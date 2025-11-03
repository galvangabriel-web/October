"""
Corner Speed Benchmarking Module - Production Grade
====================================================

Analyzes corner speeds (entry, apex, exit) and benchmarks against best lap
to identify specific corners where time is being lost or gained.

Production Features:
- Automatic corner detection using speed/G-force
- Entry/Apex/Exit speed measurement
- Benchmark comparison vs best lap
- Corner-specific coaching recommendations
- Speed trace visualization data

Key Insights:
- "Turn 3 apex: 5 km/h slower than best"
- "Turn 7 exit: carrying 8 km/h more speed"
- "Consistent entry speeds, losing time mid-corner"

Author: Production Engineering Team
Version: 1.0.0 (Phase 1.2)
License: GR Cup 2025 Hackathon
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.signal import find_peaks, savgol_filter, argrelextrema

# Production infrastructure imports
from .exceptions import (
    DataValidationError,
    InsufficientDataError,
    EmptyDatasetError
)
from .logger import logger, log_method_performance
from .config import InsightsConfig, DEFAULT_CONFIG
from .constants import TelemetrySensor


@dataclass
class CornerSpeed:
    """Corner speed analysis for a single corner"""

    # Identification (required fields first)
    corner_id: int  # Sequential corner number

    # Location (GPS distance from start/finish)
    entry_distance: float  # Turn-in point (meters)
    apex_distance: float  # Apex point (meters)
    exit_distance: float  # Track-out point (meters)

    # Speed metrics (km/h)
    entry_speed: float
    apex_speed: float  # Minimum speed in corner
    exit_speed: float

    # G-force metrics
    max_lateral_g: float  # Peak lateral G in corner
    avg_lateral_g: float  # Average lateral G

    # Performance metrics
    corner_duration: float  # Time spent in corner (seconds)
    speed_loss: float  # Entry - Apex (km/h)
    speed_gain: float  # Exit - Apex (km/h)

    # Optional fields with defaults (must come after required fields)
    corner_name: Optional[str] = None  # e.g., "Turn 1", "The Esses"

    # Comparison (vs best lap)
    delta_entry_speed: Optional[float] = None
    delta_apex_speed: Optional[float] = None
    delta_exit_speed: Optional[float] = None
    delta_time: Optional[float] = None  # Time lost/gained in this corner

    # Classification
    corner_type: str = "Unknown"  # "Slow", "Medium", "Fast", "Chicane"

    # Coaching
    recommendation: Optional[str] = None


@dataclass
class CornerBenchmarkAnalysis:
    """Complete corner benchmarking analysis"""

    vehicle_number: int
    lap_number: Optional[int]

    # Corners detected
    corners: List[CornerSpeed]
    total_corners: int

    # Overall statistics
    avg_apex_speed: float
    avg_lateral_g: float
    avg_corner_duration: float

    # Performance summary
    best_corners: List[int]  # Corner IDs where driver excels
    worst_corners: List[int]  # Corner IDs where driver loses most time

    # Coaching insights
    coaching_tips: List[str]  # Top recommendations


class CornerSpeedBenchmarking:
    """
    Benchmark corner speeds against best lap to identify improvement areas.

    Features:
    - Automatic corner detection using speed minima + lateral G
    - Entry/Apex/Exit speed extraction
    - Corner type classification (slow/medium/fast)
    - Benchmark comparison with coaching
    - Multi-lap consistency analysis

    Example:
        benchmarking = CornerSpeedBenchmarking()
        analysis = benchmarking.analyze_lap(
            telemetry_df, lap_number=5, vehicle_number=7,
            best_lap_telemetry=best_lap_df
        )
        for corner in analysis.corners:
            print(f"{corner.corner_name}: {corner.recommendation}")
    """

    def __init__(self, config: Optional[InsightsConfig] = None):
        """
        Initialize corner speed benchmarking.

        Args:
            config: Configuration object
        """
        self.config = config or DEFAULT_CONFIG
        self.min_lateral_g = 0.5  # Minimum G to consider a corner
        logger.info("CornerSpeedBenchmarking initialized")

    @log_method_performance
    def analyze_lap(
        self,
        telemetry_df: pd.DataFrame,
        lap_number: int,
        vehicle_number: int,
        best_lap_telemetry: Optional[pd.DataFrame] = None
    ) -> CornerBenchmarkAnalysis:
        """
        Analyze all corners in a lap and benchmark against best lap.

        Args:
            telemetry_df: Telemetry data (long format)
            lap_number: Lap number to analyze
            vehicle_number: Vehicle ID
            best_lap_telemetry: Optional best lap telemetry for comparison

        Returns:
            CornerBenchmarkAnalysis with all corners and insights

        Raises:
            InsufficientDataError: If lap data is missing
        """
        # Extract lap telemetry
        lap_telem = self._extract_lap_telemetry(telemetry_df, vehicle_number, lap_number)

        if lap_telem.empty:
            raise InsufficientDataError(
                f"No telemetry for vehicle {vehicle_number}, lap {lap_number}"
            )

        # Detect corners
        corners = self._detect_corners(lap_telem)

        # If best lap provided, compare each corner
        if best_lap_telemetry is not None:
            best_corners = self._detect_corners(best_lap_telemetry)
            corners = self._compare_to_best(corners, best_corners)

        # Calculate statistics
        total_corners = len(corners)
        avg_apex_speed = np.mean([c.apex_speed for c in corners]) if corners else 0
        avg_lateral_g = np.mean([c.max_lateral_g for c in corners]) if corners else 0
        avg_duration = np.mean([c.corner_duration for c in corners]) if corners else 0

        # Identify best/worst corners
        best_corners_ids, worst_corners_ids = self._identify_performance_corners(corners)

        # Generate coaching tips
        coaching_tips = self._generate_coaching_tips(corners, worst_corners_ids)

        analysis = CornerBenchmarkAnalysis(
            vehicle_number=vehicle_number,
            lap_number=lap_number,
            corners=corners,
            total_corners=total_corners,
            avg_apex_speed=avg_apex_speed,
            avg_lateral_g=avg_lateral_g,
            avg_corner_duration=avg_duration,
            best_corners=best_corners_ids,
            worst_corners=worst_corners_ids,
            coaching_tips=coaching_tips
        )

        logger.info(f"Corner analysis complete: {total_corners} corners detected")
        return analysis

    def _extract_lap_telemetry(
        self,
        telemetry_df: pd.DataFrame,
        vehicle_number: int,
        lap_number: int
    ) -> pd.DataFrame:
        """Extract and pivot telemetry for a specific lap"""
        lap_data = telemetry_df[
            (telemetry_df['vehicle_number'] == vehicle_number) &
            (telemetry_df['lap'] == lap_number)
        ].copy()

        if lap_data.empty:
            return pd.DataFrame()

        # Pivot to wide format
        pivot_data = lap_data.pivot_table(
            index='timestamp',
            columns='telemetry_name',
            values='telemetry_value',
            aggfunc='first'
        ).reset_index()

        # Calculate GPS distance if missing
        if 'gps_distance' not in pivot_data.columns:
            if 'gps_lat' in pivot_data.columns and 'gps_long' in pivot_data.columns:
                pivot_data['gps_distance'] = self._calculate_gps_distance(
                    pivot_data['gps_lat'], pivot_data['gps_long']
                )
            else:
                pivot_data['gps_distance'] = np.arange(len(pivot_data)) * 10.0

        # Ensure required columns
        for col in ['speed', 'accy_can']:
            if col not in pivot_data.columns:
                pivot_data[col] = 0.0

        return pivot_data.sort_values('timestamp')

    def _calculate_gps_distance(self, lat: pd.Series, lon: pd.Series) -> np.ndarray:
        """Calculate cumulative GPS distance (Haversine formula)"""
        if lat.empty or lon.empty:
            return np.arange(len(lat)) * 10.0

        lat_rad = np.radians(lat.values)
        lon_rad = np.radians(lon.values)

        dlat = np.diff(lat_rad)
        dlon = np.diff(lon_rad)

        a = np.sin(dlat/2)**2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        R = 6371000  # Earth radius (meters)
        distances = R * c

        return np.concatenate([[0], np.cumsum(distances)])

    def _detect_corners(self, lap_telem: pd.DataFrame) -> List[CornerSpeed]:
        """
        Detect all corners in a lap using speed minima and lateral G.

        A corner is detected as a local minimum in speed combined with
        elevated lateral G-force.

        Args:
            lap_telem: Lap telemetry (wide format)

        Returns:
            List of CornerSpeed objects
        """
        if lap_telem.empty or 'speed' not in lap_telem.columns:
            return []

        speed = lap_telem['speed'].values
        lateral_g = np.abs(lap_telem['accy_can'].values) if 'accy_can' in lap_telem.columns else np.zeros_like(speed)
        distance = lap_telem['gps_distance'].values
        timestamps = lap_telem['timestamp'].values

        # Smooth data
        if len(speed) > 20:
            speed = savgol_filter(speed, 21, 3)
            lateral_g = savgol_filter(lateral_g, 21, 3)

        # Find speed minima (apex candidates)
        apex_indices = argrelextrema(speed, np.less, order=10)[0]

        # Filter: only keep minima with significant lateral G
        apex_indices = [idx for idx in apex_indices if lateral_g[idx] > self.min_lateral_g]

        corners = []

        for corner_id, apex_idx in enumerate(apex_indices, start=1):
            # Define corner boundaries
            # Look backwards for entry (where speed starts decreasing)
            entry_idx = self._find_entry_point(speed, apex_idx)

            # Look forward for exit (where speed stops increasing)
            exit_idx = self._find_exit_point(speed, apex_idx)

            # Extract corner metrics
            entry_speed = speed[entry_idx]
            apex_speed = speed[apex_idx]
            exit_speed = speed[exit_idx]

            # G-force in corner
            corner_lateral_g = lateral_g[entry_idx:exit_idx+1]
            max_lateral_g = np.max(corner_lateral_g)
            avg_lateral_g = np.mean(corner_lateral_g)

            # Corner duration
            duration = (timestamps[exit_idx] - timestamps[entry_idx]) / 1000.0  # ms to s

            # Speed changes
            speed_loss = entry_speed - apex_speed
            speed_gain = exit_speed - apex_speed

            # Classify corner type
            corner_type = self._classify_corner(apex_speed)

            corner = CornerSpeed(
                corner_id=corner_id,
                corner_name=f"Turn {corner_id}",
                entry_distance=distance[entry_idx],
                apex_distance=distance[apex_idx],
                exit_distance=distance[exit_idx],
                entry_speed=entry_speed,
                apex_speed=apex_speed,
                exit_speed=exit_speed,
                max_lateral_g=max_lateral_g,
                avg_lateral_g=avg_lateral_g,
                corner_duration=duration,
                speed_loss=speed_loss,
                speed_gain=speed_gain,
                corner_type=corner_type
            )

            corners.append(corner)

        logger.info(f"Detected {len(corners)} corners")
        return corners

    def _find_entry_point(self, speed: np.ndarray, apex_idx: int, lookback: int = 30) -> int:
        """Find turn-in point by looking backwards from apex for speed peak"""
        start = max(0, apex_idx - lookback)
        segment = speed[start:apex_idx]

        if len(segment) == 0:
            return apex_idx

        # Find where speed peaks before apex
        peak_idx = np.argmax(segment)
        return start + peak_idx

    def _find_exit_point(self, speed: np.ndarray, apex_idx: int, lookforward: int = 30) -> int:
        """Find track-out point by looking forward from apex for speed peak"""
        end = min(len(speed), apex_idx + lookforward)
        segment = speed[apex_idx:end]

        if len(segment) == 0:
            return apex_idx

        # Find where speed peaks after apex
        peak_idx = np.argmax(segment)
        return apex_idx + peak_idx

    def _classify_corner(self, apex_speed: float) -> str:
        """Classify corner type based on apex speed"""
        if apex_speed < 80:
            return "Slow"  # Hairpins, tight corners
        elif apex_speed < 120:
            return "Medium"  # 90-degree corners
        elif apex_speed < 160:
            return "Fast"  # Sweepers, kinks
        else:
            return "Flat-out"  # High-speed corners

    def _compare_to_best(
        self,
        current_corners: List[CornerSpeed],
        best_corners: List[CornerSpeed]
    ) -> List[CornerSpeed]:
        """
        Compare current lap corners to best lap corners.

        Matches corners by proximity and calculates deltas.

        Args:
            current_corners: Corners from current lap
            best_corners: Corners from best lap

        Returns:
            Updated current_corners with comparison data
        """
        for current_corner in current_corners:
            if not best_corners:
                continue

            # Find matching corner in best lap (closest apex)
            distances = [
                abs(current_corner.apex_distance - bc.apex_distance)
                for bc in best_corners
            ]
            closest_idx = np.argmin(distances)
            best_corner = best_corners[closest_idx]

            # Only match if within 100m
            if distances[closest_idx] < 100:
                current_corner.delta_entry_speed = current_corner.entry_speed - best_corner.entry_speed
                current_corner.delta_apex_speed = current_corner.apex_speed - best_corner.apex_speed
                current_corner.delta_exit_speed = current_corner.exit_speed - best_corner.exit_speed

                # Estimate time delta (rough approximation)
                avg_speed_diff = (
                    current_corner.delta_entry_speed +
                    current_corner.delta_apex_speed +
                    current_corner.delta_exit_speed
                ) / 3
                corner_length = current_corner.exit_distance - current_corner.entry_distance
                current_corner.delta_time = (corner_length / 1000) / ((current_corner.apex_speed + avg_speed_diff) / 3600) if current_corner.apex_speed > 0 else 0

                # Generate recommendation
                current_corner.recommendation = self._generate_corner_recommendation(current_corner)

        return current_corners

    def _generate_corner_recommendation(self, corner: CornerSpeed) -> str:
        """Generate coaching recommendation for a corner"""
        if corner.delta_apex_speed is None:
            return "No comparison data available"

        tips = []

        # Entry speed
        if corner.delta_entry_speed is not None:
            if corner.delta_entry_speed < -3:
                tips.append(f"Entry: {abs(corner.delta_entry_speed):.1f} km/h slower")
            elif corner.delta_entry_speed > 3:
                tips.append(f"Entry: {corner.delta_entry_speed:.1f} km/h faster (may be overdriving)")

        # Apex speed
        if corner.delta_apex_speed < -2:
            tips.append(f"Apex: {abs(corner.delta_apex_speed):.1f} km/h slower - maintain momentum")
        elif corner.delta_apex_speed > 2:
            tips.append(f"Apex: {corner.delta_apex_speed:.1f} km/h faster - excellent!")

        # Exit speed
        if corner.delta_exit_speed is not None:
            if corner.delta_exit_speed < -3:
                tips.append(f"Exit: {abs(corner.delta_exit_speed):.1f} km/h slower - work on throttle application")
            elif corner.delta_exit_speed > 3:
                tips.append(f"Exit: {corner.delta_exit_speed:.1f} km/h faster - great exit!")

        return " | ".join(tips) if tips else "Corner speed looks good"

    def _identify_performance_corners(
        self,
        corners: List[CornerSpeed]
    ) -> Tuple[List[int], List[int]]:
        """
        Identify best and worst corners based on delta_time.

        Returns:
            Tuple of (best_corner_ids, worst_corner_ids)
        """
        # Filter corners with comparison data
        compared_corners = [c for c in corners if c.delta_time is not None]

        if not compared_corners:
            return [], []

        # Sort by delta_time
        sorted_corners = sorted(compared_corners, key=lambda c: c.delta_time)

        # Best corners (negative delta = faster)
        best_corners = [c.corner_id for c in sorted_corners[:3] if c.delta_time < 0]

        # Worst corners (positive delta = slower)
        worst_corners = [c.corner_id for c in sorted_corners[-3:] if c.delta_time > 0]

        return best_corners, worst_corners

    def _generate_coaching_tips(
        self,
        corners: List[CornerSpeed],
        worst_corners_ids: List[int]
    ) -> List[str]:
        """Generate top coaching tips"""
        tips = []

        # Focus on worst corners
        for corner_id in worst_corners_ids:
            corner = next((c for c in corners if c.corner_id == corner_id), None)
            if corner and corner.recommendation:
                tips.append(f"{corner.corner_name}: {corner.recommendation}")

        if not tips:
            tips.append("All corner speeds competitive - maintain consistency")

        return tips[:5]  # Top 5 tips


if __name__ == "__main__":
    """Example usage"""
    from data_loader import RacingDataLoader

    # Load data
    loader = RacingDataLoader()
    telemetry = loader.load_single_chunk('circuit-of-the-americas', 'race_1', 'telemetry', chunk_num=1)

    # Create benchmarking module
    benchmarking = CornerSpeedBenchmarking()

    # Analyze lap
    analysis = benchmarking.analyze_lap(telemetry, lap_number=5, vehicle_number=5)

    print(f"Corner Benchmarking for Lap {analysis.lap_number}")
    print(f"Total corners: {analysis.total_corners}")
    print(f"Average apex speed: {analysis.avg_apex_speed:.1f} km/h")
    print(f"Average lateral G: {analysis.avg_lateral_g:.2f}g")

    print("\nCorners:")
    for corner in analysis.corners:
        print(f"  {corner.corner_name} ({corner.corner_type}): "
              f"Entry={corner.entry_speed:.1f}, Apex={corner.apex_speed:.1f}, "
              f"Exit={corner.exit_speed:.1f} km/h")

    print("\nCoaching Tips:")
    for tip in analysis.coaching_tips:
        print(f"  • {tip}")
