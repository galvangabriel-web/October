"""
Brake Point Analyzer Module - Production Grade
===============================================

Analyzes braking zones to identify optimal brake points and compare
driver braking performance across laps and corners.

Production Features:
- Automatic brake zone detection
- Brake point comparison (early/late/optimal)
- Brake pressure profile analysis
- Corner-specific brake coaching
- Brake consistency tracking

Key Insights:
- "Braking 15m too early in Turn 3"
- "Inconsistent brake points (±8m variation)"
- "Peak brake pressure 12 bar lower than optimal"

Author: Production Engineering Team
Version: 1.0.0 (Phase 1.2)
License: GR Cup 2025 Hackathon
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.signal import find_peaks, savgol_filter

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
class BrakeZone:
    """Single braking zone analysis"""

    # Identification
    zone_id: int  # Sequential zone number (1, 2, 3...)
    corner_number: Optional[int]  # Associated corner (if known)

    # Location
    start_distance: float  # Brake point (meters from start/finish)
    end_distance: float  # Brake release point
    duration: float  # Braking duration (seconds)

    # Performance metrics
    peak_pressure: float  # Maximum brake pressure (bar)
    avg_pressure: float  # Average brake pressure (bar)
    entry_speed: float  # Speed at brake point (km/h)
    exit_speed: float  # Speed at brake release (km/h)
    deceleration: float  # Average deceleration (g)

    # Consistency
    brake_point_std: Optional[float] = None  # Std dev of brake point across laps

    # Comparison (vs best lap)
    delta_brake_point: Optional[float] = None  # Distance difference (m)
    delta_peak_pressure: Optional[float] = None  # Pressure difference (bar)
    delta_entry_speed: Optional[float] = None  # Speed difference (km/h)

    # Coaching
    recommendation: Optional[str] = None  # Actionable coaching tip


@dataclass
class BrakeAnalysis:
    """Complete brake point analysis for a lap or session"""

    vehicle_number: int
    lap_number: Optional[int]  # None if analyzing multiple laps

    # Brake zones found
    brake_zones: List[BrakeZone]

    # Overall statistics
    total_brake_zones: int
    avg_brake_duration: float  # Average across all zones
    avg_peak_pressure: float
    avg_deceleration: float

    # Consistency metrics
    brake_point_consistency: str  # "Excellent", "Good", "Fair", "Poor"
    most_inconsistent_zone: Optional[int]  # Zone ID with highest variation

    # Key insights
    coaching_tips: List[str]  # Top 3-5 actionable recommendations


class BrakePointAnalyzer:
    """
    Analyze braking zones and provide coaching insights.

    Features:
    - Automatic brake zone detection using pressure thresholds
    - Brake point comparison vs best lap
    - Consistency tracking across multiple laps
    - Corner-specific brake coaching

    Example:
        analyzer = BrakePointAnalyzer()
        analysis = analyzer.analyze_lap(telemetry_df, lap_number=5, vehicle_number=7)
        for zone in analysis.brake_zones:
            print(f"Corner {zone.corner_number}: {zone.recommendation}")
    """

    def __init__(self, config: Optional[InsightsConfig] = None):
        """
        Initialize brake point analyzer.

        Args:
            config: Configuration object (uses brake thresholds)
        """
        self.config = config or DEFAULT_CONFIG
        self.brake_threshold = 20.0  # Minimum bar to consider braking
        logger.info("BrakePointAnalyzer initialized")

    @log_method_performance
    def analyze_lap(
        self,
        telemetry_df: pd.DataFrame,
        lap_number: int,
        vehicle_number: int,
        best_lap_telemetry: Optional[pd.DataFrame] = None
    ) -> BrakeAnalysis:
        """
        Analyze all brake zones in a single lap.

        Args:
            telemetry_df: Telemetry data (long format)
            lap_number: Lap number to analyze
            vehicle_number: Vehicle ID
            best_lap_telemetry: Optional best lap telemetry for comparison

        Returns:
            BrakeAnalysis object with all brake zones and insights

        Raises:
            InsufficientDataError: If lap data is missing
        """
        # Extract lap telemetry
        lap_telem = self._extract_lap_telemetry(telemetry_df, vehicle_number, lap_number)

        if lap_telem.empty:
            raise InsufficientDataError(
                f"No telemetry found for vehicle {vehicle_number}, lap {lap_number}"
            )

        # Detect brake zones
        brake_zones = self._detect_brake_zones(lap_telem)

        # If best lap provided, compare each zone
        if best_lap_telemetry is not None:
            best_zones = self._detect_brake_zones(best_lap_telemetry)
            brake_zones = self._compare_to_best(brake_zones, best_zones)

        # Calculate statistics
        total_zones = len(brake_zones)
        avg_duration = np.mean([z.duration for z in brake_zones]) if brake_zones else 0
        avg_peak_pressure = np.mean([z.peak_pressure for z in brake_zones]) if brake_zones else 0
        avg_deceleration = np.mean([z.deceleration for z in brake_zones]) if brake_zones else 0

        # Assess consistency (requires multiple zones)
        consistency_rating = "N/A"
        most_inconsistent_zone = None

        # Generate coaching tips
        coaching_tips = self._generate_coaching_tips(brake_zones)

        analysis = BrakeAnalysis(
            vehicle_number=vehicle_number,
            lap_number=lap_number,
            brake_zones=brake_zones,
            total_brake_zones=total_zones,
            avg_brake_duration=avg_duration,
            avg_peak_pressure=avg_peak_pressure,
            avg_deceleration=avg_deceleration,
            brake_point_consistency=consistency_rating,
            most_inconsistent_zone=most_inconsistent_zone,
            coaching_tips=coaching_tips
        )

        logger.info(f"Brake analysis complete: {total_zones} zones detected")
        return analysis

    @log_method_performance
    def analyze_session(
        self,
        telemetry_df: pd.DataFrame,
        vehicle_number: int,
        lap_numbers: Optional[List[int]] = None
    ) -> BrakeAnalysis:
        """
        Analyze brake zones across multiple laps to assess consistency.

        Args:
            telemetry_df: Telemetry data for all laps
            vehicle_number: Vehicle ID
            lap_numbers: Optional list of laps to analyze (default: all)

        Returns:
            BrakeAnalysis with consistency metrics

        Raises:
            InsufficientDataError: If insufficient laps found
        """
        if lap_numbers is None:
            # Get all laps for vehicle
            vehicle_data = telemetry_df[telemetry_df['vehicle_number'] == vehicle_number]
            lap_numbers = sorted(vehicle_data['lap'].unique())

        if len(lap_numbers) < 2:
            raise InsufficientDataError("Need at least 2 laps for consistency analysis")

        logger.info(f"Analyzing {len(lap_numbers)} laps for brake consistency")

        # Analyze each lap
        all_zones_by_corner = {}  # corner_id -> List[BrakeZone]

        for lap_num in lap_numbers:
            lap_analysis = self.analyze_lap(telemetry_df, lap_num, vehicle_number)

            for zone in lap_analysis.brake_zones:
                corner_id = zone.zone_id  # Use zone_id as corner identifier
                if corner_id not in all_zones_by_corner:
                    all_zones_by_corner[corner_id] = []
                all_zones_by_corner[corner_id].append(zone)

        # Calculate consistency per corner
        avg_brake_zones = []

        for corner_id, zones in all_zones_by_corner.items():
            # Calculate brake point variation
            brake_points = [z.start_distance for z in zones]
            brake_point_std = np.std(brake_points)

            # Average zone
            avg_zone = BrakeZone(
                zone_id=corner_id,
                corner_number=corner_id,
                start_distance=np.mean(brake_points),
                end_distance=np.mean([z.end_distance for z in zones]),
                duration=np.mean([z.duration for z in zones]),
                peak_pressure=np.mean([z.peak_pressure for z in zones]),
                avg_pressure=np.mean([z.avg_pressure for z in zones]),
                entry_speed=np.mean([z.entry_speed for z in zones]),
                exit_speed=np.mean([z.exit_speed for z in zones]),
                deceleration=np.mean([z.deceleration for z in zones]),
                brake_point_std=brake_point_std,
                recommendation=self._assess_consistency(brake_point_std, corner_id)
            )
            avg_brake_zones.append(avg_zone)

        # Find most inconsistent zone
        most_inconsistent = max(
            avg_brake_zones,
            key=lambda z: z.brake_point_std
        ) if avg_brake_zones else None

        # Overall consistency rating
        if avg_brake_zones:
            avg_std = np.mean([z.brake_point_std for z in avg_brake_zones])
            consistency_rating = self._rate_brake_consistency(avg_std)
        else:
            consistency_rating = "N/A"
            avg_std = 0

        # Generate coaching tips
        coaching_tips = self._generate_session_coaching(
            avg_brake_zones, consistency_rating, most_inconsistent
        )

        analysis = BrakeAnalysis(
            vehicle_number=vehicle_number,
            lap_number=None,  # Session analysis
            brake_zones=avg_brake_zones,
            total_brake_zones=len(avg_brake_zones),
            avg_brake_duration=np.mean([z.duration for z in avg_brake_zones]) if avg_brake_zones else 0,
            avg_peak_pressure=np.mean([z.peak_pressure for z in avg_brake_zones]) if avg_brake_zones else 0,
            avg_deceleration=np.mean([z.deceleration for z in avg_brake_zones]) if avg_brake_zones else 0,
            brake_point_consistency=consistency_rating,
            most_inconsistent_zone=most_inconsistent.zone_id if most_inconsistent else None,
            coaching_tips=coaching_tips
        )

        logger.info(f"Session analysis complete: {consistency_rating} consistency")
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
                # Fallback: use index * 10m
                pivot_data['gps_distance'] = np.arange(len(pivot_data)) * 10.0

        # Ensure required columns
        for col in ['pbrake_f', 'speed', 'accy_can']:
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

    def _detect_brake_zones(self, lap_telem: pd.DataFrame) -> List[BrakeZone]:
        """
        Detect all brake zones in a lap using pressure threshold.

        A brake zone starts when pressure > threshold and ends when pressure < threshold.

        Args:
            lap_telem: Lap telemetry (wide format)

        Returns:
            List of BrakeZone objects
        """
        if lap_telem.empty or 'pbrake_f' not in lap_telem.columns:
            return []

        brake_pressure = lap_telem['pbrake_f'].values
        speed = lap_telem['speed'].values
        distance = lap_telem['gps_distance'].values
        timestamps = lap_telem['timestamp'].values

        # Smooth brake pressure to avoid noise
        if len(brake_pressure) > 10:
            brake_pressure = savgol_filter(brake_pressure, 11, 3)

        # Find brake zones (pressure > threshold)
        braking = brake_pressure > self.brake_threshold

        # Find start/end indices of brake zones
        zone_starts = np.where(np.diff(braking.astype(int)) == 1)[0] + 1
        zone_ends = np.where(np.diff(braking.astype(int)) == -1)[0] + 1

        # Handle edge cases
        if braking[0]:
            zone_starts = np.concatenate([[0], zone_starts])
        if braking[-1]:
            zone_ends = np.concatenate([zone_ends, [len(braking) - 1]])

        # Match starts with ends
        min_len = min(len(zone_starts), len(zone_ends))
        zone_starts = zone_starts[:min_len]
        zone_ends = zone_ends[:min_len]

        # Build BrakeZone objects
        brake_zones = []

        for i, (start_idx, end_idx) in enumerate(zip(zone_starts, zone_ends)):
            # Skip very short zones (< 0.5s)
            duration = (timestamps[end_idx] - timestamps[start_idx]) / 1000.0  # Convert ms to s
            if duration < 0.5:
                continue

            # Extract metrics
            zone_pressure = brake_pressure[start_idx:end_idx+1]
            zone_speed = speed[start_idx:end_idx+1]

            peak_pressure = np.max(zone_pressure)
            avg_pressure = np.mean(zone_pressure)
            entry_speed = speed[start_idx]
            exit_speed = speed[end_idx]

            # Calculate deceleration (g-force)
            # Use lateral_g if available, else estimate from speed change
            if 'accy_can' in lap_telem.columns:
                deceleration = np.abs(np.mean(lap_telem['accy_can'].values[start_idx:end_idx+1]))
            else:
                # Estimate: delta_v / delta_t / 9.81
                delta_v = (entry_speed - exit_speed) / 3.6  # km/h to m/s
                delta_t = duration
                deceleration = (delta_v / delta_t / 9.81) if delta_t > 0 else 0

            zone = BrakeZone(
                zone_id=i + 1,
                corner_number=i + 1,  # Assume sequential corners
                start_distance=distance[start_idx],
                end_distance=distance[end_idx],
                duration=duration,
                peak_pressure=peak_pressure,
                avg_pressure=avg_pressure,
                entry_speed=entry_speed,
                exit_speed=exit_speed,
                deceleration=deceleration
            )

            brake_zones.append(zone)

        logger.info(f"Detected {len(brake_zones)} brake zones")
        return brake_zones

    def _compare_to_best(
        self,
        current_zones: List[BrakeZone],
        best_zones: List[BrakeZone]
    ) -> List[BrakeZone]:
        """
        Compare current lap zones to best lap zones.

        Matches zones by proximity and calculates deltas.

        Args:
            current_zones: Brake zones from current lap
            best_zones: Brake zones from best lap

        Returns:
            Updated current_zones with comparison data
        """
        for current_zone in current_zones:
            # Find matching zone in best lap (closest by distance)
            if not best_zones:
                continue

            distances = [abs(current_zone.start_distance - bz.start_distance) for bz in best_zones]
            closest_idx = np.argmin(distances)
            best_zone = best_zones[closest_idx]

            # Only match if within 100m
            if distances[closest_idx] < 100:
                current_zone.delta_brake_point = current_zone.start_distance - best_zone.start_distance
                current_zone.delta_peak_pressure = current_zone.peak_pressure - best_zone.peak_pressure
                current_zone.delta_entry_speed = current_zone.entry_speed - best_zone.entry_speed

                # Generate recommendation
                current_zone.recommendation = self._generate_zone_recommendation(current_zone)

        return current_zones

    def _generate_zone_recommendation(self, zone: BrakeZone) -> str:
        """Generate coaching recommendation for a brake zone"""
        if zone.delta_brake_point is None:
            return "No comparison data available"

        tips = []

        # Brake point timing
        if zone.delta_brake_point > 10:
            tips.append(f"Braking {zone.delta_brake_point:.0f}m too early")
        elif zone.delta_brake_point < -10:
            tips.append(f"Braking {abs(zone.delta_brake_point):.0f}m too late")

        # Brake pressure
        if zone.delta_peak_pressure and zone.delta_peak_pressure < -10:
            tips.append(f"Increase peak pressure by {abs(zone.delta_peak_pressure):.0f} bar")

        # Entry speed
        if zone.delta_entry_speed and zone.delta_entry_speed < -5:
            tips.append(f"Entry speed {abs(zone.delta_entry_speed):.0f} km/h slower")

        return " | ".join(tips) if tips else "Brake point looks good"

    def _assess_consistency(self, brake_point_std: float, corner_id: int) -> str:
        """Assess brake point consistency for a corner"""
        if brake_point_std < 3:
            return f"Corner {corner_id}: Excellent consistency (±{brake_point_std:.1f}m)"
        elif brake_point_std < 8:
            return f"Corner {corner_id}: Good consistency (±{brake_point_std:.1f}m)"
        elif brake_point_std < 15:
            return f"Corner {corner_id}: Fair consistency (±{brake_point_std:.1f}m) - work on repeatability"
        else:
            return f"Corner {corner_id}: Poor consistency (±{brake_point_std:.1f}m) - focus on brake markers"

    def _rate_brake_consistency(self, avg_std: float) -> str:
        """Rate overall brake consistency"""
        if avg_std < 3:
            return "Excellent"
        elif avg_std < 8:
            return "Good"
        elif avg_std < 15:
            return "Fair"
        else:
            return "Poor"

    def _generate_coaching_tips(self, brake_zones: List[BrakeZone]) -> List[str]:
        """Generate top coaching tips for a single lap"""
        tips = []

        if not brake_zones:
            return ["No brake zones detected"]

        # Check for recommendations from comparison
        for zone in brake_zones:
            if zone.recommendation and zone.recommendation != "Brake point looks good":
                tips.append(f"Corner {zone.corner_number}: {zone.recommendation}")

        if not tips:
            tips.append("All brake points look good - maintain consistency")

        return tips[:5]  # Top 5 tips

    def _generate_session_coaching(
        self,
        avg_zones: List[BrakeZone],
        consistency_rating: str,
        most_inconsistent: Optional[BrakeZone]
    ) -> List[str]:
        """Generate coaching tips for session analysis"""
        tips = []

        # Overall consistency
        tips.append(f"Overall brake consistency: {consistency_rating}")

        # Most inconsistent zone
        if most_inconsistent and most_inconsistent.brake_point_std > 8:
            tips.append(
                f"Most inconsistent: Corner {most_inconsistent.corner_number} "
                f"(±{most_inconsistent.brake_point_std:.1f}m variation)"
            )

        # Specific zone advice
        for zone in avg_zones[:3]:  # Top 3 zones
            if zone.brake_point_std and zone.brake_point_std > 8:
                tips.append(zone.recommendation)

        return tips


if __name__ == "__main__":
    """Example usage"""
    from data_loader import RacingDataLoader

    # Load data
    loader = RacingDataLoader()
    telemetry = loader.load_single_chunk('circuit-of-the-americas', 'race_1', 'telemetry', chunk_num=1)

    # Create analyzer
    analyzer = BrakePointAnalyzer()

    # Analyze single lap
    analysis = analyzer.analyze_lap(telemetry, lap_number=5, vehicle_number=5)

    print(f"Brake Analysis for Lap {analysis.lap_number}")
    print(f"Total brake zones: {analysis.total_brake_zones}")
    print(f"Average brake duration: {analysis.avg_brake_duration:.2f}s")
    print(f"Average peak pressure: {analysis.avg_peak_pressure:.1f} bar")

    print("\nBrake Zones:")
    for zone in analysis.brake_zones:
        print(f"  Corner {zone.corner_number}: {zone.start_distance:.0f}m - {zone.peak_pressure:.1f} bar")

    print("\nCoaching Tips:")
    for tip in analysis.coaching_tips:
        print(f"  • {tip}")
