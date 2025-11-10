"""
Winning Edge Data Processor - Real-time Processing Pipeline
============================================================

This module provides real-time data processing for the Winning Edge widget,
extracting corner-specific metrics, calculating performance gaps, and generating
improvement targets from raw telemetry data.

Features:
- Real-time corner detection and phase extraction (brake/apex/exit)
- Gap calculation between current and best performance
- Improvement target generation with specific numeric targets
- Session-by-session progress tracking
- Intelligent caching layer for performance optimization
- Incremental updates for new lap data

Architecture:
- Integrates with existing insights module (CornerAnalyzer, BrakePointAnalyzer)
- Follows established dashboard data flow patterns
- Uses dcc.Store for session-level caching
- Handles missing/incomplete data gracefully

Author: GR Cup Racing Analytics Team
Version: 1.0.0
License: GR Cup 2025 Hackathon
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import warnings
from functools import lru_cache
import hashlib
import json

warnings.filterwarnings('ignore')

# Import production-grade insights modules
try:
    from src.insights import (
        CornerAnalyzer,
        BrakePointAnalyzer,
        validate_telemetry_dataframe,
        logger as insights_logger,
        DataValidationError,
        AnalysisError,
        InsightsConfig
    )
    INSIGHTS_AVAILABLE = True
except ImportError:
    INSIGHTS_AVAILABLE = False
    insights_logger = logging.getLogger(__name__)
    DataValidationError = Exception
    AnalysisError = Exception

    # Fallback stub classes
    class InsightsConfig:
        """Fallback config when insights module unavailable."""
        def __init__(self, **kwargs):
            # Accept any keyword arguments and set as attributes
            for key, value in kwargs.items():
                setattr(self, key, value)

    # Fallback stub analyzers
    CornerAnalyzer = None
    BrakePointAnalyzer = None
    validate_telemetry_dataframe = lambda df: True

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class CornerPhase:
    """Represents a single phase (brake/apex/exit) of a corner."""

    phase_type: str  # 'brake', 'apex', 'exit'
    start_timestamp: float
    end_timestamp: float
    duration: float  # seconds

    # Performance metrics
    avg_speed: float  # km/h
    min_speed: float  # km/h (important for apex)
    max_speed: float  # km/h (important for exit)

    # Brake phase specific
    peak_brake_pressure: Optional[float] = None  # bar
    avg_brake_pressure: Optional[float] = None  # bar
    brake_point_distance: Optional[float] = None  # meters

    # Throttle phase specific
    avg_throttle: Optional[float] = None  # %
    throttle_application_point: Optional[float] = None  # meters

    # Steering
    avg_steering_angle: Optional[float] = None  # degrees
    max_steering_angle: Optional[float] = None  # degrees


@dataclass
class CornerMetrics:
    """Complete metrics for a single corner."""

    # Identification
    corner_id: int
    corner_name: str
    lap_number: int
    vehicle_number: int

    # Phases
    brake_phase: Optional[CornerPhase]
    apex_phase: Optional[CornerPhase]
    exit_phase: Optional[CornerPhase]

    # Overall corner metrics
    total_duration: float  # seconds
    entry_speed: float  # km/h
    apex_speed: float  # km/h
    exit_speed: float  # km/h

    # Advanced metrics
    time_to_apex: float  # seconds from brake point
    brake_to_throttle_transition: float  # seconds
    corner_g_force_max: Optional[float] = None

    # Consistency (std dev across laps)
    consistency_score: Optional[float] = None


@dataclass
class PerformanceGap:
    """Gap between current and target performance."""

    corner_id: int
    corner_name: str

    # Time gaps
    total_time_gap: float  # seconds
    brake_phase_gap: float  # seconds
    apex_phase_gap: float  # seconds
    exit_phase_gap: float  # seconds

    # Speed gaps
    entry_speed_gap: float  # km/h
    apex_speed_gap: float  # km/h
    exit_speed_gap: float  # km/h

    # Percentage of total lap time loss
    pct_of_total_loss: float

    # Brake point gap (optional field must come after required fields)
    brake_point_gap: Optional[float] = None  # meters (positive = too early)


@dataclass
class ImprovementTarget:
    """Specific, actionable improvement target for a corner."""

    corner_id: int
    corner_name: str
    priority: int  # 1 = highest

    # Current vs target metrics
    current_brake_point: float  # meters
    target_brake_point: float  # meters
    current_brake_pressure: float  # bar
    target_brake_pressure: float  # bar
    current_throttle_point: float  # % through corner
    target_throttle_point: float  # % through corner
    current_exit_speed: float  # km/h
    target_exit_speed: float  # km/h

    # Expected gain
    expected_time_gain: float  # seconds

    # Actionable coaching
    primary_action: str
    secondary_action: str
    reference_lap: int  # Best lap number to compare against


@dataclass
class SessionProgress:
    """Track improvement progress across sessions."""

    session_id: str
    session_date: datetime
    vehicle_number: int
    track_name: str

    # Overall metrics
    best_lap_time: float
    avg_lap_time: float
    consistency_score: float  # Lower is better

    # Corner-by-corner progress
    corner_improvements: Dict[int, float]  # corner_id -> time improvement vs baseline

    # Session-to-session deltas
    lap_time_delta: Optional[float] = None  # vs previous session
    consistency_delta: Optional[float] = None


# ============================================================================
# MAIN PROCESSOR CLASS
# ============================================================================

class WinningEdgeDataProcessor:
    """
    Real-time data processor for Winning Edge widget.

    Processes raw telemetry to extract corner metrics, calculate gaps,
    and generate actionable improvement targets.

    Features:
    - Corner detection and phase extraction
    - Real-time gap calculation
    - Target generation
    - Progress tracking
    - Intelligent caching

    Example:
        processor = WinningEdgeDataProcessor()

        # Process telemetry
        corner_metrics = processor.process_telemetry_for_corners(
            telemetry_df, vehicle_number=5
        )

        # Calculate gaps
        gaps = processor.calculate_real_time_gaps(
            corner_metrics, best_lap_number=3
        )

        # Generate targets
        targets = processor.generate_improvement_targets(gaps)
    """

    def __init__(self, config: Optional[InsightsConfig] = None):
        """
        Initialize processor.

        Args:
            config: Optional InsightsConfig for corner detection thresholds
        """
        self.config = config or InsightsConfig()

        # Initialize analyzers if insights module available
        if INSIGHTS_AVAILABLE:
            self.corner_analyzer = CornerAnalyzer(config=self.config)
            self.brake_analyzer = BrakePointAnalyzer(config=self.config)
        else:
            self.corner_analyzer = None
            self.brake_analyzer = None
            logger.warning("Insights module not available - using fallback algorithms")

        # Cache for processed metrics
        self._metrics_cache: Dict[str, List[CornerMetrics]] = {}
        self._best_lap_cache: Dict[str, int] = {}

        logger.info("WinningEdgeDataProcessor initialized")

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def process_telemetry_for_corners(
        self,
        telemetry_df: pd.DataFrame,
        vehicle_number: int,
        lap_numbers: Optional[List[int]] = None,
        use_cache: bool = True
    ) -> List[CornerMetrics]:
        """
        Extract corner-specific metrics from raw telemetry.

        Args:
            telemetry_df: Raw telemetry data (long format)
            vehicle_number: Vehicle to analyze
            lap_numbers: Specific laps to process (None = all laps)
            use_cache: Use cached results if available

        Returns:
            List of CornerMetrics for each corner in each lap

        Raises:
            DataValidationError: If telemetry data is invalid
            AnalysisError: If corner detection fails
        """
        # Generate cache key
        cache_key = self._generate_cache_key(telemetry_df, vehicle_number, lap_numbers)

        if use_cache and cache_key in self._metrics_cache:
            logger.info(f"Using cached corner metrics for vehicle {vehicle_number}")
            return self._metrics_cache[cache_key]

        logger.info(f"Processing telemetry for vehicle {vehicle_number}")

        # Validate input
        if INSIGHTS_AVAILABLE:
            try:
                validate_telemetry_dataframe(telemetry_df)
            except DataValidationError as e:
                logger.error(f"Telemetry validation failed: {e}")
                raise

        # Filter to vehicle
        vehicle_data = telemetry_df[telemetry_df['vehicle_number'] == vehicle_number].copy()

        if len(vehicle_data) == 0:
            logger.warning(f"No data for vehicle {vehicle_number}")
            return []

        # Get lap numbers
        if lap_numbers is None:
            lap_numbers = sorted(vehicle_data['lap'].unique())

        # Process each lap
        all_corner_metrics = []

        for lap_num in lap_numbers:
            lap_data = vehicle_data[vehicle_data['lap'] == lap_num]

            if len(lap_data) == 0:
                continue

            # Extract corner metrics for this lap
            try:
                lap_corners = self._process_single_lap(lap_data, vehicle_number, lap_num)
                all_corner_metrics.extend(lap_corners)
            except Exception as e:
                logger.error(f"Failed to process lap {lap_num} for vehicle {vehicle_number}: {e}")
                continue

        # Cache results
        self._metrics_cache[cache_key] = all_corner_metrics

        logger.info(f"Processed {len(all_corner_metrics)} corners across {len(lap_numbers)} laps")

        return all_corner_metrics

    def calculate_real_time_gaps(
        self,
        corner_metrics: List[CornerMetrics],
        best_lap_number: Optional[int] = None,
        reference_lap_number: Optional[int] = None
    ) -> List[PerformanceGap]:
        """
        Calculate gaps between current and best performance.

        Args:
            corner_metrics: List of CornerMetrics from process_telemetry_for_corners()
            best_lap_number: Use specific lap as benchmark (None = auto-detect fastest)
            reference_lap_number: Compare this lap against benchmark (None = analyze all)

        Returns:
            List of PerformanceGap for each corner
        """
        if not corner_metrics:
            logger.warning("No corner metrics provided for gap calculation")
            return []

        # Auto-detect best lap if not provided
        if best_lap_number is None:
            best_lap_number = self._find_best_lap(corner_metrics)
            logger.info(f"Auto-detected best lap: {best_lap_number}")

        # Get best lap corners
        best_corners = {
            c.corner_id: c for c in corner_metrics
            if c.lap_number == best_lap_number
        }

        if not best_corners:
            logger.error(f"No corners found for best lap {best_lap_number}")
            return []

        # Calculate total lap time for best lap
        total_best_time = sum(c.total_duration for c in best_corners.values())

        # Calculate gaps
        gaps = []

        # Group corners by ID
        corners_by_id: Dict[int, List[CornerMetrics]] = {}
        for c in corner_metrics:
            if c.corner_id not in corners_by_id:
                corners_by_id[c.corner_id] = []
            corners_by_id[c.corner_id].append(c)

        # Process each corner
        for corner_id, corner_list in corners_by_id.items():
            if corner_id not in best_corners:
                continue

            best_corner = best_corners[corner_id]

            # Calculate average for non-best laps
            other_laps = [c for c in corner_list if c.lap_number != best_lap_number]

            if reference_lap_number:
                other_laps = [c for c in other_laps if c.lap_number == reference_lap_number]

            if not other_laps:
                continue

            # Average across laps
            avg_metrics = self._average_corner_metrics(other_laps)

            # Calculate gaps
            gap = PerformanceGap(
                corner_id=corner_id,
                corner_name=best_corner.corner_name,
                total_time_gap=avg_metrics['total_duration'] - best_corner.total_duration,
                brake_phase_gap=self._phase_gap(avg_metrics, best_corner, 'brake_phase'),
                apex_phase_gap=self._phase_gap(avg_metrics, best_corner, 'apex_phase'),
                exit_phase_gap=self._phase_gap(avg_metrics, best_corner, 'exit_phase'),
                entry_speed_gap=avg_metrics['entry_speed'] - best_corner.entry_speed,
                apex_speed_gap=avg_metrics['apex_speed'] - best_corner.apex_speed,
                exit_speed_gap=avg_metrics['exit_speed'] - best_corner.exit_speed,
                brake_point_gap=self._brake_point_gap(avg_metrics, best_corner),
                pct_of_total_loss=(
                    (avg_metrics['total_duration'] - best_corner.total_duration) / total_best_time * 100
                    if total_best_time > 0 else 0
                )
            )

            gaps.append(gap)

        # Sort by time gap (largest first)
        gaps.sort(key=lambda g: g.total_time_gap, reverse=True)

        logger.info(f"Calculated gaps for {len(gaps)} corners")

        return gaps

    def generate_improvement_targets(
        self,
        gaps: List[PerformanceGap],
        corner_metrics: List[CornerMetrics],
        top_n: int = 3
    ) -> List[ImprovementTarget]:
        """
        Generate specific numeric targets based on gap analysis.

        Args:
            gaps: List of PerformanceGap from calculate_real_time_gaps()
            corner_metrics: Original corner metrics for reference values
            top_n: Number of top priority corners to target

        Returns:
            List of ImprovementTarget with specific actions
        """
        if not gaps:
            logger.warning("No gaps provided for target generation")
            return []

        # Take top N corners by time loss
        priority_gaps = gaps[:top_n]

        targets = []

        for priority, gap in enumerate(priority_gaps, 1):
            # Find best and average corners for this corner_id
            best_corner = self._find_corner(
                corner_metrics, gap.corner_id, is_best=True
            )
            avg_corners = [
                c for c in corner_metrics
                if c.corner_id == gap.corner_id and c.lap_number != best_corner.lap_number
            ]

            if not best_corner or not avg_corners:
                continue

            avg_metrics = self._average_corner_metrics(avg_corners)

            # Extract current and target values
            target = ImprovementTarget(
                corner_id=gap.corner_id,
                corner_name=gap.corner_name,
                priority=priority,
                current_brake_point=avg_metrics.get('brake_point_distance', 0),
                target_brake_point=best_corner.brake_phase.brake_point_distance if best_corner.brake_phase else 0,
                current_brake_pressure=avg_metrics.get('peak_brake_pressure', 0),
                target_brake_pressure=best_corner.brake_phase.peak_brake_pressure if best_corner.brake_phase else 0,
                current_throttle_point=self._calculate_throttle_point(avg_corners),
                target_throttle_point=self._calculate_throttle_point([best_corner]),
                current_exit_speed=avg_metrics['exit_speed'],
                target_exit_speed=best_corner.exit_speed,
                expected_time_gain=gap.total_time_gap,
                primary_action=self._generate_primary_action(gap, best_corner, avg_metrics),
                secondary_action=self._generate_secondary_action(gap, best_corner, avg_metrics),
                reference_lap=best_corner.lap_number
            )

            targets.append(target)

        logger.info(f"Generated {len(targets)} improvement targets")

        return targets

    def update_progress_tracking(
        self,
        current_session: SessionProgress,
        historical_sessions: List[SessionProgress]
    ) -> Dict[str, Any]:
        """
        Track session-by-session improvements.

        Args:
            current_session: Current session data
            historical_sessions: Previous sessions for comparison

        Returns:
            Dictionary with progress metrics and trends
        """
        if not historical_sessions:
            logger.info("No historical data for progress tracking")
            return {
                'session_count': 1,
                'overall_improvement': 0.0,
                'consistency_improvement': 0.0,
                'corner_trends': {},
                'is_first_session': True
            }

        # Sort by date
        historical_sessions = sorted(historical_sessions, key=lambda s: s.session_date)
        latest_previous = historical_sessions[-1]

        # Calculate deltas
        lap_time_delta = current_session.best_lap_time - latest_previous.best_lap_time
        consistency_delta = current_session.consistency_score - latest_previous.consistency_score

        # Calculate corner-by-corner trends
        corner_trends = {}
        for corner_id, current_improvement in current_session.corner_improvements.items():
            prev_improvement = latest_previous.corner_improvements.get(corner_id, 0.0)
            corner_trends[corner_id] = current_improvement - prev_improvement

        # Overall improvement vs first session
        first_session = historical_sessions[0]
        overall_improvement = first_session.best_lap_time - current_session.best_lap_time
        consistency_improvement = first_session.consistency_score - current_session.consistency_score

        progress = {
            'session_count': len(historical_sessions) + 1,
            'lap_time_delta': lap_time_delta,
            'consistency_delta': consistency_delta,
            'overall_improvement': overall_improvement,
            'consistency_improvement': consistency_improvement,
            'corner_trends': corner_trends,
            'is_first_session': False,
            'trending_up': lap_time_delta < 0,  # Negative delta = faster
            'most_improved_corner': max(corner_trends.items(), key=lambda x: x[1])[0] if corner_trends else None
        }

        logger.info(f"Progress tracked: {lap_time_delta:.3f}s delta, {len(corner_trends)} corners")

        return progress

    # ========================================================================
    # CORNER PHASE EXTRACTION
    # ========================================================================

    def extract_corner_phases(
        self,
        lap_data: pd.DataFrame,
        corner_id: int
    ) -> Tuple[Optional[CornerPhase], Optional[CornerPhase], Optional[CornerPhase]]:
        """
        Identify brake/apex/exit phases from telemetry.

        Args:
            lap_data: Telemetry for single lap (long format)
            corner_id: Corner identifier

        Returns:
            Tuple of (brake_phase, apex_phase, exit_phase)
        """
        # Pivot to wide format for easier processing
        wide_data = self._pivot_telemetry(lap_data)

        if wide_data.empty:
            return None, None, None

        # Ensure required columns exist
        required = ['speed', 'pbrake_f', 'timestamp']
        if not all(col in wide_data.columns for col in required):
            logger.warning(f"Missing required columns for phase extraction")
            return None, None, None

        # Detect corner boundaries using speed threshold
        speed_threshold = self.config.corner_speed_threshold
        in_corner = wide_data['speed'] < speed_threshold

        if not in_corner.any():
            return None, None, None

        # Find corner start/end
        corner_start_idx = in_corner.idxmax()
        corner_region = wide_data.loc[corner_start_idx:]
        corner_end_idx = (~corner_region.iloc[1:][in_corner[corner_start_idx:]]).idxmax() if (~corner_region[in_corner[corner_start_idx:]]).any() else len(wide_data) - 1

        corner_data = wide_data.loc[corner_start_idx:corner_end_idx]

        if len(corner_data) < 10:  # Too short to analyze
            return None, None, None

        # Phase 1: Brake phase (high brake pressure)
        brake_phase = self._extract_brake_phase(corner_data)

        # Phase 2: Apex phase (minimum speed)
        apex_phase = self._extract_apex_phase(corner_data)

        # Phase 3: Exit phase (increasing speed + throttle)
        exit_phase = self._extract_exit_phase(corner_data)

        return brake_phase, apex_phase, exit_phase

    def calculate_phase_durations(
        self,
        phases: Tuple[Optional[CornerPhase], Optional[CornerPhase], Optional[CornerPhase]]
    ) -> Dict[str, float]:
        """
        Calculate time spent in each phase.

        Args:
            phases: Tuple of (brake_phase, apex_phase, exit_phase)

        Returns:
            Dictionary with phase durations in seconds
        """
        brake, apex, exit = phases

        return {
            'brake_duration': brake.duration if brake else 0.0,
            'apex_duration': apex.duration if apex else 0.0,
            'exit_duration': exit.duration if exit else 0.0,
            'total_duration': sum([
                brake.duration if brake else 0.0,
                apex.duration if apex else 0.0,
                exit.duration if exit else 0.0
            ])
        }

    def identify_brake_points(
        self,
        lap_data: pd.DataFrame
    ) -> List[Dict[str, float]]:
        """
        Detect brake application points from telemetry.

        Args:
            lap_data: Telemetry for single lap

        Returns:
            List of brake points with timestamp and distance
        """
        wide_data = self._pivot_telemetry(lap_data)

        if 'pbrake_f' not in wide_data.columns:
            return []

        # Detect brake application (threshold crossing)
        brake_threshold = self.config.hard_brake_threshold
        brake_signal = wide_data['pbrake_f'] > brake_threshold

        # Find rising edges (brake application points)
        brake_points = []
        prev_braking = False

        for idx, is_braking in brake_signal.items():
            if is_braking and not prev_braking:
                # Brake point detected
                brake_point = {
                    'timestamp': wide_data.loc[idx, 'timestamp'],
                    'distance': wide_data.loc[idx, 'Laptrigger_lapdist_dls'] if 'Laptrigger_lapdist_dls' in wide_data.columns else 0.0,
                    'speed': wide_data.loc[idx, 'speed'] if 'speed' in wide_data.columns else 0.0,
                    'brake_pressure': wide_data.loc[idx, 'pbrake_f']
                }
                brake_points.append(brake_point)
            prev_braking = is_braking

        return brake_points

    def calculate_exit_speeds(
        self,
        lap_data: pd.DataFrame,
        corners: List[Dict]
    ) -> Dict[int, float]:
        """
        Extract exit velocities for each corner.

        Args:
            lap_data: Telemetry for single lap
            corners: List of detected corners with boundaries

        Returns:
            Dictionary mapping corner_id to exit speed (km/h)
        """
        wide_data = self._pivot_telemetry(lap_data)

        if 'speed' not in wide_data.columns:
            return {}

        exit_speeds = {}

        for corner in corners:
            corner_id = corner.get('corner_id', 0)
            end_timestamp = corner.get('end_timestamp')

            if end_timestamp is None:
                continue

            # Find data point closest to corner exit
            time_diffs = abs(wide_data['timestamp'] - end_timestamp)
            exit_idx = time_diffs.idxmin()

            # Get speed at exit (and a few points after for better estimate)
            exit_region = wide_data.loc[exit_idx:exit_idx+5]
            exit_speed = exit_region['speed'].max()  # Max speed in exit region

            exit_speeds[corner_id] = exit_speed

        return exit_speeds

    # ========================================================================
    # CACHING & UTILITIES
    # ========================================================================

    def clear_cache(self):
        """Clear all cached data."""
        self._metrics_cache.clear()
        self._best_lap_cache.clear()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'metrics_cache_size': len(self._metrics_cache),
            'best_lap_cache_size': len(self._best_lap_cache)
        }

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _process_single_lap(
        self,
        lap_data: pd.DataFrame,
        vehicle_number: int,
        lap_number: int
    ) -> List[CornerMetrics]:
        """Process a single lap to extract corner metrics."""
        # Detect corners using insights module or fallback
        if self.corner_analyzer and INSIGHTS_AVAILABLE:
            try:
                corners = self.corner_analyzer.detect_corners(lap_data)
            except Exception as e:
                logger.error(f"Corner detection failed: {e}")
                corners = self._fallback_corner_detection(lap_data)
        else:
            corners = self._fallback_corner_detection(lap_data)

        if not corners:
            return []

        # Extract metrics for each corner
        corner_metrics_list = []

        for corner in corners:
            corner_id = corner.get('corner_id', 0)

            # Extract phases
            brake, apex, exit = self.extract_corner_phases(lap_data, corner_id)

            # Calculate overall metrics
            metrics = CornerMetrics(
                corner_id=corner_id,
                corner_name=f"Turn {corner_id + 1}",
                lap_number=lap_number,
                vehicle_number=vehicle_number,
                brake_phase=brake,
                apex_phase=apex,
                exit_phase=exit,
                total_duration=sum([
                    brake.duration if brake else 0,
                    apex.duration if apex else 0,
                    exit.duration if exit else 0
                ]),
                entry_speed=brake.avg_speed if brake else corner.get('entry_speed', 0),
                apex_speed=apex.min_speed if apex else corner.get('apex_speed', 0),
                exit_speed=exit.max_speed if exit else corner.get('exit_speed', 0),
                time_to_apex=(brake.duration if brake else 0) + (apex.duration if apex else 0),
                brake_to_throttle_transition=(brake.duration if brake else 0)
            )

            corner_metrics_list.append(metrics)

        return corner_metrics_list

    def _extract_brake_phase(self, corner_data: pd.DataFrame) -> Optional[CornerPhase]:
        """Extract brake phase from corner data."""
        if 'pbrake_f' not in corner_data.columns:
            return None

        # Find region with significant braking
        brake_threshold = 30  # bar (minimum pressure to consider braking)
        braking = corner_data['pbrake_f'] > brake_threshold

        if not braking.any():
            return None

        # Find start/end of braking
        brake_start_idx = braking.idxmax()
        brake_region = corner_data.loc[brake_start_idx:]
        brake_end_idx = (~brake_region[braking[brake_start_idx:]]).idxmax() if (~brake_region[braking[brake_start_idx:]]).any() else corner_data.index[-1]

        brake_data = corner_data.loc[brake_start_idx:brake_end_idx]

        if len(brake_data) < 3:
            return None

        return CornerPhase(
            phase_type='brake',
            start_timestamp=brake_data['timestamp'].iloc[0],
            end_timestamp=brake_data['timestamp'].iloc[-1],
            duration=(brake_data['timestamp'].iloc[-1] - brake_data['timestamp'].iloc[0]),
            avg_speed=brake_data['speed'].mean(),
            min_speed=brake_data['speed'].min(),
            max_speed=brake_data['speed'].max(),
            peak_brake_pressure=brake_data['pbrake_f'].max(),
            avg_brake_pressure=brake_data['pbrake_f'].mean(),
            brake_point_distance=brake_data['Laptrigger_lapdist_dls'].iloc[0] if 'Laptrigger_lapdist_dls' in brake_data.columns else None,
            avg_steering_angle=brake_data['Steering_Angle'].mean() if 'Steering_Angle' in brake_data.columns else None
        )

    def _extract_apex_phase(self, corner_data: pd.DataFrame) -> Optional[CornerPhase]:
        """Extract apex phase from corner data."""
        if 'speed' not in corner_data.columns:
            return None

        # Apex is minimum speed point
        apex_idx = corner_data['speed'].idxmin()

        # Take small window around apex
        window_size = min(5, len(corner_data) // 3)
        apex_start = max(corner_data.index[0], apex_idx - window_size)
        apex_end = min(corner_data.index[-1], apex_idx + window_size)

        apex_data = corner_data.loc[apex_start:apex_end]

        return CornerPhase(
            phase_type='apex',
            start_timestamp=apex_data['timestamp'].iloc[0],
            end_timestamp=apex_data['timestamp'].iloc[-1],
            duration=(apex_data['timestamp'].iloc[-1] - apex_data['timestamp'].iloc[0]),
            avg_speed=apex_data['speed'].mean(),
            min_speed=apex_data['speed'].min(),
            max_speed=apex_data['speed'].max(),
            avg_throttle=apex_data['aps'].mean() if 'aps' in apex_data.columns else None,
            avg_steering_angle=apex_data['Steering_Angle'].mean() if 'Steering_Angle' in apex_data.columns else None,
            max_steering_angle=apex_data['Steering_Angle'].abs().max() if 'Steering_Angle' in apex_data.columns else None
        )

    def _extract_exit_phase(self, corner_data: pd.DataFrame) -> Optional[CornerPhase]:
        """Extract exit phase from corner data."""
        if 'speed' not in corner_data.columns:
            return None

        # Exit is after apex with increasing speed
        apex_idx = corner_data['speed'].idxmin()
        exit_data = corner_data.loc[apex_idx:]

        if len(exit_data) < 3:
            return None

        return CornerPhase(
            phase_type='exit',
            start_timestamp=exit_data['timestamp'].iloc[0],
            end_timestamp=exit_data['timestamp'].iloc[-1],
            duration=(exit_data['timestamp'].iloc[-1] - exit_data['timestamp'].iloc[0]),
            avg_speed=exit_data['speed'].mean(),
            min_speed=exit_data['speed'].min(),
            max_speed=exit_data['speed'].max(),
            avg_throttle=exit_data['aps'].mean() if 'aps' in exit_data.columns else None,
            throttle_application_point=exit_data['Laptrigger_lapdist_dls'].iloc[0] if 'Laptrigger_lapdist_dls' in exit_data.columns else None
        )

    def _pivot_telemetry(self, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Convert long format telemetry to wide format."""
        if 'telemetry_name' not in telemetry_df.columns:
            return telemetry_df  # Already wide format

        try:
            wide = telemetry_df.pivot_table(
                index='timestamp',
                columns='telemetry_name',
                values='telemetry_value',
                aggfunc='first'
            ).reset_index()
            return wide
        except Exception as e:
            logger.error(f"Failed to pivot telemetry: {e}")
            return pd.DataFrame()

    def _fallback_corner_detection(self, lap_data: pd.DataFrame) -> List[Dict]:
        """Fallback corner detection when insights module unavailable."""
        wide_data = self._pivot_telemetry(lap_data)

        if 'speed' not in wide_data.columns or len(wide_data) < 20:
            return []

        # Simple threshold-based detection
        speed_threshold = 100  # km/h
        in_corner = wide_data['speed'] < speed_threshold

        corners = []
        corner_id = 0
        in_corner_section = False
        start_idx = None

        for idx, is_corner in in_corner.items():
            if is_corner and not in_corner_section:
                # Corner start
                start_idx = idx
                in_corner_section = True
            elif not is_corner and in_corner_section:
                # Corner end
                corner_data = wide_data.loc[start_idx:idx]
                corners.append({
                    'corner_id': corner_id,
                    'start_timestamp': corner_data['timestamp'].iloc[0],
                    'end_timestamp': corner_data['timestamp'].iloc[-1],
                    'entry_speed': corner_data['speed'].iloc[0],
                    'apex_speed': corner_data['speed'].min(),
                    'exit_speed': corner_data['speed'].iloc[-1]
                })
                corner_id += 1
                in_corner_section = False

        return corners

    def _generate_cache_key(
        self,
        telemetry_df: pd.DataFrame,
        vehicle_number: int,
        lap_numbers: Optional[List[int]]
    ) -> str:
        """Generate unique cache key for telemetry data."""
        # Use hash of key parameters
        key_data = {
            'vehicle': vehicle_number,
            'laps': str(sorted(lap_numbers)) if lap_numbers else 'all',
            'data_hash': hashlib.md5(str(len(telemetry_df)).encode()).hexdigest()[:8]
        }
        return json.dumps(key_data, sort_keys=True)

    def _find_best_lap(self, corner_metrics: List[CornerMetrics]) -> int:
        """Find fastest lap from corner metrics."""
        # Group by lap
        lap_times = {}
        for c in corner_metrics:
            if c.lap_number not in lap_times:
                lap_times[c.lap_number] = 0
            lap_times[c.lap_number] += c.total_duration

        if not lap_times:
            return 1

        return min(lap_times.items(), key=lambda x: x[1])[0]

    def _average_corner_metrics(self, corners: List[CornerMetrics]) -> Dict:
        """Average metrics across multiple corner instances."""
        if not corners:
            return {}

        return {
            'total_duration': np.mean([c.total_duration for c in corners]),
            'entry_speed': np.mean([c.entry_speed for c in corners]),
            'apex_speed': np.mean([c.apex_speed for c in corners]),
            'exit_speed': np.mean([c.exit_speed for c in corners]),
            'brake_point_distance': np.mean([
                c.brake_phase.brake_point_distance for c in corners
                if c.brake_phase and c.brake_phase.brake_point_distance
            ]) if any(c.brake_phase for c in corners) else 0,
            'peak_brake_pressure': np.mean([
                c.brake_phase.peak_brake_pressure for c in corners
                if c.brake_phase and c.brake_phase.peak_brake_pressure
            ]) if any(c.brake_phase for c in corners) else 0
        }

    def _phase_gap(
        self,
        avg_metrics: Dict,
        best_corner: CornerMetrics,
        phase_name: str
    ) -> float:
        """Calculate phase duration gap."""
        best_phase = getattr(best_corner, phase_name)
        if not best_phase:
            return 0.0

        # This is simplified - would need to store phase durations in avg_metrics
        return 0.0

    def _brake_point_gap(self, avg_metrics: Dict, best_corner: CornerMetrics) -> Optional[float]:
        """Calculate brake point distance gap."""
        if not best_corner.brake_phase or not best_corner.brake_phase.brake_point_distance:
            return None

        avg_brake_point = avg_metrics.get('brake_point_distance', 0)
        best_brake_point = best_corner.brake_phase.brake_point_distance

        return avg_brake_point - best_brake_point  # Positive = braking too early

    def _find_corner(
        self,
        corner_metrics: List[CornerMetrics],
        corner_id: int,
        is_best: bool = True
    ) -> Optional[CornerMetrics]:
        """Find best corner instance by ID."""
        candidates = [c for c in corner_metrics if c.corner_id == corner_id]

        if not candidates:
            return None

        if is_best:
            return min(candidates, key=lambda c: c.total_duration)
        else:
            return max(candidates, key=lambda c: c.total_duration)

    def _calculate_throttle_point(self, corners: List[CornerMetrics]) -> float:
        """Calculate average throttle application point as % through corner."""
        if not corners:
            return 0.0

        points = []
        for c in corners:
            if c.exit_phase and c.total_duration > 0:
                # Throttle point as % through corner
                throttle_pct = (c.time_to_apex / c.total_duration) * 100
                points.append(throttle_pct)

        return np.mean(points) if points else 50.0

    def _generate_primary_action(
        self,
        gap: PerformanceGap,
        best_corner: CornerMetrics,
        avg_metrics: Dict
    ) -> str:
        """Generate primary coaching action."""
        # Determine biggest gap
        if abs(gap.brake_point_gap or 0) > 5:  # More than 5 meters difference
            if gap.brake_point_gap > 0:
                return f"Brake {abs(gap.brake_point_gap):.1f}m later (move from {avg_metrics.get('brake_point_distance', 0):.0f}m to {best_corner.brake_phase.brake_point_distance:.0f}m)"
            else:
                return f"Brake {abs(gap.brake_point_gap):.1f}m earlier"
        elif abs(gap.exit_speed_gap) > 2:  # More than 2 km/h difference
            return f"Increase exit speed by {abs(gap.exit_speed_gap):.1f} km/h (target: {best_corner.exit_speed:.1f} km/h)"
        else:
            return f"Reduce time loss of {gap.total_time_gap:.3f}s through smoother inputs"

    def _generate_secondary_action(
        self,
        gap: PerformanceGap,
        best_corner: CornerMetrics,
        avg_metrics: Dict
    ) -> str:
        """Generate secondary coaching action."""
        if best_corner.brake_phase and avg_metrics.get('peak_brake_pressure', 0) > 0:
            pressure_diff = avg_metrics['peak_brake_pressure'] - best_corner.brake_phase.peak_brake_pressure
            if abs(pressure_diff) > 10:  # More than 10 bar difference
                if pressure_diff > 0:
                    return f"Reduce peak brake pressure by {pressure_diff:.0f} bar (brake more progressively)"
                else:
                    return f"Increase peak brake pressure by {abs(pressure_diff):.0f} bar"

        return "Focus on maintaining smooth throttle application through exit"


# ============================================================================
# HELPER FUNCTIONS FOR DASHBOARD INTEGRATION
# ============================================================================

def convert_metrics_to_dict(metrics: List[CornerMetrics]) -> List[Dict]:
    """Convert CornerMetrics to JSON-serializable dicts."""
    return [asdict(m) for m in metrics]


def convert_gaps_to_dict(gaps: List[PerformanceGap]) -> List[Dict]:
    """Convert PerformanceGap to JSON-serializable dicts."""
    return [asdict(g) for g in gaps]


def convert_targets_to_dict(targets: List[ImprovementTarget]) -> List[Dict]:
    """Convert ImprovementTarget to JSON-serializable dicts."""
    return [asdict(t) for t in targets]


def format_for_winning_edge_widget(
    corner_metrics: List[CornerMetrics],
    gaps: List[PerformanceGap],
    targets: List[ImprovementTarget]
) -> Dict[str, Any]:
    """
    Format processed data for Winning Edge widget consumption.

    Returns dictionary with all data needed for visualizations.
    """
    return {
        'corner_metrics': convert_metrics_to_dict(corner_metrics),
        'performance_gaps': convert_gaps_to_dict(gaps),
        'improvement_targets': convert_targets_to_dict(targets),
        'metadata': {
            'total_corners': len(set(c.corner_id for c in corner_metrics)),
            'total_laps': len(set(c.lap_number for c in corner_metrics)),
            'best_lap': min(corner_metrics, key=lambda c: c.total_duration).lap_number if corner_metrics else None,
            'total_time_to_gain': sum(g.total_time_gap for g in gaps),
            'top_priority_corner': targets[0].corner_name if targets else None,
            'timestamp': datetime.now().isoformat()
        }
    }
