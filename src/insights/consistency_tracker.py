"""
Consistency Tracker Module

Production-grade consistency tracking system for driver performance analysis.
Tracks driver consistency across sessions, identifies performance trends,
and detects improvements or regressions over time.

Features:
- Session-over-session performance tracking
- Lap time variance analysis with statistical rigor
- Performance trend detection (improving/stable/declining)
- Outlier lap identification
- Comprehensive progress reporting

Design Pattern: Stateful Tracker Pattern
- Maintains session history across calls
- Provides historical trend analysis
- Detects performance patterns over time

Production Enhancements:
- Full input validation at all entry points
- Comprehensive error handling with context
- Performance logging and monitoring
- Configuration-driven thresholds
- Type-safe with constants and enums
- Graceful handling of edge cases (div-by-zero, insufficient data)

Author: Production Engineering Team
Version: 2.0.0 (Production)
License: GR Cup 2025 Hackathon
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings

# Production infrastructure imports
from .exceptions import (
    ConsistencyAnalysisError,
    InsufficientDataError,
    InvalidParameterError
)
from .validation import (
    validate_consistency_inputs,
    validate_lap_times_dataframe,
    validate_vehicle_number,
    validate_sufficient_data,
    validate_string_not_empty,
    validate_positive_number
)
from .logger import (
    logger,
    log_method_performance,
    log_metric_calculation,
    log_data_quality_warning,
    log_analysis_start
)
from .config import InsightsConfig, DEFAULT_CONFIG
from .constants import (
    LapTimesColumns,
    PerformanceTrend,
    OutlierType,
    MinimumDataRequirements,
    MetricNames
)
from .models import SessionMetrics, OutlierLap

warnings.filterwarnings('ignore')


class ConsistencyTracker:
    """
    Production-grade driver performance consistency tracker.

    Tracks consistency metrics across sessions, detects performance trends,
    identifies outlier laps, and generates comprehensive progress reports.

    Features:
    - Session-over-session tracking with historical analysis
    - Statistical lap time variance analysis
    - Performance trend detection (improving/stable/declining/inconsistent)
    - Outlier lap identification with z-score analysis
    - Improvement/regression alerts with configurable thresholds

    Example:
        # Initialize tracker
        tracker = ConsistencyTracker()

        # Track session
        metrics = tracker.track_session(lap_times_df, vehicle_number=5, session_id="session_1")
        print(f"Consistency: {metrics['consistency_score']:.1f}/100")

        # Detect trends
        trend = tracker.detect_performance_trend(vehicle_number=5)
        print(f"Trend: {trend['trend']}")

        # Generate progress report
        report = tracker.generate_progress_report(vehicle_number=5)
        print(report['insights'])
    """

    def __init__(self, config: Optional[InsightsConfig] = None):
        """
        Initialize consistency tracker.

        Parameters:
            config: Configuration object (uses DEFAULT_CONFIG if None)
        """
        self.config = config or DEFAULT_CONFIG
        self.session_history: Dict[int, List[Dict]] = {}
        self.trends: Dict[str, Dict] = {}

        logger.info("ConsistencyTracker initialized")

    @log_method_performance
    def track_session(
        self,
        lap_times_df: pd.DataFrame,
        vehicle_number: int,
        session_id: str
    ) -> Dict:
        """
        Track consistency metrics for a session.

        Calculates comprehensive consistency metrics including lap time statistics,
        variance analysis, and within-session performance trajectory.

        Parameters:
            lap_times_df: Lap times DataFrame with required columns
            vehicle_number: Driver/vehicle identifier (0-19)
            session_id: Unique session identifier (non-empty string)

        Returns:
            Dictionary with session consistency metrics:
            - session_id: Session identifier
            - vehicle_number: Vehicle ID
            - total_laps: Number of laps
            - fastest_lap: Fastest lap time (seconds)
            - slowest_lap: Slowest lap time (seconds)
            - average_lap: Mean lap time (seconds)
            - median_lap: Median lap time (seconds)
            - std_dev: Lap time standard deviation
            - variance: Lap time variance
            - coefficient_of_variation: CV (std/mean)
            - consistency_score: Consistency score (0-100)
            - improvement_trajectory: Within-session trend

        Raises:
            InvalidDataFrameError: If lap_times_df is invalid
            MissingColumnsError: If required columns missing
            InvalidParameterError: If vehicle_number or session_id invalid
            EmptyDatasetError: If no data for vehicle
            ConsistencyAnalysisError: If analysis fails

        Example:
            metrics = tracker.track_session(
                lap_times_df,
                vehicle_number=5,
                session_id="race_1"
            )
            print(f"Consistency: {metrics['consistency_score']:.1f}/100")
        """
        try:
            # Validate inputs
            validate_consistency_inputs(lap_times_df, vehicle_number, session_id)

            log_analysis_start(
                "session consistency tracking",
                vehicle_number=vehicle_number,
                session_id=session_id
            )

            # Filter to driver (no .copy() - safe to modify filtered view)
            vehicle_col = LapTimesColumns.VEHICLE_NUMBER
            driver_laps = lap_times_df[lap_times_df[vehicle_col] == vehicle_number]

            # Extract lap times
            lap_duration_col = LapTimesColumns.LAP_DURATION
            if lap_duration_col not in driver_laps.columns:
                raise ConsistencyAnalysisError(
                    "Missing lap_duration column in filtered data",
                    context={
                        "vehicle_number": vehicle_number,
                        "session_id": session_id,
                        "available_columns": list(driver_laps.columns)
                    }
                )

            lap_times = driver_laps[lap_duration_col].values

            # Check sufficient data for consistency
            if len(lap_times) < self.config.min_laps_for_consistency:
                log_data_quality_warning(
                    "Insufficient laps for robust consistency analysis",
                    vehicle_number=vehicle_number,
                    lap_count=len(lap_times),
                    minimum_required=self.config.min_laps_for_consistency
                )

            # Calculate metrics
            metrics = {
                'session_id': session_id,
                'vehicle_number': vehicle_number,
                'total_laps': len(lap_times),
                'fastest_lap': float(np.min(lap_times)),
                'slowest_lap': float(np.max(lap_times)),
                'average_lap': float(np.mean(lap_times)),
                'median_lap': float(np.median(lap_times)),
                'std_dev': float(np.std(lap_times)),
                'variance': float(np.var(lap_times)),
                'coefficient_of_variation': self._calculate_coefficient_of_variation(lap_times),
                'consistency_score': self._calculate_consistency_score(lap_times),
                'improvement_trajectory': self._calculate_trajectory(lap_times)
            }

            # Log key metrics
            log_metric_calculation(MetricNames.CONSISTENCY_SCORE, metrics['consistency_score'])
            log_metric_calculation('improvement_trajectory', metrics['improvement_trajectory'])

            # Store session in history
            if vehicle_number not in self.session_history:
                self.session_history[vehicle_number] = []

            self.session_history[vehicle_number].append(metrics)
            logger.debug(f"Session {session_id} stored in history for vehicle {vehicle_number}")

            return metrics

        except (InvalidParameterError, InsufficientDataError) as e:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            raise ConsistencyAnalysisError(
                f"Failed to track session consistency: {str(e)}",
                context={
                    "vehicle_number": vehicle_number,
                    "session_id": session_id,
                    "error_type": type(e).__name__
                }
            ) from e

    def _calculate_coefficient_of_variation(self, lap_times: np.ndarray) -> float:
        """
        Calculate coefficient of variation (CV).

        Handles edge cases:
        - Less than 2 laps: CV = 0 (no variance)
        - Zero mean: CV = 0 (degenerate case)

        Parameters:
            lap_times: Array of lap times

        Returns:
            Coefficient of variation (std/mean)
        """
        if len(lap_times) < 2:
            return 0.0

        mean_time = np.mean(lap_times)

        # Handle zero mean (degenerate case)
        if mean_time == 0:
            logger.warning("Zero mean lap time encountered - returning CV=0")
            return 0.0

        cv = np.std(lap_times) / mean_time
        log_metric_calculation('coefficient_of_variation', cv)

        return float(cv)

    def _calculate_consistency_score(self, lap_times: np.ndarray) -> float:
        """
        Calculate consistency score (0-100).

        Higher score = more consistent (lower variance).

        Formula: score = max(0, 100 * (1 - CV * multiplier))
        - CV = coefficient of variation (std/mean)
        - Multiplier from config (default: 20)
        - CV = 0% → score = 100
        - CV = 5% → score = 0

        Parameters:
            lap_times: Array of lap times

        Returns:
            Consistency score (0-100)
        """
        if len(lap_times) < 2:
            log_data_quality_warning(
                "Less than 2 laps - returning consistency score 0",
                lap_count=len(lap_times)
            )
            return 0.0

        # Calculate coefficient of variation
        cv = self._calculate_coefficient_of_variation(lap_times)

        # Scale: Use config multiplier
        # Default: CV=0 → 100, CV=0.05 → 0
        score = max(0, 100 * (1 - cv * self.config.consistency_score_multiplier))

        return round(score, 2)

    def _calculate_trajectory(self, lap_times: np.ndarray) -> str:
        """
        Detect performance trajectory within session.

        Analyzes session by thirds to determine if driver is:
        - Improving: Getting faster (>0.5s improvement first-to-last third)
        - Declining: Getting slower (>0.5s slower first-to-last third)
        - Stable: Consistent pace (<2% variation)
        - Inconsistent: High variance without clear trend

        Parameters:
            lap_times: Array of lap times

        Returns:
            Performance trajectory string (uses PerformanceTrend enum values)
        """
        if len(lap_times) < self.config.min_laps_for_consistency:
            log_data_quality_warning(
                "Insufficient laps for trajectory analysis",
                lap_count=len(lap_times),
                minimum=self.config.min_laps_for_consistency
            )
            return PerformanceTrend.INCONSISTENT.value

        # Split session into thirds
        third = len(lap_times) // 3

        # Handle edge case: very few laps
        if third == 0:
            third = 1

        first_third = lap_times[:third]
        middle_third = lap_times[third:2*third]
        last_third = lap_times[2*third:]

        avg_first = np.mean(first_third)
        avg_middle = np.mean(middle_third) if len(middle_third) > 0 else avg_first
        avg_last = np.mean(last_third) if len(last_third) > 0 else avg_middle

        # Check trend (positive = improving/faster)
        improvement_first_to_last = avg_first - avg_last

        # Use config threshold instead of hardcoded 0.5
        if improvement_first_to_last > self.config.min_improvement_delta:
            return PerformanceTrend.IMPROVING.value
        elif improvement_first_to_last < -self.config.min_improvement_delta:
            return PerformanceTrend.DECLINING.value
        else:
            # Check if stable or inconsistent
            overall_std = np.std(lap_times)
            mean_time = np.mean(lap_times)

            # Avoid div-by-zero
            if mean_time == 0:
                return PerformanceTrend.INCONSISTENT.value

            # Use config threshold for consistency
            if overall_std < mean_time * self.config.consistency_variance_threshold:
                return PerformanceTrend.STABLE.value
            else:
                return PerformanceTrend.INCONSISTENT.value

    @log_method_performance
    def compare_sessions(
        self,
        vehicle_number: int,
        session_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare consistency across multiple sessions.

        Parameters:
            vehicle_number: Driver/vehicle identifier (0-19)
            session_ids: Specific sessions to compare (None = all sessions)

        Returns:
            DataFrame with session comparison table

        Raises:
            InvalidParameterError: If vehicle_number invalid
            EmptyDatasetError: If no session history found

        Example:
            comparison = tracker.compare_sessions(vehicle_number=5)
            print(comparison)
        """
        try:
            # Validate vehicle number
            validate_vehicle_number(vehicle_number)

            if vehicle_number not in self.session_history:
                raise InsufficientDataError(
                    "No session history found for vehicle",
                    context={
                        "vehicle_number": vehicle_number,
                        "available_vehicles": list(self.session_history.keys())
                    }
                )

            sessions = self.session_history[vehicle_number]

            # Filter sessions if specified
            if session_ids:
                sessions = [s for s in sessions if s['session_id'] in session_ids]

                if len(sessions) == 0:
                    log_data_quality_warning(
                        "No matching sessions found",
                        vehicle_number=vehicle_number,
                        requested_sessions=session_ids
                    )
                    return pd.DataFrame()

            # Build comparison table
            comparison_data = []

            for session in sessions:
                comparison_data.append({
                    'Session': session['session_id'],
                    'Laps': session['total_laps'],
                    'Fastest': f"{session['fastest_lap']:.3f}s",
                    'Average': f"{session['average_lap']:.3f}s",
                    'Std Dev': f"{session['std_dev']:.3f}s",
                    'Consistency': f"{session['consistency_score']:.1f}/100",
                    'Trajectory': session['improvement_trajectory']
                })

            logger.info(f"Compared {len(comparison_data)} sessions for vehicle {vehicle_number}")

            return pd.DataFrame(comparison_data)

        except (InvalidParameterError, InsufficientDataError) as e:
            raise
        except Exception as e:
            raise ConsistencyAnalysisError(
                f"Failed to compare sessions: {str(e)}",
                context={
                    "vehicle_number": vehicle_number,
                    "error_type": type(e).__name__
                }
            ) from e

    @log_method_performance
    def detect_performance_trend(
        self,
        vehicle_number: int,
        metric: str = MetricNames.FASTEST_LAP,
        min_sessions: Optional[int] = None
    ) -> Dict:
        """
        Detect long-term performance trends across sessions.

        Uses linear regression to identify trends in performance metrics
        over time (improving, stable, or declining).

        Parameters:
            vehicle_number: Driver/vehicle identifier (0-19)
            metric: Metric to track (default: 'fastest_lap')
                Options: 'fastest_lap', 'average_lap', 'consistency_score', etc.
            min_sessions: Minimum sessions required (default: from config)

        Returns:
            Dictionary with trend analysis:
            - vehicle_number: Vehicle ID
            - metric: Metric analyzed
            - sessions_analyzed: Number of sessions
            - trend: Trend classification (Improving/Stable/Declining)
            - slope: Linear regression slope
            - avg_change_per_session: Average change per session
            - current_value: Latest session value
            - starting_value: First session value
            - total_change: Total change from first to last

        Raises:
            InvalidParameterError: If vehicle_number invalid
            InsufficientDataError: If insufficient sessions

        Example:
            trend = tracker.detect_performance_trend(
                vehicle_number=5,
                metric='fastest_lap'
            )
            print(f"Trend: {trend['trend']}")
        """
        try:
            # Validate vehicle number
            validate_vehicle_number(vehicle_number)

            if vehicle_number not in self.session_history:
                raise InsufficientDataError(
                    "No session history found for vehicle",
                    context={"vehicle_number": vehicle_number}
                )

            sessions = self.session_history[vehicle_number]

            # Use config default if not specified
            min_required = min_sessions or self.config.min_sessions_for_trend

            if len(sessions) < min_required:
                raise InsufficientDataError(
                    f"Insufficient sessions for trend analysis",
                    context={
                        "vehicle_number": vehicle_number,
                        "required_sessions": min_required,
                        "available_sessions": len(sessions)
                    }
                )

            # Extract metric values
            values = [s.get(metric, 0) for s in sessions]

            if not values or all(v == 0 for v in values):
                raise ConsistencyAnalysisError(
                    f"No valid data for metric",
                    context={
                        "vehicle_number": vehicle_number,
                        "metric": metric
                    }
                )

            # Calculate trend (linear regression)
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)

            # Interpret trend based on metric type
            if metric in [MetricNames.FASTEST_LAP, MetricNames.AVERAGE_LAP, 'slowest_lap']:
                # For lap times, negative slope = improvement (getting faster)
                if slope < -0.1:
                    trend = PerformanceTrend.IMPROVING.value + ' (getting faster)'
                elif slope > 0.1:
                    trend = PerformanceTrend.DECLINING.value + ' (getting slower)'
                else:
                    trend = PerformanceTrend.STABLE.value
            else:
                # For scores, positive slope = improvement
                if slope > 1:
                    trend = PerformanceTrend.IMPROVING.value
                elif slope < -1:
                    trend = PerformanceTrend.DECLINING.value
                else:
                    trend = PerformanceTrend.STABLE.value

            # Calculate improvement rate
            if len(values) >= 2:
                total_change = values[-1] - values[0]
                sessions_span = len(values) - 1
                avg_change_per_session = total_change / sessions_span if sessions_span > 0 else 0
            else:
                total_change = 0
                avg_change_per_session = 0

            trend_analysis = {
                'vehicle_number': vehicle_number,
                'metric': metric,
                'sessions_analyzed': len(sessions),
                'trend': trend,
                'slope': round(slope, 4),
                'avg_change_per_session': round(avg_change_per_session, 4),
                'current_value': values[-1],
                'starting_value': values[0],
                'total_change': round(total_change, 4)
            }

            # Store trend
            self.trends[f"{vehicle_number}_{metric}"] = trend_analysis

            log_metric_calculation('performance_trend', trend)
            logger.info(
                f"Trend analysis complete: {trend} "
                f"(slope={slope:.4f}, sessions={len(sessions)})"
            )

            return trend_analysis

        except (InvalidParameterError, InsufficientDataError) as e:
            raise
        except Exception as e:
            raise ConsistencyAnalysisError(
                f"Failed to detect performance trend: {str(e)}",
                context={
                    "vehicle_number": vehicle_number,
                    "metric": metric,
                    "error_type": type(e).__name__
                }
            ) from e

    @log_method_performance
    def identify_outlier_laps(
        self,
        lap_times_df: pd.DataFrame,
        vehicle_number: int,
        std_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Identify outlier laps (unusually fast or slow).

        Uses z-score analysis to detect laps that deviate significantly
        from the session mean.

        Parameters:
            lap_times_df: Lap times DataFrame
            vehicle_number: Driver/vehicle identifier (0-19)
            std_threshold: Number of standard deviations for outlier
                (default: from config, typically 2.0)

        Returns:
            DataFrame with outlier laps:
            - lap: Lap number
            - lap_duration: Lap time (seconds)
            - z_score: Z-score relative to mean
            - outlier_type: 'Exceptionally Fast' or 'Exceptionally Slow'
            - deviation_from_mean: Deviation in seconds

            Empty DataFrame if no outliers found.

        Raises:
            InvalidDataFrameError: If lap_times_df invalid
            InvalidParameterError: If vehicle_number invalid
            EmptyDatasetError: If no data for vehicle

        Example:
            outliers = tracker.identify_outlier_laps(
                lap_times_df,
                vehicle_number=5,
                std_threshold=2.0
            )
            print(f"Found {len(outliers)} outlier laps")

        Note:
            The .apply() call is inefficient for large datasets.
            TODO: Vectorize outlier_type classification for better performance.
        """
        try:
            # Validate inputs
            validate_lap_times_dataframe(lap_times_df, name="lap_times_df")
            validate_vehicle_number(vehicle_number)

            # Use config default if not specified
            threshold = std_threshold or self.config.outlier_std_threshold

            validate_positive_number(threshold, "std_threshold")

            # Filter to driver (no .copy() - safe for read-only operations)
            vehicle_col = LapTimesColumns.VEHICLE_NUMBER
            lap_duration_col = LapTimesColumns.LAP_DURATION

            driver_laps = lap_times_df[lap_times_df[vehicle_col] == vehicle_number]

            if len(driver_laps) == 0 or lap_duration_col not in driver_laps.columns:
                log_data_quality_warning(
                    "No lap data for outlier detection",
                    vehicle_number=vehicle_number
                )
                return pd.DataFrame()

            # Calculate statistics
            mean_time = driver_laps[lap_duration_col].mean()
            std_time = driver_laps[lap_duration_col].std()

            # Handle zero std (all laps identical)
            if std_time == 0:
                logger.warning(
                    f"Zero standard deviation for vehicle {vehicle_number} - "
                    f"all laps are identical, no outliers"
                )
                return pd.DataFrame()

            # Create working copy for outlier analysis
            driver_laps_analysis = driver_laps.copy()

            # Calculate z-scores
            driver_laps_analysis[TelemetryColumns.Z_SCORE] = (
                (driver_laps_analysis[lap_duration_col] - mean_time) / std_time
            )

            # Identify outliers
            driver_laps_analysis[TelemetryColumns.IS_OUTLIER] = (
                driver_laps_analysis[TelemetryColumns.Z_SCORE].abs() > threshold
            )

            outliers = driver_laps_analysis[driver_laps_analysis[TelemetryColumns.IS_OUTLIER]].copy()

            if len(outliers) == 0:
                logger.debug(f"No outliers found for vehicle {vehicle_number}")
                return pd.DataFrame()

            # Categorize outliers (using enum constants)
            # TODO: Vectorize this instead of .apply() for better performance
            outliers['outlier_type'] = outliers[TelemetryColumns.Z_SCORE].apply(
                lambda z: OutlierType.EXCEPTIONALLY_FAST.value
                if z < -threshold
                else OutlierType.EXCEPTIONALLY_SLOW.value
            )

            # Format output
            lap_col = LapTimesColumns.LAP
            result = outliers[[
                lap_col, lap_duration_col, TelemetryColumns.Z_SCORE, 'outlier_type'
            ]].copy()

            result['deviation_from_mean'] = outliers[lap_duration_col] - mean_time

            logger.info(
                f"Identified {len(result)} outlier laps for vehicle {vehicle_number} "
                f"(threshold={threshold}σ)"
            )

            return result.sort_values(TelemetryColumns.Z_SCORE)

        except (InvalidParameterError, InsufficientDataError) as e:
            raise
        except Exception as e:
            raise ConsistencyAnalysisError(
                f"Failed to identify outlier laps: {str(e)}",
                context={
                    "vehicle_number": vehicle_number,
                    "error_type": type(e).__name__
                }
            ) from e

    @log_method_performance
    def generate_progress_report(
        self,
        vehicle_number: int
    ) -> Dict:
        """
        Generate comprehensive progress report for a driver.

        Analyzes session history to provide insights on:
        - Overall performance trends
        - Consistency trends
        - Best achievements
        - Actionable recommendations

        Parameters:
            vehicle_number: Driver/vehicle identifier (0-19)

        Returns:
            Dictionary with progress report:
            - vehicle_number: Vehicle ID
            - total_sessions: Number of sessions tracked
            - best_lap_ever: Best lap time across all sessions
            - current_consistency: Latest session consistency score
            - avg_consistency: Average consistency across sessions
            - performance_trend: Overall performance trend
            - consistency_trend: Consistency trend over time
            - latest_session: Most recent session ID
            - insights: List of insight strings with recommendations

        Raises:
            InvalidParameterError: If vehicle_number invalid
            InsufficientDataError: If no session history found

        Example:
            report = tracker.generate_progress_report(vehicle_number=5)
            print(f"Total Sessions: {report['total_sessions']}")
            for insight in report['insights']:
                print(f"- {insight}")
        """
        try:
            # Validate vehicle number
            validate_vehicle_number(vehicle_number)

            if vehicle_number not in self.session_history:
                raise InsufficientDataError(
                    "No session history found for vehicle",
                    context={"vehicle_number": vehicle_number}
                )

            sessions = self.session_history[vehicle_number]

            if len(sessions) == 0:
                raise InsufficientDataError(
                    "No sessions tracked for vehicle",
                    context={"vehicle_number": vehicle_number}
                )

            log_analysis_start(
                "progress report generation",
                vehicle_number=vehicle_number,
                total_sessions=len(sessions)
            )

            # Analyze trends (need min 2 sessions)
            if len(sessions) >= self.config.min_sessions_for_trend:
                fastest_trend = self.detect_performance_trend(
                    vehicle_number,
                    MetricNames.FASTEST_LAP
                )
                consistency_trend = self.detect_performance_trend(
                    vehicle_number,
                    MetricNames.CONSISTENCY_SCORE
                )
            else:
                fastest_trend = {'trend': 'Insufficient sessions'}
                consistency_trend = {'trend': 'Insufficient sessions'}

            # Overall metrics
            all_fastest_laps = [s['fastest_lap'] for s in sessions]
            all_consistency_scores = [s['consistency_score'] for s in sessions]

            report = {
                'vehicle_number': vehicle_number,
                'total_sessions': len(sessions),
                'best_lap_ever': min(all_fastest_laps),
                'current_consistency': all_consistency_scores[-1] if all_consistency_scores else 0,
                'avg_consistency': float(np.mean(all_consistency_scores)) if all_consistency_scores else 0,
                'performance_trend': fastest_trend.get('trend', 'Unknown'),
                'consistency_trend': consistency_trend.get('trend', 'Unknown'),
                'latest_session': sessions[-1]['session_id'],
                'insights': []
            }

            # Generate insights
            if fastest_trend.get('trend') == 'Improving (getting faster)':
                avg_change = fastest_trend.get('avg_change_per_session', 0)
                report['insights'].append(
                    f"Lap times improving! "
                    f"{abs(avg_change):.3f}s faster per session on average."
                )
            elif fastest_trend.get('trend') == 'Declining (getting slower)':
                report['insights'].append(
                    "Lap times declining. "
                    "Review recent changes and focus on fundamentals."
                )

            # Consistency insights
            if report['current_consistency'] > 85:
                report['insights'].append(
                    "Excellent consistency! Ready to push pace."
                )
            elif report['current_consistency'] < 60:
                report['insights'].append(
                    "Focus on consistency before pushing harder."
                )

            logger.info(
                f"Progress report generated: {len(report['insights'])} insights, "
                f"trend={report['performance_trend']}"
            )

            return report

        except (InvalidParameterError, InsufficientDataError) as e:
            raise
        except Exception as e:
            raise ConsistencyAnalysisError(
                f"Failed to generate progress report: {str(e)}",
                context={
                    "vehicle_number": vehicle_number,
                    "error_type": type(e).__name__
                }
            ) from e
