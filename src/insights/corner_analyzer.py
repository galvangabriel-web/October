"""
Corner Analyzer Module - Production Grade

Analyzes corner-by-corner performance, identifies optimal racing lines,
and provides turn-specific insights for driver improvement.

Features:
- Corner detection from GPS/speed data
- Entry/apex/exit speed analysis
- Braking point identification
- Comparison vs. optimal line
- Production-grade validation, logging, and error handling

Design Pattern: Analyzer Pattern
- Stateful corner tracking
- Optimal line calculation and caching
- Comprehensive error handling
- Performance monitoring

Author: Production Engineering Team
Version: 2.0.0 (Refactored from demo quality to production grade)
License: GR Cup 2025 Hackathon
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Production infrastructure imports
from .exceptions import (
    CornerDetectionError,
    AnalysisError,
    InvalidParameterError
)
from .validation import (
    validate_corner_detection_inputs,
    validate_telemetry_dataframe,
    validate_lap_times_dataframe,
    validate_dataframe,
    validate_vehicle_number,
    validate_positive_number
)
from .logger import (
    logger,
    log_method_performance,
    log_corner_detection,
    log_data_load,
    log_metric_calculation,
    log_data_quality_warning,
    log_exception_with_context
)
from .config import InsightsConfig, DEFAULT_CONFIG
from .constants import (
    TelemetrySensor,
    TelemetryColumns,
    LapTimesColumns
)
from .models import CornerData, CornerPerformance, OptimalLine


class CornerAnalyzer:
    """
    Analyzes individual corner performance and racing lines.

    Production-grade implementation with:
    - Comprehensive input validation
    - Structured error handling
    - Performance logging
    - Configuration-driven thresholds
    - Type-safe return structures

    Features:
    - Corner detection from GPS/speed data
    - Entry/apex/exit speed analysis
    - Braking point identification
    - Comparison vs. optimal line

    Example:
        from src.insights.corner_analyzer import CornerAnalyzer
        from src.insights.config import InsightsConfig

        config = InsightsConfig(corner_speed_threshold=130.0)
        analyzer = CornerAnalyzer(config=config)

        # Detect corners
        corners = analyzer.detect_corners(telemetry_df)

        # Analyze specific corner
        corner_perf = analyzer.analyze_corner_performance(telemetry_df, corners[0])

        # Find optimal line
        optimal = analyzer.find_optimal_line(telemetry_df, lap_times_df, corner_id=0)
    """

    def __init__(self, config: Optional[InsightsConfig] = None):
        """
        Initialize corner analyzer.

        Parameters:
            config: Configuration object (uses DEFAULT_CONFIG if None)
        """
        self.config = config or DEFAULT_CONFIG
        self.corners = []
        self.optimal_lines = {}

        logger.debug(f"CornerAnalyzer initialized with config: corner_speed_threshold={self.config.corner_speed_threshold}")

    @log_method_performance
    def detect_corners(
        self,
        telemetry_df: pd.DataFrame,
        speed_threshold: Optional[float] = None,
        min_duration_ms: Optional[int] = None
    ) -> List[Dict]:
        """
        Detect corners from speed and acceleration data.

        Uses speed threshold to identify low-speed sections (corners) and
        validates minimum duration to filter out transient speed drops.

        Parameters:
            telemetry_df: Telemetry data with speed and acceleration
            speed_threshold: Speed below which is considered cornering (km/h).
                           Uses config.corner_speed_threshold if None.
            min_duration_ms: Minimum corner duration in milliseconds.
                           Uses config.min_corner_duration_ms if None.

        Returns:
            List of detected corners with metadata (corner_id, timestamps, speeds)

        Raises:
            InvalidDataFrameError: If telemetry_df is invalid
            MissingColumnsError: If required columns missing
            SensorDataError: If speed sensor data missing
            CornerDetectionError: If corner detection fails

        Example:
            corners = analyzer.detect_corners(telemetry_df)
            # [{'corner_id': 0, 'start_timestamp': 1234567890,
            #   'entry_speed': 140.5, 'apex_speed': 95.2, ...}, ...]
        """
        # Use config defaults if not provided
        speed_threshold = speed_threshold or self.config.corner_speed_threshold
        min_duration_ms = min_duration_ms or self.config.min_corner_duration_ms

        # Validate inputs
        try:
            validate_corner_detection_inputs(telemetry_df, speed_threshold, min_duration_ms)
        except Exception as e:
            log_exception_with_context(
                e,
                context={
                    "method": "detect_corners",
                    "speed_threshold": speed_threshold,
                    "min_duration_ms": min_duration_ms
                }
            )
            raise

        try:
            # Extract speed data
            speed_data = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.SPEED
            ].copy()

            if len(speed_data) == 0:
                log_data_quality_warning(
                    "No speed data found in telemetry",
                    telemetry_records=len(telemetry_df)
                )
                return []

            log_data_load("speed telemetry", len(speed_data))

            # Ensure timestamp is numeric
            if TelemetryColumns.TIMESTAMP in speed_data.columns:
                speed_data[TelemetryColumns.TIMESTAMP] = pd.to_numeric(
                    speed_data[TelemetryColumns.TIMESTAMP],
                    errors='coerce'
                )

            # Sort by timestamp
            speed_data = speed_data.sort_values(TelemetryColumns.TIMESTAMP)

            # Find low-speed sections (corners)
            speed_data['is_corner'] = speed_data[TelemetryColumns.TELEMETRY_VALUE] < speed_threshold

            # Identify corner segments
            speed_data['corner_change'] = speed_data['is_corner'].astype(int).diff()

            corners = []
            corner_id = 0
            corner_start = None

            # TODO: This iterrows() loop is inefficient. Consider vectorization for production scale.
            # Current implementation is readable but may be slow for large datasets.
            # Future improvement: Use pandas groupby with consecutive groups detection.
            for idx, row in speed_data.iterrows():
                if row['corner_change'] == 1:  # Corner start
                    corner_start = {
                        'corner_id': corner_id,
                        'start_timestamp': row[TelemetryColumns.TIMESTAMP],
                        'entry_speed': row[TelemetryColumns.TELEMETRY_VALUE],
                        'lap': row.get(TelemetryColumns.LAP, None),
                        'vehicle_number': row.get(TelemetryColumns.VEHICLE_NUMBER, None)
                    }
                elif row['corner_change'] == -1 and corner_start:  # Corner end
                    duration = row[TelemetryColumns.TIMESTAMP] - corner_start['start_timestamp']

                    if duration >= min_duration_ms:
                        corner_start['end_timestamp'] = row[TelemetryColumns.TIMESTAMP]
                        corner_start['exit_speed'] = row[TelemetryColumns.TELEMETRY_VALUE]
                        corner_start['duration_ms'] = duration

                        # Find apex (minimum speed in corner)
                        corner_segment = speed_data[
                            (speed_data[TelemetryColumns.TIMESTAMP] >= corner_start['start_timestamp']) &
                            (speed_data[TelemetryColumns.TIMESTAMP] <= row[TelemetryColumns.TIMESTAMP])
                        ]

                        if len(corner_segment) > 0:
                            apex_row = corner_segment.loc[
                                corner_segment[TelemetryColumns.TELEMETRY_VALUE].idxmin()
                            ]
                            corner_start['apex_speed'] = apex_row[TelemetryColumns.TELEMETRY_VALUE]
                            corner_start['apex_timestamp'] = apex_row[TelemetryColumns.TIMESTAMP]

                        corners.append(corner_start)
                        corner_id += 1

                    corner_start = None

            self.corners = corners
            log_corner_detection(len(corners), speed_threshold)

            # Warn if no corners detected
            if len(corners) == 0:
                log_data_quality_warning(
                    "No corners detected - consider adjusting thresholds",
                    speed_threshold=speed_threshold,
                    min_duration_ms=min_duration_ms,
                    speed_records=len(speed_data)
                )

            return corners

        except Exception as e:
            context = {
                "method": "detect_corners",
                "speed_threshold": speed_threshold,
                "min_duration_ms": min_duration_ms,
                "telemetry_records": len(telemetry_df)
            }
            log_exception_with_context(e, context=context)
            raise CornerDetectionError(
                f"Corner detection failed: {str(e)}",
                context=context
            )

    @log_method_performance
    def analyze_corner_performance(
        self,
        telemetry_df: pd.DataFrame,
        corner: Dict
    ) -> Dict:
        """
        Detailed analysis of a single corner.

        Analyzes braking, throttle, and g-force characteristics for a
        specific corner instance.

        Parameters:
            telemetry_df: Full telemetry data
            corner: Corner metadata from detect_corners()

        Returns:
            Corner performance metrics including braking, throttle, g-forces

        Raises:
            InvalidDataFrameError: If telemetry_df is invalid
            InvalidParameterError: If corner metadata incomplete
            AnalysisError: If analysis fails

        Example:
            corner = corners[0]
            perf = analyzer.analyze_corner_performance(telemetry_df, corner)
            # {'corner_id': 0, 'entry_speed': 140.5, 'apex_speed': 95.2,
            #  'braking': {...}, 'throttle': {...}, 'g_forces': {...}}
        """
        # Validate inputs
        validate_telemetry_dataframe(telemetry_df, name="telemetry_df")

        # Validate corner has required fields
        required_fields = ['corner_id', 'start_timestamp', 'end_timestamp']
        missing = [f for f in required_fields if f not in corner]
        if missing:
            raise InvalidParameterError(
                f"Corner metadata missing required fields: {missing}",
                context={"missing_fields": missing, "corner": corner}
            )

        try:
            # Extract corner telemetry
            corner_data = telemetry_df[
                (telemetry_df[TelemetryColumns.TIMESTAMP] >= corner['start_timestamp']) &
                (telemetry_df[TelemetryColumns.TIMESTAMP] <= corner['end_timestamp'])
            ].copy()

            if len(corner_data) == 0:
                raise AnalysisError(
                    "No telemetry data found for corner time window",
                    context={
                        "corner_id": corner['corner_id'],
                        "start_timestamp": corner['start_timestamp'],
                        "end_timestamp": corner['end_timestamp']
                    }
                )

            # Analyze braking
            brake_data = corner_data[
                corner_data[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.BRAKE_PRESSURE_FRONT
            ]

            braking_analysis = {}
            if len(brake_data) > 0:
                braking_analysis = {
                    'max_brake_pressure': float(brake_data[TelemetryColumns.TELEMETRY_VALUE].max()),
                    'avg_brake_pressure': float(brake_data[TelemetryColumns.TELEMETRY_VALUE].mean()),
                    'brake_application_time_ms': len(brake_data) * 10  # Approximate
                }

                # Find braking point (first significant brake application)
                hard_brake = brake_data[brake_data[TelemetryColumns.TELEMETRY_VALUE] > 50]
                if len(hard_brake) > 0:
                    braking_analysis['brake_point_timestamp'] = int(
                        hard_brake.iloc[0][TelemetryColumns.TIMESTAMP]
                    )

                log_metric_calculation(
                    "corner_braking",
                    f"max={braking_analysis['max_brake_pressure']:.1f} bar"
                )

            # Analyze throttle
            throttle_data = corner_data[
                corner_data[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.THROTTLE
            ]

            throttle_analysis = {}
            if len(throttle_data) > 0:
                throttle_analysis = {
                    'min_throttle': float(throttle_data[TelemetryColumns.TELEMETRY_VALUE].min()),
                    'avg_throttle': float(throttle_data[TelemetryColumns.TELEMETRY_VALUE].mean()),
                    'throttle_on_exit_pct': float(
                        throttle_data.iloc[-3:][TelemetryColumns.TELEMETRY_VALUE].mean()
                        if len(throttle_data) >= 3 else 0
                    )
                }

            # Analyze g-forces
            accel_x = corner_data[
                corner_data[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.ACCEL_LONGITUDINAL
            ]
            accel_y = corner_data[
                corner_data[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.ACCEL_LATERAL
            ]

            g_force_analysis = {}
            if len(accel_y) > 0:
                g_force_analysis['max_lateral_g'] = float(
                    accel_y[TelemetryColumns.TELEMETRY_VALUE].abs().max()
                )
                g_force_analysis['avg_lateral_g'] = float(
                    accel_y[TelemetryColumns.TELEMETRY_VALUE].abs().mean()
                )

            if len(accel_x) > 0:
                g_force_analysis['max_longitudinal_g'] = float(
                    accel_x[TelemetryColumns.TELEMETRY_VALUE].abs().max()
                )

            # Compile full analysis
            analysis = {
                'corner_id': corner['corner_id'],
                'entry_speed': corner['entry_speed'],
                'apex_speed': corner.get('apex_speed', 0),
                'exit_speed': corner['exit_speed'],
                'speed_loss': corner['entry_speed'] - corner.get('apex_speed', corner['entry_speed']),
                'speed_gain': corner['exit_speed'] - corner.get('apex_speed', corner['exit_speed']),
                'duration_ms': corner['duration_ms'],
                'braking': braking_analysis,
                'throttle': throttle_analysis,
                'g_forces': g_force_analysis
            }

            logger.debug(
                f"Analyzed corner {corner['corner_id']}: "
                f"entry={analysis['entry_speed']:.1f}, "
                f"apex={analysis['apex_speed']:.1f}, "
                f"exit={analysis['exit_speed']:.1f}"
            )

            return analysis

        except AnalysisError:
            # Re-raise AnalysisError as-is
            raise
        except Exception as e:
            context = {
                "method": "analyze_corner_performance",
                "corner_id": corner.get('corner_id', 'unknown'),
                "corner_data_records": len(corner_data) if 'corner_data' in locals() else 0
            }
            log_exception_with_context(e, context=context)
            raise AnalysisError(
                f"Corner performance analysis failed: {str(e)}",
                context=context
            )

    @log_method_performance
    def find_optimal_line(
        self,
        telemetry_df: pd.DataFrame,
        lap_times_df: pd.DataFrame,
        corner_id: int
    ) -> Dict:
        """
        Identify optimal racing line for a corner based on fastest laps.

        Analyzes all instances of a specific corner across different laps,
        finds the fastest lap, and extracts that corner's characteristics
        as the "optimal" reference line.

        Parameters:
            telemetry_df: All telemetry data
            lap_times_df: Lap times data
            corner_id: Corner identifier

        Returns:
            Optimal line characteristics (empty dict if cannot determine)

        Raises:
            InvalidDataFrameError: If inputs are invalid
            InvalidParameterError: If corner_id is invalid
            AnalysisError: If optimal line calculation fails

        Example:
            optimal = analyzer.find_optimal_line(telemetry_df, lap_times_df, corner_id=0)
            # {'corner_id': 0, 'optimal_entry_speed': 142.5,
            #  'optimal_apex_speed': 98.3, 'optimal_exit_speed': 125.7, ...}
        """
        # Validate inputs
        validate_telemetry_dataframe(telemetry_df, name="telemetry_df")
        validate_lap_times_dataframe(lap_times_df, name="lap_times_df")
        validate_positive_number(corner_id, "corner_id", allow_zero=True)

        try:
            # Validate corner_id exists in detected corners
            if len(self.corners) == 0:
                log_data_quality_warning(
                    "No corners detected - call detect_corners() first",
                    corner_id=corner_id
                )
                return {}

            # Find corners matching this corner_id across all laps
            matching_corners = [c for c in self.corners if c['corner_id'] == corner_id]

            if len(matching_corners) == 0:
                log_data_quality_warning(
                    "Corner ID not found in detected corners",
                    corner_id=corner_id,
                    available_corner_ids=[c['corner_id'] for c in self.corners[:5]]  # First 5
                )
                return {}

            logger.debug(f"Found {len(matching_corners)} instances of corner {corner_id}")

            # Analyze each corner instance
            corner_analyses = []

            for corner in matching_corners:
                try:
                    analysis = self.analyze_corner_performance(telemetry_df, corner)

                    # Get lap time for this corner's lap
                    if corner.get(TelemetryColumns.LAP) and corner.get(TelemetryColumns.VEHICLE_NUMBER):
                        lap_info = lap_times_df[
                            (lap_times_df[LapTimesColumns.LAP] == corner[TelemetryColumns.LAP]) &
                            (lap_times_df[LapTimesColumns.VEHICLE_NUMBER] == corner[TelemetryColumns.VEHICLE_NUMBER])
                        ]

                        if len(lap_info) > 0 and LapTimesColumns.LAP_DURATION in lap_info.columns:
                            analysis['lap_time'] = float(lap_info.iloc[0][LapTimesColumns.LAP_DURATION])

                    corner_analyses.append(analysis)

                except AnalysisError as e:
                    # Skip corners that fail analysis
                    logger.debug(f"Skipping corner instance: {str(e)}")
                    continue

            # Find fastest lap's corner technique
            valid_analyses = [a for a in corner_analyses if 'lap_time' in a]

            if not valid_analyses:
                log_data_quality_warning(
                    "No valid corner analyses with lap times",
                    corner_id=corner_id,
                    analyzed_corners=len(corner_analyses)
                )
                return {}

            fastest = min(valid_analyses, key=lambda x: x['lap_time'])

            # This becomes the "optimal" reference line
            optimal_line = {
                'corner_id': corner_id,
                'optimal_entry_speed': fastest['entry_speed'],
                'optimal_apex_speed': fastest['apex_speed'],
                'optimal_exit_speed': fastest['exit_speed'],
                'optimal_brake_pressure': fastest['braking'].get('max_brake_pressure', 0),
                'optimal_duration_ms': fastest['duration_ms'],
                'reference_lap_time': fastest['lap_time']
            }

            self.optimal_lines[corner_id] = optimal_line

            log_metric_calculation(
                "optimal_line",
                f"corner={corner_id}, entry={optimal_line['optimal_entry_speed']:.1f} km/h"
            )

            return optimal_line

        except (AnalysisError, InvalidParameterError):
            # Re-raise validation/analysis errors as-is
            raise
        except Exception as e:
            context = {
                "method": "find_optimal_line",
                "corner_id": corner_id,
                "total_corners": len(self.corners),
                "telemetry_records": len(telemetry_df)
            }
            log_exception_with_context(e, context=context)
            raise AnalysisError(
                f"Optimal line calculation failed: {str(e)}",
                context=context
            )

    @log_method_performance
    def compare_to_optimal(
        self,
        corner_analysis: Dict,
        corner_id: int
    ) -> Dict:
        """
        Compare driver's corner to optimal line.

        Calculates deltas between driver's corner performance and the
        optimal line, generates actionable recommendations.

        Parameters:
            corner_analysis: Driver's corner analysis from analyze_corner_performance()
            corner_id: Corner identifier

        Returns:
            Comparison with deltas and recommendations

        Raises:
            InvalidParameterError: If inputs are invalid
            AnalysisError: If no optimal line found for corner

        Example:
            corner_perf = analyzer.analyze_corner_performance(telemetry_df, corners[0])
            comparison = analyzer.compare_to_optimal(corner_perf, corner_id=0)
            # {'entry_speed_delta': -2.5, 'recommendations': [...], ...}
        """
        # Validate inputs
        validate_positive_number(corner_id, "corner_id", allow_zero=True)

        if not isinstance(corner_analysis, dict):
            raise InvalidParameterError(
                f"corner_analysis must be a dictionary, got {type(corner_analysis).__name__}",
                context={"parameter": "corner_analysis"}
            )

        # Check if optimal line exists for this corner
        if corner_id not in self.optimal_lines:
            raise AnalysisError(
                f"No optimal line found for corner {corner_id}. Call find_optimal_line() first.",
                context={"corner_id": corner_id, "available_corners": list(self.optimal_lines.keys())}
            )

        try:
            optimal = self.optimal_lines[corner_id]

            comparison = {
                'corner_id': corner_id,
                'entry_speed_delta': corner_analysis['entry_speed'] - optimal['optimal_entry_speed'],
                'apex_speed_delta': corner_analysis['apex_speed'] - optimal['optimal_apex_speed'],
                'exit_speed_delta': corner_analysis['exit_speed'] - optimal['optimal_exit_speed'],
                'duration_delta_ms': corner_analysis['duration_ms'] - optimal['optimal_duration_ms']
            }

            # Generate recommendations
            recommendations = []

            if comparison['entry_speed_delta'] > 5:
                recommendations.append(
                    f"Entry speed {comparison['entry_speed_delta']:.1f} km/h too high. "
                    "Brake earlier or harder."
                )
            elif comparison['entry_speed_delta'] < -5:
                recommendations.append(
                    f"Entry speed {abs(comparison['entry_speed_delta']):.1f} km/h too low. "
                    "Carry more speed into the corner."
                )

            if comparison['apex_speed_delta'] < -5:
                recommendations.append(
                    f"Apex speed {abs(comparison['apex_speed_delta']):.1f} km/h slower than optimal. "
                    "Work on maintaining momentum through the turn."
                )

            if comparison['exit_speed_delta'] < -3:
                recommendations.append(
                    f"Exit speed {abs(comparison['exit_speed_delta']):.1f} km/h slower. "
                    "Apply throttle earlier or more aggressively."
                )

            if comparison['duration_delta_ms'] > 200:
                recommendations.append(
                    f"Corner duration {comparison['duration_delta_ms']}ms too long. "
                    "Tighten racing line and increase corner speed."
                )

            comparison['recommendations'] = recommendations if recommendations else [
                'Corner execution matches optimal line!'
            ]

            logger.debug(
                f"Corner {corner_id} comparison: "
                f"entry_delta={comparison['entry_speed_delta']:.1f}, "
                f"recommendations={len(recommendations)}"
            )

            return comparison

        except Exception as e:
            context = {
                "method": "compare_to_optimal",
                "corner_id": corner_id
            }
            log_exception_with_context(e, context=context)
            raise AnalysisError(
                f"Corner comparison failed: {str(e)}",
                context=context
            )

    @log_method_performance
    def generate_corner_report(
        self,
        telemetry_df: pd.DataFrame,
        lap_times_df: pd.DataFrame,
        vehicle_number: int
    ) -> pd.DataFrame:
        """
        Generate comprehensive corner-by-corner report for a driver.

        Parameters:
            telemetry_df: Telemetry data
            lap_times_df: Lap times data
            vehicle_number: Driver/vehicle identifier

        Returns:
            DataFrame with corner performance summary

        Raises:
            InvalidDataFrameError: If inputs are invalid
            InvalidParameterError: If vehicle_number is invalid
            EmptyDatasetError: If no data for vehicle

        Example:
            report = analyzer.generate_corner_report(telemetry_df, lap_times_df, vehicle_number=5)
            print(report)
        """
        # Validate inputs
        validate_telemetry_dataframe(telemetry_df, name="telemetry_df")
        validate_lap_times_dataframe(lap_times_df, name="lap_times_df")
        validate_vehicle_number(vehicle_number)

        try:
            # Filter to driver
            driver_telemetry = telemetry_df[
                telemetry_df[TelemetryColumns.VEHICLE_NUMBER] == vehicle_number
            ]

            if len(driver_telemetry) == 0:
                from .exceptions import EmptyDatasetError
                raise EmptyDatasetError(
                    f"No telemetry data found for vehicle {vehicle_number}",
                    context={"vehicle_number": vehicle_number}
                )

            # Detect corners
            corners = self.detect_corners(driver_telemetry)

            if len(corners) == 0:
                log_data_quality_warning(
                    "No corners detected for vehicle",
                    vehicle_number=vehicle_number
                )
                return pd.DataFrame()  # Empty report

            # Analyze each corner
            report_data = []

            for corner in corners:
                try:
                    analysis = self.analyze_corner_performance(telemetry_df, corner)

                    report_data.append({
                        'Corner': corner['corner_id'],
                        'Lap': corner.get(TelemetryColumns.LAP, 'N/A'),
                        'Entry Speed': f"{analysis['entry_speed']:.1f} km/h",
                        'Apex Speed': f"{analysis['apex_speed']:.1f} km/h",
                        'Exit Speed': f"{analysis['exit_speed']:.1f} km/h",
                        'Max Brake': f"{analysis['braking'].get('max_brake_pressure', 0):.1f} bar",
                        'Max Lateral G': f"{analysis['g_forces'].get('max_lateral_g', 0):.2f}g",
                        'Duration': f"{analysis['duration_ms']}ms"
                    })
                except AnalysisError as e:
                    # Skip corners that fail analysis
                    logger.debug(f"Skipping corner {corner['corner_id']}: {str(e)}")
                    continue

            logger.info(f"Generated corner report with {len(report_data)} corners for vehicle {vehicle_number}")

            return pd.DataFrame(report_data)

        except Exception as e:
            context = {
                "method": "generate_corner_report",
                "vehicle_number": vehicle_number
            }
            log_exception_with_context(e, context=context)
            raise AnalysisError(
                f"Corner report generation failed: {str(e)}",
                context=context
            )
