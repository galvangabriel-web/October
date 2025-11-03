"""
Driver Profiler Module - Production Grade

Analyzes driver-specific patterns, strengths, and weaknesses from telemetry data.
Generates comprehensive driver performance profiles for training and improvement.

Production Features:
- Structured exception handling with context
- Comprehensive input validation
- Performance logging and monitoring
- Configuration-driven thresholds
- Type-safe return structures
- Memory-efficient data processing
- Domain-appropriate error messages

Author: Production Engineering Team
Version: 2.0.0 (Production Grade)
License: GR Cup 2025 Hackathon
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Production infrastructure imports
from .exceptions import (
    ProfileGenerationError,
    EmptyDatasetError,
    InsufficientDataError,
    DataValidationError
)
from .validation import validate_profile_inputs
from .logger import (
    logger,
    log_method_performance,
    log_data_load,
    log_analysis_start,
    log_metric_calculation,
    log_data_quality_warning,
    log_exception_with_context
)
from .config import InsightsConfig, DEFAULT_CONFIG
from .constants import (
    TelemetrySensor,
    TelemetryColumns,
    LapTimesColumns,
    DrivingStyle,
    MetricNames
)
from .models import DriverProfile, BrakingProfile, ThrottleProfile, CorneringProfile, DrivingStyleEnum


class DriverProfiler:
    """
    Analyzes driver performance patterns and generates actionable insights.

    Production-grade driver profiling with:
    - Comprehensive input validation
    - Structured error handling
    - Performance monitoring
    - Configuration-driven thresholds
    - Type-safe outputs

    Features:
    - Driver skill assessment (aggression, smoothness, consistency)
    - Strength/weakness identification
    - Performance benchmarking against best laps
    - Sector-by-sector analysis

    Example:
        config = InsightsConfig(hard_brake_threshold=110)
        profiler = DriverProfiler(config=config)
        profile = profiler.create_profile(telemetry_df, lap_times_df, vehicle_number=5)
    """

    def __init__(self, config: Optional[InsightsConfig] = None):
        """
        Initialize driver profiler with configuration.

        Parameters:
            config: Configuration object (uses DEFAULT_CONFIG if None)

        Example:
            # Use default configuration
            profiler = DriverProfiler()

            # Custom configuration
            config = InsightsConfig(hard_brake_threshold=110)
            profiler = DriverProfiler(config=config)
        """
        self.config = config or DEFAULT_CONFIG
        self.profiles: Dict[int, Dict] = {}
        self.benchmarks: Dict[int, Dict] = {}

        logger.info(f"DriverProfiler initialized with config: log_level={self.config.log_level}")

    @log_method_performance
    def create_profile(
        self,
        telemetry_df: pd.DataFrame,
        lap_times_df: Optional[pd.DataFrame],
        vehicle_number: int
    ) -> Dict[str, Any]:
        """
        Create comprehensive driver profile.

        Performs full validation, analysis, and structured error handling.
        All thresholds are configuration-driven.

        Parameters:
            telemetry_df: Telemetry data in long format
            lap_times_df: Lap times data (optional - if None, some metrics will use defaults)
            vehicle_number: Driver/vehicle identifier (0-19)

        Returns:
            Dictionary with driver profile metrics and insights

        Raises:
            DataValidationError: If input data is invalid
            EmptyDatasetError: If no data found for driver
            ProfileGenerationError: If profile generation fails

        Example:
            profile = profiler.create_profile(telemetry_df, lap_times_df, vehicle_number=5)
            profile = profiler.create_profile(telemetry_df, None, vehicle_number=5)
        """
        log_analysis_start("driver profiling", vehicle_number=vehicle_number)

        try:
            # Validate inputs at entry point (fail-fast)
            validate_profile_inputs(telemetry_df, lap_times_df, vehicle_number)

            # Filter data for specific driver
            driver_telemetry = telemetry_df[
                telemetry_df[TelemetryColumns.VEHICLE_NUMBER] == vehicle_number
            ]

            # Filter lap times only if provided
            driver_laps = None
            if lap_times_df is not None:
                driver_laps = lap_times_df[
                    lap_times_df[LapTimesColumns.VEHICLE_NUMBER] == vehicle_number
                ]

            log_data_load("telemetry", len(driver_telemetry), vehicle_number=vehicle_number)
            if driver_laps is not None:
                log_data_load("lap_times", len(driver_laps), vehicle_number=vehicle_number)

            # Double-check filtered data (defensive programming)
            if len(driver_telemetry) == 0:
                raise EmptyDatasetError(
                    f"No telemetry data found for vehicle {vehicle_number}",
                    context={
                        "vehicle_number": vehicle_number,
                        "total_telemetry_records": len(telemetry_df)
                    }
                )

            if driver_laps is not None and len(driver_laps) == 0:
                logger.warning(
                    f"No lap times found for vehicle {vehicle_number}, "
                    f"some metrics will use default values"
                )
                driver_laps = None

            # Extract performance metrics with comprehensive error handling
            try:
                profile = {
                    'vehicle_number': vehicle_number,
                    'total_laps': len(driver_laps) if driver_laps is not None else 0,
                    'driving_style': self._analyze_driving_style(driver_telemetry),
                    'strengths': self._identify_strengths(driver_telemetry, driver_laps),
                    'weaknesses': self._identify_weaknesses(driver_telemetry, driver_laps),
                    'consistency_score': self._calculate_consistency(driver_laps),
                    'aggression_index': self._calculate_aggression(driver_telemetry),
                    'smoothness_index': self._calculate_smoothness(driver_telemetry),
                    'braking_profile': self._analyze_braking(driver_telemetry),
                    'throttle_profile': self._analyze_throttle(driver_telemetry),
                    'cornering_profile': self._analyze_cornering(driver_telemetry)
                }

                # Cache profile
                self.profiles[vehicle_number] = profile

                logger.info(
                    f"Profile created for vehicle {vehicle_number}: "
                    f"style={profile['driving_style']}, "
                    f"consistency={profile['consistency_score']:.1f}, "
                    f"aggression={profile['aggression_index']:.1f}"
                )

                # Return Pydantic model instead of dict for type safety
                # Fix for Issue #001: API Type Mismatch
                return DriverProfile(**profile)

            except Exception as e:
                raise ProfileGenerationError(
                    f"Failed to extract performance metrics for vehicle {vehicle_number}",
                    context={
                        "vehicle_number": vehicle_number,
                        "stage": "metric_extraction",
                        "error": str(e)
                    }
                )

        except (DataValidationError, EmptyDatasetError, ProfileGenerationError):
            # Re-raise structured exceptions
            raise
        except Exception as e:
            # Catch unexpected errors and wrap them
            log_exception_with_context(e, context={"vehicle_number": vehicle_number})
            raise ProfileGenerationError(
                f"Unexpected error during profile generation for vehicle {vehicle_number}",
                context={
                    "vehicle_number": vehicle_number,
                    "error_type": type(e).__name__,
                    "error": str(e)
                }
            )

    def _analyze_driving_style(self, telemetry_df: pd.DataFrame) -> str:
        """
        Classify driver's overall driving style.

        Uses configuration-driven thresholds for classification.

        Parameters:
            telemetry_df: Driver's telemetry data

        Returns:
            One of: 'Aggressive', 'Smooth', 'Conservative', 'Balanced', 'Unknown'

        Raises:
            ProfileGenerationError: If analysis fails
        """
        try:
            # Extract sensor data
            brake_data = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.BRAKE_PRESSURE_FRONT
            ]
            throttle_data = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.THROTTLE
            ]

            # Handle missing sensor data gracefully
            if len(brake_data) == 0 or len(throttle_data) == 0:
                log_data_quality_warning(
                    "Insufficient sensor data for driving style analysis",
                    brake_records=len(brake_data),
                    throttle_records=len(throttle_data)
                )
                return DrivingStyle.BALANCED.value  # Default fallback

            # Calculate metrics using DataFrame columns
            value_col = TelemetryColumns.TELEMETRY_VALUE
            avg_brake = brake_data[value_col].mean()
            max_brake = brake_data[value_col].max()
            brake_variance = brake_data[value_col].std()

            throttle_full_pct = (throttle_data[value_col] > self.config.full_throttle_threshold).mean() * 100

            log_metric_calculation("max_brake_pressure", max_brake)
            log_metric_calculation("brake_variance", brake_variance)
            log_metric_calculation("full_throttle_pct", throttle_full_pct)

            # Classification logic using config thresholds
            if max_brake > self.config.aggressive_brake_threshold and throttle_full_pct > self.config.aggressive_throttle_pct:
                style = DrivingStyle.AGGRESSIVE.value
            elif brake_variance < self.config.smooth_brake_variance and throttle_full_pct < self.config.smooth_throttle_pct:
                style = DrivingStyle.SMOOTH.value
            elif max_brake < self.config.conservative_brake_threshold and throttle_full_pct < self.config.conservative_throttle_pct:
                style = DrivingStyle.CONSERVATIVE.value
            else:
                style = DrivingStyle.BALANCED.value

            log_metric_calculation("driving_style", style)
            return style

        except Exception as e:
            log_exception_with_context(e, context={"stage": "driving_style_analysis"})
            raise ProfileGenerationError(
                "Failed to analyze driving style",
                context={"stage": "driving_style_analysis", "error": str(e)}
            )

    def _identify_strengths(
        self,
        telemetry_df: pd.DataFrame,
        lap_times_df: Optional[pd.DataFrame]
    ) -> List[str]:
        """
        Identify driver's key strengths.

        Parameters:
            telemetry_df: Driver's telemetry data
            lap_times_df: Driver's lap times (optional)

        Returns:
            List of strength descriptions
        """
        try:
            strengths = []
            value_col = TelemetryColumns.TELEMETRY_VALUE

            # Check braking strength (using config threshold)
            brake_data = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.BRAKE_PRESSURE_FRONT
            ]
            if len(brake_data) > 0:
                max_brake = brake_data[value_col].max()
                # Aggressive braking threshold + 10 bar for "strong" braking
                if max_brake > (self.config.aggressive_brake_threshold + 10):
                    strengths.append('Strong braking performance')
                    log_metric_calculation("braking_strength", max_brake)

            # Check consistency (using config threshold) - only if lap times provided
            if lap_times_df is not None and len(lap_times_df) >= self.config.min_laps_for_consistency:
                if LapTimesColumns.LAP_DURATION in lap_times_df.columns:
                    lap_times = lap_times_df[LapTimesColumns.LAP_DURATION].values
                    if len(lap_times) >= self.config.min_laps_for_consistency:
                        consistency = np.std(lap_times) / np.mean(lap_times)
                        if consistency < self.config.consistency_variance_threshold:
                            strengths.append('High consistency')
                            log_metric_calculation("lap_consistency", consistency)

            # Check throttle control
            throttle_data = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.THROTTLE
            ]
            if len(throttle_data) > 0:
                throttle_smoothness = 100 - (throttle_data[value_col].diff().abs().mean())
                if throttle_smoothness > 85:
                    strengths.append('Smooth throttle application')
                    log_metric_calculation("throttle_smoothness", throttle_smoothness)

            # Check cornering (using config threshold)
            accel_y = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.ACCEL_LATERAL
            ]
            if len(accel_y) > 0:
                max_lateral_g = accel_y[value_col].abs().max()
                if max_lateral_g > self.config.high_grip_lateral_g:
                    strengths.append('Strong cornering speed')
                    log_metric_calculation("max_lateral_g", max_lateral_g)

            return strengths if strengths else ['Developing driver - no clear strengths yet']

        except Exception as e:
            log_exception_with_context(e, context={"stage": "strengths_identification"})
            # Non-critical: return default on failure
            return ['Analysis incomplete - unable to identify strengths']

    def _identify_weaknesses(
        self,
        telemetry_df: pd.DataFrame,
        lap_times_df: Optional[pd.DataFrame]
    ) -> List[str]:
        """
        Identify areas for improvement.

        Parameters:
            telemetry_df: Driver's telemetry data
            lap_times_df: Driver's lap times (optional)

        Returns:
            List of weakness descriptions
        """
        try:
            weaknesses = []
            value_col = TelemetryColumns.TELEMETRY_VALUE

            # Check for excessive steering corrections
            steering_data = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.STEERING_ANGLE
            ]
            if len(steering_data) > 0:
                steering_changes = steering_data[value_col].diff().abs()
                if steering_changes.mean() > 5:  # Frequent large changes
                    weaknesses.append('Excessive steering corrections - work on line precision')

            # Check braking consistency
            brake_data = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.BRAKE_PRESSURE_FRONT
            ]
            if len(brake_data) > 0:
                brake_variance = brake_data[value_col].std()
                if brake_variance > 25:
                    weaknesses.append('Inconsistent braking points - focus on brake markers')

            # Check lap time variance (using config threshold) - only if lap times provided
            if lap_times_df is not None and len(lap_times_df) >= self.config.min_laps_for_consistency:
                if LapTimesColumns.LAP_DURATION in lap_times_df.columns:
                    lap_times = lap_times_df[LapTimesColumns.LAP_DURATION].values
                    if len(lap_times) >= self.config.min_laps_for_consistency:
                        consistency = np.std(lap_times) / np.mean(lap_times)
                        # 5% variance threshold for weakness
                        if consistency > 0.05:
                            weaknesses.append('Lap time inconsistency - work on repeatable execution')

            return weaknesses if weaknesses else ['No major weaknesses identified']

        except Exception as e:
            log_exception_with_context(e, context={"stage": "weaknesses_identification"})
            # Non-critical: return default on failure
            return ['Analysis incomplete - unable to identify weaknesses']

    def _calculate_consistency(self, lap_times_df: Optional[pd.DataFrame]) -> float:
        """
        Calculate consistency score (0-100).

        Higher score = more consistent lap times.

        Parameters:
            lap_times_df: Lap times DataFrame (optional - returns 0.0 if None)

        Returns:
            Consistency score (0-100)
        """
        try:
            if lap_times_df is None:
                log_data_quality_warning("No lap times provided for consistency calculation")
                return 0.0

            if len(lap_times_df) < 2:
                log_data_quality_warning("Insufficient laps for consistency", lap_count=len(lap_times_df))
                return 0.0

            if LapTimesColumns.LAP_DURATION not in lap_times_df.columns:
                log_data_quality_warning("Missing lap_duration column")
                return 0.0

            lap_times = lap_times_df[LapTimesColumns.LAP_DURATION].values

            # Coefficient of variation (inverted and scaled using config)
            cv = np.std(lap_times) / np.mean(lap_times)
            consistency_score = max(0, 100 * (1 - cv * self.config.consistency_score_multiplier))

            log_metric_calculation("consistency_score", consistency_score)
            return round(consistency_score, 2)

        except Exception as e:
            log_exception_with_context(e, context={"stage": "consistency_calculation"})
            return 0.0

    def _calculate_aggression(self, telemetry_df: pd.DataFrame) -> float:
        """
        Calculate aggression index (0-100).

        Higher score = more aggressive driving (hard braking, full throttle).

        Parameters:
            telemetry_df: Telemetry DataFrame

        Returns:
            Aggression index (0-100)
        """
        try:
            value_col = TelemetryColumns.TELEMETRY_VALUE

            brake_data = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.BRAKE_PRESSURE_FRONT
            ]
            throttle_data = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.THROTTLE
            ]

            if len(brake_data) == 0 or len(throttle_data) == 0:
                log_data_quality_warning("Insufficient data for aggression calculation")
                return 0.0

            # High brake pressure percentage (using config threshold)
            hard_brake_pct = (brake_data[value_col] > self.config.hard_brake_threshold).mean() * 100

            # Full throttle percentage (using config threshold)
            full_throttle_pct = (throttle_data[value_col] > self.config.full_throttle_threshold).mean() * 100

            # Combined aggression index
            aggression = (hard_brake_pct * 0.5 + full_throttle_pct * 0.5)

            log_metric_calculation("aggression_index", aggression)
            return round(aggression, 2)

        except Exception as e:
            log_exception_with_context(e, context={"stage": "aggression_calculation"})
            return 0.0

    def _calculate_smoothness(self, telemetry_df: pd.DataFrame) -> float:
        """
        Calculate smoothness index (0-100).

        Higher score = smoother inputs (less abrupt changes).

        Parameters:
            telemetry_df: Telemetry DataFrame

        Returns:
            Smoothness index (0-100)
        """
        try:
            value_col = TelemetryColumns.TELEMETRY_VALUE

            steering_data = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.STEERING_ANGLE
            ]
            throttle_data = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.THROTTLE
            ]
            brake_data = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.BRAKE_PRESSURE_FRONT
            ]

            smoothness_scores = []

            # Steering smoothness
            if len(steering_data) > 0:
                steering_rate = steering_data[value_col].diff().abs().mean()
                steering_smoothness = max(0, 100 - steering_rate * 2)
                smoothness_scores.append(steering_smoothness)

            # Throttle smoothness
            if len(throttle_data) > 0:
                throttle_rate = throttle_data[value_col].diff().abs().mean()
                throttle_smoothness = max(0, 100 - throttle_rate)
                smoothness_scores.append(throttle_smoothness)

            # Brake smoothness
            if len(brake_data) > 0:
                brake_rate = brake_data[value_col].diff().abs().mean()
                brake_smoothness = max(0, 100 - brake_rate * 0.5)
                smoothness_scores.append(brake_smoothness)

            if not smoothness_scores:
                log_data_quality_warning("No data for smoothness calculation")
                return 0.0

            smoothness = np.mean(smoothness_scores)
            log_metric_calculation("smoothness_index", smoothness)
            return round(smoothness, 2)

        except Exception as e:
            log_exception_with_context(e, context={"stage": "smoothness_calculation"})
            return 0.0

    def _analyze_braking(self, telemetry_df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze braking characteristics.

        Parameters:
            telemetry_df: Telemetry DataFrame

        Returns:
            Dictionary with braking metrics
        """
        try:
            value_col = TelemetryColumns.TELEMETRY_VALUE

            brake_f = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.BRAKE_PRESSURE_FRONT
            ]

            if len(brake_f) == 0:
                log_data_quality_warning("No brake data found")
                return {}

            # Calculate metrics (using config threshold for hard braking)
            profile = {
                'max_brake_pressure': round(brake_f[value_col].max(), 2),
                'avg_brake_pressure': round(brake_f[value_col].mean(), 2),
                'brake_consistency': round(100 - brake_f[value_col].std(), 2),
                'brake_variance': round(brake_f[value_col].std(), 2),
                'hard_braking_events': int((brake_f[value_col] > self.config.aggressive_brake_threshold).sum())
            }

            log_metric_calculation("braking_profile", profile)
            return profile

        except Exception as e:
            log_exception_with_context(e, context={"stage": "braking_analysis"})
            return {}

    def _analyze_throttle(self, telemetry_df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze throttle application.

        Parameters:
            telemetry_df: Telemetry DataFrame

        Returns:
            Dictionary with throttle metrics
        """
        try:
            value_col = TelemetryColumns.TELEMETRY_VALUE

            throttle = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.THROTTLE
            ]

            if len(throttle) == 0:
                log_data_quality_warning("No throttle data found")
                return {}

            # Calculate metrics (using config thresholds)
            profile = {
                'avg_throttle': round(throttle[value_col].mean(), 2),
                'full_throttle_percentage': round((throttle[value_col] > self.config.full_throttle_threshold).mean() * 100, 2),
                'throttle_smoothness': round(100 - throttle[value_col].diff().abs().mean(), 2),
                'partial_throttle_percentage': round(
                    (throttle[value_col].between(self.config.partial_throttle_threshold, self.config.full_throttle_threshold)).mean() * 100, 2
                )
            }

            log_metric_calculation("throttle_profile", profile)
            return profile

        except Exception as e:
            log_exception_with_context(e, context={"stage": "throttle_analysis"})
            return {}

    def _analyze_cornering(self, telemetry_df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze cornering performance.

        Parameters:
            telemetry_df: Telemetry DataFrame

        Returns:
            Dictionary with cornering metrics
        """
        try:
            value_col = TelemetryColumns.TELEMETRY_VALUE

            accel_x = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.ACCEL_LONGITUDINAL
            ]
            accel_y = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.ACCEL_LATERAL
            ]
            speed = telemetry_df[
                telemetry_df[TelemetryColumns.TELEMETRY_NAME] == TelemetrySensor.SPEED
            ]

            if len(accel_y) == 0:
                log_data_quality_warning("No lateral acceleration data found")
                return {}

            profile = {
                'max_lateral_g': round(accel_y[value_col].abs().max(), 2),
                'avg_lateral_g': round(accel_y[value_col].abs().mean(), 2),
                'avg_cornering_speed': 0.0  # Default value
            }

            if len(accel_x) > 0:
                profile['max_longitudinal_g'] = round(accel_x[value_col].abs().max(), 2)

            # Cornering speed index (using config threshold)
            if len(speed) > 0 and len(accel_y) > 0:
                high_g_moments = accel_y[value_col].abs() > self.config.cornering_g_threshold
                if high_g_moments.any():
                    speed_aligned = speed[speed.index.isin(accel_y[high_g_moments].index)]
                    if len(speed_aligned) > 0:
                        profile['avg_cornering_speed'] = round(speed_aligned[value_col].mean(), 2)

            log_metric_calculation("cornering_profile", profile)
            return profile

        except Exception as e:
            log_exception_with_context(e, context={"stage": "cornering_analysis"})
            return {}

    @log_method_performance
    def compare_drivers(
        self,
        vehicle_numbers: List[int]
    ) -> pd.DataFrame:
        """
        Compare multiple drivers side-by-side.

        Parameters:
            vehicle_numbers: List of vehicle numbers to compare

        Returns:
            DataFrame with comparison table

        Example:
            comparison_df = profiler.compare_drivers([5, 7, 12])
        """
        try:
            log_analysis_start("driver comparison", vehicle_count=len(vehicle_numbers))

            comparison_data = []

            for vehicle_num in vehicle_numbers:
                if vehicle_num in self.profiles:
                    profile = self.profiles[vehicle_num]
                    comparison_data.append({
                        'Vehicle': vehicle_num,
                        'Style': profile.get('driving_style', 'Unknown'),
                        'Consistency': profile.get('consistency_score', 0),
                        'Aggression': profile.get('aggression_index', 0),
                        'Smoothness': profile.get('smoothness_index', 0),
                        'Laps': profile.get('total_laps', 0)
                    })
                else:
                    log_data_quality_warning(
                        "Profile not found for comparison",
                        vehicle_number=vehicle_num
                    )

            if not comparison_data:
                logger.warning("No profiles available for comparison")
                return pd.DataFrame()

            return pd.DataFrame(comparison_data)

        except Exception as e:
            log_exception_with_context(e, context={"stage": "driver_comparison"})
            raise ProfileGenerationError(
                "Failed to compare drivers",
                context={"vehicle_numbers": vehicle_numbers, "error": str(e)}
            )

    @log_method_performance
    def generate_recommendations(
        self,
        vehicle_number: int
    ) -> List[str]:
        """
        Generate training recommendations for a driver.

        Parameters:
            vehicle_number: Driver/vehicle identifier

        Returns:
            List of actionable recommendations

        Example:
            recommendations = profiler.generate_recommendations(vehicle_number=5)
        """
        try:
            if vehicle_number not in self.profiles:
                log_data_quality_warning(
                    "No profile data available for recommendations",
                    vehicle_number=vehicle_number
                )
                return ['No profile data available. Create profile first.']

            profile = self.profiles[vehicle_number]
            recommendations = []

            # Consistency recommendations (using config threshold)
            consistency = profile.get('consistency_score', 0)
            if consistency < 70:
                recommendations.append(
                    f"Focus on consistency (current: {consistency:.1f}/100). "
                    "Practice repeatable braking points and turn-in markers."
                )

            # Smoothness recommendations
            smoothness = profile.get('smoothness_index', 0)
            if smoothness < 75:
                recommendations.append(
                    f"Improve input smoothness (current: {smoothness:.1f}/100). "
                    "Work on gradual steering, throttle, and brake application."
                )

            # Braking recommendations
            braking = profile.get('braking_profile', {})
            if braking.get('brake_consistency', 100) < 80:
                recommendations.append(
                    "Inconsistent braking detected. "
                    "Use consistent brake markers and practice trail braking."
                )

            # Cornering recommendations (using config threshold)
            cornering = profile.get('cornering_profile', {})
            max_lateral_g = cornering.get('max_lateral_g', 0)
            if max_lateral_g < 1.0:
                recommendations.append(
                    f"Low cornering forces detected ({max_lateral_g:.2f}g). "
                    "Work on carrying more speed through corners and later braking."
                )

            # Add strengths to reinforce
            strengths = profile.get('strengths', [])
            if strengths:
                recommendations.append(
                    f"Maintain your strengths: {', '.join(strengths[:2])}"
                )

            return recommendations if recommendations else ['Profile looks strong! Keep refining your technique.']

        except Exception as e:
            log_exception_with_context(e, context={"vehicle_number": vehicle_number})
            return ['Error generating recommendations. Please try again.']
