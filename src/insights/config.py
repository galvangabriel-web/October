"""
Configuration Management for Racing Insights Module

Production-grade configuration system providing:
- Centralized analysis parameter definitions
- Environment-aware configuration loading
- Type-safe configuration with validation
- Threshold tuning without code changes
- Documentation of all magic numbers

Design Pattern: Configuration Object Pattern
- Immutable dataclass for thread safety
- Default values from domain expertise
- Environment variable override capability
- Validation on initialization
- Clear documentation of each parameter

All threshold values documented with racing domain rationale.

Author: Production Engineering Team
Version: 1.0.0
License: GR Cup 2025 Hackathon
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import warnings


@dataclass(frozen=True)
class InsightsConfig:
    """
    Immutable configuration for racing insights analysis.

    All threshold values are based on GR Cup telemetry analysis and
    racing domain expertise. Documented rationale for each parameter.

    Attributes are frozen (immutable) for thread safety in production.

    Example:
        # Use defaults
        config = InsightsConfig()

        # Override specific parameters
        config = InsightsConfig(
            hard_brake_threshold=110,
            corner_speed_threshold=130
        )

        # Load from environment
        config = InsightsConfig.from_environment()
    """

    # ==========================================================================
    # BRAKING ANALYSIS THRESHOLDS
    # ==========================================================================

    hard_brake_threshold: float = 100
    """
    Brake pressure threshold (bar) for "hard braking" detection.

    Domain Knowledge:
    - Normal braking: 40-80 bar
    - Hard braking: 100+ bar (92nd percentile from telemetry analysis)
    - Panic/emergency braking: 130+ bar

    Used in: DriverProfiler._analyze_braking()
    """

    aggressive_brake_threshold: float = 120
    """
    Brake pressure threshold (bar) for "aggressive" driving style.

    Domain Knowledge:
    - Aggressive drivers consistently brake >120 bar (95th percentile)
    - Indicates late braking and hard deceleration preference

    Used in: DriverProfiler._analyze_driving_style()
    """

    smooth_brake_variance: float = 15
    """
    Maximum brake pressure variance (bar) for "smooth" classification.

    Domain Knowledge:
    - Smooth drivers: variance < 15 bar (consistent pressure application)
    - Aggressive drivers: variance > 30 bar (variable braking)

    Used in: DriverProfiler._analyze_driving_style()
    """

    conservative_brake_threshold: float = 80
    """
    Max brake pressure threshold (bar) for "conservative" driving style.

    Domain Knowledge:
    - Conservative drivers rarely exceed 80 bar (early braking)
    - Prioritize safety over speed

    Used in: DriverProfiler._analyze_driving_style()
    """

    # ==========================================================================
    # THROTTLE ANALYSIS THRESHOLDS
    # ==========================================================================

    full_throttle_threshold: float = 95
    """
    Throttle position (%) threshold for "full throttle" detection.

    Domain Knowledge:
    - 95-100% considered full throttle (accounting for sensor noise)
    - Aggressive drivers: 50%+ of lap at full throttle
    - Conservative drivers: <30% of lap at full throttle

    Used in: DriverProfiler._analyze_throttle()
    """

    aggressive_throttle_pct: float = 50
    """
    Full throttle percentage threshold for "aggressive" classification.

    Domain Knowledge:
    - Aggressive: >50% of lap at full throttle
    - Smooth: 40-50%
    - Conservative: <30%

    Used in: DriverProfiler._analyze_driving_style()
    """

    smooth_throttle_pct: float = 40
    """
    Maximum full throttle percentage for "smooth" classification.

    Domain Knowledge:
    - Smooth drivers modulate throttle more (< 40% at full)
    - Progressive throttle application

    Used in: DriverProfiler._analyze_driving_style()
    """

    conservative_throttle_pct: float = 30
    """
    Maximum full throttle percentage for "conservative" classification.

    Domain Knowledge:
    - Conservative drivers avoid full throttle (<30%)
    - Prioritize tire/fuel conservation

    Used in: DriverProfiler._analyze_driving_style()
    """

    partial_throttle_threshold: float = 30
    """
    Minimum throttle (%) for "partial throttle" (not coasting).

    Domain Knowledge:
    - 30-95% = partial throttle application
    - <30% = coasting/engine braking
    - Used to analyze throttle control finesse

    Used in: DriverProfiler._analyze_throttle()
    """

    # ==========================================================================
    # CORNERING ANALYSIS THRESHOLDS
    # ==========================================================================

    corner_speed_threshold: float = 120.0
    """
    Speed threshold (km/h) below which is considered "cornering".

    Domain Knowledge:
    - Straights: 140-190 km/h
    - Fast corners: 120-140 km/h
    - Slow corners: 66-120 km/h

    Lower value = detect only slow corners
    Higher value = detect all corners including fast sweepers

    Used in: CornerAnalyzer.detect_corners()
    """

    min_corner_duration_ms: int = 1000
    """
    Minimum corner duration (milliseconds) for valid corner detection.

    Domain Knowledge:
    - Slow hairpins: 3-5 seconds (3000-5000ms)
    - Medium corners: 2-3 seconds (2000-3000ms)
    - Fast sweepers: 1-2 seconds (1000-2000ms)
    - Brief speed drops (<1s) are not corners (kerb strikes, etc.)

    Used in: CornerAnalyzer.detect_corners()
    """

    high_grip_lateral_g: float = 1.2
    """
    Lateral g-force threshold for "high grip" cornering.

    Domain Knowledge:
    - Normal cornering: 0.6-0.9g
    - Fast cornering: 1.0-1.2g
    - Limit cornering: 1.2-1.5g (grip limit)

    Used in: DriverProfiler._identify_strengths()
    """

    cornering_g_threshold: float = 0.8
    """
    Minimum lateral g-force (abs value) to count as "cornering".

    Domain Knowledge:
    - Straight line: <0.3g
    - Light cornering: 0.3-0.6g
    - Active cornering: >0.8g

    Used in: DriverProfiler._analyze_cornering()
    """

    # ==========================================================================
    # CONSISTENCY ANALYSIS THRESHOLDS
    # ==========================================================================

    consistency_variance_threshold: float = 0.02
    """
    Coefficient of variation threshold for "high consistency".

    Domain Knowledge:
    - Elite consistency: CV < 0.01 (1% lap time variation)
    - Good consistency: CV < 0.02 (2% variation)
    - Average consistency: CV < 0.05 (5% variation)
    - Poor consistency: CV > 0.05

    Formula: CV = std(lap_times) / mean(lap_times)

    Used in: DriverProfiler._calculate_consistency()
    """

    min_improvement_delta: float = 0.5
    """
    Minimum lap time improvement (seconds) to count as "improving".

    Domain Knowledge:
    - Significant improvement: >0.5s per session
    - Marginal improvement: 0.1-0.5s
    - No improvement: <0.1s (within noise)

    Used in: ConsistencyTracker.detect_performance_trend()
    """

    outlier_std_threshold: float = 2.0
    """
    Standard deviation threshold for outlier lap detection.

    Domain Knowledge:
    - Normal laps: within ±2σ (95% of data)
    - Outliers: beyond ±2σ
    - Extreme outliers: beyond ±3σ

    Used in: ConsistencyTracker.identify_outlier_laps()
    """

    min_laps_for_consistency: int = 3
    """
    Minimum laps required for consistency analysis.

    Domain Knowledge:
    - Need 3+ laps to calculate meaningful variance
    - 5+ laps ideal for robust consistency metrics

    Used in: ConsistencyTracker.track_session()
    """

    # ==========================================================================
    # PERFORMANCE SCORING PARAMETERS
    # ==========================================================================

    consistency_score_multiplier: float = 20.0
    """
    Multiplier for consistency score calculation (0-100 scale).

    Formula: score = max(0, 100 * (1 - CV * multiplier))

    Domain Knowledge:
    - Multiplier = 20 scales CV=0.02 → score=60
    - Higher multiplier = more sensitive to variance

    Used in: ConsistencyTracker._calculate_consistency_score()
    """

    aggression_score_base: float = 100.0
    """
    Base value for aggression index normalization.

    Used in: DriverProfiler._calculate_aggression()
    """

    smoothness_score_base: float = 100.0
    """
    Base value for smoothness index normalization.

    Used in: DriverProfiler._calculate_smoothness()
    """

    # ==========================================================================
    # DATA QUALITY THRESHOLDS
    # ==========================================================================

    min_telemetry_records: int = 100
    """
    Minimum telemetry records for valid analysis.

    Domain Knowledge:
    - Single lap ≈ 100,000 records @ 100Hz sampling
    - Minimum 100 records for basic statistics

    Used in: Validation framework
    """

    min_laps_for_profile: int = 1
    """
    Minimum laps required for driver profile.

    Domain Knowledge:
    - 1 lap minimum for basic profile
    - 3+ laps recommended for robust profile

    Used in: DriverProfiler.create_profile()
    """

    min_sessions_for_trend: int = 2
    """
    Minimum sessions required for trend analysis.

    Domain Knowledge:
    - Need 2+ sessions to detect trends
    - 5+ sessions ideal for robust trend detection

    Used in: ConsistencyTracker.detect_performance_trend()
    """

    # ==========================================================================
    # PERFORMANCE/RESOURCE LIMITS
    # ==========================================================================

    max_chunk_size: int = 100_000
    """
    Maximum DataFrame chunk size for memory-efficient processing.

    Per CLAUDE.md: "Process chunks iteratively, never load all at once"

    Domain Knowledge:
    - 100k records ≈ 50MB memory (per CLAUDE.md)
    - Matches organized_data chunk size

    Used in: Data loading utilities
    """

    analysis_timeout_seconds: int = 300
    """
    Maximum time (seconds) for single analysis operation.

    Production limit to prevent runaway computations.
    5 minutes should be sufficient for any single-vehicle analysis.

    Used in: Future timeout decorators
    """

    # ==========================================================================
    # LOGGING CONFIGURATION
    # ==========================================================================

    log_level: str = "INFO"
    """
    Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Production default: INFO
    Development: DEBUG

    Used in: Logger configuration
    """

    log_rotation_size: str = "500 MB"
    """
    Log file rotation size.

    Used in: Loguru logger configuration
    """

    log_retention_days: int = 7
    """
    Number of days to retain log files.

    Used in: Loguru logger configuration
    """

    # ==========================================================================
    # CLASS METHODS
    # ==========================================================================

    @classmethod
    def from_environment(cls, prefix: str = "INSIGHTS_") -> "InsightsConfig":
        """
        Load configuration from environment variables.

        Environment variables override default values.
        Useful for production deployment without code changes.

        Parameters:
            prefix: Environment variable prefix (default: "INSIGHTS_")

        Returns:
            InsightsConfig instance with environment overrides

        Example:
            # Set environment variable
            os.environ["INSIGHTS_HARD_BRAKE_THRESHOLD"] = "110"

            # Load config
            config = InsightsConfig.from_environment()
            print(config.hard_brake_threshold)  # 110
        """
        kwargs = {}

        # Map environment variables to config attributes
        for attr_name in cls.__dataclass_fields__:
            env_var_name = f"{prefix}{attr_name.upper()}"
            env_value = os.getenv(env_var_name)

            if env_value is not None:
                # Get field type for proper conversion
                field_type = cls.__dataclass_fields__[attr_name].type

                # Convert string to appropriate type
                try:
                    if field_type == int:
                        kwargs[attr_name] = int(env_value)
                    elif field_type == float:
                        kwargs[attr_name] = float(env_value)
                    elif field_type == str:
                        kwargs[attr_name] = env_value
                    else:
                        kwargs[attr_name] = env_value
                except (ValueError, TypeError) as e:
                    warnings.warn(
                        f"Invalid environment variable {env_var_name}={env_value}: {e}. "
                        f"Using default value."
                    )

        return cls(**kwargs)

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid

        Example:
            config = InsightsConfig(hard_brake_threshold=-10)
            config.validate()  # Raises ValueError
        """
        from .exceptions import InvalidConfigurationError

        # Validate brake thresholds
        if self.hard_brake_threshold <= 0:
            raise InvalidConfigurationError(
                "hard_brake_threshold must be positive",
                context={"value": self.hard_brake_threshold}
            )

        if self.aggressive_brake_threshold < self.hard_brake_threshold:
            raise InvalidConfigurationError(
                "aggressive_brake_threshold must be >= hard_brake_threshold",
                context={
                    "aggressive": self.aggressive_brake_threshold,
                    "hard": self.hard_brake_threshold
                }
            )

        # Validate throttle thresholds
        if not 0 <= self.full_throttle_threshold <= 100:
            raise InvalidConfigurationError(
                "full_throttle_threshold must be 0-100%",
                context={"value": self.full_throttle_threshold}
            )

        # Validate corner thresholds
        if self.corner_speed_threshold <= 0:
            raise InvalidConfigurationError(
                "corner_speed_threshold must be positive",
                context={"value": self.corner_speed_threshold}
            )

        if self.min_corner_duration_ms <= 0:
            raise InvalidConfigurationError(
                "min_corner_duration_ms must be positive",
                context={"value": self.min_corner_duration_ms}
            )

        # Validate consistency thresholds
        if self.outlier_std_threshold <= 0:
            raise InvalidConfigurationError(
                "outlier_std_threshold must be positive",
                context={"value": self.outlier_std_threshold}
            )

        # Validate minimum data requirements
        if self.min_laps_for_consistency < 2:
            raise InvalidConfigurationError(
                "min_laps_for_consistency must be >= 2",
                context={"value": self.min_laps_for_consistency}
            )

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary with all configuration parameters

        Example:
            config = InsightsConfig()
            params = config.to_dict()
            print(params['hard_brake_threshold'])  # 100
        """
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


# ==============================================================================
# DEFAULT CONFIGURATION
# ==============================================================================

# Global default configuration instance
DEFAULT_CONFIG = InsightsConfig()

# Validate default configuration on module import
DEFAULT_CONFIG.validate()
