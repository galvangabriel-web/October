"""
Racing Insights & Analysis Module - Production Grade

Enterprise-grade module providing advanced driver profiling, racing line optimization,
and performance insights based on telemetry data.

Features:
- Driver Performance Profiling: Comprehensive driver analysis with driving style classification
- Racing Line Optimization: Corner-by-corner analysis and optimal line identification
- Consistency Tracking: Session-over-session performance monitoring
- Production Infrastructure: Validation, logging, exceptions, configuration

Architecture:
- Analysis Classes: DriverProfiler, CornerAnalyzer, ConsistencyTracker
- Infrastructure: Validation, logging, exceptions, configuration
- Type Safety: Pydantic models for all return types
- Error Handling: Structured exceptions with context

Author: Production Engineering Team
Version: 2.0.0 (Production)
License: GR Cup 2025 Hackathon
"""

# ==============================================================================
# MAIN ANALYSIS CLASSES
# ==============================================================================

from .driver_profiler import DriverProfiler
from .corner_analyzer import CornerAnalyzer
from .consistency_tracker import ConsistencyTracker

# Phase 1.2: Enhanced Driver Insights
from .ghost_lap_comparator import GhostLapComparator, GhostLapComparison
from .brake_point_analyzer import BrakePointAnalyzer, BrakeAnalysis, BrakeZone
from .corner_speed_benchmarking import CornerSpeedBenchmarking, CornerBenchmarkAnalysis, CornerSpeed


# ==============================================================================
# PRODUCTION INFRASTRUCTURE
# ==============================================================================

# Configuration
from .config import InsightsConfig, DEFAULT_CONFIG

# Logging
from .logger import (
    logger,
    configure_logger,
    log_performance,
    log_method_performance,
    log_data_load,
    log_analysis_start,
    log_metric_calculation,
    log_data_quality_warning,
    log_corner_detection,
    log_exception_with_context
)

# Exceptions
from .exceptions import (
    InsightsError,
    DataValidationError,
    InvalidDataFrameError,
    MissingColumnsError,
    InvalidDataTypeError,
    EmptyDatasetError,
    InvalidParameterError,
    AnalysisError,
    InsufficientDataError,
    CornerDetectionError,
    ProfileGenerationError,
    ConsistencyAnalysisError,
    ConfigurationError,
    InvalidConfigurationError,
    MissingConfigurationError,
    PerformanceError,
    MemoryExhaustedError,
    TimeoutError,
    SensorDataError,
    validate_and_raise
)

# Constants
from .constants import (
    TelemetrySensor,
    TelemetryColumns,
    LapTimesColumns,
    SensorRanges,
    DrivingStyle,
    PerformanceTrend,
    OutlierType,
    MetricNames,
    TrackName,
    MinimumDataRequirements,
    ErrorMessages,
    LogMessages,
    get_sensor_range,
    is_valid_vehicle_number
)

# Validation
from .validation import (
    validate_dataframe,
    validate_columns,
    validate_column_type,
    validate_telemetry_dataframe,
    validate_lap_times_dataframe,
    validate_vehicle_number,
    validate_positive_number,
    validate_percentage,
    validate_string_not_empty,
    validate_sufficient_data,
    validate_vehicle_has_data,
    validate_profile_inputs,
    validate_corner_detection_inputs,
    validate_consistency_inputs
)

# Type-Safe Models
from .models import (
    # Enums
    DrivingStyleEnum,
    PerformanceTrendEnum,
    OutlierTypeEnum,

    # Profile Models
    BrakingProfile,
    ThrottleProfile,
    CorneringProfile,
    DriverProfile,

    # Corner Analysis Models
    CornerData,
    CornerPerformance,
    OptimalLine,

    # Consistency Models
    SessionMetrics,
    PerformanceTrend as PerformanceTrendModel,
    OutlierLap,

    # Utility Models
    DriverComparison,
    ErrorResponse,
    AnalysisProgress,

    # Conversion Functions
    driver_profile_from_dict
)


# ==============================================================================
# PUBLIC API
# ==============================================================================

__all__ = [
    # Main Analysis Classes
    'DriverProfiler',
    'CornerAnalyzer',
    'ConsistencyTracker',

    # Phase 1.2: Enhanced Driver Insights
    'GhostLapComparator',
    'GhostLapComparison',
    'BrakePointAnalyzer',
    'BrakeAnalysis',
    'BrakeZone',
    'CornerSpeedBenchmarking',
    'CornerBenchmarkAnalysis',
    'CornerSpeed',

    # Configuration
    'InsightsConfig',
    'DEFAULT_CONFIG',

    # Logging
    'logger',
    'configure_logger',
    'log_performance',
    'log_method_performance',
    'log_data_load',
    'log_analysis_start',
    'log_metric_calculation',
    'log_data_quality_warning',
    'log_corner_detection',
    'log_exception_with_context',

    # Exceptions
    'InsightsError',
    'DataValidationError',
    'InvalidDataFrameError',
    'MissingColumnsError',
    'InvalidDataTypeError',
    'EmptyDatasetError',
    'InvalidParameterError',
    'AnalysisError',
    'InsufficientDataError',
    'CornerDetectionError',
    'ProfileGenerationError',
    'ConsistencyAnalysisError',
    'ConfigurationError',
    'InvalidConfigurationError',
    'MissingConfigurationError',
    'PerformanceError',
    'MemoryExhaustedError',
    'TimeoutError',
    'SensorDataError',
    'validate_and_raise',

    # Constants
    'TelemetrySensor',
    'TelemetryColumns',
    'LapTimesColumns',
    'SensorRanges',
    'DrivingStyle',
    'PerformanceTrend',
    'OutlierType',
    'MetricNames',
    'TrackName',
    'MinimumDataRequirements',
    'ErrorMessages',
    'LogMessages',
    'get_sensor_range',
    'is_valid_vehicle_number',

    # Validation
    'validate_dataframe',
    'validate_columns',
    'validate_column_type',
    'validate_telemetry_dataframe',
    'validate_lap_times_dataframe',
    'validate_vehicle_number',
    'validate_positive_number',
    'validate_percentage',
    'validate_string_not_empty',
    'validate_sufficient_data',
    'validate_vehicle_has_data',
    'validate_profile_inputs',
    'validate_corner_detection_inputs',
    'validate_consistency_inputs',

    # Models
    'DrivingStyleEnum',
    'PerformanceTrendEnum',
    'OutlierTypeEnum',
    'BrakingProfile',
    'ThrottleProfile',
    'CorneringProfile',
    'DriverProfile',
    'CornerData',
    'CornerPerformance',
    'OptimalLine',
    'SessionMetrics',
    'PerformanceTrendModel',
    'OutlierLap',
    'DriverComparison',
    'ErrorResponse',
    'AnalysisProgress',
    'driver_profile_from_dict',
]


# ==============================================================================
# VERSION & METADATA
# ==============================================================================

__version__ = "2.1.0"  # Phase 1.2: Enhanced Driver Insights
__author__ = "Production Engineering Team"
__status__ = "Production"
