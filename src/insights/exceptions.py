"""
Custom Exception Hierarchy for Racing Insights Module

Production-grade exception framework providing structured error handling
for telemetry analysis, driver profiling, and performance tracking.

Design Pattern: Exception Hierarchy Pattern
- Base exception for module-level catching
- Specific exceptions for different failure scenarios
- Standardized error messages with context
- Machine-readable error codes for monitoring/alerting

Exception Hierarchy:
    InsightsError (base)
    ├── DataValidationError (input validation failures)
    │   ├── InvalidDataFrameError
    │   ├── MissingColumnsError
    │   ├── InvalidDataTypeError
    │   └── EmptyDatasetError
    ├── AnalysisError (analysis execution failures)
    │   ├── InsufficientDataError
    │   ├── CornerDetectionError
    │   └── ProfileGenerationError
    ├── ConfigurationError (configuration issues)
    │   ├── InvalidConfigurationError
    │   └── MissingConfigurationError
    └── PerformanceError (performance/resource issues)
        ├── MemoryExhaustedError
        └── TimeoutError

Author: Production Engineering Team
Version: 1.0.0
License: GR Cup 2025 Hackathon
"""

from typing import Optional, Dict, Any, List


class InsightsError(Exception):
    """
    Base exception for all insights module errors.

    All custom exceptions in the insights module inherit from this base class,
    allowing for module-level exception catching and handling.

    Attributes:
        message (str): Human-readable error message
        error_code (str): Machine-readable error code for monitoring
        context (Dict[str, Any]): Additional context information

    Example:
        try:
            profile = profiler.create_profile(telemetry, lap_times, vehicle_id)
        except InsightsError as e:
            logger.error(f"Analysis failed [{e.error_code}]: {e.message}")
            logger.debug(f"Error context: {e.context}")
    """

    def __init__(
        self,
        message: str,
        error_code: str = "INSIGHTS_ERROR",
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base exception.

        Parameters:
            message: Human-readable error description
            error_code: Machine-readable error code
            context: Additional context (vehicle_number, track, etc.)
        """
        self.message = message
        self.error_code = error_code
        self.context = context or {}

        # Construct detailed exception message
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format exception message with code and context."""
        msg = f"[{self.error_code}] {self.message}"

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            msg += f" | Context: {context_str}"

        return msg

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for API responses.

        Returns:
            Dictionary with error details suitable for JSON serialization
        """
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context
        }


# ==============================================================================
# DATA VALIDATION ERRORS
# ==============================================================================

class DataValidationError(InsightsError):
    """
    Base exception for input data validation failures.

    Raised when input data does not meet expected format, type, or quality
    requirements before analysis can proceed.
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATA_VALIDATION_ERROR",
            context=context
        )


class InvalidDataFrameError(DataValidationError):
    """
    Raised when input is not a valid pandas DataFrame.

    Example:
        raise InvalidDataFrameError(
            "Expected pandas DataFrame, got None",
            context={"parameter": "telemetry_df", "type_received": "NoneType"}
        )
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, context=context)
        self.error_code = "INVALID_DATAFRAME"


class MissingColumnsError(DataValidationError):
    """
    Raised when required DataFrame columns are missing.

    Example:
        raise MissingColumnsError(
            "Missing required columns for analysis",
            context={
                "required": ["telemetry_name", "telemetry_value"],
                "found": ["timestamp", "lap"],
                "missing": ["telemetry_name", "telemetry_value"]
            }
        )
    """

    def __init__(
        self,
        message: str,
        required_columns: List[str],
        found_columns: List[str],
        context: Optional[Dict[str, Any]] = None
    ):
        ctx = context or {}
        ctx.update({
            "required_columns": required_columns,
            "found_columns": found_columns,
            "missing_columns": list(set(required_columns) - set(found_columns))
        })
        super().__init__(message=message, context=ctx)
        self.error_code = "MISSING_COLUMNS"


class InvalidDataTypeError(DataValidationError):
    """
    Raised when DataFrame column has incorrect data type.

    Example:
        raise InvalidDataTypeError(
            "Column 'vehicle_number' must be integer",
            context={
                "column": "vehicle_number",
                "expected_type": "int64",
                "actual_type": "object"
            }
        )
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, context=context)
        self.error_code = "INVALID_DATA_TYPE"


class EmptyDatasetError(DataValidationError):
    """
    Raised when dataset is empty or has insufficient records.

    Example:
        raise EmptyDatasetError(
            "No telemetry data found for vehicle",
            context={"vehicle_number": 5, "record_count": 0}
        )
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, context=context)
        self.error_code = "EMPTY_DATASET"


class InvalidParameterError(DataValidationError):
    """
    Raised when method parameters are invalid.

    Example:
        raise InvalidParameterError(
            "Vehicle number must be in valid range",
            context={
                "parameter": "vehicle_number",
                "value": 25,
                "valid_range": "0-19"
            }
        )
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, context=context)
        self.error_code = "INVALID_PARAMETER"


# ==============================================================================
# ANALYSIS ERRORS
# ==============================================================================

class AnalysisError(InsightsError):
    """
    Base exception for analysis execution failures.

    Raised when analysis can proceed with validated input but fails during
    computation due to data characteristics or algorithmic issues.
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="ANALYSIS_ERROR",
            context=context
        )


class InsufficientDataError(AnalysisError):
    """
    Raised when data is valid but insufficient for meaningful analysis.

    Example:
        raise InsufficientDataError(
            "Need at least 3 laps for consistency analysis",
            context={"vehicle_number": 5, "lap_count": 1, "minimum_required": 3}
        )
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, context=context)
        self.error_code = "INSUFFICIENT_DATA"


class CornerDetectionError(AnalysisError):
    """
    Raised when corner detection algorithm fails.

    Example:
        raise CornerDetectionError(
            "No corners detected in telemetry data",
            context={
                "speed_threshold": 120.0,
                "telemetry_records": 50000,
                "corners_found": 0
            }
        )
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, context=context)
        self.error_code = "CORNER_DETECTION_FAILED"


class ProfileGenerationError(AnalysisError):
    """
    Raised when driver profile generation fails.

    Example:
        raise ProfileGenerationError(
            "Failed to calculate driving style metrics",
            context={
                "vehicle_number": 5,
                "stage": "driving_style_analysis",
                "reason": "No brake telemetry found"
            }
        )
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, context=context)
        self.error_code = "PROFILE_GENERATION_FAILED"


class ConsistencyAnalysisError(AnalysisError):
    """
    Raised when consistency tracking analysis fails.

    Example:
        raise ConsistencyAnalysisError(
            "Cannot calculate variance with single lap",
            context={"session_id": "session_1", "lap_count": 1}
        )
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, context=context)
        self.error_code = "CONSISTENCY_ANALYSIS_FAILED"


# ==============================================================================
# CONFIGURATION ERRORS
# ==============================================================================

class ConfigurationError(InsightsError):
    """
    Base exception for configuration-related errors.

    Raised when module configuration is invalid or missing.
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            context=context
        )


class InvalidConfigurationError(ConfigurationError):
    """
    Raised when configuration values are invalid.

    Example:
        raise InvalidConfigurationError(
            "Speed threshold must be positive",
            context={
                "parameter": "speed_threshold",
                "value": -120.0,
                "constraint": "> 0"
            }
        )
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, context=context)
        self.error_code = "INVALID_CONFIGURATION"


class MissingConfigurationError(ConfigurationError):
    """
    Raised when required configuration is missing.

    Example:
        raise MissingConfigurationError(
            "Required configuration parameter not set",
            context={"parameter": "corner_speed_threshold"}
        )
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, context=context)
        self.error_code = "MISSING_CONFIGURATION"


# ==============================================================================
# PERFORMANCE ERRORS
# ==============================================================================

class PerformanceError(InsightsError):
    """
    Base exception for performance and resource-related errors.

    Raised when analysis cannot complete due to resource constraints.
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="PERFORMANCE_ERROR",
            context=context
        )


class MemoryExhaustedError(PerformanceError):
    """
    Raised when memory limits are exceeded.

    Example:
        raise MemoryExhaustedError(
            "Dataset too large for available memory",
            context={
                "telemetry_records": 5000000,
                "estimated_memory_mb": 2500,
                "available_memory_mb": 1024
            }
        )
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, context=context)
        self.error_code = "MEMORY_EXHAUSTED"


class TimeoutError(PerformanceError):
    """
    Raised when analysis exceeds time limit.

    Example:
        raise TimeoutError(
            "Corner detection exceeded timeout",
            context={
                "timeout_seconds": 30,
                "elapsed_seconds": 45,
                "records_processed": 100000
            }
        )
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, context=context)
        self.error_code = "TIMEOUT"


# ==============================================================================
# SENSOR DATA ERRORS
# ==============================================================================

class SensorDataError(DataValidationError):
    """
    Raised when sensor-specific data validation fails.

    Example:
        raise SensorDataError(
            "Unknown sensor type in telemetry data",
            context={
                "sensor_name": "unknown_sensor",
                "valid_sensors": ["speed", "pbrake_f", "aps"]
            }
        )
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, context=context)
        self.error_code = "SENSOR_DATA_ERROR"


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def validate_and_raise(
    condition: bool,
    exception_class: type,
    message: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Validate condition and raise exception if False.

    Convenience function for inline validation with structured exceptions.

    Parameters:
        condition: Boolean condition to validate
        exception_class: Exception class to raise if condition is False
        message: Error message
        context: Additional context information

    Example:
        validate_and_raise(
            len(df) > 0,
            EmptyDatasetError,
            "DataFrame is empty",
            context={"parameter": "telemetry_df"}
        )

    Raises:
        exception_class: If condition is False
    """
    if not condition:
        raise exception_class(message=message, context=context)
