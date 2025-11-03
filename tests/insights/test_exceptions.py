"""
Unit Tests for Exception Hierarchy

Tests all custom exceptions in src.insights.exceptions:
- Base exception functionality
- Exception hierarchy
- Error codes and context
- Message formatting
- to_dict() serialization
"""

import pytest
from src.insights.exceptions import (
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


class TestBaseException:
    """Test InsightsError base exception."""

    def test_basic_exception(self):
        """Test exception with just a message."""
        exc = InsightsError("Something went wrong")
        assert exc.message == "Something went wrong"
        assert exc.error_code == "INSIGHTS_ERROR"
        assert exc.context == {}

    def test_exception_with_code(self):
        """Test exception with custom error code."""
        exc = InsightsError("Error", error_code="CUSTOM_001")
        assert exc.error_code == "CUSTOM_001"

    def test_exception_with_context(self, error_context):
        """Test exception with context dictionary."""
        exc = InsightsError("Error", context=error_context)
        assert exc.context == error_context
        assert 'vehicle_number' in exc.context

    def test_exception_message_format(self):
        """Test formatted exception message includes code and context."""
        exc = InsightsError(
            "Test error",
            error_code="TEST_001",
            context={'vehicle': 5, 'lap': 3}
        )
        msg = str(exc)
        assert "[TEST_001]" in msg
        assert "Test error" in msg
        assert "vehicle=5" in msg
        assert "lap=3" in msg

    def test_exception_to_dict(self):
        """Test conversion to dictionary for API responses."""
        exc = InsightsError(
            "Test error",
            error_code="TEST_001",
            context={'vehicle': 5}
        )
        result = exc.to_dict()

        assert result['error'] is True
        assert result['error_code'] == "TEST_001"
        assert result['message'] == "Test error"
        assert result['context'] == {'vehicle': 5}

    def test_exception_inheritance(self):
        """Test all exceptions inherit from InsightsError."""
        exceptions = [
            DataValidationError("test"),
            AnalysisError("test"),
            ConfigurationError("test"),
            PerformanceError("test")
        ]

        for exc in exceptions:
            assert isinstance(exc, InsightsError)
            assert isinstance(exc, Exception)


class TestDataValidationExceptions:
    """Test data validation exception classes."""

    def test_invalid_dataframe_error(self):
        """Test InvalidDataFrameError."""
        exc = InvalidDataFrameError(
            "Not a DataFrame",
            context={'type_received': 'list'}
        )
        assert exc.error_code == "INVALID_DATAFRAME"
        assert "DataFrame" in exc.message

    def test_missing_columns_error(self):
        """Test MissingColumnsError."""
        exc = MissingColumnsError(
            "Missing speed column",
            context={'missing': ['speed'], 'expected': ['speed', 'brake']}
        )
        assert exc.error_code == "MISSING_COLUMNS"
        assert exc.context['missing'] == ['speed']

    def test_invalid_data_type_error(self):
        """Test InvalidDataTypeError."""
        exc = InvalidDataTypeError(
            "Column should be numeric",
            context={'column': 'speed', 'type': 'object'}
        )
        assert exc.error_code == "INVALID_DATA_TYPE"

    def test_empty_dataset_error(self):
        """Test EmptyDatasetError."""
        exc = EmptyDatasetError("No data", context={'row_count': 0})
        assert exc.error_code == "EMPTY_DATASET"

    def test_invalid_parameter_error(self):
        """Test InvalidParameterError."""
        exc = InvalidParameterError(
            "Invalid vehicle number",
            context={'parameter': 'vehicle_number', 'value': 999}
        )
        assert exc.error_code == "INVALID_PARAMETER"

    def test_sensor_data_error(self):
        """Test SensorDataError."""
        exc = SensorDataError(
            "Sensor out of range",
            context={'sensor': 'speed', 'value': -100}
        )
        assert exc.error_code == "SENSOR_DATA_ERROR"


class TestAnalysisExceptions:
    """Test analysis exception classes."""

    def test_analysis_error(self):
        """Test base AnalysisError."""
        exc = AnalysisError("Analysis failed")
        assert exc.error_code == "ANALYSIS_ERROR"
        assert isinstance(exc, InsightsError)

    def test_insufficient_data_error(self):
        """Test InsufficientDataError."""
        exc = InsufficientDataError(
            "Not enough laps",
            context={'required': 10, 'actual': 3}
        )
        assert exc.error_code == "INSUFFICIENT_DATA"

    def test_corner_detection_error(self):
        """Test CornerDetectionError."""
        exc = CornerDetectionError(
            "No corners found",
            context={'vehicle': 5, 'lap': 3}
        )
        assert exc.error_code == "CORNER_DETECTION_ERROR"

    def test_profile_generation_error(self):
        """Test ProfileGenerationError."""
        exc = ProfileGenerationError(
            "Cannot generate profile",
            context={'reason': 'insufficient_data'}
        )
        assert exc.error_code == "PROFILE_GENERATION_ERROR"

    def test_consistency_analysis_error(self):
        """Test ConsistencyAnalysisError."""
        exc = ConsistencyAnalysisError(
            "Consistency check failed",
            context={'sessions': 1, 'required': 2}
        )
        assert exc.error_code == "CONSISTENCY_ANALYSIS_ERROR"


class TestConfigurationExceptions:
    """Test configuration exception classes."""

    def test_configuration_error(self):
        """Test base ConfigurationError."""
        exc = ConfigurationError("Config issue")
        assert exc.error_code == "CONFIGURATION_ERROR"

    def test_invalid_configuration_error(self):
        """Test InvalidConfigurationError."""
        exc = InvalidConfigurationError(
            "Invalid threshold",
            context={'parameter': 'brake_threshold', 'value': -10}
        )
        assert exc.error_code == "INVALID_CONFIGURATION"

    def test_missing_configuration_error(self):
        """Test MissingConfigurationError."""
        exc = MissingConfigurationError(
            "Config missing",
            context={'required_key': 'api_endpoint'}
        )
        assert exc.error_code == "MISSING_CONFIGURATION"


class TestPerformanceExceptions:
    """Test performance exception classes."""

    def test_performance_error(self):
        """Test base PerformanceError."""
        exc = PerformanceError("Performance issue")
        assert exc.error_code == "PERFORMANCE_ERROR"

    def test_memory_exhausted_error(self):
        """Test MemoryExhaustedError."""
        exc = MemoryExhaustedError(
            "Out of memory",
            context={'requested_mb': 16000, 'available_mb': 1000}
        )
        assert exc.error_code == "MEMORY_EXHAUSTED"

    def test_timeout_error(self):
        """Test TimeoutError."""
        exc = TimeoutError(
            "Operation timeout",
            context={'timeout_seconds': 30, 'elapsed_seconds': 35}
        )
        assert exc.error_code == "TIMEOUT_ERROR"


class TestValidateAndRaise:
    """Test validate_and_raise utility function."""

    def test_raise_on_true_condition(self):
        """Test that exception is raised when condition is True."""
        with pytest.raises(InvalidParameterError) as exc_info:
            validate_and_raise(
                condition=True,
                exception_class=InvalidParameterError,
                message="Invalid value",
                context={'value': 999}
            )

        assert exc_info.value.message == "Invalid value"
        assert exc_info.value.context['value'] == 999

    def test_no_raise_on_false_condition(self):
        """Test that no exception is raised when condition is False."""
        # Should not raise
        validate_and_raise(
            condition=False,
            exception_class=InvalidParameterError,
            message="Invalid value"
        )

    def test_raise_with_default_exception(self):
        """Test raising with default InsightsError."""
        with pytest.raises(InsightsError):
            validate_and_raise(
                condition=True,
                message="Generic error"
            )

    def test_raise_different_exception_types(self):
        """Test raising different exception types."""
        exception_types = [
            (DataValidationError, "DATA_VALIDATION_ERROR"),
            (AnalysisError, "ANALYSIS_ERROR"),
            (ConfigurationError, "CONFIGURATION_ERROR")
        ]

        for exc_class, expected_code in exception_types:
            with pytest.raises(exc_class) as exc_info:
                validate_and_raise(
                    condition=True,
                    exception_class=exc_class,
                    message="Test"
                )
            assert exc_info.value.error_code == expected_code


class TestExceptionHierarchy:
    """Test exception inheritance relationships."""

    def test_data_validation_hierarchy(self):
        """Test DataValidationError inheritance."""
        exceptions = [
            InvalidDataFrameError("test"),
            MissingColumnsError("test"),
            InvalidDataTypeError("test"),
            EmptyDatasetError("test"),
            InvalidParameterError("test"),
            SensorDataError("test")
        ]

        for exc in exceptions:
            assert isinstance(exc, DataValidationError)
            assert isinstance(exc, InsightsError)

    def test_analysis_error_hierarchy(self):
        """Test AnalysisError inheritance."""
        exceptions = [
            InsufficientDataError("test"),
            CornerDetectionError("test"),
            ProfileGenerationError("test"),
            ConsistencyAnalysisError("test")
        ]

        for exc in exceptions:
            assert isinstance(exc, AnalysisError)
            assert isinstance(exc, InsightsError)

    def test_configuration_error_hierarchy(self):
        """Test ConfigurationError inheritance."""
        exceptions = [
            InvalidConfigurationError("test"),
            MissingConfigurationError("test")
        ]

        for exc in exceptions:
            assert isinstance(exc, ConfigurationError)
            assert isinstance(exc, InsightsError)

    def test_performance_error_hierarchy(self):
        """Test PerformanceError inheritance."""
        exceptions = [
            MemoryExhaustedError("test"),
            TimeoutError("test")
        ]

        for exc in exceptions:
            assert isinstance(exc, PerformanceError)
            assert isinstance(exc, InsightsError)

    def test_catch_all_with_base(self):
        """Test that all exceptions can be caught with InsightsError."""
        all_exceptions = [
            DataValidationError("test"),
            InvalidDataFrameError("test"),
            AnalysisError("test"),
            InsufficientDataError("test"),
            ConfigurationError("test"),
            PerformanceError("test")
        ]

        for exc in all_exceptions:
            try:
                raise exc
            except InsightsError as e:
                assert e is exc  # Caught successfully
