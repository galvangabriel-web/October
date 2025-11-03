"""
Unit Tests for Validation Framework

Tests all validation functions in src.insights.validation:
- DataFrame validation
- Column validation
- Type validation
- Parameter validation
- Data quality checks
- Composite validators
"""

import pytest
import pandas as pd
import numpy as np
from src.insights.validation import (
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
from src.insights.exceptions import (
    InvalidDataFrameError,
    MissingColumnsError,
    InvalidDataTypeError,
    EmptyDatasetError,
    InvalidParameterError,
    InsufficientDataError
)


class TestDataFrameValidation:
    """Test validate_dataframe function."""

    def test_valid_dataframe(self, sample_telemetry_df):
        """Test validation passes for valid DataFrame."""
        # Should not raise
        validate_dataframe(sample_telemetry_df, name="test_df")

    def test_none_dataframe(self):
        """Test validation fails for None."""
        with pytest.raises(InvalidDataFrameError) as exc_info:
            validate_dataframe(None, name="test_df")

        assert "test_df" in exc_info.value.message
        assert "None" in exc_info.value.message

    def test_non_dataframe_object(self):
        """Test validation fails for non-DataFrame objects."""
        invalid_inputs = [
            [1, 2, 3],
            {'a': 1},
            "not a dataframe",
            123
        ]

        for invalid in invalid_inputs:
            with pytest.raises(InvalidDataFrameError):
                validate_dataframe(invalid, name="test_df")

    def test_empty_dataframe_not_allowed(self, empty_telemetry_df):
        """Test validation fails for empty DataFrame when not allowed."""
        with pytest.raises(EmptyDatasetError) as exc_info:
            validate_dataframe(empty_telemetry_df, name="test_df", allow_empty=False)

        assert "empty" in exc_info.value.message.lower()
        assert exc_info.value.context['row_count'] == 0

    def test_empty_dataframe_allowed(self, empty_telemetry_df):
        """Test validation passes for empty DataFrame when allowed."""
        # Should not raise
        validate_dataframe(empty_telemetry_df, name="test_df", allow_empty=True)


class TestColumnValidation:
    """Test validate_columns function."""

    def test_all_columns_present(self, sample_telemetry_df):
        """Test validation passes when all columns present."""
        required = ['telemetry_name', 'telemetry_value', 'vehicle_number']
        # Should not raise
        validate_columns(sample_telemetry_df, required, name="test_df")

    def test_missing_columns(self, sample_telemetry_df):
        """Test validation fails when columns missing."""
        required = ['telemetry_name', 'missing_column', 'another_missing']

        with pytest.raises(MissingColumnsError) as exc_info:
            validate_columns(sample_telemetry_df, required, name="test_df")

        assert 'missing_column' in str(exc_info.value.context['missing'])
        assert 'another_missing' in str(exc_info.value.context['missing'])

    def test_extra_columns_ok(self, sample_telemetry_df):
        """Test validation passes with extra columns."""
        required = ['telemetry_name', 'vehicle_number']
        # Should not raise (extra columns are fine)
        validate_columns(sample_telemetry_df, required)


class TestColumnTypeValidation:
    """Test validate_column_type function."""

    def test_numeric_column_valid(self, sample_telemetry_df):
        """Test numeric column validation."""
        # Should not raise
        validate_column_type(
            sample_telemetry_df,
            'telemetry_value',
            expected_type='numeric',
            name="telemetry"
        )

    def test_numeric_column_invalid(self):
        """Test numeric validation fails for non-numeric."""
        df = pd.DataFrame({'col': ['a', 'b', 'c']})

        with pytest.raises(InvalidDataTypeError):
            validate_column_type(df, 'col', expected_type='numeric')

    def test_string_column_valid(self, sample_telemetry_df):
        """Test string column validation."""
        # Should not raise
        validate_column_type(
            sample_telemetry_df,
            'telemetry_name',
            expected_type='string'
        )

    def test_integer_column_valid(self, sample_telemetry_df):
        """Test integer column validation."""
        # Should not raise
        validate_column_type(
            sample_telemetry_df,
            'vehicle_number',
            expected_type='integer'
        )

    def test_column_not_exists(self, sample_telemetry_df):
        """Test validation fails for non-existent column."""
        with pytest.raises(MissingColumnsError):
            validate_column_type(
                sample_telemetry_df,
                'nonexistent_column',
                expected_type='numeric'
            )


class TestTelemetryValidation:
    """Test validate_telemetry_dataframe function."""

    def test_valid_telemetry(self, sample_telemetry_df):
        """Test validation passes for valid telemetry."""
        # Should not raise
        validate_telemetry_dataframe(sample_telemetry_df)

    def test_missing_required_columns(self, invalid_telemetry_df):
        """Test validation fails with missing columns."""
        with pytest.raises(MissingColumnsError):
            validate_telemetry_dataframe(invalid_telemetry_df)

    def test_empty_telemetry(self, empty_telemetry_df):
        """Test validation fails for empty telemetry."""
        with pytest.raises(EmptyDatasetError):
            validate_telemetry_dataframe(empty_telemetry_df)


class TestLapTimesValidation:
    """Test validate_lap_times_dataframe function."""

    def test_valid_lap_times(self, sample_lap_times_df):
        """Test validation passes for valid lap times."""
        # Should not raise
        validate_lap_times_dataframe(sample_lap_times_df)

    def test_missing_columns(self):
        """Test validation fails with missing columns."""
        df = pd.DataFrame({'wrong': [1, 2, 3]})

        with pytest.raises(MissingColumnsError):
            validate_lap_times_dataframe(df)

    def test_empty_lap_times(self):
        """Test validation fails for empty lap times."""
        df = pd.DataFrame(columns=['vehicle_number', 'lap', 'lap_duration'])

        with pytest.raises(EmptyDatasetError):
            validate_lap_times_dataframe(df)


class TestParameterValidation:
    """Test parameter validation functions."""

    def test_valid_vehicle_number(self, valid_vehicle_number):
        """Test valid vehicle number."""
        # Should not raise
        validate_vehicle_number(valid_vehicle_number)

    def test_invalid_vehicle_number(self):
        """Test invalid vehicle numbers."""
        invalid_numbers = [-1, 21, 100, None, 'abc']

        for invalid in invalid_numbers:
            with pytest.raises(InvalidParameterError):
                validate_vehicle_number(invalid)

    def test_positive_number_valid(self):
        """Test valid positive numbers."""
        valid_numbers = [1, 10.5, 0.001, 1e6]

        for num in valid_numbers:
            # Should not raise
            validate_positive_number(num, name="test_param")

    def test_positive_number_invalid(self):
        """Test invalid positive numbers."""
        invalid_numbers = [0, -1, -10.5, None]

        for invalid in invalid_numbers:
            with pytest.raises(InvalidParameterError):
                validate_positive_number(invalid, name="test_param")

    def test_percentage_valid(self):
        """Test valid percentages."""
        valid_percentages = [0, 50, 100, 0.5, 99.9]

        for pct in valid_percentages:
            # Should not raise
            validate_percentage(pct, name="test_pct")

    def test_percentage_invalid(self):
        """Test invalid percentages."""
        invalid_percentages = [-1, 101, -0.1, 150, None]

        for invalid in invalid_percentages:
            with pytest.raises(InvalidParameterError):
                validate_percentage(invalid, name="test_pct")

    def test_string_not_empty_valid(self):
        """Test valid non-empty strings."""
        valid_strings = ["hello", "a", "   text   "]

        for s in valid_strings:
            # Should not raise
            validate_string_not_empty(s, name="test_str")

    def test_string_not_empty_invalid(self):
        """Test invalid empty strings."""
        invalid_strings = ["", "   ", None]

        for invalid in invalid_strings:
            with pytest.raises(InvalidParameterError):
                validate_string_not_empty(invalid, name="test_str")


class TestDataQualityValidation:
    """Test data quality validation functions."""

    def test_sufficient_data_valid(self, sample_telemetry_df):
        """Test sufficient data validation passes."""
        # Should not raise (we have >1000 rows)
        validate_sufficient_data(
            sample_telemetry_df,
            min_rows=100,
            name="telemetry"
        )

    def test_sufficient_data_invalid(self):
        """Test insufficient data validation fails."""
        df = pd.DataFrame({'col': [1, 2, 3]})  # Only 3 rows

        with pytest.raises(InsufficientDataError):
            validate_sufficient_data(df, min_rows=100, name="telemetry")

    def test_vehicle_has_data_valid(self, sample_telemetry_df):
        """Test vehicle has data validation passes."""
        # Should not raise (vehicle 5 exists)
        validate_vehicle_has_data(
            sample_telemetry_df,
            vehicle_number=5,
            vehicle_column='vehicle_number'
        )

    def test_vehicle_has_data_invalid(self, sample_telemetry_df):
        """Test vehicle has no data validation fails."""
        # Vehicle 999 doesn't exist
        with pytest.raises(InsufficientDataError):
            validate_vehicle_has_data(
                sample_telemetry_df,
                vehicle_number=999,
                vehicle_column='vehicle_number'
            )


class TestCompositeValidators:
    """Test composite validation functions."""

    def test_validate_profile_inputs_valid(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test profile inputs validation passes."""
        # Should not raise
        validate_profile_inputs(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

    def test_validate_profile_inputs_invalid_telemetry(
        self,
        invalid_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test profile validation fails with invalid telemetry."""
        with pytest.raises(MissingColumnsError):
            validate_profile_inputs(
                invalid_telemetry_df,
                sample_lap_times_df,
                valid_vehicle_number
            )

    def test_validate_profile_inputs_no_vehicle_data(
        self,
        sample_telemetry_df,
        sample_lap_times_df
    ):
        """Test profile validation fails with non-existent vehicle."""
        with pytest.raises(InsufficientDataError):
            validate_profile_inputs(
                sample_telemetry_df,
                sample_lap_times_df,
                vehicle_number=999  # Doesn't exist
            )

    def test_validate_corner_detection_inputs_valid(
        self,
        sample_telemetry_df,
        valid_vehicle_number
    ):
        """Test corner detection inputs validation passes."""
        # Should not raise
        validate_corner_detection_inputs(
            sample_telemetry_df,
            valid_vehicle_number
        )

    def test_validate_corner_detection_inputs_invalid(self):
        """Test corner detection validation fails with invalid inputs."""
        df = pd.DataFrame({'wrong': [1, 2, 3]})

        with pytest.raises(MissingColumnsError):
            validate_corner_detection_inputs(df, vehicle_number=5)

    def test_validate_consistency_inputs_valid(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test consistency inputs validation passes."""
        # Should not raise
        validate_consistency_inputs(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

    def test_validate_consistency_inputs_insufficient_laps(
        self,
        sample_telemetry_df,
        valid_vehicle_number
    ):
        """Test consistency validation with too few laps."""
        # Create lap_times with only 1 lap
        lap_times = pd.DataFrame({
            'vehicle_number': [5],
            'lap': [1],
            'lap_duration': [90.0]
        })

        with pytest.raises(InsufficientDataError):
            validate_consistency_inputs(
                sample_telemetry_df,
                lap_times,
                valid_vehicle_number,
                min_laps=5  # Require at least 5 laps
            )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_dataframe_with_nan_values(self):
        """Test validation with NaN values (should pass for existence check)."""
        df = pd.DataFrame({
            'telemetry_name': ['speed', 'brake', None],
            'telemetry_value': [100, 50, np.nan],
            'vehicle_number': [5, 5, 5],
            'timestamp': [1000, 2000, 3000],
            'lap': [1, 1, 1]
        })

        # Structure validation should pass
        validate_dataframe(df)
        validate_columns(df, ['telemetry_name', 'telemetry_value'])

    def test_very_large_dataframe(self):
        """Test validation with large DataFrame (performance check)."""
        # Create large DataFrame
        n_rows = 1_000_000
        df = pd.DataFrame({
            'telemetry_name': ['speed'] * n_rows,
            'telemetry_value': np.random.rand(n_rows) * 100,
            'vehicle_number': [5] * n_rows,
            'timestamp': range(n_rows),
            'lap': [1] * n_rows
        })

        # Should complete quickly
        validate_dataframe(df)
        validate_columns(df, ['telemetry_name', 'telemetry_value'])

    def test_mixed_type_column(self):
        """Test column with mixed types."""
        df = pd.DataFrame({
            'mixed': [1, 'two', 3.0, None]
        })

        # Should fail numeric type check
        with pytest.raises(InvalidDataTypeError):
            validate_column_type(df, 'mixed', expected_type='numeric')
