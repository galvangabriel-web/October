"""
Input Validation Framework for Racing Insights Module

Production-grade validation system providing:
- DataFrame structure and type validation
- Sensor data validation
- Parameter range validation
- Early error detection with clear messages
- Data quality checks

Design Pattern: Validator Pattern
- Fail-fast validation at entry points
- Structured exceptions with context
- Clear, actionable error messages
- Type checking and schema validation

All validators follow the same pattern:
1. Check input is not None
2. Validate type/structure
3. Check required fields
4. Validate data types
5. Check value ranges
6. Raise specific exceptions with context

Author: Production Engineering Team
Version: 1.0.0
License: GR Cup 2025 Hackathon
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Set, Union, Any

from .exceptions import (
    InvalidDataFrameError,
    MissingColumnsError,
    InvalidDataTypeError,
    EmptyDatasetError,
    InvalidParameterError,
    SensorDataError
)
from .constants import (
    TelemetrySensor,
    TelemetryColumns,
    LapTimesColumns,
    SensorRanges,
    is_valid_vehicle_number
)


# ==============================================================================
# DATAFRAME VALIDATION
# ==============================================================================

def validate_dataframe(
    df: Any,
    name: str = "DataFrame",
    allow_empty: bool = False
) -> None:
    """
    Validate input is a pandas DataFrame.

    Parameters:
        df: Object to validate
        name: Parameter name for error messages
        allow_empty: Whether to allow empty DataFrames

    Raises:
        InvalidDataFrameError: If not a DataFrame or None
        EmptyDatasetError: If DataFrame is empty and not allowed

    Example:
        validate_dataframe(telemetry_df, name="telemetry_df")
    """
    # Check not None
    if df is None:
        raise InvalidDataFrameError(
            f"{name} cannot be None",
            context={"parameter": name, "type_received": "NoneType"}
        )

    # Check is DataFrame
    if not isinstance(df, pd.DataFrame):
        raise InvalidDataFrameError(
            f"{name} must be a pandas DataFrame, got {type(df).__name__}",
            context={"parameter": name, "type_received": type(df).__name__}
        )

    # Check not empty (if required)
    if not allow_empty and len(df) == 0:
        raise EmptyDatasetError(
            f"{name} is empty (0 rows)",
            context={"parameter": name, "row_count": 0}
        )


def validate_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    name: str = "DataFrame"
) -> None:
    """
    Validate DataFrame has required columns.

    Parameters:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Parameter name for error messages

    Raises:
        MissingColumnsError: If required columns are missing

    Example:
        validate_columns(
            telemetry_df,
            TelemetryColumns.required_columns(),
            name="telemetry_df"
        )
    """
    df_columns = set(df.columns)
    required_set = set(required_columns)
    missing = required_set - df_columns

    if missing:
        raise MissingColumnsError(
            f"{name} is missing required columns",
            required_columns=required_columns,
            found_columns=list(df.columns),
            context={"parameter": name}
        )


def validate_column_type(
    df: pd.DataFrame,
    column: str,
    expected_type: Union[type, str],
    name: str = "DataFrame"
) -> None:
    """
    Validate DataFrame column has expected data type.

    Parameters:
        df: DataFrame to validate
        column: Column name to check
        expected_type: Expected numpy/pandas dtype
        name: Parameter name for error messages

    Raises:
        InvalidDataTypeError: If column type doesn't match

    Example:
        validate_column_type(
            telemetry_df,
            'vehicle_number',
            'int64',
            name='telemetry_df'
        )
    """
    if column not in df.columns:
        raise MissingColumnsError(
            f"Column '{column}' not found in {name}",
            required_columns=[column],
            found_columns=list(df.columns),
            context={"parameter": name}
        )

    actual_type = str(df[column].dtype)

    # Handle type checking
    if isinstance(expected_type, str):
        # String type comparison (e.g., 'int64')
        if actual_type != expected_type:
            # Allow compatible types (int32/int64, float32/float64)
            if not (
                ('int' in expected_type and 'int' in actual_type) or
                ('float' in expected_type and 'float' in actual_type)
            ):
                raise InvalidDataTypeError(
                    f"Column '{column}' in {name} has incorrect type",
                    context={
                        "column": column,
                        "expected_type": expected_type,
                        "actual_type": actual_type
                    }
                )


# ==============================================================================
# TELEMETRY VALIDATION
# ==============================================================================

def validate_telemetry_dataframe(
    df: pd.DataFrame,
    name: str = "telemetry_df",
    required_sensors: Optional[List[str]] = None
) -> None:
    """
    Comprehensive validation for telemetry DataFrame.

    Validates:
    - DataFrame structure
    - Required columns present
    - Sensor types valid
    - Data types correct

    Parameters:
        df: Telemetry DataFrame to validate
        name: Parameter name for error messages
        required_sensors: Specific sensors that must be present (optional)

    Raises:
        InvalidDataFrameError: If not a valid DataFrame
        MissingColumnsError: If required columns missing
        SensorDataError: If sensor types invalid

    Example:
        validate_telemetry_dataframe(
            telemetry_df,
            required_sensors=[TelemetrySensor.SPEED, TelemetrySensor.BRAKE_PRESSURE_FRONT]
        )
    """
    # Validate DataFrame basics
    validate_dataframe(df, name=name, allow_empty=False)

    # Validate required columns
    validate_columns(
        df,
        TelemetryColumns.required_columns(),
        name=name
    )

    # Validate sensor types
    sensor_col = TelemetryColumns.TELEMETRY_NAME
    if sensor_col in df.columns:
        unique_sensors = df[sensor_col].unique()
        valid_sensors = TelemetrySensor.all_sensors()

        invalid_sensors = [s for s in unique_sensors if s not in valid_sensors]
        if invalid_sensors:
            raise SensorDataError(
                f"Unknown sensor types found in {name}",
                context={
                    "invalid_sensors": invalid_sensors,
                    "valid_sensors": valid_sensors
                }
            )

        # Check required sensors if specified
        if required_sensors:
            available_sensors = set(unique_sensors)
            required_set = set(required_sensors)
            missing_sensors = required_set - available_sensors

            if missing_sensors:
                raise SensorDataError(
                    f"Required sensors not found in {name}",
                    context={
                        "required_sensors": list(required_sensors),
                        "available_sensors": list(available_sensors),
                        "missing_sensors": list(missing_sensors)
                    }
                )


def validate_lap_times_dataframe(
    df: pd.DataFrame,
    name: str = "lap_times_df"
) -> None:
    """
    Comprehensive validation for lap times DataFrame.

    Validates:
    - DataFrame structure
    - Required columns present
    - Data types correct
    - Lap durations are positive

    Parameters:
        df: Lap times DataFrame to validate
        name: Parameter name for error messages

    Raises:
        InvalidDataFrameError: If not a valid DataFrame
        MissingColumnsError: If required columns missing
        InvalidDataTypeError: If lap durations invalid

    Example:
        validate_lap_times_dataframe(lap_times_df)
    """
    # Validate DataFrame basics
    validate_dataframe(df, name=name, allow_empty=False)

    # Validate required columns
    validate_columns(
        df,
        LapTimesColumns.required_columns(),
        name=name
    )

    # Validate lap durations are positive
    if LapTimesColumns.LAP_DURATION in df.columns:
        invalid_durations = df[df[LapTimesColumns.LAP_DURATION] <= 0]
        if len(invalid_durations) > 0:
            raise InvalidDataTypeError(
                f"Lap durations must be positive in {name}",
                context={
                    "invalid_count": len(invalid_durations),
                    "total_laps": len(df),
                    "example_invalid": float(invalid_durations[LapTimesColumns.LAP_DURATION].iloc[0])
                }
            )


# ==============================================================================
# PARAMETER VALIDATION
# ==============================================================================

def validate_vehicle_number(vehicle_number: Any) -> None:
    """
    Validate vehicle number is valid integer in range.

    Per CLAUDE.md: 20 vehicles (0-19)

    Parameters:
        vehicle_number: Vehicle ID to validate

    Raises:
        InvalidParameterError: If vehicle number invalid

    Example:
        validate_vehicle_number(5)  # OK
        validate_vehicle_number(25)  # Raises
    """
    # Check is integer
    if not isinstance(vehicle_number, (int, np.integer)):
        raise InvalidParameterError(
            f"vehicle_number must be an integer, got {type(vehicle_number).__name__}",
            context={
                "parameter": "vehicle_number",
                "value": vehicle_number,
                "expected_type": "int"
            }
        )

    # Check in valid range
    if not is_valid_vehicle_number(int(vehicle_number)):
        raise InvalidParameterError(
            f"vehicle_number must be 0-{SensorRanges.VEHICLE_COUNT-1}, got {vehicle_number}",
            context={
                "parameter": "vehicle_number",
                "value": vehicle_number,
                "valid_range": f"0-{SensorRanges.VEHICLE_COUNT-1}"
            }
        )


def validate_positive_number(
    value: Any,
    name: str,
    allow_zero: bool = False
) -> None:
    """
    Validate parameter is a positive number.

    Parameters:
        value: Value to validate
        name: Parameter name for error messages
        allow_zero: Whether to allow zero

    Raises:
        InvalidParameterError: If not positive number

    Example:
        validate_positive_number(120.5, "speed_threshold")
        validate_positive_number(0, "min_laps", allow_zero=True)
    """
    # Check is numeric
    if not isinstance(value, (int, float, np.number)):
        raise InvalidParameterError(
            f"{name} must be a number, got {type(value).__name__}",
            context={
                "parameter": name,
                "value": value,
                "expected_type": "numeric"
            }
        )

    # Check positive
    if allow_zero:
        if value < 0:
            raise InvalidParameterError(
                f"{name} must be non-negative, got {value}",
                context={
                    "parameter": name,
                    "value": value,
                    "constraint": ">= 0"
                }
            )
    else:
        if value <= 0:
            raise InvalidParameterError(
                f"{name} must be positive, got {value}",
                context={
                    "parameter": name,
                    "value": value,
                    "constraint": "> 0"
                }
            )


def validate_percentage(value: Any, name: str) -> None:
    """
    Validate parameter is a valid percentage (0-100).

    Parameters:
        value: Value to validate
        name: Parameter name for error messages

    Raises:
        InvalidParameterError: If not in 0-100 range

    Example:
        validate_percentage(95.5, "full_throttle_threshold")
    """
    # Check is numeric
    if not isinstance(value, (int, float, np.number)):
        raise InvalidParameterError(
            f"{name} must be a number, got {type(value).__name__}",
            context={
                "parameter": name,
                "value": value,
                "expected_type": "numeric"
            }
        )

    # Check 0-100 range
    if not 0 <= value <= 100:
        raise InvalidParameterError(
            f"{name} must be 0-100%, got {value}",
            context={
                "parameter": name,
                "value": value,
                "valid_range": "0-100"
            }
        )


def validate_string_not_empty(value: Any, name: str) -> None:
    """
    Validate parameter is a non-empty string.

    Parameters:
        value: Value to validate
        name: Parameter name for error messages

    Raises:
        InvalidParameterError: If not a non-empty string

    Example:
        validate_string_not_empty("session_1", "session_id")
    """
    # Check is string
    if not isinstance(value, str):
        raise InvalidParameterError(
            f"{name} must be a string, got {type(value).__name__}",
            context={
                "parameter": name,
                "value": value,
                "expected_type": "str"
            }
        )

    # Check not empty
    if len(value.strip()) == 0:
        raise InvalidParameterError(
            f"{name} cannot be empty",
            context={
                "parameter": name,
                "value": repr(value)
            }
        )


# ==============================================================================
# DATA QUALITY VALIDATION
# ==============================================================================

def validate_sufficient_data(
    df: pd.DataFrame,
    min_records: int,
    data_type: str = "records",
    context: Optional[dict] = None
) -> None:
    """
    Validate DataFrame has sufficient records for analysis.

    Parameters:
        df: DataFrame to validate
        min_records: Minimum number of records required
        data_type: Type of data for error message
        context: Additional context for error message

    Raises:
        EmptyDatasetError: If insufficient records

    Example:
        validate_sufficient_data(
            telemetry_df,
            min_records=100,
            data_type="telemetry",
            context={"vehicle_number": 5}
        )
    """
    actual_count = len(df)

    if actual_count < min_records:
        ctx = context or {}
        ctx.update({
            "data_type": data_type,
            "required_count": min_records,
            "actual_count": actual_count
        })

        raise EmptyDatasetError(
            f"Insufficient {data_type} for analysis: need {min_records}, found {actual_count}",
            context=ctx
        )


def validate_vehicle_has_data(
    df: pd.DataFrame,
    vehicle_number: int,
    data_type: str = "data"
) -> None:
    """
    Validate specific vehicle has data in DataFrame.

    Parameters:
        df: DataFrame to check
        vehicle_number: Vehicle ID to look for
        data_type: Type of data for error message

    Raises:
        EmptyDatasetError: If no data for vehicle

    Example:
        validate_vehicle_has_data(telemetry_df, vehicle_number=5, data_type="telemetry")
    """
    vehicle_col = TelemetryColumns.VEHICLE_NUMBER

    if vehicle_col not in df.columns:
        raise MissingColumnsError(
            f"Cannot filter by vehicle: '{vehicle_col}' column not found",
            required_columns=[vehicle_col],
            found_columns=list(df.columns)
        )

    vehicle_data = df[df[vehicle_col] == vehicle_number]

    if len(vehicle_data) == 0:
        raise EmptyDatasetError(
            f"No {data_type} found for vehicle {vehicle_number}",
            context={
                "vehicle_number": vehicle_number,
                "data_type": data_type,
                "total_vehicles_in_data": len(df[vehicle_col].unique())
            }
        )


# ==============================================================================
# COMPOSITE VALIDATORS
# ==============================================================================

def validate_profile_inputs(
    telemetry_df: pd.DataFrame,
    lap_times_df: Optional[pd.DataFrame],
    vehicle_number: int
) -> None:
    """
    Comprehensive validation for driver profile creation.

    Validates all inputs required for DriverProfiler.create_profile().

    Parameters:
        telemetry_df: Telemetry DataFrame
        lap_times_df: Lap times DataFrame (optional)
        vehicle_number: Vehicle ID

    Raises:
        Various validation exceptions if inputs invalid

    Example:
        validate_profile_inputs(telemetry_df, lap_times_df, vehicle_number=5)
        validate_profile_inputs(telemetry_df, None, vehicle_number=5)
    """
    # Validate telemetry DataFrame
    validate_telemetry_dataframe(telemetry_df, name="telemetry_df")

    # Validate lap times DataFrame (only if provided)
    if lap_times_df is not None:
        validate_lap_times_dataframe(lap_times_df, name="lap_times_df")

    # Validate vehicle number
    validate_vehicle_number(vehicle_number)

    # Validate vehicle has telemetry data
    validate_vehicle_has_data(telemetry_df, vehicle_number, data_type="telemetry")

    # Validate vehicle has lap times (only if provided)
    if lap_times_df is not None:
        validate_vehicle_has_data(lap_times_df, vehicle_number, data_type="lap times")


def validate_corner_detection_inputs(
    telemetry_df: pd.DataFrame,
    speed_threshold: float,
    min_duration_ms: int
) -> None:
    """
    Comprehensive validation for corner detection.

    Validates all inputs required for CornerAnalyzer.detect_corners().

    Parameters:
        telemetry_df: Telemetry DataFrame
        speed_threshold: Speed threshold for corner detection
        min_duration_ms: Minimum corner duration

    Raises:
        Various validation exceptions if inputs invalid

    Example:
        validate_corner_detection_inputs(
            telemetry_df,
            speed_threshold=120.0,
            min_duration_ms=1000
        )
    """
    # Validate telemetry DataFrame
    validate_telemetry_dataframe(
        telemetry_df,
        name="telemetry_df",
        required_sensors=[TelemetrySensor.SPEED]
    )

    # Validate speed threshold
    validate_positive_number(speed_threshold, "speed_threshold")

    # Validate min duration
    validate_positive_number(min_duration_ms, "min_duration_ms")


def validate_consistency_inputs(
    lap_times_df: pd.DataFrame,
    vehicle_number: int,
    session_id: str
) -> None:
    """
    Comprehensive validation for consistency tracking.

    Validates all inputs required for ConsistencyTracker.track_session().

    Parameters:
        lap_times_df: Lap times DataFrame
        vehicle_number: Vehicle ID
        session_id: Session identifier

    Raises:
        Various validation exceptions if inputs invalid

    Example:
        validate_consistency_inputs(
            lap_times_df,
            vehicle_number=5,
            session_id="session_1"
        )
    """
    # Validate lap times DataFrame
    validate_lap_times_dataframe(lap_times_df, name="lap_times_df")

    # Validate vehicle number
    validate_vehicle_number(vehicle_number)

    # Validate session ID
    validate_string_not_empty(session_id, "session_id")

    # Validate vehicle has data
    validate_vehicle_has_data(lap_times_df, vehicle_number, data_type="lap times")
