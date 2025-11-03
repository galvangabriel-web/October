"""
Constants and Enumerations for Racing Insights Module

Production-grade constants module providing centralized definitions for:
- Telemetry sensor types (per CLAUDE.md specification)
- DataFrame column names
- Valid value ranges
- Analysis thresholds (default values)
- Driving style classifications

Design Pattern: Constants Module Pattern
- Single source of truth for magic values
- Type-safe enumerations for categorical data
- Documented value ranges from domain knowledge
- Prevents typos in string literals
- Enables IDE autocomplete

Reference: CLAUDE.md Data Schema section for sensor specifications

Author: Production Engineering Team
Version: 1.0.0
License: GR Cup 2025 Hackathon
"""

from enum import Enum
from typing import Dict, List, Any


# ==============================================================================
# TELEMETRY SENSOR TYPES
# ==============================================================================

class TelemetrySensor(str, Enum):
    """
    Enumeration of all telemetry sensor types in the GR Cup dataset.

    Based on CLAUDE.md specification: 12 sensor types total.
    Inherits from str for easy comparison with DataFrame string values.

    Value Ranges (from CLAUDE.md):
    - SPEED: 66-190 km/h
    - BRAKE_PRESSURE_FRONT/REAR: 0-153 bar
    - THROTTLE: 0-100%
    - ACCEL_LONGITUDINAL: -3 to +2g
    - ACCEL_LATERAL: -3 to +2g
    - STEERING_ANGLE: -109° to 130°
    - GEAR: 1-5
    - RPM: Engine RPM (range varies by vehicle)

    Example:
        # Instead of: df[df['telemetry_name'] == 'speed']
        # Use: df[df['telemetry_name'] == TelemetrySensor.SPEED]
    """

    SPEED = 'speed'
    BRAKE_PRESSURE_FRONT = 'pbrake_f'
    BRAKE_PRESSURE_REAR = 'pbrake_r'
    THROTTLE = 'aps'  # Accelerator Pedal Sensor
    ACCEL_LONGITUDINAL = 'accx_can'  # Forward/backward acceleration
    ACCEL_LATERAL = 'accy_can'  # Side-to-side acceleration (cornering)
    STEERING_ANGLE = 'Steering_Angle'
    GEAR = 'gear'
    RPM = 'nmot'  # Engine RPM
    GPS_LONGITUDE = 'VBOX_Long_Minutes'
    GPS_LATITUDE = 'VBOX_Lat_Min'
    LAP_TRIGGER = 'Laptrigger_lapdist_dls'  # Lap distance trigger
    ATH = 'ath'  # Additional sensor (ambient temperature or angle)

    @classmethod
    def all_sensors(cls) -> List[str]:
        """Get list of all sensor names as strings."""
        return [sensor.value for sensor in cls]

    @classmethod
    def is_valid_sensor(cls, sensor_name: str) -> bool:
        """Check if sensor name is valid."""
        return sensor_name in cls.all_sensors()


# ==============================================================================
# DATAFRAME COLUMN NAMES
# ==============================================================================

class TelemetryColumns:
    """Column names for telemetry DataFrames."""

    TELEMETRY_NAME = 'telemetry_name'
    TELEMETRY_VALUE = 'telemetry_value'
    VEHICLE_NUMBER = 'vehicle_number'
    TIMESTAMP = 'timestamp'
    LAP = 'lap'

    # Derived columns (may be added during processing)
    DATETIME = 'datetime'  # pd.to_datetime(timestamp, unit='ms')
    CORNER_CHANGE = 'corner_change'
    Z_SCORE = 'z_score'
    IS_OUTLIER = 'is_outlier'

    @classmethod
    def required_columns(cls) -> List[str]:
        """Get list of required columns for telemetry data."""
        return [
            cls.TELEMETRY_NAME,
            cls.TELEMETRY_VALUE,
            cls.VEHICLE_NUMBER,
            cls.TIMESTAMP,
            cls.LAP
        ]


class LapTimesColumns:
    """Column names for lap times DataFrames."""

    VEHICLE_NUMBER = 'vehicle_number'
    LAP = 'lap'
    LAP_DURATION = 'lap_duration'
    LAP_START_TIME = 'lap_start_time'
    LAP_END_TIME = 'lap_end_time'

    @classmethod
    def required_columns(cls) -> List[str]:
        """
        Get list of required columns for lap times data.

        Note: LAP_DURATION is optional - if not present, it can be calculated
        from timestamp differences. Only VEHICLE_NUMBER and LAP are strictly required.
        """
        return [
            cls.VEHICLE_NUMBER,
            cls.LAP  # Changed from LAP_DURATION to LAP for compatibility with existing data
        ]


# ==============================================================================
# VALID VALUE RANGES (from CLAUDE.md)
# ==============================================================================

class SensorRanges:
    """
    Valid value ranges for telemetry sensors.

    Based on actual data ranges from CLAUDE.md analysis.
    Used for data validation and outlier detection context.

    Note: Per CLAUDE.md, DO NOT remove "outliers" - extreme values are
    features (hard braking, high g-forces) not errors!
    """

    SPEED = {'min': 66, 'max': 190, 'unit': 'km/h'}
    BRAKE_PRESSURE = {'min': 0, 'max': 153, 'unit': 'bar'}
    THROTTLE = {'min': 0, 'max': 100, 'unit': '%'}
    ACCELERATION = {'min': -3.0, 'max': 2.0, 'unit': 'g'}
    STEERING_ANGLE = {'min': -109, 'max': 130, 'unit': 'degrees'}
    GEAR = {'min': 1, 'max': 5, 'unit': 'gear'}
    VEHICLE_COUNT = 20  # 0-19 per CLAUDE.md


# ==============================================================================
# DRIVING STYLE CLASSIFICATIONS
# ==============================================================================

class DrivingStyle(str, Enum):
    """
    Driver style classifications based on telemetry patterns.

    Categories derived from braking aggression, throttle application,
    and smoothness metrics.
    """

    AGGRESSIVE = 'Aggressive'
    SMOOTH = 'Smooth'
    CONSERVATIVE = 'Conservative'
    BALANCED = 'Balanced'


# ==============================================================================
# PERFORMANCE TRAJECTORY CLASSIFICATIONS
# ==============================================================================

class PerformanceTrend(str, Enum):
    """Session-over-session performance trajectory classifications."""

    IMPROVING = 'Improving'
    STABLE = 'Stable'
    DECLINING = 'Declining'
    INCONSISTENT = 'Inconsistent'


# ==============================================================================
# OUTLIER CLASSIFICATIONS
# ==============================================================================

class OutlierType(str, Enum):
    """Lap time outlier classifications."""

    EXCEPTIONALLY_FAST = 'Exceptionally Fast'
    EXCEPTIONALLY_SLOW = 'Exceptionally Slow'


# ==============================================================================
# ANALYSIS METRICS
# ==============================================================================

class MetricNames:
    """Standard metric names for consistent reporting."""

    # Driver Profile Metrics
    CONSISTENCY_SCORE = 'consistency_score'
    AGGRESSION_INDEX = 'aggression_index'
    SMOOTHNESS_INDEX = 'smoothness_index'
    DRIVING_STYLE = 'driving_style'

    # Braking Metrics
    MAX_BRAKE_PRESSURE = 'max_brake_pressure'
    AVG_BRAKE_PRESSURE = 'avg_brake_pressure'
    BRAKE_CONSISTENCY = 'brake_consistency'
    BRAKE_VARIANCE = 'brake_variance'
    HARD_BRAKING_EVENTS = 'hard_braking_events'

    # Throttle Metrics
    FULL_THROTTLE_PCT = 'full_throttle_percentage'
    AVG_THROTTLE = 'avg_throttle'
    THROTTLE_SMOOTHNESS = 'throttle_smoothness'
    PARTIAL_THROTTLE_PCT = 'partial_throttle_percentage'

    # Cornering Metrics
    MAX_LATERAL_G = 'max_lateral_g'
    AVG_LATERAL_G = 'avg_lateral_g'
    AVG_CORNERING_SPEED = 'avg_cornering_speed'
    CORNERING_SPEED_INDEX = 'cornering_speed_index'

    # Speed Metrics
    MAX_SPEED = 'max_speed'
    AVG_SPEED = 'avg_speed'
    SPEED_VARIANCE = 'speed_variance'

    # Lap Metrics
    FASTEST_LAP = 'fastest_lap'
    AVERAGE_LAP = 'average_lap'
    TOTAL_LAPS = 'total_laps'
    LAP_TIME_STD = 'lap_time_std'

    # Corner Analysis Metrics
    ENTRY_SPEED = 'entry_speed'
    APEX_SPEED = 'apex_speed'
    EXIT_SPEED = 'exit_speed'
    BRAKING_POINT = 'braking_point'
    THROTTLE_POINT = 'throttle_point'


# ==============================================================================
# TRACK NAMES
# ==============================================================================

class TrackName(str, Enum):
    """
    GR Cup track names.

    Based on CLAUDE.md specification: 6 tracks total.
    """

    BARBER = 'barber-motorsports-park'
    COTA = 'circuit-of-the-americas'
    ROAD_AMERICA = 'road-america'
    SEBRING = 'sebring'
    SONOMA = 'sonoma'
    VIR = 'virginia-international-raceway'

    @classmethod
    def all_tracks(cls) -> List[str]:
        """Get list of all track names."""
        return [track.value for track in cls]


# ==============================================================================
# MINIMUM DATA REQUIREMENTS
# ==============================================================================

class MinimumDataRequirements:
    """
    Minimum data requirements for various analyses.

    Defines how much data is needed for meaningful analysis.
    """

    # Minimum laps required
    MIN_LAPS_FOR_CONSISTENCY = 3
    MIN_LAPS_FOR_TREND_ANALYSIS = 5
    MIN_LAPS_FOR_PROFILE = 1

    # Minimum telemetry records
    MIN_TELEMETRY_RECORDS = 100

    # Minimum sessions for trend detection
    MIN_SESSIONS_FOR_TREND = 2


# ==============================================================================
# MESSAGE TEMPLATES
# ==============================================================================

class ErrorMessages:
    """Standard error message templates for consistent error reporting."""

    # Data validation messages
    INVALID_DATAFRAME = "Expected pandas DataFrame, got {type_name}"
    MISSING_COLUMNS = "Missing required columns: {missing_cols}"
    EMPTY_DATASET = "No {data_type} data found for vehicle {vehicle_number}"
    INVALID_VEHICLE_NUMBER = "Vehicle number {vehicle_number} is invalid (must be 0-{max_vehicle})"
    INVALID_SENSOR = "Unknown sensor type '{sensor_name}' (valid: {valid_sensors})"

    # Analysis messages
    INSUFFICIENT_LAPS = "Need at least {min_laps} laps for {analysis_type}, found {actual_laps}"
    NO_CORNERS_DETECTED = "No corners detected with threshold {threshold}"
    PROFILE_GENERATION_FAILED = "Failed to generate profile: {reason}"

    # Configuration messages
    INVALID_THRESHOLD = "{parameter} must be {constraint}, got {value}"
    MISSING_CONFIG = "Required configuration parameter '{parameter}' not set"


# ==============================================================================
# LOGGING MESSAGES
# ==============================================================================

class LogMessages:
    """Standard log message templates for consistent logging."""

    # Info-level messages
    ANALYSIS_STARTED = "Starting {analysis_type} for vehicle {vehicle_number}"
    ANALYSIS_COMPLETED = "{analysis_type} completed in {duration:.2f}s"
    DATA_LOADED = "Loaded {record_count} {data_type} records"

    # Debug-level messages
    FILTERING_DATA = "Filtering {data_type} for vehicle {vehicle_number}"
    CALCULATING_METRIC = "Calculating {metric_name}"
    CORNER_DETECTED = "Detected corner {corner_id} at timestamp {timestamp}"

    # Warning-level messages
    LOW_DATA_QUALITY = "Low data quality for {analysis_type}: {reason}"
    MISSING_SENSOR_DATA = "No {sensor_name} data found for vehicle {vehicle_number}"
    OUTLIER_DETECTED = "Outlier lap detected: {lap_time}s (z-score: {z_score:.2f})"


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_sensor_range(sensor: TelemetrySensor) -> Dict[str, Any]:
    """
    Get valid value range for a sensor.

    Parameters:
        sensor: Telemetry sensor type

    Returns:
        Dictionary with min, max, and unit

    Example:
        range_info = get_sensor_range(TelemetrySensor.SPEED)
        # {'min': 66, 'max': 190, 'unit': 'km/h'}
    """
    sensor_range_map = {
        TelemetrySensor.SPEED: SensorRanges.SPEED,
        TelemetrySensor.BRAKE_PRESSURE_FRONT: SensorRanges.BRAKE_PRESSURE,
        TelemetrySensor.BRAKE_PRESSURE_REAR: SensorRanges.BRAKE_PRESSURE,
        TelemetrySensor.THROTTLE: SensorRanges.THROTTLE,
        TelemetrySensor.ACCEL_LONGITUDINAL: SensorRanges.ACCELERATION,
        TelemetrySensor.ACCEL_LATERAL: SensorRanges.ACCELERATION,
        TelemetrySensor.STEERING_ANGLE: SensorRanges.STEERING_ANGLE,
        TelemetrySensor.GEAR: SensorRanges.GEAR,
    }
    return sensor_range_map.get(sensor, {'min': None, 'max': None, 'unit': 'unknown'})


def is_valid_vehicle_number(vehicle_number: int) -> bool:
    """
    Validate vehicle number is in valid range.

    Per CLAUDE.md: 20 vehicles (0-19)

    Parameters:
        vehicle_number: Vehicle ID to validate

    Returns:
        True if valid, False otherwise

    Example:
        is_valid_vehicle_number(5)   # True
        is_valid_vehicle_number(25)  # False
    """
    return 0 <= vehicle_number < SensorRanges.VEHICLE_COUNT
