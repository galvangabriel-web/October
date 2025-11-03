"""
Pytest Fixtures for Racing Insights Tests

Provides reusable test data and fixtures for all test modules:
- Sample telemetry DataFrames
- Sample lap_times DataFrames
- Configuration objects
- Common test parameters
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ==============================================================================
# TELEMETRY DATA FIXTURES
# ==============================================================================

@pytest.fixture
def sample_telemetry_df():
    """
    Create a sample telemetry DataFrame with realistic racing data.

    Simulates 2 laps of telemetry for vehicle 5 with all sensor types.
    """
    np.random.seed(42)

    # Generate 2000 samples (2 laps worth of data)
    n_samples = 2000
    base_timestamp = int(datetime(2025, 10, 25, 10, 0, 0).timestamp() * 1000)

    data = []

    # Create data for 2 laps
    for lap in [1, 2]:
        lap_samples = n_samples // 2
        timestamps = np.arange(base_timestamp + (lap - 1) * 90000,
                              base_timestamp + lap * 90000,
                              90)  # 90ms intervals

        for i, ts in enumerate(timestamps[:lap_samples]):
            # Simulate racing line: speed varies through corners
            progress = i / lap_samples
            speed = 100 + 40 * np.sin(progress * 2 * np.pi * 3)  # 3 corners per lap

            # Brake pressure (high in corners)
            brake = max(0, 50 * np.cos(progress * 2 * np.pi * 3 + np.pi/4))

            # Throttle (inverse of brake)
            throttle = 100 - brake

            # Lateral g (high in corners)
            lat_g = 1.5 * np.sin(progress * 2 * np.pi * 3)

            # Steering angle
            steering = 50 * np.sin(progress * 2 * np.pi * 3)

            # Add all sensor readings
            sensors = {
                'speed': speed + np.random.normal(0, 2),
                'pbrake_f': brake + np.random.normal(0, 5),
                'pbrake_r': brake * 0.7 + np.random.normal(0, 3),
                'aps': throttle + np.random.normal(0, 3),
                'accx_can': np.random.normal(0, 0.3),
                'accy_can': lat_g + np.random.normal(0, 0.1),
                'Steering_Angle': steering + np.random.normal(0, 2),
                'gear': min(5, max(2, int(speed / 40))),
                'nmot': speed * 50 + np.random.normal(0, 100),
                'VBOX_Long_Minutes': 33.0 + progress * 0.01,
                'VBOX_Lat_Min': -84.0 + progress * 0.01,
                'Laptrigger_lapdist_dls': progress * 4000
            }

            # Add each sensor as a separate row
            for sensor_name, sensor_value in sensors.items():
                data.append({
                    'telemetry_name': sensor_name,
                    'telemetry_value': sensor_value,
                    'vehicle_number': 5,
                    'timestamp': ts,
                    'lap': lap
                })

    return pd.DataFrame(data)


@pytest.fixture
def sample_lap_times_df():
    """
    Create a sample lap_times DataFrame.

    Simulates 5 laps for vehicle 5.
    """
    base_timestamp = int(datetime(2025, 10, 25, 10, 0, 0).timestamp() * 1000)

    data = []
    for lap in range(1, 6):
        lap_start = base_timestamp + (lap - 1) * 90000
        lap_end = lap_start + 90000 + np.random.randint(-2000, 2000)

        data.append({
            'vehicle_number': 5,
            'lap': lap,
            'lap_start_timestamp': lap_start,
            'lap_end_timestamp': lap_end,
            'lap_duration': (lap_end - lap_start) / 1000.0,  # seconds
            'track': 'barber-motorsports-park',
            'race': 'race_unknown'
        })

    return pd.DataFrame(data)


@pytest.fixture
def multi_vehicle_telemetry_df():
    """Create telemetry data for multiple vehicles (3 vehicles, 1 lap each)."""
    np.random.seed(42)

    base_timestamp = int(datetime(2025, 10, 25, 10, 0, 0).timestamp() * 1000)
    data = []

    for vehicle in [3, 5, 7]:
        for i in range(100):
            ts = base_timestamp + i * 100

            data.append({
                'telemetry_name': 'speed',
                'telemetry_value': 100 + np.random.normal(0, 10),
                'vehicle_number': vehicle,
                'timestamp': ts,
                'lap': 1
            })

            data.append({
                'telemetry_name': 'pbrake_f',
                'telemetry_value': max(0, 50 + np.random.normal(0, 20)),
                'vehicle_number': vehicle,
                'timestamp': ts,
                'lap': 1
            })

    return pd.DataFrame(data)


@pytest.fixture
def empty_telemetry_df():
    """Create an empty telemetry DataFrame with correct schema."""
    return pd.DataFrame(columns=[
        'telemetry_name', 'telemetry_value', 'vehicle_number',
        'timestamp', 'lap'
    ])


@pytest.fixture
def invalid_telemetry_df():
    """Create an invalid telemetry DataFrame (missing columns)."""
    return pd.DataFrame({
        'wrong_column': [1, 2, 3],
        'another_wrong': ['a', 'b', 'c']
    })


# ==============================================================================
# CONFIGURATION FIXTURES
# ==============================================================================

@pytest.fixture
def default_config():
    """Get default configuration."""
    from src.insights import InsightsConfig, DEFAULT_CONFIG
    return DEFAULT_CONFIG


@pytest.fixture
def custom_config():
    """Create a custom configuration with non-default values."""
    from src.insights import InsightsConfig

    return InsightsConfig(
        hard_brake_threshold=120.0,
        trail_brake_threshold=0.4,
        min_corner_duration=1.5,
        outlier_threshold=2.5,
        enable_performance_logging=True
    )


# ==============================================================================
# PARAMETER FIXTURES
# ==============================================================================

@pytest.fixture
def valid_vehicle_number():
    """Valid vehicle number."""
    return 5


@pytest.fixture
def invalid_vehicle_number():
    """Invalid vehicle number (out of range)."""
    return 999


@pytest.fixture
def track_name():
    """Sample track name."""
    return "barber-motorsports-park"


# ==============================================================================
# EXCEPTION TESTING FIXTURES
# ==============================================================================

@pytest.fixture
def error_context():
    """Sample error context dictionary."""
    return {
        'vehicle_number': 5,
        'track': 'barber-motorsports-park',
        'lap_count': 10,
        'timestamp': datetime.now().isoformat()
    }


# ==============================================================================
# INTEGRATION TEST FIXTURES
# ==============================================================================

@pytest.fixture
def real_data_loader():
    """
    Provide RacingDataLoader for integration tests.

    Only returns loader if organized_data/ exists.
    """
    from pathlib import Path
    from data_loader import RacingDataLoader

    if not Path('organized_data').exists():
        pytest.skip("organized_data/ directory not found - skipping integration test")

    return RacingDataLoader()


@pytest.fixture
def sample_track_data(real_data_loader):
    """
    Load a small chunk of real telemetry data for integration tests.

    Loads first chunk from first available track.
    """
    tracks = real_data_loader.list_tracks()
    if not tracks:
        pytest.skip("No tracks found in organized_data/")

    track = tracks[0]
    races = real_data_loader.list_races(track)
    race = races[0] if races else 'race_unknown'

    # Load single chunk (fast)
    telemetry = real_data_loader.load_single_chunk(track, race, 'telemetry', chunk_num=1)
    lap_times = real_data_loader.load_data(track, race, 'lap_times')

    return {
        'telemetry': telemetry,
        'lap_times': lap_times,
        'track': track,
        'race': race
    }


# ==============================================================================
# HELPER FIXTURES
# ==============================================================================

@pytest.fixture
def assert_dataframe_equal():
    """Helper to assert DataFrames are equal."""
    def _assert_equal(df1, df2, **kwargs):
        pd.testing.assert_frame_equal(df1, df2, **kwargs)
    return _assert_equal


@pytest.fixture
def create_telemetry_row():
    """Factory fixture to create telemetry rows."""
    def _create(sensor_name, value, vehicle=5, timestamp=None, lap=1):
        if timestamp is None:
            timestamp = int(datetime.now().timestamp() * 1000)

        return {
            'telemetry_name': sensor_name,
            'telemetry_value': value,
            'vehicle_number': vehicle,
            'timestamp': timestamp,
            'lap': lap
        }

    return _create
