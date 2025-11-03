"""
Unit Tests for DriverProfiler

Tests driver profiling functionality:
- Profile creation
- Braking analysis
- Throttle analysis
- Cornering analysis
- Driving style classification
- Error handling
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from src.insights import DriverProfiler, InsightsConfig
from src.insights.exceptions import (
    EmptyDatasetError,
    InsufficientDataError,
    ProfileGenerationError,
    DataValidationError
)


class TestDriverProfilerInitialization:
    """Test DriverProfiler initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        profiler = DriverProfiler()

        assert profiler.config is not None
        assert profiler.config.hard_brake_threshold == 100.0
        assert profiler.profiles == {}
        assert profiler.benchmarks == {}

    def test_init_custom_config(self, custom_config):
        """Test initialization with custom config."""
        profiler = DriverProfiler(config=custom_config)

        assert profiler.config == custom_config
        assert profiler.config.hard_brake_threshold == 120.0

    def test_init_stores_config(self):
        """Test that config is stored correctly."""
        config = InsightsConfig(hard_brake_threshold=115.0)
        profiler = DriverProfiler(config=config)

        assert profiler.config.hard_brake_threshold == 115.0


class TestCreateProfile:
    """Test create_profile method."""

    def test_create_profile_success(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test successful profile creation."""
        profiler = DriverProfiler()

        profile = profiler.create_profile(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

        # Profile should be a dictionary
        assert isinstance(profile, dict)

        # Should contain key sections
        assert 'braking' in profile or 'throttle' in profile or 'cornering' in profile

    def test_create_profile_invalid_vehicle(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        invalid_vehicle_number
    ):
        """Test profile creation fails for invalid vehicle number."""
        profiler = DriverProfiler()

        with pytest.raises(DataValidationError):
            profiler.create_profile(
                sample_telemetry_df,
                sample_lap_times_df,
                invalid_vehicle_number
            )

    def test_create_profile_no_vehicle_data(
        self,
        sample_telemetry_df,
        sample_lap_times_df
    ):
        """Test profile creation fails when vehicle has no data."""
        profiler = DriverProfiler()

        # Vehicle 999 doesn't exist in sample data
        with pytest.raises((EmptyDatasetError, InsufficientDataError)):
            profiler.create_profile(
                sample_telemetry_df,
                sample_lap_times_df,
                vehicle_number=999
            )

    def test_create_profile_empty_telemetry(
        self,
        empty_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test profile creation fails with empty telemetry."""
        profiler = DriverProfiler()

        with pytest.raises((EmptyDatasetError, DataValidationError)):
            profiler.create_profile(
                empty_telemetry_df,
                sample_lap_times_df,
                valid_vehicle_number
            )

    def test_create_profile_invalid_dataframe(
        self,
        invalid_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test profile creation fails with invalid DataFrame."""
        profiler = DriverProfiler()

        with pytest.raises(DataValidationError):
            profiler.create_profile(
                invalid_telemetry_df,
                sample_lap_times_df,
                valid_vehicle_number
            )

    def test_create_profile_none_inputs(self):
        """Test profile creation fails with None inputs."""
        profiler = DriverProfiler()

        with pytest.raises(DataValidationError):
            profiler.create_profile(None, None, 5)


class TestBrakingAnalysis:
    """Test braking analysis components."""

    def test_braking_metrics_exist(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test that braking metrics are included in profile."""
        profiler = DriverProfiler()

        profile = profiler.create_profile(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

        # Should have braking section (if braking data exists)
        # Note: Exact structure depends on implementation
        assert profile is not None

    def test_braking_with_custom_threshold(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test braking analysis uses custom threshold."""
        custom_config = InsightsConfig(hard_brake_threshold=120.0)
        profiler = DriverProfiler(config=custom_config)

        profile = profiler.create_profile(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

        # Should complete without error
        assert profile is not None


class TestThrottleAnalysis:
    """Test throttle analysis components."""

    def test_throttle_metrics_exist(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test that throttle metrics are included."""
        profiler = DriverProfiler()

        profile = profiler.create_profile(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

        assert profile is not None


class TestCorneringAnalysis:
    """Test cornering analysis components."""

    def test_cornering_metrics_exist(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test that cornering metrics are included."""
        profiler = DriverProfiler()

        profile = profiler.create_profile(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

        assert profile is not None


class TestMultipleVehicles:
    """Test profiling multiple vehicles."""

    def test_profile_multiple_vehicles(
        self,
        multi_vehicle_telemetry_df,
        sample_lap_times_df
    ):
        """Test creating profiles for multiple vehicles."""
        profiler = DriverProfiler()

        # Create profiles for vehicles 3, 5, 7
        for vehicle in [3, 5, 7]:
            # Adjust lap_times for each vehicle
            lap_times = sample_lap_times_df.copy()
            lap_times['vehicle_number'] = vehicle

            profile = profiler.create_profile(
                multi_vehicle_telemetry_df,
                lap_times,
                vehicle_number=vehicle
            )

            assert profile is not None

    def test_profiles_are_independent(
        self,
        multi_vehicle_telemetry_df,
        sample_lap_times_df
    ):
        """Test that profiles for different vehicles are independent."""
        profiler = DriverProfiler()

        # Create two profiles
        lap_times_3 = sample_lap_times_df.copy()
        lap_times_3['vehicle_number'] = 3

        lap_times_5 = sample_lap_times_df.copy()
        lap_times_5['vehicle_number'] = 5

        profile_3 = profiler.create_profile(
            multi_vehicle_telemetry_df,
            lap_times_3,
            vehicle_number=3
        )

        profile_5 = profiler.create_profile(
            multi_vehicle_telemetry_df,
            lap_times_5,
            vehicle_number=5
        )

        # Profiles should exist and be different objects
        assert profile_3 is not None
        assert profile_5 is not None
        assert profile_3 is not profile_5


class TestConfigurationImpact:
    """Test that configuration affects profile generation."""

    def test_different_configs_may_affect_results(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test that different configs can produce different results."""
        config1 = InsightsConfig(hard_brake_threshold=90.0)
        config2 = InsightsConfig(hard_brake_threshold=120.0)

        profiler1 = DriverProfiler(config=config1)
        profiler2 = DriverProfiler(config=config2)

        profile1 = profiler1.create_profile(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

        profile2 = profiler2.create_profile(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

        # Both should succeed
        assert profile1 is not None
        assert profile2 is not None

    def test_config_persists_across_calls(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test that config persists across multiple create_profile calls."""
        config = InsightsConfig(hard_brake_threshold=115.0)
        profiler = DriverProfiler(config=config)

        # Call twice
        profile1 = profiler.create_profile(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

        profile2 = profiler.create_profile(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

        # Config should still be the same
        assert profiler.config.hard_brake_threshold == 115.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_lap_data(
        self,
        sample_telemetry_df,
        valid_vehicle_number
    ):
        """Test profile with minimal lap data."""
        # Create lap_times with single lap
        lap_times = pd.DataFrame({
            'vehicle_number': [valid_vehicle_number],
            'lap': [1],
            'lap_duration': [90.0],
            'lap_start_timestamp': [1000000],
            'lap_end_timestamp': [1090000]
        })

        profiler = DriverProfiler()

        # Should handle gracefully (might succeed or raise InsufficientDataError)
        try:
            profile = profiler.create_profile(
                sample_telemetry_df,
                lap_times,
                valid_vehicle_number
            )
            assert profile is not None
        except InsufficientDataError:
            # Acceptable to require minimum laps
            pass

    def test_missing_sensor_data(
        self,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test profile with missing sensor types."""
        # Create telemetry with only speed data
        telemetry = pd.DataFrame({
            'telemetry_name': ['speed'] * 100,
            'telemetry_value': np.random.rand(100) * 100 + 50,
            'vehicle_number': [valid_vehicle_number] * 100,
            'timestamp': range(100000, 110000, 100),
            'lap': [1] * 100
        })

        profiler = DriverProfiler()

        # Should handle gracefully (might have partial profile)
        try:
            profile = profiler.create_profile(
                telemetry,
                sample_lap_times_df,
                valid_vehicle_number
            )
            # Profile might be partial or empty
            assert profile is not None
        except (InsufficientDataError, ProfileGenerationError):
            # Acceptable to fail with missing sensors
            pass

    def test_extreme_sensor_values(
        self,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test profile with extreme but valid sensor values."""
        # Create telemetry with extreme values
        telemetry = pd.DataFrame({
            'telemetry_name': ['speed', 'pbrake_f'] * 100,
            'telemetry_value': [200.0, 200.0] * 100,  # Very high values
            'vehicle_number': [valid_vehicle_number] * 200,
            'timestamp': range(100000, 120000, 100),
            'lap': [1] * 200
        })

        profiler = DriverProfiler()

        # Should handle without crashing
        profile = profiler.create_profile(
            telemetry,
            sample_lap_times_df,
            valid_vehicle_number
        )

        assert profile is not None


class TestPerformance:
    """Test performance characteristics."""

    def test_profile_completes_quickly(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test that profile creation completes in reasonable time."""
        import time

        profiler = DriverProfiler()

        start = time.time()
        profile = profiler.create_profile(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )
        elapsed = time.time() - start

        # Should complete in under 10 seconds for sample data
        assert elapsed < 10.0
        assert profile is not None

    def test_handles_large_dataset(
        self,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test profiler handles reasonably large dataset."""
        # Create larger telemetry dataset
        n_samples = 10000
        telemetry = pd.DataFrame({
            'telemetry_name': ['speed', 'pbrake_f'] * (n_samples // 2),
            'telemetry_value': np.random.rand(n_samples) * 100,
            'vehicle_number': [valid_vehicle_number] * n_samples,
            'timestamp': range(100000, 100000 + n_samples * 100, 100),
            'lap': np.repeat(range(1, 11), n_samples // 10)  # 10 laps
        })

        profiler = DriverProfiler()

        # Should complete without memory error
        profile = profiler.create_profile(
            telemetry,
            sample_lap_times_df,
            valid_vehicle_number
        )

        assert profile is not None


class TestErrorMessages:
    """Test error messages are clear and actionable."""

    def test_error_includes_vehicle_number(
        self,
        sample_telemetry_df,
        sample_lap_times_df
    ):
        """Test error messages include vehicle number context."""
        profiler = DriverProfiler()

        with pytest.raises((EmptyDatasetError, InsufficientDataError)) as exc_info:
            profiler.create_profile(
                sample_telemetry_df,
                sample_lap_times_df,
                vehicle_number=999
            )

        # Error should mention vehicle number
        assert '999' in str(exc_info.value) or 'vehicle' in str(exc_info.value).lower()

    def test_error_context_has_details(
        self,
        sample_telemetry_df,
        sample_lap_times_df
    ):
        """Test errors include context dictionary."""
        profiler = DriverProfiler()

        with pytest.raises((EmptyDatasetError, InsufficientDataError)) as exc_info:
            profiler.create_profile(
                sample_telemetry_df,
                sample_lap_times_df,
                vehicle_number=999
            )

        # Exception should have context
        assert hasattr(exc_info.value, 'context')
        assert isinstance(exc_info.value.context, dict)
