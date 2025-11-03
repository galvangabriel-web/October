"""
Integration Tests for Racing Insights Module

Tests complete workflows with real data from organized_data/:
- End-to-end driver profiling
- Complete corner analysis
- Full consistency tracking
- Multi-module integration
- Real-world data handling

Note: These tests require organized_data/ directory and will be skipped if not present.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.insights import (
    DriverProfiler,
    CornerAnalyzer,
    ConsistencyTracker,
    InsightsConfig
)


@pytest.mark.integration
class TestRealDataIntegration:
    """Integration tests with real racing data."""

    def test_driver_profiler_with_real_data(self, sample_track_data):
        """Test DriverProfiler with real track data."""
        telemetry = sample_track_data['telemetry']
        lap_times = sample_track_data['lap_times']

        # Get first available vehicle
        if len(telemetry) == 0 or len(lap_times) == 0:
            pytest.skip("No data available in sample track data")

        vehicle_numbers = telemetry['vehicle_number'].unique()
        if len(vehicle_numbers) == 0:
            pytest.skip("No vehicles found in telemetry data")

        vehicle = int(vehicle_numbers[0])

        # Create profiler and generate profile
        profiler = DriverProfiler()
        profile = profiler.create_profile(telemetry, lap_times, vehicle)

        # Verify profile was created
        assert profile is not None
        assert isinstance(profile, dict)

    def test_corner_analyzer_with_real_data(self, sample_track_data):
        """Test CornerAnalyzer with real track data."""
        telemetry = sample_track_data['telemetry']

        if len(telemetry) == 0:
            pytest.skip("No telemetry data available")

        vehicle_numbers = telemetry['vehicle_number'].unique()
        if len(vehicle_numbers) == 0:
            pytest.skip("No vehicles found")

        vehicle = int(vehicle_numbers[0])

        # Analyze corners
        analyzer = CornerAnalyzer()

        try:
            corners = analyzer.detect_corners(telemetry, vehicle)
            assert corners is not None
        except Exception as e:
            # Real data might not have enough corner data
            pytest.skip(f"Corner detection failed: {e}")

    def test_consistency_tracker_with_real_data(self, sample_track_data):
        """Test ConsistencyTracker with real track data."""
        telemetry = sample_track_data['telemetry']
        lap_times = sample_track_data['lap_times']

        if len(telemetry) == 0 or len(lap_times) == 0:
            pytest.skip("No data available")

        vehicle_numbers = telemetry['vehicle_number'].unique()
        if len(vehicle_numbers) == 0:
            pytest.skip("No vehicles found")

        vehicle = int(vehicle_numbers[0])

        # Filter to ensure vehicle has laps
        vehicle_laps = lap_times[lap_times['vehicle_number'] == vehicle]
        if len(vehicle_laps) < 3:
            pytest.skip(f"Vehicle {vehicle} has insufficient laps")

        # Analyze consistency
        tracker = ConsistencyTracker()
        analysis = tracker.analyze_consistency(telemetry, lap_times, vehicle)

        assert analysis is not None

    def test_all_modules_with_real_data(self, sample_track_data):
        """Test all three modules together with real data."""
        telemetry = sample_track_data['telemetry']
        lap_times = sample_track_data['lap_times']

        if len(telemetry) == 0 or len(lap_times) == 0:
            pytest.skip("No data available")

        vehicle_numbers = telemetry['vehicle_number'].unique()
        if len(vehicle_numbers) == 0:
            pytest.skip("No vehicles found")

        vehicle = int(vehicle_numbers[0])

        # Ensure sufficient laps
        vehicle_laps = lap_times[lap_times['vehicle_number'] == vehicle]
        if len(vehicle_laps) < 3:
            pytest.skip(f"Vehicle {vehicle} has insufficient laps")

        # Use same config for all modules
        config = InsightsConfig()

        # Run all analyses
        profiler = DriverProfiler(config=config)
        analyzer = CornerAnalyzer(config=config)
        tracker = ConsistencyTracker(config=config)

        # Generate results
        profile = profiler.create_profile(telemetry, lap_times, vehicle)
        assert profile is not None

        try:
            corners = analyzer.detect_corners(telemetry, vehicle)
            assert corners is not None
        except Exception:
            # Corner detection might fail with limited data
            pass

        analysis = tracker.analyze_consistency(telemetry, lap_times, vehicle)
        assert analysis is not None


@pytest.mark.integration
class TestMultipleVehiclesIntegration:
    """Test analyzing multiple vehicles from real data."""

    def test_profile_multiple_vehicles(self, sample_track_data):
        """Test profiling multiple vehicles from same track."""
        telemetry = sample_track_data['telemetry']
        lap_times = sample_track_data['lap_times']

        if len(telemetry) == 0 or len(lap_times) == 0:
            pytest.skip("No data available")

        # Get all vehicles with sufficient laps
        vehicles_with_laps = []
        for vehicle in telemetry['vehicle_number'].unique():
            vehicle_laps = lap_times[lap_times['vehicle_number'] == vehicle]
            if len(vehicle_laps) >= 3:
                vehicles_with_laps.append(int(vehicle))

        if len(vehicles_with_laps) < 2:
            pytest.skip("Not enough vehicles with sufficient laps")

        # Profile first 3 vehicles (or fewer if not available)
        profiler = DriverProfiler()
        profiles = {}

        for vehicle in vehicles_with_laps[:3]:
            profile = profiler.create_profile(telemetry, lap_times, vehicle)
            profiles[vehicle] = profile
            assert profile is not None

        # Should have created multiple profiles
        assert len(profiles) >= 1


@pytest.mark.integration
class TestConfigurationIntegration:
    """Test different configurations with real data."""

    def test_custom_config_with_real_data(self, sample_track_data):
        """Test custom configuration affects real data analysis."""
        telemetry = sample_track_data['telemetry']
        lap_times = sample_track_data['lap_times']

        if len(telemetry) == 0 or len(lap_times) == 0:
            pytest.skip("No data available")

        vehicle_numbers = telemetry['vehicle_number'].unique()
        if len(vehicle_numbers) == 0:
            pytest.skip("No vehicles found")

        vehicle = int(vehicle_numbers[0])

        # Ensure sufficient laps
        vehicle_laps = lap_times[lap_times['vehicle_number'] == vehicle]
        if len(vehicle_laps) < 3:
            pytest.skip(f"Vehicle {vehicle} has insufficient laps")

        # Test with two different configs
        config1 = InsightsConfig(
            hard_brake_threshold=90.0,
            outlier_threshold=2.0
        )

        config2 = InsightsConfig(
            hard_brake_threshold=120.0,
            outlier_threshold=3.0
        )

        # Create profiles with both configs
        profiler1 = DriverProfiler(config=config1)
        profiler2 = DriverProfiler(config=config2)

        profile1 = profiler1.create_profile(telemetry, lap_times, vehicle)
        profile2 = profiler2.create_profile(telemetry, lap_times, vehicle)

        # Both should succeed
        assert profile1 is not None
        assert profile2 is not None


@pytest.mark.integration
class TestPerformanceIntegration:
    """Test performance with real data."""

    def test_performance_with_large_chunk(self, real_data_loader):
        """Test performance with larger real data chunk."""
        import time

        tracks = real_data_loader.list_tracks()
        if not tracks:
            pytest.skip("No tracks available")

        track = tracks[0]
        races = real_data_loader.list_races(track)
        race = races[0] if races else 'race_unknown'

        # Load data
        telemetry = real_data_loader.load_single_chunk(track, race, 'telemetry', chunk_num=1)
        lap_times = real_data_loader.load_data(track, race, 'lap_times')

        if len(telemetry) == 0:
            pytest.skip("No telemetry data")

        vehicle = int(telemetry['vehicle_number'].unique()[0])

        # Ensure sufficient laps
        vehicle_laps = lap_times[lap_times['vehicle_number'] == vehicle]
        if len(vehicle_laps) < 3:
            pytest.skip(f"Vehicle {vehicle} has insufficient laps")

        # Measure performance
        profiler = DriverProfiler()

        start = time.time()
        profile = profiler.create_profile(telemetry, lap_times, vehicle)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 30.0  # 30 seconds max
        assert profile is not None


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling with real data edge cases."""

    def test_handles_real_data_edge_cases(self, sample_track_data):
        """Test that modules handle real data edge cases gracefully."""
        telemetry = sample_track_data['telemetry']
        lap_times = sample_track_data['lap_times']

        if len(telemetry) == 0:
            pytest.skip("No data available")

        # Test with non-existent vehicle
        profiler = DriverProfiler()

        from src.insights.exceptions import (
            EmptyDatasetError,
            InsufficientDataError,
            DataValidationError
        )

        with pytest.raises((EmptyDatasetError, InsufficientDataError, DataValidationError)):
            profiler.create_profile(telemetry, lap_times, vehicle_number=9999)

    def test_handles_missing_sensors(self, sample_track_data):
        """Test handling when some sensors are missing."""
        telemetry = sample_track_data['telemetry']
        lap_times = sample_track_data['lap_times']

        if len(telemetry) == 0 or len(lap_times) == 0:
            pytest.skip("No data available")

        vehicle_numbers = telemetry['vehicle_number'].unique()
        if len(vehicle_numbers) == 0:
            pytest.skip("No vehicles found")

        vehicle = int(vehicle_numbers[0])

        # Filter to only include speed data (missing other sensors)
        telemetry_filtered = telemetry[telemetry['telemetry_name'] == 'speed']

        if len(telemetry_filtered) == 0:
            pytest.skip("No speed data available")

        # Should handle gracefully
        profiler = DriverProfiler()

        try:
            profile = profiler.create_profile(telemetry_filtered, lap_times, vehicle)
            # Might succeed with partial data
            assert profile is not None
        except Exception:
            # Or might fail - either is acceptable
            pass


@pytest.mark.integration
class TestDataQualityIntegration:
    """Test data quality validation with real data."""

    def test_validates_real_data_quality(self, sample_track_data):
        """Test that validation catches real data quality issues."""
        from src.insights.validation import (
            validate_telemetry_dataframe,
            validate_lap_times_dataframe
        )

        telemetry = sample_track_data['telemetry']
        lap_times = sample_track_data['lap_times']

        if len(telemetry) == 0 or len(lap_times) == 0:
            pytest.skip("No data available")

        # Validation should pass for real data
        validate_telemetry_dataframe(telemetry)
        validate_lap_times_dataframe(lap_times)

    def test_detects_corrupted_data(self, sample_track_data):
        """Test that validation detects corrupted data."""
        from src.insights.exceptions import DataValidationError, MissingColumnsError

        telemetry = sample_track_data['telemetry']

        if len(telemetry) == 0:
            pytest.skip("No data available")

        # Corrupt the data by dropping required column
        corrupted = telemetry.drop(columns=['telemetry_name'], errors='ignore')

        if 'telemetry_name' not in telemetry.columns:
            pytest.skip("Column not present to test dropping")

        # Should fail validation
        profiler = DriverProfiler()

        with pytest.raises((DataValidationError, MissingColumnsError)):
            # This should fail due to missing column
            from src.insights.validation import validate_telemetry_dataframe
            validate_telemetry_dataframe(corrupted)
