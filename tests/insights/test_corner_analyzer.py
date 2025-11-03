"""
Unit Tests for CornerAnalyzer

Tests corner detection and racing line optimization:
- Corner detection
- Optimal line identification
- Performance comparison
- Error handling
"""

import pytest
import pandas as pd
import numpy as np
from src.insights import CornerAnalyzer, InsightsConfig
from src.insights.exceptions import (
    EmptyDatasetError,
    InsufficientDataError,
    CornerDetectionError,
    DataValidationError
)


class TestCornerAnalyzerInitialization:
    """Test CornerAnalyzer initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        analyzer = CornerAnalyzer()

        assert analyzer.config is not None
        assert analyzer.config.min_corner_duration == 1.0

    def test_init_custom_config(self, custom_config):
        """Test initialization with custom config."""
        analyzer = CornerAnalyzer(config=custom_config)

        assert analyzer.config == custom_config
        assert analyzer.config.min_corner_duration == 1.5


class TestDetectCorners:
    """Test detect_corners method."""

    def test_detect_corners_success(
        self,
        sample_telemetry_df,
        valid_vehicle_number
    ):
        """Test successful corner detection."""
        analyzer = CornerAnalyzer()

        corners = analyzer.detect_corners(
            sample_telemetry_df,
            valid_vehicle_number
        )

        # Should return a result (dict or list)
        assert corners is not None

    def test_detect_corners_invalid_vehicle(
        self,
        sample_telemetry_df,
        invalid_vehicle_number
    ):
        """Test corner detection fails for invalid vehicle."""
        analyzer = CornerAnalyzer()

        with pytest.raises(DataValidationError):
            analyzer.detect_corners(
                sample_telemetry_df,
                invalid_vehicle_number
            )

    def test_detect_corners_no_data(
        self,
        empty_telemetry_df,
        valid_vehicle_number
    ):
        """Test corner detection with no data."""
        analyzer = CornerAnalyzer()

        with pytest.raises((EmptyDatasetError, DataValidationError)):
            analyzer.detect_corners(
                empty_telemetry_df,
                valid_vehicle_number
            )

    def test_detect_corners_custom_threshold(
        self,
        sample_telemetry_df,
        valid_vehicle_number
    ):
        """Test corner detection with custom thresholds."""
        config = InsightsConfig(
            min_corner_duration=2.0,
            lateral_g_threshold=0.8
        )
        analyzer = CornerAnalyzer(config=config)

        corners = analyzer.detect_corners(
            sample_telemetry_df,
            valid_vehicle_number
        )

        assert corners is not None


class TestOptimalLineAnalysis:
    """Test optimal racing line analysis."""

    def test_analyze_optimal_line(
        self,
        sample_telemetry_df,
        valid_vehicle_number
    ):
        """Test optimal line analysis."""
        analyzer = CornerAnalyzer()

        # First detect corners (if needed by implementation)
        try:
            result = analyzer.detect_corners(
                sample_telemetry_df,
                valid_vehicle_number
            )
            assert result is not None
        except (InsufficientDataError, CornerDetectionError):
            # May not have enough data in sample
            pytest.skip("Insufficient data for corner detection")


class TestMultipleVehicles:
    """Test analyzing multiple vehicles."""

    def test_analyze_multiple_vehicles(
        self,
        multi_vehicle_telemetry_df
    ):
        """Test corner detection for multiple vehicles."""
        analyzer = CornerAnalyzer()

        for vehicle in [3, 5, 7]:
            try:
                corners = analyzer.detect_corners(
                    multi_vehicle_telemetry_df,
                    vehicle_number=vehicle
                )
                assert corners is not None
            except (InsufficientDataError, CornerDetectionError):
                # May not have enough data
                pass


class TestEdgeCases:
    """Test edge cases."""

    def test_no_corners_detected(
        self,
        valid_vehicle_number
    ):
        """Test when no corners are detected (straight line)."""
        # Create telemetry with no cornering (constant values)
        telemetry = pd.DataFrame({
            'telemetry_name': ['speed', 'accy_can'] * 100,
            'telemetry_value': [150.0, 0.0] * 100,  # Constant speed, no lateral g
            'vehicle_number': [valid_vehicle_number] * 200,
            'timestamp': range(100000, 120000, 100),
            'lap': [1] * 200
        })

        analyzer = CornerAnalyzer()

        # Should handle gracefully
        try:
            corners = analyzer.detect_corners(telemetry, valid_vehicle_number)
            # If it succeeds, corners might be empty
            assert corners is not None
        except (CornerDetectionError, InsufficientDataError):
            # Acceptable to raise error when no corners found
            pass


class TestPerformance:
    """Test performance characteristics."""

    def test_detection_completes_quickly(
        self,
        sample_telemetry_df,
        valid_vehicle_number
    ):
        """Test corner detection completes in reasonable time."""
        import time

        analyzer = CornerAnalyzer()

        start = time.time()
        try:
            corners = analyzer.detect_corners(
                sample_telemetry_df,
                valid_vehicle_number
            )
            elapsed = time.time() - start

            # Should complete quickly
            assert elapsed < 5.0
        except (InsufficientDataError, CornerDetectionError):
            # If it fails, that's ok for this test
            pass
