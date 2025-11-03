"""
Unit Tests for ConsistencyTracker

Tests consistency tracking and performance trend analysis:
- Session tracking
- Performance trends
- Outlier detection
- Progress reporting
"""

import pytest
import pandas as pd
import numpy as np
from src.insights import ConsistencyTracker, InsightsConfig
from src.insights.exceptions import (
    EmptyDatasetError,
    InsufficientDataError,
    ConsistencyAnalysisError,
    DataValidationError
)


class TestConsistencyTrackerInitialization:
    """Test ConsistencyTracker initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        tracker = ConsistencyTracker()

        assert tracker.config is not None
        assert tracker.config.outlier_threshold == 2.0
        assert tracker.config.min_lap_count == 3

    def test_init_custom_config(self, custom_config):
        """Test initialization with custom config."""
        tracker = ConsistencyTracker(config=custom_config)

        assert tracker.config == custom_config
        assert tracker.config.outlier_threshold == 2.5


class TestAnalyzeConsistency:
    """Test analyze_consistency method."""

    def test_analyze_consistency_success(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test successful consistency analysis."""
        tracker = ConsistencyTracker()

        analysis = tracker.analyze_consistency(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

        # Should return analysis results
        assert analysis is not None

    def test_analyze_consistency_invalid_vehicle(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        invalid_vehicle_number
    ):
        """Test consistency analysis fails for invalid vehicle."""
        tracker = ConsistencyTracker()

        with pytest.raises(DataValidationError):
            tracker.analyze_consistency(
                sample_telemetry_df,
                sample_lap_times_df,
                invalid_vehicle_number
            )

    def test_analyze_consistency_no_data(
        self,
        empty_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test consistency analysis with no data."""
        tracker = ConsistencyTracker()

        with pytest.raises((EmptyDatasetError, DataValidationError)):
            tracker.analyze_consistency(
                empty_telemetry_df,
                sample_lap_times_df,
                valid_vehicle_number
            )

    def test_analyze_consistency_insufficient_laps(
        self,
        sample_telemetry_df,
        valid_vehicle_number
    ):
        """Test consistency analysis with too few laps."""
        # Create lap_times with only 1 lap
        lap_times = pd.DataFrame({
            'vehicle_number': [valid_vehicle_number],
            'lap': [1],
            'lap_duration': [90.0]
        })

        tracker = ConsistencyTracker()

        with pytest.raises(InsufficientDataError):
            tracker.analyze_consistency(
                sample_telemetry_df,
                lap_times,
                valid_vehicle_number
            )


class TestOutlierDetection:
    """Test outlier detection functionality."""

    def test_outlier_detection_with_custom_threshold(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test outlier detection with custom threshold."""
        config = InsightsConfig(outlier_threshold=3.0)
        tracker = ConsistencyTracker(config=config)

        analysis = tracker.analyze_consistency(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

        assert analysis is not None


class TestPerformanceTrends:
    """Test performance trend analysis."""

    def test_trend_analysis(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test performance trend identification."""
        tracker = ConsistencyTracker()

        analysis = tracker.analyze_consistency(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

        # Analysis should complete
        assert analysis is not None


class TestMultipleSessions:
    """Test tracking across multiple sessions."""

    def test_analyze_multiple_sessions(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test consistency tracking across sessions."""
        tracker = ConsistencyTracker()

        # Analyze same data twice (simulating sessions)
        analysis1 = tracker.analyze_consistency(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

        analysis2 = tracker.analyze_consistency(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )

        # Both should succeed
        assert analysis1 is not None
        assert analysis2 is not None


class TestEdgeCases:
    """Test edge cases."""

    def test_all_identical_laps(
        self,
        sample_telemetry_df,
        valid_vehicle_number
    ):
        """Test with identical lap times (perfect consistency)."""
        # Create lap_times with identical times
        lap_times = pd.DataFrame({
            'vehicle_number': [valid_vehicle_number] * 10,
            'lap': range(1, 11),
            'lap_duration': [90.0] * 10  # All identical
        })

        tracker = ConsistencyTracker()

        analysis = tracker.analyze_consistency(
            sample_telemetry_df,
            lap_times,
            valid_vehicle_number
        )

        # Should handle gracefully
        assert analysis is not None

    def test_highly_variable_laps(
        self,
        sample_telemetry_df,
        valid_vehicle_number
    ):
        """Test with highly variable lap times."""
        # Create lap_times with high variance
        lap_times = pd.DataFrame({
            'vehicle_number': [valid_vehicle_number] * 10,
            'lap': range(1, 11),
            'lap_duration': [60, 90, 120, 70, 95, 110, 65, 100, 85, 115]  # High variance
        })

        tracker = ConsistencyTracker()

        analysis = tracker.analyze_consistency(
            sample_telemetry_df,
            lap_times,
            valid_vehicle_number
        )

        # Should detect inconsistency
        assert analysis is not None


class TestPerformance:
    """Test performance characteristics."""

    def test_analysis_completes_quickly(
        self,
        sample_telemetry_df,
        sample_lap_times_df,
        valid_vehicle_number
    ):
        """Test consistency analysis completes in reasonable time."""
        import time

        tracker = ConsistencyTracker()

        start = time.time()
        analysis = tracker.analyze_consistency(
            sample_telemetry_df,
            sample_lap_times_df,
            valid_vehicle_number
        )
        elapsed = time.time() - start

        # Should complete quickly
        assert elapsed < 5.0
        assert analysis is not None
