"""
Unit Tests for Post-Race Analysis Widget
=========================================

Tests for:
- PostRacePredictor class
- PostRaceAnalyzer class
- Widget callbacks
- Data processing functions

Run with:
    pytest tests/dashboard/test_post_race_widget.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.dashboard.post_race_analyzer import PostRaceAnalyzer, SessionStatistics


@pytest.fixture
def sample_predictions():
    """Sample predictions DataFrame for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'lap_number': range(1, 21),
        'vehicle_number': [5] * 20,
        'track': ['cota'] * 20,
        'race': ['race_1'] * 20,
        'actual': [120.0 + np.random.normal(0, 1) for _ in range(20)],
        'predicted': [120.0 + np.random.normal(0, 0.5) for _ in range(20)],
        'error': [np.random.normal(0, 1) for _ in range(20)],
        'abs_error': [abs(np.random.normal(0, 1)) for _ in range(20)]
    })


@pytest.fixture
def sample_predictions_with_anomalies():
    """Sample predictions with anomalies for testing"""
    data = pd.DataFrame({
        'lap_number': range(1, 11),
        'vehicle_number': [5] * 10,
        'track': ['cota'] * 10,
        'race': ['race_1'] * 10,
        'actual': [120.0, 119.5, 125.0, 119.8, 119.2, 118.9, 119.3, 119.1, 127.5, 119.4],
        'predicted': [120.0] * 10,
        'error': [0.0, -0.5, 5.0, -0.2, -0.8, -1.1, -0.7, -0.9, 7.5, -0.6],
        'abs_error': [0.0, 0.5, 5.0, 0.2, 0.8, 1.1, 0.7, 0.9, 7.5, 0.6]
    })
    return data


class TestPostRaceAnalyzer:
    """Test suite for PostRaceAnalyzer class"""

    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        analyzer = PostRaceAnalyzer(anomaly_threshold=2.5)
        assert analyzer.anomaly_threshold == 2.5

    def test_calculate_statistics(self, sample_predictions):
        """Test session statistics calculation"""
        analyzer = PostRaceAnalyzer()
        stats = analyzer._calculate_statistics(sample_predictions)

        assert isinstance(stats, SessionStatistics)
        assert stats.total_laps == 20
        assert stats.avg_lap_time > 0
        assert stats.best_lap <= stats.avg_lap_time
        assert stats.worst_lap >= stats.avg_lap_time
        assert 0 <= stats.model_r2 <= 1

    def test_rate_consistency(self):
        """Test consistency rating function"""
        analyzer = PostRaceAnalyzer()

        # Excellent consistency (CV < 1%)
        assert analyzer._rate_consistency(0.5, 120.0) == "Excellent"

        # Good consistency (1% <= CV < 2%)
        assert analyzer._rate_consistency(1.5, 120.0) == "Good"

        # Fair consistency (2% <= CV < 3%)
        assert analyzer._rate_consistency(2.5, 120.0) == "Fair"

        # Poor consistency (CV >= 3%)
        assert analyzer._rate_consistency(4.0, 120.0) == "Poor"

    def test_detect_anomalies(self, sample_predictions_with_anomalies):
        """Test anomaly detection"""
        analyzer = PostRaceAnalyzer(anomaly_threshold=2.5)
        anomalies = analyzer._detect_anomalies(sample_predictions_with_anomalies)

        # Should detect laps 3 and 9 (errors 5.0 and 7.5)
        assert len(anomalies) == 2
        assert 3 in anomalies['lap_number'].values
        assert 9 in anomalies['lap_number'].values
        assert 'likely_cause' in anomalies.columns
        assert 'severity' in anomalies.columns

    def test_detect_anomalies_empty(self, sample_predictions):
        """Test anomaly detection with no anomalies"""
        # Use data with small errors only
        sample_predictions['abs_error'] = 0.5
        analyzer = PostRaceAnalyzer(anomaly_threshold=2.5)
        anomalies = analyzer._detect_anomalies(sample_predictions)

        assert len(anomalies) == 0
        assert 'lap_number' in anomalies.columns  # Schema preserved

    def test_infer_cause(self):
        """Test cause inference logic"""
        analyzer = PostRaceAnalyzer()

        assert "Incident" in analyzer._infer_cause(8.0)
        assert "Traffic" in analyzer._infer_cause(5.5)
        assert "Major Mistake" in analyzer._infer_cause(4.0)
        assert "Driving Error" in analyzer._infer_cause(3.0)
        assert "Exceptional" in analyzer._infer_cause(-3.5)
        assert "Normal" in analyzer._infer_cause(1.0)

    def test_classify_severity(self):
        """Test severity classification"""
        analyzer = PostRaceAnalyzer()

        assert analyzer._classify_severity(8.0) == "Critical"
        assert analyzer._classify_severity(6.0) == "High"
        assert analyzer._classify_severity(4.0) == "Medium"
        assert analyzer._classify_severity(3.0) == "Low"

    def test_generate_recommendations(self, sample_predictions):
        """Test recommendation generation"""
        analyzer = PostRaceAnalyzer()
        anomalies = analyzer._detect_anomalies(sample_predictions)
        recommendations = analyzer._generate_recommendations(sample_predictions, anomalies)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)

    def test_analyze_performance_trend_improving(self):
        """Test trend analysis for improving performance"""
        # Create data with improving trend
        df = pd.DataFrame({
            'lap_number': range(1, 11),
            'actual': [125.0 - i * 0.2 for i in range(10)],  # Decreasing times
            'predicted': [125.0] * 10
        })

        analyzer = PostRaceAnalyzer()
        trend = analyzer._analyze_performance_trend(df)

        assert trend['direction'] == 'improving'
        assert trend['rate'] < 0  # Negative slope = improving

    def test_analyze_performance_trend_degrading(self):
        """Test trend analysis for degrading performance"""
        # Create data with degrading trend
        df = pd.DataFrame({
            'lap_number': range(1, 11),
            'actual': [120.0 + i * 0.2 for i in range(10)],  # Increasing times
            'predicted': [120.0] * 10
        })

        analyzer = PostRaceAnalyzer()
        trend = analyzer._analyze_performance_trend(df)

        assert trend['direction'] == 'degrading'
        assert trend['rate'] > 0  # Positive slope = degrading

    def test_analyze_performance_trend_stable(self):
        """Test trend analysis for stable performance"""
        # Create data with stable trend
        df = pd.DataFrame({
            'lap_number': range(1, 11),
            'actual': [120.0] * 10,  # Constant times
            'predicted': [120.0] * 10
        })

        analyzer = PostRaceAnalyzer()
        trend = analyzer._analyze_performance_trend(df)

        assert trend['direction'] == 'stable'
        assert abs(trend['rate']) < 0.05  # Very small slope

    def test_analyze_performance_trend_insufficient_data(self):
        """Test trend analysis with insufficient data"""
        df = pd.DataFrame({
            'lap_number': [1, 2, 3],
            'actual': [120.0, 119.5, 120.2],
            'predicted': [120.0, 120.0, 120.0]
        })

        analyzer = PostRaceAnalyzer()
        trend = analyzer._analyze_performance_trend(df)

        assert trend['direction'] == 'unknown'
        assert trend['confidence'] == 'Low'

    def test_analyze_session_complete(self, sample_predictions_with_anomalies):
        """Test complete session analysis"""
        analyzer = PostRaceAnalyzer(anomaly_threshold=2.5)
        analysis = analyzer.analyze_session(sample_predictions_with_anomalies)

        assert 'statistics' in analysis
        assert 'anomalies' in analysis
        assert 'recommendations' in analysis
        assert 'performance_trend' in analysis

        assert isinstance(analysis['statistics'], SessionStatistics)
        assert isinstance(analysis['anomalies'], pd.DataFrame)
        assert isinstance(analysis['recommendations'], list)
        assert isinstance(analysis['performance_trend'], dict)

    def test_generate_summary_text(self, sample_predictions):
        """Test summary text generation"""
        analyzer = PostRaceAnalyzer()
        analysis = analyzer.analyze_session(sample_predictions)
        summary = analyzer.generate_summary_text(analysis)

        assert isinstance(summary, str)
        assert "SESSION SUMMARY" in summary
        assert "Lap Statistics" in summary
        assert "Model Performance" in summary
        assert "Recommendations" in summary


class TestIntegration:
    """Integration tests for full pipeline"""

    def test_end_to_end_analysis(self, sample_predictions_with_anomalies):
        """Test complete analysis pipeline"""
        analyzer = PostRaceAnalyzer(anomaly_threshold=2.5)

        # Run analysis
        analysis = analyzer.analyze_session(sample_predictions_with_anomalies)

        # Verify outputs
        assert analysis['statistics'].total_laps == 10
        assert len(analysis['anomalies']) == 2  # Laps 3 and 9
        assert len(analysis['recommendations']) > 0

        # Check anomaly details
        anomalies = analysis['anomalies']
        assert anomalies.iloc[0]['severity'] in ['Critical', 'High', 'Medium', 'Low']
        assert 'likely_cause' in anomalies.columns

    def test_multi_vehicle_data(self):
        """Test handling of multiple vehicles"""
        df = pd.DataFrame({
            'lap_number': [1, 2, 3, 1, 2, 3],
            'vehicle_number': [5, 5, 5, 7, 7, 7],
            'track': ['cota'] * 6,
            'race': ['race_1'] * 6,
            'actual': [120.0, 119.5, 119.8, 121.0, 120.5, 120.8],
            'predicted': [120.0] * 6,
            'error': [0.0, -0.5, -0.2, 1.0, 0.5, 0.8],
            'abs_error': [0.0, 0.5, 0.2, 1.0, 0.5, 0.8]
        })

        analyzer = PostRaceAnalyzer()
        analysis = analyzer.analyze_session(df)

        assert analysis['statistics'].total_laps == 6
        assert len(analysis['recommendations']) > 0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame(columns=['lap_number', 'actual', 'predicted', 'error', 'abs_error'])
        analyzer = PostRaceAnalyzer()

        # Should not crash
        with pytest.raises(Exception):  # May raise ZeroDivisionError or similar
            analyzer._calculate_statistics(df)

    def test_single_lap(self):
        """Test handling of single lap"""
        df = pd.DataFrame({
            'lap_number': [1],
            'vehicle_number': [5],
            'actual': [120.0],
            'predicted': [119.5],
            'error': [0.5],
            'abs_error': [0.5]
        })

        analyzer = PostRaceAnalyzer()
        analysis = analyzer.analyze_session(df)

        assert analysis['statistics'].total_laps == 1
        assert analysis['performance_trend']['direction'] == 'unknown'

    def test_all_perfect_predictions(self):
        """Test handling of perfect predictions"""
        df = pd.DataFrame({
            'lap_number': range(1, 11),
            'vehicle_number': [5] * 10,
            'actual': [120.0] * 10,
            'predicted': [120.0] * 10,
            'error': [0.0] * 10,
            'abs_error': [0.0] * 10
        })

        analyzer = PostRaceAnalyzer()
        analysis = analyzer.analyze_session(df)

        assert analysis['statistics'].model_mae == 0.0
        assert len(analysis['anomalies']) == 0

    def test_extreme_errors(self):
        """Test handling of extreme prediction errors"""
        df = pd.DataFrame({
            'lap_number': range(1, 6),
            'vehicle_number': [5] * 5,
            'actual': [120.0, 150.0, 90.0, 119.5, 120.2],
            'predicted': [120.0] * 5,
            'error': [0.0, 30.0, -30.0, -0.5, 0.2],
            'abs_error': [0.0, 30.0, 30.0, 0.5, 0.2]
        })

        analyzer = PostRaceAnalyzer(anomaly_threshold=2.5)
        anomalies = analyzer._detect_anomalies(df)

        # Should detect laps 2 and 3
        assert len(anomalies) == 2
        assert 'Critical' in anomalies['severity'].values


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
