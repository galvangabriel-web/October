"""
Integration Tests for Winning Edge Widget
===========================================

This module contains comprehensive integration tests for the Winning Edge Dashboard
widget, including:
- Edge case bug fixes (position gain predictor with various lap counts)
- Dashboard tab functionality
- Callback functionality
- Data flow from telemetry to visualization
- Error handling for edge cases

Tests cover:
1. Edge case handling (1-5 laps, 10 laps, 20+ laps)
2. Widget integration in main dashboard
3. Callback triggers and data updates
4. Visual component rendering
5. Data validation and bounds checking
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go


class TestPositionGainPredictorBugFix:
    """Test the edge case bug fix for Position Gain Predictor with <10 laps."""

    def test_position_gain_predictor_single_lap(self):
        """Test Position Gain Predictor with 1 lap (edge case)."""
        from src.dashboard.winning_edge_widget import create_position_gain_predictor

        # This should not raise IndexError
        fig = create_position_gain_predictor(time_gain_per_lap=0.44, num_laps=1)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # At least 2 traces (time and position)

        # Check that annotations were added (final lap milestone)
        assert len(fig.layout.annotations) >= 1

    def test_position_gain_predictor_five_laps(self):
        """Test Position Gain Predictor with 5 laps (edge case < 10)."""
        from src.dashboard.winning_edge_widget import create_position_gain_predictor

        fig = create_position_gain_predictor(time_gain_per_lap=0.44, num_laps=5)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

        # Should have annotations: midpoint + final lap
        assert len(fig.layout.annotations) >= 2

    def test_position_gain_predictor_ten_laps(self):
        """Test Position Gain Predictor with 10 laps (boundary case)."""
        from src.dashboard.winning_edge_widget import create_position_gain_predictor

        fig = create_position_gain_predictor(time_gain_per_lap=0.44, num_laps=10)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

        # Should have annotations: lap 10 + final lap
        assert len(fig.layout.annotations) >= 2

    def test_position_gain_predictor_twenty_laps(self):
        """Test Position Gain Predictor with 20 laps (original case)."""
        from src.dashboard.winning_edge_widget import create_position_gain_predictor

        fig = create_position_gain_predictor(time_gain_per_lap=0.44, num_laps=20)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

        # Should have annotations: lap 10 + lap 20 + final lap
        assert len(fig.layout.annotations) >= 3

    def test_position_gain_predictor_thirty_laps(self):
        """Test Position Gain Predictor with 30 laps (long race)."""
        from src.dashboard.winning_edge_widget import create_position_gain_predictor

        fig = create_position_gain_predictor(time_gain_per_lap=0.44, num_laps=30)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

        # Should have annotations: lap 10 + lap 20 + final lap
        assert len(fig.layout.annotations) >= 3

    def test_position_gain_predictor_different_time_gains(self):
        """Test Position Gain Predictor with various time gain values."""
        from src.dashboard.winning_edge_widget import create_position_gain_predictor

        test_cases = [0.1, 0.25, 0.44, 0.75, 1.0]

        for time_gain in test_cases:
            fig = create_position_gain_predictor(time_gain_per_lap=time_gain, num_laps=15)
            assert isinstance(fig, go.Figure)
            assert len(fig.data) >= 2

    def test_position_gain_predictor_zero_time_gain(self):
        """Test Position Gain Predictor with zero time gain (edge case)."""
        from src.dashboard.winning_edge_widget import create_position_gain_predictor

        fig = create_position_gain_predictor(time_gain_per_lap=0.0, num_laps=10)

        assert isinstance(fig, go.Figure)
        # All position gains should be 0
        y_values = fig.data[1]['y']  # Position gains trace
        assert all(val == 0 for val in y_values)

    def test_position_gain_predictor_no_index_errors(self):
        """Test that no IndexError is raised for any lap count."""
        from src.dashboard.winning_edge_widget import create_position_gain_predictor

        # Test a wide range of lap counts
        for num_laps in [1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 50]:
            try:
                fig = create_position_gain_predictor(time_gain_per_lap=0.44, num_laps=num_laps)
                assert isinstance(fig, go.Figure)
            except IndexError as e:
                pytest.fail(f"IndexError raised for {num_laps} laps: {e}")


class TestWinningEdgeLayout:
    """Test the Winning Edge widget layout."""

    def test_create_winning_edge_layout(self):
        """Test that layout is created without errors."""
        from src.dashboard.winning_edge_widget import create_winning_edge_layout

        layout = create_winning_edge_layout()

        # Layout should be an html.Div
        assert layout is not None
        assert hasattr(layout, 'children')

    def test_layout_contains_tabs(self):
        """Test that layout contains tab structure."""
        from src.dashboard.winning_edge_widget import create_winning_edge_layout

        layout = create_winning_edge_layout()

        # Convert to string to check for tab identifiers
        layout_str = str(layout)
        assert 'winning-edge-tab-1' in layout_str
        assert 'winning-edge-tab-2' in layout_str
        assert 'winning-edge-tab-3' in layout_str
        assert 'winning-edge-tab-4' in layout_str
        assert 'winning-edge-tab-5' in layout_str
        assert 'winning-edge-tab-6' in layout_str
        assert 'winning-edge-tab-7' in layout_str

    def test_layout_graph_ids_unique(self):
        """Test that all graph IDs in layout are unique."""
        from src.dashboard.winning_edge_widget import create_winning_edge_layout

        layout = create_winning_edge_layout()

        # Extract all IDs from layout
        layout_str = str(layout)
        import re
        graph_ids = re.findall(r"id='([^']*graph[^']*)'", layout_str)

        # Check uniqueness
        assert len(graph_ids) == len(set(graph_ids)), "Duplicate graph IDs found"

    def test_layout_contains_required_components(self):
        """Test that layout contains all required components."""
        from src.dashboard.winning_edge_widget import create_winning_edge_layout

        layout = create_winning_edge_layout()
        layout_str = str(layout)

        required_components = [
            'winning-edge-heatmap',
            'winning-edge-spider',
            'winning-edge-correlation',
            'winning-edge-cascade',
            'winning-edge-consistency',
            'winning-edge-action-card',
            'winning-edge-phase',
            'winning-edge-position-gain',
            'winning-edge-overtaking',
            'winning-edge-timeline',
            'winning-edge-curve',
            'winning-edge-visual-guide',
            'winning-edge-brake-guide',
            'winning-edge-summary'
        ]

        for component_id in required_components:
            assert component_id in layout_str, f"Missing component: {component_id}"


class TestWinningEdgeVisualizationFunctions:
    """Test individual visualization functions."""

    def test_time_loss_heatmap(self):
        """Test time loss heatmap visualization."""
        from src.dashboard.winning_edge_widget import create_time_loss_heatmap

        fig = create_time_loss_heatmap({})

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        # Check that the figure has a heatmap trace
        assert any(trace.type == 'heatmap' for trace in fig.data)

    def test_speed_gap_spider(self):
        """Test speed gap spider chart visualization."""
        from src.dashboard.winning_edge_widget import create_speed_gap_spider

        fig = create_speed_gap_spider({})

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_brake_exit_correlation(self):
        """Test brake-exit correlation visualization."""
        from src.dashboard.winning_edge_widget import create_brake_exit_correlation

        fig = create_brake_exit_correlation([], [], [])

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_speed_cascade_waterfall(self):
        """Test speed cascade waterfall chart."""
        from src.dashboard.winning_edge_widget import create_speed_cascade_waterfall

        fig = create_speed_cascade_waterfall('Turn 6', {})

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_consistency_performance_matrix(self):
        """Test consistency vs performance matrix."""
        from src.dashboard.winning_edge_widget import create_consistency_performance_matrix

        fig = create_consistency_performance_matrix([])

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_turn_action_card(self):
        """Test turn action card visualization."""
        from src.dashboard.winning_edge_widget import create_turn_action_card

        fig = create_turn_action_card('Turn 6', {}, {})

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_phase_distribution(self):
        """Test phase distribution chart."""
        from src.dashboard.winning_edge_widget import create_phase_distribution

        fig = create_phase_distribution({})

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_overtaking_opportunity_map(self):
        """Test overtaking opportunity map."""
        from src.dashboard.winning_edge_widget import create_overtaking_opportunity_map

        fig = create_overtaking_opportunity_map()

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_weekly_target_progression(self):
        """Test weekly target progression timeline."""
        from src.dashboard.winning_edge_widget import create_weekly_target_progression

        fig = create_weekly_target_progression()

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_improvement_curve(self):
        """Test improvement curve visualization."""
        from src.dashboard.winning_edge_widget import create_improvement_curve

        fig = create_improvement_curve()

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_turn_visual_guide(self):
        """Test turn visual guide visualization."""
        from src.dashboard.winning_edge_widget import create_turn_visual_guide

        fig = create_turn_visual_guide('Turn 6')

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_brake_pressure_guide(self):
        """Test brake pressure guide visualization."""
        from src.dashboard.winning_edge_widget import create_brake_pressure_guide

        fig = create_brake_pressure_guide('Turn 6')

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_comprehensive_dashboard(self):
        """Test comprehensive dashboard."""
        from src.dashboard.winning_edge_widget import create_comprehensive_dashboard

        fig = create_comprehensive_dashboard({})

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestWinningEdgeCallbacks:
    """Test callback functionality."""

    def test_callbacks_register_without_error(self):
        """Test that callbacks register without errors."""
        from src.dashboard.winning_edge_widget import create_winning_edge_callbacks
        from src.dashboard.app import app

        try:
            create_winning_edge_callbacks(app)
        except Exception as e:
            pytest.fail(f"Callbacks failed to register: {e}")

    def test_position_gain_callback_various_inputs(self):
        """Test position gain callback with various inputs."""
        from src.dashboard.winning_edge_widget import create_position_gain_predictor

        test_cases = [
            (0.44, 1),
            (0.44, 5),
            (0.44, 10),
            (0.44, 20),
            (0.1, 30),
            (1.0, 15),
            (0.0, 10),
        ]

        for time_gain, num_laps in test_cases:
            fig = create_position_gain_predictor(time_gain, num_laps)
            assert isinstance(fig, go.Figure)
            assert fig.data is not None


class TestWinningEdgeIntegration:
    """Integration tests for Winning Edge widget with main app."""

    def test_winning_edge_in_app_layout(self):
        """Test that Winning Edge widget is properly integrated in app layout."""
        from src.dashboard.app import app

        app_layout_str = str(app.layout)

        # Check for Winning Edge specific elements
        assert 'winning-edge' in app_layout_str

    def test_winning_edge_tab_accessibility(self):
        """Test that Winning Edge tab can be accessed."""
        from src.dashboard.winning_edge_widget import create_winning_edge_layout

        # Check that layout can create tabs
        layout = create_winning_edge_layout()
        layout_str = str(layout)
        assert 'winning-edge-tab-1' in layout_str

    def test_app_can_create_layout(self):
        """Test that app can create full layout without errors."""
        try:
            from src.dashboard.app import app
            layout = app.layout
            assert layout is not None
        except Exception as e:
            pytest.fail(f"App layout creation failed: {e}")


class TestEdgeCaseHandling:
    """Test handling of edge cases and error conditions."""

    def test_position_gain_predictor_very_small_num_laps(self):
        """Test with very small lap counts."""
        from src.dashboard.winning_edge_widget import create_position_gain_predictor

        for num_laps in [1]:
            fig = create_position_gain_predictor(0.44, num_laps)
            assert isinstance(fig, go.Figure)
            # Should have at least final lap annotation
            assert len(fig.layout.annotations) >= 1

    def test_position_gain_predictor_large_num_laps(self):
        """Test with large lap counts."""
        from src.dashboard.winning_edge_widget import create_position_gain_predictor

        for num_laps in [50, 100]:
            fig = create_position_gain_predictor(0.44, num_laps)
            assert isinstance(fig, go.Figure)
            assert len(fig.data) >= 2

    def test_position_gain_predictor_negative_time_gain(self):
        """Test with negative time gain (edge case)."""
        from src.dashboard.winning_edge_widget import create_position_gain_predictor

        # Should handle gracefully even with negative values
        fig = create_position_gain_predictor(time_gain_per_lap=-0.1, num_laps=10)
        assert isinstance(fig, go.Figure)

    def test_position_gain_predictor_very_large_time_gain(self):
        """Test with very large time gain."""
        from src.dashboard.winning_edge_widget import create_position_gain_predictor

        fig = create_position_gain_predictor(time_gain_per_lap=5.0, num_laps=10)
        assert isinstance(fig, go.Figure)

    def test_all_visualizations_handle_empty_data(self):
        """Test that all visualizations handle empty data gracefully."""
        from src.dashboard.winning_edge_widget import (
            create_time_loss_heatmap,
            create_speed_gap_spider,
            create_brake_exit_correlation,
            create_speed_cascade_waterfall,
            create_consistency_performance_matrix,
            create_turn_action_card,
            create_phase_distribution
        )

        functions = [
            (create_time_loss_heatmap, [{}]),
            (create_speed_gap_spider, [{}]),
            (create_brake_exit_correlation, [[], [], []]),
            (create_speed_cascade_waterfall, ['Turn 6', {}]),
            (create_consistency_performance_matrix, [[]]),
            (create_turn_action_card, ['Turn 6', {}, {}]),
            (create_phase_distribution, [{}]),
        ]

        for func, args in functions:
            try:
                fig = func(*args)
                assert isinstance(fig, go.Figure)
            except Exception as e:
                pytest.fail(f"{func.__name__} failed with empty data: {e}")


# ============================================================================
# CUSTOM TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_position_gain_data():
    """Provide sample position gain test data."""
    return {
        'time_gain_per_lap': 0.44,
        'num_laps': [1, 5, 10, 20, 30],
        'expected_annotations_min': {
            1: 1,    # Final lap milestone only
            5: 2,    # Midpoint + final lap
            10: 2,   # Lap 10 + final lap
            20: 3,   # Lap 10 + lap 20 + final lap
            30: 3    # Lap 10 + lap 20 + final lap
        }
    }


@pytest.fixture
def position_gain_test_results(sample_position_gain_data):
    """Generate position gain test results."""
    from src.dashboard.winning_edge_widget import create_position_gain_predictor

    results = {}
    for num_laps in sample_position_gain_data['num_laps']:
        fig = create_position_gain_predictor(
            sample_position_gain_data['time_gain_per_lap'],
            num_laps
        )
        results[num_laps] = {
            'figure': fig,
            'annotations_count': len(fig.layout.annotations),
            'traces_count': len(fig.data)
        }
    return results


class TestPositionGainWithFixtures:
    """Test Position Gain Predictor using pytest fixtures."""

    def test_position_gain_annotations_count(self, sample_position_gain_data, position_gain_test_results):
        """Test that correct number of annotations are added."""
        for num_laps, expected_min in sample_position_gain_data['expected_annotations_min'].items():
            actual_count = position_gain_test_results[num_laps]['annotations_count']
            assert actual_count >= expected_min, \
                f"For {num_laps} laps: expected at least {expected_min} annotations, got {actual_count}"

    def test_position_gain_traces_present(self, position_gain_test_results):
        """Test that both time and position traces are present."""
        for num_laps, results in position_gain_test_results.items():
            assert results['traces_count'] >= 2, \
                f"For {num_laps} laps: expected at least 2 traces, got {results['traces_count']}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
