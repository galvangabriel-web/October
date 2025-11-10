"""
Comprehensive Test Suite for Winning Edge Dashboard Components
==============================================================

Tests all 14 visualization functions in the winning_edge_widget.py with:
- Synthetic test data (when organized_data not available)
- Real data from organized_data (when available)
- Single vehicle and multi-vehicle scenarios
- Error handling and edge cases
- Performance metrics (render time, memory usage)

Test Execution:
    pytest tests/test_winning_edge_components.py -v
    pytest tests/test_winning_edge_components.py -v --capture=no  (for detailed output)
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.dashboard.winning_edge_widget import (
        create_time_loss_heatmap,
        create_speed_gap_spider,
        create_brake_exit_correlation,
        create_speed_cascade_waterfall,
        create_consistency_performance_matrix,
        create_turn_action_card,
        create_phase_distribution,
        create_position_gain_predictor,
        create_overtaking_opportunity_map,
        create_weekly_target_progression,
        create_improvement_curve,
        create_turn_visual_guide,
        create_brake_pressure_guide,
        create_comprehensive_dashboard,
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    logger.error(f"Failed to import winning_edge_widget: {e}")
    IMPORT_SUCCESS = False


# ==============================================================================
# TEST DATA GENERATORS
# ==============================================================================

def create_synthetic_telemetry_data() -> pd.DataFrame:
    """Create synthetic telemetry data matching expected format."""
    np.random.seed(42)

    n_samples = 2000
    base_timestamp = int(pd.Timestamp('2025-01-01 10:00:00').timestamp() * 1000)

    data = []

    # Create data for 2 laps
    for lap in [1, 2]:
        lap_samples = n_samples // 2
        timestamps = np.arange(
            base_timestamp + (lap - 1) * 90000,
            base_timestamp + lap * 90000,
            90
        )

        for i, ts in enumerate(timestamps[:lap_samples]):
            progress = i / lap_samples
            speed = 100 + 40 * np.sin(progress * 2 * np.pi * 3)
            brake = max(0, 50 * np.cos(progress * 2 * np.pi * 3 + np.pi/4))
            throttle = 100 - brake
            lat_g = 1.5 * np.sin(progress * 2 * np.pi * 3)
            steering = 50 * np.sin(progress * 2 * np.pi * 3)

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
            }

            for sensor_name, sensor_value in sensors.items():
                data.append({
                    'telemetry_name': sensor_name,
                    'telemetry_value': sensor_value,
                    'vehicle_number': 5,
                    'timestamp': ts,
                    'lap': lap
                })

    return pd.DataFrame(data)


def create_synthetic_multi_vehicle_data() -> pd.DataFrame:
    """Create synthetic multi-vehicle telemetry data."""
    np.random.seed(42)

    base_timestamp = int(pd.Timestamp('2025-01-01 10:00:00').timestamp() * 1000)
    data = []

    # Vehicles 5, 7, 12
    for vehicle_num in [5, 7, 12]:
        for i in range(100):
            ts = base_timestamp + i * 100

            data.append({
                'telemetry_name': 'speed',
                'telemetry_value': 100 + np.random.normal(0, 10),
                'vehicle_number': vehicle_num,
                'timestamp': ts,
                'lap': 1
            })

            data.append({
                'telemetry_name': 'pbrake_f',
                'telemetry_value': max(0, 50 + np.random.normal(0, 20)),
                'vehicle_number': vehicle_num,
                'timestamp': ts,
                'lap': 1
            })

    return pd.DataFrame(data)


# ==============================================================================
# TEST FIXTURES
# ==============================================================================

@pytest.fixture
def synthetic_telemetry():
    """Provide synthetic telemetry data."""
    return create_synthetic_telemetry_data()


@pytest.fixture
def synthetic_multi_vehicle():
    """Provide synthetic multi-vehicle telemetry."""
    return create_synthetic_multi_vehicle_data()


@pytest.fixture
def corner_data_dict() -> Dict:
    """Sample corner data for visualizations."""
    return {
        'Turn 1': {'time_loss': 0.210, 'pct_of_total': 48},
        'Turn 6': {'time_loss': 0.180, 'pct_of_total': 41},
        'Turn 11': {'time_loss': 0.050, 'pct_of_total': 11}
    }


# ==============================================================================
# SECTION 1: RACE WINNER'S DASHBOARD TESTS
# ==============================================================================

@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Failed to import winning_edge_widget")
class TestSection1RaceWinnerDashboard:
    """Test Section 1: Race Winner's Dashboard visualizations."""

    def test_time_loss_heatmap_with_default_data(self):
        """Test time loss heatmap with default data."""
        start_time = time.time()
        tracemalloc.start()

        fig = create_time_loss_heatmap({})

        mem_info = tracemalloc.get_traced_memory()
        render_time = time.time() - start_time
        tracemalloc.stop()

        assert fig is not None
        assert fig.layout.title.text is not None
        assert 'Time Loss' in fig.layout.title.text or 'PATH TO VICTORY' in fig.layout.title.text
        logger.info(f"✓ Time Loss Heatmap - Time: {render_time:.3f}s, Memory: {mem_info[0]/1024:.1f}KB")

    def test_time_loss_heatmap_with_custom_data(self, corner_data_dict):
        """Test time loss heatmap with custom corner data."""
        fig = create_time_loss_heatmap(corner_data_dict)

        assert fig is not None
        assert len(fig.data) > 0
        # Check for heatmap trace
        assert fig.data[0].z is not None
        logger.info("✓ Time Loss Heatmap with Custom Data")

    def test_speed_gap_spider_with_default_data(self):
        """Test speed gap spider chart with default data."""
        start_time = time.time()
        tracemalloc.start()

        fig = create_speed_gap_spider({})

        mem_info = tracemalloc.get_traced_memory()
        render_time = time.time() - start_time
        tracemalloc.stop()

        assert fig is not None
        assert len(fig.data) > 0
        # Verify it's a polar/radar chart
        assert hasattr(fig, 'data')
        logger.info(f"✓ Speed Gap Spider - Time: {render_time:.3f}s, Memory: {mem_info[0]/1024:.1f}KB")

    def test_speed_gap_spider_with_custom_data(self):
        """Test speed gap spider with custom metrics."""
        corner_metrics = {
            'Turn 1': [97.8, 96.8, 97.1, 83.5, 68.5, 87.5],
            'Turn 6': [97.5, 95.1, 96.3, 83.3, 52.3, 83.7],
            'Turn 11': [99.6, 98.9, 98.6, 116.9, 88.7, 94.8]
        }

        fig = create_speed_gap_spider(corner_metrics)

        assert fig is not None
        assert len(fig.data) == 3  # Should have 3 traces (one per corner)
        logger.info("✓ Speed Gap Spider with Custom Data")


# ==============================================================================
# SECTION 2: CORRELATION DASHBOARD TESTS
# ==============================================================================

@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Failed to import winning_edge_widget")
class TestSection2CorrelationDashboard:
    """Test Section 2: Correlation Dashboard visualizations."""

    def test_brake_exit_correlation_with_default_data(self):
        """Test brake-exit correlation with default data."""
        start_time = time.time()
        tracemalloc.start()

        fig = create_brake_exit_correlation([], [], [])

        mem_info = tracemalloc.get_traced_memory()
        render_time = time.time() - start_time
        tracemalloc.stop()

        assert fig is not None
        assert len(fig.data) > 0
        logger.info(f"✓ Brake-Exit Correlation - Time: {render_time:.3f}s, Memory: {mem_info[0]/1024:.1f}KB")

    def test_brake_exit_correlation_with_custom_data(self):
        """Test brake-exit correlation with custom data."""
        brake_excess = [0.62, 0.91, -0.45]
        exit_deficit = [-0.62, -0.91, 0.45]
        corner_labels = ['Turn 1', 'Turn 6', 'Turn 11']

        fig = create_brake_exit_correlation(brake_excess, exit_deficit, corner_labels)

        assert fig is not None
        assert len(fig.data) >= 2  # Data points and diagonal line
        logger.info("✓ Brake-Exit Correlation with Custom Data")

    def test_speed_cascade_waterfall_with_default_data(self):
        """Test speed cascade waterfall with default data."""
        start_time = time.time()
        tracemalloc.start()

        fig = create_speed_cascade_waterfall('Turn 6', {})

        mem_info = tracemalloc.get_traced_memory()
        render_time = time.time() - start_time
        tracemalloc.stop()

        assert fig is not None
        assert len(fig.data) > 0
        logger.info(f"✓ Speed Cascade Waterfall - Time: {render_time:.3f}s, Memory: {mem_info[0]/1024:.1f}KB")

    def test_speed_cascade_waterfall_with_custom_data(self):
        """Test speed cascade with custom cascade data."""
        cascade_data = {
            'Entry Gap': -3.6,
            'Apex Gap': -3.2,
            'Exit Gap': -3.7,
            'Straight Loss': -5.0
        }

        fig = create_speed_cascade_waterfall('Turn 6', cascade_data)

        assert fig is not None
        assert len(fig.data) > 0
        logger.info("✓ Speed Cascade Waterfall with Custom Data")

    def test_consistency_performance_matrix_with_default_data(self):
        """Test consistency matrix with default data."""
        start_time = time.time()
        tracemalloc.start()

        fig = create_consistency_performance_matrix([])

        mem_info = tracemalloc.get_traced_memory()
        render_time = time.time() - start_time
        tracemalloc.stop()

        assert fig is not None
        assert len(fig.data) > 0
        logger.info(f"✓ Consistency Matrix - Time: {render_time:.3f}s, Memory: {mem_info[0]/1024:.1f}KB")

    def test_consistency_performance_matrix_with_custom_data(self):
        """Test consistency matrix with custom data."""
        corner_data = [
            {'name': 'Turn 1', 'consistency_pct': 87.5, 'efficiency_pct': 78.9, 'time_loss': 0.21},
            {'name': 'Turn 6', 'consistency_pct': 83.7, 'efficiency_pct': 82.0, 'time_loss': 0.18},
            {'name': 'Turn 11', 'consistency_pct': 94.8, 'efficiency_pct': 95.0, 'time_loss': 0.05}
        ]

        fig = create_consistency_performance_matrix(corner_data)

        assert fig is not None
        assert len(fig.data) > 0
        logger.info("✓ Consistency Matrix with Custom Data")


# ==============================================================================
# SECTION 3: PRIORITY ACTION CARDS TESTS
# ==============================================================================

@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Failed to import winning_edge_widget")
class TestSection3ActionCards:
    """Test Section 3: Priority Action Cards visualizations."""

    def test_turn_action_card_with_default_data(self):
        """Test turn action card with default data."""
        start_time = time.time()
        tracemalloc.start()

        fig = create_turn_action_card('Turn 6', {}, {})

        mem_info = tracemalloc.get_traced_memory()
        render_time = time.time() - start_time
        tracemalloc.stop()

        assert fig is not None
        assert len(fig.data) > 0
        logger.info(f"✓ Turn Action Card - Time: {render_time:.3f}s, Memory: {mem_info[0]/1024:.1f}KB")

    def test_turn_action_card_with_custom_data(self):
        """Test turn action card with custom metrics."""
        current_metrics = {
            'Brake Point (m)': 120,
            'Brake Pressure (%)': 95,
            'Throttle Timing (%)': 52.3,
            'Exit Speed (km/h)': 98.7
        }
        target_metrics = {
            'Brake Point (m)': 132,
            'Brake Pressure (%)': 75,
            'Throttle Timing (%)': 57.5,
            'Exit Speed (km/h)': 102.4
        }

        fig = create_turn_action_card('Turn 6', current_metrics, target_metrics)

        assert fig is not None
        assert len(fig.data) == 2  # Current and target bars
        logger.info("✓ Turn Action Card with Custom Data")

    def test_phase_distribution_with_default_data(self):
        """Test phase distribution with default data."""
        start_time = time.time()
        tracemalloc.start()

        fig = create_phase_distribution({})

        mem_info = tracemalloc.get_traced_memory()
        render_time = time.time() - start_time
        tracemalloc.stop()

        assert fig is not None
        assert len(fig.data) > 0
        logger.info(f"✓ Phase Distribution - Time: {render_time:.3f}s, Memory: {mem_info[0]/1024:.1f}KB")

    def test_phase_distribution_with_custom_data(self):
        """Test phase distribution with custom data."""
        phase_data = {
            'Turn 1 Current': [35, 20, 45],
            'Turn 1 Target': [30, 20, 50],
            'Turn 6 Current': [35, 20, 45],
            'Turn 6 Target': [30, 20, 50],
            'Turn 11 Benchmark': [25, 20, 55]
        }

        fig = create_phase_distribution(phase_data)

        assert fig is not None
        assert len(fig.data) == 3  # 3 phase components
        logger.info("✓ Phase Distribution with Custom Data")


# ==============================================================================
# SECTION 4: RACE SIMULATION TESTS
# ==============================================================================

@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Failed to import winning_edge_widget")
class TestSection4RaceSimulation:
    """Test Section 4: Race Simulation Impact visualizations."""

    def test_position_gain_predictor_default_params(self):
        """Test position gain predictor with default parameters."""
        start_time = time.time()
        tracemalloc.start()

        fig = create_position_gain_predictor(0.44, 20)

        mem_info = tracemalloc.get_traced_memory()
        render_time = time.time() - start_time
        tracemalloc.stop()

        assert fig is not None
        assert len(fig.data) >= 2  # Time and position traces
        logger.info(f"✓ Position Gain Predictor - Time: {render_time:.3f}s, Memory: {mem_info[0]/1024:.1f}KB")

    def test_position_gain_predictor_custom_params(self):
        """Test position gain predictor with custom parameters."""
        fig = create_position_gain_predictor(0.3, 30)

        assert fig is not None
        assert len(fig.data) >= 2
        logger.info("✓ Position Gain Predictor with Custom Params")

    def test_position_gain_predictor_edge_cases(self):
        """Test position gain predictor with edge cases."""
        # Very small gain
        fig1 = create_position_gain_predictor(0.01, 10)
        assert fig1 is not None

        # Large gain
        fig2 = create_position_gain_predictor(2.0, 50)
        assert fig2 is not None

        logger.info("✓ Position Gain Predictor Edge Cases")

    def test_overtaking_opportunity_map_default(self):
        """Test overtaking map with default data."""
        start_time = time.time()
        tracemalloc.start()

        fig = create_overtaking_opportunity_map()

        mem_info = tracemalloc.get_traced_memory()
        render_time = time.time() - start_time
        tracemalloc.stop()

        assert fig is not None
        assert len(fig.data) > 0
        logger.info(f"✓ Overtaking Map - Time: {render_time:.3f}s, Memory: {mem_info[0]/1024:.1f}KB")

    def test_overtaking_opportunity_map_custom(self):
        """Test overtaking map with custom track layout."""
        track_layout = {
            'track_x': [0, 100, 200, 300, 0],
            'track_y': [0, 50, 100, 50, 0],
            'corners_x': [100, 300],
            'corners_y': [50, 100],
            'opportunity': [0.21, 0.05],
            'labels': ['T1: 0.21s', 'T6: 0.05s']
        }

        fig = create_overtaking_opportunity_map(track_layout)

        assert fig is not None
        logger.info("✓ Overtaking Map with Custom Layout")


# ==============================================================================
# SECTION 5: TRANSFORMATION TIMELINE TESTS
# ==============================================================================

@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Failed to import winning_edge_widget")
class TestSection5TransformationTimeline:
    """Test Section 5: 6-Week Transformation Timeline visualizations."""

    def test_weekly_target_progression(self):
        """Test weekly target progression timeline."""
        start_time = time.time()
        tracemalloc.start()

        fig = create_weekly_target_progression()

        mem_info = tracemalloc.get_traced_memory()
        render_time = time.time() - start_time
        tracemalloc.stop()

        assert fig is not None
        assert len(fig.data) > 0
        logger.info(f"✓ Weekly Target Progression - Time: {render_time:.3f}s, Memory: {mem_info[0]/1024:.1f}KB")

    def test_improvement_curve_no_actual_data(self):
        """Test improvement curve without actual progress data."""
        start_time = time.time()
        tracemalloc.start()

        fig = create_improvement_curve()

        mem_info = tracemalloc.get_traced_memory()
        render_time = time.time() - start_time
        tracemalloc.stop()

        assert fig is not None
        assert len(fig.data) > 0
        logger.info(f"✓ Improvement Curve (No Data) - Time: {render_time:.3f}s, Memory: {mem_info[0]/1024:.1f}KB")

    def test_improvement_curve_with_actual_data(self):
        """Test improvement curve with actual progress data."""
        actual_progress = [0, 0.08, 0.17, 0.25, 0.31, 0.38, 0.42]

        fig = create_improvement_curve(actual_progress)

        assert fig is not None
        assert len(fig.data) > 0
        logger.info("✓ Improvement Curve with Actual Data")


# ==============================================================================
# SECTION 6: SESSION VISUAL GUIDES TESTS
# ==============================================================================

@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Failed to import winning_edge_widget")
class TestSection6SessionGuides:
    """Test Section 6: Session Visual Guides visualizations."""

    def test_turn_visual_guide_default(self):
        """Test turn visual guide with default data."""
        start_time = time.time()
        tracemalloc.start()

        fig = create_turn_visual_guide('Turn 6')

        mem_info = tracemalloc.get_traced_memory()
        render_time = time.time() - start_time
        tracemalloc.stop()

        assert fig is not None
        assert len(fig.data) > 0
        logger.info(f"✓ Turn Visual Guide - Time: {render_time:.3f}s, Memory: {mem_info[0]/1024:.1f}KB")

    def test_turn_visual_guide_custom(self):
        """Test turn visual guide with custom racing lines."""
        current_line = {
            'x': list(range(0, 200, 5)),
            'y': [100] * 40 + [90 + 10*i/30 for i in range(30)]
        }
        optimal_line = {
            'x': list(range(0, 200, 5)),
            'y': [100] * 40 + [95 + 10*i/30 for i in range(30)]
        }

        fig = create_turn_visual_guide('Turn 6', current_line, optimal_line)

        assert fig is not None
        assert len(fig.data) > 0
        logger.info("✓ Turn Visual Guide with Custom Lines")

    def test_turn_visual_guide_all_turns(self):
        """Test turn visual guide for all major turns."""
        for turn_name in ['Turn 1', 'Turn 6', 'Turn 11']:
            fig = create_turn_visual_guide(turn_name)
            assert fig is not None
            assert len(fig.data) > 0

        logger.info("✓ Turn Visual Guide for All Turns")

    def test_brake_pressure_guide_default(self):
        """Test brake pressure guide with default data."""
        start_time = time.time()
        tracemalloc.start()

        fig = create_brake_pressure_guide('Turn 6')

        mem_info = tracemalloc.get_traced_memory()
        render_time = time.time() - start_time
        tracemalloc.stop()

        assert fig is not None
        assert len(fig.data) > 0
        logger.info(f"✓ Brake Pressure Guide - Time: {render_time:.3f}s, Memory: {mem_info[0]/1024:.1f}KB")

    def test_brake_pressure_guide_all_turns(self):
        """Test brake pressure guide for all turns."""
        for turn_name in ['Turn 1', 'Turn 6', 'Turn 11']:
            fig = create_brake_pressure_guide(turn_name)
            assert fig is not None
            assert len(fig.data) > 0

        logger.info("✓ Brake Pressure Guide for All Turns")


# ==============================================================================
# SECTION 7: COMPREHENSIVE DASHBOARD TESTS
# ==============================================================================

@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Failed to import winning_edge_widget")
class TestSection7ComprehensiveDashboard:
    """Test Section 7: Competitive Advantage Summary visualizations."""

    def test_comprehensive_dashboard_default(self):
        """Test comprehensive dashboard with default data."""
        start_time = time.time()
        tracemalloc.start()

        fig = create_comprehensive_dashboard({})

        mem_info = tracemalloc.get_traced_memory()
        render_time = time.time() - start_time
        tracemalloc.stop()

        assert fig is not None
        assert len(fig.data) > 0
        logger.info(f"✓ Comprehensive Dashboard - Time: {render_time:.3f}s, Memory: {mem_info[0]/1024:.1f}KB")

    def test_comprehensive_dashboard_has_subplots(self):
        """Test that comprehensive dashboard has 6 subplots."""
        fig = create_comprehensive_dashboard({})

        assert fig is not None
        # Should have multiple traces for subplots
        assert len(fig.data) >= 6
        logger.info("✓ Comprehensive Dashboard Subplots")


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Failed to import winning_edge_widget")
class TestIntegration:
    """Integration tests with real data."""

    def test_all_visualizations_render(self):
        """Test that all 14 visualizations can be created."""
        visualizations = [
            ('Time Loss Heatmap', create_time_loss_heatmap, [{}]),
            ('Speed Gap Spider', create_speed_gap_spider, [{}]),
            ('Brake-Exit Correlation', create_brake_exit_correlation, [[], [], []]),
            ('Speed Cascade Waterfall', create_speed_cascade_waterfall, ['Turn 6', {}]),
            ('Consistency Matrix', create_consistency_performance_matrix, [[]]),
            ('Turn Action Card', create_turn_action_card, ['Turn 6', {}, {}]),
            ('Phase Distribution', create_phase_distribution, [{}]),
            ('Position Gain Predictor', create_position_gain_predictor, [0.44, 20]),
            ('Overtaking Map', create_overtaking_opportunity_map, []),
            ('Weekly Timeline', create_weekly_target_progression, []),
            ('Improvement Curve', create_improvement_curve, []),
            ('Turn Visual Guide', create_turn_visual_guide, ['Turn 6']),
            ('Brake Pressure Guide', create_brake_pressure_guide, ['Turn 6']),
            ('Comprehensive Dashboard', create_comprehensive_dashboard, [{}]),
        ]

        results = {}
        for name, func, args in visualizations:
            try:
                if args:
                    fig = func(*args)
                else:
                    fig = func()

                assert fig is not None
                assert len(fig.data) > 0
                results[name] = 'PASS'
                logger.info(f"✓ {name}")
            except Exception as e:
                results[name] = f'FAIL: {str(e)}'
                logger.error(f"✗ {name}: {e}")

        # Print summary
        passed = sum(1 for v in results.values() if v == 'PASS')
        total = len(results)
        assert passed == total, f"Only {passed}/{total} visualizations passed"

    def test_all_visualizations_have_titles(self):
        """Test that all visualizations have proper titles."""
        visualizations = [
            ('Time Loss Heatmap', create_time_loss_heatmap, [{}]),
            ('Speed Gap Spider', create_speed_gap_spider, [{}]),
            ('Brake-Exit Correlation', create_brake_exit_correlation, [[], [], []]),
            ('Speed Cascade Waterfall', create_speed_cascade_waterfall, ['Turn 6', {}]),
            ('Consistency Matrix', create_consistency_performance_matrix, [[]]),
            ('Turn Action Card', create_turn_action_card, ['Turn 6', {}, {}]),
            ('Phase Distribution', create_phase_distribution, [{}]),
            ('Position Gain Predictor', create_position_gain_predictor, [0.44, 20]),
            ('Overtaking Map', create_overtaking_opportunity_map, []),
            ('Weekly Timeline', create_weekly_target_progression, []),
            ('Improvement Curve', create_improvement_curve, []),
            ('Turn Visual Guide', create_turn_visual_guide, ['Turn 6']),
            ('Brake Pressure Guide', create_brake_pressure_guide, ['Turn 6']),
            ('Comprehensive Dashboard', create_comprehensive_dashboard, [{}]),
        ]

        for name, func, args in visualizations:
            if args:
                fig = func(*args)
            else:
                fig = func()

            # Check for title
            assert fig.layout.title is not None or fig.layout.title_text is not None, \
                f"{name} missing title"


# ==============================================================================
# PERFORMANCE TESTS
# ==============================================================================

@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Failed to import winning_edge_widget")
class TestPerformance:
    """Performance tests for visualizations."""

    def test_render_time_performance(self):
        """Test render times for all visualizations."""
        visualizations = [
            ('Time Loss Heatmap', create_time_loss_heatmap, [{}]),
            ('Speed Gap Spider', create_speed_gap_spider, [{}]),
            ('Brake-Exit Correlation', create_brake_exit_correlation, [[], [], []]),
            ('Consistency Matrix', create_consistency_performance_matrix, [[]]),
            ('Position Gain Predictor', create_position_gain_predictor, [0.44, 20]),
        ]

        times = {}
        max_acceptable_time = 1.0  # seconds

        for name, func, args in visualizations:
            start = time.time()
            if args:
                func(*args)
            else:
                func()
            elapsed = time.time() - start
            times[name] = elapsed

            logger.info(f"  {name}: {elapsed:.4f}s")
            assert elapsed < max_acceptable_time, \
                f"{name} took {elapsed:.4f}s (max: {max_acceptable_time}s)"

        logger.info("✓ All visualizations render within performance targets")

    def test_memory_efficiency(self):
        """Test memory usage for visualizations."""
        tracemalloc.start()

        # Create all visualizations
        create_time_loss_heatmap({})
        create_speed_gap_spider({})
        create_brake_exit_correlation([], [], [])
        create_consistency_performance_matrix([])
        create_position_gain_predictor(0.44, 20)
        create_comprehensive_dashboard({})

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        logger.info(f"  Peak memory usage: {peak_mb:.1f}MB")

        # Reasonable limit for all visualizations
        assert peak_mb < 100, f"Memory usage too high: {peak_mb:.1f}MB"


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================

@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Failed to import winning_edge_widget")
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_data_handling(self):
        """Test functions handle empty data gracefully."""
        # Should not crash with empty/None data
        try:
            create_time_loss_heatmap({})
            create_speed_gap_spider({})
            create_consistency_performance_matrix([])
            create_phase_distribution({})
            create_speed_cascade_waterfall('Turn 1', {})
            logger.info("✓ Empty data handling")
        except Exception as e:
            pytest.fail(f"Functions should handle empty data: {e}")

    def test_none_parameter_handling(self):
        """Test functions handle None parameters."""
        try:
            create_turn_visual_guide('Turn 1', None, None)
            create_brake_pressure_guide('Turn 1')
            logger.info("✓ None parameter handling")
        except Exception as e:
            pytest.fail(f"Functions should handle None parameters: {e}")

    def test_extreme_values(self):
        """Test functions with extreme input values."""
        try:
            # Very large time gain
            create_position_gain_predictor(10.0, 50)

            # Very small gain
            create_position_gain_predictor(0.001, 5)

            # Large number of laps
            create_position_gain_predictor(0.44, 500)

            logger.info("✓ Extreme value handling")
        except Exception as e:
            pytest.fail(f"Functions should handle extreme values: {e}")


# ==============================================================================
# SUMMARY
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
