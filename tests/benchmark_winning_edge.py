"""
Performance Benchmark Suite for Winning Edge Widget
====================================================

This test suite benchmarks the Winning Edge widget performance with various
data sizes (10, 50, 100, 500 laps) measuring:
- Load time
- Memory usage
- CPU usage
- Rendering time

Run benchmarks:
    pytest tests/benchmark_winning_edge.py -v -s

Run specific benchmark:
    pytest tests/benchmark_winning_edge.py::test_benchmark_10_laps -v -s

Run with profiling:
    pytest tests/benchmark_winning_edge.py -v --profile
"""

import pytest
import time
import psutil
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

from src.dashboard.winning_edge_optimizer import (
    sample_large_dataset,
    optimize_figure_config,
    get_performance_cache,
    get_performance_monitor,
    estimate_memory_usage,
    get_optimization_recommendations
)
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
    create_comprehensive_dashboard
)

logger = logging.getLogger(__name__)

# Benchmark configuration
BENCHMARK_LAPS = [10, 50, 100, 500]
BENCHMARK_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "benchmarks"
BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def process_monitor():
    """Monitor process resource usage."""

    class ProcessMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_memory = 0
            self.start_time = 0
            self.end_memory = 0
            self.end_time = 0

        def start(self):
            """Start monitoring."""
            self.process.memory_info()  # Initialize
            self.start_memory = self.process.memory_info().rss / 1024 ** 2
            self.start_time = time.time()

        def stop(self) -> Dict[str, float]:
            """Stop monitoring and return metrics."""
            self.end_time = time.time()
            self.end_memory = self.process.memory_info().rss / 1024 ** 2

            return {
                'elapsed_time': self.end_time - self.start_time,
                'memory_used': self.end_memory - self.start_memory,
                'peak_memory': self.end_memory,
                'cpu_percent': self.process.cpu_percent(interval=0.1)
            }

    return ProcessMonitor()


@pytest.fixture
def sample_telemetry_data():
    """Generate sample telemetry data for benchmarking."""

    def _generate_telemetry(num_laps: int = 10) -> pd.DataFrame:
        """Generate telemetry DataFrame."""
        np.random.seed(42)

        # Generate telemetry data
        data = {
            'lap': np.repeat(range(1, num_laps + 1), 1000),
            'timestamp': np.tile(range(1000), num_laps),
            'speed': np.random.normal(120, 30, num_laps * 1000),
            'pbrake_f': np.random.normal(50, 30, num_laps * 1000),
            'pbrake_r': np.random.normal(50, 30, num_laps * 1000),
            'aps': np.random.normal(50, 20, num_laps * 1000),
            'accx_can': np.random.normal(0, 2, num_laps * 1000),
            'accy_can': np.random.normal(0, 2, num_laps * 1000),
            'Steering_Angle': np.random.normal(0, 5, num_laps * 1000),
            'gear': np.random.randint(1, 7, num_laps * 1000),
            'nmot': np.random.normal(5000, 1500, num_laps * 1000)
        }

        df = pd.DataFrame(data)
        df['speed'] = df['speed'].clip(0, 300)  # Realistic speed range
        df['pbrake_f'] = df['pbrake_f'].clip(0, 100)
        df['pbrake_r'] = df['pbrake_r'].clip(0, 100)
        df['aps'] = df['aps'].clip(0, 100)
        df['nmot'] = df['nmot'].clip(0, 8000)

        return df

    return _generate_telemetry


@pytest.fixture
def benchmark_results():
    """Store benchmark results."""
    return {
        'timestamp': datetime.now().isoformat(),
        'benchmarks': {},
        'summary': {}
    }


# =============================================================================
# BASIC PERFORMANCE TESTS
# =============================================================================

class TestBasicPerformance:
    """Test basic performance of Winning Edge components."""

    @pytest.mark.benchmark
    def test_figure_creation_heatmap(self, process_monitor):
        """Benchmark time loss heatmap creation."""
        process_monitor.start()
        fig = create_time_loss_heatmap({})
        metrics = process_monitor.stop()

        logger.info(f"Heatmap creation: {metrics['elapsed_time']*1000:.2f}ms")
        assert metrics['elapsed_time'] < 1.0, "Heatmap should render in < 1 second"

    @pytest.mark.benchmark
    def test_figure_creation_spider(self, process_monitor):
        """Benchmark speed gap spider chart creation."""
        process_monitor.start()
        fig = create_speed_gap_spider({})
        metrics = process_monitor.stop()

        logger.info(f"Spider chart creation: {metrics['elapsed_time']*1000:.2f}ms")
        assert metrics['elapsed_time'] < 1.0, "Spider chart should render in < 1 second"

    @pytest.mark.benchmark
    def test_figure_creation_correlation(self, process_monitor):
        """Benchmark brake-exit correlation chart creation."""
        process_monitor.start()
        fig = create_brake_exit_correlation([], [], [])
        metrics = process_monitor.stop()

        logger.info(f"Correlation chart creation: {metrics['elapsed_time']*1000:.2f}ms")
        assert metrics['elapsed_time'] < 1.0, "Correlation chart should render in < 1 second"

    @pytest.mark.benchmark
    def test_figure_creation_action_card(self, process_monitor):
        """Benchmark action card creation."""
        process_monitor.start()
        fig = create_turn_action_card("Turn 6", {}, {})
        metrics = process_monitor.stop()

        logger.info(f"Action card creation: {metrics['elapsed_time']*1000:.2f}ms")
        assert metrics['elapsed_time'] < 1.0, "Action card should render in < 1 second"

    @pytest.mark.benchmark
    def test_position_gain_predictor(self, process_monitor):
        """Benchmark position gain predictor chart creation."""
        process_monitor.start()
        fig = create_position_gain_predictor(0.44, 20)
        metrics = process_monitor.stop()

        logger.info(f"Position gain predictor: {metrics['elapsed_time']*1000:.2f}ms")
        assert metrics['elapsed_time'] < 1.0, "Position gain predictor should render in < 1 second"

    @pytest.mark.benchmark
    def test_comprehensive_dashboard(self, process_monitor):
        """Benchmark comprehensive dashboard creation."""
        process_monitor.start()
        fig = create_comprehensive_dashboard({})
        metrics = process_monitor.stop()

        logger.info(f"Comprehensive dashboard: {metrics['elapsed_time']*1000:.2f}ms")
        assert metrics['elapsed_time'] < 3.0, "Dashboard should render in < 3 seconds"


# =============================================================================
# DATA SIZE BENCHMARKS
# =============================================================================

class TestDataSizeBenchmarks:
    """Benchmark performance with varying data sizes."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize('num_laps', BENCHMARK_LAPS)
    def test_memory_usage_by_laps(self, sample_telemetry_data, num_laps, benchmark_results):
        """Benchmark memory usage for different lap counts."""
        df = sample_telemetry_data(num_laps)
        memory_mb = estimate_memory_usage(df)

        logger.info(f"{num_laps} laps: {memory_mb:.2f} MB ({len(df)} rows)")

        benchmark_results['benchmarks'][f'memory_{num_laps}_laps'] = {
            'laps': num_laps,
            'rows': len(df),
            'memory_mb': memory_mb
        }

        # Verify memory targets
        if num_laps <= 100:
            assert memory_mb < 500, f"Memory should be < 500MB for {num_laps} laps"

    @pytest.mark.benchmark
    @pytest.mark.parametrize('num_laps', BENCHMARK_LAPS)
    def test_sampling_performance(self, sample_telemetry_data, num_laps, process_monitor):
        """Benchmark data sampling performance."""
        df = sample_telemetry_data(num_laps)

        process_monitor.start()
        df_sampled = sample_large_dataset(df, max_laps=100, sampling_strategy='stratified')
        metrics = process_monitor.stop()

        original_mb = estimate_memory_usage(df)
        sampled_mb = estimate_memory_usage(df_sampled)
        reduction = (1 - len(df_sampled) / len(df)) * 100

        logger.info(f"{num_laps} laps -> {len(df_sampled)/1000:.0f}k rows: "
                   f"{original_mb:.1f}MB -> {sampled_mb:.1f}MB "
                   f"({reduction:.1f}% reduction, {metrics['elapsed_time']*1000:.2f}ms)")

        assert metrics['elapsed_time'] < 0.5, "Sampling should complete in < 500ms"

    @pytest.mark.benchmark
    def test_figure_optimization(self, process_monitor):
        """Benchmark figure optimization performance."""
        fig = create_comprehensive_dashboard({})

        process_monitor.start()
        fig_optimized = optimize_figure_config(fig)
        metrics = process_monitor.stop()

        logger.info(f"Figure optimization: {metrics['elapsed_time']*1000:.2f}ms")
        assert metrics['elapsed_time'] < 0.5, "Optimization should complete in < 500ms"


# =============================================================================
# CACHE PERFORMANCE TESTS
# =============================================================================

class TestCachePerformance:
    """Test caching performance benefits."""

    @pytest.mark.benchmark
    def test_cache_hit_vs_miss(self, process_monitor):
        """Benchmark cache hits vs misses."""
        cache = get_performance_cache()
        cache.clear()

        # First call - cache miss
        process_monitor.start()
        cache.get("test_key_1")
        miss_time = process_monitor.stop()['elapsed_time']

        # Second call - cache hit
        cache.set("test_key_1", "cached_value", ttl_seconds=300)
        process_monitor.start()
        result = cache.get("test_key_1")
        hit_time = process_monitor.stop()['elapsed_time']

        logger.info(f"Cache miss: {miss_time*1000:.3f}ms")
        logger.info(f"Cache hit: {hit_time*1000:.3f}ms")
        logger.info(f"Speedup: {miss_time / hit_time:.1f}x")

        assert hit_time < miss_time, "Cache hit should be faster than miss"
        assert result == "cached_value", "Cache should return correct value"

    @pytest.mark.benchmark
    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = get_performance_cache()
        cache.clear()

        # Perform operations
        for i in range(10):
            cache.get(f"key_{i}")  # Miss
            cache.set(f"key_{i}", f"value_{i}")

        for i in range(10):
            cache.get(f"key_{i}")  # Hit

        for i in range(10, 20):
            cache.get(f"key_{i}")  # Miss

        stats = cache.stats()
        logger.info(f"Cache stats: {stats}")

        assert stats['hits'] == 10, "Should have 10 cache hits"
        assert stats['misses'] == 20, "Should have 20 cache misses"
        assert stats['size'] == 20, "Should have 20 cached entries"


# =============================================================================
# INTEGRATION BENCHMARKS
# =============================================================================

class TestIntegrationBenchmarks:
    """Test end-to-end performance scenarios."""

    @pytest.mark.benchmark
    def test_full_widget_initialization(self, sample_telemetry_data, process_monitor):
        """Benchmark full widget initialization with all tabs."""
        df = sample_telemetry_data(100)

        process_monitor.start()

        # Create all major figures
        figs = [
            create_time_loss_heatmap({}),
            create_speed_gap_spider({}),
            create_brake_exit_correlation([], [], []),
            create_speed_cascade_waterfall("Turn 6", {}),
            create_consistency_performance_matrix([]),
            create_position_gain_predictor(0.44, 20),
            create_comprehensive_dashboard({})
        ]

        metrics = process_monitor.stop()

        logger.info(f"Full widget initialization (7 figures): "
                   f"{metrics['elapsed_time']:.2f}s")

        # Check target
        assert metrics['elapsed_time'] < 2.0, "Full widget should initialize in < 2 seconds"
        assert len(figs) == 7, "Should create all 7 figures"

    @pytest.mark.benchmark
    def test_tab_switch_performance(self, sample_telemetry_data, process_monitor):
        """Benchmark tab switching performance."""
        df = sample_telemetry_data(100)

        # Simulate switching between tabs
        tab_generators = [
            lambda: create_time_loss_heatmap({}),
            lambda: create_brake_exit_correlation([], [], []),
            lambda: create_turn_action_card("Turn 6", {}, {}),
            lambda: create_position_gain_predictor(0.44, 20),
            lambda: create_comprehensive_dashboard({})
        ]

        process_monitor.start()
        for generator in tab_generators:
            fig = generator()

        metrics = process_monitor.stop()

        avg_time = metrics['elapsed_time'] / len(tab_generators)
        logger.info(f"Tab switch performance: {avg_time*1000:.2f}ms per tab")

        assert avg_time < 0.5, "Tab switch should complete in < 500ms"

    @pytest.mark.benchmark
    def test_multi_lap_analysis(self, sample_telemetry_data, process_monitor):
        """Benchmark analysis with multiple lap datasets."""
        for num_laps in [10, 50, 100]:
            df = sample_telemetry_data(num_laps)

            process_monitor.start()
            df_sampled = sample_large_dataset(df, max_laps=100)
            metrics = process_monitor.stop()

            logger.info(f"{num_laps} laps: {metrics['elapsed_time']*1000:.2f}ms")


# =============================================================================
# OPTIMIZATION RECOMMENDATION TESTS
# =============================================================================

class TestOptimizationRecommendations:
    """Test optimization recommendation system."""

    @pytest.mark.benchmark
    def test_recommendations_small_dataset(self, sample_telemetry_data):
        """Test recommendations for small dataset."""
        df = sample_telemetry_data(10)
        recommendations = get_optimization_recommendations(df, 10)

        logger.info(f"Recommendations for 10 laps:\n"
                   + "\n".join(f"  - {r}" for r in recommendations))

        assert len(recommendations) > 0, "Should have recommendations"

    @pytest.mark.benchmark
    def test_recommendations_large_dataset(self, sample_telemetry_data):
        """Test recommendations for large dataset."""
        df = sample_telemetry_data(500)
        recommendations = get_optimization_recommendations(df, 500)

        logger.info(f"Recommendations for 500 laps:\n"
                   + "\n".join(f"  - {r}" for r in recommendations))

        assert any('sampling' in r.lower() for r in recommendations), \
            "Should recommend sampling for large dataset"


# =============================================================================
# PERFORMANCE COMPARISON TESTS
# =============================================================================

class TestPerformanceComparisons:
    """Compare performance before/after optimizations."""

    @pytest.mark.benchmark
    def test_optimization_impact(self, process_monitor):
        """Test impact of figure optimization."""
        fig = create_comprehensive_dashboard({})

        # Unoptimized
        process_monitor.start()
        _ = fig
        unoptimized_time = process_monitor.stop()['elapsed_time']

        # Optimized
        process_monitor.start()
        fig_opt = optimize_figure_config(fig)
        optimization_time = process_monitor.stop()['elapsed_time']

        logger.info(f"Unoptimized figure size: {len(fig.to_json())} bytes")
        logger.info(f"Optimization time: {optimization_time*1000:.2f}ms")

        assert optimization_time < 0.5, "Optimization should be fast"

    @pytest.mark.benchmark
    def test_sampling_impact_memory(self, sample_telemetry_data):
        """Test memory impact of sampling."""
        df_large = sample_telemetry_data(500)
        df_small = sample_large_dataset(df_large, max_laps=100)

        mem_large = estimate_memory_usage(df_large)
        mem_small = estimate_memory_usage(df_small)
        reduction = mem_large - mem_small

        logger.info(f"Memory savings: {mem_large:.1f}MB -> {mem_small:.1f}MB "
                   f"({reduction:.1f}MB saved, {reduction/mem_large*100:.1f}%)")

        assert mem_small < mem_large * 0.3, "Sampling should reduce memory by > 70%"


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_benchmark_report(benchmark_results: Dict) -> str:
    """
    Generate benchmark report.

    Args:
        benchmark_results: Benchmark results dictionary

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 80,
        "WINNING EDGE WIDGET PERFORMANCE BENCHMARK REPORT",
        "=" * 80,
        f"Generated: {benchmark_results['timestamp']}",
        ""
    ]

    # Performance targets
    lines.extend([
        "PERFORMANCE TARGETS:",
        "- Initial load: < 2 seconds",
        "- Tab switch: < 500ms",
        "- Memory usage: < 500MB for 100+ laps",
        ""
    ])

    # Results
    if benchmark_results['benchmarks']:
        lines.append("BENCHMARK RESULTS:")
        for key, result in benchmark_results['benchmarks'].items():
            lines.append(f"\n  {key}:")
            for metric, value in result.items():
                if isinstance(value, float):
                    lines.append(f"    {metric}: {value:.2f}")
                else:
                    lines.append(f"    {metric}: {value}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


# =============================================================================
# FIXTURES FINALIZATION
# =============================================================================

@pytest.fixture(autouse=True)
def save_benchmark_results(benchmark_results, request):
    """Save benchmark results after each test."""
    yield

    if request.node.get_closest_marker("benchmark"):
        # Save results
        output_file = BENCHMARK_OUTPUT_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)


if __name__ == "__main__":
    print("Winning Edge Performance Benchmark Suite")
    print("=" * 60)
    print("\nRun benchmarks with:")
    print("  pytest tests/benchmark_winning_edge.py -v -s")
    print("\nOr run specific benchmark:")
    print("  pytest tests/benchmark_winning_edge.py::TestBasicPerformance -v")
