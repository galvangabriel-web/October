"""
Winning Edge Callback Helpers - Vehicle-Specific Data Processing
==================================================================

Helper functions that process telemetry data for specific vehicles and return
formatted data structures for Winning Edge visualizations.

These functions bridge the gap between:
1. Raw telemetry data (from upload-data store)
2. WinningEdgeDataProcessor (extracts corner metrics)
3. Visualization functions (require specific dict formats)

Author: GR Cup Racing Analytics Team
Version: 1.1.0 - Vehicle Selection Fix
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from src.dashboard.winning_edge_data_processor import (
    WinningEdgeDataProcessor,
    CornerMetrics,
    PerformanceGap,
    ImprovementTarget
)

logger = logging.getLogger(__name__)

# Global processor instance (initialized once)
_processor = None


def get_processor() -> WinningEdgeDataProcessor:
    """Get or create global processor instance."""
    global _processor
    if _processor is None:
        _processor = WinningEdgeDataProcessor()
    return _processor


# ============================================================================
# HELPER FUNCTIONS FOR EACH VISUALIZATION
# ============================================================================

def process_corner_data_for_heatmap(
    telemetry_df: pd.DataFrame,
    vehicle_number: int
) -> Dict[str, Dict]:
    """
    Process telemetry to extract corner time loss data for heatmap.

    Args:
        telemetry_df: Raw telemetry dataframe
        vehicle_number: Vehicle to analyze

    Returns:
        Dict with corner names as keys:
        {'Turn 1': {'time_loss': 0.210, 'pct_of_total': 48}, ...}
    """
    try:
        processor = get_processor()

        # Extract corner metrics for this vehicle
        corner_metrics = processor.process_telemetry_for_corners(
            telemetry_df,
            vehicle_number
        )

        if not corner_metrics:
            return {}

        # Find best lap
        best_lap = _find_best_lap(corner_metrics)

        # Calculate gaps to best lap
        gaps = processor.calculate_real_time_gaps(corner_metrics, best_lap)

        if not gaps:
            return {}

        # Calculate total time loss
        total_loss = sum(gap.total_time_gap for gap in gaps)

        # Format for heatmap
        corner_data = {}
        for gap in gaps:
            corner_name = gap.corner_name.replace('_', ' ').title()
            corner_data[corner_name] = {
                'time_loss': gap.total_time_gap,
                'pct_of_total': (gap.total_time_gap / total_loss * 100) if total_loss > 0 else 0
            }

        return corner_data

    except Exception as e:
        logger.error(f"Error processing corner data for heatmap: {e}")
        return {}


def process_speed_gap_data_for_spider(
    telemetry_df: pd.DataFrame,
    vehicle_number: int
) -> Dict[str, Dict]:
    """
    Process telemetry to extract speed gap data for spider chart.

    Args:
        telemetry_df: Raw telemetry dataframe
        vehicle_number: Vehicle to analyze

    Returns:
        Dict with corner names as keys:
        {'Turn 1': {'entry_gap': 2.5, 'apex_gap': 3.2, 'exit_gap': 3.7}, ...}
    """
    try:
        processor = get_processor()
        corner_metrics = processor.process_telemetry_for_corners(
            telemetry_df,
            vehicle_number
        )

        if not corner_metrics:
            return {}

        best_lap = _find_best_lap(corner_metrics)
        gaps = processor.calculate_real_time_gaps(corner_metrics, best_lap)

        if not gaps:
            return {}

        # Format for spider chart
        speed_gaps = {}
        for gap in gaps:
            corner_name = gap.corner_name.replace('_', ' ').title()
            speed_gaps[corner_name] = {
                'entry_gap': gap.entry_speed_gap,
                'apex_gap': gap.apex_speed_gap,
                'exit_gap': gap.exit_speed_gap
            }

        return speed_gaps

    except Exception as e:
        logger.error(f"Error processing speed gap data for spider: {e}")
        return {}


def process_brake_exit_correlation_data(
    telemetry_df: pd.DataFrame,
    vehicle_number: int
) -> Tuple[List[float], List[float], List[str]]:
    """
    Process telemetry to extract brake point vs exit speed correlation.

    Args:
        telemetry_df: Raw telemetry dataframe
        vehicle_number: Vehicle to analyze

    Returns:
        Tuple of (brake_deltas, exit_deltas, corner_names)
    """
    try:
        processor = get_processor()
        corner_metrics = processor.process_telemetry_for_corners(
            telemetry_df,
            vehicle_number
        )

        if not corner_metrics:
            return [], [], []

        best_lap = _find_best_lap(corner_metrics)
        gaps = processor.calculate_real_time_gaps(corner_metrics, best_lap)

        if not gaps:
            return [], [], []

        brake_deltas = []
        exit_deltas = []
        corner_names = []

        for gap in gaps:
            if gap.brake_point_gap is not None:
                brake_deltas.append(gap.brake_point_gap)
                exit_deltas.append(gap.exit_speed_gap)
                corner_names.append(gap.corner_name.replace('_', ' ').title())

        return brake_deltas, exit_deltas, corner_names

    except Exception as e:
        logger.error(f"Error processing brake-exit correlation data: {e}")
        return [], [], []


def process_phase_breakdown_data(
    telemetry_df: pd.DataFrame,
    vehicle_number: int
) -> Dict[str, Dict]:
    """
    Process telemetry to extract phase breakdown (brake/apex/exit times).

    Args:
        telemetry_df: Raw telemetry dataframe
        vehicle_number: Vehicle to analyze

    Returns:
        Dict with corner names as keys:
        {'Turn 1': {'brake': 1.2, 'apex': 0.8, 'exit': 1.5}, ...}
    """
    try:
        processor = get_processor()
        corner_metrics = processor.process_telemetry_for_corners(
            telemetry_df,
            vehicle_number
        )

        if not corner_metrics:
            return {}

        # Group by corner and calculate average phase times
        phase_data = {}
        corner_groups = {}

        for metric in corner_metrics:
            if metric.corner_name not in corner_groups:
                corner_groups[metric.corner_name] = []
            corner_groups[metric.corner_name].append(metric)

        for corner_name, metrics in corner_groups.items():
            brake_times = []
            apex_times = []
            exit_times = []

            for metric in metrics:
                if metric.brake_phase:
                    brake_times.append(metric.brake_phase.duration)
                if metric.apex_phase:
                    apex_times.append(metric.apex_phase.duration)
                if metric.exit_phase:
                    exit_times.append(metric.exit_phase.duration)

            display_name = corner_name.replace('_', ' ').title()
            phase_data[display_name] = {
                'brake': np.mean(brake_times) if brake_times else 0,
                'apex': np.mean(apex_times) if apex_times else 0,
                'exit': np.mean(exit_times) if exit_times else 0
            }

        return phase_data

    except Exception as e:
        logger.error(f"Error processing phase breakdown data: {e}")
        return {}


def process_consistency_data(
    telemetry_df: pd.DataFrame,
    vehicle_number: int
) -> List[Dict]:
    """
    Process telemetry to extract consistency metrics for matrix chart.

    Args:
        telemetry_df: Raw telemetry dataframe
        vehicle_number: Vehicle to analyze

    Returns:
        List of corner consistency data:
        [{'corner': 'Turn 1', 'consistency': 95.2, 'time_loss': 0.210}, ...]
    """
    try:
        processor = get_processor()
        corner_metrics = processor.process_telemetry_for_corners(
            telemetry_df,
            vehicle_number
        )

        if not corner_metrics:
            return []

        # Calculate consistency per corner
        corner_groups = {}
        for metric in corner_metrics:
            if metric.corner_name not in corner_groups:
                corner_groups[metric.corner_name] = []
            corner_groups[metric.corner_name].append(metric.total_duration)

        consistency_data = []
        for corner_name, times in corner_groups.items():
            if len(times) > 1:
                mean_time = np.mean(times)
                std_time = np.std(times)
                consistency = max(0, 100 - (std_time / mean_time * 100)) if mean_time > 0 else 0
            else:
                consistency = 100  # Single lap = perfect consistency

            consistency_data.append({
                'corner': corner_name.replace('_', ' ').title(),
                'consistency': consistency,
                'avg_time': np.mean(times)
            })

        return consistency_data

    except Exception as e:
        logger.error(f"Error processing consistency data: {e}")
        return []


def process_action_card_data(
    telemetry_df: pd.DataFrame,
    vehicle_number: int,
    turn_name: str
) -> Tuple[Dict, Dict]:
    """
    Process telemetry to extract action card data for specific turn.

    Args:
        telemetry_df: Raw telemetry dataframe
        vehicle_number: Vehicle to analyze
        turn_name: Turn to analyze (e.g., "Turn 6")

    Returns:
        Tuple of (current_metrics, target_metrics)
    """
    try:
        processor = get_processor()
        corner_metrics = processor.process_telemetry_for_corners(
            telemetry_df,
            vehicle_number
        )

        if not corner_metrics:
            return {}, {}

        best_lap = _find_best_lap(corner_metrics)
        gaps = processor.calculate_real_time_gaps(corner_metrics, best_lap)
        targets = processor.generate_improvement_targets(gaps)

        # Find metrics for requested turn
        turn_key = turn_name.lower().replace(' ', '_')
        current = {}
        target = {}

        for gap in gaps:
            if turn_key in gap.corner_name.lower():
                current = {
                    'entry_speed': gap.entry_speed_gap,  # This is the gap, not absolute
                    'apex_speed': gap.apex_speed_gap,
                    'exit_speed': gap.exit_speed_gap,
                    'time_loss': gap.total_time_gap
                }

        for tgt in targets:
            if turn_key in tgt.corner_name.lower():
                target = {
                    'brake_point': tgt.brake_point_target if hasattr(tgt, 'brake_point_target') else None,
                    'brake_pressure': tgt.brake_pressure_target if hasattr(tgt, 'brake_pressure_target') else None,
                    'throttle_point': tgt.throttle_point_target if hasattr(tgt, 'throttle_point_target') else None,
                    'exit_speed': tgt.exit_speed_target if hasattr(tgt, 'exit_speed_target') else None
                }

        return current, target

    except Exception as e:
        logger.error(f"Error processing action card data: {e}")
        return {}, {}


# ============================================================================
# HELPER UTILITIES
# ============================================================================

def _find_best_lap(corner_metrics: List[CornerMetrics]) -> int:
    """Find the lap with minimum total time across all corners."""
    if not corner_metrics:
        return 1

    # Group by lap
    lap_times = {}
    for metric in corner_metrics:
        lap = metric.lap_number
        if lap not in lap_times:
            lap_times[lap] = 0
        lap_times[lap] += metric.total_duration

    # Find best lap
    best_lap = min(lap_times.items(), key=lambda x: x[1])[0]
    return best_lap


def load_telemetry_from_json(data_json: str) -> Optional[pd.DataFrame]:
    """
    Load telemetry dataframe from JSON string (from dcc.Store).

    Args:
        data_json: JSON string from upload-data store

    Returns:
        Pandas DataFrame or None if loading fails
    """
    if not data_json:
        return None

    try:
        df = pd.read_json(data_json, orient='split')
        return df
    except Exception as e:
        logger.error(f"Error loading telemetry from JSON: {e}")
        return None


def filter_telemetry_by_vehicle(
    telemetry_df: pd.DataFrame,
    vehicle_number: int
) -> pd.DataFrame:
    """
    Filter telemetry dataframe to specific vehicle.

    Args:
        telemetry_df: Full telemetry dataframe
        vehicle_number: Vehicle to filter

    Returns:
        Filtered dataframe for vehicle
    """
    if 'vehicle_number' not in telemetry_df.columns:
        logger.warning("No vehicle_number column in telemetry")
        return telemetry_df

    return telemetry_df[telemetry_df['vehicle_number'] == vehicle_number]
