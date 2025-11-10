"""
Metric calculation functions for Winning Edge Widget.

This module provides performance calculation functions for race analysis:
- Time loss calculations per corner
- Brake/exit speed correlation analysis
- Race position prediction based on improvements
- Consistency scoring algorithms
- Improvement potential analysis
- Compound effect calculations (speed cascades)

Author: GR Cup Racing Analytics Platform
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from dataclasses import dataclass
import warnings


@dataclass
class TimeLossResult:
    """Result object for time loss calculations."""
    corner: str
    time_loss: float
    brake_delta: float
    exit_delta: float
    optimal_brake_point: float
    optimal_exit_speed: float
    current_brake_point: float
    current_exit_speed: float
    improvement_potential: float


@dataclass
class CorrelationResult:
    """Result object for correlation analysis."""
    correlation_coefficient: float
    p_value: float
    brake_exit_ratio: float
    optimal_ratio: float
    confidence_interval: Tuple[float, float]
    regression_equation: str


@dataclass
class RaceSimulationResult:
    """Result object for race simulation."""
    final_position: int
    positions_gained: int
    time_advantage: float
    lap_by_lap_positions: List[int]
    probability_of_win: float


@dataclass
class ConsistencyScore:
    """Result object for consistency analysis."""
    overall_score: float
    lap_time_consistency: float
    brake_consistency: float
    exit_consistency: float
    coefficient_of_variation: float
    std_deviation: float


def calculate_time_loss(
    telemetry_df: pd.DataFrame,
    vehicle_number: int,
    reference_vehicle: Optional[int] = None,
    corner_ranges: Optional[Dict[str, Tuple[float, float]]] = None
) -> List[TimeLossResult]:
    """
    Calculate time loss per corner compared to reference (best lap or competitor).

    Args:
        telemetry_df: Telemetry DataFrame with columns [vehicle_number, lap, corner,
                      timestamp, speed, pbrake_f, distance]
        vehicle_number: Target vehicle to analyze
        reference_vehicle: Reference vehicle number (if None, uses best lap from target vehicle)
        corner_ranges: Dict mapping corner names to (start_distance, end_distance) tuples

    Returns:
        List of TimeLossResult objects for each corner

    Raises:
        ValueError: If vehicle not found or insufficient data
    """
    # Validate input
    if telemetry_df.empty:
        raise ValueError("telemetry_df cannot be empty")

    vehicle_data = telemetry_df[telemetry_df['vehicle_number'] == vehicle_number].copy()
    if vehicle_data.empty:
        raise ValueError(f"No data found for vehicle {vehicle_number}")

    # Get reference data
    if reference_vehicle is None:
        # Use best lap from same vehicle
        lap_times = vehicle_data.groupby('lap')['timestamp'].apply(
            lambda x: x.max() - x.min()
        )
        best_lap = lap_times.idxmin()
        reference_data = vehicle_data[vehicle_data['lap'] == best_lap].copy()
    else:
        reference_data = telemetry_df[
            telemetry_df['vehicle_number'] == reference_vehicle
        ].copy()
        if reference_data.empty:
            raise ValueError(f"No data found for reference vehicle {reference_vehicle}")

        # Use best lap from reference
        lap_times = reference_data.groupby('lap')['timestamp'].apply(
            lambda x: x.max() - x.min()
        )
        best_lap = lap_times.idxmin()
        reference_data = reference_data[reference_data['lap'] == best_lap].copy()

    # If no corner ranges provided, auto-detect or use distance segments
    if corner_ranges is None:
        corner_ranges = _auto_detect_corners(vehicle_data)

    results = []

    for corner_name, (start_dist, end_dist) in corner_ranges.items():
        try:
            # Extract corner data
            vehicle_corner = vehicle_data[
                (vehicle_data['distance'] >= start_dist) &
                (vehicle_data['distance'] <= end_dist)
            ]
            reference_corner = reference_data[
                (reference_data['distance'] >= start_dist) &
                (reference_data['distance'] <= end_dist)
            ]

            if vehicle_corner.empty or reference_corner.empty:
                continue

            # Calculate metrics
            vehicle_time = (
                vehicle_corner['timestamp'].max() - vehicle_corner['timestamp'].min()
            )
            reference_time = (
                reference_corner['timestamp'].max() - reference_corner['timestamp'].min()
            )
            time_loss = vehicle_time - reference_time

            # Brake point analysis
            vehicle_brake = vehicle_corner[vehicle_corner['pbrake_f'] > 50]
            reference_brake = reference_corner[reference_corner['pbrake_f'] > 50]

            if not vehicle_brake.empty and not reference_brake.empty:
                vehicle_brake_point = vehicle_brake['distance'].min()
                reference_brake_point = reference_brake['distance'].min()
                brake_delta = vehicle_brake_point - reference_brake_point
            else:
                vehicle_brake_point = start_dist
                reference_brake_point = start_dist
                brake_delta = 0.0

            # Exit speed analysis
            exit_threshold = start_dist + (end_dist - start_dist) * 0.8
            vehicle_exit = vehicle_corner[vehicle_corner['distance'] >= exit_threshold]
            reference_exit = reference_corner[reference_corner['distance'] >= exit_threshold]

            if not vehicle_exit.empty and not reference_exit.empty:
                vehicle_exit_speed = vehicle_exit['speed'].max()
                reference_exit_speed = reference_exit['speed'].max()
                exit_delta = reference_exit_speed - vehicle_exit_speed
            else:
                vehicle_exit_speed = 0.0
                reference_exit_speed = 0.0
                exit_delta = 0.0

            # Calculate improvement potential
            improvement_potential = max(0, time_loss)

            result = TimeLossResult(
                corner=corner_name,
                time_loss=time_loss,
                brake_delta=brake_delta,
                exit_delta=exit_delta,
                optimal_brake_point=reference_brake_point,
                optimal_exit_speed=reference_exit_speed,
                current_brake_point=vehicle_brake_point,
                current_exit_speed=vehicle_exit_speed,
                improvement_potential=improvement_potential
            )

            results.append(result)

        except Exception as e:
            warnings.warn(f"Error processing corner {corner_name}: {e}")
            continue

    return results


def calculate_brake_exit_correlation(
    telemetry_df: pd.DataFrame,
    vehicle_number: int,
    corner_name: Optional[str] = None
) -> CorrelationResult:
    """
    Calculate 1:1 correlation between brake point and exit speed.

    The "1:1 rule" states that braking 1 meter later can cost 1 km/h of exit speed.
    This function quantifies that relationship for specific corners or overall.

    Args:
        telemetry_df: Telemetry DataFrame
        vehicle_number: Vehicle to analyze
        corner_name: Specific corner (if None, analyzes all corners)

    Returns:
        CorrelationResult with correlation coefficient and optimal ratio

    Raises:
        ValueError: If insufficient data for correlation
    """
    vehicle_data = telemetry_df[telemetry_df['vehicle_number'] == vehicle_number].copy()

    if vehicle_data.empty:
        raise ValueError(f"No data found for vehicle {vehicle_number}")

    # Filter by corner if specified
    if corner_name is not None:
        vehicle_data = vehicle_data[vehicle_data['corner'] == corner_name]

    # Extract brake points and exit speeds per lap
    laps = vehicle_data['lap'].unique()

    brake_points = []
    exit_speeds = []

    for lap in laps:
        lap_data = vehicle_data[vehicle_data['lap'] == lap]

        # Find brake point (first significant brake application)
        brake_data = lap_data[lap_data['pbrake_f'] > 50]
        if not brake_data.empty:
            brake_point = brake_data['distance'].min()
            brake_points.append(brake_point)

            # Find exit speed (max speed after brake release)
            post_brake = lap_data[lap_data['distance'] > brake_point + 50]
            if not post_brake.empty:
                exit_speed = post_brake['speed'].max()
                exit_speeds.append(exit_speed)
            else:
                brake_points.pop()  # Remove brake point if no exit data

    if len(brake_points) < 3:
        raise ValueError(
            f"Insufficient data for correlation (need ≥3 laps, found {len(brake_points)})"
        )

    brake_points = np.array(brake_points)
    exit_speeds = np.array(exit_speeds)

    # Calculate correlation
    correlation, p_value = stats.pearsonr(brake_points, exit_speeds)

    # Calculate brake/exit ratio (meters per km/h)
    slope, intercept, r_value, _, _ = stats.linregress(brake_points, exit_speeds)

    # The "1:1 rule" suggests optimal ratio is ~1.0 (1m brake = 1 km/h exit loss)
    brake_exit_ratio = abs(1 / slope) if slope != 0 else 0
    optimal_ratio = 1.0  # Reference value

    # Calculate confidence interval (95%)
    n = len(brake_points)
    stderr = np.sqrt((1 - r_value**2) / (n - 2))
    ci = 1.96 * stderr  # 95% CI
    confidence_interval = (correlation - ci, correlation + ci)

    # Regression equation
    regression_equation = f"Exit Speed = {slope:.3f} * Brake Point + {intercept:.3f}"

    return CorrelationResult(
        correlation_coefficient=correlation,
        p_value=p_value,
        brake_exit_ratio=brake_exit_ratio,
        optimal_ratio=optimal_ratio,
        confidence_interval=confidence_interval,
        regression_equation=regression_equation
    )


def predict_race_positions(
    current_lap_times: pd.DataFrame,
    improvement_per_lap: float,
    total_laps: int,
    competitor_times: Optional[pd.DataFrame] = None
) -> RaceSimulationResult:
    """
    Simulate race positions based on consistent improvement per lap.

    Args:
        current_lap_times: DataFrame with [lap, lap_time, position]
        improvement_per_lap: Time improvement per lap (seconds)
        total_laps: Total race laps to simulate
        competitor_times: Optional DataFrame with competitor lap times

    Returns:
        RaceSimulationResult with projected final position and lap-by-lap analysis

    Raises:
        ValueError: If current_lap_times is empty or invalid
    """
    if current_lap_times.empty:
        raise ValueError("current_lap_times cannot be empty")

    if 'lap_time' not in current_lap_times.columns:
        raise ValueError("current_lap_times must contain 'lap_time' column")

    # Get baseline performance
    baseline_time = current_lap_times['lap_time'].mean()
    current_position = current_lap_times['position'].iloc[-1] if 'position' in current_lap_times.columns else 10

    # Initialize simulation
    lap_by_lap_positions = []
    cumulative_time_advantage = 0.0

    # Simulate each lap
    for lap in range(1, total_laps + 1):
        # Calculate improved lap time
        improved_time = baseline_time - improvement_per_lap

        # Update cumulative advantage
        cumulative_time_advantage += improvement_per_lap

        # Estimate position change (rough: 1 second = 1-2 positions depending on field spread)
        if competitor_times is not None:
            # Use actual competitor times for more accurate simulation
            field_spread = competitor_times['lap_time'].std()
            position_change_rate = 2.0 / field_spread  # positions per second
        else:
            # Default assumption: tight field, 1s = ~1.5 positions
            position_change_rate = 1.5

        positions_gained = int(cumulative_time_advantage * position_change_rate)
        projected_position = max(1, current_position - positions_gained)

        lap_by_lap_positions.append(projected_position)

    # Calculate final metrics
    final_position = lap_by_lap_positions[-1]
    positions_gained = current_position - final_position
    time_advantage = cumulative_time_advantage

    # Estimate probability of win (simplified model)
    if final_position == 1:
        probability_of_win = min(0.95, 0.5 + (time_advantage / 10))  # Cap at 95%
    elif final_position <= 3:
        probability_of_win = 0.3 - (final_position - 1) * 0.1
    else:
        probability_of_win = 0.05

    return RaceSimulationResult(
        final_position=final_position,
        positions_gained=positions_gained,
        time_advantage=time_advantage,
        lap_by_lap_positions=lap_by_lap_positions,
        probability_of_win=probability_of_win
    )


def calculate_consistency_score(
    telemetry_df: pd.DataFrame,
    vehicle_number: int,
    metrics: Optional[List[str]] = None
) -> ConsistencyScore:
    """
    Calculate statistical consistency score across multiple performance metrics.

    Consistency is measured using coefficient of variation (CV) and standard deviation
    across laps. Lower CV = more consistent performance.

    Args:
        telemetry_df: Telemetry DataFrame
        vehicle_number: Vehicle to analyze
        metrics: List of metrics to analyze (default: ['lap_time', 'brake_pressure', 'exit_speed'])

    Returns:
        ConsistencyScore with overall and per-metric consistency ratings

    Raises:
        ValueError: If vehicle not found or insufficient laps
    """
    if metrics is None:
        metrics = ['lap_time']

    vehicle_data = telemetry_df[telemetry_df['vehicle_number'] == vehicle_number].copy()

    if vehicle_data.empty:
        raise ValueError(f"No data found for vehicle {vehicle_number}")

    # Calculate lap times
    lap_times = vehicle_data.groupby('lap')['timestamp'].apply(
        lambda x: x.max() - x.min()
    )

    if len(lap_times) < 3:
        raise ValueError(f"Need at least 3 laps for consistency analysis, found {len(lap_times)}")

    # Lap time consistency
    lap_time_mean = lap_times.mean()
    lap_time_std = lap_times.std()
    lap_time_cv = (lap_time_std / lap_time_mean) * 100  # Coefficient of variation (%)

    # Convert CV to 0-100 score (lower CV = higher score)
    # CV < 1% = excellent (90-100), CV 1-2% = good (70-90), CV > 2% = poor (<70)
    lap_time_consistency = max(0, 100 - (lap_time_cv * 50))

    # Brake consistency
    brake_data = vehicle_data.groupby('lap')['pbrake_f'].max()
    if len(brake_data) > 0:
        brake_std = brake_data.std()
        brake_mean = brake_data.mean()
        brake_cv = (brake_std / brake_mean) * 100 if brake_mean > 0 else 100
        brake_consistency = max(0, 100 - (brake_cv * 10))
    else:
        brake_consistency = 50.0

    # Exit speed consistency
    exit_speeds = vehicle_data.groupby('lap')['speed'].max()
    if len(exit_speeds) > 0:
        exit_std = exit_speeds.std()
        exit_mean = exit_speeds.mean()
        exit_cv = (exit_std / exit_mean) * 100 if exit_mean > 0 else 100
        exit_consistency = max(0, 100 - (exit_cv * 20))
    else:
        exit_consistency = 50.0

    # Overall consistency (weighted average)
    overall_score = (
        lap_time_consistency * 0.5 +
        brake_consistency * 0.25 +
        exit_consistency * 0.25
    )

    return ConsistencyScore(
        overall_score=overall_score,
        lap_time_consistency=lap_time_consistency,
        brake_consistency=brake_consistency,
        exit_consistency=exit_consistency,
        coefficient_of_variation=lap_time_cv,
        std_deviation=lap_time_std
    )


def calculate_improvement_potential(
    telemetry_df: pd.DataFrame,
    vehicle_number: int,
    benchmark_source: str = "best_lap"
) -> Dict[str, float]:
    """
    Calculate improvement potential by comparing current performance to benchmark.

    Args:
        telemetry_df: Telemetry DataFrame
        vehicle_number: Vehicle to analyze
        benchmark_source: Source for benchmark ("best_lap", "theoretical_best", "competitor")

    Returns:
        Dictionary with improvement potential per metric

    Raises:
        ValueError: If invalid benchmark_source or insufficient data
    """
    vehicle_data = telemetry_df[telemetry_df['vehicle_number'] == vehicle_number].copy()

    if vehicle_data.empty:
        raise ValueError(f"No data found for vehicle {vehicle_number}")

    # Calculate current performance
    lap_times = vehicle_data.groupby('lap')['timestamp'].apply(
        lambda x: x.max() - x.min()
    )
    current_avg_time = lap_times.mean()

    # Determine benchmark
    if benchmark_source == "best_lap":
        benchmark_time = lap_times.min()
    elif benchmark_source == "theoretical_best":
        # Theoretical best: best sector times combined
        # Simplified: use best lap - 0.5s as theoretical
        benchmark_time = lap_times.min() - 0.5
    elif benchmark_source == "competitor":
        # Use fastest competitor (if available in data)
        all_vehicles = telemetry_df['vehicle_number'].unique()
        competitor_times = []
        for veh in all_vehicles:
            if veh != vehicle_number:
                veh_times = telemetry_df[telemetry_df['vehicle_number'] == veh].groupby('lap')['timestamp'].apply(
                    lambda x: x.max() - x.min()
                )
                if len(veh_times) > 0:
                    competitor_times.append(veh_times.min())

        if competitor_times:
            benchmark_time = min(competitor_times)
        else:
            benchmark_time = lap_times.min()
    else:
        raise ValueError(f"Invalid benchmark_source: {benchmark_source}")

    # Calculate gaps
    improvement_potential = {
        "lap_time_gap": current_avg_time - benchmark_time,
        "percentage_gap": ((current_avg_time - benchmark_time) / benchmark_time) * 100,
        "current_average": current_avg_time,
        "benchmark": benchmark_time,
        "best_lap": lap_times.min(),
        "worst_lap": lap_times.max(),
        "consistency_gap": lap_times.max() - lap_times.min()
    }

    return improvement_potential


def calculate_compound_effect(
    corner_improvements: Dict[str, float],
    track_length: float = 3000.0,
    straight_length_pct: float = 0.4
) -> Dict[str, Any]:
    """
    Calculate compound effect of corner exit speed on straight-line speed (cascade effect).

    Principle: Exiting a corner 1 km/h faster carries through the entire following straight,
    resulting in greater time savings than the corner alone.

    Args:
        corner_improvements: Dict mapping corner names to exit speed improvements (km/h)
        track_length: Total track length in meters
        straight_length_pct: Percentage of track that is straights (default 40%)

    Returns:
        Dictionary with compound effect analysis

    Raises:
        ValueError: If inputs are invalid
    """
    if not corner_improvements:
        raise ValueError("corner_improvements cannot be empty")

    if track_length <= 0:
        raise ValueError("track_length must be positive")

    if not 0 <= straight_length_pct <= 1:
        raise ValueError("straight_length_pct must be between 0 and 1")

    total_straight_length = track_length * straight_length_pct
    avg_straight_length = total_straight_length / len(corner_improvements)

    results = {}
    total_time_saved = 0.0

    for corner, exit_speed_gain in corner_improvements.items():
        # Calculate direct corner time saving (simplified)
        corner_length = 100  # Assume 100m corner
        corner_time_saved = (corner_length / 1000) / (exit_speed_gain + 100) * exit_speed_gain

        # Calculate cascade effect on following straight
        # If exiting 1 km/h faster, you're faster through entire straight
        # Time saved = distance / speed_original - distance / speed_improved
        baseline_straight_speed = 150  # km/h assumption
        improved_straight_speed = baseline_straight_speed + (exit_speed_gain * 0.7)  # 70% carries through

        straight_time_saved = (
            (avg_straight_length / 1000) / (baseline_straight_speed / 3600) -
            (avg_straight_length / 1000) / (improved_straight_speed / 3600)
        )

        # Total effect
        total_effect = corner_time_saved + straight_time_saved
        cascade_multiplier = total_effect / corner_time_saved if corner_time_saved > 0 else 1.0

        results[corner] = {
            "exit_speed_gain": exit_speed_gain,
            "direct_time_saved": corner_time_saved,
            "cascade_time_saved": straight_time_saved,
            "total_time_saved": total_effect,
            "cascade_multiplier": cascade_multiplier
        }

        total_time_saved += total_effect

    # Summary statistics
    avg_cascade_multiplier = np.mean([r["cascade_multiplier"] for r in results.values()])

    return {
        "per_corner_results": results,
        "total_time_saved": total_time_saved,
        "average_cascade_multiplier": avg_cascade_multiplier,
        "lap_time_improvement": total_time_saved,
        "theoretical_max_improvement": sum(corner_improvements.values()) * 0.1  # 10% of exit speed gains
    }


def _auto_detect_corners(telemetry_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Auto-detect corner locations based on brake pressure patterns.

    Args:
        telemetry_df: Telemetry DataFrame with brake pressure and distance

    Returns:
        Dictionary mapping corner names to (start_distance, end_distance)
    """
    # Find braking zones (brake pressure > threshold)
    brake_threshold = 30  # bar

    brake_zones = telemetry_df[telemetry_df['pbrake_f'] > brake_threshold].copy()

    if brake_zones.empty:
        # Fallback: divide track into equal segments
        max_dist = telemetry_df['distance'].max()
        num_corners = 10  # Default assumption
        segment_length = max_dist / num_corners

        return {
            f"Corner {i+1}": (i * segment_length, (i+1) * segment_length)
            for i in range(num_corners)
        }

    # Group consecutive braking zones
    brake_zones['distance_diff'] = brake_zones['distance'].diff()
    brake_zones['new_corner'] = brake_zones['distance_diff'] > 100  # 100m gap = new corner

    brake_zones['corner_id'] = brake_zones['new_corner'].cumsum()

    corners = {}
    for corner_id, group in brake_zones.groupby('corner_id'):
        corner_name = f"Corner {corner_id + 1}"
        start_dist = group['distance'].min()
        end_dist = group['distance'].max()
        corners[corner_name] = (start_dist, end_dist)

    return corners
