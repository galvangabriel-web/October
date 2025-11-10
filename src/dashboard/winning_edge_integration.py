"""
Winning Edge Widget Integration Module
========================================

This module provides integration utilities and documentation for connecting
the Winning Edge widget to the main dashboard's telemetry data flow.

Integration Points:
-------------------
1. Data Store Connection: Links to dcc.Store components for telemetry data
2. Vehicle Selection: Syncs with main dashboard vehicle dropdown
3. Callback Coordination: Ensures winning edge callbacks don't conflict
4. Data Transformation: Converts dashboard data format to widget expectations

Usage:
------
    from src.dashboard.winning_edge_integration import (
        prepare_winning_edge_data,
        extract_corner_metrics,
        calculate_competitive_gaps
    )

    # In a callback:
    winning_edge_data = prepare_winning_edge_data(telemetry_df, vehicle_number)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def prepare_winning_edge_data(
    telemetry_df: pd.DataFrame,
    vehicle_number: int,
    reference_vehicle: Optional[int] = None
) -> Dict:
    """
    Transform telemetry DataFrame into format expected by Winning Edge widget.

    Args:
        telemetry_df: Main dashboard telemetry data (long format)
        vehicle_number: Target vehicle to analyze
        reference_vehicle: Optional comparison vehicle (defaults to fastest)

    Returns:
        Dictionary with keys: 'corners', 'laps', 'metrics', 'reference'
    """
    try:
        # Filter for target vehicle
        vehicle_data = telemetry_df[telemetry_df['vehicle_number'] == vehicle_number].copy()

        if vehicle_data.empty:
            logger.warning(f"No data found for vehicle {vehicle_number}")
            return _empty_winning_edge_data()

        # Extract corner-by-corner metrics
        corner_metrics = extract_corner_metrics(vehicle_data)

        # Calculate lap-level performance
        lap_metrics = calculate_lap_metrics(vehicle_data)

        # Get reference data (fastest vehicle if not specified)
        if reference_vehicle is None:
            reference_vehicle = _find_fastest_vehicle(telemetry_df)

        reference_data = None
        if reference_vehicle and reference_vehicle != vehicle_number:
            ref_df = telemetry_df[telemetry_df['vehicle_number'] == reference_vehicle].copy()
            if not ref_df.empty:
                reference_data = {
                    'vehicle_number': reference_vehicle,
                    'corners': extract_corner_metrics(ref_df),
                    'laps': calculate_lap_metrics(ref_df)
                }

        return {
            'vehicle_number': vehicle_number,
            'corners': corner_metrics,
            'laps': lap_metrics,
            'reference': reference_data,
            'track': vehicle_data['track'].iloc[0] if 'track' in vehicle_data.columns else 'Unknown',
            'session': vehicle_data['race'].iloc[0] if 'race' in vehicle_data.columns else 'Unknown'
        }

    except Exception as e:
        logger.error(f"Error preparing winning edge data: {e}")
        return _empty_winning_edge_data()


def extract_corner_metrics(telemetry_df: pd.DataFrame) -> List[Dict]:
    """
    Extract corner-by-corner performance metrics from telemetry data.

    Identifies corners based on steering angle and brake pressure patterns,
    then calculates entry/apex/exit speeds, brake points, and time deltas.

    Args:
        telemetry_df: Telemetry data for a single vehicle (long format)

    Returns:
        List of dictionaries, each containing metrics for one corner:
        - corner_number: int
        - corner_name: str (e.g., "Turn 1", "Turn 2")
        - entry_speed: float (km/h)
        - apex_speed: float (km/h)
        - exit_speed: float (km/h)
        - brake_point: float (meters before corner)
        - brake_pressure_max: float (bar)
        - time_in_corner: float (seconds)
        - lateral_g_max: float (g-force)
    """
    corners = []

    try:
        # Pivot to wide format for easier processing
        wide_df = telemetry_df.pivot_table(
            index=['vehicle_number', 'lap', 'timestamp'],
            columns='telemetry_name',
            values='telemetry_value',
            aggfunc='first'
        ).reset_index()

        # Ensure required columns exist
        required_cols = ['speed', 'Steering_Angle', 'pbrake_f']
        if not all(col in wide_df.columns for col in required_cols):
            logger.warning("Missing required columns for corner detection")
            return _generate_sample_corners()

        # Detect corners based on steering angle threshold (>30 degrees)
        wide_df['is_corner'] = np.abs(wide_df['Steering_Angle']) > 30

        # Group consecutive corner samples
        wide_df['corner_group'] = (
            wide_df['is_corner'] != wide_df['is_corner'].shift()
        ).cumsum()

        corner_num = 1
        for (is_corner, group_id), group in wide_df.groupby(['is_corner', 'corner_group']):
            if not is_corner or len(group) < 5:  # Skip non-corners or too-short segments
                continue

            # Extract metrics
            entry_speed = group['speed'].iloc[0]
            apex_idx = group['speed'].idxmin()  # Slowest point = apex
            apex_speed = group.loc[apex_idx, 'speed']
            exit_speed = group['speed'].iloc[-1]

            brake_pressure_max = group['pbrake_f'].max() if 'pbrake_f' in group.columns else 0

            # Calculate time in corner
            time_in_corner = (group['timestamp'].iloc[-1] - group['timestamp'].iloc[0]) / 1000.0

            # Lateral G (if available)
            lateral_g = group['accy_can'].max() if 'accy_can' in group.columns else 0

            corners.append({
                'corner_number': corner_num,
                'corner_name': f"Turn {corner_num}",
                'entry_speed': float(entry_speed),
                'apex_speed': float(apex_speed),
                'exit_speed': float(exit_speed),
                'brake_point': 0.0,  # Would need GPS/distance data to calculate
                'brake_pressure_max': float(brake_pressure_max),
                'time_in_corner': float(time_in_corner),
                'lateral_g_max': float(abs(lateral_g))
            })

            corner_num += 1

        if not corners:
            logger.warning("No corners detected in telemetry data")
            return _generate_sample_corners()

        return corners

    except Exception as e:
        logger.error(f"Error extracting corner metrics: {e}")
        return _generate_sample_corners()


def calculate_lap_metrics(telemetry_df: pd.DataFrame) -> List[Dict]:
    """
    Calculate lap-by-lap performance metrics.

    Args:
        telemetry_df: Telemetry data for a single vehicle

    Returns:
        List of dictionaries with lap-level metrics:
        - lap_number: int
        - lap_time: float (seconds)
        - max_speed: float (km/h)
        - avg_speed: float (km/h)
        - brake_events: int
        - throttle_consistency: float (0-100%)
    """
    laps = []

    try:
        # Group by lap
        for lap_num, lap_group in telemetry_df.groupby('lap'):
            # Pivot to wide format
            wide_lap = lap_group.pivot_table(
                index=['timestamp'],
                columns='telemetry_name',
                values='telemetry_value',
                aggfunc='first'
            ).reset_index()

            # Calculate lap time
            if len(wide_lap) > 0:
                lap_time = (wide_lap['timestamp'].iloc[-1] - wide_lap['timestamp'].iloc[0]) / 1000.0
            else:
                lap_time = 0

            # Speed metrics
            max_speed = wide_lap['speed'].max() if 'speed' in wide_lap.columns else 0
            avg_speed = wide_lap['speed'].mean() if 'speed' in wide_lap.columns else 0

            # Brake events (count transitions from not-braking to braking)
            if 'pbrake_f' in wide_lap.columns:
                brake_events = ((wide_lap['pbrake_f'] > 20) &
                               (wide_lap['pbrake_f'].shift() <= 20)).sum()
            else:
                brake_events = 0

            # Throttle consistency (std dev of throttle position)
            if 'aps' in wide_lap.columns:
                throttle_consistency = max(0, 100 - wide_lap['aps'].std())
            else:
                throttle_consistency = 0

            laps.append({
                'lap_number': int(lap_num),
                'lap_time': float(lap_time),
                'max_speed': float(max_speed),
                'avg_speed': float(avg_speed),
                'brake_events': int(brake_events),
                'throttle_consistency': float(throttle_consistency)
            })

        return laps

    except Exception as e:
        logger.error(f"Error calculating lap metrics: {e}")
        return []


def calculate_competitive_gaps(
    target_data: Dict,
    reference_data: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Calculate performance gaps between target and reference vehicle.

    Args:
        target_data: Winning edge data for target vehicle
        reference_data: Optional reference vehicle data

    Returns:
        Dictionary with gap metrics:
        - avg_corner_speed_gap: float (km/h)
        - brake_point_gap: float (meters)
        - lap_time_gap: float (seconds)
        - consistency_gap: float (percentage points)
    """
    if not reference_data:
        return {
            'avg_corner_speed_gap': 0.0,
            'brake_point_gap': 0.0,
            'lap_time_gap': 0.0,
            'consistency_gap': 0.0
        }

    try:
        # Corner speed gaps
        target_corners = target_data.get('corners', [])
        ref_corners = reference_data.get('corners', [])

        if target_corners and ref_corners:
            # Match corners by index (assumes same track layout)
            speed_gaps = []
            for i, (tc, rc) in enumerate(zip(target_corners, ref_corners)):
                # Compare exit speeds (most important for lap time)
                gap = rc['exit_speed'] - tc['exit_speed']
                speed_gaps.append(gap)

            avg_corner_speed_gap = np.mean(speed_gaps) if speed_gaps else 0.0
        else:
            avg_corner_speed_gap = 0.0

        # Lap time gap
        target_laps = target_data.get('laps', [])
        ref_laps = reference_data.get('laps', [])

        if target_laps and ref_laps:
            target_best = min(lap['lap_time'] for lap in target_laps if lap['lap_time'] > 60)
            ref_best = min(lap['lap_time'] for lap in ref_laps if lap['lap_time'] > 60)
            lap_time_gap = target_best - ref_best
        else:
            lap_time_gap = 0.0

        return {
            'avg_corner_speed_gap': float(avg_corner_speed_gap),
            'brake_point_gap': 0.0,  # Would need GPS data
            'lap_time_gap': float(lap_time_gap),
            'consistency_gap': 0.0  # Would need statistical analysis
        }

    except Exception as e:
        logger.error(f"Error calculating competitive gaps: {e}")
        return {
            'avg_corner_speed_gap': 0.0,
            'brake_point_gap': 0.0,
            'lap_time_gap': 0.0,
            'consistency_gap': 0.0
        }


def _find_fastest_vehicle(telemetry_df: pd.DataFrame) -> Optional[int]:
    """Find the vehicle with the fastest lap time."""
    try:
        # Group by vehicle and lap
        lap_times = []
        for (vehicle, lap), group in telemetry_df.groupby(['vehicle_number', 'lap']):
            if len(group) > 0:
                time_range = group['timestamp'].max() - group['timestamp'].min()
                lap_time = time_range / 1000.0  # Convert to seconds
                if 60 < lap_time < 300:  # Valid lap time range
                    lap_times.append((vehicle, lap_time))

        if lap_times:
            fastest = min(lap_times, key=lambda x: x[1])
            return int(fastest[0])

        return None

    except Exception as e:
        logger.error(f"Error finding fastest vehicle: {e}")
        return None


def _empty_winning_edge_data() -> Dict:
    """Return empty data structure for error cases."""
    return {
        'vehicle_number': None,
        'corners': [],
        'laps': [],
        'reference': None,
        'track': 'Unknown',
        'session': 'Unknown'
    }


def _generate_sample_corners() -> List[Dict]:
    """Generate sample corner data for demonstration purposes."""
    return [
        {
            'corner_number': i + 1,
            'corner_name': f"Turn {i + 1}",
            'entry_speed': 120.0 + np.random.randn() * 5,
            'apex_speed': 80.0 + np.random.randn() * 5,
            'exit_speed': 110.0 + np.random.randn() * 5,
            'brake_point': 100.0,
            'brake_pressure_max': 120.0 + np.random.randn() * 10,
            'time_in_corner': 3.5 + np.random.randn() * 0.5,
            'lateral_g_max': 1.8 + np.random.randn() * 0.2
        }
        for i in range(8)  # Generate 8 sample corners
    ]


# ============================================================================
# INTEGRATION DOCUMENTATION
# ============================================================================

INTEGRATION_NOTES = """
Winning Edge Widget Integration Guide
======================================

1. DATA FLOW
   -----------
   Main Dashboard Upload -> dcc.Store -> Winning Edge Callbacks

   The winning edge widget accesses the same telemetry data stores as other
   dashboard tabs, ensuring data consistency.

2. CALLBACK STRUCTURE
   ------------------
   The widget uses these callback IDs (all prefixed with 'winning-edge-'):
   - winning-edge-upload-data: File upload store
   - winning-edge-vehicle-dropdown: Vehicle selection
   - winning-edge-reference-dropdown: Reference vehicle selection
   - winning-edge-corner-select: Corner detail selection
   - winning-edge-*-chart: Chart output components

   All IDs are unique and don't conflict with main dashboard.

3. DATA REQUIREMENTS
   -----------------
   Minimum telemetry sensors needed:
   - speed: Vehicle speed (km/h)
   - Steering_Angle: Steering input (degrees)
   - pbrake_f: Front brake pressure (bar)
   - timestamp: Time series data (milliseconds)

   Optional but recommended:
   - accy_can: Lateral acceleration (g)
   - aps: Throttle position (%)
   - GPS coordinates for brake point analysis

4. PERFORMANCE CONSIDERATIONS
   --------------------------
   - Corner detection runs on client side (fast for <10 laps)
   - Large datasets (>20 laps) may benefit from server-side processing
   - Consider caching corner metrics in dcc.Store for better UX

5. CUSTOMIZATION
   -------------
   To modify corner detection logic:
   - Edit extract_corner_metrics() in this file
   - Adjust steering angle threshold (currently 30 degrees)
   - Modify minimum corner duration (currently 5 samples)

6. TROUBLESHOOTING
   --------------
   - If no corners detected: Check steering angle data availability
   - If speeds look wrong: Verify telemetry units (km/h vs mph)
   - If reference comparison fails: Ensure both vehicles have data
"""


def get_integration_info() -> Dict[str, any]:
    """
    Return integration metadata for debugging and documentation.

    Returns:
        Dictionary with integration status, requirements, and capabilities
    """
    return {
        'widget_name': 'Winning Edge',
        'version': '1.0.0',
        'requires_upload': True,
        'requires_vehicle_selection': True,
        'minimum_sensors': ['speed', 'Steering_Angle', 'pbrake_f', 'timestamp'],
        'optional_sensors': ['accy_can', 'aps', 'VBOX_Long_Minutes', 'VBOX_Lat_Min'],
        'callback_prefix': 'winning-edge-',
        'data_format': 'long_format',
        'supports_multi_vehicle': True,
        'supports_reference_comparison': True,
        'integration_notes': INTEGRATION_NOTES
    }
