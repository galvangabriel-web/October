"""
Winning Edge Widget Real-Time Integration Module
==================================================

This module provides real-time integration between the Winning Edge data processor
and the dashboard callbacks, enabling automatic telemetry processing and
visualization updates.

Features:
- Connects WinningEdgeDataProcessor to dashboard data stores
- Processes uploaded telemetry automatically
- Updates visualizations with real-time data
- Manages caching and incremental updates
- Handles missing/incomplete data gracefully

Integration Pattern:
    Upload telemetry -> Store in dcc.Store -> Process with processor ->
    Update visualizations -> Cache results -> Incremental updates on new laps

Usage in app.py:
    from src.dashboard.winning_edge_integration_realtime import (
        create_winning_edge_data_store,
        create_winning_edge_processing_callbacks
    )

    # Add data store to layout
    layout = html.Div([
        create_winning_edge_data_store(),
        # ... other components
    ])

    # Register processing callbacks
    create_winning_edge_processing_callbacks(app)

Author: GR Cup Racing Analytics Team
Version: 1.0.0
License: GR Cup 2025 Hackathon
"""

from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from datetime import datetime

# Import data processor
try:
    from src.dashboard.winning_edge_data_processor import (
        WinningEdgeDataProcessor,
        CornerMetrics,
        PerformanceGap,
        ImprovementTarget,
        SessionProgress,
        CornerPhase,
        format_for_winning_edge_widget,
        convert_metrics_to_dict,
        convert_gaps_to_dict,
        convert_targets_to_dict
    )
    PROCESSOR_AVAILABLE = True
except ImportError as e:
    PROCESSOR_AVAILABLE = False
    logging.warning(f"WinningEdgeDataProcessor not available: {e}")

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STORE COMPONENTS
# ============================================================================

def create_winning_edge_data_store():
    """
    Create dcc.Store components for Winning Edge data caching.

    Returns:
        html.Div containing all necessary data stores
    """
    return html.Div([
        # Raw processed corner metrics
        dcc.Store(id='winning-edge-corner-metrics-store', storage_type='memory'),

        # Performance gaps
        dcc.Store(id='winning-edge-gaps-store', storage_type='memory'),

        # Improvement targets
        dcc.Store(id='winning-edge-targets-store', storage_type='memory'),

        # Session progress tracking
        dcc.Store(id='winning-edge-session-progress-store', storage_type='local'),

        # Processing status
        dcc.Store(id='winning-edge-processing-status', storage_type='memory', data={
            'last_processed': None,
            'vehicle_number': None,
            'num_laps': 0,
            'num_corners': 0,
            'status': 'ready'
        }),

        # Best lap cache
        dcc.Store(id='winning-edge-best-lap-store', storage_type='memory')
    ], style={'display': 'none'})


# ============================================================================
# DATA PROCESSING CALLBACKS
# ============================================================================

def create_winning_edge_processing_callbacks(app):
    """
    Register callbacks for automatic telemetry processing.

    Args:
        app: Dash app instance

    Callbacks created:
    1. Process telemetry when new data uploaded
    2. Update corner metrics store
    3. Calculate performance gaps
    4. Generate improvement targets
    5. Track session progress
    """

    if not PROCESSOR_AVAILABLE:
        logger.warning("Winning Edge processor not available - callbacks will use sample data")
        return

    # Initialize processor (singleton pattern)
    processor = WinningEdgeDataProcessor()

    # ========================================================================
    # CALLBACK 1: Process uploaded telemetry
    # ========================================================================

    @app.callback(
        Output('winning-edge-corner-metrics-store', 'data'),
        Output('winning-edge-processing-status', 'data'),
        Input('upload-data', 'data'),  # Main upload store
        State('winning-edge-processing-status', 'data')
    )
    def process_uploaded_telemetry(upload_data, current_status):
        """
        Process uploaded telemetry to extract corner metrics.

        Triggered when new telemetry is uploaded to main dashboard.
        """
        if not upload_data:
            logger.info("No upload data - skipping processing")
            return None, current_status

        try:
            # Convert upload data to DataFrame
            telemetry_df = pd.DataFrame(upload_data)

            if len(telemetry_df) == 0:
                logger.warning("Empty telemetry data")
                return None, current_status

            # Get vehicle number (use first vehicle if multiple)
            vehicle_numbers = telemetry_df['vehicle_number'].unique()
            if len(vehicle_numbers) == 0:
                logger.warning("No vehicle numbers in data")
                return None, current_status

            vehicle_number = int(vehicle_numbers[0])

            logger.info(f"Processing telemetry for vehicle {vehicle_number}")

            # Process telemetry
            corner_metrics = processor.process_telemetry_for_corners(
                telemetry_df=telemetry_df,
                vehicle_number=vehicle_number,
                use_cache=True
            )

            if not corner_metrics:
                logger.warning("No corners detected in telemetry")
                return None, {
                    'last_processed': datetime.now().isoformat(),
                    'vehicle_number': vehicle_number,
                    'num_laps': 0,
                    'num_corners': 0,
                    'status': 'no_corners_detected'
                }

            # Convert to JSON-serializable format
            metrics_data = convert_metrics_to_dict(corner_metrics)

            # Update status
            num_laps = len(set(c.lap_number for c in corner_metrics))
            num_corners = len(set(c.corner_id for c in corner_metrics))

            status = {
                'last_processed': datetime.now().isoformat(),
                'vehicle_number': vehicle_number,
                'num_laps': num_laps,
                'num_corners': num_corners,
                'status': 'success'
            }

            logger.info(f"Successfully processed {num_laps} laps, {num_corners} corners")

            return metrics_data, status

        except Exception as e:
            logger.error(f"Error processing telemetry: {e}", exc_info=True)
            return None, {
                'last_processed': datetime.now().isoformat(),
                'vehicle_number': None,
                'num_laps': 0,
                'num_corners': 0,
                'status': f'error: {str(e)}'
            }

    # ========================================================================
    # CALLBACK 2: Calculate performance gaps
    # ========================================================================

    @app.callback(
        Output('winning-edge-gaps-store', 'data'),
        Output('winning-edge-best-lap-store', 'data'),
        Input('winning-edge-corner-metrics-store', 'data'),
        State('winning-edge-best-lap-store', 'data')
    )
    def calculate_performance_gaps(metrics_data, current_best_lap):
        """
        Calculate gaps between current and best performance.

        Triggered when corner metrics are updated.
        """
        if not metrics_data:
            return None, current_best_lap

        try:
            # Reconstruct CornerMetrics objects
            corner_metrics = _deserialize_corner_metrics(metrics_data)

            if not corner_metrics:
                return None, current_best_lap

            # Calculate gaps
            gaps = processor.calculate_real_time_gaps(
                corner_metrics=corner_metrics,
                best_lap_number=current_best_lap  # Use cached best lap if available
            )

            if not gaps:
                logger.warning("No gaps calculated")
                return None, current_best_lap

            # Convert to JSON-serializable format
            gaps_data = convert_gaps_to_dict(gaps)

            # Store best lap number
            best_lap = min(corner_metrics, key=lambda c: c.total_duration).lap_number

            logger.info(f"Calculated gaps for {len(gaps)} corners, best lap: {best_lap}")

            return gaps_data, best_lap

        except Exception as e:
            logger.error(f"Error calculating gaps: {e}", exc_info=True)
            return None, current_best_lap

    # ========================================================================
    # CALLBACK 3: Generate improvement targets
    # ========================================================================

    @app.callback(
        Output('winning-edge-targets-store', 'data'),
        Input('winning-edge-gaps-store', 'data'),
        State('winning-edge-corner-metrics-store', 'data')
    )
    def generate_improvement_targets(gaps_data, metrics_data):
        """
        Generate specific improvement targets from gaps.

        Triggered when performance gaps are updated.
        """
        if not gaps_data or not metrics_data:
            return None

        try:
            # Reconstruct objects
            gaps = _deserialize_gaps(gaps_data)
            corner_metrics = _deserialize_corner_metrics(metrics_data)

            if not gaps or not corner_metrics:
                return None

            # Generate targets
            targets = processor.generate_improvement_targets(
                gaps=gaps,
                corner_metrics=corner_metrics,
                top_n=3  # Top 3 priority corners
            )

            if not targets:
                logger.warning("No targets generated")
                return None

            # Convert to JSON-serializable format
            targets_data = convert_targets_to_dict(targets)

            logger.info(f"Generated {len(targets)} improvement targets")

            return targets_data

        except Exception as e:
            logger.error(f"Error generating targets: {e}", exc_info=True)
            return None

    # ========================================================================
    # CALLBACK 4: Update session progress tracking
    # ========================================================================

    @app.callback(
        Output('winning-edge-session-progress-store', 'data'),
        Input('winning-edge-corner-metrics-store', 'data'),
        Input('winning-edge-gaps-store', 'data'),
        State('winning-edge-session-progress-store', 'data'),
        State('upload-data', 'data')
    )
    def update_session_progress(metrics_data, gaps_data, historical_data, upload_data):
        """
        Track improvement progress across sessions.

        Stores session-level metrics for trend analysis.
        """
        if not metrics_data or not upload_data:
            return historical_data

        try:
            # Reconstruct corner metrics
            corner_metrics = _deserialize_corner_metrics(metrics_data)

            if not corner_metrics:
                return historical_data

            # Calculate session metrics
            telemetry_df = pd.DataFrame(upload_data)
            vehicle_number = int(corner_metrics[0].vehicle_number)
            track_name = telemetry_df['track'].iloc[0] if 'track' in telemetry_df.columns else 'unknown'

            # Calculate best lap time
            lap_times = {}
            for c in corner_metrics:
                if c.lap_number not in lap_times:
                    lap_times[c.lap_number] = 0
                lap_times[c.lap_number] += c.total_duration

            best_lap_time = min(lap_times.values()) if lap_times else 0
            avg_lap_time = np.mean(list(lap_times.values())) if lap_times else 0

            # Calculate consistency (std dev of lap times)
            consistency_score = np.std(list(lap_times.values())) if len(lap_times) > 1 else 0

            # Calculate corner improvements
            gaps = _deserialize_gaps(gaps_data) if gaps_data else []
            corner_improvements = {
                gap.corner_id: -gap.total_time_gap  # Negative gap = improvement
                for gap in gaps
            }

            # Create session progress
            current_session = {
                'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'session_date': datetime.now().isoformat(),
                'vehicle_number': vehicle_number,
                'track_name': track_name,
                'best_lap_time': best_lap_time,
                'avg_lap_time': avg_lap_time,
                'consistency_score': consistency_score,
                'corner_improvements': corner_improvements
            }

            # Update historical data
            if historical_data is None:
                historical_data = {'sessions': []}

            historical_data['sessions'].append(current_session)

            # Keep only last 10 sessions
            if len(historical_data['sessions']) > 10:
                historical_data['sessions'] = historical_data['sessions'][-10:]

            logger.info(f"Updated session progress: {len(historical_data['sessions'])} sessions tracked")

            return historical_data

        except Exception as e:
            logger.error(f"Error updating session progress: {e}", exc_info=True)
            return historical_data


# ============================================================================
# HELPER FUNCTIONS FOR DATA CONVERSION
# ============================================================================

def get_processed_data_for_visualization(
    metrics_store: Optional[List[Dict]],
    gaps_store: Optional[List[Dict]],
    targets_store: Optional[List[Dict]]
) -> Dict[str, Any]:
    """
    Helper function to get processed data in format ready for visualizations.

    Use this in widget callbacks to convert stored data to visualization format.

    Args:
        metrics_store: Corner metrics from dcc.Store
        gaps_store: Performance gaps from dcc.Store
        targets_store: Improvement targets from dcc.Store

    Returns:
        Dictionary with formatted data for all Winning Edge visualizations

    Example:
        @app.callback(
            Output('winning-edge-heatmap', 'figure'),
            Input('winning-edge-corner-metrics-store', 'data'),
            Input('winning-edge-gaps-store', 'data')
        )
        def update_heatmap(metrics_data, gaps_data):
            data = get_processed_data_for_visualization(
                metrics_store=metrics_data,
                gaps_store=gaps_data,
                targets_store=None
            )
            return create_time_loss_heatmap(data['corner_data'])
    """
    # Convert to objects
    corner_metrics = _deserialize_corner_metrics(metrics_store) if metrics_store else []
    gaps = _deserialize_gaps(gaps_store) if gaps_store else []
    targets = _deserialize_targets(targets_store) if targets_store else []

    # Format for visualizations
    if corner_metrics and gaps and targets:
        return format_for_winning_edge_widget(corner_metrics, gaps, targets)
    else:
        # Return empty structure
        return {
            'corner_metrics': [],
            'performance_gaps': [],
            'improvement_targets': [],
            'metadata': {
                'total_corners': 0,
                'total_laps': 0,
                'best_lap': None,
                'total_time_to_gain': 0,
                'top_priority_corner': None,
                'timestamp': datetime.now().isoformat()
            }
        }


def extract_corner_data_for_heatmap(gaps_data: Optional[List[Dict]]) -> Dict[str, Dict]:
    """
    Extract corner data in format needed for time loss heatmap.

    Args:
        gaps_data: Performance gaps from store

    Returns:
        Dictionary mapping corner names to time loss data
    """
    if not gaps_data:
        return {}

    gaps = _deserialize_gaps(gaps_data)

    corner_data = {}
    for gap in gaps:
        corner_data[gap.corner_name] = {
            'time_loss': gap.total_time_gap,
            'pct_of_total': gap.pct_of_total_loss
        }

    return corner_data


def extract_speed_data_for_spider(
    metrics_data: Optional[List[Dict]],
    best_lap_number: Optional[int] = None
) -> Tuple[List[str], List[float], List[float]]:
    """
    Extract speed data for spider chart visualization.

    Args:
        metrics_data: Corner metrics from store
        best_lap_number: Best lap to use as benchmark

    Returns:
        Tuple of (corner_names, current_speeds, best_speeds)
    """
    if not metrics_data:
        return [], [], []

    corner_metrics = _deserialize_corner_metrics(metrics_data)

    if not corner_metrics:
        return [], [], []

    # Auto-detect best lap if not provided
    if best_lap_number is None:
        lap_times = {}
        for c in corner_metrics:
            if c.lap_number not in lap_times:
                lap_times[c.lap_number] = 0
            lap_times[c.lap_number] += c.total_duration
        best_lap_number = min(lap_times.items(), key=lambda x: x[1])[0]

    # Get speeds by corner
    corners_by_id = {}
    for c in corner_metrics:
        if c.corner_id not in corners_by_id:
            corners_by_id[c.corner_id] = {'best': [], 'other': []}

        if c.lap_number == best_lap_number:
            corners_by_id[c.corner_id]['best'].append(c.exit_speed)
        else:
            corners_by_id[c.corner_id]['other'].append(c.exit_speed)

    # Build lists
    corner_names = []
    current_speeds = []
    best_speeds = []

    for corner_id in sorted(corners_by_id.keys()):
        data = corners_by_id[corner_id]

        if data['best'] and data['other']:
            corner_names.append(f"Turn {corner_id + 1}")
            best_speeds.append(np.mean(data['best']))
            current_speeds.append(np.mean(data['other']))

    return corner_names, current_speeds, best_speeds


def extract_action_card_data(
    targets_data: Optional[List[Dict]],
    corner_name: str
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Extract data for action card visualization.

    Args:
        targets_data: Improvement targets from store
        corner_name: Name of corner (e.g., 'Turn 6')

    Returns:
        Tuple of (current_metrics, target_metrics) dictionaries
    """
    if not targets_data:
        return {}, {}

    targets = _deserialize_targets(targets_data)

    # Find target for this corner
    target = next((t for t in targets if t.corner_name == corner_name), None)

    if not target:
        return {}, {}

    current_metrics = {
        'Brake Point (m)': target.current_brake_point,
        'Brake Pressure (%)': (target.current_brake_pressure / 153) * 100,  # Convert to %
        'Throttle Timing (%)': target.current_throttle_point,
        'Exit Speed (km/h)': target.current_exit_speed
    }

    target_metrics = {
        'Brake Point (m)': target.target_brake_point,
        'Brake Pressure (%)': (target.target_brake_pressure / 153) * 100,  # Convert to %
        'Throttle Timing (%)': target.target_throttle_point,
        'Exit Speed (km/h)': target.target_exit_speed
    }

    return current_metrics, target_metrics


# ============================================================================
# INTERNAL DESERIALIZATION HELPERS
# ============================================================================

def _deserialize_corner_metrics(metrics_data: List[Dict]) -> List[CornerMetrics]:
    """Convert JSON dicts back to CornerMetrics objects."""
    metrics_list = []

    for data in metrics_data:
        # Reconstruct phases
        brake_phase = _dict_to_corner_phase(data.get('brake_phase')) if data.get('brake_phase') else None
        apex_phase = _dict_to_corner_phase(data.get('apex_phase')) if data.get('apex_phase') else None
        exit_phase = _dict_to_corner_phase(data.get('exit_phase')) if data.get('exit_phase') else None

        # Create CornerMetrics
        metrics = CornerMetrics(
            corner_id=data['corner_id'],
            corner_name=data['corner_name'],
            lap_number=data['lap_number'],
            vehicle_number=data['vehicle_number'],
            brake_phase=brake_phase,
            apex_phase=apex_phase,
            exit_phase=exit_phase,
            total_duration=data['total_duration'],
            entry_speed=data['entry_speed'],
            apex_speed=data['apex_speed'],
            exit_speed=data['exit_speed'],
            time_to_apex=data['time_to_apex'],
            brake_to_throttle_transition=data['brake_to_throttle_transition'],
            corner_g_force_max=data.get('corner_g_force_max'),
            consistency_score=data.get('consistency_score')
        )

        metrics_list.append(metrics)

    return metrics_list


def _deserialize_gaps(gaps_data: List[Dict]) -> List[PerformanceGap]:
    """Convert JSON dicts back to PerformanceGap objects."""
    return [PerformanceGap(**data) for data in gaps_data]


def _deserialize_targets(targets_data: List[Dict]) -> List[ImprovementTarget]:
    """Convert JSON dicts back to ImprovementTarget objects."""
    return [ImprovementTarget(**data) for data in targets_data]


def _dict_to_corner_phase(data: Dict) -> Optional[CornerPhase]:
    """Convert dict to CornerPhase dataclass."""
    if not data:
        return None

    return CornerPhase(**data)


# ============================================================================
# STATUS MONITORING
# ============================================================================

def create_winning_edge_status_indicator():
    """
    Create status indicator component for debugging.

    Shows processing status in development mode.
    """
    return html.Div([
        html.Div(id='winning-edge-status-indicator', style={
            'position': 'fixed',
            'bottom': '10px',
            'right': '10px',
            'padding': '10px',
            'backgroundColor': '#f8f9fa',
            'border': '1px solid #dee2e6',
            'borderRadius': '5px',
            'fontSize': '11px',
            'zIndex': 9999,
            'maxWidth': '300px'
        })
    ])


def create_winning_edge_status_callback(app):
    """
    Create callback to update status indicator.

    For debugging and development only.
    """

    @app.callback(
        Output('winning-edge-status-indicator', 'children'),
        Input('winning-edge-processing-status', 'data')
    )
    def update_status_indicator(status_data):
        """Update status indicator with current processing state."""
        if not status_data:
            return "Winning Edge: Waiting for data..."

        status = status_data.get('status', 'unknown')
        vehicle = status_data.get('vehicle_number', 'N/A')
        laps = status_data.get('num_laps', 0)
        corners = status_data.get('num_corners', 0)
        last_processed = status_data.get('last_processed', 'Never')

        color_map = {
            'success': '#28a745',
            'ready': '#17a2b8',
            'no_corners_detected': '#ffc107',
            'error': '#dc3545'
        }

        color = color_map.get(status if not status.startswith('error') else 'error', '#6c757d')

        return html.Div([
            html.Div([
                html.Strong("Winning Edge Status"),
                html.Div(style={
                    'width': '10px',
                    'height': '10px',
                    'borderRadius': '50%',
                    'backgroundColor': color,
                    'display': 'inline-block',
                    'marginLeft': '5px'
                })
            ]),
            html.Div(f"Vehicle: {vehicle}"),
            html.Div(f"Laps: {laps} | Corners: {corners}"),
            html.Div(f"Status: {status}", style={'fontSize': '10px', 'color': color}),
            html.Div(f"Updated: {last_processed[:19] if last_processed != 'Never' else 'Never'}", style={'fontSize': '9px', 'color': '#6c757d'})
        ])
