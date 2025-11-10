"""
Corner Analysis Widget - Sprint 2 Task 3
=========================================

Displays detailed corner-by-corner analysis with:
- Entry/apex/exit speed comparisons
- Brake pressure analysis
- Specific coaching per corner
- Time gains/losses per corner
- Category-specific metrics (9 professional categories)
"""

import dash_bootstrap_components as dbc
from dash import html
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CATEGORY-TO-METRICS MAPPING
# ============================================================================

CATEGORY_METRICS_MAP = {
    'speed': {
        'display_name': 'Speed & Acceleration',
        'primary_metrics': ['entry_speed', 'apex_speed', 'exit_speed'],
        'secondary_metrics': ['speed_delta', 'acceleration'],
        'description': 'Entry/Apex/Exit speeds and acceleration through corners'
    },
    'braking': {
        'display_name': 'Braking Performance',
        'primary_metrics': ['brake_pressure_avg', 'brake_pressure_max', 'brake_point'],
        'secondary_metrics': ['deceleration', 'braking_duration', 'trail_braking'],
        'description': 'Brake pressure, timing, and trail braking analysis'
    },
    'cornering': {
        'display_name': 'Cornering Dynamics',
        'primary_metrics': ['lateral_g_max', 'lateral_g_avg', 'corner_speed'],
        'secondary_metrics': ['g_force_consistency', 'turn_performance'],
        'description': 'Lateral G-forces and cornering performance'
    },
    'throttle': {
        'display_name': 'Throttle Management',
        'primary_metrics': ['throttle_application', 'full_throttle_pct', 'throttle_modulation'],
        'secondary_metrics': ['throttle_smoothness', 'partial_throttle_time'],
        'description': 'Throttle application timing and modulation'
    },
    'steering': {
        'display_name': 'Steering Control',
        'primary_metrics': ['steering_angle_max', 'steering_smoothness', 'steering_corrections'],
        'secondary_metrics': ['turn_in_rate', 'steering_consistency'],
        'description': 'Steering input smoothness and precision'
    },
    'powertrain': {
        'display_name': 'Powertrain & Gear Management',
        'primary_metrics': ['gear_used', 'rpm_avg', 'rpm_max'],
        'secondary_metrics': ['shift_timing', 'gear_optimization'],
        'description': 'Gear selection, RPM usage, and shift timing'
    },
    'composite': {
        'display_name': 'Composite Performance',
        'primary_metrics': ['overall_efficiency', 'time_delta', 'performance_index'],
        'secondary_metrics': ['consistency_score', 'optimization_potential'],
        'description': 'Combined performance metrics and optimization opportunities'
    },
    'lap_seg': {
        'display_name': 'Lap Segmentation & Timing',
        'primary_metrics': ['sector_time', 'time_gain_loss', 'consistency'],
        'secondary_metrics': ['optimal_line', 'lap_to_lap_delta'],
        'description': 'Sector timing and lap-to-lap consistency'
    },
    'uncategorized': {
        'display_name': 'Other Metrics',
        'primary_metrics': ['time_delta', 'performance_index', 'overall_efficiency'],
        'secondary_metrics': ['consistency_score', 'optimization_potential'],
        'description': 'Additional performance metrics and miscellaneous data'
    },
}


def get_category_metrics(category_id: str) -> Dict:
    """Get metrics configuration for a specific category"""
    return CATEGORY_METRICS_MAP.get(category_id, CATEGORY_METRICS_MAP['speed'])


def _build_metric_rows(corner: Dict, category_id: str) -> List:
    """
    Build table rows for category-specific metrics

    Args:
        corner: Corner analysis dictionary
        category_id: Category ID

    Returns:
        List of html.Tr elements
    """
    # Metric display configurations
    METRIC_DISPLAY = {
        # Speed & Acceleration
        'entry_speed': {'label': 'Entry Speed', 'icon': 'fa-arrow-right', 'color': '#2ecc71', 'unit': 'km/h', 'decimals': 1},
        'apex_speed': {'label': 'Apex Speed', 'icon': 'fa-bullseye', 'color': '#e74c3c', 'unit': 'km/h', 'decimals': 1},
        'exit_speed': {'label': 'Exit Speed', 'icon': 'fa-arrow-left', 'color': '#3498db', 'unit': 'km/h', 'decimals': 1},
        'speed_delta': {'label': 'Speed Delta', 'icon': 'fa-chart-line', 'color': '#9b59b6', 'unit': 'km/h', 'decimals': 1},
        'acceleration': {'label': 'Acceleration', 'icon': 'fa-rocket', 'color': '#2ecc71', 'unit': 'g', 'decimals': 2},

        # Braking Performance
        'brake_pressure_avg': {'label': 'Brake Pressure (Avg)', 'icon': 'fa-stop-circle', 'color': '#e74c3c', 'unit': 'bar', 'decimals': 1},
        'brake_pressure_max': {'label': 'Brake Pressure (Max)', 'icon': 'fa-stop-circle', 'color': '#c0392b', 'unit': 'bar', 'decimals': 1},
        'brake_point': {'label': 'Brake Point', 'icon': 'fa-map-marker-alt', 'color': '#e67e22', 'unit': 'm', 'decimals': 0},
        'deceleration': {'label': 'Deceleration', 'icon': 'fa-arrow-down', 'color': '#e74c3c', 'unit': 'g', 'decimals': 2},
        'braking_duration': {'label': 'Braking Duration', 'icon': 'fa-clock', 'color': '#f39c12', 'unit': 's', 'decimals': 2},
        'trail_braking': {'label': 'Trail Braking', 'icon': 'fa-wave-square', 'color': '#9b59b6', 'unit': '%', 'decimals': 1},

        # Cornering Dynamics
        'lateral_g_max': {'label': 'Lateral G (Max)', 'icon': 'fa-sync-alt', 'color': '#9b59b6', 'unit': 'g', 'decimals': 2},
        'lateral_g_avg': {'label': 'Lateral G (Avg)', 'icon': 'fa-sync', 'color': '#8e44ad', 'unit': 'g', 'decimals': 2},
        'corner_speed': {'label': 'Corner Speed', 'icon': 'fa-tachometer-alt', 'color': '#3498db', 'unit': 'km/h', 'decimals': 1},
        'g_force_consistency': {'label': 'G-Force Consistency', 'icon': 'fa-balance-scale', 'color': '#16a085', 'unit': '%', 'decimals': 1},
        'turn_performance': {'label': 'Turn Performance', 'icon': 'fa-trophy', 'color': '#f39c12', 'unit': '%', 'decimals': 1},

        # Throttle Management
        'throttle_application': {'label': 'Throttle Application', 'icon': 'fa-bolt', 'color': '#2ecc71', 'unit': '%', 'decimals': 1},
        'full_throttle_pct': {'label': 'Full Throttle %', 'icon': 'fa-fire', 'color': '#e74c3c', 'unit': '%', 'decimals': 1},
        'throttle_modulation': {'label': 'Throttle Modulation', 'icon': 'fa-sliders-h', 'color': '#3498db', 'unit': '%', 'decimals': 1},
        'throttle_smoothness': {'label': 'Throttle Smoothness', 'icon': 'fa-wave-square', 'color': '#16a085', 'unit': '%', 'decimals': 1},
        'partial_throttle_time': {'label': 'Partial Throttle Time', 'icon': 'fa-clock', 'color': '#f39c12', 'unit': 's', 'decimals': 2},

        # Steering Control
        'steering_angle_max': {'label': 'Steering Angle (Max)', 'icon': 'fa-life-ring', 'color': '#f39c12', 'unit': '°', 'decimals': 1},
        'steering_smoothness': {'label': 'Steering Smoothness', 'icon': 'fa-wave-square', 'color': '#16a085', 'unit': '%', 'decimals': 1},
        'steering_corrections': {'label': 'Steering Corrections', 'icon': 'fa-redo', 'color': '#e74c3c', 'unit': '', 'decimals': 0},
        'turn_in_rate': {'label': 'Turn-In Rate', 'icon': 'fa-angle-right', 'color': '#3498db', 'unit': '°/s', 'decimals': 1},
        'steering_consistency': {'label': 'Steering Consistency', 'icon': 'fa-balance-scale', 'color': '#9b59b6', 'unit': '%', 'decimals': 1},

        # Powertrain & Gear Management
        'gear_used': {'label': 'Gear Used', 'icon': 'fa-cog', 'color': '#34495e', 'unit': '', 'decimals': 0},
        'rpm_avg': {'label': 'RPM (Avg)', 'icon': 'fa-tachometer-alt', 'color': '#3498db', 'unit': 'rpm', 'decimals': 0},
        'rpm_max': {'label': 'RPM (Max)', 'icon': 'fa-tachometer-alt', 'color': '#e74c3c', 'unit': 'rpm', 'decimals': 0},
        'shift_timing': {'label': 'Shift Timing', 'icon': 'fa-clock', 'color': '#f39c12', 'unit': 's', 'decimals': 2},
        'gear_optimization': {'label': 'Gear Optimization', 'icon': 'fa-trophy', 'color': '#2ecc71', 'unit': '%', 'decimals': 1},

        # Composite Performance
        'overall_efficiency': {'label': 'Overall Efficiency', 'icon': 'fa-layer-group', 'color': '#16a085', 'unit': '%', 'decimals': 1},
        'time_delta': {'label': 'Time Delta', 'icon': 'fa-clock', 'color': '#e74c3c', 'unit': 's', 'decimals': 3},
        'performance_index': {'label': 'Performance Index', 'icon': 'fa-chart-line', 'color': '#3498db', 'unit': '', 'decimals': 2},
        'consistency_score': {'label': 'Consistency Score', 'icon': 'fa-balance-scale', 'color': '#9b59b6', 'unit': '%', 'decimals': 1},
        'optimization_potential': {'label': 'Optimization Potential', 'icon': 'fa-lightbulb', 'color': '#f39c12', 'unit': '%', 'decimals': 1},

        # Lap Segmentation & Timing
        'sector_time': {'label': 'Sector Time', 'icon': 'fa-flag-checkered', 'color': '#c0392b', 'unit': 's', 'decimals': 3},
        'time_gain_loss': {'label': 'Time Gain/Loss', 'icon': 'fa-chart-line', 'color': '#e74c3c', 'unit': 's', 'decimals': 3},
        'consistency': {'label': 'Consistency', 'icon': 'fa-balance-scale', 'color': '#16a085', 'unit': '%', 'decimals': 1},
        'optimal_line': {'label': 'Optimal Line', 'icon': 'fa-route', 'color': '#3498db', 'unit': '%', 'decimals': 1},
        'lap_to_lap_delta': {'label': 'Lap-to-Lap Delta', 'icon': 'fa-random', 'color': '#9b59b6', 'unit': 's', 'decimals': 3},
    }

    # Get metrics for this category
    category_config = get_category_metrics(category_id)
    primary_metrics = category_config.get('primary_metrics', [])

    rows = []
    for metric_key in primary_metrics:
        if metric_key not in METRIC_DISPLAY:
            continue  # Skip if display config not found

        display_config = METRIC_DISPLAY[metric_key]

        # Get values from corner data (with defaults)
        # Try different key patterns to handle various data formats
        # Priority: direct key > _avg suffix > default to 0
        if metric_key in corner:
            # Direct key exists (e.g., 'lateral_g_max')
            avg_value = corner[metric_key]
            best_value = corner.get(f'{metric_key}_best', avg_value * 1.1)
        elif f'{metric_key}_avg' in corner:
            # Has _avg suffix (e.g., 'entry_speed_avg')
            avg_value = corner[f'{metric_key}_avg']
            best_value = corner.get(f'{metric_key}_best', avg_value * 1.1)
        else:
            # Not found - use defaults
            avg_value = 0
            best_value = 0

        # Format values
        decimals = display_config['decimals']
        unit = display_config['unit']

        if decimals == 0:
            avg_str = f"{avg_value:.0f} {unit}".strip()
            best_str = f"{best_value:.0f} {unit}".strip()
            gap_str = f"+{best_value - avg_value:.0f} {unit}".strip()
        else:
            avg_str = f"{avg_value:.{decimals}f} {unit}".strip()
            best_str = f"{best_value:.{decimals}f} {unit}".strip()
            gap_str = f"+{best_value - avg_value:.{decimals}f} {unit}".strip()

        # Build row
        rows.append(
            html.Tr([
                html.Td([
                    html.I(className=f"fas {display_config['icon']} me-2", style={'color': display_config['color']}),
                    display_config['label']
                ]),
                html.Td(avg_str, style={'textAlign': 'center'}),
                html.Td(best_str, style={'textAlign': 'center', 'fontWeight': 'bold'}),
                html.Td(
                    gap_str,
                    style={
                        'textAlign': 'center',
                        'color': '#2ecc71' if best_value > avg_value else '#95a5a6',
                        'fontWeight': 'bold'
                    }
                )
            ])
        )

    # If no rows generated, show default message
    if not rows:
        rows.append(
            html.Tr([
                html.Td([
                    html.I(className="fas fa-info-circle me-2", style={'color': '#95a5a6'}),
                    "Data not available for this category"
                ], colSpan=4, style={'textAlign': 'center', 'color': '#95a5a6'})
            ])
        )

    return rows


def calculate_corner_phases(corner_data: Dict) -> Dict:
    """
    Calculate time spent in each corner phase: Brake, Apex, Exit

    Args:
        corner_data: Dictionary with corner metrics

    Returns:
        Dictionary with phase breakdown analysis
    """
    # Extract time in corner (default to 5.0 seconds if not available)
    total_corner_time = corner_data.get('time_in_corner', 5.0)

    # Estimate brake duration from data (or use 35% of corner time as default)
    brake_duration = corner_data.get('braking_duration', total_corner_time * 0.35)

    # Apex phase typically 15-25% of corner time
    apex_duration = total_corner_time * 0.20

    # Exit phase is remaining time
    exit_duration = total_corner_time - brake_duration - apex_duration

    # Ensure no negative durations
    if exit_duration < 0:
        # Adjust if brake duration is too long
        brake_duration = total_corner_time * 0.40
        apex_duration = total_corner_time * 0.20
        exit_duration = total_corner_time * 0.40

    # Calculate optimal distribution (ideal racing line)
    optimal_brake = total_corner_time * 0.30  # 30% braking
    optimal_apex = total_corner_time * 0.20   # 20% apex
    optimal_exit = total_corner_time * 0.50   # 50% exit (priority!)

    # Calculate deltas
    brake_delta = brake_duration - optimal_brake
    apex_delta = apex_duration - optimal_apex
    exit_delta = exit_duration - optimal_exit

    return {
        "actual": {
            "brake": brake_duration,
            "apex": apex_duration,
            "exit": exit_duration
        },
        "optimal": {
            "brake": optimal_brake,
            "apex": optimal_apex,
            "exit": optimal_exit
        },
        "delta": {
            "brake": brake_delta,
            "apex": apex_delta,
            "exit": exit_delta
        },
        "total_time": total_corner_time
    }


def calculate_session_best(all_corners: List[Dict]) -> Dict:
    """
    Calculate session best lap time and corner-by-corner best times

    Args:
        all_corners: List of all corner data dictionaries from multiple laps

    Returns:
        Dictionary with session best metrics
    """
    if not all_corners or len(all_corners) == 0:
        return {}

    # Find fastest overall lap time
    lap_times = [c.get('lap_time', 999.0) for c in all_corners if 'lap_time' in c]
    fastest_lap = min(lap_times) if lap_times else 0.0

    # Group corners by corner number
    corners_by_number = {}
    for corner in all_corners:
        corner_num = corner.get('corner_number', corner.get('corner_name', 'Unknown'))
        if corner_num not in corners_by_number:
            corners_by_number[corner_num] = []
        corners_by_number[corner_num].append(corner)

    # Find best time for each corner
    best_corner_times = {}
    for corner_num, corners in corners_by_number.items():
        corner_times = [c.get('time_in_corner', 999.0) for c in corners if 'time_in_corner' in c]
        best_corner_times[corner_num] = min(corner_times) if corner_times else 0.0

    return {
        'fastest_lap': fastest_lap,
        'best_corners': best_corner_times
    }


def create_session_best_badge(corner: Dict, session_best: Dict) -> html.Span:
    """
    Create session best badge showing delta to best time for this corner

    Args:
        corner: Current corner data
        session_best: Output from calculate_session_best()

    Returns:
        Badge component showing session best status
    """
    if not session_best:
        return html.Span()  # Empty if no session data

    corner_num = corner.get('corner_number', corner.get('corner_name', 'Unknown'))
    current_time = corner.get('time_in_corner', 0.0)
    best_time = session_best.get('best_corners', {}).get(corner_num, 0.0)

    if best_time == 0.0 or current_time == 0.0:
        return html.Span()  # No data

    delta_to_best = current_time - best_time

    # Determine badge color and text
    if abs(delta_to_best) < 0.05:  # Within 50ms of session best
        badge_color = 'success'
        badge_text = "SESSION BEST"
        icon = "fas fa-trophy"
    elif delta_to_best < 0.15:  # Within 150ms
        badge_color = 'warning'
        badge_text = f"+{delta_to_best:.3f}s"
        icon = "fas fa-clock"
    else:  # More than 150ms off
        badge_color = 'danger'
        badge_text = f"+{delta_to_best:.3f}s"
        icon = "fas fa-chart-line"

    return dbc.Badge([
        html.I(className=f"{icon} me-1"),
        badge_text
    ], color=badge_color, className="ms-2", style={'fontSize': '0.7rem'})


def create_phase_breakdown_visual(phases: Dict) -> html.Div:
    """
    Create visual phase breakdown chart with Brake/Apex/Exit phases

    Args:
        phases: Output from calculate_corner_phases()

    Returns:
        Dash Div component with phase visualization
    """
    actual = phases["actual"]
    optimal = phases["optimal"]
    delta = phases["delta"]
    total_time = phases["total_time"]

    # Calculate percentages for visualization
    actual_total = sum(actual.values())
    brake_pct = (actual["brake"] / actual_total) * 100
    apex_pct = (actual["apex"] / actual_total) * 100
    exit_pct = (actual["exit"] / actual_total) * 100

    optimal_total = sum(optimal.values())
    optimal_brake_pct = (optimal["brake"] / optimal_total) * 100
    optimal_apex_pct = (optimal["apex"] / optimal_total) * 100
    optimal_exit_pct = (optimal["exit"] / optimal_total) * 100

    return html.Div([
        html.H6([
            html.I(className="fas fa-stopwatch me-2", style={'color': '#9b59b6'}),
            "Corner Phase Breakdown"
        ], className="mb-3", style={'fontSize': '0.9rem'}),

        # Your actual phases
        html.Div([
            html.Small("Your Phases:", className="text-muted d-block mb-1", style={'fontSize': '0.8rem'}),
            html.Div([
                # Brake phase (RED)
                html.Div([
                    html.Div("BRAKE", style={
                        'fontSize': '0.7rem',
                        'fontWeight': 'bold',
                        'color': 'white',
                        'textAlign': 'center',
                        'padding': '4px'
                    })
                ], style={
                    'width': f'{brake_pct}%',
                    'backgroundColor': '#e74c3c',
                    'display': 'inline-block',
                    'minWidth': '60px'
                }),
                # Apex phase (YELLOW)
                html.Div([
                    html.Div("APEX", style={
                        'fontSize': '0.7rem',
                        'fontWeight': 'bold',
                        'color': 'black',
                        'textAlign': 'center',
                        'padding': '4px'
                    })
                ], style={
                    'width': f'{apex_pct}%',
                    'backgroundColor': '#f39c12',
                    'display': 'inline-block',
                    'minWidth': '50px'
                }),
                # Exit phase (GREEN)
                html.Div([
                    html.Div("EXIT", style={
                        'fontSize': '0.7rem',
                        'fontWeight': 'bold',
                        'color': 'white',
                        'textAlign': 'center',
                        'padding': '4px'
                    })
                ], style={
                    'width': f'{exit_pct}%',
                    'backgroundColor': '#27ae60',
                    'display': 'inline-block',
                    'minWidth': '50px'
                })
            ], style={'borderRadius': '4px', 'overflow': 'hidden', 'display': 'flex'}),

            # Time labels
            html.Div([
                html.Span(f"{actual['brake']:.2f}s", className="text-danger me-3", style={'fontSize': '0.75rem'}),
                html.Span(f"{actual['apex']:.2f}s", className="text-warning me-3", style={'fontSize': '0.75rem'}),
                html.Span(f"{actual['exit']:.2f}s", className="text-success", style={'fontSize': '0.75rem'}),
            ], className="mt-1")
        ], className="mb-3"),

        # Optimal phases (semi-transparent for comparison)
        html.Div([
            html.Small("Optimal:", className="text-muted d-block mb-1", style={'fontSize': '0.8rem'}),
            html.Div([
                html.Div(style={
                    'width': f'{optimal_brake_pct}%',
                    'backgroundColor': '#e74c3c',
                    'opacity': '0.4',
                    'height': '8px',
                    'display': 'inline-block'
                }),
                html.Div(style={
                    'width': f'{optimal_apex_pct}%',
                    'backgroundColor': '#f39c12',
                    'opacity': '0.4',
                    'height': '8px',
                    'display': 'inline-block'
                }),
                html.Div(style={
                    'width': f'{optimal_exit_pct}%',
                    'backgroundColor': '#27ae60',
                    'opacity': '0.4',
                    'height': '8px',
                    'display': 'inline-block'
                })
            ], style={'borderRadius': '4px', 'overflow': 'hidden', 'display': 'flex'}),

            # Optimal time labels
            html.Div([
                html.Span(f"{optimal['brake']:.2f}s", className="me-3", style={'fontSize': '0.7rem', 'color': '#95a5a6'}),
                html.Span(f"{optimal['apex']:.2f}s", className="me-3", style={'fontSize': '0.7rem', 'color': '#95a5a6'}),
                html.Span(f"{optimal['exit']:.2f}s", style={'fontSize': '0.7rem', 'color': '#95a5a6'}),
            ], className="mt-1")
        ], className="mb-3"),

        # Analysis
        html.Div([
            html.Small([
                html.I(className="fas fa-chart-line me-2", style={'color': '#3498db'}),
                html.Strong("Analysis:")
            ], className="d-block mb-2", style={'fontSize': '0.85rem'}),
            html.Ul([
                html.Li([
                    html.Strong("Braking: "),
                    f"{delta['brake']:+.2f}s ",
                    html.Span(
                        "(brake earlier and lighter)" if delta['brake'] > 0 else "(good brake efficiency)",
                        className="text-danger" if delta['brake'] > 0.1 else "text-success"
                    )
                ], style={'fontSize': '0.8rem'}, className="mb-1"),
                html.Li([
                    html.Strong("Exit: "),
                    f"{delta['exit']:+.2f}s ",
                    html.Span(
                        "(get on throttle earlier!)" if delta['exit'] < -0.1 else "(good exit speed)",
                        className="text-warning" if delta['exit'] < -0.1 else "text-success"
                    )
                ], style={'fontSize': '0.8rem'}, className="mb-1"),
            ], className="mb-0", style={'paddingLeft': '1.2rem'})
        ], className="p-2", style={'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})
    ], className="mb-3 p-3", style={
        'border': '1px solid #dee2e6',
        'borderRadius': '0.25rem',
        'backgroundColor': '#ffffff'
    })


def create_corner_analysis_content(
    corner_analyses: Optional[List[Dict]] = None,
    category_id: str = 'speed'
):
    """
    Create body content for corner analysis (without modal wrapper)

    Args:
        corner_analyses: List of corner analysis dictionaries
        category_id: Category ID to determine which metrics to display

    Returns:
        Dash component with corner analysis content
    """
    if not corner_analyses or len(corner_analyses) == 0:
        return dbc.Alert([
            html.I(className="fas fa-info-circle me-2"),
            "No corner analysis data available. Please upload telemetry data and ensure corner detection is successful."
        ], color="info")

    # Get category-specific metrics configuration
    category_config = get_category_metrics(category_id)

    # Calculate session best for all corners (Quick Win #4)
    session_best = calculate_session_best(corner_analyses)

    # Create category description banner
    description_banner = dbc.Alert([
        html.I(className="fas fa-info-circle me-2"),
        html.Strong(f"{category_config['display_name']}: "),
        category_config['description']
    ], color="info", className="mb-3")

    # Create corner cards with category-specific metrics and session best data
    corner_cards = [
        create_corner_card(corner, i+1, category_id, session_best)
        for i, corner in enumerate(corner_analyses)
    ]

    return html.Div([description_banner, html.Div(corner_cards)])


def create_corner_analysis_modal(
    is_open: bool = False,
    corner_analyses: Optional[List[Dict]] = None,
    category_name: str = "Performance"
) -> dbc.Modal:
    """
    Create modal showing corner-by-corner analysis

    DEPRECATED: Use create_corner_analysis_content() instead for Approach 1
    (fixed modal structure). This function is kept for backward compatibility.

    Args:
        is_open: Whether modal is open
        corner_analyses: List of corner analysis dictionaries
        category_name: Name of category being analyzed

    Returns:
        Dash Bootstrap Modal component
    """
    body_content = create_corner_analysis_content(corner_analyses)

    return dbc.Modal([
        dbc.ModalHeader([
            html.I(className="fas fa-map-marked-alt me-2"),
            f"Corner Analysis - {category_name}"
        ]),
        dbc.ModalBody([
            body_content
        ], style={'maxHeight': '600px', 'overflowY': 'auto'}),
        dbc.ModalFooter([
            dbc.Button(
                [html.I(className="fas fa-times me-2"), "Close"],
                id="close-corner-modal",
                color="secondary"
            )
        ])
    ], id="corner-analysis-modal", size="xl", is_open=is_open)


def create_corner_card(corner: Dict, index: int, category_id: str = 'speed', session_best: Optional[Dict] = None) -> dbc.Card:
    """
    Create individual corner analysis card with category-specific metrics

    Args:
        corner: Corner analysis dictionary
        index: Corner number
        category_id: Category ID to determine which metrics to display
        session_best: Optional session best data from calculate_session_best()

    Returns:
        Dash Bootstrap Card component
    """
    corner_name = corner.get('corner_name', f'Turn {index}')
    time_delta = corner.get('time_delta', 0)

    # Get category-specific opportunities (new structure)
    category_opportunities_map = corner.get('category_opportunities', {})
    opportunities = category_opportunities_map.get(category_id, [])

    # Fallback to legacy opportunities for backward compatibility
    if not opportunities:
        opportunities = corner.get('opportunities', [])

    # Get category configuration
    category_config = get_category_metrics(category_id)

    # Determine severity color
    if time_delta > 0.15:
        severity_color = '#dc3545'  # Red
        severity_badge_color = 'danger'
        severity_text = 'HIGH OPPORTUNITY'
    elif time_delta > 0.08:
        severity_color = '#ffc107'  # Yellow
        severity_badge_color = 'warning'
        severity_text = 'MEDIUM OPPORTUNITY'
    else:
        severity_color = '#17a2b8'  # Blue
        severity_badge_color = 'info'
        severity_text = 'LOW OPPORTUNITY'

    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.H5([
                    html.Span(f"#{index}", className="badge bg-dark me-2", style={'fontSize': '0.9rem'}),
                    html.Strong(corner_name),
                    create_session_best_badge(corner, session_best),  # Quick Win #4
                    dbc.Badge(
                        severity_text,
                        color=severity_badge_color,
                        className="ms-2",
                        style={'fontSize': '0.7rem'}
                    ),
                    dbc.Badge(
                        f"{time_delta:.3f}s lost/lap",
                        color="dark",
                        className="ms-2",
                        style={'fontSize': '0.7rem'}
                    )
                ], className="mb-0")
            ])
        ], style={
            'backgroundColor': '#f8f9fa',
            'borderLeft': f'4px solid {severity_color}'
        }),

        dbc.CardBody([
            # Metrics comparison table - category-specific
            html.Div([
                html.H6([
                    html.I(className="fas fa-chart-line me-2", style={'color': '#3498db'}),
                    f"{category_config['display_name']} Metrics"
                ], className="mb-3", style={'fontSize': '0.9rem'}),

                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Metric"),
                            html.Th("Your Average", style={'textAlign': 'center'}),
                            html.Th("Your Best", style={'textAlign': 'center'}),
                            html.Th("Gap", style={'textAlign': 'center'})
                        ])
                    ]),
                    html.Tbody(
                        _build_metric_rows(corner, category_id)
                    )
                ], bordered=True, hover=True, size="sm", className="mb-3")
            ]),

            # Phase breakdown section (Quick Win #3)
            create_phase_breakdown_visual(calculate_corner_phases(corner)),

            # Opportunities section
            html.Div([
                html.H6([
                    html.I(className="fas fa-lightbulb me-2", style={'color': '#f39c12'}),
                    "Specific Opportunities"
                ], className="mb-2", style={'fontSize': '0.9rem'}),

                html.Div([
                    html.Div([
                        html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                        html.Span(opp, style={'fontSize': '0.9rem'})
                    ], className="mb-2") for opp in opportunities
                ] if opportunities else [
                    html.P("No specific opportunities identified for this corner.", className="text-muted small mb-0")
                ], style={
                    'backgroundColor': '#fff9e6',
                    'padding': '1rem',
                    'borderLeft': '3px solid #f39c12',
                    'borderRadius': '0.25rem'
                })
            ])
        ])
    ], className="mb-3", style={'border': '1px solid #dee2e6'})


def create_corner_analysis_summary_card(corner_analyses: List[Dict]) -> dbc.Card:
    """
    Create summary card showing total opportunity across all corners

    Args:
        corner_analyses: List of corner analysis dictionaries

    Returns:
        Summary card component
    """
    if not corner_analyses or len(corner_analyses) == 0:
        return html.Div()

    total_time_loss = sum(c.get('time_delta', 0) for c in corner_analyses)
    high_opportunity = sum(1 for c in corner_analyses if c.get('time_delta', 0) > 0.15)
    medium_opportunity = sum(1 for c in corner_analyses if 0.08 <= c.get('time_delta', 0) <= 0.15)
    low_opportunity = sum(1 for c in corner_analyses if c.get('time_delta', 0) < 0.08)

    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-flag-checkered fa-2x mb-2", style={'color': '#3498db'}),
                        html.H3(f"{total_time_loss:.3f}s", className="mb-0"),
                        html.P("Total Lap Time", className="text-muted small mb-0")
                    ], className="text-center")
                ], md=3),

                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-exclamation-circle fa-2x mb-2", style={'color': '#dc3545'}),
                        html.H3(str(high_opportunity), className="mb-0"),
                        html.P("High Opportunity", className="text-muted small mb-0")
                    ], className="text-center")
                ], md=3),

                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle fa-2x mb-2", style={'color': '#ffc107'}),
                        html.H3(str(medium_opportunity), className="mb-0"),
                        html.P("Medium Opportunity", className="text-muted small mb-0")
                    ], className="text-center")
                ], md=3),

                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-info-circle fa-2x mb-2", style={'color': '#17a2b8'}),
                        html.H3(str(low_opportunity), className="mb-0"),
                        html.P("Low Opportunity", className="text-muted small mb-0")
                    ], className="text-center")
                ], md=3)
            ])
        ])
    ], className="mb-3", style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'color': 'white'
    })


# ============================================================================
# DEMO DATA (for testing)
# ============================================================================

DEMO_CORNER_ANALYSES = [
    {
        'corner_number': 1,
        'corner_name': 'Turn 1 (Uphill Left)',
        # Speed metrics
        'entry_speed_avg': 166.6,
        'entry_speed_best': 170.2,
        'apex_speed_avg': 77.3,
        'apex_speed_best': 79.8,
        'exit_speed_avg': 112.5,
        'exit_speed_best': 115.8,
        # Braking metrics
        'brake_pressure_avg': 143.8,
        'brake_pressure_max': 150.1,
        'brake_point': 85,
        'deceleration': 2.1,
        # Cornering metrics
        'lateral_g_max': 1.45,
        'lateral_g_avg': 1.22,
        'corner_speed': 77.3,
        # Steering metrics
        'steering_angle_max': 125.5,
        'steering_smoothness': 82,
        'steering_corrections': 3,
        # Throttle metrics
        'throttle_application': 68.5,
        'full_throttle_pct': 45.2,
        'throttle_modulation': 78.3,
        # Powertrain metrics
        'gear_used': 3,
        'rpm_avg': 5800,
        'rpm_max': 6200,
        # Lap Segmentation metrics
        'sector_time': 12.456,
        'time_gain_loss': 0.210,
        'consistency': 87.5,
        'optimal_line': 82.3,
        'lap_to_lap_delta': 0.089,
        # Composite metrics
        'overall_efficiency': 78.9,
        'performance_index': 7.89,
        'consistency_score': 85.2,
        'optimization_potential': 21.1,
        # General
        'time_delta': 0.21,
        # Category-specific opportunities
        'category_opportunities': {
            'braking': [
                'Increase peak brake pressure by 6.3 bar (143.8 → 150.1 bar)',
                'Brake 5m later for better entry speed',
                'Improve trail braking consistency into corner'
            ],
            'speed': [
                'Carry 3.6 km/h more entry speed (166.6 → 170.2 km/h)',
                'Maintain 2.5 km/h more through apex (77.3 → 79.8 km/h)',
                'Achieve 3.3 km/h higher exit speed (112.5 → 115.8 km/h)'
            ],
            'cornering': [
                'Optimize racing line to carry 2.5 km/h more mid-corner speed',
                'Maximize lateral G-forces (target: 1.45g)',
                'Hit apex 0.3m tighter for better exit'
            ],
            'throttle': [
                'Apply throttle 0.2s earlier on corner exit',
                'Progressive throttle application from apex for better traction',
                'Reduce partial throttle time by 15%'
            ],
            'steering': [
                'Improve steering smoothness to 85% (currently 82%)',
                'Minimize mid-corner corrections',
                'Maintain steady 125° angle through turn'
            ],
            'powertrain': [
                'Consider gear 3 for optimal RPM (5800 avg)',
                'Shift at 6200 RPM for maximum power',
                'Improve gear selection consistency'
            ],
            'lap_seg': [
                'High priority corner - 0.210s gain available',
                'Focus on lap-to-lap consistency in sector 1',
                'Reduce sector time variance by 0.05s'
            ],
            'composite': [
                'Total optimization potential: 0.210s per lap',
                'Primary limiter: braking point and trail braking',
                'Secondary focus: apex speed and throttle application'
            ],
            'uncategorized': [
                'Overall efficiency rating: 78.9% (target: 85%)',
                'Performance index improvement available: +0.8 points',
                'Consistency score can increase by 5.2%'
            ]
        },
        # Legacy field for backward compatibility
        'opportunities': [
            'Brake 2.3 bar harder (143.8 → 150.1 bar)',
            'Carry 2.5 km/h more speed through apex (77.3 → 79.8 km/h)',
            'Entry speed can increase 3.6 km/h with later braking'
        ]
    },
    {
        'corner_number': 2,
        'corner_name': 'Turn 6 (Hairpin Left)',
        # Speed metrics
        'entry_speed_avg': 145.2,
        'entry_speed_best': 148.8,
        'apex_speed_avg': 62.1,
        'apex_speed_best': 65.3,
        'exit_speed_avg': 98.7,
        'exit_speed_best': 102.4,
        # Braking metrics
        'brake_pressure_avg': 138.2,
        'brake_pressure_max': 145.7,
        'brake_point': 120,
        'deceleration': 2.8,
        # Cornering metrics
        'lateral_g_max': 1.28,
        'lateral_g_avg': 1.05,
        'corner_speed': 62.1,
        # Steering metrics
        'steering_angle_max': 185.2,
        'steering_smoothness': 75,
        'steering_corrections': 5,
        # Throttle metrics
        'throttle_application': 52.3,
        'full_throttle_pct': 38.7,
        'throttle_modulation': 72.1,
        # Powertrain metrics
        'gear_used': 2,
        'rpm_avg': 4500,
        'rpm_max': 5100,
        # Lap Segmentation metrics
        'sector_time': 18.234,
        'time_gain_loss': 0.180,
        'consistency': 83.7,
        'optimal_line': 79.5,
        'lap_to_lap_delta': 0.112,
        # Composite metrics
        'overall_efficiency': 82.0,
        'performance_index': 8.20,
        'consistency_score': 81.4,
        'optimization_potential': 18.0,
        # General
        'time_delta': 0.18,
        # Category-specific opportunities
        'category_opportunities': {
            'braking': [
                'Increase peak brake pressure by 7.5 bar (138.2 → 145.7 bar)',
                'Brake 8m earlier for better rotation',
                'Extend trail braking deeper into corner entry'
            ],
            'speed': [
                'Carry 3.6 km/h more entry speed (145.2 → 148.8 km/h)',
                'Apex speed can improve 3.2 km/h with better line (62.1 → 65.3 km/h)',
                'Exit speed gain of 3.7 km/h available (98.7 → 102.4 km/h)'
            ],
            'cornering': [
                'Tighten racing line through hairpin for better exit',
                'Increase lateral G loading (target: 1.28g max)',
                'Rotate car earlier for straighter exit'
            ],
            'throttle': [
                'Apply throttle 0.3s earlier on hairpin exit',
                'Focus on early throttle application for better acceleration',
                'Aggressive throttle in low-speed corner - use full 100%'
            ],
            'steering': [
                'Improve steering smoothness to 80% (currently 75%)',
                'Reduce hairpin steering angle corrections',
                'Sharp turn-in followed by smooth unwinding (185° max)'
            ],
            'powertrain': [
                'Gear 2 optimal for hairpin (4500 RPM avg)',
                'Short-shift to gear 3 earlier on exit',
                'Maximize torque delivery in low gear'
            ],
            'lap_seg': [
                'Important hairpin corner - 0.180s gain available',
                'Exit speed critical for following straight',
                'Improve consistency - target ±0.03s lap-to-lap'
            ],
            'composite': [
                'Total optimization potential: 0.180s per lap',
                'Primary limiter: exit speed and throttle application',
                'Key to fast lap: nail the exit for long straight ahead'
            ],
            'uncategorized': [
                'Overall efficiency rating: 81.4% (good performance)',
                'Performance index shows 0.6 points improvement possible',
                'Optimization potential: 18.0% remaining'
            ]
        },
        # Legacy field for backward compatibility
        'opportunities': [
            'Brake 7.5 bar harder for better entry',
            'Apex speed can improve 3.2 km/h with better line',
            'Focus on early throttle application on exit'
        ]
    },
    {
        'corner_number': 3,
        'corner_name': 'Turn 11 (Fast Left)',
        # Speed metrics
        'entry_speed_avg': 188.5,
        'entry_speed_best': 189.2,
        'apex_speed_avg': 165.3,
        'apex_speed_best': 167.1,
        'exit_speed_avg': 175.8,
        'exit_speed_best': 178.2,
        # Braking metrics
        'brake_pressure_avg': 42.1,
        'brake_pressure_max': 45.8,
        'brake_point': 40,
        'deceleration': 0.8,
        # Cornering metrics
        'lateral_g_max': 1.85,
        'lateral_g_avg': 1.62,
        'corner_speed': 165.3,
        # Steering metrics
        'steering_angle_max': 65.3,
        'steering_smoothness': 92,
        'steering_corrections': 1,
        # Throttle metrics
        'throttle_application': 88.7,
        'full_throttle_pct': 82.3,
        'throttle_modulation': 91.5,
        # Powertrain metrics
        'gear_used': 5,
        'rpm_avg': 7200,
        'rpm_max': 7600,
        # Lap Segmentation metrics
        'sector_time': 8.912,
        'time_gain_loss': 0.050,
        'consistency': 94.8,
        'optimal_line': 95.2,
        'lap_to_lap_delta': 0.023,
        # Composite metrics
        'overall_efficiency': 95.0,
        'performance_index': 9.50,
        'consistency_score': 96.1,
        'optimization_potential': 5.0,
        # General
        'time_delta': 0.05,
        # Category-specific opportunities
        'category_opportunities': {
            'braking': [
                'Minimal braking - light 45.8 bar max pressure optimal',
                'Brake point near-perfect at 40m before apex',
                'Gentle deceleration maintains momentum (0.8g)'
            ],
            'speed': [
                'Marginal entry speed gain: 0.7 km/h (188.5 → 189.2 km/h)',
                'Focus on maintaining momentum - 1.8 km/h apex speed gain possible (165.3 → 167.1 km/h)',
                'Exit speed opportunity: 2.4 km/h (175.8 → 178.2 km/h)'
            ],
            'cornering': [
                'High-speed corner - maximize lateral G (1.85g available)',
                'Optimize racing line for minimum speed loss',
                'Confidence critical - commit to 165+ km/h apex speed'
            ],
            'throttle': [
                'Maintain throttle through apex for momentum',
                'Progressive acceleration from mid-corner',
                'Full throttle earlier on exit - high-speed corner allows it'
            ],
            'steering': [
                'Excellent steering smoothness (92%) - maintain consistency',
                'Minimal corrections needed in fast corner',
                'Smooth 65° steering angle - don\'t over-steer'
            ],
            'powertrain': [
                'Gear 5 optimal for high-speed corner (7200 RPM avg)',
                'Stay in gear through entire corner - no shifting',
                'Maximize engine RPM (7600 max) for power delivery'
            ],
            'lap_seg': [
                'Fast corner - only 0.050s gain available',
                'Consistency more important than outright speed',
                'Small gains compound - focus on repeatable line'
            ],
            'composite': [
                'Total optimization potential: 0.050s (minor gain)',
                'Well-executed corner - small refinements only',
                'Maintain consistency and confidence through high-speed section'
            ],
            'uncategorized': [
                'Overall efficiency rating: 95.0% (excellent performance!)',
                'Performance index: 9.50 - near optimal execution',
                'Minimal optimization potential: only 5.0% remaining'
            ]
        },
        # Legacy field for backward compatibility
        'opportunities': [
            'Minimal braking improvement available',
            'Focus on maintaining momentum through apex',
            'Small gain possible with smoother inputs'
        ]
    }
]


if __name__ == '__main__':
    """Test the corner analysis widget components"""
    print("Corner Analysis Widget - Component Test")
    print("=" * 60)

    print("\nDemo corner analyses loaded:")
    for corner in DEMO_CORNER_ANALYSES:
        print(f"  {corner['corner_name']}: {corner['time_delta']:.3f}s opportunity")

    total = sum(c['time_delta'] for c in DEMO_CORNER_ANALYSES)
    print(f"\nTotal time opportunity: {total:.3f} seconds/lap")
    print("\nWidget components ready for integration!")
