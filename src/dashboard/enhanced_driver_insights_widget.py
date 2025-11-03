"""
Enhanced Driver Insights Widget - Phase 1.2 Integration
========================================================

Combines original API-based driver insights with Phase 1.2 modules:
- Ghost Lap Comparison
- Brake Point Analysis
- Corner Speed Benchmarking

Author: Production Engineering Team
Version: 1.0.0
License: GR Cup 2025 Hackathon
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import io
import requests
from typing import Optional, Dict, Any

from src.insights import (
    GhostLapComparator,
    BrakePointAnalyzer,
    CornerSpeedBenchmarking
)
from src.dashboard.telemetry_comparison_charts import (
    create_ghost_lap_overlay_chart,
    create_time_delta_heatmap,
    create_brake_zones_chart,
    create_brake_consistency_chart,
    create_corner_speed_chart,
    create_corner_speed_delta_chart,
    create_corner_type_distribution
)

# API configuration
API_BASE = "http://localhost:8000"

# Color scheme
COLORS = {
    'primary': '#2196F3',
    'success': '#4CAF50',
    'warning': '#FF9800',
    'danger': '#F44336',
    'info': '#00BCD4'
}


def create_enhanced_driver_insights_layout(
    df: pd.DataFrame,
    vehicle_number: int,
    lap_times_df: Optional[pd.DataFrame] = None,
    selected_lap: Optional[int] = None
) -> dbc.Container:
    """
    Create enhanced driver insights layout with Phase 1.2 features.

    Args:
        df: Telemetry DataFrame (long format)
        vehicle_number: Vehicle ID to analyze
        lap_times_df: Lap times DataFrame (optional, for Phase 1.2 features)
        selected_lap: Lap number to analyze (optional, defaults to best lap)

    Returns:
        Dash Bootstrap Container with all sections
    """
    # Section 1: Original API-based insights
    api_section = _create_api_insights_section(df, vehicle_number)

    # Section 2: Ghost Lap Comparison (Phase 1.2)
    ghost_section = _create_ghost_lap_section(df, lap_times_df, vehicle_number, selected_lap)

    # Section 3: Brake Point Analysis (Phase 1.2)
    brake_section = _create_brake_analysis_section(df, lap_times_df, vehicle_number, selected_lap)

    # Section 4: Corner Speed Benchmarking (Phase 1.2)
    corner_section = _create_corner_speed_section(df, lap_times_df, vehicle_number, selected_lap)

    # Combine all sections
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="fas fa-user-chart me-3"),
                    f"Driver Insights - Vehicle #{vehicle_number}"
                ], className="mb-4 text-primary")
            ])
        ]),

        # Section tabs or accordion
        dbc.Tabs([
            dbc.Tab(api_section, label="Performance Overview", tab_id="tab-overview"),
            dbc.Tab(ghost_section, label="Ghost Lap Comparison", tab_id="tab-ghost"),
            dbc.Tab(brake_section, label="Brake Analysis", tab_id="tab-brake"),
            dbc.Tab(corner_section, label="Corner Speed", tab_id="tab-corner"),
        ], id="insights-tabs", active_tab="tab-overview", className="mb-4"),

    ], fluid=True)


def _create_api_insights_section(df: pd.DataFrame, vehicle_number: int) -> html.Div:
    """Create original API-based insights section"""
    import time
    import random

    render_timestamp = time.time()
    unique_key = f"{vehicle_number}_{render_timestamp}_{random.randint(1000, 9999)}"

    try:
        # Call API for driver insights
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = requests.post(
            f"{API_BASE}/driver-insights",
            params={"vehicle_number": vehicle_number, "_t": render_timestamp},
            files={"file": ("telemetry.csv", csv_buffer, "text/csv")},
            timeout=30,
            headers={"Cache-Control": "no-cache", "Pragma": "no-cache"}
        )

        if response.status_code != 200:
            error_detail = response.json().get('detail', 'Unknown error')
            return html.Div([
                dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"API Error: {error_detail}"
                ], color="danger")
            ])

        insights = response.json()
        perf = insights['performance_summary']

        return html.Div([
            html.Div(id=f"render-key-{unique_key}", style={"display": "none"}),

            # Performance scores
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"{perf['consistency_score']:.1f}", className="text-center mb-0 text-primary"),
                            html.P("Consistency", className="text-muted text-center mb-0")
                        ])
                    ], className="shadow-sm")
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"{perf['aggression_index']:.1f}", className="text-center mb-0 text-warning"),
                            html.P("Aggression", className="text-muted text-center mb-0")
                        ])
                    ], className="shadow-sm")
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"{perf['smoothness_rating']:.1f}", className="text-center mb-0 text-success"),
                            html.P("Smoothness", className="text-muted text-center mb-0")
                        ])
                    ], className="shadow-sm")
                ], md=4),
            ], className="mb-4"),

            # Detailed metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6([html.I(className="fas fa-tachometer-alt me-2"), "Speed Metrics"])),
                        dbc.CardBody([
                            html.P([html.Strong("Average: "), f"{perf['avg_speed']:.1f} km/h"], className="mb-2"),
                            html.P([html.Strong("Maximum: "), f"{perf['max_speed']:.1f} km/h"], className="mb-0"),
                        ])
                    ], className="shadow-sm h-100")
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6([html.I(className="fas fa-stop me-2"), "Braking Metrics"])),
                        dbc.CardBody([
                            html.P([html.Strong("Average: "), f"{perf['avg_brake_pressure']:.1f} bar"], className="mb-2"),
                            html.P([html.Strong("Maximum: "), f"{perf['max_brake_pressure']:.1f} bar"], className="mb-0"),
                        ])
                    ], className="shadow-sm h-100")
                ], md=6),
            ], className="mb-4"),

            # Strengths and Weaknesses
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6([html.I(className="fas fa-check-circle me-2 text-success"), "Strengths"])),
                        dbc.CardBody([
                            html.Ul([html.Li(s) for s in insights['strengths']]) if insights['strengths']
                            else html.P("No specific strengths identified", className="text-muted mb-0")
                        ])
                    ], className="shadow-sm h-100")
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6([html.I(className="fas fa-exclamation-circle me-2 text-warning"), "Areas for Improvement"])),
                        dbc.CardBody([
                            html.Ul([html.Li(w) for w in insights['weaknesses']]) if insights['weaknesses']
                            else html.P("No specific weaknesses identified", className="text-muted mb-0")
                        ])
                    ], className="shadow-sm h-100")
                ], md=6),
            ], className="mb-4"),

            # Recommendations
            dbc.Card([
                dbc.CardHeader(html.H6([html.I(className="fas fa-lightbulb me-2 text-primary"), "Coaching Recommendations"])),
                dbc.CardBody([
                    html.Ol([html.Li(r) for r in insights['recommendations']])
                ])
            ], className="shadow-sm")
        ])

    except requests.exceptions.ConnectionError:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Cannot connect to API server. Please ensure the API is running at ",
                html.Code("http://localhost:8000"),
                html.Br(),
                html.Br(),
                "Showing Phase 1.2 features only (Ghost Lap, Brake Analysis, Corner Speed)."
            ], color="warning")
        ])
    except Exception as e:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-circle me-2"),
                f"Error generating insights: {str(e)}"
            ], color="danger")
        ])


def _create_ghost_lap_section(
    df: pd.DataFrame,
    lap_times_df: Optional[pd.DataFrame],
    vehicle_number: int,
    selected_lap: Optional[int]
) -> html.Div:
    """Create Ghost Lap Comparison section (Phase 1.2)"""
    if lap_times_df is None or lap_times_df.empty:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                "Ghost Lap Comparison requires lap times data. ",
                "Upload a telemetry file with lap information or select a session with lap times."
            ], color="info")
        ])

    try:
        # Initialize comparator
        comparator = GhostLapComparator()

        # Get best lap if selected_lap not provided
        vehicle_laps = lap_times_df[lap_times_df['vehicle_number'] == vehicle_number]
        if selected_lap is None:
            best_lap_idx = vehicle_laps['lap_time'].idxmin()
            selected_lap = int(vehicle_laps.loc[best_lap_idx, 'lap_number'])

        # Run comparison
        comparison = comparator.compare_to_best(
            df, lap_times_df,
            current_lap=selected_lap,
            vehicle_number=vehicle_number
        )

        # Create visualizations
        overlay_chart = create_ghost_lap_overlay_chart(comparison)
        heatmap_chart = create_time_delta_heatmap(comparison)

        return html.Div([
            # Summary cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"Lap {comparison.current_lap_number}", className="text-center mb-0"),
                            html.P("Current Lap", className="text-muted text-center mb-0 small")
                        ])
                    ], color="primary", outline=True)
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"Lap {comparison.ghost_lap_number}", className="text-center mb-0"),
                            html.P("Best Lap (Ghost)", className="text-muted text-center mb-0 small")
                        ])
                    ], color="success", outline=True)
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(
                                f"{comparison.lap_time_delta:+.3f}s",
                                className=f"text-center mb-0 {'text-danger' if comparison.lap_time_delta > 0 else 'text-success'}"
                            ),
                            html.P("Time Delta", className="text-muted text-center mb-0 small")
                        ])
                    ], outline=True)
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{len(comparison.biggest_losses)}", className="text-center mb-0 text-warning"),
                            html.P("Problem Areas", className="text-muted text-center mb-0 small")
                        ])
                    ], outline=True)
                ], md=3),
            ], className="mb-4"),

            # Overlay chart
            dbc.Card([
                dbc.CardHeader(html.H5([html.I(className="fas fa-chart-line me-2"), "Telemetry Overlay"])),
                dbc.CardBody([
                    dcc.Graph(figure=overlay_chart, config={'displayModeBar': True})
                ])
            ], className="shadow-sm mb-4"),

            # Time delta heatmap
            dbc.Card([
                dbc.CardHeader(html.H5([html.I(className="fas fa-fire me-2"), "Time Gain/Loss Heatmap"])),
                dbc.CardBody([
                    dcc.Graph(figure=heatmap_chart, config={'displayModeBar': True})
                ])
            ], className="shadow-sm mb-4"),

            # Key insights
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6([html.I(className="fas fa-arrow-down me-2 text-danger"), "Biggest Losses"])),
                        dbc.CardBody([
                            html.Ul([
                                html.Li(f"{loss['distance']:.0f}m: {loss['time_lost']:.3f}s - {loss['description']}")
                                for loss in comparison.biggest_losses
                            ]) if comparison.biggest_losses else html.P("No major losses detected", className="text-muted mb-0")
                        ])
                    ], className="shadow-sm h-100")
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6([html.I(className="fas fa-arrow-up me-2 text-success"), "Biggest Gains"])),
                        dbc.CardBody([
                            html.Ul([
                                html.Li(f"{gain['distance']:.0f}m: {gain['time_gained']:.3f}s - {gain['description']}")
                                for gain in comparison.biggest_gains
                            ]) if comparison.biggest_gains else html.P("No major gains detected", className="text-muted mb-0")
                        ])
                    ], className="shadow-sm h-100")
                ], md=6),
            ], className="mb-4"),
        ])

    except Exception as e:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-circle me-2"),
                f"Error in ghost lap comparison: {str(e)}"
            ], color="danger")
        ])


def _create_brake_analysis_section(
    df: pd.DataFrame,
    lap_times_df: Optional[pd.DataFrame],
    vehicle_number: int,
    selected_lap: Optional[int]
) -> html.Div:
    """Create Brake Point Analysis section (Phase 1.2)"""
    try:
        # Initialize analyzer
        analyzer = BrakePointAnalyzer()

        # Get best lap if selected_lap not provided
        if lap_times_df is not None and not lap_times_df.empty:
            vehicle_laps = lap_times_df[lap_times_df['vehicle_number'] == vehicle_number]
            if selected_lap is None and not vehicle_laps.empty:
                best_lap_idx = vehicle_laps['lap_time'].idxmin()
                selected_lap = int(vehicle_laps.loc[best_lap_idx, 'lap_number'])

        if selected_lap is None:
            # Fallback: analyze first available lap
            vehicle_data = df[df['vehicle_number'] == vehicle_number]
            if not vehicle_data.empty:
                selected_lap = int(vehicle_data['lap'].min())

        # Run analysis
        analysis = analyzer.analyze_lap(df, selected_lap, vehicle_number)

        # Create visualizations
        brake_chart = create_brake_zones_chart(analysis)

        return html.Div([
            # Summary cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{analysis.total_brake_zones}", className="text-center mb-0 text-primary"),
                            html.P("Brake Zones", className="text-muted text-center mb-0 small")
                        ])
                    ], outline=True)
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{analysis.avg_brake_duration:.2f}s", className="text-center mb-0"),
                            html.P("Avg Duration", className="text-muted text-center mb-0 small")
                        ])
                    ], outline=True)
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{analysis.avg_peak_pressure:.1f} bar", className="text-center mb-0"),
                            html.P("Avg Peak Pressure", className="text-muted text-center mb-0 small")
                        ])
                    ], outline=True)
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{analysis.avg_deceleration:.2f}g", className="text-center mb-0"),
                            html.P("Avg Deceleration", className="text-muted text-center mb-0 small")
                        ])
                    ], outline=True)
                ], md=3),
            ], className="mb-4"),

            # Brake zones chart
            dbc.Card([
                dbc.CardHeader(html.H5([html.I(className="fas fa-chart-bar me-2"), "Brake Zones"])),
                dbc.CardBody([
                    dcc.Graph(figure=brake_chart, config={'displayModeBar': True})
                ])
            ], className="shadow-sm mb-4"),

            # Coaching tips
            dbc.Card([
                dbc.CardHeader(html.H6([html.I(className="fas fa-comments me-2"), "Brake Coaching"])),
                dbc.CardBody([
                    html.Ul([html.Li(tip) for tip in analysis.coaching_tips])
                    if analysis.coaching_tips else html.P("No specific recommendations", className="text-muted mb-0")
                ])
            ], className="shadow-sm")
        ])

    except Exception as e:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-circle me-2"),
                f"Error in brake analysis: {str(e)}"
            ], color="danger")
        ])


def _create_corner_speed_section(
    df: pd.DataFrame,
    lap_times_df: Optional[pd.DataFrame],
    vehicle_number: int,
    selected_lap: Optional[int]
) -> html.Div:
    """Create Corner Speed Benchmarking section (Phase 1.2)"""
    try:
        # Initialize benchmarking
        benchmarking = CornerSpeedBenchmarking()

        # Get best lap if selected_lap not provided
        if lap_times_df is not None and not lap_times_df.empty:
            vehicle_laps = lap_times_df[lap_times_df['vehicle_number'] == vehicle_number]
            if selected_lap is None and not vehicle_laps.empty:
                best_lap_idx = vehicle_laps['lap_time'].idxmin()
                selected_lap = int(vehicle_laps.loc[best_lap_idx, 'lap_number'])

        if selected_lap is None:
            # Fallback: analyze first available lap
            vehicle_data = df[df['vehicle_number'] == vehicle_number]
            if not vehicle_data.empty:
                selected_lap = int(vehicle_data['lap'].min())

        # Run analysis
        analysis = benchmarking.analyze_lap(df, selected_lap, vehicle_number)

        # Create visualizations
        speed_chart = create_corner_speed_chart(analysis)
        type_dist_chart = create_corner_type_distribution(analysis)

        return html.Div([
            # Summary cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{analysis.total_corners}", className="text-center mb-0 text-primary"),
                            html.P("Corners Detected", className="text-muted text-center mb-0 small")
                        ])
                    ], outline=True)
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{analysis.avg_apex_speed:.1f} km/h", className="text-center mb-0"),
                            html.P("Avg Apex Speed", className="text-muted text-center mb-0 small")
                        ])
                    ], outline=True)
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{analysis.avg_lateral_g:.2f}g", className="text-center mb-0"),
                            html.P("Avg Lateral G", className="text-muted text-center mb-0 small")
                        ])
                    ], outline=True)
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{analysis.avg_corner_duration:.2f}s", className="text-center mb-0"),
                            html.P("Avg Duration", className="text-muted text-center mb-0 small")
                        ])
                    ], outline=True)
                ], md=3),
            ], className="mb-4"),

            # Corner speed chart
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5([html.I(className="fas fa-chart-line me-2"), "Corner Speeds"])),
                        dbc.CardBody([
                            dcc.Graph(figure=speed_chart, config={'displayModeBar': True})
                        ])
                    ], className="shadow-sm")
                ], md=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5([html.I(className="fas fa-chart-pie me-2"), "Corner Types"])),
                        dbc.CardBody([
                            dcc.Graph(figure=type_dist_chart, config={'displayModeBar': True})
                        ])
                    ], className="shadow-sm")
                ], md=4),
            ], className="mb-4"),

            # Best/Worst corners
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6([html.I(className="fas fa-star me-2 text-success"), "Best Corners"])),
                        dbc.CardBody([
                            html.Ul([html.Li(f"Corner {cid}") for cid in analysis.best_corners])
                            if analysis.best_corners else html.P("No comparison data", className="text-muted mb-0")
                        ])
                    ], className="shadow-sm h-100")
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6([html.I(className="fas fa-exclamation-triangle me-2 text-warning"), "Areas to Improve"])),
                        dbc.CardBody([
                            html.Ul([html.Li(f"Corner {cid}") for cid in analysis.worst_corners])
                            if analysis.worst_corners else html.P("No comparison data", className="text-muted mb-0")
                        ])
                    ], className="shadow-sm h-100")
                ], md=6),
            ], className="mb-4"),

            # Coaching tips
            dbc.Card([
                dbc.CardHeader(html.H6([html.I(className="fas fa-comments me-2"), "Corner Coaching"])),
                dbc.CardBody([
                    html.Ul([html.Li(tip) for tip in analysis.coaching_tips])
                    if analysis.coaching_tips else html.P("No specific recommendations", className="text-muted mb-0")
                ])
            ], className="shadow-sm")
        ])

    except Exception as e:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-circle me-2"),
                f"Error in corner speed analysis: {str(e)}"
            ], color="danger")
        ])


# Export layout creation function
__all__ = ['create_enhanced_driver_insights_layout']
