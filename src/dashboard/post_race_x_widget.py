"""
Post-Race-X Analysis Widget
============================

Extended analysis widget focusing on AI model configuration and feature inspection.

Features:
- AI Model Configuration card (40-feature vs 147-feature predictor)
- Dynamic Feature Inspector showing all engineered features
- Feature categorization and visualization
- Real-time updates based on vehicle selection

Usage:
    # In app.py:
    from src.dashboard.post_race_x_widget import create_post_race_x_layout, create_post_race_x_callbacks

    # Add to tabs
    dcc.Tab(label='Post-Race-X Analysis', value='tab-post-race-x', children=create_post_race_x_layout())

    # Register callbacks
    create_post_race_x_callbacks(app)
"""

from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from typing import Optional, List
import pandas as pd
import numpy as np
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

from src.data_processing.feature_engineering import TelemetryFeatureEngineer

# Data file path (local and production)
PARQUET_FILE = 'data/processed/combined_features_all_tracks.parquet'

# Key features for radar chart
RADAR_FEATURES = ['avg_speed', 'max_brake_f', 'full_throttle_pct', 'max_lateral_g',
                  'steering_smoothness', 'avg_rpm', 'driving_smoothness']


# ============================================================================
# FEATURE METADATA
# ============================================================================
# Comprehensive metadata for all 46 engineered features from TelemetryFeatureEngineer
# Used to dynamically generate Feature Inspector UI based on actual vehicle data

FEATURE_METADATA = {
    # Speed Features (8)
    'avg_speed': {'name': 'Average Speed', 'description': 'Mean velocity across the entire lap', 'unit': 'km/h', 'format': '{:.1f}', 'impact': 'Very High', 'icon': 'fas fa-tachometer-alt', 'color': '#3498db', 'category': 'Speed'},
    'max_speed': {'name': 'Maximum Speed', 'description': 'Peak velocity achieved on the lap', 'unit': 'km/h', 'format': '{:.1f}', 'impact': 'High', 'icon': 'fas fa-rocket', 'color': '#3498db', 'category': 'Speed'},
    'min_speed': {'name': 'Minimum Speed', 'description': 'Slowest point on the lap (usually apex of slowest corner)', 'unit': 'km/h', 'format': '{:.1f}', 'impact': 'Medium', 'icon': 'fas fa-snail', 'color': '#3498db', 'category': 'Speed'},
    'speed_variance': {'name': 'Speed Variance', 'description': 'Variation in speed throughout the lap', 'unit': 'km/h²', 'format': '{:.1f}', 'impact': 'Medium', 'icon': 'fas fa-wave-square', 'color': '#3498db', 'category': 'Speed'},
    'accel_time_pct': {'name': 'Acceleration Time %', 'description': 'Percentage of lap spent accelerating', 'unit': '%', 'format': '{:.1f}', 'impact': 'High', 'icon': 'fas fa-chart-line', 'color': '#3498db', 'category': 'Speed'},
    'decel_time_pct': {'name': 'Deceleration Time %', 'description': 'Percentage of lap spent decelerating', 'unit': '%', 'format': '{:.1f}', 'impact': 'High', 'icon': 'fas fa-chart-line-down', 'color': '#3498db', 'category': 'Speed'},
    'constant_speed_pct': {'name': 'Constant Speed %', 'description': 'Percentage of lap at steady speed (coast or maintenance)', 'unit': '%', 'format': '{:.1f}', 'impact': 'Medium', 'icon': 'fas fa-equals', 'color': '#3498db', 'category': 'Speed'},
    'speed_change_rate': {'name': 'Speed Change Rate', 'description': 'Average rate of speed changes per second', 'unit': 'km/h/s', 'format': '{:.2f}', 'impact': 'Medium', 'icon': 'fas fa-exchange-alt', 'color': '#3498db', 'category': 'Speed'},

    # Braking Features (8)
    'max_brake_f': {'name': 'Max Brake Pressure (Front)', 'description': 'Peak front brake pressure applied during the lap', 'unit': 'bar', 'format': '{:.1f}', 'impact': 'Very High', 'icon': 'fas fa-hand-paper', 'color': '#e74c3c', 'category': 'Braking'},
    'avg_brake_f': {'name': 'Avg Brake Pressure (Front)', 'description': 'Mean front brake pressure across all braking zones', 'unit': 'bar', 'format': '{:.1f}', 'impact': 'High', 'icon': 'fas fa-hand-paper', 'color': '#e74c3c', 'category': 'Braking'},
    'brake_count': {'name': 'Brake Applications', 'description': 'Number of times brakes were applied (identifies braking zones)', 'unit': 'count', 'format': '{:.0f}', 'impact': 'Medium', 'icon': 'fas fa-sort-numeric-up', 'color': '#e74c3c', 'category': 'Braking'},
    'hard_brake_pct': {'name': 'Hard Braking %', 'description': 'Percentage of lap with heavy brake pressure (>110 bar)', 'unit': '%', 'format': '{:.1f}', 'impact': 'Very High', 'icon': 'fas fa-exclamation-triangle', 'color': '#e74c3c', 'category': 'Braking'},
    'brake_variance': {'name': 'Brake Pressure Variance', 'description': 'Consistency of brake application (lower = more consistent)', 'unit': 'bar²', 'format': '{:.1f}', 'impact': 'High', 'icon': 'fas fa-wave-square', 'color': '#e74c3c', 'category': 'Braking'},
    'trail_braking_pct': {'name': 'Trail Braking %', 'description': 'Percentage of lap using trail braking technique (brake + turn)', 'unit': '%', 'format': '{:.1f}', 'impact': 'Very High', 'icon': 'fas fa-bezier-curve', 'color': '#e74c3c', 'category': 'Braking'},
    'brake_balance': {'name': 'Brake Balance', 'description': 'Ratio of front to rear brake pressure (ideal ~1.5-2.0)', 'unit': 'ratio', 'format': '{:.2f}', 'impact': 'Medium', 'icon': 'fas fa-balance-scale', 'color': '#e74c3c', 'category': 'Braking'},
    'brake_transition_smoothness': {'name': 'Brake Transition Smoothness', 'description': 'How smoothly driver modulates brake pressure', 'unit': 'score', 'format': '{:.2f}', 'impact': 'High', 'icon': 'fas fa-water', 'color': '#e74c3c', 'category': 'Braking'},

    # Throttle Features (5)
    'avg_throttle': {'name': 'Average Throttle', 'description': 'Mean throttle position throughout the lap', 'unit': '%', 'format': '{:.1f}', 'impact': 'High', 'icon': 'fas fa-bolt', 'color': '#2ecc71', 'category': 'Throttle'},
    'full_throttle_pct': {'name': 'Full Throttle %', 'description': 'Percentage of lap at 100% throttle (key performance metric)', 'unit': '%', 'format': '{:.1f}', 'impact': 'Very High', 'icon': 'fas fa-fire', 'color': '#2ecc71', 'category': 'Throttle'},
    'throttle_variance': {'name': 'Throttle Variance', 'description': 'Variation in throttle application (stability indicator)', 'unit': '%²', 'format': '{:.1f}', 'impact': 'Medium', 'icon': 'fas fa-wave-square', 'color': '#2ecc71', 'category': 'Throttle'},
    'throttle_aggression': {'name': 'Throttle Aggression', 'description': 'Rate of throttle application (how quickly driver floors it)', 'unit': '%/s', 'format': '{:.1f}', 'impact': 'High', 'icon': 'fas fa-fist-raised', 'color': '#2ecc71', 'category': 'Throttle'},
    'lift_and_coast_pct': {'name': 'Lift & Coast %', 'description': 'Percentage of lap with throttle fully released (0%)', 'unit': '%', 'format': '{:.1f}', 'impact': 'Medium', 'icon': 'fas fa-hand-rock', 'color': '#2ecc71', 'category': 'Throttle'},

    # Cornering Features (7)
    'max_lateral_g': {'name': 'Maximum Lateral G', 'description': 'Peak cornering force (grip limit indicator)', 'unit': 'g', 'format': '{:.2f}', 'impact': 'Very High', 'icon': 'fas fa-redo', 'color': '#e67e22', 'category': 'Cornering'},
    'avg_lateral_g': {'name': 'Average Lateral G', 'description': 'Mean cornering force across all turns', 'unit': 'g', 'format': '{:.2f}', 'impact': 'High', 'icon': 'fas fa-redo', 'color': '#e67e22', 'category': 'Cornering'},
    'high_g_time_pct': {'name': 'High G-Force Time %', 'description': 'Percentage of lap pulling >1.0g lateral force', 'unit': '%', 'format': '{:.1f}', 'impact': 'Very High', 'icon': 'fas fa-compress-arrows-alt', 'color': '#e67e22', 'category': 'Cornering'},
    'corner_entry_speed': {'name': 'Corner Entry Speed', 'description': 'Average speed when initiating corner turn-in', 'unit': 'km/h', 'format': '{:.1f}', 'impact': 'Very High', 'icon': 'fas fa-sign-in-alt', 'color': '#e67e22', 'category': 'Cornering'},
    'corner_exit_speed': {'name': 'Corner Exit Speed', 'description': 'Average speed when exiting corners onto straights', 'unit': 'km/h', 'format': '{:.1f}', 'impact': 'Very High', 'icon': 'fas fa-sign-out-alt', 'color': '#e67e22', 'category': 'Cornering'},
    'min_corner_speed': {'name': 'Minimum Corner Speed', 'description': 'Slowest speed at apex of corners (carrying speed indicator)', 'unit': 'km/h', 'format': '{:.1f}', 'impact': 'Very High', 'icon': 'fas fa-bullseye', 'color': '#e67e22', 'category': 'Cornering'},
    'grip_utilization': {'name': 'Grip Utilization', 'description': 'How close driver gets to traction limit (0-1 scale)', 'unit': 'ratio', 'format': '{:.2f}', 'impact': 'Very High', 'icon': 'fas fa-shoe-prints', 'color': '#e67e22', 'category': 'Cornering'},

    # Steering Features (5)
    'avg_steering': {'name': 'Average Steering Angle', 'description': 'Mean steering input magnitude throughout lap', 'unit': 'deg', 'format': '{:.1f}', 'impact': 'Medium', 'icon': 'fas fa-dharmachakra', 'color': '#3498db', 'category': 'Steering'},
    'max_steering': {'name': 'Maximum Steering Angle', 'description': 'Peak steering lock applied (identifies tightest corners)', 'unit': 'deg', 'format': '{:.1f}', 'impact': 'Medium', 'icon': 'fas fa-dharmachakra', 'color': '#3498db', 'category': 'Steering'},
    'steering_variance': {'name': 'Steering Variance', 'description': 'Variation in steering input (smoothness indicator)', 'unit': 'deg²', 'format': '{:.1f}', 'impact': 'High', 'icon': 'fas fa-wave-square', 'color': '#3498db', 'category': 'Steering'},
    'steering_changes_per_sec': {'name': 'Steering Changes/Sec', 'description': 'Frequency of steering corrections (stability metric)', 'unit': 'Hz', 'format': '{:.2f}', 'impact': 'High', 'icon': 'fas fa-sync-alt', 'color': '#3498db', 'category': 'Steering'},
    'steering_smoothness': {'name': 'Steering Smoothness', 'description': 'How gradually driver applies steering input', 'unit': 'score', 'format': '{:.2f}', 'impact': 'High', 'icon': 'fas fa-water', 'color': '#3498db', 'category': 'Steering'},

    # Powertrain Features (6)
    'avg_rpm': {'name': 'Average Engine RPM', 'description': 'Mean engine speed throughout the lap', 'unit': 'rpm', 'format': '{:.0f}', 'impact': 'Medium', 'icon': 'fas fa-cog', 'color': '#e67e22', 'category': 'Powertrain'},
    'max_rpm': {'name': 'Maximum RPM', 'description': 'Peak engine speed (redline usage)', 'unit': 'rpm', 'format': '{:.0f}', 'impact': 'Medium', 'icon': 'fas fa-tachometer-alt', 'color': '#e67e22', 'category': 'Powertrain'},
    'gear_changes': {'name': 'Gear Changes', 'description': 'Number of shifts per lap', 'unit': 'count', 'format': '{:.0f}', 'impact': 'Low', 'icon': 'fas fa-sort-numeric-up', 'color': '#e67e22', 'category': 'Powertrain'},
    'avg_gear': {'name': 'Average Gear', 'description': 'Mean gear used throughout lap', 'unit': '', 'format': '{:.1f}', 'impact': 'Low', 'icon': 'fas fa-cogs', 'color': '#e67e22', 'category': 'Powertrain'},
    'time_in_top_gear_pct': {'name': 'Time in Top Gear %', 'description': 'Percentage of lap in highest gear', 'unit': '%', 'format': '{:.1f}', 'impact': 'Medium', 'icon': 'fas fa-arrow-up', 'color': '#e67e22', 'category': 'Powertrain'},
    'shift_timing_smoothness': {'name': 'Shift Timing Smoothness', 'description': 'How smoothly driver executes gear changes', 'unit': 'score', 'format': '{:.2f}', 'impact': 'Medium', 'icon': 'fas fa-water', 'color': '#e67e22', 'category': 'Powertrain'},

    # Combined Features (7)
    'brake_throttle_overlap': {'name': 'Brake-Throttle Overlap', 'description': 'Time both brake and throttle pressed (left-foot braking)', 'unit': '%', 'format': '{:.1f}', 'impact': 'Medium', 'icon': 'fas fa-align-center', 'color': '#9b59b6', 'category': 'Combined'},
    'g_force_efficiency': {'name': 'G-Force Efficiency', 'description': 'How well driver uses available traction (accel + lateral)', 'unit': 'score', 'format': '{:.2f}', 'impact': 'Very High', 'icon': 'fas fa-certificate', 'color': '#9b59b6', 'category': 'Combined'},
    'cornering_efficiency': {'name': 'Cornering Efficiency', 'description': 'Speed through corners vs. available grip', 'unit': 'score', 'format': '{:.2f}', 'impact': 'Very High', 'icon': 'fas fa-percentage', 'color': '#9b59b6', 'category': 'Combined'},
    'throttle_on_exit_pct': {'name': 'Throttle on Exit %', 'description': 'How early driver gets on power exiting corners', 'unit': '%', 'format': '{:.1f}', 'impact': 'Very High', 'icon': 'fas fa-door-open', 'color': '#9b59b6', 'category': 'Combined'},
    'brake_to_throttle_time': {'name': 'Brake-to-Throttle Time', 'description': 'Average transition time from braking to acceleration', 'unit': 's', 'format': '{:.2f}', 'impact': 'High', 'icon': 'fas fa-exchange-alt', 'color': '#9b59b6', 'category': 'Combined'},
    'consistency_score': {'name': 'Consistency Score', 'description': 'Overall lap-to-lap consistency (lower variance = better)', 'unit': 'score', 'format': '{:.2f}', 'impact': 'High', 'icon': 'fas fa-equals', 'color': '#9b59b6', 'category': 'Combined'},
    'driving_smoothness': {'name': 'Overall Driving Smoothness', 'description': 'Combined metric of all input smoothness factors', 'unit': 'score', 'format': '{:.2f}', 'impact': 'Very High', 'icon': 'fas fa-water', 'color': '#9b59b6', 'category': 'Combined'},
}

CATEGORY_CONFIG = {
    'Speed': {'icon': 'fas fa-tachometer-alt', 'color': '#3498db', 'emoji': '🚀'},
    'Braking': {'icon': 'fas fa-hand-paper', 'color': '#e74c3c', 'emoji': '🛑'},
    'Throttle': {'icon': 'fas fa-bolt', 'color': '#2ecc71', 'emoji': '⚡'},
    'Cornering': {'icon': 'fas fa-redo', 'color': '#e67e22', 'emoji': '🔄'},
    'Steering': {'icon': 'fas fa-dharmachakra', 'color': '#3498db', 'emoji': '🎮'},
    'Powertrain': {'icon': 'fas fa-cog', 'color': '#e67e22', 'emoji': '⚙️'},
    'Combined': {'icon': 'fas fa-certificate', 'color': '#9b59b6', 'emoji': '🔧'},
}


def create_post_race_x_layout():
    """
    Create Post-Race-X Analysis layout with AI Model Configuration and Feature Inspector

    Now includes independent data loading from pre-computed features parquet file.

    Returns:
        Dash layout component
    """
    return html.Div([
        # Local data store for this tab (independent from other tabs)
        dcc.Store(id='post-race-x-local-data-store'),

        # ============================================================================
        # TRACK SELECTION (INDEPENDENT DATA LOADING)
        # ============================================================================
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-database me-2", style={'color': '#3498db'}),
                    "Select Track Data"
                ], className="mb-0")
            ], style={'backgroundColor': '#f8f9fa'}),
            dbc.CardBody([
                html.P("Load pre-computed feature data from 2,587 laps across 5 tracks:", className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id='post-race-x-track-selector',
                            options=[
                                {'label': '🏁 Barber Motorsports Park (466 laps)', 'value': 'barber-motorsports-park'},
                                {'label': '🏁 Circuit of the Americas (643 laps)', 'value': 'circuit-of-the-americas'},
                                {'label': '🏁 Road America (367 laps)', 'value': 'road-america'},
                                {'label': '🏁 Sonoma Raceway (747 laps)', 'value': 'sonoma'},
                                {'label': '🏁 Virginia International Raceway (364 laps)', 'value': 'virginia-international-raceway'},
                            ],
                            placeholder="Select a track to load features...",
                            className="mb-2"
                        )
                    ], md=8),
                    dbc.Col([
                        html.Div(id='post-race-x-load-status', children=[
                            html.Span("Select a track to begin", className="text-muted")
                        ])
                    ], md=4, className="d-flex align-items-center")
                ]),

                # Loading indicator
                dcc.Loading(
                    id="post-race-x-loading",
                    type="default",
                    children=html.Div(id='post-race-x-loading-output')
                )
            ])
        ], className="mb-4", style={'border': '1px solid #dee2e6', 'borderRadius': '10px'}),

        # ============================================================================
        # FEATURE 1: VEHICLE SELECTOR
        # ============================================================================
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-car me-2", style={'color': '#e74c3c'}),
                    "Vehicle Selection"
                ], className="mb-0")
            ], style={'backgroundColor': '#f8f9fa'}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id='post-race-x-vehicle-selector',
                            options=[],  # Populated dynamically when track loads
                            placeholder="Select vehicle(s) to analyze...",
                            multi=True,
                            className="mb-2"
                        )
                    ], md=10),
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="fas fa-download me-2"), "Export CSV"],
                            id='post-race-x-export-btn',
                            color="success",
                            size="sm",
                            disabled=True
                        ),
                        dcc.Download(id='post-race-x-download-csv')
                    ], md=2, className="d-flex align-items-center justify-content-end")
                ])
            ])
        ], className="mb-4", style={'border': '1px solid #dee2e6', 'borderRadius': '10px'}),

        # ============================================================================
        # FEATURE 2: SUMMARY STATISTICS
        # ============================================================================
        html.Div(id='post-race-x-summary-stats', children=[]),

        # ============================================================================
        # FEATURES 3-10: INTERACTIVE ANALYSIS TABS
        # ============================================================================
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-chart-bar me-2", style={'color': '#3498db'}),
                    "Interactive Analysis"
                ], className="mb-0")
            ], style={'backgroundColor': '#f8f9fa'}),
            dbc.CardBody([
                dbc.Tabs([
                    # Tab 1: Vehicle Comparison
                    dbc.Tab(label="📊 Vehicle Comparison", tab_id="tab-comparison", children=[
                        html.Div(className="p-3", children=[
                            dcc.Graph(id='post-race-x-comparison-chart')
                        ])
                    ]),

                    # Tab 2: Radar Chart
                    dbc.Tab(label="🎯 Radar Chart", tab_id="tab-radar", children=[
                        html.Div(className="p-3", children=[
                            dcc.Graph(id='post-race-x-radar-chart')
                        ])
                    ]),

                    # Tab 3: Correlation Heatmap
                    dbc.Tab(label="🔥 Correlation Map", tab_id="tab-correlation", children=[
                        html.Div(className="p-3", children=[
                            dcc.Graph(id='post-race-x-correlation-heatmap')
                        ])
                    ]),

                    # Tab 4: Scatter Plots
                    dbc.Tab(label="📈 Scatter Plot", tab_id="tab-scatter", children=[
                        html.Div(className="p-3", children=[
                            dbc.Row([
                                dbc.Col([
                                    html.Label("X-Axis Feature:"),
                                    dcc.Dropdown(id='post-race-x-scatter-x', options=[], placeholder="Select X feature...")
                                ], md=5),
                                dbc.Col([
                                    html.Label("Y-Axis Feature:"),
                                    dcc.Dropdown(id='post-race-x-scatter-y', options=[], placeholder="Select Y feature...")
                                ], md=5),
                                dbc.Col([
                                    html.Label(" "),
                                    html.Div([
                                        dbc.Button("Plot", id='post-race-x-scatter-btn', color="primary", size="sm")
                                    ])
                                ], md=2)
                            ], className="mb-3"),
                            dcc.Graph(id='post-race-x-scatter-plot')
                        ])
                    ]),

                    # Tab 5: Consistency Chart
                    dbc.Tab(label="📉 Consistency", tab_id="tab-consistency", children=[
                        html.Div(className="p-3", children=[
                            dcc.Graph(id='post-race-x-consistency-chart')
                        ])
                    ]),

                    # Tab 6: Top Performers
                    dbc.Tab(label="🏆 Top Performers", tab_id="tab-performers", children=[
                        html.Div(className="p-3", children=[
                            html.Div(id='post-race-x-top-performers-table')
                        ])
                    ]),

                    # Tab 7: Rankings
                    dbc.Tab(label="📊 Rankings", tab_id="tab-rankings", children=[
                        html.Div(className="p-3", children=[
                            html.Div(id='post-race-x-rankings-table')
                        ])
                    ])
                ], id='post-race-x-analysis-tabs', active_tab="tab-comparison")
            ])
        ], className="mb-4", style={'border': '1px solid #dee2e6', 'borderRadius': '10px'}),

        # ============================================================================
        # POST-RACE-X ANALYSIS SECTION (EXISTING - AI Model Config)
        # ============================================================================
        dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="fas fa-chart-line me-3", style={'color': '#e74c3c'}),
                    "Post-Race-X Analysis"
                ], className="mb-0", style={
                    'fontWeight': '700',
                    'fontSize': '24px',
                    'color': '#2c3e50'
                })
            ], style={
                'backgroundColor': '#ffffff',
                'borderBottom': '3px solid #e74c3c',
                'padding': '1.2rem'
            }),
            dbc.CardBody([

                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-brain me-2", style={'color': '#9b59b6'}),
                            "AI Model Configuration"
                        ], className="mb-0")
                    ], style={'backgroundColor': '#f8f9fa'}),
                    dbc.CardBody([
                        dbc.Row([
                            # Current Mode
                            dbc.Col([
                                html.Div([
                                    html.H6([
                                        html.I(className="fas fa-cog me-2", style={'color': '#3498db'}),
                                        "Current Analysis Mode"
                                    ], className="mb-3"),

                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H5("40-Feature Basic Predictor", className="mb-2", style={'color': '#2c3e50'}),
                                            html.P([
                                                "Optimized for datasets with ",
                                                html.Strong("9 core sensors"),
                                                " (no GPS required)"
                                            ], className="mb-3"),

                                            html.Div([
                                                dbc.Badge("Speed Analysis", color="primary", className="me-2 mb-2"),
                                                dbc.Badge("Braking Metrics", color="primary", className="me-2 mb-2"),
                                                dbc.Badge("Throttle Control", color="primary", className="me-2 mb-2"),
                                                dbc.Badge("G-Force Analysis", color="primary", className="me-2 mb-2"),
                                                dbc.Badge("Steering Dynamics", color="primary", className="me-2 mb-2"),
                                                dbc.Badge("Gear Usage", color="primary", className="me-2 mb-2"),
                                            ], className="mb-3"),

                                            html.Div([
                                                html.Strong("Expected Accuracy: "),
                                                html.Span("89-91% R²", style={'fontSize': '1.1rem', 'color': '#2ecc71'})
                                            ], className="mb-2"),

                                            html.Div([
                                                html.Strong("Typical Error: "),
                                                html.Span("±2-3 seconds per lap", style={'fontSize': '1.0rem', 'color': '#95a5a6'})
                                            ])
                                        ])
                                    ], style={'backgroundColor': '#ecf0f1', 'border': 'none'})
                                ])
                            ], md=6),

                            # Advanced Mode (Future)
                            dbc.Col([
                                html.Div([
                                    html.H6([
                                        html.I(className="fas fa-rocket me-2", style={'color': '#e67e22'}),
                                        "Advanced Mode (Requires GPS)"
                                    ], className="mb-3"),

                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H5("147-Feature Advanced Predictor", className="mb-2", style={'color': '#7f8c8d'}),
                                            html.P([
                                                "Requires ",
                                                html.Strong("12 sensors including GPS"),
                                                " for full spatial analysis"
                                            ], className="mb-3"),

                                            html.Div([
                                                dbc.Badge("All Basic Features", color="secondary", className="me-2 mb-2"),
                                                dbc.Badge("FFT Analysis", color="secondary", className="me-2 mb-2"),
                                                dbc.Badge("Wavelet Transform", color="secondary", className="me-2 mb-2"),
                                                dbc.Badge("Corner Detection", color="secondary", className="me-2 mb-2"),
                                                dbc.Badge("Track Encoding", color="secondary", className="me-2 mb-2"),
                                                dbc.Badge("Spatial Features", color="secondary", className="me-2 mb-2"),
                                            ], className="mb-3"),

                                            html.Div([
                                                html.Strong("Expected Accuracy: "),
                                                html.Span("97.49% R²", style={'fontSize': '1.1rem', 'color': '#bdc3c7'})
                                            ], className="mb-2"),

                                            html.Div([
                                                html.Strong("Typical Error: "),
                                                html.Span("±1.73 seconds per lap", style={'fontSize': '1.0rem', 'color': '#bdc3c7'})
                                            ])
                                        ])
                                    ], style={'backgroundColor': '#ecf0f1', 'border': '2px dashed #95a5a6'})
                                ])
                            ], md=6)
                        ]),

                        # Info footer
                        html.Hr(className="mt-3 mb-3"),
                        html.Div([
                            html.I(className="fas fa-info-circle me-2", style={'color': '#3498db'}),
                            html.Strong("Why Two Modes? "),
                            "The 40-feature predictor is specifically designed for reliability with minimal sensor data. ",
                            "It delivers excellent results (89-91% R²) using only the 9 core sensors available in your dataset. ",
                            "To unlock the advanced 147-feature predictor (97.49% R²), you'll need to add GPS sensors to capture ",
                            "spatial track position data for corner detection, FFT, and wavelet analysis."
                        ], style={'fontSize': '0.95rem', 'color': '#7f8c8d'})
                    ])
                ], className="mb-4", style={'border': '1px solid #dee2e6', 'borderRadius': '10px'}),

                # ============================================================================
                # FEATURE INSPECTOR WITH VEHICLE SELECTOR
                # ============================================================================
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-user-circle me-2", style={'color': '#667eea'}),
                            "Select Vehicle for Feature Inspector"
                        ], className="mb-0")
                    ], style={'backgroundColor': '#f8f9fa'}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Choose a vehicle to inspect its features:", className="mb-2"),
                                dcc.Dropdown(
                                    id='post-race-x-feature-inspector-vehicle',
                                    options=[],  # Populated when track loads
                                    placeholder="Select a vehicle to view detailed features...",
                                    className="mb-2"
                                )
                            ], md=8),
                            dbc.Col([
                                html.Div(id='post-race-x-inspector-vehicle-info', className="mt-4")
                            ], md=4)
                        ])
                    ])
                ], className="mb-4", style={'border': '1px solid #dee2e6', 'borderRadius': '10px'}),

                # Feature Inspector - dynamically populated based on selected vehicle
                html.Div(id='dynamic-feature-inspector-x', children=[]),

            ], style={'padding': '1.5rem'})
        ], className="mb-4", style={'border': '1px solid #dee2e6', 'borderRadius': '15px', 'boxShadow': '0 4px 15px rgba(231, 76, 60, 0.1)'}),
    ])


def create_post_race_x_callbacks(app):
    """
    Register callbacks for Post-Race-X Analysis tab

    Args:
        app: Dash app instance
    """

    # =========================================================================
    # INDEPENDENT DATA LOADING CALLBACK
    # =========================================================================
    @app.callback(
        [Output('post-race-x-local-data-store', 'data'),
         Output('post-race-x-load-status', 'children'),
         Output('post-race-x-loading-output', 'children')],
        [Input('post-race-x-track-selector', 'value')]
    )
    def load_track_features(selected_track):
        """
        Load pre-computed features from parquet file (INDEPENDENT DATA SOURCE)

        This makes Post-Race-X tab completely independent from Post-Race Analysis tab.
        Features are loaded instantly from parquet file (~0.06 seconds).
        """
        if not selected_track:
            return None, html.Span("Select a track to begin", className="text-muted"), None

        try:
            # Load parquet file (ultra-fast: <0.1 seconds)
            df = pd.read_parquet(PARQUET_FILE)

            # Filter by selected track
            track_data = df[df['track'] == selected_track].copy()

            if len(track_data) == 0:
                return None, html.Span("No data found for this track", className="text-danger"), None

            # Get unique vehicles
            vehicles = sorted(track_data['vehicle_number'].unique())

            # Prepare features_by_vehicle structure (same format as post_race_widget.py)
            features_by_vehicle = {}

            for vehicle in vehicles:
                vehicle_data = track_data[track_data['vehicle_number'] == vehicle]

                # Get average features across all laps for this vehicle
                feature_cols = [c for c in vehicle_data.columns
                               if c not in ['vehicle_number', 'lap_number', 'race', 'track']]

                avg_features = vehicle_data[feature_cols].mean().to_dict()

                # Convert to JSON-serializable format
                features_by_vehicle[str(vehicle)] = {k: float(v) if pd.notna(v) else 0.0
                                                     for k, v in avg_features.items()}

            # Status message
            status = html.Div([
                html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                html.Span(f"Loaded {len(track_data)} laps, {len(vehicles)} vehicles", style={'color': '#2ecc71'})
            ])

            # Return data in same format as post_race_widget for compatibility
            return {'features_by_vehicle': features_by_vehicle}, status, None

        except FileNotFoundError:
            error_msg = html.Div([
                html.I(className="fas fa-exclamation-triangle me-2", style={'color': '#e74c3c'}),
                html.Span(f"Data file not found: {PARQUET_FILE}", style={'color': '#e74c3c'})
            ])
            return None, error_msg, None

        except Exception as e:
            error_msg = html.Div([
                html.I(className="fas fa-times-circle me-2", style={'color': '#e74c3c'}),
                html.Span(f"Error loading data: {str(e)}", style={'color': '#e74c3c'})
            ])
            return None, error_msg, None

    # =========================================================================
    # CALLBACK 1: POPULATE VEHICLE SELECTOR & ENABLE EXPORT
    # =========================================================================
    @app.callback(
        [Output('post-race-x-vehicle-selector', 'options'),
         Output('post-race-x-vehicle-selector', 'value'),
         Output('post-race-x-export-btn', 'disabled'),
         Output('post-race-x-scatter-x', 'options'),
         Output('post-race-x-scatter-y', 'options'),
         Output('post-race-x-feature-inspector-vehicle', 'options'),
         Output('post-race-x-feature-inspector-vehicle', 'value')],
        [Input('post-race-x-local-data-store', 'data')]
    )
    def populate_vehicle_selector(stored_data):
        """Populate vehicle dropdown when track data loads"""
        if not stored_data:
            return [], None, True, [], [], [], None

        features_by_vehicle = stored_data.get('features_by_vehicle', {})
        if not features_by_vehicle:
            return [], None, True, [], [], [], None

        # Vehicle options
        vehicles = sorted([int(v) for v in features_by_vehicle.keys()])
        vehicle_options = [{'label': f'Vehicle {v}', 'value': v} for v in vehicles]

        # Feature options for scatter plot
        feature_names = list(features_by_vehicle[str(vehicles[0])].keys())
        feature_options = [{'label': f.replace('_', ' ').title(), 'value': f} for f in feature_names]

        # Auto-select all vehicles for comparison, select first vehicle for Feature Inspector
        return vehicle_options, vehicles, False, feature_options, feature_options, vehicle_options, vehicles[0]


    # =========================================================================
    # CALLBACK 2: SUMMARY STATISTICS CARD
    # =========================================================================
    @app.callback(
        Output('post-race-x-summary-stats', 'children'),
        [Input('post-race-x-local-data-store', 'data'),
         Input('post-race-x-vehicle-selector', 'value')]
    )
    def update_summary_stats(stored_data, selected_vehicles):
        """Display summary statistics for selected vehicles"""
        if not stored_data or not selected_vehicles:
            return []

        features_by_vehicle = stored_data.get('features_by_vehicle', {})
        if not features_by_vehicle:
            return []

        # Create dataframe from selected vehicles
        vehicle_list = [selected_vehicles] if isinstance(selected_vehicles, int) else selected_vehicles
        df_list = []
        for v in vehicle_list:
            if str(v) in features_by_vehicle:
                data = features_by_vehicle[str(v)].copy()
                data['vehicle'] = v
                df_list.append(data)

        if not df_list:
            return []

        df = pd.DataFrame(df_list)

        # Calculate summary stats
        feature_cols = [c for c in df.columns if c != 'vehicle']
        stats_summary = df[feature_cols].describe().loc[['mean', 'std', 'min', 'max']]

        # Create summary card
        return dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-chart-pie me-2", style={'color': '#3498db'}),
                    f"Summary Statistics ({len(vehicle_list)} vehicle{'s' if len(vehicle_list) > 1 else ''})"
                ], className="mb-0")
            ], style={'backgroundColor': '#f8f9fa'}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Quick Stats", className="text-muted mb-3"),
                        html.P([html.Strong("Features Analyzed: "), f"{len(feature_cols)}"]),
                        html.P([html.Strong("Avg Speed: "), f"{df['avg_speed'].mean():.1f} km/h"]),
                        html.P([html.Strong("Avg Max Brake: "), f"{df['max_brake_f'].mean():.1f} bar"]),
                    ], md=3),
                    dbc.Col([
                        html.H6("Performance Metrics", className="text-muted mb-3"),
                        html.P([html.Strong("Avg Full Throttle: "), f"{df.get('full_throttle_pct', pd.Series([0])).mean():.1f}%"]),
                        html.P([html.Strong("Avg Max G-Force: "), f"{df['max_lateral_g'].mean():.2f} g"]),
                        html.P([html.Strong("Avg Smoothness: "), f"{df.get('driving_smoothness', pd.Series([0])).mean():.2f}"]),
                    ], md=3),
                    dbc.Col([
                        html.H6("Consistency Indicators", className="text-muted mb-3"),
                        html.P([html.Strong("Speed Variance: "), f"{df.get('speed_variance', pd.Series([0])).mean():.1f}"]),
                        html.P([html.Strong("Brake Consistency: "), f"{df.get('brake_variance', pd.Series([0])).mean():.1f}"]),
                        html.P([html.Strong("Throttle Variance: "), f"{df.get('throttle_variance', pd.Series([0])).mean():.1f}"]),
                    ], md=3),
                    dbc.Col([
                        html.H6("Rankings Available", className="text-muted mb-3"),
                        html.P([html.I(className="fas fa-trophy me-2", style={'color': '#f39c12'}),
                               "Check Rankings tab"]),
                        html.P([html.I(className="fas fa-chart-bar me-2", style={'color': '#3498db'}),
                               "View comparisons below"]),
                    ], md=3)
                ])
            ])
        ], className="mb-4", style={'border': '1px solid #dee2e6', 'borderRadius': '10px'})


    # =========================================================================
    # CALLBACK 3: VEHICLE COMPARISON BAR CHART
    # =========================================================================
    @app.callback(
        Output('post-race-x-comparison-chart', 'figure'),
        [Input('post-race-x-local-data-store', 'data'),
         Input('post-race-x-vehicle-selector', 'value')]
    )
    def update_comparison_chart(stored_data, selected_vehicles):
        """Create side-by-side vehicle comparison"""
        if not stored_data or not selected_vehicles:
            return go.Figure().add_annotation(text="Select vehicles to compare", showarrow=False)

        features_by_vehicle = stored_data.get('features_by_vehicle', {})
        vehicle_list = [selected_vehicles] if isinstance(selected_vehicles, int) else selected_vehicles

        # Key features to compare
        key_features = ['avg_speed', 'max_brake_f', 'full_throttle_pct', 'max_lateral_g',
                       'driving_smoothness', 'avg_rpm']

        fig = go.Figure()

        for v in vehicle_list:
            if str(v) in features_by_vehicle:
                values = [features_by_vehicle[str(v)].get(f, 0) for f in key_features]
                fig.add_trace(go.Bar(
                    name=f'Vehicle {v}',
                    x=[f.replace('_', ' ').title() for f in key_features],
                    y=values,
                    text=[f'{val:.1f}' for val in values],
                    textposition='auto'
                ))

        fig.update_layout(
            title="Vehicle Performance Comparison (Key Features)",
            barmode='group',
            height=500,
            xaxis_title="Feature",
            yaxis_title="Value",
            hovermode='x unified'
        )

        return fig


    # =========================================================================
    # CALLBACK 4: EXPORT TO CSV
    # =========================================================================
    @app.callback(
        Output('post-race-x-download-csv', 'data'),
        [Input('post-race-x-export-btn', 'n_clicks')],
        [State('post-race-x-local-data-store', 'data'),
         State('post-race-x-track-selector', 'value')]
    )
    def export_to_csv(n_clicks, stored_data, track_name):
        """Export loaded features to CSV"""
        if not n_clicks or not stored_data:
            return None

        features_by_vehicle = stored_data.get('features_by_vehicle', {})

        # Convert to dataframe
        df_list = []
        for vehicle, features in features_by_vehicle.items():
            row = features.copy()
            row['vehicle_number'] = int(vehicle)
            df_list.append(row)

        df = pd.DataFrame(df_list)

        # Generate filename
        filename = f"post_race_x_features_{track_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

        return dcc.send_data_frame(df.to_csv, filename, index=False)


    # =========================================================================
    # CALLBACK 5: PERCENTILE RANKINGS
    # =========================================================================
    @app.callback(
        Output('post-race-x-rankings-table', 'children'),
        [Input('post-race-x-local-data-store', 'data'),
         Input('post-race-x-vehicle-selector', 'value')]
    )
    def update_rankings(stored_data, selected_vehicles):
        """Calculate percentile rankings for each vehicle"""
        if not stored_data or not selected_vehicles:
            return html.P("Select vehicles to view rankings", className="text-muted text-center p-3")

        features_by_vehicle = stored_data.get('features_by_vehicle', {})
        vehicle_list = [selected_vehicles] if isinstance(selected_vehicles, int) else selected_vehicles

        # Create dataframe
        df_list = []
        for v, features in features_by_vehicle.items():
            row = features.copy()
            row['vehicle'] = int(v)
            df_list.append(row)

        df = pd.DataFrame(df_list)

        # Calculate percentile ranks for key features
        key_features = ['avg_speed', 'max_brake_f', 'full_throttle_pct', 'max_lateral_g']

        rankings_data = []
        for v in vehicle_list:
            vehicle_row = df[df['vehicle'] == v].iloc[0] if len(df[df['vehicle'] == v]) > 0 else None
            if vehicle_row is None:
                continue

            rank_row = {'Vehicle': v}
            for feat in key_features:
                if feat in df.columns:
                    percentile = stats.percentileofscore(df[feat], vehicle_row[feat])
                    rank_row[feat.replace('_', ' ').title()] = f"{percentile:.0f}%"
            rankings_data.append(rank_row)

        rankings_df = pd.DataFrame(rankings_data)

        # Create table
        return dbc.Table.from_dataframe(rankings_df, striped=True, bordered=True, hover=True,
                                         responsive=True, className="mt-3")


    # =========================================================================
    # CALLBACK 6: RADAR CHART
    # =========================================================================
    @app.callback(
        Output('post-race-x-radar-chart', 'figure'),
        [Input('post-race-x-local-data-store', 'data'),
         Input('post-race-x-vehicle-selector', 'value')]
    )
    def update_radar_chart(stored_data, selected_vehicles):
        """Create radar chart for multi-dimensional comparison"""
        if not stored_data or not selected_vehicles:
            return go.Figure().add_annotation(text="Select vehicles to compare", showarrow=False)

        features_by_vehicle = stored_data.get('features_by_vehicle', {})
        vehicle_list = [selected_vehicles] if isinstance(selected_vehicles, int) else selected_vehicles

        fig = go.Figure()

        # Normalize features to 0-100 scale for radar chart
        df_list = []
        for v, features in features_by_vehicle.items():
            df_list.append(features)
        df_all = pd.DataFrame(df_list)

        for v in vehicle_list:
            if str(v) not in features_by_vehicle:
                continue

            features = features_by_vehicle[str(v)]
            values = []
            labels = []

            for feat in RADAR_FEATURES:
                if feat in features and feat in df_all.columns:
                    # Normalize to 0-100 scale
                    min_val = df_all[feat].min()
                    max_val = df_all[feat].max()
                    if max_val > min_val:
                        normalized = 100 * (features[feat] - min_val) / (max_val - min_val)
                    else:
                        normalized = 50
                    values.append(normalized)
                    labels.append(feat.replace('_', ' ').title())

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=f'Vehicle {v}'
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Multi-Dimensional Performance Comparison (Normalized 0-100)",
            height=600
        )

        return fig


    # =========================================================================
    # CALLBACK 7: CORRELATION HEATMAP
    # =========================================================================
    @app.callback(
        Output('post-race-x-correlation-heatmap', 'figure'),
        [Input('post-race-x-local-data-store', 'data')]
    )
    def update_correlation_heatmap(stored_data):
        """Create feature correlation heatmap"""
        if not stored_data:
            return go.Figure().add_annotation(text="Load track data to view correlations", showarrow=False)

        features_by_vehicle = stored_data.get('features_by_vehicle', {})

        # Create dataframe
        df_list = []
        for features in features_by_vehicle.values():
            df_list.append(features)

        df = pd.DataFrame(df_list)

        # Select subset of interesting features
        key_features = ['avg_speed', 'max_brake_f', 'full_throttle_pct', 'max_lateral_g',
                       'driving_smoothness', 'avg_rpm', 'max_steering', 'grip_utilization']

        available_features = [f for f in key_features if f in df.columns]
        corr_matrix = df[available_features].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[f.replace('_', ' ').title() for f in corr_matrix.columns],
            y=[f.replace('_', ' ').title() for f in corr_matrix.index],
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))

        fig.update_layout(
            title="Feature Correlation Matrix",
            height=600,
            xaxis_title="",
            yaxis_title=""
        )

        return fig


    # =========================================================================
    # CALLBACK 8: TOP PERFORMERS TABLE
    # =========================================================================
    @app.callback(
        Output('post-race-x-top-performers-table', 'children'),
        [Input('post-race-x-local-data-store', 'data')]
    )
    def update_top_performers(stored_data):
        """Show best vehicle for each feature category"""
        if not stored_data:
            return html.P("Load track data to view top performers", className="text-muted text-center p-3")

        features_by_vehicle = stored_data.get('features_by_vehicle', {})

        # Create dataframe
        df_list = []
        for v, features in features_by_vehicle.items():
            row = features.copy()
            row['vehicle'] = int(v)
            df_list.append(row)

        df = pd.DataFrame(df_list)

        # Define categories and their key features
        categories = {
            '🚀 Speed': ['avg_speed', 'max_speed'],
            '🛑 Braking': ['max_brake_f', 'avg_brake_f'],
            '⚡ Throttle': ['full_throttle_pct', 'avg_throttle'],
            '🔄 Cornering': ['max_lateral_g', 'grip_utilization'],
            '🎮 Smoothness': ['driving_smoothness', 'steering_smoothness'],
        }

        top_performers = []

        for category, features in categories.items():
            for feat in features:
                if feat in df.columns:
                    best_idx = df[feat].idxmax()
                    best_vehicle = df.loc[best_idx, 'vehicle']
                    best_value = df.loc[best_idx, feat]

                    top_performers.append({
                        'Category': category,
                        'Feature': feat.replace('_', ' ').title(),
                        'Best Vehicle': f"Vehicle {int(best_vehicle)}",
                        'Value': f"{best_value:.2f}"
                    })

        performers_df = pd.DataFrame(top_performers)

        return dbc.Table.from_dataframe(performers_df, striped=True, bordered=True, hover=True,
                                         responsive=True, className="mt-3")


    # =========================================================================
    # CALLBACK 9: INTERACTIVE SCATTER PLOT
    # =========================================================================
    @app.callback(
        Output('post-race-x-scatter-plot', 'figure'),
        [Input('post-race-x-scatter-btn', 'n_clicks')],
        [State('post-race-x-local-data-store', 'data'),
         State('post-race-x-scatter-x', 'value'),
         State('post-race-x-scatter-y', 'value')]
    )
    def update_scatter_plot(n_clicks, stored_data, x_feature, y_feature):
        """Create interactive scatter plot for any two features"""
        if not n_clicks or not stored_data or not x_feature or not y_feature:
            return go.Figure().add_annotation(text="Select X and Y features, then click Plot", showarrow=False)

        features_by_vehicle = stored_data.get('features_by_vehicle', {})

        # Create dataframe
        df_list = []
        for v, features in features_by_vehicle.items():
            row = features.copy()
            row['vehicle'] = int(v)
            df_list.append(row)

        df = pd.DataFrame(df_list)

        if x_feature not in df.columns or y_feature not in df.columns:
            return go.Figure().add_annotation(text="Selected features not available", showarrow=False)

        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            color='vehicle',
            size=df[x_feature].abs() + df[y_feature].abs(),
            hover_data=['vehicle'],
            labels={x_feature: x_feature.replace('_', ' ').title(),
                    y_feature: y_feature.replace('_', ' ').title()},
            title=f"{x_feature.replace('_', ' ').title()} vs {y_feature.replace('_', ' ').title()}"
        )

        fig.update_layout(height=500)

        return fig


    # =========================================================================
    # CALLBACK 10: LAP-BY-LAP CONSISTENCY CHART
    # =========================================================================
    @app.callback(
        Output('post-race-x-consistency-chart', 'figure'),
        [Input('post-race-x-local-data-store', 'data'),
         Input('post-race-x-vehicle-selector', 'value')]
    )
    def update_consistency_chart(stored_data, selected_vehicles):
        """Show variance/consistency across key features"""
        if not stored_data or not selected_vehicles:
            return go.Figure().add_annotation(text="Select vehicles to view consistency", showarrow=False)

        features_by_vehicle = stored_data.get('features_by_vehicle', {})
        vehicle_list = [selected_vehicles] if isinstance(selected_vehicles, int) else selected_vehicles

        # Features to analyze for consistency
        consistency_features = ['speed_variance', 'brake_variance', 'throttle_variance',
                               'steering_variance']

        fig = go.Figure()

        for v in vehicle_list:
            if str(v) in features_by_vehicle:
                features = features_by_vehicle[str(v)]
                values = [features.get(f, 0) for f in consistency_features]

                fig.add_trace(go.Bar(
                    name=f'Vehicle {v}',
                    x=[f.replace('_', ' ').title() for f in consistency_features],
                    y=values,
                    text=[f'{val:.1f}' for val in values],
                    textposition='auto'
                ))

        fig.update_layout(
            title="Consistency Analysis (Lower variance = Better consistency)",
            barmode='group',
            height=500,
            xaxis_title="Feature Variance",
            yaxis_title="Variance Value",
            hovermode='x unified'
        )

        return fig

    # =========================================================================
    # CALLBACK 11: VEHICLE INFO DISPLAY
    # =========================================================================
    @app.callback(
        Output('post-race-x-inspector-vehicle-info', 'children'),
        [Input('post-race-x-feature-inspector-vehicle', 'value')]
    )
    def update_inspector_vehicle_info(selected_vehicle):
        """Display selected vehicle info"""
        if not selected_vehicle:
            return html.Div([
                html.P("No vehicle selected", className="text-muted")
            ])

        return html.Div([
            html.H5([
                html.I(className="fas fa-car me-2", style={'color': '#667eea'}),
                f"Vehicle {selected_vehicle}"
            ], style={'color': '#667eea', 'fontWeight': 'bold'}),
            html.P("Inspecting features", className="text-muted mb-0", style={'fontSize': '0.9rem'})
        ])

    # =========================================================================
    # DYNAMIC FEATURE INSPECTOR CALLBACK (UPDATED FOR VEHICLE SELECTOR)
    # =========================================================================
    @app.callback(
        Output('dynamic-feature-inspector-x', 'children'),
        [Input('post-race-x-local-data-store', 'data'),
         Input('post-race-x-feature-inspector-vehicle', 'value')]
    )
    def update_dynamic_feature_inspector(stored_data, selected_vehicle):
        """
        Dynamically generate Feature Inspector based on SELECTED vehicle features

        NOW WITH VEHICLE SELECTOR: User can choose which vehicle to inspect.
        - Shows ALL 45 features from pre-computed parquet file
        - Updates instantly when vehicle is selected
        - Groups features by category with color-coded tabs
        - Displays actual values averaged across all laps
        - CLEARLY shows which vehicle is being inspected
        """
        if not stored_data:
            return html.Div([
                html.P("Select a track above to load features and see the Feature Inspector.",
                       className="text-muted text-center p-5", style={'fontSize': '1.1rem'})
            ])

        # Handle both old format (string) and new format (dict)
        if isinstance(stored_data, str):
            # Old format - no features available
            features_by_vehicle = {}
        else:
            # New format - features included
            features_by_vehicle = stored_data.get('features_by_vehicle', {})

        if not features_by_vehicle:
            return html.Div([
                html.P("No features loaded. Select a track from the dropdown above to load pre-computed features.",
                       className="text-muted text-center p-5")
            ])

        # Get features for SELECTED vehicle (or first if none selected)
        if selected_vehicle and str(selected_vehicle) in features_by_vehicle:
            vehicle = str(selected_vehicle)
        else:
            vehicle = list(features_by_vehicle.keys())[0]

        vehicle_features = features_by_vehicle.get(vehicle, {})

        if not vehicle_features:
            return html.Div([
                html.P(f"No features found for vehicle {vehicle}.",
                       className="text-muted text-center p-5")
            ])

        # Group features by category
        features_by_category = {}
        for feature_key, feature_value in vehicle_features.items():
            if feature_key in FEATURE_METADATA:
                metadata = FEATURE_METADATA[feature_key]
                category = metadata['category']
                if category not in features_by_category:
                    features_by_category[category] = []
                features_by_category[category].append((feature_key, feature_value, metadata))

        # Create tabs for each category
        tabs = []
        category_order = ['Speed', 'Braking', 'Throttle', 'Cornering', 'Steering', 'Powertrain', 'Combined']

        for category_name in category_order:
            if category_name not in features_by_category:
                continue

            features = features_by_category[category_name]
            config = CATEGORY_CONFIG[category_name]

            # Create feature cards in 2-column layout
            feature_rows = []
            for i in range(0, len(features), 2):
                cols = []
                for j in range(2):
                    if i + j < len(features):
                        feature_key, feature_value, metadata = features[i + j]

                        # Format value
                        try:
                            formatted_value = metadata['format'].format(float(feature_value))
                        except (ValueError, TypeError):
                            formatted_value = str(feature_value)

                        # Create feature card
                        cols.append(dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6([
                                        html.I(className=f"{metadata['icon']} me-2",
                                               style={'color': metadata['color']}),
                                        metadata['name']
                                    ], style={'fontWeight': '600', 'fontSize': '1.05rem'}),
                                    html.P(metadata['description'],
                                           className="text-muted mb-2",
                                           style={'fontSize': '0.9rem'}),
                                    html.Div([
                                        html.Span("Value: ", style={'fontWeight': '500'}),
                                        html.Span(f"{formatted_value} {metadata['unit']}",
                                                  style={'color': metadata['color'], 'fontWeight': '600'})
                                    ], className="mb-1"),
                                    html.Div([
                                        html.Span("Impact: ", style={'fontWeight': '500'}),
                                        html.Span(metadata['impact'],
                                                  style={'color': '#e74c3c' if metadata['impact'] == 'Very High' else '#3498db'})
                                    ])
                                ], className="p-3")
                            ], className="mb-3", style={'borderLeft': f"4px solid {metadata['color']}"})
                        ], md=6))

                feature_rows.append(dbc.Row(cols, className="mb-2"))

            # Create tab
            tab_label = f"{config['emoji']} {category_name} ({len(features)})"
            tabs.append(dbc.Tab(
                label=tab_label,
                tab_id=f"dynamic-x-{category_name.lower()}",
                children=[html.Div(feature_rows, className="p-3")]
            ))

        # Return complete card with tabs
        return dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col([
                        html.H4([
                            html.I(className="fas fa-microscope me-3", style={'color': '#667eea'}),
                            "Feature Inspector - Under the Hood"
                        ], className="mb-0", style={
                            'fontWeight': '700',
                            'fontSize': '28px',
                            'color': '#2c3e50',
                            'fontFamily': 'Inter, sans-serif'
                        })
                    ], md=8),
                    dbc.Col([
                        html.Div([
                            html.H3([
                                html.I(className="fas fa-car me-2"),
                                f"VEHICLE {vehicle}"
                            ], className="mb-0 text-end", style={
                                'fontWeight': '900',
                                'fontSize': '32px',
                                'color': '#ffffff',
                                'backgroundColor': '#667eea',
                                'padding': '10px 20px',
                                'borderRadius': '8px',
                                'boxShadow': '0 4px 8px rgba(102, 126, 234, 0.3)'
                            })
                        ])
                    ], md=4, className="d-flex align-items-center justify-content-end")
                ])
            ], style={
                'backgroundColor': '#ffffff',
                'borderBottom': '3px solid #667eea',
                'padding': '1.5rem'
            }),
            dbc.CardBody([
                dbc.Alert([
                    html.I(className="fas fa-info-circle me-2"),
                    html.Strong(f"Currently inspecting VEHICLE {vehicle}: "),
                    f"Showing {len(vehicle_features)} engineered features from this vehicle's telemetry data. ",
                    "These features are averaged across all laps for this vehicle on the selected track."
                ], color="info", className="mb-4"),
                html.P([
                    "Our AI analyzes ",
                    html.Strong(f"{len(vehicle_features)} engineered features",
                                style={'color': '#667eea', 'fontSize': '1.1rem'}),
                    f" from Vehicle {vehicle}'s telemetry to predict lap times. ",
                    "Below are the features extracted from your session, organized by category:"
                ], className="mb-4", style={'fontSize': '1.05rem', 'color': '#34495e'}),

                dbc.Tabs(tabs, active_tab=f"dynamic-x-speed", className="mt-3"),

                html.Hr(className="mt-4 mb-3"),
                dbc.Alert([
                    html.I(className="fas fa-lightbulb me-2", style={'color': '#f39c12'}),
                    html.Strong("Pro Tip: "),
                    "Focus on improving features marked as 'Very High Impact' first. Small improvements in Full Throttle %, ",
                    "Maximum Lateral G, and Minimum Corner Speed yield the biggest lap time gains."
                ], color="info", className="mb-0")
            ], style={'padding': '2rem'})
        ], className="mb-4", style={
            'border': '1px solid #dee2e6',
            'borderRadius': '15px',
            'boxShadow': '0 4px 20px rgba(102, 126, 234, 0.15)'
        })
