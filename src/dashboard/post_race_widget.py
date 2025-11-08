"""
Post-Race Analysis Widget (Tab 8)
==================================

Dashboard tab for comprehensive post-session analysis with AI-powered insights.

Features:
- Lap-by-lap timeline visualization (actual vs. predicted)
- Error distribution analysis
- Session statistics summary
- Anomaly detection and classification
- Coaching recommendations
- Export to CSV/PDF

Usage:
    # In app.py:
    from src.dashboard.post_race_widget import create_post_race_layout, create_post_race_callbacks

    # Add to tabs
    dcc.Tab(label='Post-Race Analysis', value='tab-8', children=create_post_race_layout())

    # Register callbacks
    create_post_race_callbacks(app)
"""

from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import base64
import io
from typing import Optional, List, Tuple

from src.models.inference.post_race_predictor import PostRacePredictor, load_session_data
from src.models.inference.simple_post_race_predictor import SimplePostRacePredictor
from src.dashboard.post_race_analyzer import PostRaceAnalyzer


# Color scheme
COLORS = {
    'actual': '#1f77b4',        # Blue
    'predicted': '#ff7f0e',     # Orange
    'anomaly': '#d62728',       # Red
    'good_lap': '#2ca02c',      # Green
    'background': '#f8f9fa',    # Light gray
    'text': '#212529'           # Dark gray
}


def create_post_race_layout():
    """
    Create Tab 8: Post-Race Analysis layout

    Returns:
        Dash layout component
    """
    return dbc.Container([
        # AUTO-LOAD INFRASTRUCTURE
        # Hidden store for auto-loaded data
        dcc.Store(id='post-race-autoload-store', storage_type='memory'),

        # Hidden store for radio-selected data
        dcc.Store(id='post-race-radio-data-store', storage_type='memory'),

        # Status message area (success/error messages)
        html.Div(id='post-race-autoload-status', children=[], className="mb-3"),


        # Header - Left-aligned with description
        dbc.Row([
            dbc.Col([
                html.H3("📊 Post-Race Analysis", className="mb-3", style={'textAlign': 'left'}),

                # Description area explaining data file contents
                dbc.Alert([
                    html.H5([
                        html.I(className="fas fa-info-circle me-2"),
                        "About Post-Race Analysis Data"
                    ], className="alert-heading", style={'textAlign': 'left'}),
                    html.P([
                        "This analysis tool processes comprehensive telemetry data from your racing sessions. ",
                        "Each data file contains:"
                    ], className="mb-2", style={'textAlign': 'left'}),
                    html.Ul([
                        html.Li([
                            html.Strong("Lap Data: "),
                            "Individual lap times, sectors, and timestamps for performance tracking"
                        ]),
                        html.Li([
                            html.Strong("Telemetry Channels: "),
                            "12 sensor readings @ 10Hz including speed, braking, throttle, steering, GPS, and G-forces"
                        ]),
                        html.Li([
                            html.Strong("Features: "),
                            "147 engineered features from basic metrics (speed, braking) to advanced FFT/wavelet analysis"
                        ]),
                        html.Li([
                            html.Strong("AI Predictions: "),
                            "Sequential LightGBM model (97.49% R²) for lap time forecasting and anomaly detection"
                        ])
                    ], style={'textAlign': 'left', 'marginBottom': '0'}),
                    html.Hr(),
                    html.P([
                        html.I(className="fas fa-lightbulb me-2"),
                        html.Strong("Tip: "),
                        "Select a track below to load sample data and generate detailed coaching insights, ",
                        "or upload your own telemetry CSV file for personalized analysis."
                    ], className="mb-0", style={'textAlign': 'left', 'fontSize': '0.95rem'})
                ], color="info", className="mb-3", style={'textAlign': 'left'}),

                html.P(
                    "Comprehensive session review with AI-powered insights and coaching recommendations",
                    className="text-muted mb-3",
                    style={'textAlign': 'left'}
                ),
                html.Hr()
            ])
        ], className="mb-3"),

        # SENSOR STATUS CARD - Shows which sensors are available
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-wifi me-2", style={'color': '#2ecc71'}),
                    "Sensor Status & Data Quality"
                ], className="mb-0")
            ], style={'backgroundColor': '#f8f9fa'}),
            dbc.CardBody([
                dbc.Row([
                    # Left: Sensor checklist
                    dbc.Col([
                        html.H6("Available Sensors (9/12)", className="mb-3"),
                        html.Div([
                            # Present sensors - green checkmarks
                            html.Div([
                                html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                                html.Span("Speed", className="fw-bold"),
                                html.Span(" (km/h)", className="text-muted ms-1")
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                                html.Span("Brake Pressure Front", className="fw-bold"),
                                html.Span(" (bar)", className="text-muted ms-1")
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                                html.Span("Brake Pressure Rear", className="fw-bold"),
                                html.Span(" (bar)", className="text-muted ms-1")
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                                html.Span("Throttle Position", className="fw-bold"),
                                html.Span(" (%)", className="text-muted ms-1")
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                                html.Span("Acceleration X", className="fw-bold"),
                                html.Span(" (g)", className="text-muted ms-1")
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                                html.Span("Acceleration Y", className="fw-bold"),
                                html.Span(" (g)", className="text-muted ms-1")
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                                html.Span("Steering Angle", className="fw-bold"),
                                html.Span(" (deg)", className="text-muted ms-1")
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                                html.Span("Gear", className="fw-bold"),
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                                html.Span("Engine RPM", className="fw-bold"),
                                html.Span(" (rpm)", className="text-muted ms-1")
                            ], className="mb-3"),

                            # Missing sensors - red X's
                            html.Hr(),
                            html.H6("Missing Sensors (3/12)", className="mb-3 text-danger"),
                            html.Div([
                                html.I(className="fas fa-times-circle me-2", style={'color': '#e74c3c'}),
                                html.Span("GPS Latitude", className="text-muted"),
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="fas fa-times-circle me-2", style={'color': '#e74c3c'}),
                                html.Span("GPS Longitude", className="text-muted"),
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="fas fa-times-circle me-2", style={'color': '#e74c3c'}),
                                html.Span("GPS Altitude", className="text-muted"),
                            ], className="mb-0")
                        ])
                    ], md=6),

                    # Right: Data quality metrics
                    dbc.Col([
                        html.H6("Data Quality Metrics", className="mb-3"),
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-database fa-2x mb-2", style={'color': '#3498db'}),
                                    html.H4("582,035", className="mb-0", style={'color': '#2c3e50'}),
                                    html.P("Telemetry Points", className="text-muted mb-0", style={'fontSize': '0.9rem'})
                                ], className="text-center mb-3")
                            ])
                        ], className="mb-3", style={'backgroundColor': '#f8f9fa'}),

                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-check-double fa-2x mb-2", style={'color': '#2ecc71'}),
                                    html.H4("100%", className="mb-0", style={'color': '#2c3e50'}),
                                    html.P("Data Completeness", className="text-muted mb-0", style={'fontSize': '0.9rem'})
                                ], className="text-center mb-0")
                            ])
                        ], style={'backgroundColor': '#f8f9fa'}),

                        # GPS Warning Alert
                        dbc.Alert([
                            html.I(className="fas fa-exclamation-triangle me-2"),
                            html.Strong("GPS Not Available: "),
                            "Advanced spatial analysis features (147 total features) require GPS data. ",
                            "Current analysis uses 40 basic features optimized for available sensors."
                        ], color="warning", className="mt-3 mb-0")
                    ], md=6)
                ])
            ])
        ], className="mb-4", style={'border': '1px solid #dee2e6', 'borderRadius': '10px'}),

        # FEATURE DOCUMENTATION CARD - Explains current predictor mode
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
        # FEATURE INSPECTOR PANEL - Shows all 40 baseline features
        # ============================================================================
        dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="fas fa-microscope me-3", style={'color': '#667eea'}),
                    "Feature Inspector - Under the Hood"
                ], className="mb-0", style={
                    'fontWeight': '700',
                    'fontSize': '28px',
                    'color': '#2c3e50',
                    'fontFamily': 'Inter, sans-serif'
                })
            ], style={
                'backgroundColor': '#ffffff',
                'borderBottom': '3px solid #667eea',
                'padding': '1.5rem'
            }),
            dbc.CardBody([
                # Introduction
                html.P([
                    "Our AI analyzes ",
                    html.Strong("40 engineered features", style={'color': '#667eea', 'fontSize': '1.1rem'}),
                    " from 9 telemetry sensors to predict lap times with 89-91% accuracy. ",
                    "Below are the features extracted from your session, organized by category:"
                ], className="mb-4", style={'fontSize': '1.05rem', 'color': '#34495e'}),

                # Feature Categories in Tabs
                dbc.Tabs([
                    # Speed Features Tab
                    dbc.Tab(label="🚀 Speed (8)", tab_id="speed-features", children=[
                        html.Div([
                            html.H5("Speed Analysis Features", className="mt-3 mb-3", style={'color': '#667eea'}),
                            html.P("Metrics analyzing your velocity profile throughout the lap", className="text-muted mb-3"),

                            # Feature list with descriptions
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-tachometer-alt me-2", style={'color': '#3498db'}),
                                                "Average Speed"
                                            ], className="mb-2"),
                                            html.P("Mean velocity across the entire lap", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("High - directly correlates with lap time", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-arrow-up me-2", style={'color': '#e74c3c'}),
                                                "Maximum Speed"
                                            ], className="mb-2"),
                                            html.P("Peak velocity reached during the lap", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - indicates straight-line performance", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-arrow-down me-2", style={'color': '#95a5a6'}),
                                                "Minimum Speed"
                                            ], className="mb-2"),
                                            html.P("Slowest point, typically slowest corner apex", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("High - slow corners lose time", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-chart-line me-2", style={'color': '#9b59b6'}),
                                                "Speed Variance"
                                            ], className="mb-2"),
                                            html.P("Consistency of velocity - lower is smoother", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - affects tire wear and balance", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            # Additional speed features
                            html.Hr(),
                            html.P([
                                html.Strong("Also analyzed: "),
                                "Speed Range (max-min), Time Above 170 km/h, Normalized Speed Metrics"
                            ], className="text-muted", style={'fontSize': '0.95rem'})
                        ], className="p-3")
                    ]),

                    # Braking Features Tab
                    dbc.Tab(label="🛑 Braking (8)", tab_id="braking-features", children=[
                        html.Div([
                            html.H5("Braking Performance Features", className="mt-3 mb-3", style={'color': '#e74c3c'}),
                            html.P("Metrics analyzing your braking technique and consistency", className="text-muted mb-3"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-hand-paper me-2", style={'color': '#e74c3c'}),
                                                "Maximum Brake Pressure"
                                            ], className="mb-2"),
                                            html.P("Peak front brake pressure (bar)", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("High - hard braking = shorter braking zones", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-chart-area me-2", style={'color': '#3498db'}),
                                                "Brake Consistency"
                                            ], className="mb-2"),
                                            html.P("Standard deviation of brake pressure >50 bar", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("High - consistent braking = predictable car", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-clock me-2", style={'color': '#f39c12'}),
                                                "Brake Duration"
                                            ], className="mb-2"),
                                            html.P("Percentage of lap time on brakes (>20 bar)", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - less time braking = more speed", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-shoe-prints me-2", style={'color': '#9b59b6'}),
                                                "Trail Braking Amount"
                                            ], className="mb-2"),
                                            html.P("Overlap between braking and throttle application", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - advanced technique for rotation", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            html.Hr(),
                            html.P([
                                html.Strong("Also analyzed: "),
                                "Average Brake Pressure, Number of Braking Zones, Brake Point Consistency"
                            ], className="text-muted", style={'fontSize': '0.95rem'})
                        ], className="p-3")
                    ]),

                    # Throttle Features Tab
                    dbc.Tab(label="⚡ Throttle (5)", tab_id="throttle-features", children=[
                        html.Div([
                            html.H5("Throttle Application Features", className="mt-3 mb-3", style={'color': '#2ecc71'}),
                            html.P("Metrics analyzing your power delivery and acceleration technique", className="text-muted mb-3"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-bolt me-2", style={'color': '#f39c12'}),
                                                "Full Throttle Percentage"
                                            ], className="mb-2"),
                                            html.P("Time at >95% throttle position", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Very High - more full throttle = faster laps", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-sliders-h me-2", style={'color': '#3498db'}),
                                                "Throttle Modulation"
                                            ], className="mb-2"),
                                            html.P("Standard deviation - smoothness of application", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - smooth = better traction", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-chart-bar me-2", style={'color': '#2ecc71'}),
                                                "Average Throttle"
                                            ], className="mb-2"),
                                            html.P("Mean throttle position throughout lap", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("High - overall power delivery metric", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-percentage me-2", style={'color': '#9b59b6'}),
                                                "Time Above 50% Throttle"
                                            ], className="mb-2"),
                                            html.P("Percentage of lap with significant power", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - indicates acceleration efficiency", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            html.Hr(),
                            html.P([
                                html.Strong("Also analyzed: "),
                                "Maximum Throttle Position"
                            ], className="text-muted", style={'fontSize': '0.95rem'})
                        ], className="p-3")
                    ]),

                    # Cornering Features Tab
                    dbc.Tab(label="🔄 Cornering (7)", tab_id="cornering-features", children=[
                        html.Div([
                            html.H5("Cornering Dynamics Features", className="mt-3 mb-3", style={'color': '#e67e22'}),
                            html.P("Metrics analyzing your lateral g-forces and corner performance", className="text-muted mb-3"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-circle-notch me-2", style={'color': '#e67e22'}),
                                                "Maximum Lateral G"
                                            ], className="mb-2"),
                                            html.P("Peak cornering force (g-force)", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Very High - higher G = faster corners", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-bullseye me-2", style={'color': '#3498db'}),
                                                "Minimum Corner Speed"
                                            ], className="mb-2"),
                                            html.P("Apex speed at slowest turn (km/h)", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Very High - higher apex = faster exit", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-grip-horizontal me-2", style={'color': '#2ecc71'}),
                                                "Grip Utilization"
                                            ], className="mb-2"),
                                            html.P("% of theoretical maximum lateral g used", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("High - pushing limits = faster times", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-equals me-2", style={'color': '#9b59b6'}),
                                                "Cornering Consistency"
                                            ], className="mb-2"),
                                            html.P("Standard deviation of lateral g-forces", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - smooth cornering = predictability", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            html.Hr(),
                            html.P([
                                html.Strong("Also analyzed: "),
                                "Average Lateral G, Corner Count, High-G Sections"
                            ], className="text-muted", style={'fontSize': '0.95rem'})
                        ], className="p-3")
                    ]),

                    # Steering Features Tab
                    dbc.Tab(label="🎮 Steering (5)", tab_id="steering-features", children=[
                        html.Div([
                            html.H5("Steering Technique Features", className="mt-3 mb-3", style={'color': '#3498db'}),
                            html.P("Metrics analyzing steering smoothness and precision", className="text-muted mb-3"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-dharmachakra me-2", style={'color': '#3498db'}),
                                                "Steering Smoothness"
                                            ], className="mb-2"),
                                            html.P("Standard deviation of steering angle changes", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - smooth inputs = better balance", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-undo me-2", style={'color': '#e74c3c'}),
                                                "Steering Corrections"
                                            ], className="mb-2"),
                                            html.P("Number of direction changes (overcorrections)", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - fewer = more precise", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-arrows-alt-h me-2", style={'color': '#2ecc71'}),
                                                "Maximum Steering Angle"
                                            ], className="mb-2"),
                                            html.P("Peak steering input (degrees)", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Low - track-specific characteristic", style={'color': '#95a5a6'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-chart-line me-2", style={'color': '#9b59b6'}),
                                                "Average Absolute Steering"
                                            ], className="mb-2"),
                                            html.P("Mean magnitude of steering inputs", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Low - indicates track technicality", style={'color': '#95a5a6'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ])
                        ], className="p-3")
                    ]),

                    # Other Features Tab
                    dbc.Tab(label="🔧 Other (7+)", tab_id="other-features", children=[
                        html.Div([
                            html.H5("Powertrain & Combined Features", className="mt-3 mb-3", style={'color': '#9b59b6'}),
                            html.P("Additional metrics from engine, gearing, and composite calculations", className="text-muted mb-3"),

                            # Powertrain
                            html.H6("⚙️ Powertrain Features", className="mb-3", style={'color': '#e67e22'}),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.P("RPM Efficiency, Shift Points, Gear Usage, Time in Power Band", className="mb-0")
                                        ])
                                    ], className="mb-3", style={'backgroundColor': '#f8f9fa'})
                                ], md=12),
                            ]),

                            # Combined
                            html.H6("🔬 Combined Metrics", className="mb-3 mt-3", style={'color': '#3498db'}),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.P("Traction Circle Utilization, Performance Index, Driving Aggression Score", className="mb-0")
                                        ])
                                    ], className="mb-3", style={'backgroundColor': '#f8f9fa'})
                                ], md=12),
                            ]),

                            html.Hr(),
                            html.Div([
                                html.I(className="fas fa-info-circle me-2", style={'color': '#3498db'}),
                                html.Strong("Total Feature Count: "),
                                html.Span("40 baseline features", style={'fontSize': '1.1rem', 'color': '#667eea'})
                            ], className="mb-3"),

                            html.P([
                                "These features are engineered from just 9 basic telemetry sensors, making the analysis ",
                                "reliable even without GPS data. Each feature contributes to the AI's understanding of your ",
                                "driving style and helps identify specific areas for improvement."
                            ], className="text-muted", style={'fontSize': '0.95rem'})
                        ], className="p-3")
                    ])
                ], active_tab="speed-features", className="mt-3"),

                # Summary Footer
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
        }),

                # Quick Track Selection (Radio Buttons) - Enhanced Design
        dbc.Card([
            dbc.CardBody([
                # Header Section - Left Aligned
                html.Div([
                    html.H3([
                        html.I(className="fas fa-flag-checkered me-3", style={
                            'color': '#e74c3c',
                            'fontSize': '2.2rem'
                        }),
                        "SELECT YOUR TRACK"
                    ], style={
                        'textAlign': 'left',
                        'fontWeight': '700',
                        'letterSpacing': '2px',
                        'color': '#2c3e50',
                        'marginBottom': '0.5rem',
                        'fontFamily': 'Montserrat, Arial, sans-serif',
                        'textTransform': 'uppercase'
                    }),
                    html.P("Click any track below to instantly load sample data and start analysis",
                        style={
                            'textAlign': 'left',
                            'color': '#7f8c8d',
                            'fontSize': '1.0rem',
                            'fontWeight': '400',
                            'marginBottom': '1.5rem'
                        }
                    )
                ], style={
                    'background': 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
                    'padding': '1.5rem 2rem',
                    'borderRadius': '10px 10px 0 0',
                    'marginBottom': '1.5rem',
                    'boxShadow': '0 2px 8px rgba(102, 126, 234, 0.3)'
                }),

                # Radio Buttons - Left Aligned
                html.Div([
                    dcc.RadioItems(
                        id='post-race-track-radio',
                        options=[
                            {'label': '🏁  Circuit of the Americas (COTA)', 'value': 'circuit-of-the-americas'},
                            {'label': '🏁  Road America', 'value': 'road-america'},
                            {'label': '🏁  Sonoma Raceway', 'value': 'sonoma'},
                            {'label': '🏁  Virginia International Raceway', 'value': 'virginia-international-raceway'},
                            {'label': '🏁  Barber Motorsports Park', 'value': 'barber-motorsports-park'},
                            {'label': '🏁  Sebring International Raceway', 'value': 'sebring'}
                        ],
                        value=None,
                        inline=False,
                        labelStyle={
                            'display': 'block',
                            'margin': '0.8rem 0',
                            'padding': '1rem 2rem',
                            'cursor': 'pointer',
                            'fontSize': '1.2rem',
                            'fontWeight': '600',
                            'fontFamily': 'Roboto, Arial, sans-serif',
                            'color': '#34495e',
                            'backgroundColor': '#ffffff',
                            'border': '2px solid #ecf0f1',
                            'borderRadius': '12px',
                            'transition': 'all 0.3s ease',
                            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                            'textAlign': 'left',
                            'maxWidth': '800px',
                            'letterSpacing': '0.5px',
                            'width': '100%'
                        },
                        inputStyle={
                            'marginRight': '15px',
                            'cursor': 'pointer',
                            'width': '20px',
                            'height': '20px',
                            'accentColor': '#e74c3c'
                        },
                        style={
                            'textAlign': 'left',
                            'width': '100%'
                        }
                    ),
                ], style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'flex-start',
                    'padding': '1rem',
                    'width': '100%'
                }),

                # Status Message
                html.Div(id='post-race-track-load-status', style={
                    'marginTop': '2rem',
                    'textAlign': 'center'
                })
            ], style={
                'padding': '0'
            })
        ], className="mb-4", style={
            'border': 'none',
            'borderRadius': '15px',
            'boxShadow': '0 8px 30px rgba(0,0,0,0.12)',
            'overflow': 'hidden'
        }),

        # Control Panel
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    # Upload section
                    dbc.Col([
                        html.Label("📤 Upload Session Data:", className="fw-bold"),
                        dcc.Upload(
                            id='post-race-upload',
                            children=html.Div([
                                '🔼 Drag and Drop or ',
                                html.A('Select Telemetry CSV')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'backgroundColor': COLORS['background']
                            },
                            multiple=False
                        )
                    ], width=4),

                    # Session selector
                    dbc.Col([
                        html.Label("🏁 Select Track:", className="fw-bold"),
                        dcc.Dropdown(
                            id='post-race-track-dropdown',
                            options=[
                                {'label': 'Circuit of the Americas', 'value': 'circuit-of-the-americas'},
                                {'label': 'Road America', 'value': 'road-america'},
                                {'label': 'Sonoma Raceway', 'value': 'sonoma'},
                                {'label': 'Virginia Int. Raceway', 'value': 'virginia-international-raceway'},
                                {'label': 'Barber Motorsports Park', 'value': 'barber-motorsports-park'},
                                {'label': 'Sebring', 'value': 'sebring'}
                            ],
                            placeholder="Choose a track...",
                            className="mb-2"
                        ),
                        html.Label("🏎️ Select Race:", className="fw-bold mt-2"),
                        dcc.Dropdown(
                            id='post-race-race-dropdown',
                            options=[],  # Populated by callback
                            placeholder="Choose a race session..."
                        )
                    ], width=4),

                    # Driver selector
                    dbc.Col([
                        html.Label("👤 Select Drivers:", className="fw-bold"),
                        dcc.Dropdown(
                            id='post-race-drivers-dropdown',
                            options=[],  # Populated by callback
                            multi=True,
                            placeholder="Choose drivers (optional)..."
                        ),
                        dbc.Button(
                            "🔄 Analyze Session",
                            id='post-race-analyze-btn',
                            color="primary",
                            className="mt-3 w-100",
                            size="lg"
                        )
                    ], width=4)
                ])
            ])
        ], className="mb-4"),

        # Status/Error messages
        dbc.Alert(
            id='post-race-status-message',
            children="👆 Upload telemetry or select a track/race to begin analysis",
            color="info",
            dismissable=True,
            is_open=True
        ),

        # Loading state
        dcc.Loading(
            id="post-race-loading",
            type="default",
            children=[
                # Hidden div to store processed data
                dcc.Store(id='post-race-data-store'),

                # Section 1: Lap-by-Lap Timeline
                dbc.Card([
                    dbc.CardHeader(html.H5("📈 Lap-by-Lap Performance Timeline", className="mb-0")),
                    dbc.CardBody([
                        dcc.Graph(
                            id='post-race-timeline',
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ])
                ], className="mb-4", id='post-race-timeline-card', style={'display': 'none'}),

                # Section 2: Error Distribution (Full Width)
                dbc.Card([
                    dbc.CardHeader(html.H5("📊 Error Distribution", className="mb-0")),
                    dbc.CardBody([
                        dcc.Graph(
                            id='post-race-error-histogram',
                            config={'displayModeBar': True, 'displaylogo': False},
                            style={'height': '500px'}
                        )
                    ])
                ], className="mb-4", id='post-race-error-card', style={'display': 'none'}),

                # Section 3: Session Statistics
                dbc.Card([
                    dbc.CardHeader(html.H5("📋 Session Statistics", className="mb-0")),
                    dbc.CardBody([
                        html.Div(id='post-race-statistics-table')
                    ])
                ], className="mb-4", id='post-race-statistics-card', style={'display': 'none'}),

                # Section 4: Anomaly Details
                dbc.Card([
                    dbc.CardHeader(html.H5("🔍 Anomaly Details (Problem Laps - Top 20)", className="mb-0")),
                    dbc.CardBody([
                        html.Div(id='post-race-anomalies-table')
                    ])
                ], className="mb-4", id='post-race-anomalies-card', style={'display': 'none'}),

                # Section 5: Recommendations
                dbc.Card([
                    dbc.CardHeader(html.H5("💡 AI Coaching Recommendations", className="mb-0")),
                    dbc.CardBody([
                        html.Div(id='post-race-recommendations')
                    ])
                ], className="mb-4", id='post-race-recommendations-card', style={'display': 'none'}),

                # Export Controls
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "📥 Download CSV Report",
                                    id="post-race-download-csv-btn",
                                    color="primary",
                                    className="me-2",
                                    disabled=True
                                ),
                                dbc.Button(
                                    "📄 Generate Summary Report",
                                    id="post-race-download-summary-btn",
                                    color="secondary",
                                    disabled=True
                                ),
                                dcc.Download(id="post-race-download-csv"),
                                dcc.Download(id="post-race-download-summary")
                            ])
                        ])
                    ])
                ], id='post-race-export-card', style={'display': 'none'})
            ]
        )
    ], fluid=True)




# ============================================================================
# AUTO-LOAD CALLBACK
# ============================================================================

def _create_autoload_callback(app):
    """
    Auto-load master_racing_data.csv when Tab 5 is accessed

    Triggers when user navigates to post-race tab
    Only loads once per session (cached in dcc.Store)
    """
    from dash import Output, Input, State
    from dash.exceptions import PreventUpdate

    @app.callback(
        [Output('post-race-autoload-store', 'data'),
         Output('post-race-autoload-status', 'children')],
        [Input('tabs', 'active_tab')],  # FIXED: Use 'active_tab' instead of 'value' for dbc.Tabs
        [State('post-race-autoload-store', 'data')]
    )
    def auto_load_master_csv(active_tab, current_data):
        """Auto-load CSV when tab becomes active"""
        # Only trigger on post-race tab
        if active_tab not in ['tab-post-race', 'tab-8', 'tab-5']:  # Handle multiple tab ID variations
            raise PreventUpdate

        # Don't reload if data already exists
        if current_data is not None:
            raise PreventUpdate

        import platform
        from pathlib import Path
        import pandas as pd

        # Platform detection
        is_production = platform.system() == 'Linux'

        # Determine file path
        if is_production:
            csv_path = Path('/home/tactical/racing_analytics/data/master_racing_data.csv')
        else:
            csv_path = Path('data/master_racing_data.csv')

        # Check if file exists
        if not csv_path.exists():
            error_msg = dbc.Alert(
                [
                    html.H6("Auto-Load Failed", className="alert-heading"),
                    html.P(f"File not found: {csv_path}"),
                    html.Hr(),
                    html.P("Please upload a telemetry file manually.", className="mb-0")
                ],
                color="warning",
                dismissable=True
            )
            return None, error_msg

        try:
            # Load CSV
            df = pd.read_csv(csv_path)

            # Validate required columns
            required_cols = ['telemetry_name', 'telemetry_value', 'vehicle_number',
                            'timestamp', 'lap', 'source_file']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                error_msg = dbc.Alert(
                    [
                        html.H6("Invalid CSV Format", className="alert-heading"),
                        html.P(f"Missing columns: {', '.join(missing_cols)}")
                    ],
                    color="danger",
                    dismissable=True
                )
                return None, error_msg

            # Store data
            data = {
                'telemetry': df.to_dict('records'),
                'filename': csv_path.name,
                'rows': len(df),
                'vehicles': int(df['vehicle_number'].nunique()),
                'sources': int(df['source_file'].nunique()),
                'columns': list(df.columns)
            }

            # Success message
            success_msg = dbc.Alert(
                [
                    html.H6([
                        html.I(className="fas fa-check-circle me-2"),
                        "Auto-Load Successful"
                    ], className="alert-heading"),
                    html.P([
                        f"Loaded ",
                        html.Strong(f"{len(df):,}"),
                        " telemetry samples from ",
                        html.Code(csv_path.name)
                    ]),
                    html.Hr(),
                    html.Div([
                        dbc.Badge(f"{data['vehicles']} vehicles", color="primary", className="me-2"),
                        dbc.Badge(f"{data['sources']} race sessions", color="info", className="me-2"),
                        dbc.Badge(f"{len(data['columns'])} columns", color="secondary")
                    ])
                ],
                color="success",
                dismissable=True,
                className="mb-0"
            )

            print(f"[AUTO-LOAD] Successfully loaded {len(df):,} rows from {csv_path.name}")
            return data, success_msg

        except Exception as e:
            error_msg = dbc.Alert(
                [
                    html.H6("Error Loading CSV", className="alert-heading"),
                    html.P(f"{str(e)}"),
                    html.Hr(),
                    html.P("Please check the file format and try uploading manually.", className="mb-0")
                ],
                color="danger",
                dismissable=True
            )
            print(f"[AUTO-LOAD ERROR] {e}")
            return None, error_msg


def _create_radio_load_callback(app):
    """
    Auto-load CSV template when user selects a track via radio button

    This provides a quick way to load and analyze sample data without manual upload
    """
    from dash import Output, Input
    from dash.exceptions import PreventUpdate
    import platform
    from pathlib import Path
    import pandas as pd

    @app.callback(
        [Output('post-race-radio-data-store', 'data'),
         Output('post-race-track-load-status', 'children')],
        [Input('post-race-track-radio', 'value')]
    )
    def load_track_template(selected_track):
        """Load CSV template when track radio button is selected"""
        print(f"[RADIO-LOAD] Callback triggered with track: {selected_track}")

        if not selected_track:
            print(f"[RADIO-LOAD] No track selected, preventing update")
            raise PreventUpdate

        # Determine template path (cross-platform)
        is_production = platform.system() == 'Linux'

        if is_production:
            template_dir = Path('/home/tactical/racing_analytics/post_race_templates')
        else:
            template_dir = Path('post_race_templates')

        template_file = template_dir / f"{selected_track}_sample_template.csv"

        if not template_file.exists():
            error_msg = dbc.Alert(
                f"Template not found: {template_file.name}",
                color="danger",
                dismissable=True
            )
            return None, error_msg

        try:
            # Load CSV
            df = pd.read_csv(template_file)

            # Validate format
            required_cols = ['timestamp', 'lap', 'vehicle_number', 'telemetry_name',
                            'telemetry_value', 'track', 'race']
            missing = [col for col in required_cols if col not in df.columns]

            if missing:
                error_msg = dbc.Alert(
                    f"Invalid CSV format. Missing columns: {', '.join(missing)}",
                    color="danger",
                    dismissable=True
                )
                return None, error_msg

            # Store data
            data = {
                'telemetry': df.to_dict('records'),
                'filename': template_file.name,
                'track': selected_track
            }

            # Success message
            success_msg = dbc.Alert(
                [
                    html.I(className="fas fa-check-circle me-2"),
                    html.Strong("Loaded: "),
                    f"{template_file.name} ",
                    dbc.Badge(f"{len(df):,} rows", color="primary", className="ms-2"),
                    dbc.Badge(f"{df['lap'].nunique()} laps", color="info", className="ms-2"),
                    dbc.Badge(f"Vehicle #{int(df['vehicle_number'].iloc[0])}", color="success", className="ms-2")
                ],
                color="success",
                dismissable=True,
                className="mb-0"
            )

            print(f"[RADIO-LOAD] Loaded {template_file.name}: {len(df):,} rows")
            return data, success_msg

        except Exception as e:
            error_msg = dbc.Alert(
                f"Error loading template: {str(e)}",
                color="danger",
                dismissable=True
            )
            print(f"[RADIO-LOAD ERROR] {e}")
            return None, error_msg


def _create_driver_dropdown_callback(app):
    """
    Populate driver dropdown when data is loaded via radio button or upload

    Updates the driver selection dropdown with all available vehicles
    from the loaded telemetry data.
    """
    from dash import Output, Input
    from dash.exceptions import PreventUpdate
    import pandas as pd

    @app.callback(
        Output('post-race-drivers-dropdown', 'options', allow_duplicate=True),
        [Input('post-race-radio-data-store', 'data'),
         Input('post-race-autoload-store', 'data')],
        prevent_initial_call=True
    )
    def populate_driver_dropdown(radio_data, autoload_data):
        """Populate driver dropdown with available vehicles"""
        print(f"[DRIVER-DROPDOWN] Callback triggered")
        print(f"[DRIVER-DROPDOWN] radio_data exists: {radio_data is not None}")
        print(f"[DRIVER-DROPDOWN] autoload_data exists: {autoload_data is not None}")

        # Priority: radio data > autoload data
        data_source = radio_data or autoload_data

        if not data_source:
            print(f"[DRIVER-DROPDOWN] No data source, preventing update")
            raise PreventUpdate

        try:
            # Extract telemetry from data store
            telemetry_df = pd.DataFrame(data_source['telemetry'])

            # Get unique vehicle numbers
            vehicles = sorted(telemetry_df['vehicle_number'].unique())

            # Create dropdown options
            options = [
                {'label': f'Vehicle #{int(v)}', 'value': int(v)}
                for v in vehicles
            ]

            print(f"[DRIVER-DROPDOWN] Populated with {len(options)} vehicles: {[int(v) for v in vehicles]}")
            return options

        except Exception as e:
            print(f"[DRIVER-DROPDOWN ERROR] {e}")
            raise PreventUpdate


def create_post_race_callbacks(app):
    """
    Register callbacks for Post-Race Analysis tab

    Args:
        app: Dash app instance
    """

    # Register auto-load callback
    _create_autoload_callback(app)

    # Register radio button auto-load callback
    _create_radio_load_callback(app)

    # Register driver dropdown population callback
    _create_driver_dropdown_callback(app)

    # Initialize predictor and analyzer
    # Use SimplePostRacePredictor for better compatibility with minimal sensor data
    try:
        predictor = SimplePostRacePredictor()
        print("[INFO] Using SimplePostRacePredictor (basic features only)")
    except Exception as e:
        print(f"[WARNING] SimplePostRacePredictor failed: {e}")
        print("[INFO] Falling back to PostRacePredictor")
        predictor = PostRacePredictor()

    analyzer = PostRaceAnalyzer(anomaly_threshold=2.5)

    @app.callback(
        [Output('post-race-timeline', 'figure'),
         Output('post-race-error-histogram', 'figure'),
         Output('post-race-statistics-table', 'children'),
         Output('post-race-anomalies-table', 'children'),
         Output('post-race-recommendations', 'children'),
         Output('post-race-status-message', 'children'),
         Output('post-race-status-message', 'color'),
         Output('post-race-status-message', 'is_open'),
         Output('post-race-data-store', 'data'),
         Output('post-race-timeline-card', 'style'),
         Output('post-race-error-card', 'style'),
         Output('post-race-statistics-card', 'style'),
         Output('post-race-anomalies-card', 'style'),
         Output('post-race-recommendations-card', 'style'),
         Output('post-race-export-card', 'style'),
         Output('post-race-download-csv-btn', 'disabled'),
         Output('post-race-download-summary-btn', 'disabled')],
        [Input('post-race-analyze-btn', 'n_clicks'),
         Input('post-race-radio-data-store', 'data')],  # Auto-trigger when radio selection loads data
        [State('post-race-upload', 'contents'),
         State('post-race-upload', 'filename'),
         State('post-race-track-dropdown', 'value'),
         State('post-race-race-dropdown', 'value'),
         State('post-race-drivers-dropdown', 'value')],
        prevent_initial_call=True
    )
    def update_post_race_analysis(n_clicks, radio_data, upload_contents, filename,
                                  track, race, driver_ids):
        """
        Main callback: Process data and generate visualizations

        Can be triggered by:
        1. Clicking "Analyze Session" button with uploaded data
        2. Selecting a track via radio button (auto-loads and analyzes)

        Returns:
            Tuple of 17 outputs for all dashboard components
        """
        try:
            # Step 1: Load data (priority: radio selection > upload > track/race dropdown)
            if radio_data:
                # Data loaded via radio button selection
                telemetry_df = pd.DataFrame(radio_data['telemetry'])
                lap_times_df = calculate_lap_times_from_telemetry(telemetry_df)
                source = f"template: {radio_data['filename']}"
            elif upload_contents:
                # Data uploaded via file upload
                telemetry_df, lap_times_df = parse_upload(upload_contents, filename)
                source = f"uploaded file: {filename}"
            elif track and race:
                # Data selected via dropdown
                telemetry_df, lap_times_df = load_session_data(track, race)
                source = f"{track} - {race}"
            else:
                return empty_outputs_with_message(
                    "❌ Please upload a file, select a track via radio button, or choose track/race from dropdowns",
                    "warning"
                )

            # Step 2: Filter by drivers if specified
            if driver_ids:
                telemetry_df = telemetry_df[telemetry_df['vehicle_number'].isin(driver_ids)].copy()
                lap_times_df = lap_times_df[lap_times_df['vehicle_number'].isin(driver_ids)].copy()

            # Step 3: Make predictions
            predictions_df = predictor.predict_session(telemetry_df, lap_times_df, driver_ids)

            if len(predictions_df) == 0:
                return empty_outputs_with_message(
                    "❌ No valid laps found. Check data format.",
                    "danger"
                )

            # Step 4: Analyze results
            analysis = analyzer.analyze_session(predictions_df)

            # Step 5: Create visualizations
            timeline_fig = create_timeline_chart(predictions_df)
            histogram_fig = create_error_histogram(predictions_df)
            stats_table = create_statistics_table(analysis['statistics'])
            anomalies_table = create_anomalies_table(analysis['anomalies'])
            recommendations_div = create_recommendations_div(analysis['recommendations'])

            # Success message
            status_msg = f"✅ Analysis complete! Processed {len(predictions_df)} laps from {source}"
            status_color = "success"

            # Store data for export
            data_store = predictions_df.to_json(date_format='iso', orient='split')

            # Show all cards
            show_style = {'display': 'block'}

            return (
                timeline_fig, histogram_fig, stats_table, anomalies_table,
                recommendations_div, status_msg, status_color, True, data_store,
                show_style, show_style, show_style, show_style, show_style, show_style,  # 6 cards
                False, False  # Enable download buttons
            )

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            return empty_outputs_with_message(error_msg, "danger")

    def create_timeline_chart(df: pd.DataFrame) -> go.Figure:
        """Create lap-by-lap timeline chart"""
        fig = go.Figure()

        # Group by vehicle for multi-driver support
        for vehicle in df['vehicle_number'].unique():
            vehicle_df = df[df['vehicle_number'] == vehicle]

            # Actual lap times (solid line)
            fig.add_trace(go.Scatter(
                x=vehicle_df['lap_number'],
                y=vehicle_df['actual'],
                mode='lines+markers',
                name=f'Vehicle {vehicle} - Actual',
                line=dict(color=COLORS['actual'], width=2),
                marker=dict(size=6),
                hovertemplate='<b>Lap %{x}</b><br>' +
                             'Actual: %{y:.3f}s<br>' +
                             '<extra></extra>'
            ))

            # Predicted lap times (dashed line)
            fig.add_trace(go.Scatter(
                x=vehicle_df['lap_number'],
                y=vehicle_df['predicted'],
                mode='lines',
                name=f'Vehicle {vehicle} - Predicted',
                line=dict(color=COLORS['predicted'], width=2, dash='dash'),
                hovertemplate='<b>Lap %{x}</b><br>' +
                             'Predicted: %{y:.3f}s<br>' +
                             '<extra></extra>'
            ))

            # Anomalies (red markers)
            anomalies = vehicle_df[vehicle_df['abs_error'] > 2.5]
            if len(anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=anomalies['lap_number'],
                    y=anomalies['actual'],
                    mode='markers',
                    name=f'Vehicle {vehicle} - Anomalies',
                    marker=dict(size=14, color=COLORS['anomaly'], symbol='x', line=dict(width=2)),
                    hovertemplate='<b>Lap %{x}</b><br>' +
                                 'Actual: %{y:.3f}s<br>' +
                                 'Error: %{customdata:.3f}s<br>' +
                                 '<b>⚠️ ANOMALY</b><extra></extra>',
                    customdata=anomalies['error']
                ))

        fig.update_layout(
            title='Lap-by-Lap Performance: Actual vs. Predicted',
            xaxis_title='Lap Number',
            yaxis_title='Lap Time (seconds)',
            hovermode='x unified',
            height=450,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    def create_error_histogram(df: pd.DataFrame) -> go.Figure:
        """Create error distribution histogram"""
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=df['error'],
            nbinsx=30,
            marker=dict(
                color=COLORS['actual'],
                line=dict(color='white', width=1)
            ),
            name='Error Distribution',
            hovertemplate='Error: %{x:.2f}s<br>Count: %{y}<extra></extra>'
        ))

        # Add vertical line at zero
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color=COLORS['anomaly'],
            line_width=2,
            annotation_text="Perfect Prediction",
            annotation_position="top"
        )

        # Add threshold lines
        fig.add_vline(x=2.5, line_dash="dot", line_color=COLORS['anomaly'],
                     annotation_text="+2.5s", annotation_position="top")
        fig.add_vline(x=-2.5, line_dash="dot", line_color=COLORS['anomaly'],
                     annotation_text="-2.5s", annotation_position="top")

        fig.update_layout(
            title='Prediction Error Distribution',
            xaxis_title='Error (seconds)',
            yaxis_title='Count',
            height=500,  # Increased height for full-width display
            template='plotly_white',
            showlegend=False
        )

        return fig

    def create_statistics_table(stats) -> html.Div:
        """Create session statistics table"""
        return dbc.Table([
            html.Tbody([
                html.Tr([html.Td("Total Laps:", className="fw-bold"),
                        html.Td(f"{stats.total_laps}")]),
                html.Tr([html.Td("Average Lap Time:", className="fw-bold"),
                        html.Td(f"{stats.avg_lap_time:.3f}s")]),
                html.Tr([html.Td("Best Lap:", className="fw-bold"),
                        html.Td(f"{stats.best_lap:.3f}s", className="text-success fw-bold")]),
                html.Tr([html.Td("Worst Lap:", className="fw-bold"),
                        html.Td(f"{stats.worst_lap:.3f}s", className="text-danger")]),
                html.Tr([html.Td("Std Deviation:", className="fw-bold"),
                        html.Td(f"{stats.std_lap_time:.3f}s")]),
                html.Tr([html.Td("Consistency Rating:", className="fw-bold"),
                        html.Td(stats.consistency_rating,
                               className=get_consistency_color(stats.consistency_rating))]),
                html.Tr([html.Td(html.Hr(), colSpan=2)]),
                html.Tr([html.Td("Model MAE:", className="fw-bold"),
                        html.Td(f"{stats.model_mae:.3f}s")]),
                html.Tr([html.Td("Model RMSE:", className="fw-bold"),
                        html.Td(f"{stats.model_rmse:.3f}s")]),
                html.Tr([html.Td("Model R²:", className="fw-bold"),
                        html.Td(f"{stats.model_r2:.1%}")]),
                html.Tr([html.Td(html.Hr(), colSpan=2)]),
                html.Tr([html.Td("Anomalies Detected:", className="fw-bold"),
                        html.Td(f"{stats.anomaly_count} laps",
                               className="text-danger fw-bold" if stats.anomaly_count > 0 else "text-success")]),
                html.Tr([html.Td("Max Error:", className="fw-bold"),
                        html.Td(f"{stats.max_error:.3f}s")])
            ])
        ], bordered=True, hover=True, striped=True, size="sm", className="mb-0")

    def get_consistency_color(rating: str) -> str:
        """Get CSS class for consistency rating"""
        colors = {
            'Excellent': 'text-success fw-bold',
            'Good': 'text-primary fw-bold',
            'Fair': 'text-warning fw-bold',
            'Poor': 'text-danger fw-bold'
        }
        return colors.get(rating, '')

    def create_anomalies_table(anomalies: pd.DataFrame) -> html.Div:
        """Create anomalies table - shows top 20 problem laps"""
        if len(anomalies) == 0:
            return html.Div([
                html.P("✅ No anomalies detected!", className="text-success fw-bold mb-2"),
                html.P("All laps within expected range (error < 2.5s)", className="text-muted")
            ])

        # Create formatted table - show top 20 worst laps
        table_data = anomalies.head(20).to_dict('records')

        return dash_table.DataTable(
            data=table_data,
            columns=[
                {'name': 'Lap', 'id': 'lap_number'},
                {'name': 'Vehicle', 'id': 'vehicle_number'},
                {'name': 'Actual', 'id': 'actual', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                {'name': 'Predicted', 'id': 'predicted', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                {'name': 'Error', 'id': 'error', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                {'name': 'Likely Cause', 'id': 'likely_cause'},
                {'name': 'Severity', 'id': 'severity'}
            ],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': COLORS['background'], 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{severity} = "Critical"'},
                    'backgroundColor': '#ffcccc'
                },
                {
                    'if': {'filter_query': '{severity} = "High"'},
                    'backgroundColor': '#ffe6cc'
                }
            ]
        )

    def create_recommendations_div(recommendations: List[str]) -> html.Div:
        """Create recommendations display"""
        if not recommendations:
            return html.P("No specific recommendations at this time.", className="text-muted")

        return html.Div([
            dbc.Alert(
                [
                    html.P(rec, className="mb-2")
                    for rec in recommendations
                ],
                color="info"
            )
        ])

    def empty_outputs_with_message(message: str, color: str):
        """Return empty outputs with error message"""
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No Data",
            template='plotly_white',
            annotations=[dict(text="Upload data to begin analysis",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False)]
        )

        hide_style = {'display': 'none'}

        return (
            empty_fig, empty_fig, "", "", "",  # Empty visualizations
            message, color, True, None,  # Status message
            hide_style, hide_style, hide_style, hide_style, hide_style, hide_style,  # Hide all cards
            True, True  # Disable buttons
        )

    # Callback to populate race dropdown based on track selection
    @app.callback(
        Output('post-race-race-dropdown', 'options'),
        Input('post-race-track-dropdown', 'value')
    )
    def update_race_dropdown(track):
        """
        Populate race options based on selected track

        Works in two modes:
        1. Development (Windows): Browse organized_data/ directory
        2. Production (Linux): Extract from master_racing_data.csv
        """
        if not track:
            return []

        import platform
        from pathlib import Path

        # Detect platform
        is_production = platform.system() == 'Linux'

        if is_production:
            # PRODUCTION MODE: Extract races from CSV
            try:
                import pandas as pd

                # Platform-aware CSV path
                if is_production:
                    csv_path = Path("/home/tactical/racing_analytics/data/master_racing_data.csv")
                else:
                    csv_path = Path("data/master_racing_data.csv")

                if not csv_path.exists():
                    print(f"Info: {csv_path} not found, race selection disabled")
                    return []

                # Read CSV and extract source_file column
                df = pd.read_csv(csv_path, usecols=['source_file'])

                # Extract track-specific files
                # source_file format: "track-name_session_type"
                track_files = df[df['source_file'].str.contains(track, case=False, na=False)]

                if track_files.empty:
                    return []

                # Extract unique race/session names
                unique_sessions = track_files['source_file'].unique()

                # Create options from unique sessions
                options = []
                for session in unique_sessions:
                    # Clean up display name and map to user-friendly format
                    session_suffix = session.replace(track + '_', '')

                    # Map common session types to user-friendly names
                    if session_suffix == 'full' or session.endswith('_full'):
                        display_name = 'Race 1'
                    elif session_suffix == 'simplified' or session.endswith('_simplified'):
                        display_name = 'Race 2'
                    elif session_suffix.startswith('race_'):
                        # Already in race_N format
                        display_name = session_suffix.replace('_', ' ').title()
                    else:
                        # Generic cleanup for other formats
                        display_name = session_suffix.replace('_', ' ').title()

                    options.append({'label': display_name, 'value': session})

                return sorted(options, key=lambda x: x['label'])

            except Exception as e:
                print(f"Error extracting races from CSV: {e}")
                import traceback
                traceback.print_exc()
                return []
        else:
            # DEVELOPMENT MODE: Browse organized_data/ directory
            try:
                from data_loader import RacingDataLoader
                loader = RacingDataLoader()
                races = loader.list_races(track)

                # Filter to only races with telemetry data
                valid_races = []
                for race in races:
                    try:
                        categories = loader.list_categories(track, race)
                        if 'telemetry' in categories:
                            telemetry_dir = Path(f"organized_data/{track}/{race}/telemetry")
                            if telemetry_dir.exists():
                                telemetry_files = list(telemetry_dir.glob("*.csv"))
                                if len(telemetry_files) > 0:
                                    valid_races.append(race)
                    except:
                        continue

                # Format race names nicely
                options = []
                for race in valid_races:
                    if race.endswith('.csv') or len(race) > 30:
                        continue
                    label = race.replace('_', ' ').replace('-', ' ').title()
                    options.append({'label': label, 'value': race})

                return sorted(options, key=lambda x: x['value'])

            except Exception as e:
                print(f"Error loading races (development mode): {e}")
                return []

    # Callback to populate driver dropdown based on track/race selection
    @app.callback(
        Output('post-race-drivers-dropdown', 'options'),
        [Input('post-race-track-dropdown', 'value'),
         Input('post-race-race-dropdown', 'value')]
    )
    def update_drivers_dropdown(track, race):
        """
        Populate driver options based on selected track and race

        Works in two modes:
        1. Development: Browse organized_data/
        2. Production: Extract from master_racing_data.csv
        """
        if not track or not race:
            return []

        import platform
        from pathlib import Path

        is_production = platform.system() == 'Linux'

        if is_production:
            # PRODUCTION MODE: Extract vehicles from CSV
            try:
                import pandas as pd

                # Platform-aware CSV path
                if is_production:
                    csv_path = Path("/home/tactical/racing_analytics/data/master_racing_data.csv")
                else:
                    csv_path = Path("data/master_racing_data.csv")

                if not csv_path.exists():
                    return []

                # Read CSV and filter by source_file (race)
                df = pd.read_csv(csv_path, usecols=['source_file', 'vehicle_number'])
                race_data = df[df['source_file'] == race]

                if race_data.empty:
                    return []

                # Get unique vehicle numbers
                vehicles = sorted(race_data['vehicle_number'].dropna().unique())
                return [{'label': f'Vehicle {int(v)}', 'value': int(v)} for v in vehicles if pd.notna(v)]

            except Exception as e:
                print(f"Error extracting vehicles from CSV: {e}")
                return []
        else:
            # DEVELOPMENT MODE: Browse organized_data/
            try:
                from data_loader import RacingDataLoader
                loader = RacingDataLoader()

                telemetry_dir = Path(f"organized_data/{track}/{race}/telemetry")
                if not telemetry_dir.exists():
                    return []

                chunk_files = sorted(telemetry_dir.glob("*.csv"))[:5]
                all_vehicles = set()

                for chunk_file in chunk_files:
                    chunk_num = int(chunk_file.stem.split('_')[-1])
                    chunk_df = loader.load_single_chunk(track, race, 'telemetry', chunk_num=chunk_num)
                    if 'vehicle_number' in chunk_df.columns:
                        all_vehicles.update(chunk_df['vehicle_number'].dropna().unique())

                vehicles = sorted(all_vehicles)
                return [{'label': f'Vehicle {int(v)}', 'value': int(v)} for v in vehicles]

            except Exception as e:
                print(f"Error loading drivers (development mode): {e}")
                return []

    # Download CSV callback
    @app.callback(
        Output("post-race-download-csv", "data"),
        Input("post-race-download-csv-btn", "n_clicks"),
        State("post-race-data-store", "data"),
        prevent_initial_call=True
    )
    def download_csv(n_clicks, data_json):
        """Export analysis results as CSV"""
        if not data_json:
            return None

        df = pd.read_json(data_json, orient='split')
        return dcc.send_data_frame(df.to_csv, "post_race_analysis.csv", index=False)

    # Download Summary Report callback
    @app.callback(
        Output("post-race-download-summary", "data"),
        Input("post-race-download-summary-btn", "n_clicks"),
        State("post-race-data-store", "data"),
        prevent_initial_call=True
    )
    def download_summary_report(n_clicks, data_json):
        """Generate and download comprehensive text summary report"""
        if not data_json:
            return None

        # Parse data
        df = pd.read_json(data_json, orient='split')

        # Re-analyze for summary
        analyzer_temp = PostRaceAnalyzer(anomaly_threshold=2.5)
        analysis = analyzer_temp.analyze_session(df)
        stats = analysis['statistics']
        anomalies = analysis['anomalies']
        recommendations = analysis['recommendations']

        # Build summary report
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("POST-RACE ANALYSIS - SUMMARY REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Session Overview
        report_lines.append("-" * 70)
        report_lines.append("SESSION OVERVIEW")
        report_lines.append("-" * 70)
        report_lines.append(f"Total Laps:                 {stats.total_laps}")
        report_lines.append(f"Vehicles Analyzed:          {df['vehicle_number'].nunique()}")
        report_lines.append(f"Track:                      {df['track'].iloc[0] if 'track' in df.columns else 'Unknown'}")
        report_lines.append(f"Race:                       {df['race'].iloc[0] if 'race' in df.columns else 'Unknown'}")
        report_lines.append("")

        # Performance Statistics
        report_lines.append("-" * 70)
        report_lines.append("PERFORMANCE STATISTICS")
        report_lines.append("-" * 70)
        report_lines.append(f"Average Lap Time:           {stats.avg_lap_time:.3f} seconds")
        report_lines.append(f"Best Lap Time:              {stats.best_lap:.3f} seconds")
        report_lines.append(f"Worst Lap Time:             {stats.worst_lap:.3f} seconds")
        report_lines.append(f"Standard Deviation:         {stats.std_lap_time:.3f} seconds")
        report_lines.append(f"Lap Time Range:             {stats.worst_lap - stats.best_lap:.3f} seconds")
        report_lines.append(f"Consistency Rating:         {stats.consistency_rating}")
        report_lines.append("")

        # Model Performance
        report_lines.append("-" * 70)
        report_lines.append("MODEL PREDICTION ACCURACY")
        report_lines.append("-" * 70)
        report_lines.append(f"Mean Absolute Error (MAE):  {stats.model_mae:.3f} seconds")
        report_lines.append(f"Root Mean Square Error:     {stats.model_rmse:.3f} seconds")
        report_lines.append(f"R² Score:                   {stats.model_r2:.1%}")
        report_lines.append(f"Max Prediction Error:       {stats.max_error:.3f} seconds")
        report_lines.append("")

        # Anomaly Detection
        report_lines.append("-" * 70)
        report_lines.append("ANOMALY DETECTION")
        report_lines.append("-" * 70)
        report_lines.append(f"Anomalies Detected:         {stats.anomaly_count} laps")
        report_lines.append(f"Anomaly Rate:               {stats.anomaly_count / stats.total_laps * 100:.1f}%")
        report_lines.append("")

        if len(anomalies) > 0:
            report_lines.append("Top Problem Laps (sorted by error magnitude):")
            report_lines.append("")
            report_lines.append(f"{'Lap':<6} {'Vehicle':<8} {'Actual':<10} {'Predicted':<12} {'Error':<10} {'Severity':<10} Cause")
            report_lines.append("-" * 70)
            for _, row in anomalies.head(10).iterrows():
                report_lines.append(
                    f"{int(row['lap_number']):<6} "
                    f"{int(row['vehicle_number']):<8} "
                    f"{row['actual']:<10.3f} "
                    f"{row['predicted']:<12.3f} "
                    f"{row['error']:>9.3f} "
                    f"{row['severity']:<10} "
                    f"{row['likely_cause']}"
                )
            report_lines.append("")
        else:
            report_lines.append("No anomalies detected - all laps within expected range.")
            report_lines.append("")

        # Coaching Recommendations
        if recommendations:
            report_lines.append("-" * 70)
            report_lines.append("AI COACHING RECOMMENDATIONS")
            report_lines.append("-" * 70)
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")

        # Per-Vehicle Breakdown (if multiple vehicles)
        if df['vehicle_number'].nunique() > 1:
            report_lines.append("-" * 70)
            report_lines.append("PER-VEHICLE BREAKDOWN")
            report_lines.append("-" * 70)
            for vehicle in sorted(df['vehicle_number'].unique()):
                vehicle_df = df[df['vehicle_number'] == vehicle]
                report_lines.append(f"Vehicle #{int(vehicle)}:")
                report_lines.append(f"  Laps:           {len(vehicle_df)}")
                report_lines.append(f"  Avg Lap Time:   {vehicle_df['actual'].mean():.3f}s")
                report_lines.append(f"  Best Lap:       {vehicle_df['actual'].min():.3f}s")
                report_lines.append(f"  Std Dev:        {vehicle_df['actual'].std():.3f}s")
                report_lines.append(f"  Avg Error:      {vehicle_df['abs_error'].mean():.3f}s")
                anomaly_count = len(vehicle_df[vehicle_df['abs_error'] > 2.5])
                report_lines.append(f"  Anomalies:      {anomaly_count}")
                report_lines.append("")

        # Footer
        report_lines.append("=" * 70)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 70)

        # Join lines and return as downloadable text file
        report_text = "\n".join(report_lines)
        return dict(content=report_text, filename="post_race_summary_report.txt")


def calculate_lap_times_from_telemetry(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate lap times from telemetry data

    Args:
        telemetry_df: Telemetry in long format with timestamp, lap, vehicle_number columns

    Returns:
        DataFrame with lap_number, vehicle_number, lap_time, track, race
    """
    lap_times_list = []

    # Ensure required columns exist
    if 'vehicle_number' not in telemetry_df.columns:
        raise ValueError("CSV must contain 'vehicle_number' column")

    if 'lap' not in telemetry_df.columns:
        raise ValueError("CSV must contain 'lap' column")

    # Convert timestamp to numeric if needed
    if 'timestamp' in telemetry_df.columns:
        if telemetry_df['timestamp'].dtype == 'object':
            telemetry_df['timestamp'] = pd.to_datetime(telemetry_df['timestamp'])
            telemetry_df['timestamp'] = telemetry_df['timestamp'].astype(np.int64) // 10**6  # Convert to milliseconds

    # Group by vehicle and lap to calculate lap times
    for (vehicle, lap), group in telemetry_df.groupby(['vehicle_number', 'lap']):
        if 'timestamp' in group.columns and len(group) > 0:
            min_time = group['timestamp'].min()
            max_time = group['timestamp'].max()
            lap_time = (max_time - min_time) / 1000.0  # Convert to seconds

            # Get track and race info
            track = group['track'].iloc[0] if 'track' in group.columns and len(group) > 0 else 'unknown'
            race = group['race'].iloc[0] if 'race' in group.columns and len(group) > 0 else 'unknown'

            # Only add valid lap times (60-300 seconds)
            if 60 <= lap_time <= 300:
                lap_times_list.append({
                    'vehicle_number': vehicle,
                    'lap_number': int(lap),
                    'lap_time': lap_time,
                    'track': track,
                    'race': race
                })

    lap_times_df = pd.DataFrame(lap_times_list)

    if len(lap_times_df) == 0:
        raise ValueError("No valid lap times found (laps must be 60-300 seconds)")

    return lap_times_df


def parse_upload(contents: str, filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse uploaded CSV file

    Args:
        contents: Base64 encoded file contents
        filename: Original filename

    Returns:
        Tuple of (telemetry_df, lap_times_df)
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # Read CSV
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Check if data is in wide format (separate columns) or long format (telemetry_name/value)
    if 'telemetry_name' in df.columns and 'telemetry_value' in df.columns:
        # Already in long format
        telemetry_df = df.copy()
    else:
        # Convert wide format to long format
        # Identify sensor columns (exclude metadata columns)
        metadata_cols = ['timestamp', 'lap', 'vehicle_number', 'track', 'race', 'distance']
        sensor_cols = [col for col in df.columns if col not in metadata_cols]

        # Melt to long format
        id_vars = [col for col in metadata_cols if col in df.columns]
        telemetry_df = df.melt(
            id_vars=id_vars,
            value_vars=sensor_cols,
            var_name='telemetry_name',
            value_name='telemetry_value'
        )

        # Drop NaN values
        telemetry_df = telemetry_df.dropna(subset=['telemetry_value'])

    # Calculate lap times using helper function
    lap_times_df = calculate_lap_times_from_telemetry(telemetry_df)

    return telemetry_df, lap_times_df


# ============================================================================
# EXPORTABLE COMPONENTS FOR MAIN DASHBOARD
# ============================================================================

def create_sensor_status_card():
    """
    Create standalone sensor status card for main dashboard

    Returns:
        dbc.Card: Sensor status and data quality card
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-wifi me-2", style={'color': '#2ecc71'}),
                "Sensor Status & Data Quality"
            ], className="mb-0")
        ], style={'backgroundColor': '#f8f9fa'}),
        dbc.CardBody([
            dbc.Row([
                # Left: Sensor checklist
                dbc.Col([
                    html.H6("Available Sensors (9/12)", className="mb-3"),
                    html.Div([
                        # Present sensors - green checkmarks
                        html.Div([
                            html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                            html.Span("Speed", className="fw-bold"),
                            html.Span(" (km/h)", className="text-muted ms-1")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                            html.Span("Brake Pressure Front", className="fw-bold"),
                            html.Span(" (bar)", className="text-muted ms-1")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                            html.Span("Brake Pressure Rear", className="fw-bold"),
                            html.Span(" (bar)", className="text-muted ms-1")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                            html.Span("Throttle Position", className="fw-bold"),
                            html.Span(" (%)", className="text-muted ms-1")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                            html.Span("Acceleration X", className="fw-bold"),
                            html.Span(" (g)", className="text-muted ms-1")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                            html.Span("Acceleration Y", className="fw-bold"),
                            html.Span(" (g)", className="text-muted ms-1")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                            html.Span("Steering Angle", className="fw-bold"),
                            html.Span(" (deg)", className="text-muted ms-1")
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                            html.Span("Gear", className="fw-bold"),
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-check-circle me-2", style={'color': '#2ecc71'}),
                            html.Span("Engine RPM", className="fw-bold"),
                            html.Span(" (rpm)", className="text-muted ms-1")
                        ], className="mb-3"),

                        # Missing sensors - red X's
                        html.Hr(),
                        html.H6("Missing Sensors (3/12)", className="mb-3 text-danger"),
                        html.Div([
                            html.I(className="fas fa-times-circle me-2", style={'color': '#e74c3c'}),
                            html.Span("GPS Latitude", className="text-muted"),
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-times-circle me-2", style={'color': '#e74c3c'}),
                            html.Span("GPS Longitude", className="text-muted"),
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-times-circle me-2", style={'color': '#e74c3c'}),
                            html.Span("GPS Altitude", className="text-muted"),
                        ], className="mb-0")
                    ])
                ], md=6),

                # Right: Data quality metrics
                dbc.Col([
                    html.H6("Data Quality Metrics", className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-database fa-2x mb-2", style={'color': '#3498db'}),
                                html.H4("582,035", className="mb-0", style={'color': '#2c3e50'}),
                                html.P("Telemetry Points", className="text-muted mb-0", style={'fontSize': '0.9rem'})
                            ], className="text-center mb-3")
                        ])
                    ], className="mb-3", style={'backgroundColor': '#f8f9fa'}),

                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-check-double fa-2x mb-2", style={'color': '#2ecc71'}),
                                html.H4("100%", className="mb-0", style={'color': '#2c3e50'}),
                                html.P("Data Completeness", className="text-muted mb-0", style={'fontSize': '0.9rem'})
                            ], className="text-center mb-0")
                        ])
                    ], style={'backgroundColor': '#f8f9fa'}),

                    # GPS Warning Alert
                    dbc.Alert([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        html.Strong("GPS Not Available: "),
                        "Advanced spatial analysis features require GPS data. ",
                        "Current analysis uses core sensors only."
                    ], color="warning", className="mt-3 mb-0")
                ], md=6)
            ])
        ])
    ], className="mb-4", style={'border': '1px solid #dee2e6', 'borderRadius': '10px'})


def create_ai_model_config_card():
    """
    Create standalone AI model configuration card for main dashboard

    Returns:
        dbc.Card: AI model configuration card
    """
    return dbc.Card([
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
    ], className="mb-4", style={'border': '1px solid #dee2e6', 'borderRadius': '10px'})
