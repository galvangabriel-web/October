"""
Data Structure Widget
=====================

Widget displaying sensor status and data quality metrics.

Features:
- Sensor availability checklist (9 available, 3 missing)
- Data quality metrics
- GPS availability warning
- Data completeness indicators

Usage:
    # In app.py:
    from src.dashboard.data_structure_widget import create_data_structure_layout

    # Add to tabs
    dcc.Tab(label='Data Structure', value='tab-data-structure', children=create_data_structure_layout())

    # No callbacks needed - this is a static widget
"""

from dash import html
import dash_bootstrap_components as dbc


def create_data_structure_layout():
    """
    Create Data Structure layout with sensor status and quality metrics

    Returns:
        Dash layout component
    """
    return html.Div([
        # ============================================================================
        # DATA STRUCTURE SECTION
        # ============================================================================
        dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="fas fa-database me-3", style={'color': '#3498db'}),
                    "Data Structure"
                ], className="mb-0", style={
                    'fontWeight': '700',
                    'fontSize': '24px',
                    'color': '#2c3e50'
                })
            ], style={
                'backgroundColor': '#ffffff',
                'borderBottom': '3px solid #3498db',
                'padding': '1.2rem'
            }),
            dbc.CardBody([

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
            ], style={'padding': '1.5rem'})
        ], className="mb-4", style={'border': '1px solid #dee2e6', 'borderRadius': '15px', 'boxShadow': '0 4px 15px rgba(52, 152, 219, 0.1)'}),
    ])
