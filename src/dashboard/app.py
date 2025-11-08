"""
Racing Analytics Dashboard - Main Application
==============================================

Interactive dashboard for racing telemetry analysis with:
- Telemetry file upload
- Driver performance comparison
- Telemetry overlay charts
- Coaching insights
- Lap time predictions
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import base64
import io
import requests
import logging
from PIL import Image
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Week 1 widgets
try:
    # Add parent directory to path for module imports
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    from src.dashboard.weather_widget import create_weather_layout, create_weather_callbacks
    from src.dashboard.sector_widget import create_sector_layout, create_sector_callbacks
    from src.dashboard.animation_widget import create_animation_layout, create_animation_callbacks
    from src.dashboard.post_race_widget import (
        create_post_race_layout,
        create_post_race_callbacks,
        create_sensor_status_card,
        create_ai_model_config_card
    )
    from src.dashboard.enhanced_driver_insights_widget import create_enhanced_driver_insights_layout
    from src.data_processing.weather_loader import WeatherDataLoader
    from src.data_processing.lap_analysis_loader import LapAnalysisLoader
    from src.data_processing.championship_loader import ChampionshipLoader
    WEEK1_ENABLED = True
    ENHANCED_INSIGHTS_ENABLED = True
    logger.info("Week 1 widgets and Enhanced Insights loaded successfully")
except ImportError as e:
    WEEK1_ENABLED = False
    ENHANCED_INSIGHTS_ENABLED = False
    logger.warning(f"Week 1 widgets not available: {e}")

# Import tour components (separate from Week 1 to avoid dependency issues)
try:
    from src.dashboard.tour import create_welcome_modal
    TOUR_ENABLED = True
    logger.info("Tour system loaded successfully")
except ImportError as e:
    TOUR_ENABLED = False
    logger.warning(f"Tour system not available: {e}")
    # Create dummy function if tour is not available
    def create_welcome_modal():
        return html.Div()

# Import help documentation system
try:
    from src.dashboard.help_documentation import (
        create_help_button,
        create_help_documentation_modal,
        create_help_callbacks
    )
    HELP_ENABLED = True
    logger.info("Help documentation system loaded successfully")
except ImportError as e:
    HELP_ENABLED = False
    logger.warning(f"Help documentation system not available: {e}")
    # Create dummy functions if help is not available
    def create_help_button():
        return html.Div()
    def create_help_documentation_modal():
        return html.Div()
    def create_help_callbacks(app):
        return app

# Import chatbot widget
print("=" * 80)
print("DEBUG: ATTEMPTING CHATBOT IMPORT")
print("=" * 80)
logger.info("=" * 80)
logger.info("DEBUG: ATTEMPTING CHATBOT IMPORT")
logger.info("=" * 80)
try:
    from src.dashboard.chatbot_modern_widget import create_modern_chatbot_layout as create_chatbot_layout, create_modern_chatbot_callbacks as create_chatbot_callbacks
    CHATBOT_ENABLED = True
    print("DEBUG: [OK] CHATBOT IMPORT SUCCEEDED - CHATBOT_ENABLED = True")
    logger.info("[OK] Chatbot widget loaded successfully - CHATBOT_ENABLED = True")
except ImportError as e:
    CHATBOT_ENABLED = False
    print(f"DEBUG: [X] CHATBOT IMPORT FAILED - {e}")
    logger.warning(f"Chatbot widget not available: {e}")
    # Create dummy functions if chatbot is not available
    def create_chatbot_layout():
        return html.Div()
    def create_chatbot_callbacks(app):
        return app
except Exception as e:
    CHATBOT_ENABLED = False
    print(f"DEBUG: [X] CHATBOT IMPORT EXCEPTION - {e}")
    logger.error(f"Chatbot import exception: {e}")
    # Create dummy functions if chatbot is not available
    def create_chatbot_layout():
        return html.Div()
    def create_chatbot_callbacks(app):
        return app

print(f"DEBUG: FINAL CHATBOT_ENABLED STATE = {CHATBOT_ENABLED}")
logger.info(f"DEBUG: FINAL CHATBOT_ENABLED STATE = {CHATBOT_ENABLED}")
print("=" * 80)

# Note: All data processing is done via API calls
# No direct imports needed - dashboard communicates with API

#================================================================
# VERSION TRACKING
#================================================================

DASHBOARD_VERSION = "3.2.0-help-documentation"
VERSION_DATE = "2025-11-07"
VERSION_NOTES = """
Version 3.2.0 - Help Documentation System
- Comprehensive help guide for all 10 dashboard tabs
- Detailed explanations of purpose and data interpretation
- Special documentation for Post-Race Analysis variables
- Advanced Model Predictions tab documentation
- Vehicle selection requirement notices
- Floating help button with modal documentation
- Tab-specific technical insights and metrics
- Previous version: 3.1.0-tour-system-mvp
"""

#================================================================
# CONFIGURATION
#================================================================

API_BASE = "http://localhost:8000"
COLORS = px.colors.qualitative.Set2

# Production mode: Auto-load data from server path
PRODUCTION_MODE = True
DATA_FILE_PATH = "/home/tactical/racing_analytics/data/master_racing_data.csv"

# Global cache for corner analyses (Sprint 2 Task 5)
_cached_corner_analyses = None

# Global storage for auto-loaded data
_auto_loaded_data = None
_auto_loaded_stats = {
    'num_samples': 0,
    'vehicle_options': [],
    'num_vehicles': 0,
    'num_laps': 0,
    'avg_time': '--'
}

# Initialize Dash app with Bootstrap theme and custom stylesheets
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
    ],
    title="Racing Analytics Dashboard",
    suppress_callback_exceptions=True
)

# Initialize Week 1 data loaders
if WEEK1_ENABLED:
    try:
        weather_loader = WeatherDataLoader()
        lap_loader = LapAnalysisLoader()
        championship_loader = ChampionshipLoader()
        logger.info("Week 1 data loaders initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Week 1 loaders: {e}")
        WEEK1_ENABLED = False

#================================================================
# AUTO-LOAD DATA FUNCTION (Production Mode)
#================================================================

def load_data_on_startup():
    """
    Load data from server path on dashboard startup (Production Mode).

    This function:
    1. Checks if data file exists at DATA_FILE_PATH
    2. Loads CSV into memory
    3. Calculates statistics
    4. Populates global variables for dashboard

    Returns:
        bool: True if data loaded successfully, False otherwise
    """
    global _auto_loaded_data, _auto_loaded_stats

    if not PRODUCTION_MODE:
        logger.info("Production mode disabled, skipping auto-load")
        return False

    try:
        # Check if running on Windows (development) or Linux (production)
        import platform
        is_windows = platform.system() == 'Windows'

        # Use local path for Windows development, server path for Linux production
        if is_windows:
            data_path = Path(__file__).parent.parent.parent / "master_racing_data.csv"
            logger.info(f"[DEVELOPMENT MODE] Loading from local path: {data_path}")
        else:
            data_path = Path(DATA_FILE_PATH)
            logger.info(f"[PRODUCTION MODE] Loading from server path: {data_path}")

        # Check if file exists
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return False

        # Load CSV
        logger.info(f"Loading telemetry data from {data_path}...")
        df = pd.read_csv(data_path)
        logger.info(f"[OK] Loaded {len(df):,} rows from {data_path}")

        # Store data as JSON
        _auto_loaded_data = df.to_json(date_format='iso', orient='split')

        # Calculate statistics
        num_samples = len(df)
        vehicles = sorted(df['vehicle_number'].unique()) if 'vehicle_number' in df.columns else []
        num_vehicles = len(vehicles)
        num_laps = df['lap'].nunique() if 'lap' in df.columns else 0

        # Create vehicle dropdown options
        vehicle_options = [{'label': f'Vehicle #{v}', 'value': v} for v in vehicles]

        # Store stats
        _auto_loaded_stats = {
            'num_samples': num_samples,
            'vehicle_options': vehicle_options,
            'num_vehicles': num_vehicles,
            'num_laps': num_laps,
            'avg_time': '--'
        }

        logger.info(f"[OK] Auto-load complete: {num_samples:,} samples, {num_vehicles} vehicles, {num_laps} laps")
        return True

    except Exception as e:
        logger.error(f"Failed to auto-load data: {e}", exc_info=True)
        return False

# Auto-load data on startup if in production mode
if PRODUCTION_MODE:
    logger.info("="*60)
    logger.info("PRODUCTION MODE: Auto-loading telemetry data...")
    logger.info("="*60)
    if load_data_on_startup():
        logger.info("[OK] Data auto-loaded successfully")
    else:
        logger.warning("[WARN] Failed to auto-load data, dashboard will start empty")

#================================================================
# LAYOUT
#================================================================

# Navigation bar
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.I(className="fas fa-flag-checkered", style={'font-size': '24px', 'color': 'white'}), width="auto"),
            dbc.Col(dbc.NavbarBrand("Racing Analytics Dashboard", className="ms-2", style={'font-size': '24px'}), width="auto"),
        ], align="center", className="g-0"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Dashboard", href="#")),
            dbc.NavItem(dbc.NavLink("API Docs", href=f"{API_BASE}/docs", target="_blank")),
        ], className="ms-auto", navbar=True),
    ], fluid=True),
    color="dark",
    dark=True,
    className="mb-4"
)

# Stats cards
def create_stat_card(title, value, icon, color):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"fas fa-{icon}", style={'font-size': '24px', 'color': color}),
                html.H4(value, className="mt-2 mb-0"),
                html.P(title, className="text-muted mb-0"),
            ], className="text-center")
        ])
    ], className="shadow-sm")

# File upload section
upload_section = dbc.Card([
    dbc.CardHeader(html.H5([html.I(className="fas fa-upload me-2"), "Upload Telemetry Data"])),
    dbc.CardBody([
        dcc.Upload(
            id='upload-telemetry',
            children=html.Div([
                html.I(className="fas fa-cloud-upload-alt", style={'font-size': '48px', 'color': '#6c757d'}),
                html.Br(),
                html.Br(),
                'Drag and Drop or ',
                html.A('Select CSV File', style={'color': '#007bff', 'cursor': 'pointer'})
            ]),
            style={
                'width': '100%',
                'height': '150px',
                'lineHeight': '150px',
                'borderWidth': '2px',
                'borderStyle': 'dashed',
                'borderRadius': '10px',
                'textAlign': 'center',
                'background': '#f8f9fa'
            },
            multiple=False
        ),
        html.Div(id='upload-status', className="mt-3"),
    ])
], className="mb-4 shadow-sm")

# Vehicle selector
vehicle_selector = dbc.Card([
    dbc.CardHeader(html.H5([html.I(className="fas fa-car me-2"), "Select Vehicle"])),
    dbc.CardBody([
        dcc.Dropdown(
            id='vehicle-dropdown',
            placeholder="Select a vehicle number...",
            className="mb-2"
        ),
        dbc.Button(
            [html.I(className="fas fa-chart-line me-2"), "Analyze Performance"],
            id='analyze-button',
            color="primary",
            className="w-100",
            disabled=True
        ),
    ])
], className="mb-4 shadow-sm")

# Create upload page layout (Page 1)
def create_upload_page():
    """Create Windows 11 Fluent Design upload page

    Custom CSS animations are loaded from assets/windows11.css
    """
    return html.Div([
        # Main container with centered content
        html.Div([
            # Header section with modern branding
            html.Div([
                # Icon with glow effect
                html.Div([
                    html.I(className="fas fa-flag-checkered", style={
                        'fontSize': '64px',
                        'color': '#0078D4',
                        'filter': 'drop-shadow(0 8px 16px rgba(0, 120, 212, 0.3))',
                        'marginBottom': '24px',
                        'animation': 'fadeInUp 0.6s ease-out'
                    })
                ], style={'textAlign': 'center'}),

                # Title with modern typography
                html.H1("Racing Analytics Dashboard", style={
                    'textAlign': 'center',
                    'color': '#FFFFFF',
                    'fontSize': '42px',
                    'fontWeight': '600',
                    'letterSpacing': '-0.5px',
                    'marginBottom': '12px',
                    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI Variable", "Segoe UI", system-ui, sans-serif',
                    'animation': 'fadeInUp 0.6s ease-out 0.1s both'
                }),
                html.P("Unlock professional racing insights with AI-powered telemetry analysis", style={
                    'textAlign': 'center',
                    'color': 'rgba(255, 255, 255, 0.85)',
                    'fontSize': '16px',
                    'fontWeight': '400',
                    'marginBottom': '48px',
                    'lineHeight': '1.5',
                    'animation': 'fadeInUp 0.6s ease-out 0.2s both'
                }),
            ]),

            # Main upload card with acrylic effect
            html.Div([
                dbc.CardBody([
                    # Upload zone with hover effect
                    dcc.Upload(
                        id='upload-telemetry',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt", style={
                                'fontSize': '48px',
                                'color': '#0078D4',
                                'marginBottom': '20px',
                                'transition': 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
                            }),
                            html.H4("Drop your telemetry file here", style={
                                'color': '#201F1E',
                                'fontSize': '20px',
                                'fontWeight': '600',
                                'marginBottom': '8px',
                                'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI Variable", "Segoe UI", system-ui, sans-serif'
                            }),
                            html.P("or click to browse", style={
                                'color': '#605E5C',
                                'fontSize': '14px',
                                'marginBottom': '24px'
                            }),
                            html.Div([
                                html.I(className="fas fa-file-csv me-2"),
                                "Select CSV File"
                            ], style={
                                'display': 'inline-block',
                                'padding': '12px 32px',
                                'background': '#0078D4',
                                'color': '#FFFFFF',
                                'borderRadius': '6px',
                                'fontSize': '14px',
                                'fontWeight': '600',
                                'cursor': 'pointer',
                                'transition': 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                                'boxShadow': '0 2px 6px rgba(0, 120, 212, 0.3)'
                            })
                        ], style={
                            'textAlign': 'center',
                            'padding': '64px 48px',
                            'transition': 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
                        }),
                        style={
                            'width': '100%',
                            'borderRadius': '12px',
                            'border': '2px dashed rgba(0, 120, 212, 0.3)',
                            'background': 'rgba(243, 242, 241, 0.6)',
                            'backdropFilter': 'blur(20px)',
                            'cursor': 'pointer',
                            'transition': 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                            'position': 'relative',
                            'overflow': 'hidden'
                        },
                        multiple=False
                    ),

                    # Status message
                    html.Div(id='upload-status', className="mt-4"),

                    # Hidden elements for callback compatibility
                    html.Div([
                        dcc.Dropdown(id='vehicle-dropdown', options=[], style={'display': 'none'}),
                        html.Span(id='stat-samples', children="0", style={'display': 'none'}),
                        html.Span(id='stat-vehicles', children="0", style={'display': 'none'}),
                        html.Span(id='stat-laps', children="0", style={'display': 'none'}),
                        html.Span(id='stat-avg-time', children="--", style={'display': 'none'}),
                        dbc.Button(id='analyze-button', children="Analyze", disabled=True, style={'display': 'none'}),
                    ], style={'display': 'none'}),

                    # Feature cards with Fluent Design
                    html.Div([
                        html.H6("Powerful features at your fingertips", style={
                            'color': '#201F1E',
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'marginBottom': '24px',
                            'textAlign': 'center',
                            'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI Variable", "Segoe UI", system-ui, sans-serif'
                        }),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Div(style={
                                        'width': '48px',
                                        'height': '48px',
                                        'borderRadius': '12px',
                                        'background': 'linear-gradient(135deg, #0078D4 0%, #00BCF2 100%)',
                                        'display': 'flex',
                                        'alignItems': 'center',
                                        'justifyContent': 'center',
                                        'margin': '0 auto 12px auto',
                                        'boxShadow': '0 4px 12px rgba(0, 120, 212, 0.25)'
                                    }, children=[
                                        html.I(className="fas fa-chart-line", style={'fontSize': '20px', 'color': '#FFFFFF'})
                                    ]),
                                    html.P("Driver Insights", style={'fontSize': '14px', 'fontWeight': '600', 'color': '#201F1E', 'marginBottom': '4px'}),
                                    html.Small("Performance metrics", style={'color': '#605E5C', 'fontSize': '12px'})
                                ], style={'textAlign': 'center', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
                            ], md=3),
                            dbc.Col([
                                html.Div([
                                    html.Div(style={
                                        'width': '48px',
                                        'height': '48px',
                                        'borderRadius': '12px',
                                        'background': 'linear-gradient(135deg, #8764B8 0%, #A675C6 100%)',
                                        'display': 'flex',
                                        'alignItems': 'center',
                                        'justifyContent': 'center',
                                        'margin': '0 auto 12px auto',
                                        'boxShadow': '0 4px 12px rgba(135, 100, 184, 0.25)'
                                    }, children=[
                                        html.I(className="fas fa-brain", style={'fontSize': '20px', 'color': '#FFFFFF'})
                                    ]),
                                    html.P("AI Predictions", style={'fontSize': '14px', 'fontWeight': '600', 'color': '#201F1E', 'marginBottom': '4px'}),
                                    html.Small("Lap time forecasting", style={'color': '#605E5C', 'fontSize': '12px'})
                                ], style={'textAlign': 'center', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
                            ], md=3),
                            dbc.Col([
                                html.Div([
                                    html.Div(style={
                                        'width': '48px',
                                        'height': '48px',
                                        'borderRadius': '12px',
                                        'background': 'linear-gradient(135deg, #107C10 0%, #0B6A0B 100%)',
                                        'display': 'flex',
                                        'alignItems': 'center',
                                        'justifyContent': 'center',
                                        'margin': '0 auto 12px auto',
                                        'boxShadow': '0 4px 12px rgba(16, 124, 16, 0.25)'
                                    }, children=[
                                        html.I(className="fas fa-map-marked-alt", style={'fontSize': '20px', 'color': '#FFFFFF'})
                                    ]),
                                    html.P("Track Maps", style={'fontSize': '14px', 'fontWeight': '600', 'color': '#201F1E', 'marginBottom': '4px'}),
                                    html.Small("Visual overlays", style={'color': '#605E5C', 'fontSize': '12px'})
                                ], style={'textAlign': 'center', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
                            ], md=3),
                            dbc.Col([
                                html.Div([
                                    html.Div(style={
                                        'width': '48px',
                                        'height': '48px',
                                        'borderRadius': '12px',
                                        'background': 'linear-gradient(135deg, #FFB900 0%, #FF8C00 100%)',
                                        'display': 'flex',
                                        'alignItems': 'center',
                                        'justifyContent': 'center',
                                        'margin': '0 auto 12px auto',
                                        'boxShadow': '0 4px 12px rgba(255, 185, 0, 0.25)'
                                    }, children=[
                                        html.I(className="fas fa-trophy", style={'fontSize': '20px', 'color': '#FFFFFF'})
                                    ]),
                                    html.P("Championships", style={'fontSize': '14px', 'fontWeight': '600', 'color': '#201F1E', 'marginBottom': '4px'}),
                                    html.Small("Live standings", style={'color': '#605E5C', 'fontSize': '12px'})
                                ], style={'textAlign': 'center', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
                            ], md=3),
                        ], className="g-4")
                    ], style={'marginTop': '48px'})
                ], style={'padding': '48px'})
            ], style={
                'maxWidth': '1000px',
                'margin': '0 auto',
                'background': 'rgba(255, 255, 255, 0.7)',
                'backdropFilter': 'blur(40px) saturate(180%)',
                'borderRadius': '16px',
                'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.12), 0 2px 8px rgba(0, 0, 0, 0.08)',
                'border': '1px solid rgba(255, 255, 255, 0.18)',
                'animation': 'fadeInUp 0.6s ease-out 0.3s both'
            }),

        ], style={
            'maxWidth': '1200px',
            'margin': '0 auto',
            'padding': '80px 20px'
        })
    ], style={
        'minHeight': '100vh',
        'background': 'linear-gradient(135deg, #0078D4 0%, #8764B8 50%, #0078D4 100%)',
        'backgroundSize': '200% 200%',
        'animation': 'gradientShift 15s ease infinite',
        'position': 'relative',
        'overflow': 'hidden'
    })

# Create dashboard page layout (Page 2)
def create_dashboard_page():
    """Create full dashboard with all tabs"""
    return html.Div([
        # Header
        navbar,

        # Main content container - FULL WIDTH
        dbc.Container([
            # ============================================================================
            # HERO SECTION - Marketing-focused introduction
            # ============================================================================
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        # Main headline
                        html.H1([
                            "AI-Powered Racing Intelligence"
                        ], style={
                            'fontSize': '48px',
                            'fontWeight': '700',
                            'color': 'white',
                            'textAlign': 'center',
                            'marginBottom': '1rem',
                            'textShadow': '0 2px 4px rgba(0,0,0,0.3)',
                            'fontFamily': 'Inter, sans-serif'
                        }),

                        # Sub-headline
                        html.H4([
                            "97.49% Accuracy. 18.5GB Training Data. Real Results."
                        ], style={
                            'color': 'rgba(255,255,255,0.95)',
                            'textAlign': 'center',
                            'marginBottom': '2.5rem',
                            'fontWeight': '500',
                            'fontSize': '24px',
                            'fontFamily': 'Inter, sans-serif'
                        }),

                        # Floating stats cards
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.H2("71,000+", style={'color': 'white', 'fontWeight': '700', 'marginBottom': '0.5rem'}),
                                    html.P("Telemetry Samples", style={'color': 'rgba(255,255,255,0.9)', 'marginBottom': '0', 'fontSize': '14px'})
                                ], style={
                                    'backgroundColor': 'rgba(255,255,255,0.15)',
                                    'padding': '1.5rem',
                                    'borderRadius': '12px',
                                    'textAlign': 'center',
                                    'backdropFilter': 'blur(10px)',
                                    'border': '1px solid rgba(255,255,255,0.2)'
                                })
                            ], md=4, className="mb-3 mb-md-0"),
                            dbc.Col([
                                html.Div([
                                    html.H2("97.49%", style={'color': 'white', 'fontWeight': '700', 'marginBottom': '0.5rem'}),
                                    html.P("Prediction Accuracy", style={'color': 'rgba(255,255,255,0.9)', 'marginBottom': '0', 'fontSize': '14px'})
                                ], style={
                                    'backgroundColor': 'rgba(255,255,255,0.15)',
                                    'padding': '1.5rem',
                                    'borderRadius': '12px',
                                    'textAlign': 'center',
                                    'backdropFilter': 'blur(10px)',
                                    'border': '1px solid rgba(255,255,255,0.2)'
                                })
                            ], md=4, className="mb-3 mb-md-0"),
                            dbc.Col([
                                html.Div([
                                    html.H2("±1.73s", style={'color': 'white', 'fontWeight': '700', 'marginBottom': '0.5rem'}),
                                    html.P("Lap Time Precision", style={'color': 'rgba(255,255,255,0.9)', 'marginBottom': '0', 'fontSize': '14px'})
                                ], style={
                                    'backgroundColor': 'rgba(255,255,255,0.15)',
                                    'padding': '1.5rem',
                                    'borderRadius': '12px',
                                    'textAlign': 'center',
                                    'backdropFilter': 'blur(10px)',
                                    'border': '1px solid rgba(255,255,255,0.2)'
                                })
                            ], md=4),
                        ])
                    ], style={'padding': '3rem 2rem'})
                ], style={
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    'border': 'none',
                    'borderRadius': '16px',
                    'boxShadow': '0 8px 32px rgba(102, 126, 234, 0.4)',
                    'marginBottom': '2rem'
                })
            ]),


            # ============================================================================
            # STAR FEATURE SPOTLIGHT - Post-Race Analysis Highlight
            # ============================================================================
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        # Feature badge
                        html.Div([
                            dbc.Badge([
                                html.I(className='fas fa-trophy me-2'),
                                'FLAGSHIP FEATURE'
                            ], color='warning', className='mb-3', style={'fontSize': '0.9rem', 'fontWeight': '600'})
                        ], style={'textAlign': 'center'}),

                        # Feature title
                        html.H2([
                            html.I(className='fas fa-brain me-3', style={'color': '#667eea'}),
                            'Post-Race Analysis - Your AI Race Engineer'
                        ], style={
                            'fontSize': '32px',
                            'fontWeight': '700',
                            'color': '#2c3e50',
                            'textAlign': 'center',
                            'marginBottom': '1.5rem',
                            'fontFamily': 'Inter, sans-serif'
                        }),

                        # Feature description
                        html.P([
                            'Upload your telemetry and receive instant, personalized coaching insights powered by our championship-winning AI. ',
                            'Get lap-by-lap analysis, corner-specific recommendations, and pattern detection that professional race engineers use.'
                        ], style={
                            'fontSize': '18px',
                            'color': '#34495e',
                            'textAlign': 'center',
                            'marginBottom': '2rem',
                            'lineHeight': '1.7'
                        }),

                        # Key selling points
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.I(className='fas fa-check-circle fa-2x mb-2', style={'color': '#2ecc71'}),
                                    html.H6('Multi-Dimensional Patterns', className='mb-1', style={'fontWeight': '600'}),
                                    html.P('WHAT × WHERE × WHEN analysis', className='text-muted mb-0', style={'fontSize': '14px'})
                                ], style={'textAlign': 'center'})
                            ], md=4),
                            dbc.Col([
                                html.Div([
                                    html.I(className='fas fa-chart-line fa-2x mb-2', style={'color': '#3498db'}),
                                    html.H6('Coaching Insights', className='mb-1', style={'fontWeight': '600'}),
                                    html.P('Specific, actionable recommendations', className='text-muted mb-0', style={'fontSize': '14px'})
                                ], style={'textAlign': 'center'})
                            ], md=4),
                            dbc.Col([
                                html.Div([
                                    html.I(className='fas fa-stopwatch fa-2x mb-2', style={'color': '#e74c3c'}),
                                    html.H6('Time Savings', className='mb-1', style={'fontWeight': '600'}),
                                    html.P('Identify seconds per lap improvements', className='text-muted mb-0', style={'fontSize': '14px'})
                                ], style={'textAlign': 'center'})
                            ], md=4),
                        ], className='mb-3'),

                        # CTA Button (will be linked to Post-Race tab via callback)
                        html.Div([
                            dbc.Button([
                                html.I(className='fas fa-rocket me-2'),
                                'Analyze Your Session Now',
                                html.I(className='fas fa-arrow-right ms-2')
                            ],
                            id='hero-analyze-btn',
                            color='primary',
                            size='lg',
                            style={
                                'fontSize': '18px',
                                'padding': '12px 48px',
                                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                'border': 'none',
                                'borderRadius': '50px',
                                'fontWeight': '600',
                                'boxShadow': '0 4px 15px rgba(102, 126, 234, 0.4)'
                            })
                        ], style={'textAlign': 'center'})
                    ], style={'padding': '2.5rem'})
                ], style={
                    'border': '2px solid #667eea',
                    'borderRadius': '16px',
                    'boxShadow': '0 8px 32px rgba(102, 126, 234, 0.2)',
                    'marginBottom': '2rem',
                    'background': 'linear-gradient(to bottom, #ffffff 0%, #f8f9ff 100%)'
                })
            ]),

            # ============================================================================
            # FEATURE SHOWCASE GRID - 6 Premium Capability Cards
            # ============================================================================
            html.Div([
                html.H3([
                    "Platform Capabilities"
                ], style={
                    'fontSize': '32px',
                    'fontWeight': '700',
                    'color': '#2c3e50',
                    'textAlign': 'center',
                    'marginBottom': '2rem',
                    'fontFamily': 'Inter, sans-serif'
                }),

                dbc.Row([
                    # Card 1: Real-Time Telemetry
                    dbc.Col([
                        dbc.Card([
                            html.Div(style={
                                'height': '4px',
                                'background': 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
                                'borderRadius': '4px 4px 0 0'
                            }),
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-broadcast-tower fa-3x mb-3", style={'color': '#667eea'}),
                                    html.H5("Real-Time Telemetry", className="mb-2", style={'fontWeight': '600'}),
                                    html.P("12-channel sensor data @ 10Hz. Speed, braking, throttle, steering, GPS, and G-forces.",
                                           className="text-muted mb-0", style={'fontSize': '14px'})
                                ], style={'textAlign': 'center'})
                            ])
                        ], style={'border': '1px solid #e1e8ed', 'borderRadius': '12px'}, className="h-100")
                    ], md=4, className="mb-4"),

                    # Card 2: AI Predictions
                    dbc.Col([
                        dbc.Card([
                            html.Div(style={
                                'height': '4px',
                                'background': 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
                                'borderRadius': '4px 4px 0 0'
                            }),
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-brain fa-3x mb-3", style={'color': '#3498db'}),
                                    html.H5("AI Predictions", className="mb-2", style={'fontWeight': '600'}),
                                    html.P("Sequential LightGBM model with 97.49% R² accuracy trained on 18.5GB of championship data.",
                                           className="text-muted mb-0", style={'fontSize': '14px'})
                                ], style={'textAlign': 'center'})
                            ])
                        ], style={'border': '1px solid #e1e8ed', 'borderRadius': '12px'}, className="h-100")
                    ], md=4, className="mb-4"),

                    # Card 3: Track Coaching
                    dbc.Col([
                        dbc.Card([
                            html.Div(style={
                                'height': '4px',
                                'background': 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
                                'borderRadius': '4px 4px 0 0'
                            }),
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-chalkboard-teacher fa-3x mb-3", style={'color': '#2ecc71'}),
                                    html.H5("Track Coaching", className="mb-2", style={'fontWeight': '600'}),
                                    html.P("Corner-specific recommendations across 6 circuits. Learn where to brake, accelerate, and improve.",
                                           className="text-muted mb-0", style={'fontSize': '14px'})
                                ], style={'textAlign': 'center'})
                            ])
                        ], style={'border': '1px solid #e1e8ed', 'borderRadius': '12px'}, className="h-100")
                    ], md=4, className="mb-4"),

                    # Card 4: Pattern Detection
                    dbc.Col([
                        dbc.Card([
                            html.Div(style={
                                'height': '4px',
                                'background': 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
                                'borderRadius': '4px 4px 0 0'
                            }),
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-project-diagram fa-3x mb-3", style={'color': '#e74c3c'}),
                                    html.H5("Pattern Detection", className="mb-2", style={'fontWeight': '600'}),
                                    html.P("Multi-dimensional cube analysis identifies hidden driving patterns and improvement opportunities.",
                                           className="text-muted mb-0", style={'fontSize': '14px'})
                                ], style={'textAlign': 'center'})
                            ])
                        ], style={'border': '1px solid #e1e8ed', 'borderRadius': '12px'}, className="h-100")
                    ], md=4, className="mb-4"),

                    # Card 5: Visualizations
                    dbc.Col([
                        dbc.Card([
                            html.Div(style={
                                'height': '4px',
                                'background': 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
                                'borderRadius': '4px 4px 0 0'
                            }),
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-chart-area fa-3x mb-3", style={'color': '#f39c12'}),
                                    html.H5("Interactive Visualizations", className="mb-2", style={'fontWeight': '600'}),
                                    html.P("GPS track maps, animated laps, synchronized telemetry comparison, and real-time charts.",
                                           className="text-muted mb-0", style={'fontSize': '14px'})
                                ], style={'textAlign': 'center'})
                            ])
                        ], style={'border': '1px solid #e1e8ed', 'borderRadius': '12px'}, className="h-100")
                    ], md=4, className="mb-4"),

                    # Card 6: Benchmarking
                    dbc.Col([
                        dbc.Card([
                            html.Div(style={
                                'height': '4px',
                                'background': 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
                                'borderRadius': '4px 4px 0 0'
                            }),
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-trophy fa-3x mb-3", style={'color': '#9b59b6'}),
                                    html.H5("Performance Benchmarking", className="mb-2", style={'fontWeight': '600'}),
                                    html.P("Compare against top drivers, track records, and your personal bests across all sessions.",
                                           className="text-muted mb-0", style={'fontSize': '14px'})
                                ], style={'textAlign': 'center'})
                            ])
                        ], style={'border': '1px solid #e1e8ed', 'borderRadius': '12px'}, className="h-100")
                    ], md=4, className="mb-4"),
                ])
            ], className="mb-4"),

            # ============================================================================
            # TECH POWER BAR - Impressive Platform Specifications
            # ============================================================================
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-database me-2", style={'color': '#667eea'}),
                                    html.Strong("18.5GB", style={'fontSize': '18px', 'color': '#2c3e50'}),
                                    html.Span(" Training Data", className="text-muted ms-1", style={'fontSize': '14px'})
                                ], style={'textAlign': 'center'})
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-car me-2", style={'color': '#3498db'}),
                                    html.Strong("20", style={'fontSize': '18px', 'color': '#2c3e50'}),
                                    html.Span(" Vehicles", className="text-muted ms-1", style={'fontSize': '14px'})
                                ], style={'textAlign': 'center'})
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-flag-checkered me-2", style={'color': '#2ecc71'}),
                                    html.Strong("4,881", style={'fontSize': '18px', 'color': '#2c3e50'}),
                                    html.Span(" Laps", className="text-muted ms-1", style={'fontSize': '14px'})
                                ], style={'textAlign': 'center'})
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-route me-2", style={'color': '#e74c3c'}),
                                    html.Strong("6", style={'fontSize': '18px', 'color': '#2c3e50'}),
                                    html.Span(" Tracks", className="text-muted ms-1", style={'fontSize': '14px'})
                                ], style={'textAlign': 'center'})
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-tachometer-alt me-2", style={'color': '#f39c12'}),
                                    html.Strong("10Hz", style={'fontSize': '18px', 'color': '#2c3e50'}),
                                    html.Span(" Sampling", className="text-muted ms-1", style={'fontSize': '14px'})
                                ], style={'textAlign': 'center'})
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-bolt me-2", style={'color': '#9b59b6'}),
                                    html.Strong("<200ms", style={'fontSize': '18px', 'color': '#2c3e50'}),
                                    html.Span(" API Response", className="text-muted ms-1", style={'fontSize': '14px'})
                                ], style={'textAlign': 'center'})
                            ], md=2),
                        ])
                    ], style={'padding': '1.5rem'})
                ], style={
                    'border': '1px solid #e1e8ed',
                    'borderRadius': '12px',
                    'background': 'linear-gradient(to right, #f8f9fa 0%, #ffffff 100%)',
                    'marginBottom': '3rem'
                })
            ]),

            # Analysis Tabs Section - Enhanced Visual Container
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-chart-line me-3", style={'color': '#667eea'}),
                            "Choose Your Analysis Tool"
                        ], className="mb-0", style={'fontWeight': '600', 'fontSize': '24px', 'color': '#2c3e50'})
                    ])
                ], style={
                    'backgroundColor': '#ffffff',
                    'borderBottom': '3px solid #667eea',
                    'padding': '1.5rem'
                }),
                dbc.CardBody([
                    # Vehicle selector - moved from footer to tab container
                    dbc.Row([
                        dbc.Col([
                            html.Label([
                                html.I(className="fas fa-car me-2"),
                                "Select Vehicle"
                            ], className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id='vehicle-dropdown-display',
                                placeholder="Select vehicle number...",
                                className="mb-3"
                            )
                        ], md=6),
                        dbc.Col([
                            html.Div([
                                html.Span("Data loaded: ", className="fw-bold"),
                                html.Span(id='stat-samples', children="0"),
                                html.Span(" samples, ", className="text-muted"),
                                html.Span(id='stat-vehicles', children="0"),
                                html.Span(" vehicles, ", className="text-muted"),
                                html.Span(id='stat-laps', children="0"),
                                html.Span(" laps", className="text-muted")
                            ], style={'padding': '0.5rem', 'marginTop': '1.8rem'})
                        ], md=6)
                    ], className="mb-3"),

                    # FULL WIDTH TABS - Maximum space for visualizations
                    dbc.Row([
                        dbc.Col([
                            dbc.Tabs([
                                dbc.Tab(label="Driver Insights", tab_id="tab-insights", label_style={"cursor": "pointer"}),
                                dbc.Tab(label="Post-Race Analysis", tab_id="tab-post-race", label_style={"cursor": "pointer"}),
                                dbc.Tab(label="Telemetry Charts", tab_id="tab-telemetry", label_style={"cursor": "pointer"}),
                                dbc.Tab(label="Model Predictions", tab_id="tab-predictions", label_style={"cursor": "pointer"}),
                                dbc.Tab(label="Track Maps", tab_id="tab-track-maps", label_style={"cursor": "pointer"}),
                                dbc.Tab(label="Weather Conditions", tab_id="tab-weather", label_style={"cursor": "pointer"}),
                                dbc.Tab(label="Sector Benchmarking", tab_id="tab-sectors", label_style={"cursor": "pointer"}),
                                dbc.Tab(label="Championships", tab_id="tab-championships", label_style={"cursor": "pointer"}),
                                dbc.Tab(label="Track Animation", tab_id="tab-animation", label_style={"cursor": "pointer"}),
                            ], id="tabs", active_tab="tab-insights", className="mb-3"),
                            html.Div(id="tab-content", style={'min-height': '70vh'})  # Minimum height for content
                        ], width=12),
                    ]),
                ], style={'padding': '1.5rem'}),
            ], className="mb-3", style={'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        ], fluid=True, style={'padding': '0 1rem'}),

        # CHATBOT WIDGET - Floating panel on right side
        html.Div([
            html.Div([
                create_chatbot_layout()
            ], style={
                'position': 'fixed',
                'right': '20px',
                'bottom': '220px',
                'width': '400px',
                'maxHeight': '600px',
                'zIndex': '999',
                'boxShadow': '0 4px 12px rgba(0,0,0,0.15)',
                'borderRadius': '0.5rem',
                'overflow': 'hidden'
            })
        ], id="chatbot-container"),

        # Vehicle selector and hidden components (moved from footer)
        html.Div([
            # Hidden vehicle dropdown for callbacks
            dcc.Dropdown(id='vehicle-dropdown', style={'display': 'none'}),
            dbc.Button(id='analyze-button', style={'display': 'none'}),
            html.Div(id='upload-status', style={'display': 'none'}),
            # Hidden upload for development mode compatibility
            dcc.Upload(id='upload-telemetry', style={'display': 'none'})
        ], style={'display': 'none'}),

        # Corner Analysis Modal - Must exist in base layout for callbacks (Sprint 2 Task 3)
        # Fixed structure: only content areas are updated, close button always exists
        dbc.Modal([
            dbc.ModalHeader(
                html.Div("Corner Analysis", id="modal-header-content")
            ),
            dbc.ModalBody(
                html.Div("Select a vehicle and click an Analyze button to view corner analysis.", id="modal-body-content"),
                style={'maxHeight': '600px', 'overflowY': 'auto'}
            ),
            dbc.ModalFooter([
                dbc.Button(
                    [html.I(className="fas fa-times me-2"), "Close"],
                    id="close-corner-modal",
                    color="secondary"
                )
            ])
        ], id="corner-analysis-modal", size="xl", is_open=False),

    ], style={'background': '#f0f2f5', 'min-height': '100vh', 'paddingBottom': '200px'})

# Main layout - Production Mode (No Upload Page, Auto-load Only)
if PRODUCTION_MODE:
    # Direct dashboard layout with auto-loaded data
    app.layout = html.Div([
        # Store for telemetry data (populated from auto-loaded data)
        dcc.Store(id='upload-data', data=_auto_loaded_data),

        # Theme state storage
        dcc.Store(id='theme-state', storage_type='local', data='light'),

        # Tour state storage
        dcc.Store(id='tour-state', data={
            'welcome_shown': False,
            'tour_completed': False,
            'dont_show_again': False
        }),

        # Processing indicator overlay
        html.Div([
            dbc.Spinner(color="primary", size="lg"),
            html.Span("Loading...", style={'fontSize': '1.2rem', 'marginLeft': '1rem'})
        ], id="processing-overlay", className="processing-overlay", style={'display': 'none'}),

        # Theme toggle button
        dbc.Button(
            html.I(id="theme-icon", className="fas fa-moon"),
            id="theme-toggle",
            className="theme-toggle-btn",
            title="Toggle Light/Dark Mode"
        ),

        # Show dashboard directly (no upload page)
        create_dashboard_page(),

        # Add tour welcome modal
        create_welcome_modal(),

        # Add help documentation system
        create_help_button(),
        create_help_documentation_modal(),
    ], id="main-app-container")
else:
    # Development mode: Keep two-page flow with manual upload
    app.layout = html.Div([
        # Store to track page state
        dcc.Store(id='show-dashboard', data=False),

        # Store to persist uploaded telemetry data across page switches
        dcc.Store(id='upload-data'),

        # Tour state storage
        dcc.Store(id='tour-state', data={
            'welcome_shown': False,
            'tour_completed': False,
            'dont_show_again': False
        }),

        # Page container (shows upload page initially, then switches to dashboard after upload)
        html.Div(id='page-container', children=create_upload_page()),

        # Add tour welcome modal
        create_welcome_modal(),

        # Add help documentation system
        create_help_button(),
        create_help_documentation_modal(),

        # CHATBOT WIDGET - Floating panel on right side
        html.Div([
            html.Div([
                create_chatbot_layout()
            ], style={
                'position': 'fixed',
                'right': '20px',
                'bottom': '220px',
                'width': '400px',
                'maxHeight': '600px',
                'zIndex': '999',
                'boxShadow': '0 4px 12px rgba(0,0,0,0.15)',
                'borderRadius': '0.5rem',
                'overflow': 'hidden'
            })
        ], id="chatbot-container")
    ])

#================================================================
# CALLBACKS
#================================================================

# Callback to render initial page based on show-dashboard state (Development Mode Only)
if not PRODUCTION_MODE:
    @app.callback(
        Output('page-container', 'children'),
        Input('show-dashboard', 'data')
    )
    def render_page(show_dashboard):
        """Render upload page or dashboard based on state (Development Mode)"""
        if show_dashboard:
            return create_dashboard_page()
        else:
            return create_upload_page()

# Callback to populate stats on page load (Production Mode)
if PRODUCTION_MODE:
    @app.callback(
        [Output('vehicle-dropdown', 'options'),
         Output('stat-samples', 'children'),
         Output('stat-vehicles', 'children'),
         Output('stat-laps', 'children'),
         Output('analyze-button', 'disabled')],
        [Input('upload-data', 'data')],  # Triggers when page loads with auto-loaded data
        prevent_initial_call=False
    )
    def populate_stats_on_load(data_json):
        """Populate dashboard stats from auto-loaded data (Production Mode)"""
        if data_json is None or not _auto_loaded_stats:
            # No data loaded
            return ([], "0", "0", "0", True)

        # Use pre-calculated stats from auto-load
        return (
            _auto_loaded_stats['vehicle_options'],
            f"{_auto_loaded_stats['num_samples']:,}",
            str(_auto_loaded_stats['num_vehicles']),
            str(_auto_loaded_stats['num_laps']),
            False  # Enable analyze button
        )

# Upload callback (Development Mode Only - Production Mode uses auto-loaded data)
if not PRODUCTION_MODE:
    @app.callback(
        [Output('upload-status', 'children'),
         Output('upload-data', 'data'),
         Output('vehicle-dropdown', 'options'),
         Output('stat-samples', 'children'),
         Output('stat-vehicles', 'children'),
         Output('stat-laps', 'children'),
         Output('analyze-button', 'disabled'),
         Output('show-dashboard', 'data')],  # Add output to switch to dashboard
        [Input('upload-telemetry', 'contents')],
        [State('upload-telemetry', 'filename')]
    )
    def handle_upload(contents, filename):
        """Handle telemetry file upload (Development Mode Only)"""
        if contents is None:
            return ("", "", [], "0", "0", "0", True, False)  # Stay on upload page

        try:
            # Decode uploaded file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            # Check file size (limit to 500MB for web upload)
            file_size_mb = len(decoded) / (1024 * 1024)
            if file_size_mb > 500:
                error_msg = dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    html.Strong("File too large! "),
                    f"({file_size_mb:.1f}MB) ",
                    html.Br(),
                    "Maximum file size for web upload is 500MB. ",
                    html.Br(),
                    html.Small("For larger files (like 2.5GB), please use one of these options:", className="d-block mt-2"),
                    html.Ul([
                        html.Li("Use organize_and_chunk_data.py to split into smaller chunks"),
                        html.Li("Use data_loader.py to load data directly from organized_data/"),
                        html.Li("Sample your data first (e.g., every 10th row)")
                    ], className="small mt-1")
                ], color="danger", className="mb-0")
                return (error_msg, "", [], "0", "0", "0", "--", True, False)  # Stay on upload page

            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            # Store data as JSON
            data_json = df.to_json(date_format='iso', orient='split')

            # Get statistics
            num_samples = len(df)
            vehicles = sorted(df['vehicle_number'].unique()) if 'vehicle_number' in df.columns else []
            num_vehicles = len(vehicles)
            num_laps = df['lap'].nunique() if 'lap' in df.columns else 0

            # Calculate average lap time if lap_times data
            avg_time = "--"

            # Create vehicle dropdown options
            vehicle_options = [{'label': f'Vehicle #{v}', 'value': v} for v in vehicles]

            # Success message
            status_msg = dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                f"Successfully loaded {filename} ({num_samples:,} samples)"
            ], color="success", className="mb-0")

            return (
                status_msg,
                data_json,
                vehicle_options,
                f"{num_samples:,}",
                str(num_vehicles),
                str(num_laps),
                avg_time,
                False,  # Enable analyze button
                True    # Switch to dashboard page
            )

        except Exception as e:
            error_msg = dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"Error loading file: {str(e)}"
            ], color="danger", className="mb-0")
            return (error_msg, "", [], "0", "0", "0", "--", True, False)  # Stay on upload page on error

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'active_tab'),
    Input('analyze-button', 'n_clicks'),
    Input('vehicle-dropdown', 'value'),
    State('upload-data', 'data'),
    prevent_initial_call=False
)
def render_tab_content(active_tab, n_clicks, vehicle_number, data_json):
    """Render content based on selected tab"""
    # Add logging to debug
    import datetime
    logger.info(f"[{datetime.datetime.now()}] Callback triggered - Tab: {active_tab}, Vehicle: {vehicle_number}")

    # Week 1 tabs, animation tab, and post-race tab don't require telemetry upload
    if active_tab in ["tab-weather", "tab-sectors", "tab-championships", "tab-animation", "tab-post-race"]:
        if active_tab == "tab-animation":
            # Animation tab has its own upload mechanism
            return create_animation_layout()

        if active_tab == "tab-post-race":
            # Post-Race Analysis tab has its own upload mechanism
            return create_post_race_layout()

        if not WEEK1_ENABLED:
            return dbc.Alert("Week 1 features not available. Missing data loaders.", color="warning")

        if active_tab == "tab-weather":
            return render_weather_tab()
        elif active_tab == "tab-sectors":
            return render_sectors_tab()
        elif active_tab == "tab-championships":
            return render_championships_tab()

    # Original tabs require telemetry data
    if not data_json or vehicle_number is None:
        return dbc.Alert("Please upload telemetry data and select a vehicle to analyze.", color="info")

    try:
        # Load data
        df = pd.read_json(io.StringIO(data_json), orient='split')

        if active_tab == "tab-insights":
            logger.info(f"Rendering driver insights for vehicle #{vehicle_number}")
            return render_driver_insights(df, vehicle_number)
        elif active_tab == "tab-telemetry":
            return render_telemetry_charts(df, vehicle_number)
        elif active_tab == "tab-predictions":
            return render_predictions(df, vehicle_number)
        elif active_tab == "tab-track-maps":
            return render_track_maps(df, vehicle_number)

    except Exception as e:
        logger.error(f"Error rendering content: {str(e)}")
        return dbc.Alert(f"Error rendering content: {str(e)}", color="danger")

def _extract_lap_times_from_telemetry(df: pd.DataFrame, vehicle_number: int) -> pd.DataFrame:
    """
    Extract lap times from telemetry DataFrame.

    Creates a lap_times DataFrame from telemetry data by aggregating
    timestamps per lap.

    Args:
        df: Telemetry DataFrame (long format)
        vehicle_number: Vehicle ID

    Returns:
        DataFrame with columns: vehicle_number, lap_number, lap_time, track, race
    """
    # Filter for vehicle
    vehicle_df = df[df['vehicle_number'] == vehicle_number].copy()

    if vehicle_df.empty:
        return pd.DataFrame(columns=['vehicle_number', 'lap_number', 'lap_time', 'track', 'race'])

    # Group by lap and calculate lap time (max timestamp - min timestamp)
    lap_groups = vehicle_df.groupby('lap')

    lap_times_list = []
    for lap_num, group in lap_groups:
        # Calculate lap time in seconds
        min_time = group['timestamp'].min()
        max_time = group['timestamp'].max()
        lap_time = (max_time - min_time) / 1000.0  # Convert ms to seconds

        # Get track and race info if available
        track = group['track'].iloc[0] if 'track' in group.columns and len(group) > 0 else 'unknown'
        race = group['race'].iloc[0] if 'race' in group.columns and len(group) > 0 else 'unknown'

        lap_times_list.append({
            'vehicle_number': vehicle_number,
            'lap_number': int(lap_num),
            'lap_time': lap_time,
            'track': track,
            'race': race
        })

    lap_times_df = pd.DataFrame(lap_times_list)

    # Filter out obviously bad lap times (pit laps, incomplete laps)
    if len(lap_times_df) > 0:
        # Remove laps < 60s or > 300s (likely incomplete/pit laps)
        lap_times_df = lap_times_df[
            (lap_times_df['lap_time'] >= 60) &
            (lap_times_df['lap_time'] <= 300)
        ].copy()

    logger.info(f"Extracted {len(lap_times_df)} valid laps for vehicle {vehicle_number}")
    return lap_times_df


def render_driver_insights(df, vehicle_number):
    """Render driver insights tab - Enhanced with Phase 1.2 features"""
    import time
    import random

    # Try using enhanced widget if available
    if ENHANCED_INSIGHTS_ENABLED:
        try:
            # Extract lap times from telemetry if available
            lap_times_df = _extract_lap_times_from_telemetry(df, vehicle_number)

            # Use enhanced driver insights widget
            return create_enhanced_driver_insights_layout(
                df=df,
                vehicle_number=vehicle_number,
                lap_times_df=lap_times_df,
                selected_lap=None  # Auto-select best lap
            )
        except Exception as e:
            logger.warning(f"Enhanced insights failed, falling back to API-only mode: {e}")
            # Fall through to API-only mode

    # Fallback: Original API-based insights
    render_timestamp = time.time()
    unique_key = f"{vehicle_number}_{render_timestamp}_{random.randint(1000, 9999)}"

    try:
        # Call API for driver insights with cache-busting timestamp
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        logger.info(f"[RENDER] Calling API for vehicle #{vehicle_number} at {render_timestamp}")

        response = requests.post(
            f"{API_BASE}/driver-insights",
            params={
                "vehicle_number": vehicle_number,
                "_t": render_timestamp  # Cache buster
            },
            files={"file": ("telemetry.csv", csv_buffer, "text/csv")},
            timeout=30,
            headers={"Cache-Control": "no-cache", "Pragma": "no-cache"}
        )

        if response.status_code != 200:
            error_detail = response.json().get('detail', 'Unknown error')
            return dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"API Error: {error_detail}"
            ], color="danger", id=f"error-{unique_key}")

        insights = response.json()
        perf = insights['performance_summary']

        logger.info(f"[RENDER] Received insights for vehicle #{vehicle_number}: max_speed={perf['max_speed']}")

        return dbc.Container([
            # Add hidden div with unique key to force re-render
            html.Div(id=f"render-key-{unique_key}", style={"display": "none"}),
            # Performance scores
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{perf['consistency_score']:.1f}", className="text-center mb-0"),
                            html.P("Consistency", className="text-muted text-center mb-0 small")
                        ])
                    ], color="primary", outline=True)
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{perf['aggression_index']:.1f}", className="text-center mb-0"),
                            html.P("Aggression", className="text-muted text-center mb-0 small")
                        ])
                    ], color="warning", outline=True)
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{perf['smoothness_rating']:.1f}", className="text-center mb-0"),
                            html.P("Smoothness", className="text-muted text-center mb-0 small")
                        ])
                    ], color="success", outline=True)
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
                            html.Ul([html.Li(s) for s in insights['strengths']]) if insights['strengths'] else html.P("No specific strengths identified", className="text-muted mb-0")
                        ])
                    ], className="shadow-sm h-100")
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6([html.I(className="fas fa-exclamation-circle me-2 text-warning"), "Areas for Improvement"])),
                        dbc.CardBody([
                            html.Ul([html.Li(w) for w in insights['weaknesses']]) if insights['weaknesses'] else html.P("No specific weaknesses identified", className="text-muted mb-0")
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
        ], fluid=True)

    except requests.exceptions.ConnectionError:
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Cannot connect to API server. Please ensure the API is running at ",
            html.Code("http://localhost:8000")
        ], color="danger")
    except requests.exceptions.Timeout:
        return dbc.Alert([
            html.I(className="fas fa-clock me-2"),
            "Request timed out. The analysis may take longer for large files."
        ], color="warning")
    except Exception as e:
        return dbc.Alert([
            html.I(className="fas fa-exclamation-circle me-2"),
            f"Error generating insights: {str(e)}"
        ], color="danger")

def render_telemetry_charts(df, vehicle_number):
    """Render telemetry overlay charts"""
    try:
        # Filter data for selected vehicle
        vehicle_df = df[df['vehicle_number'] == vehicle_number].copy()

        if len(vehicle_df) == 0:
            return dbc.Alert(f"No data found for vehicle #{vehicle_number}", color="warning")

        # Pivot data for plotting
        pivot_df = vehicle_df.pivot_table(
            index='timestamp',
            columns='telemetry_name',
            values='telemetry_value',
            aggfunc='first'
        ).reset_index()

        # Create subplots for different sensors
        charts = []

        # Speed chart
        if 'speed' in pivot_df.columns:
            speed_fig = go.Figure()
            speed_fig.add_trace(go.Scatter(
                x=pivot_df.index,
                y=pivot_df['speed'],
                mode='lines',
                name='Speed',
                line=dict(color=COLORS[0], width=2)
            ))
            speed_fig.update_layout(
                title="Speed (km/h)",
                height=200,
                margin=dict(l=50, r=20, t=40, b=30),
                showlegend=False
            )
            charts.append(dcc.Graph(figure=speed_fig, config={'displayModeBar': False}))

        # Brake pressure
        if 'pbrake_f' in pivot_df.columns:
            brake_fig = go.Figure()
            brake_fig.add_trace(go.Scatter(
                x=pivot_df.index,
                y=pivot_df['pbrake_f'],
                mode='lines',
                name='Brake Pressure',
                line=dict(color=COLORS[1], width=2),
                fill='tozeroy'
            ))
            brake_fig.update_layout(
                title="Brake Pressure (bar)",
                height=200,
                margin=dict(l=50, r=20, t=40, b=30),
                showlegend=False
            )
            charts.append(dcc.Graph(figure=brake_fig, config={'displayModeBar': False}))

        # Throttle
        if 'aps' in pivot_df.columns:
            throttle_fig = go.Figure()
            throttle_fig.add_trace(go.Scatter(
                x=pivot_df.index,
                y=pivot_df['aps'],
                mode='lines',
                name='Throttle',
                line=dict(color=COLORS[2], width=2),
                fill='tozeroy'
            ))
            throttle_fig.update_layout(
                title="Throttle Position (%)",
                height=200,
                margin=dict(l=50, r=20, t=40, b=30),
                showlegend=False
            )
            charts.append(dcc.Graph(figure=throttle_fig, config={'displayModeBar': False}))

        # Lateral G
        if 'accy_can' in pivot_df.columns:
            g_fig = go.Figure()
            g_fig.add_trace(go.Scatter(
                x=pivot_df.index,
                y=pivot_df['accy_can'],
                mode='lines',
                name='Lateral G',
                line=dict(color=COLORS[3], width=2)
            ))
            g_fig.update_layout(
                title="Lateral G-Force",
                height=200,
                margin=dict(l=50, r=20, t=40, b=30),
                showlegend=False
            )
            charts.append(dcc.Graph(figure=g_fig, config={'displayModeBar': False}))

        return html.Div(charts)

    except Exception as e:
        return dbc.Alert(f"Error rendering charts: {str(e)}", color="danger")

def render_predictions(df, vehicle_number):
    """Render model predictions tab with enhanced categorized features (Sprint 1 + Sprint 2)"""
    try:
        # Import enhanced widget
        from src.dashboard.model_predictions_widget import (
            create_model_predictions_layout,
            create_model_predictions_error
        )

        # Step 1: Extract features using API
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        features_response = requests.post(
            f"{API_BASE}/extract-features",
            files={"file": ("telemetry.csv", csv_buffer, "text/csv")},
            timeout=30
        )

        if features_response.status_code != 200:
            error_detail = features_response.json().get('detail', 'Unknown error')
            return create_model_predictions_error('extraction', f"Feature extraction failed: {error_detail}")

        features_data = features_response.json()

        # Step 2: Make prediction if sample features available
        prediction_result = None
        if features_data.get('sample_features'):
            try:
                pred_response = requests.post(
                    f"{API_BASE}/predict",
                    json={"features": features_data['sample_features']},
                    timeout=10
                )

                if pred_response.status_code == 200:
                    prediction_result = pred_response.json()
                elif pred_response.status_code == 503:
                    # Model not loaded - show features anyway
                    logger.warning("Model not loaded, showing features only")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Prediction request failed: {e}")

        # Step 3: Run cube analysis on telemetry (Sprint 2 Task 2 + Task 5)
        patterns_data = None
        corner_analyses = None
        try:
            from src.services.telemetry_analyzer import get_telemetry_analyzer
            logger.info(f"Running cube analysis on {len(df):,} rows...")

            analyzer = get_telemetry_analyzer()

            # Analyze patterns
            patterns_data = analyzer.analyze_telemetry(df, use_cache=True)
            if patterns_data and len(patterns_data) > 0:
                logger.info(f"[OK] Cube analysis complete: {len(patterns_data)} patterns detected")
            else:
                logger.info("No patterns detected in telemetry")

            # Analyze corners (Sprint 2 Task 5)
            corner_analyses = analyzer.analyze_corners(df, use_cache=True)
            if corner_analyses and len(corner_analyses) > 0:
                logger.info(f"[OK] Corner analysis complete: {len(corner_analyses)} corners detected")
                # Store globally for corner analysis callback access
                global _cached_corner_analyses
                _cached_corner_analyses = corner_analyses
            else:
                logger.info("No corners detected in telemetry")

        except Exception as e:
            logger.warning(f"Cube analysis failed, will use demo patterns: {str(e)}")
            patterns_data = None
            corner_analyses = None

        # Step 4: Render using enhanced widget (Sprint 1 + Sprint 2 + Task 6)
        # Default to COTA if track not detectable from data
        track_name = 'circuit-of-the-americas'  # TODO: Detect from GPS data or user input

        return create_model_predictions_layout(
            features_data,
            prediction_result,
            patterns_data,
            corner_analyses,  # Sprint 2 Task 6
            track_name  # Sprint 2 Task 6
        )

    except requests.exceptions.ConnectionError:
        from src.dashboard.model_predictions_widget import create_model_predictions_error
        return create_model_predictions_error('connection', "Cannot connect to API server at http://localhost:8000")
    except requests.exceptions.Timeout:
        from src.dashboard.model_predictions_widget import create_model_predictions_error
        return create_model_predictions_error('timeout', "Request timed out. The file may be too large or the server is busy.")
    except Exception as e:
        from src.dashboard.model_predictions_widget import create_model_predictions_error
        return create_model_predictions_error('extraction', f"Error generating predictions: {str(e)}")

def render_track_maps(df, vehicle_number):
    """Render track maps tab with telemetry overlay"""
    try:
        # Import track modules - adjust import path since we're in src/dashboard
        import sys
        from pathlib import Path
        # Add parent directory to path if not already there
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from track_data.track_metadata import get_available_tracks, get_track_metadata
        from track_data.track_images import TrackImageLoader, get_track_image_path

        # Initialize track image loader
        try:
            loader = TrackImageLoader()
        except FileNotFoundError:
            return dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Track map images not found. Please run: python convert_track_maps_to_images.py"
            ], color="warning")

        # Get available tracks
        tracks = get_available_tracks()

        # Detect track from telemetry data if possible
        # For now, default to first track
        default_track = tracks[0] if tracks else None

        # Create layout with improved spacing and structure
        return dbc.Container([
            # Main track map card
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-map me-2"),
                            "Track Map Visualization"
                        ]),
                        dbc.CardBody([
                            # Track selector
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Track:", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id='track-selector',
                                        options=[
                                            {'label': track.replace('-', ' ').title(), 'value': track}
                                            for track in tracks
                                        ],
                                        value=default_track,
                                        clearable=False
                                    )
                                ], md=6, className="mb-3"),
                                dbc.Col([
                                    html.Label("Map Quality:", className="fw-bold mb-2"),
                                    dbc.RadioItems(
                                        id='map-quality',
                                        options=[
                                            {'label': 'Standard (150 DPI)', 'value': 'standard'},
                                            {'label': 'HD (300 DPI)', 'value': 'hd'}
                                        ],
                                        value='hd',
                                        inline=True
                                    )
                                ], md=6, className="mb-3")
                            ]),

                            # Track map display with full width and responsive height
                            dcc.Graph(
                                id='track-map-display',
                                style={'height': '900px', 'width': '100%'},
                                config={
                                    'responsive': True,
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'toImageButtonOptions': {
                                        'format': 'png',
                                        'filename': 'track_map',
                                        'height': 900,
                                        'width': 1400,
                                        'scale': 2
                                    }
                                }
                            ),

                            # Telemetry overlay options
                            html.Hr(className="my-4"),
                            html.H6([
                                html.I(className="fas fa-layer-group me-2"),
                                "Telemetry Overlay Options"
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Checklist(
                                        id='overlay-options',
                                        options=[
                                            {'label': ' Show Speed Trace', 'value': 'speed'},
                                            {'label': ' Show Braking Zones', 'value': 'brake'},
                                            {'label': ' Show Throttle Application', 'value': 'throttle'},
                                            {'label': ' Show Lateral G-Force', 'value': 'lateral_g'},
                                        ],
                                        value=['speed', 'brake'],
                                        switch=True
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("Lap Selection:", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id='lap-selector',
                                        options=[],
                                        value=None,
                                        placeholder="Select lap..."
                                    )
                                ], md=6)
                            ])
                        ])
                    ], className="shadow-sm")
                ], width=12)
            ], className="mb-4"),

            # Track information cards with proper spacing
            dbc.Row(id='track-info-cards', className="mb-4"),

            # Circuit Configuration Table - OPTION 2 STACKED LAYOUT
            html.Hr(className="my-4"),
            dbc.Row([
                dbc.Col(html.Div(id='circuit-config-table'), width=12)
            ], className="mb-4"),

            # Sector performance with proper spacing
            dbc.Row(id='sector-performance')
        ], fluid=True)

    except Exception as e:
        logger.error(f"Error in render_track_maps: {e}", exc_info=True)
        return dbc.Alert([
            html.I(className="fas fa-exclamation-circle me-2"),
            f"Error loading track maps: {str(e)}"
        ], color="danger")

# Add callback for track map display
@app.callback(
    Output('track-map-display', 'figure'),
    Output('track-info-cards', 'children'),
    Output('circuit-config-table', 'children'),
    Output('lap-selector', 'options'),
    Input('track-selector', 'value'),
    Input('map-quality', 'value'),
    Input('overlay-options', 'value'),
    Input('lap-selector', 'value'),
    State('upload-data', 'data'),
    State('vehicle-dropdown', 'value')
)
def update_track_map(track_name, quality, overlay_options, selected_lap, data_json, vehicle_number):
    """Update track map display with telemetry overlay and circuit configuration"""
    empty_fig = go.Figure()
    empty_fig.update_layout(title="Select a track to display")

    if not track_name:
        return empty_fig, [], html.Div(), []

    try:
        # Adjust import paths for track modules
        import sys
        from pathlib import Path
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from track_data.track_metadata import get_track_metadata
        from track_data.track_visualizer import TrackVisualizer
        from dashboard.circuit_config_table import create_circuit_config_table, create_circuit_summary_badge

        # Get track metadata
        metadata = get_track_metadata(track_name)
        if not metadata:
            return empty_fig, [], html.Div(), []

        # Create track visualizer
        visualizer = TrackVisualizer(track_name)

        # Get lap options from telemetry
        lap_options = []
        telemetry_df = None

        if data_json and vehicle_number is not None:
            try:
                telemetry_df = pd.read_json(io.StringIO(data_json), orient='split')
                vehicle_df = telemetry_df[telemetry_df['vehicle_number'] == vehicle_number]
                laps = sorted(vehicle_df['lap'].unique()) if 'lap' in vehicle_df.columns else []
                lap_options = [{'label': f'Lap {lap}', 'value': lap} for lap in laps]
            except Exception as e:
                logger.error(f"Error loading telemetry: {e}")
                telemetry_df = None

        # Create figure based on options
        if telemetry_df is not None and vehicle_number is not None and len(overlay_options) > 0:
            # Create overlay visualization
            show_speed = 'speed' in overlay_options
            show_braking = 'brake' in overlay_options
            show_throttle = 'throttle' in overlay_options
            show_lateral_g = 'lateral_g' in overlay_options

            fig = visualizer.create_track_overlay(
                telemetry_df,
                vehicle_number,
                lap=selected_lap,
                show_speed=show_speed,
                show_braking=show_braking,
                show_sectors=True
            )
        else:
            # Show base track map without overlay
            fig = visualizer.create_base_figure()

        # Create track info cards
        info_cards = [
            dbc.Col([
                create_stat_card("Track Length", f"{metadata['length_miles']} mi", "road", "#2e7d32")
            ], md=3),
            dbc.Col([
                create_stat_card("Turns", str(len(metadata['turns'])), "flag", "#1565c0")
            ], md=3),
            dbc.Col([
                create_stat_card("Sectors", "3", "chart-pie", "#f57c00")
            ], md=3),
            dbc.Col([
                create_stat_card("Elevation", f"{metadata['elevation_ft']} ft", "mountain", "#6a1b9a")
            ], md=3),
        ]

        # Create circuit configuration table - OPTION 2 STACKED LAYOUT
        circuit_config = html.Div([
            create_circuit_summary_badge(metadata),
            create_circuit_config_table(metadata)
        ])

        return fig, info_cards, circuit_config, lap_options

    except Exception as e:
        logger.error(f"Error updating track map: {e}", exc_info=True)
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error: {str(e)}")
        error_alert = dbc.Alert(f"Error loading circuit data: {str(e)}", color="danger")
        return error_fig, [], error_alert, []

#================================================================
# WEEK 1 TAB RENDERERS
#================================================================

def render_weather_tab():
    """Render weather conditions tab"""
    if not WEEK1_ENABLED:
        return dbc.Alert("Weather data not available", color="warning")

    try:
        return html.Div([
            html.H3([
                html.I(className="fas fa-cloud-sun me-2"),
                "Weather Conditions Analysis"
            ], className="mb-4", style={'color': '#2c3e50'}),
            create_weather_layout()
        ])
    except Exception as e:
        logger.error(f"Error rendering weather tab: {e}")
        return dbc.Alert(f"Error loading weather data: {str(e)}", color="danger")

def render_sectors_tab():
    """Render sector benchmarking tab"""
    if not WEEK1_ENABLED:
        return dbc.Alert("Sector data not available", color="warning")

    try:
        return html.Div([
            html.H3([
                html.I(className="fas fa-stopwatch me-2"),
                "Sector Time Benchmarking"
            ], className="mb-4", style={'color': '#2c3e50'}),
            create_sector_layout()
        ])
    except Exception as e:
        logger.error(f"Error rendering sectors tab: {e}")
        return dbc.Alert(f"Error loading sector data: {str(e)}", color="danger")

def render_championships_tab():
    """Render championships tab"""
    if not WEEK1_ENABLED:
        return dbc.Alert("Championship data not available", color="warning")

    try:
        from dash import dash_table
        champ_data = championship_loader.load_all_championships()

        if not champ_data:
            return dbc.Alert("No championship data found", color="warning")

        champ_names = list(champ_data.keys())

        return html.Div([
            html.H3([
                html.I(className="fas fa-trophy me-2"),
                "Championship Standings"
            ], className="mb-4", style={'color': '#2c3e50'}),

            # Championship selector
            dbc.Row([
                dbc.Col([
                    html.Label("Select Championship:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='championship-dropdown-integrated',
                        options=[{'label': name, 'value': name} for name in champ_names],
                        value=champ_names[0] if champ_names else None,
                        style={'marginBottom': '20px'}
                    )
                ], md=12)
            ], className="mb-4"),

            # Championship content
            html.Div(id='championship-content-integrated')
        ])
    except Exception as e:
        logger.error(f"Error rendering championships tab: {e}")
        return dbc.Alert(f"Error loading championship data: {str(e)}", color="danger")

# Championship callback (integrated version)
if WEEK1_ENABLED:
    @app.callback(
        Output('championship-content-integrated', 'children'),
        Input('championship-dropdown-integrated', 'value')
    )
    def update_championship_integrated(championship_name):
        """Update championship display in integrated dashboard"""
        if not championship_name:
            return dbc.Alert("Select a championship", color="info")

        try:
            from dash import dash_table
            champ_data = championship_loader.load_all_championships()

            if championship_name not in champ_data:
                return dbc.Alert("Championship not found", color="warning")

            df = champ_data[championship_name]

            # Prepare standings table
            display_cols = ['Pos', 'Participant', 'Points']
            if 'TEAM' in df.columns:
                display_cols.append('TEAM')

            df_display = df[df['Participant'].notna()].copy()
            if 'Points' in df_display.columns:
                df_display['Points'] = pd.to_numeric(df_display['Points'], errors='coerce').fillna(0)

            if 'Pos' in df_display.columns:
                df_display = df_display.sort_values('Pos')
            else:
                df_display = df_display.sort_values('Points', ascending=False)
                df_display['Pos'] = range(1, len(df_display) + 1)

            table_data = df_display[display_cols].head(20)

            # Create standings table
            standings_table = dash_table.DataTable(
                data=table_data.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in display_cols],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '14px'},
                style_header={
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'fontSize': '16px'
                },
                style_data_conditional=[
                    {'if': {'row_index': 0}, 'backgroundColor': '#f1c40f', 'fontWeight': 'bold'},
                    {'if': {'row_index': 1}, 'backgroundColor': '#ecf0f1'},
                    {'if': {'row_index': 2}, 'backgroundColor': '#cd7f32', 'color': 'white'}
                ]
            )

            # Create points chart
            top10 = df_display.head(10)
            points_fig = go.Figure()
            points_fig.add_trace(go.Bar(
                x=top10['Participant'],
                y=top10['Points'],
                marker_color='#3498db',
                text=top10['Points'],
                textposition='outside'
            ))
            points_fig.update_layout(
                title="Top 10 Championship Points",
                xaxis_title="Participant",
                yaxis_title="Points",
                height=400,
                showlegend=False,
                xaxis={'tickangle': -45}
            )

            return html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Current Standings")),
                            dbc.CardBody(standings_table)
                        ], className="shadow-sm mb-4")
                    ], md=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Points Distribution")),
                            dbc.CardBody(dcc.Graph(figure=points_fig, config={'displayModeBar': False}))
                        ], className="shadow-sm")
                    ], md=12)
                ])
            ])

        except Exception as e:
            logger.error(f"Error updating championship: {e}")
            return dbc.Alert(f"Error: {str(e)}", color="danger")

# ============================================================================
# CORNER ANALYSIS MODAL CALLBACKS (Sprint 2 Task 3)
# ============================================================================

@app.callback(
    [Output('corner-analysis-modal', 'is_open'),
     Output('modal-header-content', 'children'),
     Output('modal-body-content', 'children')],
    [Input('analyze-btn-speed', 'n_clicks'),
     Input('analyze-btn-braking', 'n_clicks'),
     Input('analyze-btn-cornering', 'n_clicks'),
     Input('analyze-btn-throttle', 'n_clicks'),
     Input('analyze-btn-steering', 'n_clicks'),
     Input('analyze-btn-powertrain', 'n_clicks'),
     Input('analyze-btn-composite', 'n_clicks'),
     Input('analyze-btn-lap_seg', 'n_clicks'),
     Input('analyze-btn-uncategorized', 'n_clicks'),
     Input('close-corner-modal', 'n_clicks')],
    [State('corner-analysis-modal', 'is_open')],
    prevent_initial_call=True
)
def toggle_corner_modal(speed_clicks, brake_clicks, corner_clicks, throttle_clicks,
                       steering_clicks, powertrain_clicks, composite_clicks,
                       lap_seg_clicks, uncategorized_clicks, close_clicks, is_open):
    """
    Toggle corner analysis modal when Analyze buttons are clicked.

    This callback handles:
    - Opening modal when any category's "Analyze" button is clicked
    - Closing modal when "Close" button is clicked
    - Loading real corner analysis from uploaded telemetry (Sprint 2 Task 5)

    Approach 1: Fixed modal structure - only updates header and body content,
    never replaces the close button (prevents DOM removal bug).
    """
    ctx = callback_context
    if not ctx.triggered:
        return False, "Corner Analysis", "Select a vehicle and click an Analyze button to view corner analysis."

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Close button clicked
    if button_id == 'close-corner-modal':
        return False, "Corner Analysis", "Select a vehicle and click an Analyze button to view corner analysis."

    # Analyze button clicked - open modal
    if button_id.startswith('analyze-btn-'):
        # Extract category name from button ID
        category_id = button_id.replace('analyze-btn-', '')
        category_names = {
            'speed': 'Speed & Acceleration',
            'braking': 'Braking Performance',
            'cornering': 'Cornering Dynamics',
            'throttle': 'Throttle Management',
            'steering': 'Steering Control',
            'powertrain': 'Powertrain & Gear Management',
            'composite': 'Composite Performance Metrics',
            'lap_seg': 'Lap Segmentation & Timing',
            'uncategorized': 'Other Features'
        }
        category_name = category_names.get(category_id, 'Performance')

        # Load corner analysis widget
        from src.dashboard.corner_analysis_widget import create_corner_analysis_content, DEMO_CORNER_ANALYSES

        # Sprint 2 Task 5: Use real corner analysis from telemetry
        global _cached_corner_analyses
        corner_analyses = _cached_corner_analyses if _cached_corner_analyses else DEMO_CORNER_ANALYSES

        if corner_analyses == DEMO_CORNER_ANALYSES:
            logger.info("Using demo corner data (no telemetry analyzed yet)")
        else:
            logger.info(f"Using real corner analysis: {len(corner_analyses)} corners detected")

        # Get header and body content separately
        header_content = [
            html.I(className="fas fa-map-marked-alt me-2"),
            f"Corner Analysis - {category_name}"
        ]
        body_content = create_corner_analysis_content(corner_analyses, category_id)

        return True, header_content, body_content

    return is_open, "Corner Analysis", "Select a vehicle and click an Analyze button to view corner analysis."


# ============================================================================
# CATEGORY FILTER CALLBACKS (Sprint 2 Task 4)
# ============================================================================

@app.callback(
    [
        # Button style outputs (to show active state)
        Output('filter-btn-all', 'color'),
        Output('filter-btn-critical', 'color'),
        Output('filter-btn-important', 'color'),
        Output('filter-btn-advanced', 'color'),
        Output('filter-btn-all', 'outline'),
        Output('filter-btn-critical', 'outline'),
        Output('filter-btn-important', 'outline'),
        Output('filter-btn-advanced', 'outline'),
        # Category visibility outputs (10 categories)
        Output('category-container-speed', 'style'),
        Output('category-container-braking', 'style'),
        Output('category-container-throttle', 'style'),
        Output('category-container-cornering', 'style'),
        Output('category-container-steering', 'style'),
        Output('category-container-powertrain', 'style'),
        Output('category-container-composite', 'style'),
        Output('category-container-fft', 'style'),
        Output('category-container-wavelet', 'style'),
        Output('category-container-lap_seg', 'style'),
    ],
    [
        Input('filter-btn-all', 'n_clicks'),
        Input('filter-btn-critical', 'n_clicks'),
        Input('filter-btn-important', 'n_clicks'),
        Input('filter-btn-advanced', 'n_clicks'),
    ],
    prevent_initial_call=True
)
def filter_categories_by_importance(all_clicks, critical_clicks, important_clicks, advanced_clicks):
    """
    Filter feature categories by importance level.

    This callback handles:
    - Filtering categories when importance filter buttons are clicked
    - Highlighting the active filter button
    - Showing/hiding categories based on their importance level

    Category importance levels:
    - CRITICAL: speed, braking, throttle, cornering, lap_seg (5 categories)
    - IMPORTANT: steering, powertrain, composite (3 categories)
    - ADVANCED: fft, wavelet (2 categories)
    """
    ctx = callback_context
    if not ctx.triggered:
        # Default: show all categories
        return (
            'primary', 'danger', 'warning', 'info',  # Button colors
            False, True, True, True,  # Button outline states (All is solid)
            *[{'border': '1px solid #dee2e6', 'borderRadius': '0.25rem', 'overflow': 'hidden'}] * 10  # All visible
        )

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Base style for visible categories
    visible_style = {
        'border': '1px solid #dee2e6',
        'borderRadius': '0.25rem',
        'overflow': 'hidden',
        'display': 'block'
    }

    # Style for hidden categories
    hidden_style = {
        'display': 'none'
    }

    # Category importance mapping (matches feature_categories.py)
    # Order: speed, braking, throttle, cornering, steering, powertrain, composite, fft, wavelet, lap_seg
    category_importance = [
        'critical',  # speed
        'critical',  # braking
        'critical',  # throttle
        'critical',  # cornering
        'important', # steering
        'important', # powertrain
        'important', # composite
        'advanced',  # fft
        'advanced',  # wavelet
        'critical',  # lap_seg
    ]

    if button_id == 'filter-btn-all':
        # Show all categories
        return (
            'primary', 'danger', 'warning', 'info',  # Button colors
            False, True, True, True,  # Outline states (All is solid, others outline)
            *[visible_style] * 10  # All categories visible
        )

    elif button_id == 'filter-btn-critical':
        # Show only CRITICAL categories
        category_styles = [
            visible_style if imp == 'critical' else hidden_style
            for imp in category_importance
        ]
        return (
            'primary', 'danger', 'warning', 'info',  # Button colors
            True, False, True, True,  # Outline states (Critical is solid)
            *category_styles
        )

    elif button_id == 'filter-btn-important':
        # Show only IMPORTANT categories
        category_styles = [
            visible_style if imp == 'important' else hidden_style
            for imp in category_importance
        ]
        return (
            'primary', 'danger', 'warning', 'info',  # Button colors
            True, True, False, True,  # Outline states (Important is solid)
            *category_styles
        )

    elif button_id == 'filter-btn-advanced':
        # Show only ADVANCED categories
        category_styles = [
            visible_style if imp == 'advanced' else hidden_style
            for imp in category_importance
        ]
        return (
            'primary', 'danger', 'warning', 'info',  # Button colors
            True, True, True, False,  # Outline states (Advanced is solid)
            *category_styles
        )

    # Default fallback: show all
    return (
        'primary', 'danger', 'warning', 'info',
        False, True, True, True,
        *[visible_style] * 10
    )


# ============================================================================
# WEEK 1 WIDGET CALLBACKS
# ============================================================================

if WEEK1_ENABLED:
    # Register weather callbacks
    try:
        create_weather_callbacks(app, weather_loader)
        logger.info("Weather callbacks registered")
    except Exception as e:
        logger.error(f"Failed to register weather callbacks: {e}")

    # Register sector callbacks
    try:
        create_sector_callbacks(app, lap_loader)
        logger.info("Sector callbacks registered")
    except Exception as e:
        logger.error(f"Failed to register sector callbacks: {e}")

    # Register animation callbacks
    try:
        create_animation_callbacks(app)
        logger.info("Animation callbacks registered")
    except Exception as e:
        logger.error(f"Failed to register animation callbacks: {e}")

    # Register post-race analysis callbacks
    try:
        create_post_race_callbacks(app)
        logger.info("Post-race analysis callbacks registered")
    except Exception as e:
        logger.error(f"Failed to register post-race analysis callbacks: {e}")

    # Register help documentation callbacks
    try:
        create_help_callbacks(app)
        logger.info("Help documentation callbacks registered")
    except Exception as e:
        logger.error(f"Failed to register help documentation callbacks: {e}")

# Register chatbot callbacks
print("=" * 80)
print(f"DEBUG: CHATBOT CALLBACK REGISTRATION - CHATBOT_ENABLED = {CHATBOT_ENABLED}")
print("=" * 80)
logger.info("=" * 80)
logger.info(f"DEBUG: CHATBOT CALLBACK REGISTRATION - CHATBOT_ENABLED = {CHATBOT_ENABLED}")
logger.info("=" * 80)
if CHATBOT_ENABLED:
    print("DEBUG: [OK] CHATBOT_ENABLED is True, registering callbacks...")
    logger.info("DEBUG: [OK] CHATBOT_ENABLED is True, registering callbacks...")
    try:
        create_chatbot_callbacks(app)
        print("DEBUG: [OK] CHATBOT CALLBACKS REGISTERED SUCCESSFULLY")
        logger.info("[OK] Chatbot callbacks registered successfully")
    except Exception as e:
        print(f"DEBUG: [X] CHATBOT CALLBACK REGISTRATION FAILED - {e}")
        logger.error(f"Failed to register chatbot callbacks: {e}")
else:
    print("DEBUG: [X] CHATBOT_ENABLED is False, skipping callback registration")
    logger.info("DEBUG: [X] CHATBOT_ENABLED is False, skipping callback registration")
print("=" * 80)

#================================================================
# TOUR SYSTEM CALLBACKS
#================================================================

@app.callback(
    [Output('tour-welcome-modal', 'is_open'),
     Output('tour-state', 'data')],
    [Input('tour-welcome-start', 'n_clicks'),
     Input('tour-welcome-skip', 'n_clicks'),
     Input('upload-data', 'data')],
    [State('tour-state', 'data'),
     State('tour-welcome-dont-show', 'value')],
    prevent_initial_call=False
)
def handle_welcome_modal(start_clicks, skip_clicks, upload_data, tour_state, dont_show):
    """
    Handles welcome modal interactions.

    - Shows modal on page load if data is loaded and modal hasn't been shown
    - Start Tour: Closes modal, starts guided tour (Phase 2)
    - Skip Tour: Closes modal, marks as completed
    - Respects "Don't show again" preference
    """
    ctx = callback_context

    # Check if we should show the modal on page load
    if not ctx.triggered or ctx.triggered[0]['prop_id'] == 'upload-data.data':
        # Show modal if:
        # 1. Data is loaded (upload_data is not None)
        # 2. Welcome hasn't been shown yet
        # 3. User hasn't selected "don't show again"
        if upload_data and not tour_state.get('welcome_shown', False) and not tour_state.get('dont_show_again', False):
            return True, tour_state
        else:
            return False, tour_state

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Start tour
    if trigger_id == 'tour-welcome-start':
        tour_state['welcome_shown'] = True
        tour_state['dont_show_again'] = dont_show if dont_show else False
        # TODO: In Phase 2, this will activate the tour overlay
        logger.info("User started dashboard tour")
        return False, tour_state

    # Skip tour
    if trigger_id == 'tour-welcome-skip':
        tour_state['welcome_shown'] = True
        tour_state['tour_completed'] = True
        tour_state['dont_show_again'] = dont_show if dont_show else False
        logger.info("User skipped dashboard tour")
        return False, tour_state

    return False, tour_state

#================================================================
# THEME TOGGLE AND UI ENHANCEMENTS
#================================================================

@app.callback(
    [Output('main-app-container', 'data-theme'),
     Output('theme-icon', 'className'),
     Output('theme-state', 'data')],
    Input('theme-toggle', 'n_clicks'),
    State('theme-state', 'data'),
    prevent_initial_call=True
)
def toggle_theme(n_clicks, current_theme):
    """Toggle between light and dark theme"""
    if not n_clicks:
        return 'light', 'fas fa-moon', 'light'

    new_theme = 'dark' if current_theme == 'light' else 'light'
    icon_class = 'fas fa-sun' if new_theme == 'dark' else 'fas fa-moon'

    return new_theme, icon_class, new_theme


@app.callback(
    Output('processing-overlay', 'style'),
    Input('tabs', 'active_tab'),
    prevent_initial_call=True
)
def show_processing_indicator(active_tab):
    """Show processing indicator when switching tabs"""
    # Show overlay briefly (CSS handles the display)
    # This will trigger, then the actual tab content loads
    # We'll hide it again when tab content is ready
    return {'display': 'flex'}


@app.callback(
    Output('processing-overlay', 'style', allow_duplicate=True),
    Input('tab-content', 'children'),
    prevent_initial_call=True
)
def hide_processing_indicator(content):
    """Hide processing indicator when tab content is loaded"""
    return {'display': 'none'}


# Sync vehicle dropdown value (display <-> hidden)
@app.callback(
    Output('vehicle-dropdown', 'value'),
    Input('vehicle-dropdown-display', 'value'),
    prevent_initial_call=True
)
def sync_vehicle_dropdown_value(selected_vehicle):
    """Synchronize the displayed vehicle dropdown value with the hidden one"""
    return selected_vehicle


@app.callback(
    Output('vehicle-dropdown-display', 'options'),
    Input('upload-data', 'data'),
    prevent_initial_call=False
)
def update_vehicle_dropdown_display_options(data_json):
    """Update vehicle dropdown options when data is loaded"""
    if data_json and _auto_loaded_stats:
        return _auto_loaded_stats['vehicle_options']
    return []


#================================================================
# RUN APPLICATION
#================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("RACING ANALYTICS DASHBOARD - INTEGRATED")
    print("="*60)
    print(f"\nVersion: {DASHBOARD_VERSION}")
    print(f"Date: {VERSION_DATE}")
    print("\nDashboard running at: http://localhost:8050")
    print("API running at: http://localhost:8000")
    print("\nFEATURES:")
    print("  - Driver Insights & Telemetry Analysis")
    if ENHANCED_INSIGHTS_ENABLED:
        print("    [NEW] Enhanced with Phase 1.2: Ghost Lap, Brake Analysis, Corner Speed")
    print("  - Model Predictions & Track Maps")
    if WEEK1_ENABLED:
        print("  - [OK] Weather Conditions (Week 1)")
        print("  - [OK] Sector Benchmarking (Week 1)")
        print("  - [OK] Championship Tracker (Week 1)")
    else:
        print("  - [WARN] Week 1 features unavailable")
    print("  - [NEW] Track Animation (Quick Win Path)")
    print("  - [NEW] Post-Race Analysis - AI-powered session insights (97.49% R²)")
    print("\nNOTE: API at http://localhost:8000 is optional for Tab 1 (Enhanced Insights works without it)")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=8050)
# trigger reload
