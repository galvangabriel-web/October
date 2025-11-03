"""
Track Animation Dashboard Widget
=================================

Tab 8 widget for the unified dashboard providing interactive track animations
with telemetry data overlays.

Usage:
    from src.dashboard.animation_widget import create_animation_layout, create_animation_callbacks

    app = dash.Dash(__name__)
    app.layout = create_animation_layout()
    create_animation_callbacks(app)
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import io
import base64
from pathlib import Path

from src.track_data.track_animator import TrackAnimator
from src.track_data.track_metadata import get_available_tracks


def create_animation_layout():
    """Create the animation tab layout"""

    # Get available tracks that have animation paths
    all_tracks = get_available_tracks()
    animation_dir = Path("track_maps/animations")

    available_tracks = []
    for track in all_tracks:
        if (animation_dir / f"{track}_path.json").exists():
            available_tracks.append(track)

    # Track display names
    track_names = {
        'barber-motorsports-park': 'Barber Motorsports Park',
        'circuit-of-the-americas': 'Circuit of the Americas',
        'road-america': 'Road America',
        'sebring': 'Sebring International Raceway',
        'sonoma': 'Sonoma Raceway',
        'virginia-international-raceway': 'Virginia International Raceway'
    }

    track_options = [
        {'label': track_names.get(track, track.replace('-', ' ').title()), 'value': track}
        for track in available_tracks
    ]

    if not track_options:
        # No tracks available yet
        return html.Div([
            dbc.Alert(
                [
                    html.H4("Track Animation Not Available", className="alert-heading"),
                    html.P([
                        "No animation paths have been generated yet. ",
                        "To create an animation path for a track, run:"
                    ]),
                    html.Pre(
                        "python scripts/vectorization/extract_track_path_from_telemetry.py --track TRACK_NAME",
                        style={'background': '#f8f9fa', 'padding': '10px', 'border-radius': '5px'}
                    ),
                    html.Hr(),
                    html.P([
                        "See ",
                        html.Code("QUICK_WIN_ANIMATION_COMPLETE.md"),
                        " for full instructions."
                    ], className="mb-0")
                ],
                color="info"
            )
        ], style={'padding': '20px'})

    layout = html.Div([
        # Header
        html.Div([
            html.H2("🏁 Track Animation", style={'margin-bottom': '10px'}),
            html.P(
                "Upload telemetry data to see animated car movement with speed overlays.",
                style={'color': '#666', 'margin-bottom': '20px'}
            )
        ]),

        # Controls Row
        dbc.Row([
            # Track Selector
            dbc.Col([
                html.Label("Track:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                dcc.Dropdown(
                    id='animation-track-selector',
                    options=track_options,
                    value=track_options[0]['value'] if track_options else None,
                    clearable=False,
                    style={'width': '100%'}
                )
            ], md=3),

            # Vehicle Selector
            dbc.Col([
                html.Label("Vehicle:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                dcc.Dropdown(
                    id='animation-vehicle-selector',
                    options=[],
                    placeholder="Upload telemetry first",
                    clearable=False,
                    disabled=True,
                    style={'width': '100%'}
                )
            ], md=2),

            # Overlay Type
            dbc.Col([
                html.Label("Overlay:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                dcc.Dropdown(
                    id='animation-overlay-selector',
                    options=[
                        {'label': 'Speed Heatmap', 'value': 'speed'},
                        {'label': 'None', 'value': 'none'}
                    ],
                    value='speed',
                    clearable=False,
                    style={'width': '100%'}
                )
            ], md=2),

            # Animation Type
            dbc.Col([
                html.Label("Type:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                dcc.Dropdown(
                    id='animation-type-selector',
                    options=[
                        {'label': 'Static Overlay', 'value': 'static'},
                        {'label': 'Animated', 'value': 'animated'}
                    ],
                    value='static',
                    clearable=False,
                    style={'width': '100%'}
                )
            ], md=2),

            # Generate Button
            dbc.Col([
                html.Label('\u00A0', style={'display': 'block', 'margin-bottom': '5px'}),
                dbc.Button(
                    "Generate Animation",
                    id='animation-generate-button',
                    color="primary",
                    disabled=True,
                    style={'width': '100%'}
                )
            ], md=3)
        ], style={'margin-bottom': '20px'}),

        # File Upload
        html.Div([
            dcc.Upload(
                id='animation-upload-telemetry',
                children=html.Div([
                    html.I(className='fas fa-upload', style={'margin-right': '10px', 'font-size': '20px'}),
                    'Drag and Drop or ',
                    html.A('Select Telemetry CSV File', style={'text-decoration': 'underline', 'cursor': 'pointer'})
                ]),
                style={
                    'width': '100%',
                    'height': '80px',
                    'lineHeight': '80px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'background': '#f8f9fa',
                    'cursor': 'pointer',
                    'transition': 'all 0.3s ease'
                },
                multiple=False
            ),
            html.Div(id='animation-upload-status', style={'margin-top': '10px'})
        ], style={'margin-bottom': '20px'}),

        # Loading Indicator
        dcc.Loading(
            id="animation-loading",
            type="default",
            children=[
                # Animation Display
                html.Div(
                    id='animation-display',
                    style={'margin-top': '20px'}
                )
            ]
        ),

        # Hidden storage for telemetry data
        dcc.Store(id='animation-telemetry-store'),

        # Instructions
        html.Div([
            dbc.Collapse(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("📖 Instructions", className="card-title"),
                        html.Ol([
                            html.Li("Select a track from the dropdown"),
                            html.Li("Upload telemetry CSV file (must be from organized_data/)"),
                            html.Li("Select a vehicle from the dropdown (populated after upload)"),
                            html.Li("Choose overlay type (Speed Heatmap shows red=fast, blue=slow)"),
                            html.Li("Choose animation type (Static or Animated with playback)"),
                            html.Li("Click 'Generate Animation' to create visualization")
                        ]),
                        html.Hr(),
                        html.P([
                            html.Strong("Tips:"),
                            html.Ul([
                                html.Li("Static overlays load faster and show the full lap path with colors"),
                                html.Li("Animated visualizations have playback controls (Play/Pause)"),
                                html.Li("Speed heatmap: Red = high speed, Blue = low speed"),
                                html.Li("The car icon in animated mode shows real-time position")
                            ])
                        ])
                    ])
                ),
                id="animation-instructions-collapse",
                is_open=False
            ),
            html.Div([
                dbc.Button(
                    "Show Instructions",
                    id="animation-instructions-button",
                    color="link",
                    size="sm",
                    style={'padding': '5px 10px', 'margin-top': '10px'}
                )
            ])
        ])

    ], style={'padding': '20px'})

    return layout


def create_animation_callbacks(app):
    """Create callbacks for animation tab"""

    # Callback: Upload telemetry and extract vehicle list
    @app.callback(
        [Output('animation-telemetry-store', 'data'),
         Output('animation-upload-status', 'children'),
         Output('animation-vehicle-selector', 'options'),
         Output('animation-vehicle-selector', 'disabled'),
         Output('animation-generate-button', 'disabled')],
        Input('animation-upload-telemetry', 'contents'),
        State('animation-upload-telemetry', 'filename')
    )
    def upload_telemetry(contents, filename):
        if contents is None:
            return None, None, [], True, True

        try:
            # Parse uploaded file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            # Read CSV
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            # Validate telemetry structure
            required_cols = ['vehicle_number', 'timestamp']
            if not all(col in df.columns for col in required_cols):
                return None, dbc.Alert(
                    f"Invalid telemetry file. Required columns: {required_cols}",
                    color="danger"
                ), [], True, True

            # Get unique vehicles
            if 'vehicle_number' in df.columns:
                vehicles = sorted(df['vehicle_number'].unique())
                vehicle_options = [
                    {'label': f'Vehicle {v}', 'value': int(v)}
                    for v in vehicles
                ]
            else:
                vehicle_options = []

            # Store telemetry (sample for memory efficiency)
            # Store up to 100k rows
            if len(df) > 100000:
                df_sample = df.sample(n=100000, random_state=42)
            else:
                df_sample = df

            # Convert to JSON for storage
            telemetry_json = df_sample.to_json(orient='split')

            status = dbc.Alert([
                html.I(className='fas fa-check-circle', style={'margin-right': '10px'}),
                f"Loaded {filename}: {len(df):,} records, {len(vehicles)} vehicle(s)"
            ], color="success")

            return telemetry_json, status, vehicle_options, False, False

        except Exception as e:
            return None, dbc.Alert(
                f"Error loading file: {str(e)}",
                color="danger"
            ), [], True, True

    # Callback: Generate animation
    @app.callback(
        Output('animation-display', 'children'),
        Input('animation-generate-button', 'n_clicks'),
        [State('animation-track-selector', 'value'),
         State('animation-vehicle-selector', 'value'),
         State('animation-overlay-selector', 'value'),
         State('animation-type-selector', 'value'),
         State('animation-telemetry-store', 'data')]
    )
    def generate_animation(n_clicks, track, vehicle, overlay, anim_type, telemetry_json):
        if n_clicks is None or telemetry_json is None or vehicle is None:
            raise PreventUpdate

        try:
            # Load telemetry from storage
            df = pd.read_json(io.StringIO(telemetry_json), orient='split')

            # Initialize animator
            animator = TrackAnimator(track)

            # Generate visualization
            if anim_type == 'static':
                fig = animator.create_static_overlay(
                    df,
                    vehicle_number=vehicle,
                    overlay=overlay
                )
            else:  # animated
                fig = animator.create_plotly_animation(
                    df,
                    vehicle_number=vehicle,
                    fps=30,
                    overlay=overlay
                )

            # Return Plotly graph
            return dcc.Graph(
                figure=fig,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
                },
                style={'height': '800px'}
            )

        except Exception as e:
            return dbc.Alert(
                [
                    html.H5("Animation Generation Failed", className="alert-heading"),
                    html.P(str(e)),
                    html.Hr(),
                    html.P([
                        "Common issues:",
                        html.Ul([
                            html.Li("Track animation path not found - run path extraction script"),
                            html.Li("Telemetry format mismatch - ensure CSV is from organized_data/"),
                            html.Li("No GPS data for selected vehicle/lap")
                        ])
                    ])
                ],
                color="danger"
            )

    # Callback: Toggle instructions
    @app.callback(
        [Output("animation-instructions-collapse", "is_open"),
         Output("animation-instructions-button", "children")],
        Input("animation-instructions-button", "n_clicks"),
        State("animation-instructions-collapse", "is_open")
    )
    def toggle_instructions(n_clicks, is_open):
        if n_clicks:
            if is_open:
                return False, "Show Instructions"
            else:
                return True, "Hide Instructions"
        return False, "Show Instructions"
