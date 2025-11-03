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
        # Header
        dbc.Row([
            dbc.Col([
                html.H3("📊 Post-Race Analysis", className="mb-2"),
                html.P(
                    "Comprehensive session review with AI-powered insights and coaching recommendations",
                    className="text-muted"
                ),
                html.Hr()
            ])
        ], className="mb-3"),

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

                # Section 2: Performance Analysis
                dbc.Row([
                    # Error distribution
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("📊 Error Distribution", className="mb-0")),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='post-race-error-histogram',
                                    config={'displayModeBar': True, 'displaylogo': False}
                                )
                            ])
                        ])
                    ], width=6),

                    # Session statistics
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("📋 Session Statistics", className="mb-0")),
                            dbc.CardBody([
                                html.Div(id='post-race-statistics-table')
                            ])
                        ])
                    ], width=6)
                ], className="mb-4", id='post-race-analysis-row', style={'display': 'none'}),

                # Section 3: Anomaly Details
                dbc.Card([
                    dbc.CardHeader(html.H5("🔍 Anomaly Details (Problem Laps)", className="mb-0")),
                    dbc.CardBody([
                        html.Div(id='post-race-anomalies-table')
                    ])
                ], className="mb-4", id='post-race-anomalies-card', style={'display': 'none'}),

                # Section 4: Recommendations
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


def create_post_race_callbacks(app):
    """
    Register callbacks for Post-Race Analysis tab

    Args:
        app: Dash app instance
    """

    # Initialize predictor and analyzer
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
         Output('post-race-analysis-row', 'style'),
         Output('post-race-anomalies-card', 'style'),
         Output('post-race-recommendations-card', 'style'),
         Output('post-race-export-card', 'style'),
         Output('post-race-download-csv-btn', 'disabled'),
         Output('post-race-download-summary-btn', 'disabled')],
        [Input('post-race-analyze-btn', 'n_clicks')],
        [State('post-race-upload', 'contents'),
         State('post-race-upload', 'filename'),
         State('post-race-track-dropdown', 'value'),
         State('post-race-race-dropdown', 'value'),
         State('post-race-drivers-dropdown', 'value')],
        prevent_initial_call=True
    )
    def update_post_race_analysis(n_clicks, upload_contents, filename,
                                  track, race, driver_ids):
        """
        Main callback: Process data and generate visualizations

        Returns:
            Tuple of 16 outputs for all dashboard components
        """
        try:
            # Step 1: Load data
            if upload_contents:
                telemetry_df, lap_times_df = parse_upload(upload_contents, filename)
                source = f"uploaded file: {filename}"
            elif track and race:
                telemetry_df, lap_times_df = load_session_data(track, race)
                source = f"{track} - {race}"
            else:
                return empty_outputs_with_message(
                    "❌ Please upload a file or select a track/race",
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
                show_style, show_style, show_style, show_style, show_style,
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
            height=400,
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
        """Create anomalies table"""
        if len(anomalies) == 0:
            return html.Div([
                html.P("✅ No anomalies detected!", className="text-success fw-bold mb-2"),
                html.P("All laps within expected range (error < 2.5s)", className="text-muted")
            ])

        # Create formatted table
        table_data = anomalies.head(10).to_dict('records')

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
            hide_style, hide_style, hide_style, hide_style, hide_style,  # Hide cards
            True, True  # Disable buttons
        )

    # Callback to populate race dropdown based on track selection
    @app.callback(
        Output('post-race-race-dropdown', 'options'),
        Input('post-race-track-dropdown', 'value')
    )
    def update_race_dropdown(track):
        """Populate race options based on selected track - ONLY VALID RACES WITH TELEMETRY"""
        if not track:
            return []

        from data_loader import RacingDataLoader
        from pathlib import Path

        loader = RacingDataLoader()

        try:
            races = loader.list_races(track)

            # Filter to only races with telemetry data
            valid_races = []
            for race in races:
                try:
                    categories = loader.list_categories(track, race)

                    # Check if telemetry exists
                    if 'telemetry' in categories:
                        # Verify telemetry files exist
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
                # Clean up display name
                if race.endswith('.csv') or len(race) > 30:
                    # Skip CSV filenames or very long names
                    continue

                label = race.replace('_', ' ').replace('-', ' ').title()
                options.append({'label': label, 'value': race})

            return sorted(options, key=lambda x: x['value'])

        except Exception as e:
            print(f"Error loading races for {track}: {e}")
            return []

    # Callback to populate driver dropdown based on track/race selection
    @app.callback(
        Output('post-race-drivers-dropdown', 'options'),
        [Input('post-race-track-dropdown', 'value'),
         Input('post-race-race-dropdown', 'value')]
    )
    def update_drivers_dropdown(track, race):
        """Populate driver options based on selected track and race"""
        if not track or not race:
            return []

        from data_loader import RacingDataLoader
        from pathlib import Path

        loader = RacingDataLoader()

        try:
            # Sample first 5 chunks to get vehicle list quickly
            # Most vehicles appear in early chunks
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
            print(f"Error loading drivers for {track}/{race}: {e}")
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

    # Extract lap times from telemetry
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

    return telemetry_df, lap_times_df
