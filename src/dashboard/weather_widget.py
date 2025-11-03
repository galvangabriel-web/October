"""
Weather Widget for Racing Telemetry Dashboard

This module provides weather visualization components for the Dash dashboard.
Displays air temperature, track temperature, humidity, wind, and rain conditions.

Usage:
    from src.dashboard.weather_widget import create_weather_layout, create_weather_callbacks

    # In your Dash app:
    app.layout = html.Div([
        create_weather_layout(),
        # ... other components
    ])

    create_weather_callbacks(app)
"""

import plotly.graph_objects as go
import plotly.express as px
from dash import html, dcc, Input, Output, State
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def create_weather_gauge(value: float, title: str, min_val: float, max_val: float,
                        unit: str = "", color_scale: str = "RdYlGn_r") -> go.Figure:
    """
    Create a gauge chart for a weather metric.

    Args:
        value: Current value
        title: Chart title
        min_val: Minimum value for gauge
        max_val: Maximum value for gauge
        unit: Unit of measurement
        color_scale: Plotly color scale name

    Returns:
        Plotly Figure with gauge chart
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{title} ({unit})", 'font': {'size': 16}},
        number={'suffix': f" {unit}"},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_val, (max_val - min_val) * 0.33 + min_val], 'color': 'lightgreen'},
                {'range': [(max_val - min_val) * 0.33 + min_val, (max_val - min_val) * 0.66 + min_val], 'color': 'yellow'},
                {'range': [(max_val - min_val) * 0.66 + min_val, max_val], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )

    return fig


def create_weather_timeline(weather_df: pd.DataFrame, metric: str, title: str, unit: str) -> go.Figure:
    """
    Create a timeline chart for a weather metric.

    Args:
        weather_df: DataFrame with weather data
        metric: Column name for the metric
        title: Chart title
        unit: Unit of measurement

    Returns:
        Plotly Figure with timeline chart
    """
    if weather_df.empty or metric not in weather_df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
        return fig

    fig = go.Figure()

    # Plot metric over time
    fig.add_trace(go.Scatter(
        x=weather_df['timestamp'] if 'timestamp' in weather_df.columns else weather_df.index,
        y=weather_df[metric],
        mode='lines+markers',
        name=title,
        line=dict(color='rgb(31, 119, 180)', width=2),
        marker=dict(size=6),
        hovertemplate=f'<b>{title}</b><br>Time: %{{x}}<br>Value: %{{y:.2f}} {unit}<extra></extra>'
    ))

    # Add average line
    avg_value = weather_df[metric].mean()
    fig.add_hline(
        y=avg_value,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg: {avg_value:.2f} {unit}",
        annotation_position="right"
    )

    fig.update_layout(
        title=f"{title} Over Time",
        xaxis_title="Time",
        yaxis_title=f"{title} ({unit})",
        height=300,
        margin=dict(l=60, r=20, t=50, b=60),
        hovermode='x unified',
        showlegend=False
    )

    return fig


def create_weather_summary_card(summary: Dict[str, float], session_name: str) -> html.Div:
    """
    Create a summary card showing weather statistics.

    Args:
        summary: Dictionary with weather statistics
        session_name: Name of the session

    Returns:
        Dash HTML Div component
    """
    rain_badge = html.Span(
        "🌧️ RAIN",
        style={
            'backgroundColor': '#ff6b6b',
            'color': 'white',
            'padding': '5px 10px',
            'borderRadius': '5px',
            'fontSize': '12px',
            'fontWeight': 'bold'
        }
    ) if summary.get('rain_detected', False) else html.Span(
        "☀️ DRY",
        style={
            'backgroundColor': '#51cf66',
            'color': 'white',
            'padding': '5px 10px',
            'borderRadius': '5px',
            'fontSize': '12px',
            'fontWeight': 'bold'
        }
    )

    card_content = [
        html.H4(f"📊 {session_name}", style={'marginBottom': '15px', 'color': '#2c3e50'}),
        html.Div(rain_badge, style={'marginBottom': '15px'}),
        html.Hr(),
        html.Div([
            html.Div([
                html.Strong("🌡️ Air Temp: "),
                html.Span(f"{summary.get('avg_air_temp', 0):.1f}°C")
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Strong("🏁 Track Temp: "),
                html.Span(f"{summary.get('avg_track_temp', 0):.1f}°C")
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Strong("💧 Humidity: "),
                html.Span(f"{summary.get('avg_humidity', 0):.1f}%")
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Strong("💨 Wind Speed: "),
                html.Span(f"{summary.get('avg_wind_speed', 0):.1f} km/h")
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Strong("🔢 Readings: "),
                html.Span(f"{summary.get('num_readings', 0)}")
            ], style={'marginBottom': '10px'}),
        ])
    ]

    return html.Div(
        card_content,
        style={
            'backgroundColor': '#f8f9fa',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '20px'
        }
    )


def create_weather_layout() -> html.Div:
    """
    Create the complete weather widget layout for the dashboard.

    Returns:
        Dash HTML Div component with weather visualization
    """
    layout = html.Div([
        # Header
        html.Div([
            html.H2("🌤️ Weather Conditions", style={
                'textAlign': 'center',
                'color': '#2c3e50',
                'marginBottom': '10px'
            }),
            html.P("Real-time weather data from Alkamel timing system", style={
                'textAlign': 'center',
                'color': '#7f8c8d',
                'marginBottom': '30px'
            })
        ]),

        # Session selector
        html.Div([
            html.Label("Select Session:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='weather-session-dropdown',
                options=[],  # Will be populated by callback
                value=None,
                placeholder="Select a session...",
                style={'marginBottom': '20px'}
            )
        ]),

        # Weather summary card
        html.Div(id='weather-summary-card', children=[
            html.P("Select a session to view weather data", style={
                'textAlign': 'center',
                'color': '#7f8c8d',
                'padding': '40px'
            })
        ]),

        # Weather gauges - Row 1: Air Temp and Track Temp
        html.Div([
            html.Div([
                dcc.Graph(id='weather-gauge-air-temp')
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='weather-gauge-track-temp')
            ], className='six columns'),
        ], className='row', style={'marginBottom': '20px'}),

        # Weather gauges - Row 2: Humidity and Wind Speed
        html.Div([
            html.Div([
                dcc.Graph(id='weather-gauge-humidity')
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='weather-gauge-wind')
            ], className='six columns'),
        ], className='row', style={'marginBottom': '20px'}),

        # Timeline charts
        html.Div([
            dcc.Graph(id='weather-timeline-air-temp')
        ], style={'marginBottom': '20px'}),

        html.Div([
            dcc.Graph(id='weather-timeline-track-temp')
        ], style={'marginBottom': '20px'}),

        html.Div([
            dcc.Graph(id='weather-timeline-humidity')
        ], style={'marginBottom': '20px'}),

        # Store for weather data
        dcc.Store(id='weather-data-store'),

    ], style={'padding': '20px'})

    return layout


def create_weather_callbacks(app, weather_loader):
    """
    Create Dash callbacks for weather widget interactivity.

    Args:
        app: Dash app instance
        weather_loader: Instance of WeatherDataLoader
    """

    @app.callback(
        Output('weather-session-dropdown', 'options'),
        Output('weather-session-dropdown', 'value'),
        Input('weather-session-dropdown', 'id')  # Trigger on page load
    )
    def populate_session_dropdown(_):
        """Populate session dropdown with available sessions."""
        try:
            sessions = weather_loader.get_available_sessions()
            options = [{'label': s.replace('%20', ' '), 'value': s} for s in sessions]

            # Default to first session if available
            default_value = sessions[0] if sessions else None

            return options, default_value
        except Exception as e:
            logger.error(f"Error populating session dropdown: {e}")
            return [], None

    @app.callback(
        Output('weather-data-store', 'data'),
        Input('weather-session-dropdown', 'value')
    )
    def load_weather_data(session):
        """Load weather data for selected session."""
        if not session:
            return None

        try:
            weather_df = weather_loader.load_weather_by_session(session)
            if weather_df.empty:
                return None

            # Convert to dict for storage
            return weather_df.to_dict('records')
        except Exception as e:
            logger.error(f"Error loading weather data: {e}")
            return None

    @app.callback(
        Output('weather-summary-card', 'children'),
        Output('weather-gauge-air-temp', 'figure'),
        Output('weather-gauge-track-temp', 'figure'),
        Output('weather-gauge-humidity', 'figure'),
        Output('weather-gauge-wind', 'figure'),
        Output('weather-timeline-air-temp', 'figure'),
        Output('weather-timeline-track-temp', 'figure'),
        Output('weather-timeline-humidity', 'figure'),
        Input('weather-data-store', 'data'),
        State('weather-session-dropdown', 'value')
    )
    def update_weather_displays(weather_data, session):
        """Update all weather displays when data changes."""
        if not weather_data or not session:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No data selected",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            empty_fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))

            empty_card = html.P("Select a session to view weather data", style={
                'textAlign': 'center',
                'color': '#7f8c8d',
                'padding': '40px'
            })

            return empty_card, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

        try:
            # Convert back to DataFrame
            weather_df = pd.DataFrame(weather_data)

            # Convert timestamp strings back to datetime
            if 'timestamp' in weather_df.columns:
                weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])

            # Get summary
            summary = weather_loader.get_session_summary(weather_df)
            session_name = session.replace('%20', ' ')

            # Create summary card
            summary_card = create_weather_summary_card(summary, session_name)

            # Create gauges
            air_temp_gauge = create_weather_gauge(
                summary.get('avg_air_temp', 20),
                "Air Temperature",
                10, 40, "°C"
            )

            track_temp_gauge = create_weather_gauge(
                summary.get('avg_track_temp', 30),
                "Track Temperature",
                15, 60, "°C"
            )

            humidity_gauge = create_weather_gauge(
                summary.get('avg_humidity', 50),
                "Humidity",
                0, 100, "%"
            )

            wind_gauge = create_weather_gauge(
                summary.get('avg_wind_speed', 5),
                "Wind Speed",
                0, 30, "km/h"
            )

            # Create timelines
            air_temp_timeline = create_weather_timeline(
                weather_df, 'AIR_TEMP', 'Air Temperature', '°C'
            )

            track_temp_timeline = create_weather_timeline(
                weather_df, 'TRACK_TEMP', 'Track Temperature', '°C'
            )

            humidity_timeline = create_weather_timeline(
                weather_df, 'HUMIDITY', 'Humidity', '%'
            )

            return (summary_card, air_temp_gauge, track_temp_gauge, humidity_gauge,
                   wind_gauge, air_temp_timeline, track_temp_timeline, humidity_timeline)

        except Exception as e:
            logger.error(f"Error updating weather displays: {e}")
            empty_fig = go.Figure()
            empty_card = html.P(f"Error loading weather data: {str(e)}", style={
                'textAlign': 'center',
                'color': 'red',
                'padding': '40px'
            })
            return empty_card, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig


if __name__ == "__main__":
    print("This module is meant to be imported, not run directly.")
    print("Import it in your Dash app with:")
    print("  from src.dashboard.weather_widget import create_weather_layout, create_weather_callbacks")
