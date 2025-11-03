"""
Sector Time Benchmarking Widget for Racing Telemetry Dashboard

This module provides sector time analysis and benchmarking visualizations.
Displays S1, S2, S3 sector times with percentile rankings and comparison to field.

Usage:
    from src.dashboard.sector_widget import create_sector_layout, create_sector_callbacks

    # In your Dash app:
    app.layout = html.Div([
        create_sector_layout(),
        # ... other components
    ])

    create_sector_callbacks(app, lap_loader)
"""

import plotly.graph_objects as go
import plotly.express as px
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def create_sector_heatmap(driver_times: Dict[str, float], field_percentiles: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Create a heatmap showing driver performance vs field in each sector.

    Args:
        driver_times: Dict with driver's best times (S1_best, S2_best, S3_best)
        field_percentiles: Dict with percentile breakdowns for each sector

    Returns:
        Plotly Figure with heatmap
    """
    sectors = ['S1', 'S2', 'S3']

    # Create color scale data
    driver_percentiles = []
    driver_time_values = []

    for sector in sectors:
        time_key = f'{sector}_best'
        if time_key in driver_times:
            driver_time = driver_times[time_key]
            driver_time_values.append(driver_time)

            # Calculate percentile based on field distribution
            if sector in field_percentiles:
                p_data = field_percentiles[sector]

                # Determine percentile bucket
                if driver_time <= p_data.get('p10', float('inf')):
                    percentile = 95  # Top 10%
                elif driver_time <= p_data.get('p25', float('inf')):
                    percentile = 82.5  # Top 25%
                elif driver_time <= p_data.get('p50', float('inf')):
                    percentile = 62.5  # Top 50%
                elif driver_time <= p_data.get('p75', float('inf')):
                    percentile = 37.5  # Top 75%
                elif driver_time <= p_data.get('p90', float('inf')):
                    percentile = 17.5  # Top 90%
                else:
                    percentile = 5  # Bottom 10%

                driver_percentiles.append(percentile)
            else:
                driver_percentiles.append(50)
        else:
            driver_percentiles.append(0)
            driver_time_values.append(0)

    fig = go.Figure(data=go.Heatmap(
        z=[driver_percentiles],
        x=sectors,
        y=['Performance'],
        text=[[f"{t:.2f}s<br>{p:.0f}%" for t, p in zip(driver_time_values, driver_percentiles)]],
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale=[
            [0.0, '#ef5350'],    # Red (slow)
            [0.25, '#ffa726'],   # Orange
            [0.5, '#ffee58'],    # Yellow
            [0.75, '#66bb6a'],   # Light green
            [1.0, '#26a69a']     # Dark green (fast)
        ],
        colorbar=dict(
            title="Percentile",
            tickvals=[10, 30, 50, 70, 90],
            ticktext=['10%', '30%', '50%', '70%', '90%']
        ),
        hovertemplate='Sector: %{x}<br>Time: %{text}<extra></extra>'
    ))

    fig.update_layout(
        title="Sector Performance Heatmap",
        height=200,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def create_sector_comparison_chart(driver_name: str, driver_times: Dict[str, float],
                                  field_stats: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Create bar chart comparing driver's sector times to field statistics.

    Args:
        driver_name: Name of the driver
        driver_times: Dict with driver's best times
        field_stats: Dict with field statistics for each sector

    Returns:
        Plotly Figure with comparison chart
    """
    sectors = ['S1', 'S2', 'S3']

    # Prepare data
    driver_best = [driver_times.get(f'{s}_best', 0) for s in sectors]
    field_best = [field_stats.get(s, {}).get('min', 0) for s in sectors]
    field_median = [field_stats.get(s, {}).get('median', 0) for s in sectors]
    field_p90 = [field_stats.get(s, {}).get('p90', 0) for s in sectors]

    fig = go.Figure()

    # Field best (fastest in entire field)
    fig.add_trace(go.Bar(
        name='Field Best',
        x=sectors,
        y=field_best,
        marker_color='#26a69a',
        text=[f"{t:.2f}s" for t in field_best],
        textposition='outside'
    ))

    # Driver best
    fig.add_trace(go.Bar(
        name=f'{driver_name} Best',
        x=sectors,
        y=driver_best,
        marker_color='#5c6bc0',
        text=[f"{t:.2f}s" for t in driver_best],
        textposition='outside'
    ))

    # Field median
    fig.add_trace(go.Bar(
        name='Field Median',
        x=sectors,
        y=field_median,
        marker_color='#ffa726',
        text=[f"{t:.2f}s" for t in field_median],
        textposition='outside'
    ))

    fig.update_layout(
        title=f"Sector Time Comparison: {driver_name} vs Field",
        xaxis_title="Sector",
        yaxis_title="Time (seconds)",
        barmode='group',
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_theoretical_lap_widget(theoretical_data: Dict[str, float]) -> html.Div:
    """
    Create widget showing theoretical best lap vs actual best.

    Args:
        theoretical_data: Dict with theoretical lap data

    Returns:
        Dash HTML Div component
    """
    theoretical_time = theoretical_data.get('theoretical_lap_time', 0)
    actual_best = theoretical_data.get('actual_best_lap', 0)
    potential = theoretical_data.get('improvement_potential', 0)

    # Determine color based on potential
    if potential is not None:
        if potential < 0.1:
            badge_color = '#26a69a'  # Green - already very close
            badge_text = "⭐ EXCELLENT"
        elif potential < 0.3:
            badge_color = '#66bb6a'  # Light green - good
            badge_text = "✅ GOOD"
        elif potential < 0.5:
            badge_color = '#ffa726'  # Orange - room for improvement
            badge_text = "📈 IMPROVING"
        else:
            badge_color = '#ef5350'  # Red - significant potential
            badge_text = "🎯 POTENTIAL"
    else:
        badge_color = '#9e9e9e'
        badge_text = "N/A"

    widget_content = [
        html.Div([
            html.H3("🏁 Best Theoretical Lap", style={'marginBottom': '15px', 'color': '#2c3e50'}),
            html.Div(
                badge_text,
                style={
                    'backgroundColor': badge_color,
                    'color': 'white',
                    'padding': '8px 15px',
                    'borderRadius': '5px',
                    'fontSize': '14px',
                    'fontWeight': 'bold',
                    'display': 'inline-block',
                    'marginBottom': '20px'
                }
            ),
        ]),
        html.Hr(),
        html.Div([
            html.Div([
                html.Strong("⚡ Theoretical Best: "),
                html.Span(f"{theoretical_time:.3f}s", style={'fontSize': '20px', 'color': '#26a69a', 'fontWeight': 'bold'})
            ], style={'marginBottom': '15px'}),
            html.Div([
                html.Strong("🏎️ Actual Best: "),
                html.Span(f"{actual_best:.3f}s", style={'fontSize': '20px', 'color': '#5c6bc0', 'fontWeight': 'bold'})
            ], style={'marginBottom': '15px'}),
            html.Div([
                html.Strong("📊 Improvement Potential: "),
                html.Span(
                    f"{potential:.3f}s" if potential is not None else "N/A",
                    style={
                        'fontSize': '20px',
                        'color': badge_color,
                        'fontWeight': 'bold'
                    }
                )
            ], style={'marginBottom': '15px'}),
        ]),
        html.Hr(),
        html.Div([
            html.P("💡 Theoretical lap = Best S1 + Best S2 + Best S3", style={
                'fontSize': '12px',
                'color': '#7f8c8d',
                'fontStyle': 'italic'
            })
        ])
    ]

    return html.Div(
        widget_content,
        style={
            'backgroundColor': '#f8f9fa',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '20px'
        }
    )


def create_percentile_gauges(percentiles: Dict[str, float]) -> html.Div:
    """
    Create gauge displays for sector percentile rankings.

    Args:
        percentiles: Dict with percentile rankings (S1, S2, S3)

    Returns:
        Dash HTML Div with gauges
    """
    gauges = []

    for sector in ['S1', 'S2', 'S3']:
        percentile = percentiles.get(sector, 50)

        # Determine color and label
        if percentile >= 90:
            color = '#26a69a'
            label = "Elite"
        elif percentile >= 75:
            color = '#66bb6a'
            label = "Strong"
        elif percentile >= 50:
            color = '#ffa726'
            label = "Average"
        elif percentile >= 25:
            color = '#ff7043'
            label = "Below Avg"
        else:
            color = '#ef5350'
            label = "Needs Work"

        gauge = html.Div([
            html.H4(sector, style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.Div([
                html.Div(
                    f"{percentile:.1f}%",
                    style={
                        'fontSize': '32px',
                        'fontWeight': 'bold',
                        'color': color,
                        'textAlign': 'center',
                        'marginBottom': '5px'
                    }
                ),
                html.Div(
                    label,
                    style={
                        'fontSize': '14px',
                        'color': '#7f8c8d',
                        'textAlign': 'center'
                    }
                )
            ], style={
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '10px',
                'border': f'3px solid {color}'
            })
        ], className='four columns', style={'marginBottom': '20px'})

        gauges.append(gauge)

    return html.Div(gauges, className='row')


def create_sector_layout() -> html.Div:
    """
    Create the complete sector benchmarking layout for the dashboard.

    Returns:
        Dash HTML Div component with sector analysis
    """
    layout = html.Div([
        # Header
        html.Div([
            html.H2("📊 Sector Time Benchmarking", style={
                'textAlign': 'center',
                'color': '#2c3e50',
                'marginBottom': '10px'
            }),
            html.P("Corner-by-corner performance analysis with percentile rankings", style={
                'textAlign': 'center',
                'color': '#7f8c8d',
                'marginBottom': '30px'
            })
        ]),

        # Driver selector
        html.Div([
            html.Label("Select Driver:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='sector-driver-dropdown',
                options=[],  # Will be populated by callback
                value=None,
                placeholder="Select a driver...",
                style={'marginBottom': '20px'}
            )
        ]),

        # Theoretical lap widget
        html.Div(id='theoretical-lap-widget', children=[
            html.P("Select a driver to view sector analysis", style={
                'textAlign': 'center',
                'color': '#7f8c8d',
                'padding': '40px'
            })
        ]),

        # Percentile rankings
        html.Div([
            html.H3("🏆 Percentile Rankings", style={'marginBottom': '20px', 'color': '#2c3e50'}),
            html.Div(id='percentile-gauges')
        ], style={'marginBottom': '30px'}),

        # Sector heatmap
        html.Div([
            dcc.Graph(id='sector-heatmap')
        ], style={'marginBottom': '20px'}),

        # Comparison chart
        html.Div([
            dcc.Graph(id='sector-comparison-chart')
        ], style={'marginBottom': '20px'}),

        # Store for lap data
        dcc.Store(id='sector-data-store'),

    ], style={'padding': '20px'})

    return layout


def create_sector_callbacks(app, lap_loader):
    """
    Create Dash callbacks for sector widget interactivity.

    Args:
        app: Dash app instance
        lap_loader: Instance of LapAnalysisLoader
    """

    @app.callback(
        Output('sector-driver-dropdown', 'options'),
        Output('sector-driver-dropdown', 'value'),
        Output('sector-data-store', 'data'),
        Input('sector-driver-dropdown', 'id')  # Trigger on page load
    )
    def load_sector_data(_):
        """Load all lap data and populate driver dropdown."""
        try:
            # Load all lap data
            all_laps = lap_loader.load_all_lap_data()

            if all_laps.empty:
                return [], None, None

            # Get drivers
            drivers = lap_loader.get_available_drivers(all_laps)
            options = [{'label': d, 'value': d} for d in drivers]

            # Default to first driver
            default_driver = drivers[0] if drivers else None

            # Store lap data (convert to dict for JSON serialization)
            # Only store necessary columns to reduce size
            cols_to_keep = ['DRIVER_NAME', 'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS',
                          'LAP_TIME_SECONDS', 'CLASS', 'TEAM']
            stored_data = all_laps[cols_to_keep].to_dict('records')

            return options, default_driver, stored_data

        except Exception as e:
            logger.error(f"Error loading sector data: {e}")
            return [], None, None

    @app.callback(
        Output('theoretical-lap-widget', 'children'),
        Output('percentile-gauges', 'children'),
        Output('sector-heatmap', 'figure'),
        Output('sector-comparison-chart', 'figure'),
        Input('sector-driver-dropdown', 'value'),
        State('sector-data-store', 'data')
    )
    def update_sector_displays(driver_name, lap_data):
        """Update all sector displays when driver changes."""
        if not driver_name or not lap_data:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No driver selected",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            empty_fig.update_layout(height=300)

            empty_widget = html.P("Select a driver to view sector analysis", style={
                'textAlign': 'center',
                'color': '#7f8c8d',
                'padding': '40px'
            })

            empty_gauges = html.Div()

            return empty_widget, empty_gauges, empty_fig, empty_fig

        try:
            # Convert back to DataFrame
            all_laps_df = pd.DataFrame(lap_data)

            # Get driver performance
            driver_perf = lap_loader.get_driver_sector_performance(all_laps_df, driver_name)

            # Get field statistics
            field_stats = lap_loader.get_sector_statistics(all_laps_df)

            # Get theoretical lap
            theoretical = lap_loader.get_best_theoretical_lap(all_laps_df, driver_name)

            # Calculate percentile ranks
            percentiles = {}
            for sector in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
                rank = lap_loader.calculate_percentile_rank(all_laps_df, driver_name, sector)
                if rank is not None:
                    percentiles[sector.replace('_SECONDS', '')] = rank

            # Create widgets
            theoretical_widget = create_theoretical_lap_widget(theoretical)
            percentile_gauge_display = create_percentile_gauges(percentiles)
            heatmap = create_sector_heatmap(driver_perf, field_stats)
            comparison = create_sector_comparison_chart(driver_name, driver_perf, field_stats)

            return theoretical_widget, percentile_gauge_display, heatmap, comparison

        except Exception as e:
            logger.error(f"Error updating sector displays: {e}")
            empty_fig = go.Figure()
            empty_widget = html.P(f"Error: {str(e)}", style={'color': 'red'})
            return empty_widget, html.Div(), empty_fig, empty_fig


if __name__ == "__main__":
    print("This module is meant to be imported, not run directly.")
    print("Import it in your Dash app with:")
    print("  from src.dashboard.sector_widget import create_sector_layout, create_sector_callbacks")
