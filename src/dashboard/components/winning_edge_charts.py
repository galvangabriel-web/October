"""
Advanced chart generation for Winning Edge Widget.

This module provides specialized Plotly visualizations for race performance analysis:
- Heatmaps with custom annotations for time loss patterns
- 3D surface plots for multi-dimensional optimization
- Animated transitions showing performance improvements
- Custom hover templates with detailed metrics
- Export functionality for all chart types

Author: GR Cup Racing Analytics Platform
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime


def create_time_loss_heatmap(
    corner_data: pd.DataFrame,
    metric: str = "time_loss",
    title: str = "Time Loss by Corner and Lap",
    color_scale: str = "RdYlGn_r",
    show_annotations: bool = True
) -> go.Figure:
    """
    Create advanced heatmap showing time loss patterns across corners and laps.

    Args:
        corner_data: DataFrame with columns [corner, lap, time_loss, brake_delta, exit_delta]
        metric: Metric to visualize ('time_loss', 'brake_delta', 'exit_delta')
        title: Chart title
        color_scale: Plotly color scale name
        show_annotations: Whether to show cell annotations

    Returns:
        Plotly Figure object

    Raises:
        ValueError: If corner_data is empty or missing required columns
    """
    if corner_data.empty:
        raise ValueError("corner_data cannot be empty")

    required_cols = ['corner', 'lap', metric]
    missing_cols = [col for col in required_cols if col not in corner_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Pivot data for heatmap
    heatmap_data = corner_data.pivot(
        index='corner',
        columns='lap',
        values=metric
    )

    # Create custom hover text
    hover_text = []
    for i, corner in enumerate(heatmap_data.index):
        hover_row = []
        for j, lap in enumerate(heatmap_data.columns):
            value = heatmap_data.iloc[i, j]
            if pd.notna(value):
                hover_row.append(
                    f"Corner: {corner}<br>"
                    f"Lap: {lap}<br>"
                    f"{metric.replace('_', ' ').title()}: {value:.3f}s<br>"
                    f"Click for details"
                )
            else:
                hover_row.append("")
        hover_text.append(hover_row)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=color_scale,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(
            title=dict(
                text=metric.replace('_', ' ').title() + " (s)",
                side="right"
            ),
            tickmode="linear",
            tick0=0,
            dtick=0.1
        )
    ))

    # Add annotations if requested
    if show_annotations:
        annotations = []
        for i, corner in enumerate(heatmap_data.index):
            for j, lap in enumerate(heatmap_data.columns):
                value = heatmap_data.iloc[i, j]
                if pd.notna(value):
                    annotations.append(
                        dict(
                            x=lap,
                            y=corner,
                            text=f"{value:.2f}",
                            showarrow=False,
                            font=dict(
                                size=10,
                                color='white' if abs(value) > 0.2 else 'black'
                            )
                        )
                    )
        fig.update_layout(annotations=annotations)

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Lap Number",
        yaxis_title="Corner",
        height=max(400, len(heatmap_data.index) * 30),
        xaxis=dict(side='bottom'),
        yaxis=dict(side='left'),
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white'
    )

    return fig


def create_3d_performance_surface(
    corner_data: pd.DataFrame,
    x_metric: str = "brake_pressure",
    y_metric: str = "entry_speed",
    z_metric: str = "exit_speed",
    title: str = "Performance Optimization Surface"
) -> go.Figure:
    """
    Create 3D surface plot showing relationship between multiple performance metrics.

    Args:
        corner_data: DataFrame with performance metrics
        x_metric: Metric for X axis
        y_metric: Metric for Y axis
        z_metric: Metric for Z axis (response variable)
        title: Chart title

    Returns:
        Plotly Figure object with 3D surface

    Raises:
        ValueError: If required metrics not in corner_data
    """
    required_metrics = [x_metric, y_metric, z_metric]
    missing_metrics = [m for m in required_metrics if m not in corner_data.columns]
    if missing_metrics:
        raise ValueError(f"Missing required metrics: {missing_metrics}")

    if len(corner_data) < 3:
        raise ValueError("Need at least 3 data points for surface plot")

    # Create mesh grid for surface
    x_range = np.linspace(
        corner_data[x_metric].min(),
        corner_data[x_metric].max(),
        20
    )
    y_range = np.linspace(
        corner_data[y_metric].min(),
        corner_data[y_metric].max(),
        20
    )

    X, Y = np.meshgrid(x_range, y_range)

    # Interpolate Z values
    from scipy.interpolate import griddata

    points = corner_data[[x_metric, y_metric]].values
    values = corner_data[z_metric].values

    Z = griddata(points, values, (X, Y), method='cubic', fill_value=np.nan)

    # Create 3D surface
    fig = go.Figure(data=[
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(
                    text=z_metric.replace('_', ' ').title(),
                    side='right'
                )
            ),
            hovertemplate=(
                f"{x_metric.replace('_', ' ').title()}: %{{x:.2f}}<br>"
                f"{y_metric.replace('_', ' ').title()}: %{{y:.2f}}<br>"
                f"{z_metric.replace('_', ' ').title()}: %{{z:.2f}}<br>"
                "<extra></extra>"
            )
        )
    ])

    # Add scatter points for actual data
    fig.add_trace(
        go.Scatter3d(
            x=corner_data[x_metric],
            y=corner_data[y_metric],
            z=corner_data[z_metric],
            mode='markers',
            marker=dict(
                size=5,
                color='red',
                symbol='circle',
                line=dict(color='white', width=1)
            ),
            name='Actual Data',
            hovertemplate=(
                f"{x_metric.replace('_', ' ').title()}: %{{x:.2f}}<br>"
                f"{y_metric.replace('_', ' ').title()}: %{{y:.2f}}<br>"
                f"{z_metric.replace('_', ' ').title()}: %{{z:.2f}}<br>"
                "<extra></extra>"
            )
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis_title=x_metric.replace('_', ' ').title(),
            yaxis_title=y_metric.replace('_', ' ').title(),
            zaxis_title=z_metric.replace('_', ' ').title(),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=600,
        showlegend=True
    )

    return fig


def create_animated_improvement_chart(
    current_data: pd.DataFrame,
    target_data: pd.DataFrame,
    metric: str = "lap_time",
    frames: int = 30,
    title: str = "Performance Improvement Animation"
) -> go.Figure:
    """
    Create animated chart showing transition from current to target performance.

    Args:
        current_data: DataFrame with current performance (columns: corner, metric)
        target_data: DataFrame with target performance (columns: corner, metric)
        metric: Metric to animate
        frames: Number of animation frames
        title: Chart title

    Returns:
        Plotly Figure with animation controls

    Raises:
        ValueError: If data formats don't match
    """
    if current_data.empty or target_data.empty:
        raise ValueError("Both current_data and target_data must be non-empty")

    if metric not in current_data.columns or metric not in target_data.columns:
        raise ValueError(f"Metric '{metric}' not found in data")

    # Merge data
    merged = pd.merge(
        current_data[['corner', metric]],
        target_data[['corner', metric]],
        on='corner',
        suffixes=('_current', '_target')
    )

    # Create frames
    animation_frames = []
    for i in range(frames + 1):
        progress = i / frames
        interpolated_values = (
            merged[f'{metric}_current'] * (1 - progress) +
            merged[f'{metric}_target'] * progress
        )

        frame_data = go.Bar(
            x=merged['corner'],
            y=interpolated_values,
            name=f"Progress: {progress*100:.0f}%",
            marker=dict(
                color=interpolated_values,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text=metric.replace('_', ' ').title()
                    )
                )
            ),
            hovertemplate=(
                "Corner: %{x}<br>"
                f"{metric.replace('_', ' ').title()}: %{{y:.3f}}s<br>"
                f"Progress: {progress*100:.0f}%<br>"
                "<extra></extra>"
            )
        )

        animation_frames.append(go.Frame(
            data=[frame_data],
            name=f"frame_{i}"
        ))

    # Create initial figure with first frame
    fig = go.Figure(
        data=[animation_frames[0].data[0]],
        frames=animation_frames
    )

    # Add target reference line
    fig.add_trace(
        go.Scatter(
            x=merged['corner'],
            y=merged[f'{metric}_target'],
            mode='lines+markers',
            name='Target',
            line=dict(color='green', width=2, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        )
    )

    # Update layout with animation controls
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Corner",
        yaxis_title=metric.replace('_', ' ').title() + " (s)",
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate'
                        }]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate'
                        }]
                    )
                ],
                x=0.1,
                y=1.15
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        method='animate',
                        args=[[f'frame_{i}'], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate'
                        }],
                        label=f"{(i/frames)*100:.0f}%"
                    )
                    for i in range(frames + 1)
                ],
                x=0.1,
                len=0.85,
                xanchor='left',
                y=0,
                yanchor='top'
            )
        ],
        height=600,
        showlegend=True
    )

    return fig


def create_detailed_hover_template(
    data: pd.DataFrame,
    primary_metric: str,
    secondary_metrics: Optional[List[str]] = None
) -> str:
    """
    Create custom hover template with detailed metrics.

    Args:
        data: DataFrame with metrics
        primary_metric: Main metric to display
        secondary_metrics: Additional metrics to include in hover

    Returns:
        Formatted hover template string
    """
    if secondary_metrics is None:
        secondary_metrics = []

    template = f"<b>{primary_metric.replace('_', ' ').title()}: %{{y:.3f}}</b><br>"
    template += "Corner: %{x}<br>"

    for metric in secondary_metrics:
        if metric in data.columns:
            template += f"{metric.replace('_', ' ').title()}: %{{customdata[{secondary_metrics.index(metric)}]:.3f}}<br>"

    template += "<extra></extra>"

    return template


def export_chart_to_file(
    fig: go.Figure,
    filename: str,
    format: str = "html",
    width: int = 1200,
    height: int = 800
) -> str:
    """
    Export Plotly figure to file (PNG or HTML).

    Args:
        fig: Plotly Figure object
        filename: Output filename (without extension)
        format: Export format ('png', 'html', 'json')
        width: Image width in pixels (for PNG)
        height: Image height in pixels (for PNG)

    Returns:
        Path to exported file

    Raises:
        ValueError: If format is not supported
    """
    supported_formats = ['png', 'html', 'json']
    if format.lower() not in supported_formats:
        raise ValueError(f"Format must be one of {supported_formats}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{filename}_{timestamp}.{format.lower()}"

    if format.lower() == 'html':
        fig.write_html(
            output_path,
            include_plotlyjs='cdn',
            config={'displayModeBar': True, 'displaylogo': False}
        )
    elif format.lower() == 'png':
        try:
            fig.write_image(output_path, width=width, height=height)
        except Exception as e:
            raise RuntimeError(
                f"PNG export failed. Ensure kaleido is installed: pip install kaleido. Error: {e}"
            )
    elif format.lower() == 'json':
        fig.write_json(output_path)

    return output_path


def create_consistency_radar_chart(
    consistency_data: Dict[str, float],
    title: str = "Consistency Analysis"
) -> go.Figure:
    """
    Create radar chart showing consistency across different metrics.

    Args:
        consistency_data: Dictionary mapping metric names to consistency scores (0-100)
        title: Chart title

    Returns:
        Plotly Figure with radar chart
    """
    if not consistency_data:
        raise ValueError("consistency_data cannot be empty")

    categories = list(consistency_data.keys())
    values = list(consistency_data.values())

    # Close the radar chart by repeating first value
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name='Consistency Score',
        line=dict(color='rgb(0, 123, 255)', width=2),
        fillcolor='rgba(0, 123, 255, 0.3)',
        hovertemplate=(
            "<b>%{theta}</b><br>"
            "Score: %{r:.1f}/100<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickmode='linear',
                tick0=0,
                dtick=20
            )
        ),
        title=dict(text=title, x=0.5, xanchor='center'),
        showlegend=False,
        height=500
    )

    return fig


def create_race_position_simulation_chart(
    lap_data: pd.DataFrame,
    improvement_scenarios: Dict[str, float],
    title: str = "Race Position Simulation"
) -> go.Figure:
    """
    Create chart showing projected race positions under different improvement scenarios.

    Args:
        lap_data: DataFrame with columns [lap, current_position, current_time]
        improvement_scenarios: Dict mapping scenario names to improvement amounts (seconds)
        title: Chart title

    Returns:
        Plotly Figure with multiple traces
    """
    if lap_data.empty:
        raise ValueError("lap_data cannot be empty")

    fig = go.Figure()

    # Current performance baseline
    fig.add_trace(go.Scatter(
        x=lap_data['lap'],
        y=lap_data['current_position'],
        mode='lines+markers',
        name='Current',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))

    # Improvement scenarios
    colors = px.colors.qualitative.Plotly
    for idx, (scenario_name, improvement_time) in enumerate(improvement_scenarios.items()):
        # Simulate improved position (simplified - assumes linear relationship)
        improved_position = lap_data['current_position'] - (
            improvement_time / lap_data['current_time'] * 3  # Rough estimate: 1s = ~3 positions
        )
        improved_position = improved_position.clip(lower=1)  # Can't be better than 1st

        fig.add_trace(go.Scatter(
            x=lap_data['lap'],
            y=improved_position,
            mode='lines+markers',
            name=scenario_name,
            line=dict(color=colors[idx % len(colors)], width=2, dash='dash'),
            marker=dict(size=6),
            hovertemplate=(
                f"<b>{scenario_name}</b><br>"
                "Lap: %{x}<br>"
                "Position: %{y:.1f}<br>"
                f"Improvement: {improvement_time:.3f}s/lap<br>"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Lap",
        yaxis_title="Race Position",
        yaxis=dict(autorange='reversed'),  # Position 1 at top
        height=500,
        showlegend=True,
        legend=dict(x=1.05, y=1, xanchor='left'),
        hovermode='x unified'
    )

    return fig
