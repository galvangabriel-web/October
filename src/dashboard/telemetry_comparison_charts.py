"""
Telemetry Comparison Charts - Production Grade
===============================================

Visualization functions for Phase 1.2 driver insights enhancements:
- Ghost lap overlay charts
- Brake point analysis charts
- Corner speed benchmarking charts

Uses Plotly for interactive, dashboard-ready visualizations.

Author: Production Engineering Team
Version: 1.0.0 (Phase 1.2)
License: GR Cup 2025 Hackathon
"""

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

from src.insights.ghost_lap_comparator import GhostLapComparison
from src.insights.brake_point_analyzer import BrakeAnalysis, BrakeZone
from src.insights.corner_speed_benchmarking import CornerBenchmarkAnalysis, CornerSpeed


# Color scheme
COLORS = {
    'current': '#2196F3',  # Blue
    'ghost': '#FF9800',    # Orange
    'delta_positive': '#F44336',  # Red (slower)
    'delta_negative': '#4CAF50',  # Green (faster)
    'neutral': '#9E9E9E',  # Gray
    'accent': '#9C27B0'    # Purple
}


def create_ghost_lap_overlay_chart(comparison: GhostLapComparison) -> go.Figure:
    """
    Create a multi-panel chart comparing current lap vs ghost lap.

    Panels:
    1. Speed overlay
    2. Brake pressure overlay
    3. Throttle overlay
    4. Cumulative time delta

    Args:
        comparison: GhostLapComparison object from GhostLapComparator

    Returns:
        Plotly Figure with 4 subplots
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f'Speed Comparison (Lap {comparison.current_lap_number} vs Lap {comparison.ghost_lap_number})',
            'Brake Pressure',
            'Throttle Position',
            f'Cumulative Time Delta ({comparison.lap_time_delta:+.3f}s total)'
        ),
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )

    distance = comparison.distance

    # Panel 1: Speed
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=comparison.current_speed,
            mode='lines',
            name=f'Lap {comparison.current_lap_number}',
            line=dict(color=COLORS['current'], width=2),
            showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=comparison.ghost_speed,
            mode='lines',
            name=f'Lap {comparison.ghost_lap_number} (Best)',
            line=dict(color=COLORS['ghost'], width=2, dash='dash'),
            showlegend=True
        ),
        row=1, col=1
    )

    # Panel 2: Brake Pressure
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=comparison.current_brake_f,
            mode='lines',
            name='Current Brake',
            line=dict(color=COLORS['current'], width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=comparison.ghost_brake_f,
            mode='lines',
            name='Best Brake',
            line=dict(color=COLORS['ghost'], width=2, dash='dash'),
            showlegend=False
        ),
        row=2, col=1
    )

    # Panel 3: Throttle
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=comparison.current_throttle,
            mode='lines',
            name='Current Throttle',
            line=dict(color=COLORS['current'], width=2),
            showlegend=False
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=comparison.ghost_throttle,
            mode='lines',
            name='Best Throttle',
            line=dict(color=COLORS['ghost'], width=2, dash='dash'),
            showlegend=False
        ),
        row=3, col=1
    )

    # Panel 4: Cumulative Time Delta
    delta_color = np.where(
        comparison.cumulative_time_delta > 0,
        COLORS['delta_positive'],  # Red where slower
        COLORS['delta_negative']   # Green where faster
    )

    fig.add_trace(
        go.Scatter(
            x=distance,
            y=comparison.cumulative_time_delta,
            mode='lines',
            name='Time Delta',
            line=dict(color=COLORS['accent'], width=3),
            fill='tozeroy',
            fillcolor='rgba(156, 39, 176, 0.2)',
            showlegend=False
        ),
        row=4, col=1
    )

    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS['neutral'], row=4, col=1)

    # Update axes
    fig.update_xaxes(title_text="Distance (m)", row=4, col=1)
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
    fig.update_yaxes(title_text="Brake (bar)", row=2, col=1)
    fig.update_yaxes(title_text="Throttle (%)", row=3, col=1)
    fig.update_yaxes(title_text="Delta (s)", row=4, col=1)

    fig.update_layout(
        height=900,
        title_text=f"Ghost Lap Analysis - Vehicle {comparison.vehicle_number}",
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def create_time_delta_heatmap(comparison: GhostLapComparison) -> go.Figure:
    """
    Create a heatmap showing where time is gained/lost around the lap.

    Args:
        comparison: GhostLapComparison object

    Returns:
        Plotly Figure with time delta heatmap
    """
    # Create bins for distance
    num_bins = 50
    distance_bins = np.linspace(comparison.distance.min(), comparison.distance.max(), num_bins)
    delta_binned = []

    for i in range(len(distance_bins) - 1):
        mask = (comparison.distance >= distance_bins[i]) & (comparison.distance < distance_bins[i+1])
        if np.any(mask):
            avg_delta = np.mean(comparison.cumulative_time_delta[mask])
            delta_binned.append(avg_delta)
        else:
            delta_binned.append(0)

    # Create figure
    fig = go.Figure()

    # Add heatmap bar
    fig.add_trace(go.Heatmap(
        x=distance_bins[:-1],
        y=[0],
        z=[delta_binned],
        colorscale='RdYlGn_r',  # Red = slower, Green = faster
        zmid=0,
        colorbar=dict(title="Time Delta (s)", x=1.1),
        showscale=True
    ))

    # Add annotations for biggest gains/losses
    for gain in comparison.biggest_gains:
        fig.add_annotation(
            x=gain['distance'],
            y=0,
            text=f"+{gain['time_gained']:.2f}s",
            showarrow=True,
            arrowhead=2,
            arrowcolor='green',
            ay=-40
        )

    for loss in comparison.biggest_losses:
        fig.add_annotation(
            x=loss['distance'],
            y=0,
            text=f"-{loss['time_lost']:.2f}s",
            showarrow=True,
            arrowhead=2,
            arrowcolor='red',
            ay=40
        )

    fig.update_layout(
        title=f"Time Gain/Loss Heatmap - Lap {comparison.current_lap_number} vs Best",
        xaxis_title="Distance (m)",
        yaxis=dict(showticklabels=False, showgrid=False),
        height=200,
        template='plotly_white'
    )

    return fig


def create_brake_zones_chart(analysis: BrakeAnalysis) -> go.Figure:
    """
    Create a chart showing all brake zones with metrics.

    Args:
        analysis: BrakeAnalysis object from BrakePointAnalyzer

    Returns:
        Plotly Figure with brake zone visualization
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f'Brake Zones - Lap {analysis.lap_number if analysis.lap_number else "Session Average"}',
            'Brake Pressure'
        ),
        vertical_spacing=0.12,
        row_heights=[0.3, 0.7]
    )

    # Panel 1: Brake zone markers
    for zone in analysis.brake_zones:
        # Add vertical span for each brake zone
        fig.add_vrect(
            x0=zone.start_distance,
            x1=zone.end_distance,
            fillcolor=COLORS['accent'],
            opacity=0.3,
            layer="below",
            line_width=0,
            row=1, col=1
        )

        # Add zone label
        mid_point = (zone.start_distance + zone.end_distance) / 2
        fig.add_annotation(
            x=mid_point,
            y=0.5,
            text=f"C{zone.corner_number}",
            showarrow=False,
            font=dict(size=10, color='white'),
            bgcolor=COLORS['accent'],
            row=1, col=1
        )

    # Panel 2: Brake pressure profile (if we have continuous data)
    # For now, show bar chart of peak pressures
    corner_ids = [z.corner_number for z in analysis.brake_zones]
    peak_pressures = [z.peak_pressure for z in analysis.brake_zones]

    fig.add_trace(
        go.Bar(
            x=[f"Corner {cid}" for cid in corner_ids],
            y=peak_pressures,
            name='Peak Brake Pressure',
            marker_color=COLORS['current'],
            showlegend=False
        ),
        row=2, col=1
    )

    # If we have comparison data, add delta bars
    if analysis.brake_zones and analysis.brake_zones[0].delta_peak_pressure is not None:
        deltas = [z.delta_peak_pressure for z in analysis.brake_zones if z.delta_peak_pressure is not None]
        if deltas:
            fig.add_trace(
                go.Bar(
                    x=[f"Corner {z.corner_number}" for z in analysis.brake_zones if z.delta_peak_pressure is not None],
                    y=deltas,
                    name='Delta vs Best',
                    marker_color=[COLORS['delta_positive'] if d > 0 else COLORS['delta_negative'] for d in deltas],
                    showlegend=True
                ),
                row=2, col=1
            )

    fig.update_xaxes(title_text="Corner", row=2, col=1)
    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="Brake Pressure (bar)", row=2, col=1)

    fig.update_layout(
        height=600,
        title_text=f"Brake Zone Analysis - Vehicle {analysis.vehicle_number}",
        template='plotly_white',
        barmode='group'
    )

    return fig


def create_brake_consistency_chart(analysis: BrakeAnalysis) -> go.Figure:
    """
    Create a chart showing brake point consistency across laps.

    Args:
        analysis: BrakeAnalysis with consistency data (from analyze_session)

    Returns:
        Plotly Figure with consistency visualization
    """
    if not analysis.brake_zones or analysis.brake_zones[0].brake_point_std is None:
        # No consistency data available
        return go.Figure().add_annotation(
            text="No consistency data available<br>(requires multi-lap analysis)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )

    corner_ids = [z.corner_number for z in analysis.brake_zones]
    brake_point_stds = [z.brake_point_std for z in analysis.brake_zones]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[f"Corner {cid}" for cid in corner_ids],
        y=brake_point_stds,
        name='Brake Point Variation',
        marker_color=[
            COLORS['delta_negative'] if std < 5 else
            COLORS['neutral'] if std < 10 else
            COLORS['delta_positive']
            for std in brake_point_stds
        ],
        text=[f"±{std:.1f}m" for std in brake_point_stds],
        textposition='auto'
    ))

    # Add threshold lines
    fig.add_hline(y=5, line_dash="dash", line_color='green', annotation_text="Excellent (<5m)")
    fig.add_hline(y=10, line_dash="dash", line_color='orange', annotation_text="Good (<10m)")

    fig.update_layout(
        title=f"Brake Point Consistency - {analysis.brake_point_consistency}",
        xaxis_title="Corner",
        yaxis_title="Standard Deviation (m)",
        height=400,
        template='plotly_white'
    )

    return fig


def create_corner_speed_chart(analysis: CornerBenchmarkAnalysis) -> go.Figure:
    """
    Create a chart showing entry/apex/exit speeds for all corners.

    Args:
        analysis: CornerBenchmarkAnalysis object

    Returns:
        Plotly Figure with corner speed comparison
    """
    corner_names = [c.corner_name for c in analysis.corners]
    entry_speeds = [c.entry_speed for c in analysis.corners]
    apex_speeds = [c.apex_speed for c in analysis.corners]
    exit_speeds = [c.exit_speed for c in analysis.corners]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=corner_names,
        y=entry_speeds,
        mode='lines+markers',
        name='Entry Speed',
        line=dict(color=COLORS['current'], width=2),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=corner_names,
        y=apex_speeds,
        mode='lines+markers',
        name='Apex Speed',
        line=dict(color=COLORS['delta_positive'], width=2),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=corner_names,
        y=exit_speeds,
        mode='lines+markers',
        name='Exit Speed',
        line=dict(color=COLORS['delta_negative'], width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=f"Corner Speed Profile - Lap {analysis.lap_number if analysis.lap_number else 'Session'}",
        xaxis_title="Corner",
        yaxis_title="Speed (km/h)",
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )

    return fig


def create_corner_speed_delta_chart(analysis: CornerBenchmarkAnalysis) -> go.Figure:
    """
    Create a chart showing speed deltas vs best lap for each corner.

    Args:
        analysis: CornerBenchmarkAnalysis with comparison data

    Returns:
        Plotly Figure with delta visualization
    """
    # Filter corners with comparison data
    corners_with_deltas = [c for c in analysis.corners if c.delta_apex_speed is not None]

    if not corners_with_deltas:
        return go.Figure().add_annotation(
            text="No comparison data available<br>(requires best lap comparison)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )

    corner_names = [c.corner_name for c in corners_with_deltas]
    entry_deltas = [c.delta_entry_speed for c in corners_with_deltas]
    apex_deltas = [c.delta_apex_speed for c in corners_with_deltas]
    exit_deltas = [c.delta_exit_speed for c in corners_with_deltas]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Entry Delta',
        x=corner_names,
        y=entry_deltas,
        marker_color=[COLORS['delta_negative'] if d > 0 else COLORS['delta_positive'] for d in entry_deltas]
    ))

    fig.add_trace(go.Bar(
        name='Apex Delta',
        x=corner_names,
        y=apex_deltas,
        marker_color=[COLORS['delta_negative'] if d > 0 else COLORS['delta_positive'] for d in apex_deltas]
    ))

    fig.add_trace(go.Bar(
        name='Exit Delta',
        x=corner_names,
        y=exit_deltas,
        marker_color=[COLORS['delta_negative'] if d > 0 else COLORS['delta_positive'] for d in exit_deltas]
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS['neutral'])

    fig.update_layout(
        title=f"Corner Speed Delta vs Best Lap",
        xaxis_title="Corner",
        yaxis_title="Speed Delta (km/h) - Positive = Faster",
        height=500,
        template='plotly_white',
        barmode='group'
    )

    return fig


def create_corner_type_distribution(analysis: CornerBenchmarkAnalysis) -> go.Figure:
    """
    Create a pie chart showing distribution of corner types.

    Args:
        analysis: CornerBenchmarkAnalysis object

    Returns:
        Plotly Figure with corner type distribution
    """
    corner_types = [c.corner_type for c in analysis.corners]
    type_counts = pd.Series(corner_types).value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=type_counts.index,
        values=type_counts.values,
        hole=0.4,
        marker_colors=[COLORS['delta_positive'], COLORS['current'], COLORS['delta_negative'], COLORS['accent']]
    )])

    fig.update_layout(
        title="Corner Type Distribution",
        height=400,
        template='plotly_white'
    )

    return fig


if __name__ == "__main__":
    """Example usage"""
    print("Telemetry Comparison Charts Module")
    print("=" * 50)
    print("\nFunctions available:")
    print("  - create_ghost_lap_overlay_chart()")
    print("  - create_time_delta_heatmap()")
    print("  - create_brake_zones_chart()")
    print("  - create_brake_consistency_chart()")
    print("  - create_corner_speed_chart()")
    print("  - create_corner_speed_delta_chart()")
    print("  - create_corner_type_distribution()")
    print("\nUse these functions in the dashboard to visualize Phase 1.2 enhancements.")
