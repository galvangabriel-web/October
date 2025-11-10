"""
Winning Edge Widget for Racing Telemetry Dashboard
====================================================

This module provides "The Winning Edge" performance dashboard - a visual-first,
race-day decision support system for driver + race engineer collaboration.

Features:
- Section 1: Race Winner's Dashboard (Time Loss Heat Map, Speed Gap Spider Chart)
- Section 2: Correlation Dashboard (Brake-Exit Correlation, Speed Cascade, Consistency Matrix)
- Section 3: Priority Action Cards (Turn-specific improvement targets)
- Section 4: Race Simulation Impact (Position Gain Predictor, Overtaking Map)
- Section 5: 6-Week Transformation Timeline (Progress tracking)
- Section 6: Session Visual Guides (Turn reference maps, brake pressure profiles)
- Section 7: Competitive Advantage Summary (Performance overview dashboard)

Usage:
    from src.dashboard.winning_edge_widget import create_winning_edge_layout, create_winning_edge_callbacks

    # In your Dash app:
    app.layout = dbc.Tabs([
        dbc.Tab(create_winning_edge_layout(), label="Winning Edge", tab_id="winning-edge"),
        # ... other tabs
    ])

    create_winning_edge_callbacks(app)

Philosophy:
- Visual-first approach for rapid decision-making
- Instant pattern recognition: RED = Fix This, GREEN = Copy This
- Mathematical proof showing WHY improvements work
- Race context: translates seconds into positions
- Action-oriented: every chart leads to specific action
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Color scheme for consistent branding
COLORS = {
    'winning': '#FFD700',      # Gold
    'opportunity': '#ff6b6b',  # Red - needs work
    'benchmark': '#51cf66',    # Green - target level
    'current': '#3498db',      # Blue - current performance
    'background': '#f8f9fa',   # Light gray
    'text': '#2c3e50'          # Dark gray
}


# ============================================================================
# SECTION 1: RACE WINNER'S DASHBOARD
# ============================================================================

def create_time_loss_heatmap(corner_data: Dict[str, Dict]) -> go.Figure:
    """
    Create interactive heatmap showing time loss per corner.

    Args:
        corner_data: Dictionary with corner names as keys and metrics as values
                    e.g., {'Turn 1': {'time_loss': 0.210, 'pct_of_total': 48}, ...}

    Returns:
        Plotly Figure with heatmap visualization
    """
    if not corner_data:
        # Default example data
        corner_data = {
            'Turn 1<br>Uphill': {'time_loss': 0.210, 'pct_of_total': 48},
            'Turn 6<br>Hairpin': {'time_loss': 0.180, 'pct_of_total': 41},
            'Turn 11<br>Fast': {'time_loss': 0.050, 'pct_of_total': 11}
        }

    corners = list(corner_data.keys())
    time_losses = [data['time_loss'] for data in corner_data.values()]
    pct_totals = [data['pct_of_total'] for data in corner_data.values()]

    # Create text labels
    text_labels = [[f"{loss:.3f}s<br>{pct}% of total"
                   for loss, pct in zip(time_losses, pct_totals)]]

    fig = go.Figure(data=go.Heatmap(
        z=[time_losses],
        x=corners,
        y=['Time Loss'],
        colorscale=[
            [0, COLORS['benchmark']],     # Low loss (green)
            [0.5, 'yellow'],              # Medium loss
            [1, COLORS['opportunity']]    # High loss (red)
        ],
        text=text_labels,
        texttemplate='%{text}',
        textfont={"size": 16, "color": "white"},
        showscale=True,
        colorbar=dict(title="Time Loss (s)")
    ))

    total_opportunity = sum(time_losses)

    fig.update_layout(
        title={
            'text': f"🏁 YOUR PATH TO VICTORY: {total_opportunity:.2f}s Available Per Lap",
            'font': {'size': 24, 'color': COLORS['text']}
        },
        height=300,
        margin=dict(l=60, r=60, t=100, b=60),
        annotations=[
            dict(
                text=f"89% of gains in just 2 corners!",
                x=0.5, y=1.15,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=18, color=COLORS['opportunity'])
            )
        ]
    )

    return fig


def create_speed_gap_spider(corner_metrics: Dict[str, List[float]]) -> go.Figure:
    """
    Create radar/spider chart showing performance gaps across metrics.

    Args:
        corner_metrics: Dict with corner names and performance values (% of optimal)
                       e.g., {'Turn 1': [97.8, 96.8, 97.1, 83.5, 68.5, 87.5], ...}

    Returns:
        Plotly Figure with radar chart
    """
    categories = [
        'Entry Speed', 'Apex Speed', 'Exit Speed',
        'Brake Efficiency', 'Throttle Application', 'Consistency'
    ]

    if not corner_metrics:
        # Default example data
        corner_metrics = {
            'Turn 1': [97.8, 96.8, 97.1, 83.5, 68.5, 87.5],
            'Turn 6': [97.5, 95.1, 96.3, 83.3, 52.3, 83.7],
            'Turn 11 (Target)': [99.6, 98.9, 98.6, 116.9, 88.7, 94.8]
        }

    fig = go.Figure()

    colors = {
        'Turn 1': COLORS['current'],
        'Turn 6': COLORS['opportunity'],
        'Turn 11 (Target)': COLORS['benchmark']
    }

    for corner_name, values in corner_metrics.items():
        color = colors.get(corner_name, COLORS['current'])
        opacity = 0.6 if 'Target' in corner_name else 0.4

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=corner_name,
            line_color=color,
            opacity=opacity
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 120],
                ticktext=['0%', '50%', '100%', 'Optimal'],
                tickvals=[0, 50, 100, 120]
            )
        ),
        showlegend=True,
        title={
            'text': "Performance Profile: Your Shape vs Champion Shape",
            'font': {'size': 20}
        },
        height=500,
        legend=dict(
            x=0.85, y=0.95,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )

    return fig


# ============================================================================
# SECTION 2: CORRELATION DASHBOARD
# ============================================================================

def create_brake_exit_correlation(brake_excess: List[float],
                                  exit_deficit: List[float],
                                  corner_labels: List[str]) -> go.Figure:
    """
    Create scatter plot showing 1:1 brake-exit correlation.

    Args:
        brake_excess: Brake phase excess time (seconds)
        exit_deficit: Exit phase deficit time (seconds)
        corner_labels: Corner names

    Returns:
        Plotly Figure with correlation scatter plot
    """
    if not brake_excess:
        # Default example data
        brake_excess = [0.62, 0.91, -0.45]
        exit_deficit = [-0.62, -0.91, 0.45]
        corner_labels = ['Turn 1', 'Turn 6', 'Turn 11']

    fig = go.Figure()

    # Data points
    colors_map = [COLORS['opportunity'], COLORS['opportunity'], COLORS['benchmark']]
    sizes = [20, 25, 15]

    fig.add_trace(go.Scatter(
        x=brake_excess,
        y=exit_deficit,
        mode='markers+text',
        text=corner_labels,
        textposition="top center",
        marker=dict(
            size=sizes,
            color=colors_map,
            showscale=False,
            line=dict(width=2, color='white')
        ),
        name='Your Performance',
        hovertemplate='<b>%{text}</b><br>Brake Excess: %{x:.2f}s<br>Exit Deficit: %{y:.2f}s<extra></extra>'
    ))

    # Perfect correlation line
    x_line = np.linspace(-1, 1, 100)
    fig.add_trace(go.Scatter(
        x=x_line,
        y=-x_line,
        mode='lines',
        name='Perfect 1:1 Ratio',
        line=dict(dash='dash', color='gray', width=2),
        hoverinfo='skip'
    ))

    # Add improvement arrows for opportunity corners
    for i in range(min(2, len(corner_labels))):
        if brake_excess[i] > 0:  # Only for corners with excess braking
            fig.add_annotation(
                x=brake_excess[i],
                y=exit_deficit[i],
                ax=0,
                ay=0,
                xref="x", yref="y",
                axref="x", ayref="y",
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=3,
                arrowcolor=COLORS['benchmark']
            )

    fig.update_layout(
        title={
            'text': "🎯 THE KEY DISCOVERY: Every 0.1s Less Braking = 0.1s More Acceleration",
            'font': {'size': 18}
        },
        xaxis_title="Brake Phase Excess (seconds)",
        yaxis_title="Exit Phase Deficit (seconds)",
        height=500,
        hovermode='closest',
        annotations=[
            dict(
                text="Move Turn 1 & 6<br>toward Turn 11!",
                x=0, y=-1,
                showarrow=False,
                font=dict(size=14, color=COLORS['benchmark'])
            )
        ]
    )

    return fig


def create_speed_cascade_waterfall(corner_name: str,
                                   cascade_data: Dict[str, float]) -> go.Figure:
    """
    Create waterfall chart showing how speed losses compound.

    Args:
        corner_name: Name of the corner being analyzed
        cascade_data: Dictionary with phase names and speed losses

    Returns:
        Plotly Figure with waterfall chart
    """
    if not cascade_data:
        # Default example data for Turn 6
        cascade_data = {
            'Entry Gap': -3.6,
            'Apex Gap': -3.2,
            'Exit Gap': -3.7,
            'Straight Loss': -5.0
        }

    phases = list(cascade_data.keys())
    values = list(cascade_data.values())

    fig = go.Figure(go.Waterfall(
        name=corner_name,
        orientation="v",
        measure=["relative"] * len(phases) + ["total"],
        x=phases + ["Total Impact"],
        y=values + [None],
        text=[f"{v:.1f} km/h" for v in values] + [f"{sum(values):.1f} km/h"],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": COLORS['opportunity']}},
        totals={"marker": {"color": 'darkred'}}
    ))

    fig.update_layout(
        title={
            'text': f"How {abs(values[0]):.1f} km/h Entry Loss Becomes {abs(sum(values)):.1f} km/h Disadvantage",
            'font': {'size': 18}
        },
        showlegend=False,
        height=400,
        yaxis_title="Speed Change (km/h)"
    )

    return fig


def create_consistency_performance_matrix(corner_data: List[Dict]) -> go.Figure:
    """
    Create bubble chart showing consistency vs performance relationship.

    Args:
        corner_data: List of dicts with consistency_pct, efficiency_pct, time_loss, name

    Returns:
        Plotly Figure with bubble scatter plot
    """
    if not corner_data:
        # Default example data
        corner_data = [
            {'name': 'Turn 1', 'consistency_pct': 87.5, 'efficiency_pct': 78.9, 'time_loss': 0.21},
            {'name': 'Turn 6', 'consistency_pct': 83.7, 'efficiency_pct': 82.0, 'time_loss': 0.18},
            {'name': 'Turn 11', 'consistency_pct': 94.8, 'efficiency_pct': 95.0, 'time_loss': 0.05}
        ]

    fig = go.Figure()

    consistency = [d['consistency_pct'] for d in corner_data]
    efficiency = [d['efficiency_pct'] for d in corner_data]
    time_losses = [d['time_loss'] for d in corner_data]
    names = [d['name'] for d in corner_data]

    # Determine colors based on time loss
    colors = [COLORS['benchmark'] if tl < 0.1 else COLORS['opportunity'] for tl in time_losses]

    fig.add_trace(go.Scatter(
        x=consistency,
        y=efficiency,
        mode='markers+text',
        text=[f"{name}<br>{tl:.2f}s loss" for name, tl in zip(names, time_losses)],
        textposition="top center",
        marker=dict(
            size=[tl * 1000 for tl in time_losses],  # Size = time loss in milliseconds
            color=colors,
            opacity=0.6,
            line=dict(width=2, color='black')
        ),
        hovertemplate='<b>%{text}</b><br>Consistency: %{x:.1f}%<br>Efficiency: %{y:.1f}%<extra></extra>'
    ))

    # Add target zone
    fig.add_shape(
        type="rect",
        x0=92, y0=92, x1=100, y1=100,
        line=dict(color=COLORS['benchmark'], width=2, dash="dash"),
        fillcolor=COLORS['benchmark'],
        opacity=0.1
    )

    fig.add_annotation(
        text="TARGET ZONE",
        x=96, y=96,
        showarrow=False,
        font=dict(color=COLORS['benchmark'], size=16, family='Arial Black')
    )

    fig.update_layout(
        title="The Consistency-Performance Sweet Spot",
        xaxis_title="Consistency % (Your Average vs Your Best)",
        yaxis_title="Overall Efficiency %",
        height=500,
        xaxis=dict(range=[80, 100]),
        yaxis=dict(range=[75, 100])
    )

    return fig


# ============================================================================
# SECTION 3: PRIORITY ACTION CARDS
# ============================================================================

def create_turn_action_card(turn_name: str,
                            current_metrics: Dict[str, float],
                            target_metrics: Dict[str, float]) -> go.Figure:
    """
    Create action card showing current vs target for specific turn.

    Args:
        turn_name: Name of the turn
        current_metrics: Current performance metrics
        target_metrics: Target performance metrics

    Returns:
        Plotly Figure with grouped bar chart
    """
    if not current_metrics or not target_metrics:
        # Default example data for Turn 6
        current_metrics = {
            'Brake Point (m)': 120,
            'Brake Pressure (%)': 95,
            'Throttle Timing (%)': 52.3,
            'Exit Speed (km/h)': 98.7
        }
        target_metrics = {
            'Brake Point (m)': 132,
            'Brake Pressure (%)': 75,
            'Throttle Timing (%)': 57.5,
            'Exit Speed (km/h)': 102.4
        }

    categories = list(current_metrics.keys())
    current_vals = list(current_metrics.values())
    target_vals = list(target_metrics.values())

    fig = go.Figure()

    x = np.arange(len(categories))
    width = 0.35

    fig.add_trace(go.Bar(
        x=x - width/2,
        y=current_vals,
        name='Current',
        marker_color=COLORS['opportunity'],
        text=[f"{val:.1f}" for val in current_vals],
        textposition='outside',
        width=width
    ))

    fig.add_trace(go.Bar(
        x=x + width/2,
        y=target_vals,
        name='Target',
        marker_color=COLORS['benchmark'],
        text=[f"{val:.1f}" for val in target_vals],
        textposition='outside',
        width=width
    ))

    # Add improvement deltas
    for i in range(len(categories)):
        delta = abs(target_vals[i] - current_vals[i])
        fig.add_annotation(
            x=i,
            y=max(current_vals[i], target_vals[i]) + max(current_vals[i], target_vals[i]) * 0.1,
            text=f"↑ {delta:.1f}",
            showarrow=False,
            font=dict(color=COLORS['current'], size=14, family='Arial Black')
        )

    fig.update_layout(
        title=f"🎯 {turn_name.upper()} ACTION CARD",
        xaxis=dict(ticktext=categories, tickvals=x, tickangle=-15),
        barmode='group',
        height=450,
        showlegend=True,
        legend=dict(x=0.8, y=0.95)
    )

    return fig


def create_phase_distribution(phase_data: Dict[str, List[float]]) -> go.Figure:
    """
    Create stacked bar chart showing phase distribution.

    Args:
        phase_data: Dict with corner names and [brake%, apex%, exit%] values

    Returns:
        Plotly Figure with stacked bar chart
    """
    if not phase_data:
        # Default example data
        phase_data = {
            'Turn 1 Current': [35, 20, 45],
            'Turn 1 Target': [30, 20, 50],
            'Turn 6 Current': [35, 20, 45],
            'Turn 6 Target': [30, 20, 50],
            'Turn 11 Benchmark': [25, 20, 55]
        }

    corners = list(phase_data.keys())
    brake_pct = [data[0] for data in phase_data.values()]
    apex_pct = [data[1] for data in phase_data.values()]
    exit_pct = [data[2] for data in phase_data.values()]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Brake Phase',
        x=corners,
        y=brake_pct,
        marker_color=COLORS['opportunity'],
        text=[f"{val}%" for val in brake_pct],
        textposition='inside',
        textfont=dict(color='white', size=12)
    ))

    fig.add_trace(go.Bar(
        name='Apex Phase',
        x=corners,
        y=apex_pct,
        marker_color='gold',
        text=[f"{val}%" for val in apex_pct],
        textposition='inside',
        textfont=dict(color='white', size=12)
    ))

    fig.add_trace(go.Bar(
        name='Exit Phase',
        x=corners,
        y=exit_pct,
        marker_color=COLORS['benchmark'],
        text=[f"{val}%" for val in exit_pct],
        textposition='inside',
        textfont=dict(color='white', size=12)
    ))

    fig.update_layout(
        barmode='stack',
        title="Phase Distribution: Shift Red to Green!",
        yaxis_title="% of Corner Time",
        height=400,
        annotations=[
            dict(
                text="MORE GREEN = MORE SPEED",
                x=0.5, y=1.1,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=16, color=COLORS['benchmark'], family='Arial Black')
            )
        ]
    )

    return fig


# ============================================================================
# SECTION 4: RACE SIMULATION IMPACT
# ============================================================================

def create_position_gain_predictor(time_gain_per_lap: float,
                                   num_laps: int = 20) -> go.Figure:
    """
    Create line chart showing cumulative advantage over race distance.

    Args:
        time_gain_per_lap: Expected time gain per lap (seconds)
        num_laps: Number of laps in race

    Returns:
        Plotly Figure with dual-axis line chart
    """
    laps = list(range(1, num_laps + 1))
    cumulative_gain = [time_gain_per_lap * lap for lap in laps]
    position_gains = [gain // 2.2 for gain in cumulative_gain]  # ~2.2s per position

    fig = go.Figure()

    # Time advantage line
    fig.add_trace(go.Scatter(
        x=laps,
        y=cumulative_gain,
        mode='lines+markers',
        name='Time Advantage (seconds)',
        line=dict(color=COLORS['current'], width=3),
        marker=dict(size=8),
        yaxis='y1'
    ))

    # Position gains line
    fig.add_trace(go.Scatter(
        x=laps,
        y=position_gains,
        mode='lines+markers',
        name='Positions Gained',
        line=dict(color=COLORS['benchmark'], width=3, dash='dash'),
        marker=dict(size=8),
        yaxis='y2'
    ))

    # Add milestone annotations (with bounds checking for all lap counts)
    milestones = []

    # Only add lap 10 milestone if we have at least 10 laps
    if num_laps >= 10:
        milestones.append(
            (10, cumulative_gain[9], f"Lap 10: {int(position_gains[9])} positions gained!")
        )

    # Only add lap 20 milestone if we have at least 20 laps
    if num_laps >= 20:
        milestones.append(
            (20, cumulative_gain[19], f"Lap 20: {int(position_gains[19])} positions = PODIUM!")
        )

    # If race is short (< 10 laps), add milestone at midpoint
    if num_laps < 10 and num_laps >= 1:
        midpoint = (num_laps // 2)
        if midpoint >= 1:
            milestones.append(
                (midpoint, cumulative_gain[midpoint - 1],
                 f"Lap {midpoint}: {int(position_gains[midpoint - 1])} positions gained!")
            )

    # Always add a final lap milestone
    if num_laps >= 1:
        final_idx = num_laps - 1
        milestones.append(
            (num_laps, cumulative_gain[final_idx],
             f"Lap {num_laps}: {int(position_gains[final_idx])} total positions = TARGET!")
        )

    for lap, time_val, text in milestones:
        if lap <= num_laps:
            fig.add_annotation(
                x=lap, y=time_val,
                text=text,
                showarrow=True,
                arrowhead=2,
                bgcolor=COLORS['winning'],
                bordercolor='black',
                borderwidth=1,
                font=dict(size=11, color='black')
            )

    fig.update_layout(
        title=f"🏆 Your Path to the Podium: {time_gain_per_lap:.2f}s × {num_laps} Laps = Positions",
        xaxis_title="Race Lap",
        yaxis=dict(
            title="Cumulative Time Gain (seconds)",
            side='left'
        ),
        yaxis2=dict(
            title="Grid Positions Gained",
            overlaying='y',
            side='right'
        ),
        height=500,
        hovermode='x unified',
        legend=dict(x=0.05, y=0.95)
    )

    return fig


def create_overtaking_opportunity_map(track_layout: Dict = None) -> go.Figure:
    """
    Create track map showing overtaking zones.

    Args:
        track_layout: Dict with track coordinates and corner data (optional)

    Returns:
        Plotly Figure with track visualization
    """
    # Simplified track layout (example)
    if not track_layout:
        track_x = [0, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 0]
        track_y = [0, 0, 50, 100, 100, 150, 200, 200, 150, 100, 50, 0, 0]
        corners_x = [150, 350, 550]
        corners_y = [50, 200, 50]
        opportunity = [0.21, 0.18, 0.05]
        labels = ['T1: 0.21s', 'T6: 0.18s', 'T11: 0.05s']
    else:
        track_x = track_layout.get('track_x', [])
        track_y = track_layout.get('track_y', [])
        corners_x = track_layout.get('corners_x', [])
        corners_y = track_layout.get('corners_y', [])
        opportunity = track_layout.get('opportunity', [])
        labels = track_layout.get('labels', [])

    fig = go.Figure()

    # Track outline
    fig.add_trace(go.Scatter(
        x=track_x,
        y=track_y,
        mode='lines',
        line=dict(color='gray', width=20),
        name='Track',
        hoverinfo='skip'
    ))

    # Corner markers with opportunity size
    colors = [COLORS['opportunity'] if opp > 0.1 else COLORS['benchmark'] for opp in opportunity]

    fig.add_trace(go.Scatter(
        x=corners_x,
        y=corners_y,
        mode='markers+text',
        marker=dict(
            size=[s * 200 for s in opportunity],
            color=colors,
            line=dict(width=2, color='black')
        ),
        text=labels,
        textposition="top center",
        name='Time Gain Zones',
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))

    # Add overtaking zone annotation
    if len(corners_x) >= 2:
        fig.add_annotation(
            x=corners_x[1] + 50, y=corners_y[1] - 20,
            text="PRIMARY OVERTAKE<br>3.7 km/h exit advantage<br>= EASY PASS!",
            showarrow=True,
            arrowhead=2,
            ax=-50,
            ay=-30,
            bgcolor=COLORS['winning'],
            bordercolor='black',
            borderwidth=1
        )

    fig.update_layout(
        title="Track Dominance Map: Where You'll Make The Moves",
        showlegend=False,
        height=400,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor='x', scaleratio=1)
    )

    return fig


# ============================================================================
# SECTION 5: 6-WEEK TRANSFORMATION TIMELINE
# ============================================================================

def create_weekly_target_progression() -> go.Figure:
    """
    Create Gantt-style progress tracker for 6-week improvement plan.

    Returns:
        Plotly Figure with timeline chart
    """
    tasks = [
        dict(Task="Week 1-2: T6 Brake Point", Start='2025-01-01', Finish='2025-01-14', Resource='Brake'),
        dict(Task="Week 1-2: T6 Throttle", Start='2025-01-01', Finish='2025-01-14', Resource='Throttle'),
        dict(Task="Week 3-4: T1 Brake", Start='2025-01-15', Finish='2025-01-28', Resource='Brake'),
        dict(Task="Week 3-4: T1 Exit", Start='2025-01-15', Finish='2025-01-28', Resource='Throttle'),
        dict(Task="Week 5-6: Consistency", Start='2025-01-29', Finish='2025-02-11', Resource='Consistency'),
        dict(Task="Week 5-6: Race Sim", Start='2025-01-29', Finish='2025-02-11', Resource='Race')
    ]

    df = pd.DataFrame(tasks)

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Resource",
        color_discrete_map={
            'Brake': COLORS['opportunity'],
            'Throttle': COLORS['benchmark'],
            'Consistency': COLORS['current'],
            'Race': COLORS['winning']
        },
        title="Your 6-Week Path to 0.44s Improvement"
    )

    fig.update_layout(
        height=400,
        xaxis_title="Timeline",
        yaxis_title="",
        showlegend=True
    )

    # Add milestone markers
    milestones = [
        ('2025-01-14', 'T6 Complete: 0.18s gained'),
        ('2025-01-28', 'T1 Complete: 0.39s total'),
        ('2025-02-11', 'Race Ready: 0.44s total')
    ]

    for date, text in milestones:
        fig.add_vline(
            x=pd.to_datetime(date).timestamp() * 1000,
            line_dash="dash",
            line_color=COLORS['winning'],
            annotation_text=text,
            annotation_position="top"
        )

    return fig


def create_improvement_curve(actual_progress: Optional[List[float]] = None) -> go.Figure:
    """
    Create progress tracking chart with confidence intervals.

    Args:
        actual_progress: List of actual time gains achieved (optional)

    Returns:
        Plotly Figure with progress curve
    """
    weeks = list(range(0, 7))
    expected_gain = [0, 0.09, 0.18, 0.26, 0.33, 0.39, 0.44]
    lower_bound = [0, 0.05, 0.12, 0.18, 0.24, 0.30, 0.36]
    upper_bound = [0, 0.12, 0.24, 0.34, 0.42, 0.48, 0.52]

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=weeks + weeks[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Range',
        showlegend=True,
        hoverinfo='skip'
    ))

    # Expected progress line
    fig.add_trace(go.Scatter(
        x=weeks,
        y=expected_gain,
        mode='lines+markers',
        line=dict(color=COLORS['current'], width=3),
        marker=dict(size=10),
        name='Expected Progress'
    ))

    # Actual progress (if provided)
    if actual_progress:
        fig.add_trace(go.Scatter(
            x=list(range(len(actual_progress))),
            y=actual_progress,
            mode='markers+lines',
            marker=dict(size=15, color=COLORS['benchmark']),
            line=dict(color=COLORS['benchmark'], width=2, dash='dot'),
            name='Your Actual Progress'
        ))
    else:
        # Placeholder starting point
        fig.add_trace(go.Scatter(
            x=[0],
            y=[0],
            mode='markers',
            marker=dict(size=15, color=COLORS['benchmark']),
            name='Your Actual Progress (Start Here)'
        ))

    # Phase labels
    phase_labels = ['T6 Brake', 'T6 Complete', 'T1 Brake', 'T1 Complete', 'Consistency', 'Race Ready']
    for i, (week, gain) in enumerate(zip(weeks[1:], expected_gain[1:])):
        fig.add_annotation(
            x=week,
            y=gain,
            text=phase_labels[i],
            showarrow=False,
            yshift=20,
            font=dict(size=9)
        )

    fig.update_layout(
        title="Track Your Journey to 0.44s Improvement",
        xaxis_title="Week",
        yaxis_title="Cumulative Time Gain (seconds)",
        height=400,
        hovermode='x unified'
    )

    return fig


# ============================================================================
# SECTION 6: SESSION VISUAL GUIDES
# ============================================================================

def create_turn_visual_guide(turn_name: str,
                             current_line: Dict = None,
                             optimal_line: Dict = None) -> go.Figure:
    """
    Create detailed corner visualization with racing lines.

    Args:
        turn_name: Name of the turn
        current_line: Dict with x, y coordinates for current line
        optimal_line: Dict with x, y coordinates for optimal line

    Returns:
        Plotly Figure with corner visualization
    """
    # Default example data
    if not current_line:
        approach_x = list(range(0, 150, 5))
        approach_y = [100] * len(approach_x)
        current_x = list(range(120, 250, 5))
        current_y = [100 - 0.01 * (x - 120) ** 2 for x in current_x]
        current_line = {'x': approach_x + current_x, 'y': approach_y + current_y}

    if not optimal_line:
        approach_x = list(range(0, 150, 5))
        approach_y = [100] * len(approach_x)
        optimal_x = list(range(132, 250, 5))
        optimal_y = [100 - 0.008 * (x - 132) ** 2 for x in optimal_x]
        optimal_line = {'x': approach_x + optimal_x, 'y': approach_y + optimal_y}

    fig = go.Figure()

    # Current line
    fig.add_trace(go.Scatter(
        x=current_line['x'],
        y=current_line['y'],
        mode='lines',
        line=dict(color=COLORS['opportunity'], width=3, dash='dash'),
        name='Current Line'
    ))

    # Optimal line
    fig.add_trace(go.Scatter(
        x=optimal_line['x'],
        y=optimal_line['y'],
        mode='lines',
        line=dict(color=COLORS['benchmark'], width=3),
        name='Optimal Line'
    ))

    # Reference points
    markers = [
        (120, 100, "Current Brake", COLORS['opportunity']),
        (132, 100, "New Brake", COLORS['benchmark']),
    ]

    for x, y, label, color in markers:
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(size=15, color=color),
            text=[label],
            textposition="top center",
            showlegend=False,
            hoverinfo='text',
            hovertext=label
        ))

    fig.update_layout(
        title=f"{turn_name}: Your New Racing Line",
        xaxis_title="Distance (meters)",
        yaxis_title="Track Position",
        height=500,
        annotations=[
            dict(
                text="12m Earlier Brake = 3.7 km/h Faster Exit",
                x=0.5, y=0.05,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=16, color=COLORS['current'])
            )
        ]
    )

    return fig


def create_brake_pressure_guide(turn_name: str) -> go.Figure:
    """
    Create brake pressure trace comparison.

    Args:
        turn_name: Name of the turn

    Returns:
        Plotly Figure with brake pressure profiles
    """
    distance = list(range(0, 200, 2))

    # Current brake trace (late and hard)
    current_brake = [0] * 60 + [95] * 20 + [60] * 20 + [30] * 20 + [0] * 80

    # Optimal brake trace (early and progressive)
    optimal_brake = [0] * 54 + [75] * 15 + [60] * 15 + [40] * 20 + [20] * 16 + [0] * 80

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=distance,
        y=current_brake,
        mode='lines',
        line=dict(color=COLORS['opportunity'], width=3),
        fill='tozeroy',
        name='Current (Late & Hard)',
        opacity=0.5
    ))

    fig.add_trace(go.Scatter(
        x=distance,
        y=optimal_brake,
        mode='lines',
        line=dict(color=COLORS['benchmark'], width=3),
        fill='tozeroy',
        name='Target (Early & Progressive)',
        opacity=0.5
    ))

    # Trail brake zone
    fig.add_vrect(
        x0=120, x1=140,
        fillcolor="yellow",
        opacity=0.2,
        annotation_text="Trail Brake Zone",
        annotation_position="top"
    )

    fig.update_layout(
        title=f"{turn_name}: Brake Pressure Profile - The Art of Trail Braking",
        xaxis_title="Distance to Apex (meters)",
        yaxis_title="Brake Pressure %",
        height=400,
        annotations=[
            dict(text="Start 12m earlier", x=108, y=80, showarrow=True, ax=-30, ay=0),
            dict(text="20% less pressure", x=120, y=75, showarrow=True, ax=30, ay=0)
        ]
    )

    return fig


# ============================================================================
# SECTION 7: COMPETITIVE ADVANTAGE SUMMARY
# ============================================================================

def create_comprehensive_dashboard(summary_data: Dict) -> go.Figure:
    """
    Create comprehensive multi-chart dashboard overview.

    Args:
        summary_data: Dictionary with all performance metrics

    Returns:
        Plotly Figure with subplots
    """
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Time Loss by Corner', 'Brake vs Exit Correlation',
            'Speed Cascade', 'Consistency Map',
            'Phase Distribution', 'Race Advantage'
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}, {"type": "waterfall"}],
            [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}]
        ]
    )

    # Subplot 1: Time loss bars
    corners = ['T1', 'T6', 'T11']
    time_losses = [0.21, 0.18, 0.05]
    colors_bars = [COLORS['opportunity'], COLORS['opportunity'], COLORS['benchmark']]

    fig.add_trace(
        go.Bar(x=corners, y=time_losses, marker_color=colors_bars, showlegend=False),
        row=1, col=1
    )

    # Subplot 2: Correlation
    brake_excess = [0.62, 0.91, -0.45]
    exit_deficit = [-0.62, -0.91, 0.45]

    fig.add_trace(
        go.Scatter(
            x=brake_excess, y=exit_deficit,
            mode='markers',
            marker=dict(size=20, color=colors_bars),
            showlegend=False
        ),
        row=1, col=2
    )

    # Subplot 3: Waterfall
    phases = ['Entry', 'Apex', 'Exit', 'Straight']
    cascade_vals = [-3.6, -3.2, -3.7, -5.0]

    fig.add_trace(
        go.Waterfall(
            x=phases,
            y=cascade_vals,
            connector={"line": {"color": "gray"}},
            decreasing={"marker": {"color": COLORS['opportunity']}},
            showlegend=False
        ),
        row=1, col=3
    )

    # Subplot 4: Consistency scatter
    consistency = [87.5, 83.7, 94.8]
    efficiency = [78.9, 82.0, 95.0]

    fig.add_trace(
        go.Scatter(
            x=consistency, y=efficiency,
            mode='markers',
            marker=dict(size=[210, 180, 50], color=colors_bars),
            showlegend=False
        ),
        row=2, col=1
    )

    # Subplot 5: Phase distribution
    phase_corners = ['T1', 'T6', 'T11']
    brake_phase = [35, 35, 25]
    exit_phase = [45, 45, 55]

    fig.add_trace(
        go.Bar(x=phase_corners, y=brake_phase, name='Brake', marker_color=COLORS['opportunity']),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(x=phase_corners, y=exit_phase, name='Exit', marker_color=COLORS['benchmark']),
        row=2, col=2
    )

    # Subplot 6: Race advantage
    laps = list(range(1, 21))
    cumulative = [0.44 * lap for lap in laps]

    fig.add_trace(
        go.Scatter(x=laps, y=cumulative, mode='lines', line=dict(color=COLORS['current'], width=2), showlegend=False),
        row=2, col=3
    )

    fig.update_layout(
        height=800,
        title_text="Your Complete Race-Winning Dashboard",
        title_font_size=24,
        showlegend=True
    )

    # Update axis labels
    fig.update_xaxes(title_text="Corner", row=1, col=1)
    fig.update_yaxes(title_text="Time Loss (s)", row=1, col=1)

    fig.update_xaxes(title_text="Brake Excess (s)", row=1, col=2)
    fig.update_yaxes(title_text="Exit Deficit (s)", row=1, col=2)

    fig.update_xaxes(title_text="Phase", row=1, col=3)
    fig.update_yaxes(title_text="Speed Loss (km/h)", row=1, col=3)

    fig.update_xaxes(title_text="Consistency %", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency %", row=2, col=1)

    fig.update_xaxes(title_text="Corner", row=2, col=2)
    fig.update_yaxes(title_text="% of Corner", row=2, col=2)

    fig.update_xaxes(title_text="Lap", row=2, col=3)
    fig.update_yaxes(title_text="Time Gain (s)", row=2, col=3)

    return fig


# ============================================================================
# LAYOUT AND CALLBACKS
# ============================================================================

def create_winning_edge_layout() -> html.Div:
    """
    Create the complete Winning Edge widget layout.

    Returns:
        Dash HTML Div component with full dashboard layout
    """
    layout = html.Div([
        # Header
        html.Div([
            html.H1([
                "🏁 The Winning Edge Dashboard",
                html.Span(" - Race-Day Decision Support System",
                         style={'fontSize': '20px', 'fontWeight': 'normal', 'color': '#7f8c8d'})
            ], style={
                'textAlign': 'center',
                'color': COLORS['text'],
                'marginBottom': '10px',
                'marginTop': '20px'
            }),
            html.P([
                "Visual-first approach for rapid decision-making | ",
                html.Strong("Target Audience: "),
                "Driver + Race Engineer | ",
                html.Strong("Key Advantage: "),
                "Instant visual pattern recognition"
            ], style={
                'textAlign': 'center',
                'color': '#7f8c8d',
                'marginBottom': '30px',
                'fontSize': '14px'
            })
        ]),

        # Tab structure for different sections
        dbc.Tabs([
            # ===== TAB 1: RACE WINNER'S DASHBOARD =====
            dbc.Tab(label="1. Race Winner's Dashboard", tab_id="winning-edge-tab-1", children=[
                html.Div([
                    html.H3("Your 0.44s Advantage Visualized",
                           style={'textAlign': 'center', 'marginTop': '20px', 'color': COLORS['text']}),
                    html.Hr(),

                    # Time Loss Heatmap
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("🔥 Time Loss Heat Map",
                                             style={'backgroundColor': COLORS['background'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="winning-edge-loading-heatmap",
                                        type="default",
                                        children=[dcc.Graph(id='winning-edge-heatmap')]
                                    )
                                ])
                            ], className='mb-4')
                        ], width=12)
                    ]),

                    # Speed Gap Spider Chart
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("🕸️ Speed Gap Spider Chart",
                                             style={'backgroundColor': COLORS['background'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    html.P([
                                        html.Strong("How to Read: "),
                                        "GREEN (Turn 11) = Your target shape - you CAN do this! | ",
                                        "RED (Turn 6) = Biggest opportunity | ",
                                        "BLUE (Turn 1) = Second priority | ",
                                        html.Strong("Goal: "),
                                        "Make red and blue match green shape"
                                    ], style={'fontSize': '12px', 'marginBottom': '15px'}),
                                    dcc.Loading(
                                        id="winning-edge-loading-spider",
                                        type="default",
                                        children=[dcc.Graph(id='winning-edge-spider')]
                                    )
                                ])
                            ], className='mb-4')
                        ], width=12)
                    ])
                ], style={'padding': '20px'})
            ]),

            # ===== TAB 2: CORRELATION DASHBOARD =====
            dbc.Tab(label="2. Correlation Dashboard", tab_id="winning-edge-tab-2", children=[
                html.Div([
                    html.H3("Understanding What Drives What",
                           style={'textAlign': 'center', 'marginTop': '20px', 'color': COLORS['text']}),
                    html.Hr(),

                    # Brake-Exit Correlation
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("🎯 1:1 Brake-Exit Correlation (Game Changer!)",
                                             style={'backgroundColor': COLORS['background'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    dbc.Alert([
                                        html.Strong("Mathematical Fact: "),
                                        "1:1 trade-off between braking and accelerating | ",
                                        html.Strong("Turn 11 Green Dot: "),
                                        "Shows you CAN brake efficiently | ",
                                        html.Strong("Action: "),
                                        "Brake 0.62s less in Turn 1 → Gain 0.62s of acceleration"
                                    ], color="info", style={'fontSize': '12px'}),
                                    dcc.Loading(
                                        id="winning-edge-loading-correlation",
                                        type="default",
                                        children=[dcc.Graph(id='winning-edge-correlation')]
                                    )
                                ])
                            ], className='mb-4')
                        ], width=12)
                    ]),

                    # Speed Cascade & Consistency Matrix
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("📉 Speed Cascade Waterfall",
                                             style={'backgroundColor': COLORS['background'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    html.P("How 3.6 km/h entry loss becomes 15.5 km/h disadvantage",
                                          style={'fontSize': '12px', 'fontStyle': 'italic'}),
                                    dcc.Loading(
                                        id="winning-edge-loading-cascade",
                                        type="default",
                                        children=[dcc.Graph(id='winning-edge-cascade')]
                                    )
                                ])
                            ], className='mb-4')
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("📊 Consistency vs Performance Matrix",
                                             style={'backgroundColor': COLORS['background'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    html.P("Bubble size = time loss (make bubbles smaller!)",
                                          style={'fontSize': '12px', 'fontStyle': 'italic'}),
                                    dcc.Loading(
                                        id="winning-edge-loading-consistency",
                                        type="default",
                                        children=[dcc.Graph(id='winning-edge-consistency')]
                                    )
                                ])
                            ], className='mb-4')
                        ], width=6)
                    ])
                ], style={'padding': '20px'})
            ]),

            # ===== TAB 3: PRIORITY ACTION CARDS =====
            dbc.Tab(label="3. Action Cards", tab_id="winning-edge-tab-3", children=[
                html.Div([
                    html.H3("Your Race-Winning Checklist",
                           style={'textAlign': 'center', 'marginTop': '20px', 'color': COLORS['text']}),
                    html.Hr(),

                    # Turn selector
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Turn for Action Card:", style={'fontWeight': 'bold'}),
                            dcc.Dropdown(
                                id='winning-edge-turn-selector',
                                options=[
                                    {'label': 'Turn 1 - Priority 2', 'value': 'Turn 1'},
                                    {'label': 'Turn 6 - Priority 1 (Highest Opportunity)', 'value': 'Turn 6'},
                                    {'label': 'Turn 11 - Benchmark', 'value': 'Turn 11'}
                                ],
                                value='Turn 6',
                                style={'marginBottom': '20px'}
                            )
                        ], width=6)
                    ]),

                    # Action Card
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(id='winning-edge-action-card-header',
                                             style={'backgroundColor': COLORS['background'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    html.Div(id='winning-edge-action-card-instructions',
                                            style={'marginBottom': '15px'}),
                                    dcc.Loading(
                                        id="winning-edge-loading-action-card",
                                        type="default",
                                        children=[dcc.Graph(id='winning-edge-action-card')]
                                    )
                                ])
                            ], className='mb-4')
                        ], width=12)
                    ]),

                    # Phase Distribution
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("⏱️ Phase Distribution Fix",
                                             style={'backgroundColor': COLORS['background'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    dbc.Alert([
                                        html.Strong("The Golden Rule: "),
                                        "🔴 Red (Brake): Minimize to 30% | ",
                                        "🟡 Yellow (Apex): Keep at 20% | ",
                                        "🟢 Green (Exit): Maximize to 50%+"
                                    ], color="success", style={'fontSize': '12px'}),
                                    dcc.Loading(
                                        id="winning-edge-loading-phase",
                                        type="default",
                                        children=[dcc.Graph(id='winning-edge-phase')]
                                    )
                                ])
                            ], className='mb-4')
                        ], width=12)
                    ])
                ], style={'padding': '20px'})
            ]),

            # ===== TAB 4: RACE SIMULATION =====
            dbc.Tab(label="4. Race Simulation", tab_id="winning-edge-tab-4", children=[
                html.Div([
                    html.H3("What These Changes Mean for Your Race",
                           style={'textAlign': 'center', 'marginTop': '20px', 'color': COLORS['text']}),
                    html.Hr(),

                    # Race parameters
                    dbc.Row([
                        dbc.Col([
                            html.Label("Time Gain Per Lap (seconds):", style={'fontWeight': 'bold'}),
                            dcc.Input(
                                id='winning-edge-time-gain-input',
                                type='number',
                                value=0.44,
                                min=0,
                                max=2,
                                step=0.01,
                                style={'width': '100%', 'marginBottom': '10px'}
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Number of Race Laps:", style={'fontWeight': 'bold'}),
                            dcc.Input(
                                id='winning-edge-num-laps-input',
                                type='number',
                                value=20,
                                min=5,
                                max=50,
                                step=1,
                                style={'width': '100%', 'marginBottom': '10px'}
                            )
                        ], width=3)
                    ]),

                    # Position Gain Predictor
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("🏆 Position Gain Predictor",
                                             style={'backgroundColor': COLORS['background'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="winning-edge-loading-position",
                                        type="default",
                                        children=[dcc.Graph(id='winning-edge-position-gain')]
                                    )
                                ])
                            ], className='mb-4')
                        ], width=12)
                    ]),

                    # Overtaking Map
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("🗺️ Overtaking Opportunity Map",
                                             style={'backgroundColor': COLORS['background'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    html.P([
                                        html.Strong("Your Overtaking Advantage: "),
                                        "Turn 6 Exit: +3.7 km/h = 2-3 car lengths | ",
                                        "Turn 1 Exit: +3.3 km/h = 1-2 car lengths | ",
                                        "Defense: Better exits make you impossible to pass!"
                                    ], style={'fontSize': '12px', 'marginBottom': '15px'}),
                                    dcc.Loading(
                                        id="winning-edge-loading-overtaking",
                                        type="default",
                                        children=[dcc.Graph(id='winning-edge-overtaking')]
                                    )
                                ])
                            ], className='mb-4')
                        ], width=12)
                    ])
                ], style={'padding': '20px'})
            ]),

            # ===== TAB 5: TRANSFORMATION TIMELINE =====
            dbc.Tab(label="5. 6-Week Timeline", tab_id="winning-edge-tab-5", children=[
                html.Div([
                    html.H3("Visual Progress Tracking System",
                           style={'textAlign': 'center', 'marginTop': '20px', 'color': COLORS['text']}),
                    html.Hr(),

                    # Weekly Timeline
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("📅 Weekly Target Progression",
                                             style={'backgroundColor': COLORS['background'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="winning-edge-loading-timeline",
                                        type="default",
                                        children=[dcc.Graph(id='winning-edge-timeline')]
                                    )
                                ])
                            ], className='mb-4')
                        ], width=12)
                    ]),

                    # Improvement Curve
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("📈 Expected Improvement Curve",
                                             style={'backgroundColor': COLORS['background'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    html.P("Track your actual progress against expected gains with confidence intervals",
                                          style={'fontSize': '12px', 'fontStyle': 'italic'}),
                                    dcc.Loading(
                                        id="winning-edge-loading-curve",
                                        type="default",
                                        children=[dcc.Graph(id='winning-edge-curve')]
                                    )
                                ])
                            ], className='mb-4')
                        ], width=12)
                    ])
                ], style={'padding': '20px'})
            ]),

            # ===== TAB 6: VISUAL GUIDES =====
            dbc.Tab(label="6. Session Guides", tab_id="winning-edge-tab-6", children=[
                html.Div([
                    html.H3("Practical Track Maps for Your Next Session",
                           style={'textAlign': 'center', 'marginTop': '20px', 'color': COLORS['text']}),
                    html.Hr(),

                    # Turn selector for visual guide
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Turn for Visual Guide:", style={'fontWeight': 'bold'}),
                            dcc.Dropdown(
                                id='winning-edge-visual-turn-selector',
                                options=[
                                    {'label': 'Turn 6 Hairpin - Primary Target', 'value': 'Turn 6'},
                                    {'label': 'Turn 1 Uphill - Secondary Target', 'value': 'Turn 1'},
                                    {'label': 'Turn 11 Fast - Benchmark', 'value': 'Turn 11'}
                                ],
                                value='Turn 6',
                                style={'marginBottom': '20px'}
                            )
                        ], width=6)
                    ]),

                    # Racing Line Visual
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(id='winning-edge-visual-guide-header',
                                             style={'backgroundColor': COLORS['background'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="winning-edge-loading-visual",
                                        type="default",
                                        children=[dcc.Graph(id='winning-edge-visual-guide')]
                                    )
                                ])
                            ], className='mb-4')
                        ], width=12)
                    ]),

                    # Brake Pressure Profile
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(id='winning-edge-brake-guide-header',
                                             style={'backgroundColor': COLORS['background'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="winning-edge-loading-brake",
                                        type="default",
                                        children=[dcc.Graph(id='winning-edge-brake-guide')]
                                    )
                                ])
                            ], className='mb-4')
                        ], width=12)
                    ])
                ], style={'padding': '20px'})
            ]),

            # ===== TAB 7: SUMMARY DASHBOARD =====
            dbc.Tab(label="7. Summary", tab_id="winning-edge-tab-7", children=[
                html.Div([
                    html.H3("Your Personal Race-Winning Dashboard",
                           style={'textAlign': 'center', 'marginTop': '20px', 'color': COLORS['text']}),
                    html.Hr(),

                    # Champion's Mindset Card
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert([
                                html.H4("🏆 THE CHAMPION'S MINDSET", style={'marginBottom': '15px'}),
                                html.P([
                                    html.Strong("YOUR COMPETITIVE ADVANTAGES:"), html.Br(),
                                    "1️⃣ You've Already Done It: 'Your Best' proves capability", html.Br(),
                                    "2️⃣ Clear Mathematical Edge: 0.44s is 4 grid positions", html.Br(),
                                    "3️⃣ Simple Focus: 89% of gains in just 2 corners", html.Br(),
                                    "4️⃣ Proven Template: Turn 11 shows the way", html.Br(),
                                    "5️⃣ Compound Benefits: Each improvement multiplies", html.Br(), html.Br(),
                                    html.Strong("THE WINNING FORMULA:", style={'fontSize': '16px'}), html.Br(),
                                    html.Code("Less Brake Time + Earlier Throttle = 0.44s/lap",
                                             style={'fontSize': '14px'}), html.Br(),
                                    html.Code("0.44s × 20 laps = 8.8 seconds = PODIUM",
                                             style={'fontSize': '14px'})
                                ], style={'marginBottom': '0'})
                            ], color="success", className='mb-4')
                        ], width=12)
                    ]),

                    # Comprehensive Dashboard
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("📊 Complete Performance Overview",
                                             style={'backgroundColor': COLORS['background'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="winning-edge-loading-summary",
                                        type="default",
                                        children=[dcc.Graph(id='winning-edge-summary')]
                                    )
                                ])
                            ], className='mb-4')
                        ], width=12)
                    ])
                ], style={'padding': '20px'})
            ])
        ], id='winning-edge-tabs', active_tab='winning-edge-tab-1')
    ], style={'backgroundColor': 'white', 'minHeight': '100vh'})

    return layout


def create_winning_edge_callbacks(app):
    """
    Register all callbacks for the Winning Edge widget.

    Args:
        app: Dash app instance
    """

    # ===== SECTION 1 CALLBACKS =====
    @app.callback(
        Output('winning-edge-heatmap', 'figure'),
        Input('winning-edge-tabs', 'active_tab')
    )
    def update_heatmap(active_tab):
        """Update time loss heatmap when tab is activated."""
        if active_tab == 'winning-edge-tab-1':
            return create_time_loss_heatmap({})
        return go.Figure()

    @app.callback(
        Output('winning-edge-spider', 'figure'),
        Input('winning-edge-tabs', 'active_tab')
    )
    def update_spider(active_tab):
        """Update speed gap spider chart when tab is activated."""
        if active_tab == 'winning-edge-tab-1':
            return create_speed_gap_spider({})
        return go.Figure()

    # ===== SECTION 2 CALLBACKS =====
    @app.callback(
        Output('winning-edge-correlation', 'figure'),
        Input('winning-edge-tabs', 'active_tab')
    )
    def update_correlation(active_tab):
        """Update brake-exit correlation chart."""
        if active_tab == 'winning-edge-tab-2':
            return create_brake_exit_correlation([], [], [])
        return go.Figure()

    @app.callback(
        Output('winning-edge-cascade', 'figure'),
        Input('winning-edge-tabs', 'active_tab')
    )
    def update_cascade(active_tab):
        """Update speed cascade waterfall chart."""
        if active_tab == 'winning-edge-tab-2':
            return create_speed_cascade_waterfall('Turn 6', {})
        return go.Figure()

    @app.callback(
        Output('winning-edge-consistency', 'figure'),
        Input('winning-edge-tabs', 'active_tab')
    )
    def update_consistency(active_tab):
        """Update consistency matrix chart."""
        if active_tab == 'winning-edge-tab-2':
            return create_consistency_performance_matrix([])
        return go.Figure()

    # ===== SECTION 3 CALLBACKS =====
    @app.callback(
        Output('winning-edge-action-card-header', 'children'),
        Output('winning-edge-action-card-instructions', 'children'),
        Output('winning-edge-action-card', 'figure'),
        Input('winning-edge-turn-selector', 'value')
    )
    def update_action_card(turn_name):
        """Update action card based on selected turn."""
        instructions_map = {
            'Turn 6': html.Div([
                html.P([
                    html.Strong("🎯 Your Turn 6 Mission:"), html.Br(),
                    "1️⃣ BRAKE: Move from 120m → 132m marker (+12m earlier)", html.Br(),
                    "2️⃣ PRESSURE: Reduce from 95% → 75% (-20% lighter)", html.Br(),
                    "3️⃣ THROTTLE: Apply at 52.3% → 57.5% (+5.2% earlier)", html.Br(),
                    "4️⃣ RESULT: Exit at 102.4 km/h (+3.7 km/h = overtaking advantage!)"
                ], style={'fontSize': '12px'})
            ]),
            'Turn 1': html.Div([
                html.P([
                    html.Strong("🎯 Your Turn 1 Mission:"), html.Br(),
                    "Similar brake/throttle optimization as Turn 6", html.Br(),
                    "Focus on early brake point and progressive pressure application"
                ], style={'fontSize': '12px'})
            ]),
            'Turn 11': html.Div([
                html.P([
                    html.Strong("✅ Turn 11 - Your Benchmark:"), html.Br(),
                    "This is your target performance level!", html.Br(),
                    "Use this corner as template for Turn 1 and Turn 6"
                ], style={'fontSize': '12px'})
            ])
        }

        header = f"🏁 {turn_name} Transformation Card"
        instructions = instructions_map.get(turn_name, html.Div())
        figure = create_turn_action_card(turn_name, {}, {})

        return header, instructions, figure

    @app.callback(
        Output('winning-edge-phase', 'figure'),
        Input('winning-edge-tabs', 'active_tab')
    )
    def update_phase_distribution(active_tab):
        """Update phase distribution chart."""
        if active_tab == 'winning-edge-tab-3':
            return create_phase_distribution({})
        return go.Figure()

    # ===== SECTION 4 CALLBACKS =====
    @app.callback(
        Output('winning-edge-position-gain', 'figure'),
        Input('winning-edge-time-gain-input', 'value'),
        Input('winning-edge-num-laps-input', 'value')
    )
    def update_position_gain(time_gain, num_laps):
        """Update position gain predictor based on inputs."""
        if time_gain and num_laps:
            return create_position_gain_predictor(time_gain, num_laps)
        return go.Figure()

    @app.callback(
        Output('winning-edge-overtaking', 'figure'),
        Input('winning-edge-tabs', 'active_tab')
    )
    def update_overtaking_map(active_tab):
        """Update overtaking opportunity map."""
        if active_tab == 'winning-edge-tab-4':
            return create_overtaking_opportunity_map()
        return go.Figure()

    # ===== SECTION 5 CALLBACKS =====
    @app.callback(
        Output('winning-edge-timeline', 'figure'),
        Input('winning-edge-tabs', 'active_tab')
    )
    def update_timeline(active_tab):
        """Update weekly target progression timeline."""
        if active_tab == 'winning-edge-tab-5':
            return create_weekly_target_progression()
        return go.Figure()

    @app.callback(
        Output('winning-edge-curve', 'figure'),
        Input('winning-edge-tabs', 'active_tab')
    )
    def update_improvement_curve(active_tab):
        """Update improvement curve with progress tracking."""
        if active_tab == 'winning-edge-tab-5':
            return create_improvement_curve()
        return go.Figure()

    # ===== SECTION 6 CALLBACKS =====
    @app.callback(
        Output('winning-edge-visual-guide-header', 'children'),
        Output('winning-edge-visual-guide', 'figure'),
        Input('winning-edge-visual-turn-selector', 'value')
    )
    def update_visual_guide(turn_name):
        """Update turn visual guide based on selection."""
        header = f"🏁 {turn_name}: Your New Racing Line"
        figure = create_turn_visual_guide(turn_name)
        return header, figure

    @app.callback(
        Output('winning-edge-brake-guide-header', 'children'),
        Output('winning-edge-brake-guide', 'figure'),
        Input('winning-edge-visual-turn-selector', 'value')
    )
    def update_brake_guide(turn_name):
        """Update brake pressure guide based on selection."""
        header = f"🛑 {turn_name}: Brake Pressure Visualization"
        figure = create_brake_pressure_guide(turn_name)
        return header, figure

    # ===== SECTION 7 CALLBACKS =====
    @app.callback(
        Output('winning-edge-summary', 'figure'),
        Input('winning-edge-tabs', 'active_tab')
    )
    def update_summary_dashboard(active_tab):
        """Update comprehensive summary dashboard."""
        if active_tab == 'winning-edge-tab-7':
            return create_comprehensive_dashboard({})
        return go.Figure()

    logger.info("Winning Edge widget callbacks registered successfully")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("This module is meant to be imported, not run directly.")
    print("Import it in your Dash app with:")
    print("  from src.dashboard.winning_edge_widget import create_winning_edge_layout, create_winning_edge_callbacks")
