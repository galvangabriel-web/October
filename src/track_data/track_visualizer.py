"""
Track Map Visualizer with Telemetry Overlay
============================================

Advanced visualization module for overlaying telemetry data on track maps.

Features:
- Speed trace visualization
- Braking zone highlighting
- Throttle application display
- Sector performance overlay
- Interactive plotly-based maps

Usage:
    from src.track_data.track_visualizer import TrackVisualizer

    visualizer = TrackVisualizer(track_name='barber-motorsports-park')
    fig = visualizer.create_track_overlay(telemetry_df, vehicle_number=5)
    fig.show()
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import base64
from PIL import Image
import io


class TrackVisualizer:
    """Create interactive track maps with telemetry overlay"""

    def __init__(self, track_name: str):
        """
        Initialize track visualizer

        Args:
            track_name: Track identifier (e.g., 'barber-motorsports-park')
        """
        self.track_name = track_name
        self.track_metadata = None
        self.track_image = None
        self._load_track_data()

    def _load_track_data(self):
        """Load track metadata and image"""
        try:
            from src.track_data.track_metadata import get_track_metadata
            from src.track_data.track_images import TrackImageLoader

            self.track_metadata = get_track_metadata(self.track_name)

            # Load track image
            loader = TrackImageLoader()
            self.track_image = loader.get_track_image(self.track_name, page=1, hd=True)
        except Exception as e:
            print(f"Error loading track data: {e}")

    def create_base_figure(self, width: int = 1400, height: int = 900) -> go.Figure:
        """
        Create base plotly figure with track map background

        Args:
            width: Figure width (default: 1400 for full-width dashboard layout)
            height: Figure height (default: 900 to show full track including tables)

        Returns:
            Plotly figure with track map as background
        """
        if self.track_image is None:
            # Create empty figure if no image
            fig = go.Figure()
            fig.update_layout(
                title=f"Track Map - {self.track_name}",
                width=width,
                height=height,
                autosize=True
            )
            return fig

        # Convert PIL image to base64 for plotly
        buffered = io.BytesIO()
        self.track_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Create figure with image background
        fig = go.Figure()

        # Add invisible scatter to set up axes
        fig.add_trace(go.Scatter(
            x=[0, self.track_image.width],
            y=[0, self.track_image.height],
            mode='markers',
            marker=dict(size=0.1, opacity=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add track map as background image
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{img_base64}",
                xref="x",
                yref="y",
                x=0,
                y=self.track_image.height,
                sizex=self.track_image.width,
                sizey=self.track_image.height,
                sizing="stretch",
                opacity=1,
                layer="below"
            )
        )

        # Configure layout
        track_title = self.track_metadata['name'] if self.track_metadata else self.track_name
        fig.update_layout(
            title=f"{track_title} - Track Map Visualization",
            width=width,
            height=height,
            autosize=True,  # Enable responsive resizing
            xaxis=dict(
                range=[0, self.track_image.width],
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            yaxis=dict(
                range=[0, self.track_image.height],
                showgrid=False,
                zeroline=False,
                visible=False,
                scaleanchor="x"
            ),
            plot_bgcolor='white',
            margin=dict(l=10, r=150, t=50, b=10),  # Increased right margin for legend/colorbar
            legend=dict(
                x=1.02,
                y=0.5,
                xanchor='left',
                yanchor='middle',
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            )
        )

        return fig

    def add_speed_trace(
        self,
        fig: go.Figure,
        telemetry_df: pd.DataFrame,
        vehicle_number: int,
        lap: Optional[int] = None
    ) -> go.Figure:
        """
        Add speed trace overlay to track map

        Args:
            fig: Base plotly figure
            telemetry_df: Telemetry DataFrame
            vehicle_number: Vehicle to visualize
            lap: Specific lap (optional)

        Returns:
            Updated figure with speed trace
        """
        # Filter telemetry for vehicle
        vehicle_df = telemetry_df[telemetry_df['vehicle_number'] == vehicle_number]

        if lap is not None:
            vehicle_df = vehicle_df[vehicle_df['lap'] == lap]

        # Get speed data
        speed_data = vehicle_df[vehicle_df['telemetry_name'] == 'speed'].copy()

        if len(speed_data) == 0:
            return fig

        # Normalize speed for color mapping
        speed_values = speed_data['telemetry_value'].values
        speed_norm = (speed_values - speed_values.min()) / (speed_values.max() - speed_values.min())

        # Create synthetic track coordinates (placeholder)
        # In real implementation, would use GPS or track position data
        num_points = len(speed_data)
        t = np.linspace(0, 2 * np.pi, num_points)

        # Create oval track shape as example
        center_x = self.track_image.width / 2 if self.track_image else 500
        center_y = self.track_image.height / 2 if self.track_image else 400
        radius_x = center_x * 0.7
        radius_y = center_y * 0.6

        x_coords = center_x + radius_x * np.cos(t)
        y_coords = center_y + radius_y * np.sin(t)

        # Add speed trace
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=3,
                color=speed_values,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Speed<br>(km/h)",
                        side="right"
                    ),
                    x=1.15,
                    y=0.5,
                    yanchor='middle',
                    len=0.6,
                    thickness=15,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='rgba(0, 0, 0, 0.2)',
                    borderwidth=1
                ),
                cmin=speed_values.min(),
                cmax=speed_values.max()
            ),
            name=f"Vehicle {vehicle_number}",
            hovertemplate="Speed: %{marker.color:.1f} km/h<extra></extra>"
        ))

        return fig

    def add_braking_zones(
        self,
        fig: go.Figure,
        telemetry_df: pd.DataFrame,
        vehicle_number: int,
        threshold: float = 50.0
    ) -> go.Figure:
        """
        Highlight braking zones on track map

        Args:
            fig: Base plotly figure
            telemetry_df: Telemetry DataFrame
            vehicle_number: Vehicle to visualize
            threshold: Brake pressure threshold

        Returns:
            Updated figure with braking zones
        """
        # Filter telemetry
        vehicle_df = telemetry_df[telemetry_df['vehicle_number'] == vehicle_number]

        # Get brake data
        brake_data = vehicle_df[vehicle_df['telemetry_name'] == 'pbrake_f'].copy()

        if len(brake_data) == 0:
            return fig

        # Find hard braking zones
        hard_braking = brake_data[brake_data['telemetry_value'] > threshold]

        if len(hard_braking) > 0:
            # Create synthetic positions for braking zones
            num_zones = min(5, len(hard_braking) // 10)  # Example: up to 5 zones

            if self.track_image:
                # Place braking zones at strategic points
                zone_positions = [
                    (self.track_image.width * 0.2, self.track_image.height * 0.3),
                    (self.track_image.width * 0.8, self.track_image.height * 0.3),
                    (self.track_image.width * 0.8, self.track_image.height * 0.7),
                    (self.track_image.width * 0.2, self.track_image.height * 0.7),
                    (self.track_image.width * 0.5, self.track_image.height * 0.5),
                ]
            else:
                zone_positions = [(100, 100), (400, 100), (400, 300), (100, 300), (250, 200)]

            for i in range(num_zones):
                x, y = zone_positions[i]
                fig.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers',
                    marker=dict(
                        size=30,
                        color='red',
                        opacity=0.5,
                        symbol='square'
                    ),
                    name="Braking Zones",
                    showlegend=(i == 0),
                    legendgroup="braking",
                    hovertemplate=f"Braking Zone {i+1}<extra></extra>"
                ))

        return fig

    def add_sector_boundaries(self, fig: go.Figure) -> go.Figure:
        """
        Add sector boundary lines to track map

        Args:
            fig: Base plotly figure

        Returns:
            Updated figure with sector boundaries
        """
        if not self.track_metadata or not self.track_image:
            return fig

        # Define sector boundary positions (example)
        # In real implementation, would use actual track coordinates
        boundaries = [
            {'name': 'S1/S2', 'x': [0, self.track_image.width], 'y': [self.track_image.height * 0.33] * 2},
            {'name': 'S2/S3', 'x': [0, self.track_image.width], 'y': [self.track_image.height * 0.66] * 2}
        ]

        for i, boundary in enumerate(boundaries):
            fig.add_trace(go.Scatter(
                x=boundary['x'],
                y=boundary['y'],
                mode='lines',
                line=dict(color='yellow', width=2, dash='dash'),
                name="Sector Boundaries" if i == 0 else boundary['name'],
                showlegend=(i == 0),  # Only show first boundary in legend
                legendgroup="sectors",
                hovertemplate=f"{boundary['name']}<extra></extra>"
            ))

        return fig

    def create_track_overlay(
        self,
        telemetry_df: pd.DataFrame,
        vehicle_number: int,
        lap: Optional[int] = None,
        show_speed: bool = True,
        show_braking: bool = True,
        show_sectors: bool = True
    ) -> go.Figure:
        """
        Create complete track visualization with telemetry overlay

        Args:
            telemetry_df: Telemetry DataFrame
            vehicle_number: Vehicle to visualize
            lap: Specific lap (optional)
            show_speed: Show speed trace
            show_braking: Show braking zones
            show_sectors: Show sector boundaries

        Returns:
            Complete plotly figure with overlays
        """
        # Create base figure
        fig = self.create_base_figure()

        # Add overlays based on options
        if show_sectors:
            fig = self.add_sector_boundaries(fig)

        if show_speed:
            fig = self.add_speed_trace(fig, telemetry_df, vehicle_number, lap)

        if show_braking:
            fig = self.add_braking_zones(fig, telemetry_df, vehicle_number)

        # Add title with lap info
        lap_text = f"Lap {lap}" if lap else "All Laps"
        title = f"{self.track_metadata['name'] if self.track_metadata else self.track_name} - Vehicle {vehicle_number} - {lap_text}"
        fig.update_layout(title=title)

        return fig

    def create_sector_performance_chart(
        self,
        telemetry_df: pd.DataFrame,
        vehicle_number: int
    ) -> go.Figure:
        """
        Create sector performance comparison chart

        Args:
            telemetry_df: Telemetry DataFrame
            vehicle_number: Vehicle number

        Returns:
            Plotly figure with sector performance
        """
        if not self.track_metadata:
            return go.Figure()

        # Create example sector times (would calculate from telemetry in real implementation)
        sectors = ['Sector 1', 'Sector 2', 'Sector 3']

        # Example data - would be calculated from actual telemetry
        best_times = [28.5, 42.3, 31.2]
        current_times = [29.1, 43.1, 31.8]

        fig = go.Figure()

        # Add bars for best times
        fig.add_trace(go.Bar(
            name='Best Time',
            x=sectors,
            y=best_times,
            marker_color='green',
            opacity=0.7
        ))

        # Add bars for current times
        fig.add_trace(go.Bar(
            name='Current Lap',
            x=sectors,
            y=current_times,
            marker_color='blue',
            opacity=0.7
        ))

        fig.update_layout(
            title=f"Sector Performance - Vehicle {vehicle_number}",
            xaxis_title="Sector",
            yaxis_title="Time (seconds)",
            barmode='group',
            height=400
        )

        return fig


# ============================================================================
# Convenience Functions
# ============================================================================

def create_track_visualization(
    track_name: str,
    telemetry_df: pd.DataFrame,
    vehicle_number: int,
    lap: Optional[int] = None
) -> go.Figure:
    """
    Quick function to create track visualization

    Args:
        track_name: Track identifier
        telemetry_df: Telemetry DataFrame
        vehicle_number: Vehicle to visualize
        lap: Specific lap (optional)

    Returns:
        Plotly figure with track overlay

    Example:
        >>> from src.track_data.track_visualizer import create_track_visualization
        >>> fig = create_track_visualization('barber-motorsports-park', df, 5)
        >>> fig.show()
    """
    visualizer = TrackVisualizer(track_name)
    return visualizer.create_track_overlay(telemetry_df, vehicle_number, lap)


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Track Visualizer - Demo")
    print("=" * 60)

    # Create sample telemetry data
    sample_data = pd.DataFrame({
        'vehicle_number': [5] * 100,
        'lap': [1] * 50 + [2] * 50,
        'telemetry_name': ['speed'] * 50 + ['pbrake_f'] * 50,
        'telemetry_value': np.random.uniform(100, 180, 50).tolist() +
                          np.random.uniform(0, 100, 50).tolist()
    })

    # Create visualizer
    visualizer = TrackVisualizer('barber-motorsports-park')

    # Create visualization
    fig = visualizer.create_track_overlay(sample_data, vehicle_number=5, lap=1)

    print("Visualization created successfully")
    print("Use fig.show() to display in browser")
    print("=" * 60)