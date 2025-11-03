"""
Track Animation Engine
======================

Creates animated track visualizations with telemetry data overlays.

Usage:
    from src.track_data.track_animator import TrackAnimator

    animator = TrackAnimator('barber-motorsports-park')
    fig = animator.create_plotly_animation(telemetry_df, vehicle=5)
    fig.show()
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d

from src.track_data.track_metadata import get_track_metadata


class TrackAnimator:
    """Create animated track visualizations with telemetry overlays"""

    def __init__(self, track_name: str):
        """
        Initialize track animator

        Args:
            track_name: Track identifier (e.g., 'barber-motorsports-park')
        """
        self.track_name = track_name
        self.metadata = get_track_metadata(track_name)

        if not self.metadata:
            raise ValueError(f"Track metadata not found: {track_name}")

        # Load animation path
        path_file = Path(f"track_maps/animations/{track_name}_path.json")
        if not path_file.exists():
            raise FileNotFoundError(
                f"Animation path not found: {path_file}\n"
                f"Run: python scripts/vectorization/extract_track_path_from_telemetry.py --track {track_name}"
            )

        with open(path_file) as f:
            self.path_data = json.load(f)

        # Load track image
        self.image_path = Path(f"track_maps/images/{track_name}_page_1_hd.png")
        if not self.image_path.exists():
            raise FileNotFoundError(f"Track image not found: {self.image_path}")

        print(f"Loaded track animator for {self.metadata['name']}")
        print(f"  Animation points: {len(self.path_data['pixel_path'])}")
        print(f"  Track length: {self.path_data['path_info']['total_length_meters']:.1f}m")

    def sync_telemetry_to_path(
        self,
        telemetry: pd.DataFrame,
        vehicle_number: int,
        lap_number: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Synchronize telemetry data to animation path

        Args:
            telemetry: Telemetry data (long or wide format)
            vehicle_number: Vehicle to animate
            lap_number: Specific lap (None = best lap)

        Returns:
            Synchronized DataFrame with path indices and telemetry values
        """
        print(f"\nSynchronizing telemetry for vehicle {vehicle_number}...")

        # Filter by vehicle
        if 'vehicle_number' in telemetry.columns:
            telemetry = telemetry[telemetry['vehicle_number'] == vehicle_number].copy()

        if telemetry.empty:
            raise ValueError(f"No telemetry data found for vehicle {vehicle_number}")

        # Select lap
        if lap_number is not None:
            if 'lap' in telemetry.columns:
                telemetry = telemetry[telemetry['lap'] == lap_number].copy()
        else:
            # Use lap with most GPS data
            if 'lap' in telemetry.columns:
                lap_counts = telemetry.groupby('lap').size()
                lap_number = lap_counts.idxmax()
                telemetry = telemetry[telemetry['lap'] == lap_number].copy()
                print(f"  Auto-selected lap {lap_number} ({len(telemetry)} records)")

        # Check if telemetry is in long format
        if 'telemetry_name' in telemetry.columns and 'telemetry_value' in telemetry.columns:
            # Pivot to wide format
            print("  Converting long format to wide format...")
            telemetry = telemetry.pivot_table(
                index=['timestamp', 'vehicle_number', 'lap'],
                columns='telemetry_name',
                values='telemetry_value',
                aggfunc='first'
            ).reset_index()

        # Find GPS columns
        gps_cols = self._find_gps_columns(telemetry)
        if not gps_cols:
            raise ValueError("No GPS columns found in telemetry")

        lat_col, lon_col = gps_cols
        print(f"  Found GPS: {lat_col}, {lon_col}")

        # Extract GPS data
        gps_telemetry = telemetry[[lat_col, lon_col]].dropna()

        # Convert VBOX format if needed
        if gps_telemetry[lat_col].abs().max() > 90 or gps_telemetry[lon_col].abs().max() > 180:
            print("  Converting VBOX format to decimal degrees...")
            gps_telemetry[lat_col] = gps_telemetry[lat_col] / 60
            gps_telemetry[lon_col] = gps_telemetry[lon_col] / 60

        # Match telemetry GPS to path GPS
        path_gps = np.array(self.path_data['gps_path'])
        telemetry_gps = gps_telemetry[[lat_col, lon_col]].values

        # Find closest path point for each telemetry point
        print(f"  Matching {len(telemetry_gps)} telemetry points to path...")
        path_indices = []

        for telem_point in telemetry_gps:
            # Calculate distances to all path points
            distances = np.sqrt(
                (path_gps[:, 0] - telem_point[0])**2 +
                (path_gps[:, 1] - telem_point[1])**2
            )
            closest_idx = np.argmin(distances)
            path_indices.append(closest_idx)

        # Add path index to telemetry
        telemetry_subset = telemetry.loc[gps_telemetry.index].copy()
        telemetry_subset['path_index'] = path_indices

        print(f"  Synchronized {len(telemetry_subset)} points")

        return telemetry_subset

    def _find_gps_columns(self, df: pd.DataFrame) -> Optional[Tuple[str, str]]:
        """Find GPS column names in DataFrame"""
        gps_patterns = [
            ['gps_lat', 'gps_long'],
            ['VBOX_Lat_Min', 'VBOX_Long_Minutes'],
            ['gps_latitude', 'gps_longitude']
        ]

        for lat_col, lon_col in gps_patterns:
            if lat_col in df.columns and lon_col in df.columns:
                return (lat_col, lon_col)

        return None

    def create_speed_heatmap(
        self,
        telemetry: pd.DataFrame,
        speed_column: str = 'speed'
    ) -> np.ndarray:
        """
        Create speed heatmap colors for the path

        Args:
            telemetry: Synchronized telemetry with path_index
            speed_column: Name of speed column

        Returns:
            RGB color array for each path point
        """
        path_length = len(self.path_data['pixel_path'])

        # Initialize with default speed (track average)
        if speed_column in telemetry.columns:
            speeds = np.full(path_length, telemetry[speed_column].median())

            # Fill in known speeds
            for _, row in telemetry.iterrows():
                idx = int(row['path_index'])
                if 0 <= idx < path_length:
                    speeds[idx] = row[speed_column]

            # Interpolate missing speeds
            known_indices = telemetry['path_index'].values
            known_speeds = telemetry[speed_column].values

            if len(known_indices) > 1:
                f = interp1d(
                    known_indices,
                    known_speeds,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                all_indices = np.arange(path_length)
                speeds = f(all_indices)

            # Normalize speeds to [0, 1]
            speed_min, speed_max = speeds.min(), speeds.max()
            if speed_max > speed_min:
                speed_norm = (speeds - speed_min) / (speed_max - speed_min)
            else:
                speed_norm = np.ones_like(speeds) * 0.5

        else:
            speed_norm = np.ones(path_length) * 0.5

        # Convert to RGB colors (blue = slow, red = fast)
        colors = np.zeros((path_length, 3))
        colors[:, 0] = speed_norm  # Red channel
        colors[:, 2] = 1 - speed_norm  # Blue channel

        return colors

    def create_plotly_animation(
        self,
        telemetry: pd.DataFrame,
        vehicle_number: int,
        lap_number: Optional[int] = None,
        overlay: str = 'speed',
        fps: int = 30,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create Plotly animation figure

        Args:
            telemetry: Telemetry data
            vehicle_number: Vehicle to animate
            lap_number: Specific lap (None = auto-select)
            overlay: Overlay type ('speed', 'none')
            fps: Frames per second
            title: Custom title

        Returns:
            Plotly figure with animation
        """
        print(f"\nCreating animation for {self.metadata['name']}...")

        # Sync telemetry
        synced_telem = self.sync_telemetry_to_path(telemetry, vehicle_number, lap_number)

        # Get path data
        pixel_path = np.array(self.path_data['pixel_path'])
        path_x = pixel_path[:, 0]
        path_y = pixel_path[:, 1]

        # Create speed heatmap if requested
        if overlay == 'speed':
            colors = self.create_speed_heatmap(synced_telem, 'speed')
            color_data = colors[:, 0]  # Use red channel for color intensity
        else:
            color_data = np.ones(len(path_x)) * 0.5

        # Load track image
        from PIL import Image
        img = Image.open(self.image_path)
        img_width, img_height = img.size

        # Create figure
        fig = go.Figure()

        # Add track image as background
        fig.add_layout_image(
            dict(
                source=img,
                xref="x",
                yref="y",
                x=0,
                y=img_height,  # Position at top of inverted y-axis
                sizex=img_width,
                sizey=img_height,
                sizing="stretch",
                opacity=1.0,
                layer="below"
            )
        )

        # Add path trace with speed colors
        fig.add_trace(go.Scatter(
            x=path_x,
            y=path_y,
            mode='markers',
            marker=dict(
                color=color_data.tolist(),
                colorscale='RdYlBu_r',  # Red = fast, Blue = slow
                size=4,
                cmin=0,
                cmax=1,
                showscale=False
            ),
            name='Track Path',
            showlegend=False
        ))

        # Add animated car marker
        car_positions = []
        for idx in range(0, len(pixel_path), max(1, len(pixel_path) // (fps * 10))):
            car_positions.append(idx)

        frames = []
        for pos_idx in car_positions:
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=[path_x[pos_idx]],
                        y=[path_y[pos_idx]],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='yellow',
                            symbol='circle',
                            line=dict(color='black', width=2)
                        ),
                        showlegend=False
                    )
                ],
                name=str(pos_idx)
            )
            frames.append(frame)

        fig.frames = frames

        # Add play/pause buttons
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 1000 // fps, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0}
                            }]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }]
                        )
                    ],
                    x=0.1,
                    xanchor="left",
                    y=0,
                    yanchor="top"
                )
            ],
            xaxis=dict(
                range=[0, img_width],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                range=[img_height, 0],  # Inverted Y
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            title=title or f"{self.metadata['name']} - Vehicle {vehicle_number}",
            showlegend=False,
            height=800,
            margin=dict(l=0, r=0, t=50, b=0)
        )

        print(f"  Animation created: {len(frames)} frames at {fps} FPS")

        return fig

    def create_static_overlay(
        self,
        telemetry: pd.DataFrame,
        vehicle_number: int,
        lap_number: Optional[int] = None,
        overlay: str = 'speed'
    ) -> go.Figure:
        """
        Create static track visualization with telemetry overlay

        Args:
            telemetry: Telemetry data
            vehicle_number: Vehicle number
            lap_number: Specific lap (None = auto-select)
            overlay: Overlay type ('speed', 'brake', 'throttle')

        Returns:
            Plotly figure with static overlay
        """
        print(f"\nCreating static overlay for {self.metadata['name']}...")

        # Sync telemetry
        synced_telem = self.sync_telemetry_to_path(telemetry, vehicle_number, lap_number)

        # Get path data
        pixel_path = np.array(self.path_data['pixel_path'])
        path_x = pixel_path[:, 0]
        path_y = pixel_path[:, 1]

        # Create overlay colors
        if overlay == 'speed':
            colors = self.create_speed_heatmap(synced_telem, 'speed')
            color_data = colors[:, 0]
            colorscale = 'RdYlBu_r'
            colorbar_title = 'Speed'
        else:
            color_data = np.ones(len(path_x)) * 0.5
            colorscale = 'Viridis'
            colorbar_title = overlay.title()

        # Load track image
        from PIL import Image
        img = Image.open(self.image_path)
        img_width, img_height = img.size

        # Create figure
        fig = go.Figure()

        # Add track image
        fig.add_layout_image(
            dict(
                source=img,
                xref="x",
                yref="y",
                x=0,
                y=img_height,  # Position at top of inverted y-axis
                sizex=img_width,
                sizey=img_height,
                sizing="stretch",
                opacity=1.0,
                layer="below"
            )
        )

        # Add path with overlay
        fig.add_trace(go.Scatter(
            x=path_x,
            y=path_y,
            mode='markers',
            marker=dict(
                color=color_data.tolist(),
                colorscale=colorscale,
                size=6,
                cmin=0,
                cmax=1,
                colorbar=dict(title=colorbar_title),
                showscale=True
            ),
            name='Track Path',
            showlegend=False
        ))

        # Update layout
        fig.update_layout(
            xaxis=dict(
                range=[0, img_width],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                range=[img_height, 0],
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            title=f"{self.metadata['name']} - {overlay.title()} Overlay - Vehicle {vehicle_number}",
            height=800,
            margin=dict(l=0, r=0, t=50, b=0)
        )

        print(f"  Static overlay created")

        return fig
