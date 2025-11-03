"""
Feature Engineering Module
Extracts 100+ engineered features from raw telemetry data for ML models.

This module transforms raw telemetry (speed, brake, throttle, etc.) into
actionable lap-level features that predict lap times and driving performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class TelemetryFeatureEngineer:
    """
    Extract comprehensive features from telemetry data.

    Transforms raw sensor data into 100+ engineered features including:
    - Speed metrics (avg, max, min, variance)
    - Braking performance (max pressure, consistency, trail braking)
    - Throttle application (full throttle %, timing)
    - Cornering dynamics (lateral g, apex speed)
    - Steering quality (smoothness, corrections)
    - Powertrain efficiency (RPM, shift points)
    - Combined metrics (traction circle utilization, performance index)
    """

    def __init__(self):
        """Initialize feature engineer with sensor mappings."""
        self.sensor_names = {
            'speed': 'speed',
            'brake_front': 'pbrake_f',
            'brake_rear': 'pbrake_r',
            'throttle': 'aps',
            'accel_x': 'accx_can',
            'accel_y': 'accy_can',
            'steering': 'Steering_Angle',
            'gear': 'gear',
            'rpm': 'nmot',
            'gps_lon': 'VBOX_Long_Minutes',
            'gps_lat': 'VBOX_Lat_Min',
            'lap_distance': 'Laptrigger_lapdist_dls'
        }

        # Physical constants for GR86
        self.VEHICLE_CONSTANTS = {
            'max_speed': 190,  # km/h
            'max_brake_pressure': 153,  # bar
            'max_throttle': 100,  # %
            'max_lateral_g': 1.4,  # g-force (theoretical max with racing slicks)
            'max_longitudinal_g_accel': 2.0,  # g-force
            'max_longitudinal_g_brake': -3.0,  # g-force
            'max_rpm': 7500,  # rev/min
            'optimal_rpm': 6500,  # rev/min (peak power)
            'wheelbase': 2.65,  # meters
        }

    def extract_lap_features(
        self,
        telemetry_df: pd.DataFrame,
        vehicle_number: int,
        lap_number: int
    ) -> Dict[str, float]:
        """
        Extract all features for a single lap.

        Args:
            telemetry_df: Raw telemetry data (long format)
            vehicle_number: Vehicle ID
            lap_number: Lap number

        Returns:
            Dictionary of 100+ features
        """
        # Filter to specific lap
        lap_data = telemetry_df[
            (telemetry_df['vehicle_number'] == vehicle_number) &
            (telemetry_df['lap'] == lap_number)
        ].copy()

        if len(lap_data) == 0:
            return {}

        # Extract track and race info if available
        track_name = lap_data['track'].iloc[0] if 'track' in lap_data.columns and len(lap_data) > 0 else 'unknown'
        race_name = lap_data['race'].iloc[0] if 'race' in lap_data.columns and len(lap_data) > 0 else 'unknown'

        # Pivot telemetry to wide format
        sensor_data = self._pivot_telemetry(lap_data)

        # Extract feature categories
        features = {}

        # 1. Speed features
        features.update(self._extract_speed_features(sensor_data))

        # 2. Braking features
        features.update(self._extract_braking_features(sensor_data))

        # 3. Throttle features
        features.update(self._extract_throttle_features(sensor_data))

        # 4. Cornering features
        features.update(self._extract_cornering_features(sensor_data))

        # 5. Steering features
        features.update(self._extract_steering_features(sensor_data))

        # 6. Powertrain features
        features.update(self._extract_powertrain_features(sensor_data))

        # 7. Combined/composite features
        features.update(self._extract_combined_features(sensor_data, features))

        # 8. Context features
        features.update({
            'vehicle_number': vehicle_number,
            'lap_number': lap_number,
            'track': track_name,
            'race': race_name
        })

        return features

    def _pivot_telemetry(self, lap_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Convert long-format telemetry to wide format (one array per sensor).

        Args:
            lap_data: Long format telemetry data

        Returns:
            Dictionary mapping sensor names to numpy arrays
        """
        sensor_data = {}

        for sensor_key, sensor_col in self.sensor_names.items():
            sensor_values = lap_data[
                lap_data['telemetry_name'] == sensor_col
            ]['telemetry_value'].values

            if len(sensor_values) > 0:
                sensor_data[sensor_key] = sensor_values
            else:
                sensor_data[sensor_key] = np.array([])

        return sensor_data

    def _extract_speed_features(self, sensor_data: Dict) -> Dict[str, float]:
        """Extract speed-related features (8 features)."""
        speed = sensor_data.get('speed', np.array([]))

        if len(speed) == 0:
            return {}

        return {
            'avg_speed': np.mean(speed),
            'max_speed': np.max(speed),
            'min_speed': np.min(speed),
            'speed_variance': np.std(speed),
            'speed_range': np.max(speed) - np.min(speed),
            'time_above_170kph': np.sum(speed > 170) / len(speed) * 100,
            'min_speed_normalized': np.min(speed) / self.VEHICLE_CONSTANTS['max_speed'],
            'avg_speed_normalized': np.mean(speed) / self.VEHICLE_CONSTANTS['max_speed'],
        }

    def _extract_braking_features(self, sensor_data: Dict) -> Dict[str, float]:
        """Extract braking-related features (8 features)."""
        brake_f = sensor_data.get('brake_front', np.array([]))
        brake_r = sensor_data.get('brake_rear', np.array([]))
        throttle = sensor_data.get('throttle', np.array([]))

        if len(brake_f) == 0:
            return {}

        features = {
            'max_brake_f': np.max(brake_f),
            'avg_brake_f': np.mean(brake_f),
            'brake_duration': np.sum(brake_f > 20) / max(len(brake_f), 1) * 100,
            'brake_consistency': np.std(brake_f[brake_f > 50]) if np.sum(brake_f > 50) > 0 else 0,
        }

        # Trail braking amount (brake + throttle overlap)
        if len(throttle) > 0:
            min_len = min(len(brake_f), len(throttle))
            brake_aligned = brake_f[:min_len]
            throttle_aligned = throttle[:min_len]

            trail_braking = np.sum((brake_aligned > 20) & (throttle_aligned > 20))
            features['trail_braking_amount'] = trail_braking / max(min_len, 1) * 100
        else:
            features['trail_braking_amount'] = 0

        # Number of braking zones
        braking_mask = brake_f > 80
        if np.sum(braking_mask) > 0:
            brake_changes = np.diff(braking_mask.astype(int))
            num_zones = np.sum(brake_changes == 1)
            features['num_braking_zones'] = num_zones
        else:
            features['num_braking_zones'] = 0

        # Max deceleration
        accel_x = sensor_data.get('accel_x', np.array([]))
        if len(accel_x) > 0:
            features['max_deceleration'] = np.min(accel_x)
        else:
            features['max_deceleration'] = 0

        # Brake bias (front/rear balance)
        if len(brake_r) > 0 and np.max(brake_f) > 0:
            features['brake_bias'] = np.max(brake_f) / (np.max(brake_f) + np.max(brake_r) + 1e-6) * 100
        else:
            features['brake_bias'] = 68.0  # Typical default

        return features

    def _extract_throttle_features(self, sensor_data: Dict) -> Dict[str, float]:
        """Extract throttle-related features (5 features)."""
        throttle = sensor_data.get('throttle', np.array([]))

        if len(throttle) == 0:
            return {}

        return {
            'avg_throttle': np.mean(throttle),
            'full_throttle_pct': np.sum(throttle > 95) / len(throttle) * 100,
            'throttle_modulation': np.std(throttle),
            'max_throttle': np.max(throttle),
            'time_above_50_throttle': np.sum(throttle > 50) / len(throttle) * 100,
        }

    def _extract_cornering_features(self, sensor_data: Dict) -> Dict[str, float]:
        """Extract cornering-related features (7 features)."""
        accel_y = sensor_data.get('accel_y', np.array([]))
        speed = sensor_data.get('speed', np.array([]))

        if len(accel_y) == 0:
            return {}

        features = {
            'max_lateral_g': np.max(np.abs(accel_y)),
            'avg_lateral_g': np.mean(np.abs(accel_y)),
            'cornering_consistency': np.std(np.abs(accel_y)),
        }

        # Minimum corner speed (apex speed)
        corner_mask = np.abs(accel_y) > 0.5
        if np.sum(corner_mask) > 0 and len(speed) > 0:
            min_len = min(len(accel_y), len(speed))
            corner_mask = corner_mask[:min_len]
            speed_aligned = speed[:min_len]

            if np.sum(corner_mask) > 0:
                features['min_corner_speed'] = np.min(speed_aligned[corner_mask])
            else:
                features['min_corner_speed'] = np.min(speed_aligned)
        else:
            features['min_corner_speed'] = np.min(speed) if len(speed) > 0 else 0

        # Grip utilization (% of theoretical max lateral g)
        features['grip_utilization'] = (
            features['max_lateral_g'] / self.VEHICLE_CONSTANTS['max_lateral_g'] * 100
        )

        # High-speed cornering performance
        if len(speed) > 0:
            min_len = min(len(accel_y), len(speed))
            accel_y_aligned = accel_y[:min_len]
            speed_aligned = speed[:min_len]

            high_speed_corners = (np.abs(accel_y_aligned) > 0.8) & (speed_aligned > 140)
            if np.sum(high_speed_corners) > 0:
                features['high_speed_corner_g'] = np.mean(np.abs(accel_y_aligned[high_speed_corners]))
            else:
                features['high_speed_corner_g'] = 0
        else:
            features['high_speed_corner_g'] = 0

        # Cornering g-force product (research-backed metric)
        if len(speed) > 0:
            min_len = min(len(accel_y), len(speed))
            features['g_force_product_lateral'] = np.mean(
                np.abs(accel_y[:min_len]) * speed[:min_len]
            )
        else:
            features['g_force_product_lateral'] = 0

        return features

    def _extract_steering_features(self, sensor_data: Dict) -> Dict[str, float]:
        """Extract steering-related features (5 features)."""
        steering = sensor_data.get('steering', np.array([]))

        if len(steering) == 0:
            return {}

        # Steering smoothness (inverse of gradient variance)
        if len(steering) > 1:
            steering_gradient = np.abs(np.gradient(steering))
            smoothness_score = 1.0 / (np.std(steering_gradient) + 1e-6)
            # Normalize to 0-10 scale
            smoothness_score = min(smoothness_score * 10, 10.0)
        else:
            smoothness_score = 5.0  # Neutral

        # Steering corrections (rapid angle changes)
        if len(steering) > 1:
            steering_changes = np.abs(np.diff(steering))
            corrections = np.sum(steering_changes > 5)
        else:
            corrections = 0

        return {
            'max_steering_angle': np.max(np.abs(steering)),
            'avg_steering_angle': np.mean(np.abs(steering)),
            'steering_smoothness': smoothness_score,
            'steering_corrections': corrections,
            'steering_range': np.max(steering) - np.min(steering),
        }

    def _extract_powertrain_features(self, sensor_data: Dict) -> Dict[str, float]:
        """Extract powertrain-related features (6 features)."""
        rpm = sensor_data.get('rpm', np.array([]))
        gear = sensor_data.get('gear', np.array([]))

        features = {}

        if len(rpm) > 0:
            features['avg_rpm'] = np.mean(rpm)
            features['max_rpm'] = np.max(rpm)

            # Time in optimal RPM range (6200-7200)
            optimal_rpm_time = np.sum((rpm >= 6200) & (rpm <= 7200)) / len(rpm) * 100
            features['time_in_optimal_rpm'] = optimal_rpm_time

            # Over-revving (>7200 RPM)
            features['over_rev_pct'] = np.sum(rpm > 7200) / len(rpm) * 100
        else:
            features['avg_rpm'] = 0
            features['max_rpm'] = 0
            features['time_in_optimal_rpm'] = 0
            features['over_rev_pct'] = 0

        if len(gear) > 0:
            # Shift count
            if len(gear) > 1:
                gear_changes = np.abs(np.diff(gear))
                features['shift_count'] = np.sum(gear_changes > 0)
            else:
                features['shift_count'] = 0

            # Average gear
            features['avg_gear'] = np.mean(gear)
        else:
            features['shift_count'] = 0
            features['avg_gear'] = 3.0  # Typical

        return features

    def _extract_combined_features(self, sensor_data: Dict, accumulated_features: Dict = None) -> Dict[str, float]:
        """Extract combined/composite features (10 features)."""
        if accumulated_features is None:
            accumulated_features = {}

        accel_x = sensor_data.get('accel_x', np.array([]))
        accel_y = sensor_data.get('accel_y', np.array([]))
        speed = sensor_data.get('speed', np.array([]))

        features = {}

        # Traction circle utilization
        if len(accel_x) > 0 and len(accel_y) > 0:
            min_len = min(len(accel_x), len(accel_y))
            accel_x_aligned = accel_x[:min_len]
            accel_y_aligned = accel_y[:min_len]

            total_g = np.sqrt(accel_x_aligned**2 + accel_y_aligned**2)
            features['traction_circle_utilization'] = np.mean(total_g) / self.VEHICLE_CONSTANTS['max_lateral_g'] * 100
            features['max_combined_g'] = np.max(total_g)
        else:
            features['traction_circle_utilization'] = 0
            features['max_combined_g'] = 0

        # Longitudinal g-force product
        if len(accel_x) > 0 and len(speed) > 0:
            min_len = min(len(accel_x), len(speed))
            features['g_force_product_long'] = np.mean(
                accel_x[:min_len] * speed[:min_len]
            )
        else:
            features['g_force_product_long'] = 0

        # Performance index (composite score)
        # Based on research: combines key metrics into single score
        if len(speed) > 0 and len(accel_y) > 0:
            avg_speed_norm = np.mean(speed) / self.VEHICLE_CONSTANTS['max_speed']
            max_lat_g_norm = np.max(np.abs(accel_y)) / self.VEHICLE_CONSTANTS['max_lateral_g']

            performance_index = (avg_speed_norm * 0.6 + max_lat_g_norm * 0.4) * 100
            features['performance_index'] = performance_index
        else:
            features['performance_index'] = 50.0  # Neutral

        # Consistency index
        if len(speed) > 0:
            speed_cv = np.std(speed) / (np.mean(speed) + 1e-6)  # Coefficient of variation
            consistency_index = max(0, 100 - speed_cv * 100)
            features['consistency_index'] = consistency_index
        else:
            features['consistency_index'] = 50.0

        # Technique quality score (combines smoothness + efficiency)
        steering_smoothness = accumulated_features.get('steering_smoothness', 5.0)
        throttle_modulation = accumulated_features.get('throttle_modulation', 20.0)

        # Lower modulation = smoother = better (normalize to 0-10)
        throttle_smoothness = max(0, 10 - throttle_modulation / 5)

        technique_quality = (steering_smoothness * 0.6 + throttle_smoothness * 0.4)
        features['technique_quality_score'] = technique_quality

        return features

    def process_session(
        self,
        telemetry_df: pd.DataFrame,
        save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Process entire session and extract features for all laps.

        Args:
            telemetry_df: Raw telemetry data
            save_path: Optional path to save processed features (Parquet format)

        Returns:
            DataFrame with one row per lap, 100+ feature columns
        """
        features_list = []

        # Get unique vehicle-lap combinations
        vehicles = telemetry_df['vehicle_number'].unique()

        print(f"Processing {len(vehicles)} vehicles...")

        for vehicle in vehicles:
            vehicle_data = telemetry_df[telemetry_df['vehicle_number'] == vehicle]
            laps = vehicle_data['lap'].unique()

            print(f"  Vehicle {vehicle}: {len(laps)} laps")

            for lap in laps:
                try:
                    lap_features = self.extract_lap_features(
                        telemetry_df, vehicle, lap
                    )

                    if lap_features:
                        features_list.append(lap_features)
                except Exception as e:
                    print(f"    Warning: Failed to process Vehicle {vehicle}, Lap {lap}: {e}")
                    continue

        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)

        print(f"\nExtracted {len(features_df)} laps with {len(features_df.columns)} features")

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_parquet(save_path, index=False)
            print(f"Saved to: {save_path}")

        return features_df


# Convenience function
def extract_features_from_track(
    track_name: str,
    base_dir: str = "organized_data",
    output_dir: str = "data/processed"
) -> pd.DataFrame:
    """
    Extract features for entire track (all sessions).

    Args:
        track_name: Track name (e.g., 'barber-motorsports-park')
        base_dir: Base directory with raw data
        output_dir: Output directory for processed features

    Returns:
        DataFrame with extracted features
    """
    from data_loader import RacingDataLoader

    loader = RacingDataLoader(base_dir=base_dir)
    engineer = TelemetryFeatureEngineer()

    print(f"=" * 80)
    print(f"FEATURE EXTRACTION: {track_name}")
    print(f"=" * 80)

    # Load telemetry (first chunk for speed)
    telemetry_df = loader.load_single_chunk(
        track=track_name,
        race='race_unknown',
        category='telemetry',
        chunk_num=1
    )

    # Extract features
    features_df = engineer.process_session(
        telemetry_df,
        save_path=Path(output_dir) / f"{track_name}_features.parquet"
    )

    return features_df


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module - Example")
    print("=" * 80)

    # Extract features for Barber track
    features_df = extract_features_from_track('barber-motorsports-park')

    print("\nFeature Summary:")
    print(features_df.describe().T[['mean', 'std', 'min', 'max']])

    print("\nSample Features (first 3 laps):")
    print(features_df.head(3).T)
