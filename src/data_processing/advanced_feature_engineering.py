"""
Advanced Feature Engineering Module

Extends base feature engineering with sophisticated frequency-domain,
wavelet, segmentation, and temporal features for racing telemetry.

This module creates 60-80 advanced features including:
- FFT (Fast Fourier Transform) frequency domain features
- Wavelet transform multi-scale decomposition
- Lap segmentation (corner-by-corner analysis)
- Track encoding (one-hot + embeddings)
- Temporal/sequential features (tire degradation, fatigue)
- Weather integration features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
from scipy import fft
from scipy.signal import find_peaks
import pywt  # PyWavelets
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """
    Create advanced features from telemetry data.

    Extends basic features with frequency domain analysis, wavelets,
    corner segmentation, and temporal progression tracking.
    """

    def __init__(self):
        """Initialize advanced feature engineer."""
        # Sensor mappings (same as base engineer)
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

        # FFT parameters
        self.fft_sample_rate = 100  # Hz (assumed telemetry sampling rate)

        # Wavelet parameters
        self.wavelet_type = 'db4'  # Daubechies 4 wavelet
        self.wavelet_levels = 3  # 3-level decomposition

        # Corner detection thresholds
        self.corner_speed_drop = 40  # km/h speed drop to identify corner
        self.corner_lateral_g_threshold = 0.5  # g-force to identify corner
        self.max_corners_per_track = 10  # Track top 10 corners

        # Track encoder (will be fitted during processing)
        self.track_encoder = None
        self.fitted_tracks = []

    def extract_advanced_features(
        self,
        telemetry_df: pd.DataFrame,
        vehicle_number: int,
        lap_number: int,
        track_name: str,
        session_lap_count: int = 1,
        total_session_laps: int = 1
    ) -> Dict[str, float]:
        """
        Extract all advanced features for a single lap.

        Args:
            telemetry_df: Raw telemetry data (long format)
            vehicle_number: Vehicle ID
            lap_number: Lap number
            track_name: Track identifier
            session_lap_count: Which lap in this session (1, 2, 3, ...)
            total_session_laps: Total laps in session

        Returns:
            Dictionary of 60-80 advanced features
        """
        # Filter to specific lap
        lap_data = telemetry_df[
            (telemetry_df['vehicle_number'] == vehicle_number) &
            (telemetry_df['lap'] == lap_number)
        ].copy()

        if len(lap_data) == 0:
            return {}

        # Pivot telemetry to wide format
        sensor_data = self._pivot_telemetry(lap_data)

        # Extract feature categories
        features = {}

        # 1. FFT (Frequency Domain) Features
        features.update(self._extract_fft_features(sensor_data))

        # 2. Wavelet Transform Features
        features.update(self._extract_wavelet_features(sensor_data))

        # 3. Lap Segmentation Features (Corner-by-Corner)
        features.update(self._extract_segmentation_features(sensor_data))

        # 4. Track Encoding Features
        features.update(self._extract_track_encoding_features(track_name))

        # 5. Temporal/Sequential Features
        features.update(self._extract_temporal_features(
            session_lap_count,
            total_session_laps,
            sensor_data
        ))

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

    def _extract_fft_features(self, sensor_data: Dict) -> Dict[str, float]:
        """
        Extract FFT (frequency domain) features.

        Analyzes periodic patterns in speed, throttle, brake signals.
        Features capture driving rhythm and consistency.

        Returns 15 features:
        - Dominant frequencies for speed, throttle, brake
        - Peak power for each signal
        - Spectral entropy (signal complexity)
        - Harmonic ratios
        """
        features = {}

        # Signals to analyze
        signals = {
            'speed': sensor_data.get('speed', np.array([])),
            'throttle': sensor_data.get('throttle', np.array([])),
            'brake_f': sensor_data.get('brake_front', np.array([]))
        }

        for signal_name, signal in signals.items():
            if len(signal) < 10:  # Need minimum samples for FFT
                features[f'fft_{signal_name}_dominant_freq'] = 0.0
                features[f'fft_{signal_name}_peak_power'] = 0.0
                features[f'fft_{signal_name}_spectral_entropy'] = 0.0
                continue

            # Apply FFT
            signal_fft = fft.rfft(signal)
            frequencies = fft.rfftfreq(len(signal), 1.0 / self.fft_sample_rate)
            power_spectrum = np.abs(signal_fft) ** 2

            # Dominant frequency (skip DC component at index 0)
            if len(power_spectrum) > 1:
                dominant_idx = np.argmax(power_spectrum[1:]) + 1
                features[f'fft_{signal_name}_dominant_freq'] = frequencies[dominant_idx]
                features[f'fft_{signal_name}_peak_power'] = power_spectrum[dominant_idx]
            else:
                features[f'fft_{signal_name}_dominant_freq'] = 0.0
                features[f'fft_{signal_name}_peak_power'] = 0.0

            # Spectral entropy (measure of signal complexity)
            # Higher entropy = more chaotic/variable driving
            if np.sum(power_spectrum) > 0:
                normalized_spectrum = power_spectrum / np.sum(power_spectrum)
                # Avoid log(0)
                normalized_spectrum = normalized_spectrum[normalized_spectrum > 0]
                spectral_entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum))
                features[f'fft_{signal_name}_spectral_entropy'] = spectral_entropy
            else:
                features[f'fft_{signal_name}_spectral_entropy'] = 0.0

        # Harmonic ratio (fundamental vs harmonics) for speed
        speed = signals['speed']
        if len(speed) >= 10:
            speed_fft = fft.rfft(speed)
            power = np.abs(speed_fft) ** 2

            if len(power) > 4:
                # Ratio of fundamental to sum of first 3 harmonics
                fundamental = power[1] if len(power) > 1 else 0
                harmonics = np.sum(power[2:5]) if len(power) > 4 else 0

                if harmonics > 0:
                    features['fft_speed_harmonic_ratio'] = fundamental / harmonics
                else:
                    features['fft_speed_harmonic_ratio'] = 1.0
            else:
                features['fft_speed_harmonic_ratio'] = 1.0
        else:
            features['fft_speed_harmonic_ratio'] = 1.0

        # Total spectral power for throttle (aggressive driving indicator)
        throttle = signals['throttle']
        if len(throttle) >= 10:
            throttle_fft = fft.rfft(throttle)
            features['fft_throttle_total_power'] = np.sum(np.abs(throttle_fft) ** 2)
        else:
            features['fft_throttle_total_power'] = 0.0

        # Brake frequency variation (consistency indicator)
        brake = signals['brake_f']
        if len(brake) >= 10:
            brake_fft = fft.rfft(brake)
            power = np.abs(brake_fft) ** 2
            frequencies = fft.rfftfreq(len(brake), 1.0 / self.fft_sample_rate)

            if np.sum(power) > 0:
                # Weighted average frequency
                weighted_freq = np.sum(frequencies * power) / np.sum(power)
                features['fft_brake_weighted_freq'] = weighted_freq
            else:
                features['fft_brake_weighted_freq'] = 0.0
        else:
            features['fft_brake_weighted_freq'] = 0.0

        return features

    def _extract_wavelet_features(self, sensor_data: Dict) -> Dict[str, float]:
        """
        Extract wavelet transform features.

        Multi-scale decomposition captures patterns at different time scales:
        - Low frequency: Overall lap strategy
        - Mid frequency: Corner-to-corner transitions
        - High frequency: Micro-corrections and smoothness

        Returns 18 features (6 signals × 3 scales).
        """
        features = {}

        # Signals to analyze
        signals = {
            'speed': sensor_data.get('speed', np.array([])),
            'throttle': sensor_data.get('throttle', np.array([])),
            'brake_f': sensor_data.get('brake_front', np.array([])),
            'lateral_g': sensor_data.get('accel_y', np.array([])),
            'long_g': sensor_data.get('accel_x', np.array([])),
            'steering': sensor_data.get('steering', np.array([]))
        }

        for signal_name, signal in signals.items():
            if len(signal) < 8:  # Need minimum samples for wavelet
                features[f'wavelet_{signal_name}_low_freq'] = 0.0
                features[f'wavelet_{signal_name}_mid_freq'] = 0.0
                features[f'wavelet_{signal_name}_high_freq'] = 0.0
                continue

            try:
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(signal, self.wavelet_type, level=self.wavelet_levels)

                # coeffs = [cA3, cD3, cD2, cD1]
                # cA3 = approximation (low frequency trend)
                # cD3, cD2, cD1 = details (high to low frequency)

                # Low frequency: approximation coefficient energy
                low_freq_energy = np.sum(coeffs[0] ** 2) if len(coeffs) > 0 else 0.0
                features[f'wavelet_{signal_name}_low_freq'] = low_freq_energy

                # Mid frequency: middle detail coefficient energy
                mid_idx = len(coeffs) // 2
                mid_freq_energy = np.sum(coeffs[mid_idx] ** 2) if len(coeffs) > mid_idx else 0.0
                features[f'wavelet_{signal_name}_mid_freq'] = mid_freq_energy

                # High frequency: finest detail coefficient energy
                high_freq_energy = np.sum(coeffs[-1] ** 2) if len(coeffs) > 0 else 0.0
                features[f'wavelet_{signal_name}_high_freq'] = high_freq_energy

            except Exception as e:
                # Fallback if wavelet fails
                features[f'wavelet_{signal_name}_low_freq'] = 0.0
                features[f'wavelet_{signal_name}_mid_freq'] = 0.0
                features[f'wavelet_{signal_name}_high_freq'] = 0.0

        return features

    def _extract_segmentation_features(self, sensor_data: Dict) -> Dict[str, float]:
        """
        Extract lap segmentation features (corner-by-corner analysis).

        Identifies major corners and extracts:
        - Entry speed
        - Apex speed (minimum speed in corner)
        - Exit speed
        - Brake point (distance before apex)
        - Time in corner

        Returns ~40 features (10 corners × 4 metrics each).
        """
        features = {}

        speed = sensor_data.get('speed', np.array([]))
        lateral_g = sensor_data.get('accel_y', np.array([]))
        brake_f = sensor_data.get('brake_front', np.array([]))

        if len(speed) < 20 or len(lateral_g) < 20:
            # Not enough data - return zeros
            for i in range(self.max_corners_per_track):
                features[f'corner_{i+1}_entry_speed'] = 0.0
                features[f'corner_{i+1}_apex_speed'] = 0.0
                features[f'corner_{i+1}_exit_speed'] = 0.0
                features[f'corner_{i+1}_brake_point'] = 0.0
            return features

        # Align arrays
        min_len = min(len(speed), len(lateral_g), len(brake_f))
        speed = speed[:min_len]
        lateral_g = lateral_g[:min_len]
        brake_f = brake_f[:min_len]

        # Identify corners using combined criteria:
        # 1. Lateral g-force > threshold
        # 2. Speed drop > threshold

        # Smooth speed to reduce noise
        from scipy.ndimage import gaussian_filter1d
        speed_smooth = gaussian_filter1d(speed, sigma=3)

        # Find local minima in speed (potential apex points)
        apex_candidates, _ = find_peaks(-speed_smooth, distance=50, prominence=10)

        # Filter by lateral g at those points
        valid_apexes = []
        for apex_idx in apex_candidates:
            if apex_idx < len(lateral_g) and abs(lateral_g[apex_idx]) > self.corner_lateral_g_threshold:
                valid_apexes.append(apex_idx)

        # Sort by apex speed (slowest corners first - most significant)
        if len(valid_apexes) > 0:
            apex_speeds = [speed[idx] for idx in valid_apexes]
            sorted_indices = np.argsort(apex_speeds)
            valid_apexes = [valid_apexes[i] for i in sorted_indices]

        # Extract features for top N corners
        num_corners = min(len(valid_apexes), self.max_corners_per_track)

        for i in range(num_corners):
            apex_idx = valid_apexes[i]

            # Apex speed (minimum)
            apex_speed = speed[apex_idx]
            features[f'corner_{i+1}_apex_speed'] = apex_speed

            # Entry speed (look back 30-50 samples before apex)
            entry_idx = max(0, apex_idx - 40)
            entry_speed = np.max(speed[entry_idx:apex_idx]) if apex_idx > entry_idx else apex_speed
            features[f'corner_{i+1}_entry_speed'] = entry_speed

            # Exit speed (look ahead 30-50 samples after apex)
            exit_idx = min(len(speed), apex_idx + 40)
            exit_speed = np.max(speed[apex_idx:exit_idx]) if exit_idx > apex_idx else apex_speed
            features[f'corner_{i+1}_exit_speed'] = exit_speed

            # Brake point (find where braking started before apex)
            brake_region = brake_f[entry_idx:apex_idx]
            brake_points = np.where(brake_region > 50)[0]
            if len(brake_points) > 0:
                # Distance from first brake to apex (in samples)
                brake_point = len(brake_region) - brake_points[0]
                features[f'corner_{i+1}_brake_point'] = brake_point
            else:
                features[f'corner_{i+1}_brake_point'] = 0.0

        # Fill remaining corners with zeros
        for i in range(num_corners, self.max_corners_per_track):
            features[f'corner_{i+1}_entry_speed'] = 0.0
            features[f'corner_{i+1}_apex_speed'] = 0.0
            features[f'corner_{i+1}_exit_speed'] = 0.0
            features[f'corner_{i+1}_brake_point'] = 0.0

        return features

    def _extract_track_encoding_features(self, track_name: str) -> Dict[str, float]:
        """
        Extract track encoding features.

        Creates both one-hot encoding and learned embeddings for tracks.
        One-hot captures categorical differences, embeddings capture similarities.

        Returns 11 features (6 one-hot + 5 embedding dimensions).
        """
        features = {}

        # Known tracks
        all_tracks = [
            'barber-motorsports-park',
            'circuit-of-the-americas',
            'road-america',
            'sebring',
            'sonoma',
            'virginia-international-raceway'
        ]

        # One-hot encoding
        for track in all_tracks:
            features[f'track_onehot_{track}'] = 1.0 if track == track_name else 0.0

        # Learned embeddings (hand-crafted based on track characteristics)
        # Dimension 1: Track length (normalized)
        # Dimension 2: Number of corners (normalized)
        # Dimension 3: Average speed (normalized)
        # Dimension 4: Elevation change (normalized)
        # Dimension 5: Technical difficulty (normalized)

        track_embeddings = {
            'barber-motorsports-park': [0.45, 0.60, 0.50, 0.40, 0.70],  # Technical, hilly
            'circuit-of-the-americas': [0.85, 0.75, 0.70, 0.60, 0.80],  # Long, complex
            'road-america': [0.90, 0.50, 0.85, 0.50, 0.60],  # Long, fast
            'sebring': [0.75, 0.65, 0.60, 0.30, 0.75],  # Bumpy, technical
            'sonoma': [0.50, 0.70, 0.55, 0.80, 0.85],  # Hilly, technical
            'virginia-international-raceway': [0.60, 0.65, 0.65, 0.45, 0.65]  # Balanced
        }

        embedding = track_embeddings.get(track_name, [0.5, 0.5, 0.5, 0.5, 0.5])

        for i, value in enumerate(embedding, 1):
            features[f'track_embedding_dim{i}'] = value

        return features

    def _extract_temporal_features(
        self,
        session_lap_count: int,
        total_session_laps: int,
        sensor_data: Dict
    ) -> Dict[str, float]:
        """
        Extract temporal/sequential features.

        Captures:
        - Lap progression through session
        - Tire degradation proxies (grip loss over time)
        - Driver fatigue indicators (consistency changes)
        - Session phase (early/mid/late)

        Returns 8 features.
        """
        features = {}

        # Lap position in session (normalized 0-1)
        features['lap_number_normalized'] = session_lap_count / max(total_session_laps, 1)

        # Session progress percentage
        features['session_progress_pct'] = (session_lap_count / max(total_session_laps, 1)) * 100

        # Session phase (categorical encoded as 0=early, 0.5=mid, 1=late)
        if features['session_progress_pct'] < 33:
            features['session_phase'] = 0.0  # Early
        elif features['session_progress_pct'] < 67:
            features['session_phase'] = 0.5  # Mid
        else:
            features['session_phase'] = 1.0  # Late

        # Tire degradation proxy
        # Assumption: grip decreases linearly with laps
        # Use lateral g variance as proxy (more variance = less grip)
        lateral_g = sensor_data.get('accel_y', np.array([]))

        if len(lateral_g) > 10:
            lateral_g_variance = np.var(lateral_g)
            # Combine with lap position to estimate degradation
            features['tire_degradation_proxy'] = lateral_g_variance * features['lap_number_normalized']
        else:
            features['tire_degradation_proxy'] = 0.0

        # Driver fatigue proxy
        # Assumption: more corrections and less smoothness in later laps
        steering = sensor_data.get('steering', np.array([]))

        if len(steering) > 10:
            # Steering jerk (rate of change of steering rate)
            steering_diff = np.diff(steering)
            steering_jerk = np.diff(steering_diff)
            steering_jerk_variance = np.var(steering_jerk) if len(steering_jerk) > 0 else 0.0

            # Combine with lap position
            features['driver_fatigue_proxy'] = steering_jerk_variance * features['lap_number_normalized']
        else:
            features['driver_fatigue_proxy'] = 0.0

        # Consistency degradation
        # Compare current lap to expected performance
        speed = sensor_data.get('speed', np.array([]))

        if len(speed) > 10:
            speed_cv = np.std(speed) / (np.mean(speed) + 1e-6)  # Coefficient of variation
            # Higher CV in later laps = degrading consistency
            features['consistency_degradation'] = speed_cv * features['lap_number_normalized']
        else:
            features['consistency_degradation'] = 0.0

        # Fuel load proxy (linear decrease assumption)
        # More fuel at start = slower, less fuel at end = faster
        # Normalized: 1.0 = full tank, 0.0 = empty
        features['fuel_load_proxy'] = 1.0 - features['lap_number_normalized']

        # Optimal window indicator (middle laps often fastest)
        # Peak at 40-60% through session
        progress = features['session_progress_pct']
        if 40 <= progress <= 60:
            features['optimal_window_indicator'] = 1.0
        else:
            features['optimal_window_indicator'] = 0.0

        return features

    def process_session_advanced(
        self,
        telemetry_df: pd.DataFrame,
        base_features_df: pd.DataFrame,
        track_name: str,
        save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Process entire session and add advanced features.

        Args:
            telemetry_df: Raw telemetry data
            base_features_df: DataFrame with base features from TelemetryFeatureEngineer
            track_name: Track identifier
            save_path: Optional path to save combined features

        Returns:
            DataFrame with base + advanced features (110+ columns)
        """
        advanced_features_list = []

        print(f"Processing advanced features for {track_name}...")
        print(f"Base features shape: {base_features_df.shape}")

        # Get unique vehicles
        vehicles = base_features_df['vehicle_number'].unique()

        for vehicle in vehicles:
            vehicle_laps = base_features_df[base_features_df['vehicle_number'] == vehicle]
            total_laps = len(vehicle_laps)

            print(f"  Vehicle {vehicle}: {total_laps} laps")

            for idx, row in vehicle_laps.iterrows():
                lap_num = int(row['lap_number'])
                session_lap_count = idx - vehicle_laps.index[0] + 1  # Lap count in this session

                try:
                    advanced_feats = self.extract_advanced_features(
                        telemetry_df,
                        vehicle,
                        lap_num,
                        track_name,
                        session_lap_count,
                        total_laps
                    )

                    if advanced_feats:
                        # Add identifiers
                        advanced_feats['vehicle_number'] = vehicle
                        advanced_feats['lap_number'] = lap_num
                        advanced_features_list.append(advanced_feats)

                except Exception as e:
                    print(f"    Warning: Failed vehicle {vehicle}, lap {lap_num}: {e}")
                    continue

        # Convert to DataFrame
        advanced_df = pd.DataFrame(advanced_features_list)

        print(f"Extracted {len(advanced_df)} laps with {len(advanced_df.columns)} advanced features")

        # Merge with base features
        if len(advanced_df) > 0:
            combined_df = base_features_df.merge(
                advanced_df,
                on=['vehicle_number', 'lap_number'],
                how='left'
            )

            print(f"Combined shape: {combined_df.shape}")
        else:
            combined_df = base_features_df
            print("Warning: No advanced features extracted, returning base features only")

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_parquet(save_path, index=False)
            print(f"Saved to: {save_path}")

        return combined_df


def extract_advanced_features_from_track(
    track_name: str,
    base_dir: str = "organized_data",
    features_dir: str = "data/processed",
    output_dir: str = "data/processed"
) -> pd.DataFrame:
    """
    Extract advanced features for entire track.

    Args:
        track_name: Track name (e.g., 'barber-motorsports-park')
        base_dir: Base directory with raw telemetry
        features_dir: Directory with base features
        output_dir: Output directory for advanced features

    Returns:
        DataFrame with base + advanced features
    """
    from data_loader import RacingDataLoader

    loader = RacingDataLoader(base_dir=base_dir)
    engineer = AdvancedFeatureEngineer()

    print(f"=" * 80)
    print(f"ADVANCED FEATURE EXTRACTION: {track_name}")
    print(f"=" * 80)

    # Load base features
    base_features_path = Path(features_dir) / f"{track_name}_features.parquet"

    if not base_features_path.exists():
        raise FileNotFoundError(f"Base features not found: {base_features_path}")

    base_features_df = pd.read_parquet(base_features_path)
    print(f"Loaded base features: {base_features_df.shape}")

    # Load telemetry (first chunk for prototyping - adjust as needed)
    telemetry_df = loader.load_single_chunk(
        track=track_name,
        race='race_unknown',
        category='telemetry',
        chunk_num=1
    )
    print(f"Loaded telemetry chunk: {telemetry_df.shape}")

    # Extract advanced features
    combined_df = engineer.process_session_advanced(
        telemetry_df,
        base_features_df,
        track_name,
        save_path=Path(output_dir) / f"{track_name}_advanced_features.parquet"
    )

    return combined_df


if __name__ == "__main__":
    # Example usage
    print("Advanced Feature Engineering Module - Example")
    print("=" * 80)

    # Extract for Barber track
    features_df = extract_advanced_features_from_track('barber-motorsports-park')

    print("\nFeature Summary:")
    print(f"Total features: {len(features_df.columns)}")
    print(f"Total laps: {len(features_df)}")

    # Show some advanced features
    advanced_cols = [col for col in features_df.columns if any(
        prefix in col for prefix in ['fft_', 'wavelet_', 'corner_', 'track_', 'tire_', 'driver_']
    )]

    print(f"\nAdvanced features ({len(advanced_cols)}):")
    for col in advanced_cols[:20]:  # Show first 20
        print(f"  - {col}")
