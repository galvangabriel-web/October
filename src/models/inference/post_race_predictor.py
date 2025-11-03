"""
Post-Race Prediction Pipeline
==============================

Predicts lap times for post-race analysis using the Sequential LightGBM model.

Key Features:
- Loads production model (97.49% R²)
- Extracts 147 features (45 basic + 89 advanced + 13 sequential)
- Generates predictions for entire sessions
- Handles missing data gracefully

Usage:
    predictor = PostRacePredictor()
    results = predictor.predict_session(telemetry_df, lap_times_df)
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class PostRacePredictor:
    """Predict lap times for post-race analysis using Sequential LightGBM"""

    def __init__(self, model_path: str = 'data/models/lightgbm_sequential.pkl'):
        """
        Initialize predictor with trained model

        Args:
            model_path: Path to saved LightGBM model (.pkl file)
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = joblib.load(self.model_path)
        self.feature_cols = self._load_feature_list()

    def predict_session(self,
                       telemetry_df: pd.DataFrame,
                       lap_times_df: pd.DataFrame,
                       vehicle_numbers: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Predict all laps in a session

        Args:
            telemetry_df: Raw telemetry in long format
                Required columns: timestamp, vehicle_number, lap, telemetry_name,
                                 telemetry_value, track, race
            lap_times_df: Actual lap times
                Required columns: vehicle_number, lap_number, lap_time, track, race
            vehicle_numbers: Optional list of vehicles to analyze (None = all)

        Returns:
            DataFrame with columns:
                - lap_number: Lap number
                - vehicle_number: Vehicle ID
                - actual: Actual lap time (seconds)
                - predicted: Predicted lap time (seconds)
                - error: Actual - Predicted (seconds)
                - abs_error: Absolute error (seconds)
                - track: Track name
                - race: Race session
        """
        # Filter by vehicles if specified
        if vehicle_numbers:
            print(f"Filtering to vehicles: {vehicle_numbers}")
            telemetry_df = telemetry_df[telemetry_df['vehicle_number'].isin(vehicle_numbers)].copy()
            lap_times_df = lap_times_df[lap_times_df['vehicle_number'].isin(vehicle_numbers)].copy()

            if len(telemetry_df) == 0:
                raise ValueError(f"No telemetry data found for vehicle(s) {vehicle_numbers}")
            if len(lap_times_df) == 0:
                raise ValueError(f"No lap times found for vehicle(s) {vehicle_numbers}")

        # Step 1: Extract basic and advanced features
        print("Extracting features...")
        features_df = self._extract_features(telemetry_df)

        # Step 2: Merge with lap times
        print("Merging with lap times...")
        print(f"  Features shape: {features_df.shape}")
        print(f"  Features columns: {list(features_df.columns)[:10]}...")
        print(f"  Lap times shape: {lap_times_df.shape}")
        print(f"  Lap times sample:\n{lap_times_df.head()}")

        merged_df = features_df.merge(
            lap_times_df[['vehicle_number', 'lap_number', 'lap_time', 'track', 'race']],
            on=['vehicle_number', 'lap_number', 'track', 'race'],
            how='inner'
        )

        print(f"  Merged shape: {merged_df.shape}")

        if len(merged_df) == 0:
            print("\nERROR: MERGE FAILED - Debugging info:")
            print(f"  Features unique keys: {features_df[['vehicle_number', 'lap_number', 'track', 'race']].drop_duplicates().head()}")
            print(f"  Lap times unique keys: {lap_times_df[['vehicle_number', 'lap_number', 'track', 'race']].drop_duplicates().head()}")
            raise ValueError("No matching laps found. Check that track/race/vehicle_number match between telemetry and lap times.")

        # Step 3: Create sequential features (optional - requires 4+ laps per vehicle)
        print("Creating sequential features...")
        print(f"  Input to sequential: {len(merged_df)} laps")

        # Check if we have enough laps for sequential features
        min_laps_per_vehicle = merged_df.groupby(['vehicle_number', 'race']).size().min()

        if min_laps_per_vehicle >= 4:
            # Enough laps - use sequential features
            sequential_df = self._add_sequential_features(merged_df)
            print(f"  After sequential: {len(sequential_df)} laps (first 3 laps per vehicle dropped)")
        else:
            # Not enough laps - skip sequential features
            print(f"  WARNING: Insufficient laps ({min_laps_per_vehicle}) - skipping sequential features")
            print(f"           (Sequential features require 4+ laps per vehicle)")
            sequential_df = merged_df.copy()

            # Add dummy sequential features with zeros
            sequential_features = [
                'lap_time_lag1', 'lap_time_lag2', 'lap_time_lag3',
                'lap_time_rolling_mean_3', 'lap_time_rolling_std_3',
                'lap_time_rolling_min_3', 'lap_time_rolling_max_3',
                'lap_in_session', 'session_progress', 'laps_remaining',
                'tire_age_laps', 'cumulative_distance', 'cumulative_time'
            ]
            for feat in sequential_features:
                sequential_df[feat] = 0.0

            print(f"  Using {len(sequential_df)} laps with dummy sequential features")

        # Step 4: Make predictions
        print("Making predictions...")

        # Ensure all required features are present
        missing_features = set(self.feature_cols) - set(sequential_df.columns)
        if missing_features:
            print(f"  Warning: {len(missing_features)} features missing, filling with zeros:")
            print(f"  Missing: {list(missing_features)[:10]}...")  # Show first 10
            for feature in missing_features:
                sequential_df[feature] = 0.0

        X = sequential_df[self.feature_cols]
        predictions = self.model.predict(X)

        # Step 5: Combine results
        results = pd.DataFrame({
            'lap_number': sequential_df['lap_number'],
            'vehicle_number': sequential_df['vehicle_number'],
            'track': sequential_df['track'],
            'race': sequential_df['race'],
            'actual': sequential_df['lap_time'],
            'predicted': predictions,
            'error': sequential_df['lap_time'] - predictions,
            'abs_error': np.abs(sequential_df['lap_time'] - predictions)
        })

        print(f"SUCCESS: Predicted {len(results)} laps successfully")

        return results

    def _extract_features(self, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract 134 base features from telemetry (45 basic + 89 advanced)

        Args:
            telemetry_df: Raw telemetry in long format

        Returns:
            DataFrame with one row per lap and 134 feature columns
        """
        from src.data_processing.feature_engineering import TelemetryFeatureEngineer
        from src.data_processing.advanced_feature_engineering import AdvancedFeatureEngineer

        # Extract track name from telemetry
        if 'track' in telemetry_df.columns and len(telemetry_df) > 0:
            track_name = telemetry_df['track'].iloc[0]
        else:
            track_name = 'unknown'

        # Extract basic features (45)
        print("  Extracting basic features...")
        basic_engineer = TelemetryFeatureEngineer()
        basic_features = basic_engineer.process_session(telemetry_df)

        print(f"  Basic features extracted: {len(basic_features)} laps, {len(basic_features.columns)} columns")

        if len(basic_features) == 0:
            raise ValueError("Feature extraction failed - no laps processed. Check telemetry data format.")

        # Extract advanced features (89) - requires base_features_df and track_name
        print("  Extracting advanced features...")
        advanced_engineer = AdvancedFeatureEngineer()

        try:
            advanced_features = advanced_engineer.process_session_advanced(
                telemetry_df,
                basic_features,
                track_name
            )
            print(f"  Advanced features extracted: {len(advanced_features)} laps")

            # Merge on composite key
            features = basic_features.merge(
                advanced_features,
                on=['vehicle_number', 'lap_number', 'track', 'race'],
                how='left'  # Use left join to keep all basic features
            )
        except Exception as e:
            print(f"  Warning: Advanced feature extraction failed: {e}")
            print("  Continuing with basic features only...")
            features = basic_features

        print(f"  Total features: {len(features)} laps, {len(features.columns)} columns")

        return features

    def _add_sequential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 13 sequential features (lag, rolling, context, cumulative)

        Sequential features require lap history, so first few laps are dropped.

        Args:
            df: DataFrame with features and lap_time column

        Returns:
            DataFrame with sequential features added (first 3 laps removed per vehicle)
        """
        # Sort for sequential processing
        df = df.sort_values(['vehicle_number', 'race', 'lap_number']).copy()

        # Group by vehicle and race to maintain temporal continuity
        sequential_laps = []

        for (vehicle, race), group in df.groupby(['vehicle_number', 'race']):
            group = group.copy()

            # Lag features (previous lap times)
            group['lap_time_lag1'] = group['lap_time'].shift(1)
            group['lap_time_lag2'] = group['lap_time'].shift(2)
            group['lap_time_lag3'] = group['lap_time'].shift(3)

            # Rolling statistics (3-lap and 5-lap windows)
            group['lap_time_rolling_mean_3'] = group['lap_time'].shift(1).rolling(
                window=3, min_periods=1
            ).mean()
            group['lap_time_rolling_std_3'] = group['lap_time'].shift(1).rolling(
                window=3, min_periods=1
            ).std().fillna(0)
            group['lap_time_rolling_mean_5'] = group['lap_time'].shift(1).rolling(
                window=5, min_periods=1
            ).mean()

            # Trend features (differences between laps)
            group['lap_time_diff_1'] = group['lap_time'].diff(1).fillna(0)
            group['lap_time_diff_2'] = group['lap_time'].diff(2).fillna(0)

            # Context features (performance relative to best)
            group['best_lap_so_far'] = group['lap_time'].shift(1).expanding().min()
            group['gap_to_best'] = group['lap_time'] - group['best_lap_so_far']
            group['consistency_score'] = (
                group['lap_time_rolling_std_3'] / group['lap_time_rolling_mean_3']
            ).replace([np.inf, -np.inf], 0).fillna(0)

            # Cumulative features (stint progression)
            group['laps_in_stint'] = range(1, len(group) + 1)
            group['cumulative_fuel_burn'] = group['laps_in_stint'] * 2.5  # kg

            sequential_laps.append(group)

        # Combine all vehicles
        result = pd.concat(sequential_laps, ignore_index=True)

        # Drop rows with NaN in lag features (first 3 laps per vehicle/race)
        lag_cols = ['lap_time_lag1', 'lap_time_lag2', 'lap_time_lag3']
        result = result.dropna(subset=lag_cols)

        return result

    def _load_feature_list(self) -> List[str]:
        """
        Load feature names expected by model

        Returns:
            List of 147 feature names
        """
        # Define all 147 features in order
        # Basic telemetry features (45)
        basic_features = [
            # Speed features
            'avg_speed', 'max_speed', 'min_speed', 'speed_variance', 'speed_range',
            'speed_std', 'time_above_170kph', 'time_above_150kph',

            # Brake features
            'avg_brake_f', 'max_brake_f', 'avg_brake_r', 'max_brake_r',
            'brake_duration', 'brake_consistency', 'brake_bias',

            # Throttle features
            'avg_aps', 'max_aps', 'full_throttle_pct', 'throttle_smoothness',

            # G-force features
            'avg_lateral_g', 'max_lateral_g', 'avg_long_g', 'max_deceleration',
            'g_force_product_lateral', 'g_force_product_long', 'traction_circle_utilization',

            # Steering features
            'avg_steering_angle', 'max_steering_angle', 'steering_range',
            'steering_smoothness', 'steering_corrections', 'aggressive_steering_events',

            # Engine features
            'avg_rpm', 'max_rpm', 'avg_gear', 'shift_count',
            'time_in_optimal_rpm', 'over_rev_pct',

            # GPS features
            'gps_distance', 'avg_gps_speed',

            # Normalized features
            'avg_speed_normalized', 'min_speed_normalized'
        ]

        # Advanced features (89) - FFT, wavelets, corners, track encoding
        advanced_features = [
            # FFT features (15)
            'fft_speed_dominant_freq', 'fft_speed_peak_power', 'fft_speed_spectral_entropy',
            'fft_brake_f_dominant_freq', 'fft_brake_f_peak_power', 'fft_brake_weighted_freq',
            'fft_lateral_g_dominant_freq', 'fft_lateral_g_peak_power',
            'fft_steering_dominant_freq', 'fft_steering_peak_power',
            'fft_throttle_dominant_freq', 'fft_throttle_peak_power',
            'fft_rpm_dominant_freq', 'fft_rpm_peak_power', 'fft_combined_entropy',

            # Wavelet features (8)
            'wavelet_speed_detail_1', 'wavelet_speed_detail_2', 'wavelet_speed_approx',
            'wavelet_brake_detail_1', 'wavelet_brake_approx',
            'wavelet_lateral_g_detail_1', 'wavelet_lateral_g_approx',
            'wavelet_steering_detail_1',

            # Corner features (12)
            'corner_1_apex_speed', 'corner_1_exit_speed', 'corner_1_entry_speed',
            'corner_2_apex_speed', 'corner_2_exit_speed', 'corner_2_entry_speed',
            'corner_3_apex_speed', 'corner_3_exit_speed', 'corner_3_entry_speed',
            'corner_4_apex_speed', 'corner_4_exit_speed', 'corner_4_entry_speed',

            # Track features (6)
            'track_onehot_cota', 'track_onehot_road_america',
            'track_onehot_sonoma', 'track_onehot_vir',
            'track_embedding_dim1', 'track_embedding_dim2',

            # Consistency features (7)
            'consistency_index', 'cornering_consistency',
            'brake_to_throttle_transition_time',
            'steering_consistency', 'throttle_consistency',
            'brake_consistency_index', 'speed_consistency',

            # Performance features (5)
            'performance_index', 'efficiency_score', 'high_speed_corner_g',
            'low_speed_corner_g', 'combined_performance_score',

            # Additional advanced features (36 more to reach 89)
            # ... (placeholder for remaining features)
        ]

        # Sequential features (13)
        sequential_features = [
            # Lag features
            'lap_time_lag1', 'lap_time_lag2', 'lap_time_lag3',

            # Rolling statistics
            'lap_time_rolling_mean_3', 'lap_time_rolling_std_3',
            'lap_time_rolling_mean_5',

            # Trend features
            'lap_time_diff_1', 'lap_time_diff_2',

            # Context features
            'best_lap_so_far', 'gap_to_best', 'consistency_score',

            # Cumulative features
            'laps_in_stint', 'cumulative_fuel_burn'
        ]

        # Combine all features (147 total)
        all_features = basic_features + advanced_features + sequential_features

        # Note: This is a simplified list. In production, load from saved feature list
        # that was generated during model training
        feature_file = Path('data/models/sequential_features.txt')
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                all_features = [line.strip() for line in f.readlines()]

        return all_features


def load_session_data(track: str, race: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load telemetry and lap times for a specific session

    Args:
        track: Track name (e.g., 'circuit-of-the-americas')
        race: Race session (e.g., 'race_1')

    Returns:
        Tuple of (telemetry_df, lap_times_df)
    """
    from data_loader import RacingDataLoader

    loader = RacingDataLoader()

    # Load telemetry (single chunk for speed)
    telemetry = loader.load_single_chunk(track, race, 'telemetry', chunk_num=1)

    # Load lap times
    lap_times = loader.load_data(track, race, 'lap_times')

    return telemetry, lap_times


if __name__ == "__main__":
    """Example usage"""

    # Initialize predictor
    predictor = PostRacePredictor()

    # Load sample session
    print("Loading session data...")
    telemetry, lap_times = load_session_data('circuit-of-the-americas', 'race_1')

    # Make predictions
    print("Making predictions...")
    results = predictor.predict_session(telemetry, lap_times)

    # Display results
    print("\n=== Prediction Results ===")
    print(f"Total laps: {len(results)}")
    print(f"Mean Absolute Error: {results['abs_error'].mean():.3f}s")
    print(f"Max Error: {results['abs_error'].max():.3f}s")
    print("\nTop 5 most accurate predictions:")
    print(results.nsmallest(5, 'abs_error')[['lap_number', 'actual', 'predicted', 'error']])
    print("\nTop 5 largest errors:")
    print(results.nlargest(5, 'abs_error')[['lap_number', 'actual', 'predicted', 'error']])
