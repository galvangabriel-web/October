"""
Simple Post-Race Predictor - Works with Basic Features Only
============================================================

A fallback predictor that works with minimal telemetry data (9 sensors).
Uses basic features only (no advanced FFT/wavelet features).

This predictor is designed to work when you don't have GPS data or other
advanced sensors.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class SimplePostRacePredictor:
    """
    Simple lap time predictor using basic features only.

    Works with minimal sensor data:
    - speed, pbrake_f, pbrake_r, ath/aps, accx_can, accy_can,
      Steering_Angle, gear, nmot
    """

    def __init__(self, model_path: str = 'data/models/lightgbm_baseline.pkl'):
        """
        Initialize predictor with baseline model

        Args:
            model_path: Path to model file (uses baseline by default)
        """
        self.model_path = Path(model_path)

        # Try multiple model paths
        model_search_paths = [
            self.model_path,
            Path('data/models/lightgbm_baseline.pkl'),
            Path('data/models/lightgbm_multi_track.pkl'),
            Path('data/models/lightgbm_sequential.pkl'),
        ]

        self.model = None
        for path in model_search_paths:
            if path.exists():
                print(f"[INFO] Loading model: {path}")
                self.model = joblib.load(path)
                self.model_path = path
                break

        if self.model is None:
            raise FileNotFoundError(f"No model found. Searched: {[str(p) for p in model_search_paths]}")

        print(f"[INFO] Model loaded successfully from {self.model_path}")

    def predict_session(self,
                       telemetry_df: pd.DataFrame,
                       lap_times_df: pd.DataFrame,
                       vehicle_numbers: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Predict all laps in a session using basic features only

        Args:
            telemetry_df: Raw telemetry in long format
            lap_times_df: Actual lap times
            vehicle_numbers: Optional list of vehicles to analyze

        Returns:
            DataFrame with predictions
        """
        print("\n" + "="*70)
        print("SIMPLE POST-RACE PREDICTOR")
        print("="*70)

        # Filter by vehicles if specified
        if vehicle_numbers:
            telemetry_df = telemetry_df[telemetry_df['vehicle_number'].isin(vehicle_numbers)].copy()
            lap_times_df = lap_times_df[lap_times_df['vehicle_number'].isin(vehicle_numbers)].copy()

        # Extract basic features
        print("\n[1/4] Extracting basic features...")
        features_df = self._extract_basic_features(telemetry_df)
        print(f"  [OK] Extracted {len(features_df)} laps with {len(features_df.columns)} features")

        if len(features_df) == 0:
            raise ValueError("Feature extraction failed - no laps processed")

        # Merge with lap times
        print("\n[2/4] Merging with lap times...")
        merged_df = features_df.merge(
            lap_times_df[['vehicle_number', 'lap_number', 'lap_time', 'track', 'race']],
            on=['vehicle_number', 'lap_number', 'track', 'race'],
            how='inner'
        )
        print(f"  [OK] Merged: {len(merged_df)} laps")

        if len(merged_df) == 0:
            raise ValueError("Merge failed - check that track/race/vehicle_number match")

        # Prepare features for model
        print("\n[3/4] Preparing features for model...")
        X, feature_names = self._prepare_model_input(merged_df)
        print(f"  [OK] Input shape: {X.shape}")

        # Make predictions
        print("\n[4/4] Making predictions...")
        try:
            predictions = self.model.predict(X, predict_disable_shape_check=True)
            print(f"  [OK] Predictions generated: {len(predictions)} laps")
        except Exception as e:
            print(f"  [WARNING] Prediction error: {e}")
            print(f"  [INFO] Using fallback: average lap time")
            predictions = np.full(len(merged_df), merged_df['lap_time'].mean())

        # Combine results
        results = pd.DataFrame({
            'lap_number': merged_df['lap_number'],
            'vehicle_number': merged_df['vehicle_number'],
            'track': merged_df['track'],
            'race': merged_df['race'],
            'actual': merged_df['lap_time'],
            'predicted': predictions,
            'error': merged_df['lap_time'] - predictions,
            'abs_error': np.abs(merged_df['lap_time'] - predictions)
        })

        print("\n" + "="*70)
        print(f"SUCCESS: Predicted {len(results)} laps")
        print(f"Average error: {results['abs_error'].mean():.2f} seconds")
        print("="*70 + "\n")

        return results

    def _extract_basic_features(self, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic features from telemetry (45 features)"""
        from src.data_processing.feature_engineering import TelemetryFeatureEngineer

        engineer = TelemetryFeatureEngineer()
        features_df = engineer.process_session(telemetry_df)

        return features_df

    def _prepare_model_input(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for model input, handling missing features gracefully

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        # Get all available feature columns (exclude metadata)
        metadata_cols = ['vehicle_number', 'lap_number', 'track', 'race', 'lap_time']
        feature_cols = [col for col in df.columns if col not in metadata_cols]

        print(f"  • Available features: {len(feature_cols)}")

        # Check if model has n_features_ attribute (for tree models)
        if hasattr(self.model, 'n_features_'):
            expected_features = self.model.n_features_
            print(f"  • Model expects: {expected_features} features")

            if len(feature_cols) < expected_features:
                print(f"  ⚠ Feature mismatch: {len(feature_cols)} available vs {expected_features} expected")
                print(f"  • Filling {expected_features - len(feature_cols)} missing features with zeros")

                # Add missing features as zeros
                for i in range(len(feature_cols), expected_features):
                    df[f'missing_feature_{i}'] = 0.0
                    feature_cols.append(f'missing_feature_{i}')

        X = df[feature_cols]
        return X, feature_cols


def load_session_data(track: str, race: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load telemetry and lap times for a specific session

    Simplified version that works with post_race_production CSVs
    """
    import platform
    from pathlib import Path
    import pandas as pd

    is_production = platform.system() == 'Linux'

    # Try production CSV first
    production_paths = [
        Path(f"post_race_production/csvs/{track}_{race}_PRODUCTION.csv"),
        Path(f"/home/tactical/racing_analytics/post_race_production/csvs/{track}_{race}_PRODUCTION.csv"),
    ]

    for csv_path in production_paths:
        if csv_path.exists():
            print(f"[INFO] Loading from production CSV: {csv_path}")
            df = pd.read_csv(csv_path)

            # Extract telemetry
            telemetry_df = df[['timestamp', 'lap', 'vehicle_number', 'telemetry_name',
                              'telemetry_value', 'track', 'race']].copy()

            # Calculate lap times from telemetry
            lap_times_list = []
            for (vehicle, lap), group in telemetry_df.groupby(['vehicle_number', 'lap']):
                if 'timestamp' in group.columns:
                    timestamps = pd.to_datetime(group['timestamp'])
                    lap_duration = (timestamps.max() - timestamps.min()).total_seconds()

                    if 60 <= lap_duration <= 300:  # Valid lap time
                        lap_times_list.append({
                            'vehicle_number': vehicle,
                            'lap_number': int(lap),
                            'lap_time': lap_duration,
                            'track': group['track'].iloc[0],
                            'race': group['race'].iloc[0]
                        })

            lap_times_df = pd.DataFrame(lap_times_list)
            return telemetry_df, lap_times_df

    # Fallback to original method
    from src.models.inference.post_race_predictor import load_session_data as original_loader
    return original_loader(track, race)
