"""
Weather-Enhanced LightGBM Model Training Pipeline
=================================================

Extends the baseline LightGBM model by adding weather features:
- Air temperature
- Track temperature
- Humidity
- Wind speed and direction
- Rain conditions
- Derived weather features (grip index, temperature deltas)

Target Performance:
- R² improvement > 2% vs baseline
- Better predictions in varying weather conditions
- Temperature-adjusted lap time estimates

Usage:
    python -m src.models.baseline.train_lightgbm_weather --tracks all --save-model
"""

import sys
import os
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from data_loader import RacingDataLoader
from src.data_processing.feature_engineering import TelemetryFeatureEngineer
from src.data_processing.weather_loader import WeatherDataLoader


class WeatherFeatureEngineer:
    """
    Add weather-based features to telemetry features.

    Creates ~10-15 weather features including:
    - Direct: air_temp, track_temp, humidity, wind_speed, rain
    - Derived: grip_index, temp_deltas, wind_impact
    """

    def __init__(self):
        """Initialize with optimal racing conditions."""
        # Optimal racing conditions (based on domain knowledge)
        self.OPTIMAL_CONDITIONS = {
            'air_temp': 20.0,  # °C (68°F)
            'track_temp': 25.0,  # °C (77°F)
            'humidity': 50.0,  # %
            'wind_speed': 5.0,  # km/h (light wind)
        }

    def calculate_grip_index(self, row: pd.Series) -> float:
        """
        Calculate track grip index based on weather conditions.

        Higher grip = better lap times. Range: 0-100

        Factors:
        - Optimal track temp (25°C) = best grip
        - Rain = major grip reduction (-50%)
        - High humidity = slight reduction (-10%)
        - Extreme temps = reduced grip
        """
        grip = 100.0

        # Track temperature effect (parabolic - optimal at 25°C)
        if pd.notna(row.get('TRACK_TEMP')):
            temp_delta = abs(row['TRACK_TEMP'] - 25.0)
            grip -= min(temp_delta * 1.5, 30)  # Max 30 point penalty

        # Rain effect (major impact)
        if row.get('RAIN', 0) > 0:
            grip *= 0.5  # 50% grip reduction

        # Humidity effect (minor)
        if pd.notna(row.get('HUMIDITY')):
            if row['HUMIDITY'] > 70:
                grip -= (row['HUMIDITY'] - 70) * 0.3

        return max(0, min(100, grip))

    def calculate_wind_components(self, wind_speed: float, wind_direction: float,
                                   track_direction: float = 0) -> Dict[str, float]:
        """
        Calculate headwind and crosswind components.

        Args:
            wind_speed: Wind speed in km/h
            wind_direction: Wind direction in degrees
            track_direction: Average track direction (default 0)

        Returns:
            Dict with headwind and crosswind components
        """
        if pd.isna(wind_speed) or pd.isna(wind_direction):
            return {'headwind': 0.0, 'crosswind': 0.0}

        # Convert to radians
        angle_diff = np.radians(wind_direction - track_direction)

        # Calculate components
        headwind = wind_speed * np.cos(angle_diff)
        crosswind = abs(wind_speed * np.sin(angle_diff))

        return {'headwind': headwind, 'crosswind': crosswind}

    def add_weather_features(self, features_df: pd.DataFrame,
                            weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge weather data with telemetry features and create derived features.

        Uses session-based matching since weather and telemetry data are from
        different racing events (different dates).

        Args:
            features_df: DataFrame with lap-level telemetry features
            weather_df: DataFrame with weather data (from WeatherDataLoader)

        Returns:
            Enhanced DataFrame with weather features added
        """
        print(f"\n{'='*60}")
        print("Adding Weather Features (Session-Based Matching)")
        print(f"{'='*60}")
        print(f"Telemetry features: {len(features_df)} laps")
        print(f"Weather data: {len(weather_df)} records")

        # Map weather session names to lap session codes
        # Weather sessions: "Race 1", "Race 2", "Practice 1", etc.
        # Lap sessions: "R1", "R2", "P1", etc.
        session_mapping = {
            'Race 1': 'R1',
            'Race 2': 'R2',
            'Race%201': 'R1',  # URL-encoded version
            'Race%202': 'R2',
            'Practice 1': 'P1',
            'Practice 2': 'P2',
            'Practice%201': 'P1',
            'Practice%202': 'P2',
            'Qualifying': 'Q',
            'Qualify 1': 'Q1',
            'Qualify 2': 'Q2',
            'Qualify%201': 'Q1',
            'Qualify%202': 'Q2',
            'Test Session 1': 'T1',
            'Test Session 2': 'T2',
            'Test%20Session%201': 'T1',
            'Test%20Session%202': 'T2',
        }

        # Map weather sessions
        weather_df['session_code'] = weather_df['session'].map(session_mapping)

        print(f"\nWeather sessions found:")
        for session in weather_df['session'].unique():
            mapped = session_mapping.get(session, 'UNKNOWN')
            count = len(weather_df[weather_df['session'] == session])
            print(f"  {session:30s} -> {mapped:5s} ({count:3d} records)")

        # Calculate average weather conditions per session
        weather_summary = weather_df.groupby('session_code').agg({
            'AIR_TEMP': 'mean',
            'TRACK_TEMP': 'mean',
            'HUMIDITY': 'mean',
            'WIND_SPEED': 'mean',
            'WIND_DIRECTION': 'mean',
            'RAIN': 'max'  # Use max to detect if it rained at all during session
        }).reset_index()

        print(f"\nSession-averaged weather conditions:")
        print(weather_summary.to_string())

        # Check if features_df has session information
        if 'session' not in features_df.columns:
            print(f"\n⚠️  WARNING: No 'session' column in features_df")
            print(f"   Available columns: {features_df.columns.tolist()}")
            # Try to infer from other columns or use default
            features_df['session'] = 'R1'  # Default to R1 if unknown
            print(f"   Defaulting all laps to session 'R1'")

        # Merge weather summary with features based on session
        merged_df = features_df.merge(
            weather_summary,
            left_on='session',
            right_on='session_code',
            how='left'
        )

        # Calculate derived weather features
        print("Calculating derived weather features...")

        # 1. Temperature deltas from optimal
        merged_df['air_temp_delta'] = merged_df['AIR_TEMP'] - self.OPTIMAL_CONDITIONS['air_temp']
        merged_df['track_temp_delta'] = merged_df['TRACK_TEMP'] - self.OPTIMAL_CONDITIONS['track_temp']
        merged_df['track_air_temp_diff'] = merged_df['TRACK_TEMP'] - merged_df['AIR_TEMP']

        # 2. Temperature categories (hot, optimal, cold)
        merged_df['temp_category'] = pd.cut(
            merged_df['TRACK_TEMP'],
            bins=[-np.inf, 15, 30, np.inf],
            labels=[0, 1, 2]  # 0=cold, 1=optimal, 2=hot
        ).astype(float)

        # 3. Humidity categories
        merged_df['humidity_category'] = pd.cut(
            merged_df['HUMIDITY'],
            bins=[-np.inf, 40, 70, np.inf],
            labels=[0, 1, 2]  # 0=dry, 1=optimal, 2=humid
        ).astype(float)

        # 4. Grip index (0-100 scale)
        merged_df['grip_index'] = merged_df.apply(self.calculate_grip_index, axis=1)

        # 5. Wind impact (simplified - average track direction = 0)
        wind_components = merged_df.apply(
            lambda row: self.calculate_wind_components(
                row.get('WIND_SPEED', 0),
                row.get('WIND_DIRECTION', 0)
            ),
            axis=1,
            result_type='expand'
        )
        merged_df['headwind'] = wind_components['headwind']
        merged_df['crosswind'] = wind_components['crosswind']

        # 6. Rain indicator (binary)
        merged_df['is_raining'] = (merged_df['RAIN'] > 0).astype(int)

        # 7. Combined weather severity index (0-100, higher = worse conditions)
        merged_df['weather_severity'] = (
            abs(merged_df['air_temp_delta']) * 2 +  # Temperature impact
            abs(merged_df['track_temp_delta']) * 2 +  # Track temp impact
            merged_df['WIND_SPEED'] * 0.5 +  # Wind impact
            merged_df['is_raining'] * 50  # Rain impact (major)
        )

        # Report statistics
        weather_feature_cols = [
            'AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'WIND_SPEED', 'RAIN',
            'air_temp_delta', 'track_temp_delta', 'track_air_temp_diff',
            'temp_category', 'humidity_category', 'grip_index',
            'headwind', 'crosswind', 'is_raining', 'weather_severity'
        ]

        print(f"\nWeather features added: {len(weather_feature_cols)}")
        print(f"Laps with weather data: {merged_df['AIR_TEMP'].notna().sum()} / {len(merged_df)}")
        print(f"Coverage: {merged_df['AIR_TEMP'].notna().sum() / len(merged_df) * 100:.1f}%")

        print(f"\nWeather conditions summary:")
        print(f"  Air temp: {merged_df['AIR_TEMP'].mean():.1f}°C (range: {merged_df['AIR_TEMP'].min():.1f}-{merged_df['AIR_TEMP'].max():.1f}°C)")
        print(f"  Track temp: {merged_df['TRACK_TEMP'].mean():.1f}°C (range: {merged_df['TRACK_TEMP'].min():.1f}-{merged_df['TRACK_TEMP'].max():.1f}°C)")
        print(f"  Humidity: {merged_df['HUMIDITY'].mean():.1f}% (range: {merged_df['HUMIDITY'].min():.1f}-{merged_df['HUMIDITY'].max():.1f}%)")
        print(f"  Wind speed: {merged_df['WIND_SPEED'].mean():.1f} km/h")
        print(f"  Rainy laps: {merged_df['is_raining'].sum()} ({merged_df['is_raining'].sum() / len(merged_df) * 100:.1f}%)")
        print(f"  Grip index: {merged_df['grip_index'].mean():.1f}/100")

        return merged_df


class WeatherEnhancedLightGBMTrainer:
    """Train LightGBM model with weather features."""

    def __init__(self, data_dir='organized_data', output_dir='data'):
        """Initialize trainer with weather support."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.processed_dir = self.output_dir / 'processed'
        self.models_dir = self.output_dir / 'models'

        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.loader = RacingDataLoader(base_dir=str(self.data_dir))
        self.feature_engineer = TelemetryFeatureEngineer()
        self.weather_engineer = WeatherFeatureEngineer()
        self.weather_loader = WeatherDataLoader()
        self.model = None
        self.feature_names = None

    def load_baseline_features(self, track: str) -> Optional[pd.DataFrame]:
        """Load pre-computed baseline features (without weather)."""
        feature_file = self.processed_dir / f"{track}_features.parquet"
        if feature_file.exists():
            print(f"Loading baseline features from {feature_file}")
            return pd.read_parquet(feature_file)
        return None

    def load_lap_times_for_track(self, track: str) -> pd.DataFrame:
        """
        Load lap times with session info and calculate lap durations.

        Returns DataFrame with: vehicle_number, lap, lap_time, session
        """
        from pathlib import Path

        # Find the race_unknown directory (where actual lap time data lives)
        track_dir = self.data_dir / track / 'race_unknown' / 'lap_times'

        if not track_dir.exists():
            print(f"  ✗ Lap times directory not found: {track_dir}")
            return pd.DataFrame()

        # Find lap_start and lap_end files
        lap_start_files = list(track_dir.glob('*_lap_start.csv'))
        lap_end_files = list(track_dir.glob('*_lap_end.csv'))

        if not lap_start_files or not lap_end_files:
            print(f"  ✗ No lap start/end files found")
            return pd.DataFrame()

        # Load and combine start/end files
        start_dfs = [pd.read_csv(f) for f in lap_start_files]
        end_dfs = [pd.read_csv(f) for f in lap_end_files]

        lap_start_df = pd.concat(start_dfs, ignore_index=True)
        lap_end_df = pd.concat(end_dfs, ignore_index=True)

        # Merge to calculate lap times
        lap_times_df = lap_start_df[['vehicle_number', 'lap', 'timestamp', 'meta_session']].merge(
            lap_end_df[['vehicle_number', 'lap', 'timestamp']],
            on=['vehicle_number', 'lap'],
            suffixes=('_start', '_end'),
            how='inner'
        )

        # Convert timestamps to datetime
        lap_times_df['timestamp_start'] = pd.to_datetime(lap_times_df['timestamp_start'], errors='coerce')
        lap_times_df['timestamp_end'] = pd.to_datetime(lap_times_df['timestamp_end'], errors='coerce')

        # Calculate lap duration
        lap_times_df = lap_times_df.dropna(subset=['timestamp_start', 'timestamp_end'])
        lap_times_df['lap_time'] = (lap_times_df['timestamp_end'] - lap_times_df['timestamp_start']).dt.total_seconds()

        # Filter to realistic lap times (60-200 seconds)
        lap_times_df = lap_times_df[
            (lap_times_df['lap_time'] >= 60) &
            (lap_times_df['lap_time'] <= 200)
        ].copy()

        # Rename session column
        lap_times_df['session'] = lap_times_df['meta_session']

        # Select final columns
        lap_times_df = lap_times_df[['vehicle_number', 'lap', 'lap_time', 'session']]

        print(f"  ✓ Loaded {len(lap_times_df)} lap times")
        print(f"  Lap time range: {lap_times_df['lap_time'].min():.2f}s - {lap_times_df['lap_time'].max():.2f}s")
        print(f"  Sessions: {lap_times_df['session'].unique()}")

        return lap_times_df

    def add_weather_to_features(self, features_df: pd.DataFrame, track: str) -> pd.DataFrame:
        """Add weather features and lap times to existing telemetry features."""
        # Load lap times with session information
        print(f"Loading lap times for {track}...")
        lap_times_df = self.load_lap_times_for_track(track)

        if lap_times_df.empty:
            print("  ERROR: No lap times found - cannot add weather features!")
            return features_df

        # Merge features with lap times (adds lap_time and session columns)
        features_with_times = features_df.merge(
            lap_times_df,
            left_on=['vehicle_number', 'lap_number'],
            right_on=['vehicle_number', 'lap'],
            how='inner'  # Only keep laps with valid lap times
        )

        print(f"  ✓ Merged lap times: {len(features_with_times)} / {len(features_df)} laps")
        print(f"  Session distribution: {features_with_times['session'].value_counts().to_dict()}")

        # Load weather and add features
        weather_df = self.weather_loader.load_all_weather_data()
        return self.weather_engineer.add_weather_features(features_with_times, weather_df)

    def train_model(self, tracks: list, test_size=0.2, save_model=True):
        """
        Train weather-enhanced LightGBM model.

        Args:
            tracks: List of track names to include
            test_size: Fraction of data for testing
            save_model: Whether to save the trained model
        """
        print(f"\n{'='*60}")
        print("WEATHER-ENHANCED LIGHTGBM TRAINING")
        print(f"{'='*60}")
        print(f"Tracks: {tracks}")
        print(f"Test size: {test_size}")

        # Load baseline features and add weather for each track
        all_features = []
        for track in tracks:
            features = self.load_baseline_features(track)
            if features is not None:
                print(f"✓ {track}: {len(features)} laps")
                # Add weather features for this track
                features_with_weather = self.add_weather_to_features(features, track)
                all_features.append(features_with_weather)

        if not all_features:
            print("ERROR: No baseline features found. Run train_lightgbm.py first!")
            return

        # Combine all tracks
        features_df = pd.concat(all_features, ignore_index=True)
        print(f"\nTotal laps: {len(features_df)}")

        # Report weather data coverage
        weather_coverage = features_df['AIR_TEMP'].notna().sum()
        print(f"Laps with weather data: {weather_coverage} / {len(features_df)} ({weather_coverage/len(features_df)*100:.1f}%)")

        # Don't remove laps without weather - fill with track-average conditions instead
        # This allows the model to train on all data with degraded weather features for missing sessions
        if weather_coverage == 0:
            print("\n⚠️  WARNING: No weather data matched! Using default conditions.")
            # Use typical racing conditions as defaults
            features_df['AIR_TEMP'] = 20.0
            features_df['TRACK_TEMP'] = 25.0
            features_df['HUMIDITY'] = 50.0
            features_df['WIND_SPEED'] = 5.0
            features_df['WIND_DIRECTION'] = 0.0
            features_df['RAIN'] = 0.0

        # Prepare X and y
        exclude_cols = ['lap_time', 'vehicle_number', 'lap_number', 'lap', 'timestamp', 'session', 'session_code', 'track_name', 'track', 'race']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        X = features_df[feature_cols].fillna(0)
        y = features_df['lap_time']

        self.feature_names = feature_cols
        print(f"\nTotal features: {len(feature_cols)}")
        print(f"Weather features: {[col for col in feature_cols if any(w in col.lower() for w in ['temp', 'humid', 'wind', 'rain', 'grip', 'weather'])]}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        print(f"\nTraining set: {len(X_train)} laps")
        print(f"Test set: {len(X_test)} laps")

        # Train LightGBM
        print(f"\n{'='*60}")
        print("Training LightGBM Model...")
        print(f"{'='*60}")

        start_time = time.time()

        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        )

        train_time = time.time() - start_time
        print(f"\nTraining completed in {train_time:.1f}s")

        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        train_metrics = {
            'mae': mean_absolute_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'r2': r2_score(y_train, y_pred_train)
        }

        test_metrics = {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'r2': r2_score(y_test, y_pred_test)
        }

        print(f"\n{'='*60}")
        print("WEATHER-ENHANCED MODEL PERFORMANCE")
        print(f"{'='*60}")
        print(f"Train - MAE: {train_metrics['mae']:.3f}s | RMSE: {train_metrics['rmse']:.3f}s | R²: {train_metrics['r2']:.4f}")
        print(f"Test  - MAE: {test_metrics['mae']:.3f}s | RMSE: {test_metrics['rmse']:.3f}s | R²: {test_metrics['r2']:.4f}")

        # Save model
        if save_model:
            model_path = self.models_dir / 'lightgbm_weather_enhanced.pkl'
            joblib.dump(self.model, model_path)
            print(f"\n✓ Model saved: {model_path}")

            # Save feature names
            feature_names_path = self.models_dir / 'lightgbm_weather_enhanced_features.txt'
            with open(feature_names_path, 'w') as f:
                f.write('\n'.join(self.feature_names))
            print(f"✓ Feature names saved: {feature_names_path}")

        return test_metrics


def main():
    """Main training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Train weather-enhanced LightGBM model')
    parser.add_argument('--tracks', nargs='+', default=['barber-motorsports-park'],
                       help='Tracks to include (or "all")')
    parser.add_argument('--save-model', action='store_true', help='Save trained model')
    args = parser.parse_args()

    # Get track list
    if 'all' in args.tracks:
        tracks = [
            'barber-motorsports-park',
            'circuit-of-the-americas',
            'road-america',
            'sebring',
            'sonoma',
            'virginia-international-raceway'
        ]
    else:
        tracks = args.tracks

    # Train model
    trainer = WeatherEnhancedLightGBMTrainer()
    trainer.train_model(tracks, save_model=args.save_model)


if __name__ == '__main__':
    main()
