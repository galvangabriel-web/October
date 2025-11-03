"""
Baseline LightGBM Model Training Pipeline
==========================================

Trains a LightGBM model for lap time prediction using engineered features.

Target Performance:
- R² > 0.90 on test set
- MAE < 0.6 seconds
- Training time < 5 minutes

Usage:
    python -m src.models.baseline.train_lightgbm --tracks all --save-model
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

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from data_loader import RacingDataLoader
from src.data_processing.feature_engineering import TelemetryFeatureEngineer


class LightGBMTrainer:
    """Train and evaluate LightGBM model for lap time prediction."""

    def __init__(self, data_dir='organized_data', output_dir='data'):
        """
        Initialize trainer.

        Parameters:
        -----------
        data_dir : str
            Directory containing organized telemetry data
        output_dir : str
            Directory to save processed features and models
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.processed_dir = self.output_dir / 'processed'
        self.models_dir = self.output_dir / 'models'

        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.loader = RacingDataLoader(base_dir=str(self.data_dir))
        self.feature_engineer = TelemetryFeatureEngineer()
        self.model = None

    def extract_features_for_track(self, track, sample_size=None, max_chunks=10):
        """
        Extract features for a specific track (all races automatically detected).

        Parameters:
        -----------
        track : str
            Track name (e.g., 'barber-motorsports-park')
        sample_size : int, optional
            Number of laps to sample (for testing)
        max_chunks : int, optional
            Maximum number of telemetry chunks to process (default: 10)

        Returns:
        --------
        pd.DataFrame
            Feature dataframe with lap-level features
        """
        print(f"\n{'='*60}")
        print(f"Processing: {track}")
        print(f"{'='*60}")

        # Check if features already exist
        feature_file = self.processed_dir / f"{track}_features.parquet"
        if feature_file.exists():
            print(f"✓ Loading existing features from {feature_file}")
            return pd.read_parquet(feature_file)

        # Detect available races dynamically
        track_dir = self.data_dir / track
        if not track_dir.exists():
            print(f"✗ Track directory not found: {track_dir}")
            return None

        # Find race directories
        available_races = [d.name for d in track_dir.iterdir() if d.is_dir() and (d / 'telemetry').exists()]

        if not available_races:
            print(f"✗ No race directories with telemetry found in {track}")
            return None

        print(f"Found {len(available_races)} race(s): {available_races}")

        # Process all races
        all_features = []
        for race in available_races:
            print(f"\n--- Processing {track} / {race} ---")
            race_features = self._extract_features_for_race(track, race, sample_size, max_chunks)
            if race_features is not None and len(race_features) > 0:
                race_features['race'] = race  # Track which race this data is from
                all_features.append(race_features)

        if not all_features:
            print(f"✗ No features extracted for {track}")
            return None

        # Combine all races
        combined_features = pd.concat(all_features, ignore_index=True)
        print(f"\n✓ Total features extracted: {len(combined_features)} laps from {len(available_races)} race(s)")

        # Save combined features
        combined_features.to_parquet(feature_file, index=False)
        print(f"✓ Saved features to {feature_file}")

        return combined_features

    def _extract_features_for_race(self, track, race, sample_size=None, max_chunks=10):
        """
        Extract features for a specific race.

        Parameters:
        -----------
        track : str
            Track name
        race : str
            Race identifier
        sample_size : int, optional
            Number of laps to sample
        max_chunks : int, optional
            Maximum chunks to process

        Returns:
        --------
        pd.DataFrame
            Feature dataframe for this race
        """
        try:
            # Load telemetry data - load multiple chunks
            print(f"Loading telemetry data (up to {max_chunks} chunks)...")
            start_time = time.time()

            # Load multiple chunks
            telemetry_chunks = []
            for chunk_num in range(1, max_chunks + 1):
                chunk_df = self.loader.load_single_chunk(track, race, 'telemetry', chunk_num=chunk_num)
                if chunk_df is not None and len(chunk_df) > 0:
                    telemetry_chunks.append(chunk_df)
                else:
                    break  # No more chunks available

            if not telemetry_chunks:
                print(f"✗ No telemetry data found for {track}")
                return None

            num_chunks = len(telemetry_chunks)
            telemetry_df = pd.concat(telemetry_chunks, ignore_index=True)
            del telemetry_chunks  # Free memory

            load_time = time.time() - start_time
            print(f"✓ Loaded {len(telemetry_df):,} telemetry rows from {num_chunks} chunks in {load_time:.1f}s")

            # Get unique vehicle-lap combinations
            vehicle_laps = telemetry_df[['vehicle_number', 'lap']].drop_duplicates()
            print(f"✓ Found {len(vehicle_laps)} unique laps across {telemetry_df['vehicle_number'].nunique()} vehicles")

            # Sample if requested
            if sample_size and len(vehicle_laps) > sample_size:
                vehicle_laps = vehicle_laps.sample(n=sample_size, random_state=42)
                print(f"✓ Sampling {sample_size} laps for testing")

            # Extract features for each lap
            print(f"Extracting features...")
            features_list = []
            extraction_start = time.time()

            for idx, (_, row) in enumerate(vehicle_laps.iterrows(), 1):
                vehicle = row['vehicle_number']
                lap = row['lap']

                features = self.feature_engineer.extract_lap_features(
                    telemetry_df, vehicle, lap
                )

                if features:
                    features_list.append(features)

                if idx % 50 == 0:
                    elapsed = time.time() - extraction_start
                    rate = idx / elapsed
                    remaining = (len(vehicle_laps) - idx) / rate
                    print(f"  Progress: {idx}/{len(vehicle_laps)} laps ({rate:.1f} laps/s, {remaining:.0f}s remaining)")

            if not features_list:
                print(f"✗ No features extracted for {track}")
                return None

            # Create features DataFrame
            features_df = pd.DataFrame(features_list)
            extraction_time = time.time() - extraction_start
            print(f"✓ Extracted {len(features_df)} feature rows in {extraction_time:.1f}s")
            print(f"  Rate: {len(features_df)/extraction_time:.1f} laps/second")

            # Free memory before returning
            del telemetry_df
            return features_df

        except Exception as e:
            print(f"✗ Error processing {track}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_lap_times_for_track(self, track):
        """
        Load lap times for a track (all races automatically detected).

        Parameters:
        -----------
        track : str
            Track name

        Returns:
        --------
        pd.DataFrame
            Lap times with vehicle_number, lap, and race as keys
        """
        # Detect available races
        track_dir = self.data_dir / track
        available_races = [d.name for d in track_dir.iterdir() if d.is_dir() and (d / 'lap_times').exists()]

        if not available_races:
            print(f"✗ No race directories with lap_times found for {track}")
            return None

        # Load lap times from all races
        all_lap_times = []
        for race in available_races:
            race_lap_times = self._load_lap_times_for_race(track, race)
            if race_lap_times is not None and len(race_lap_times) > 0:
                race_lap_times['race'] = race  # Track which race this data is from
                all_lap_times.append(race_lap_times)

        if not all_lap_times:
            print(f"✗ No lap times loaded for {track}")
            return None

        # Combine all races
        combined_lap_times = pd.concat(all_lap_times, ignore_index=True)
        print(f"✓ Loaded {len(combined_lap_times)} total lap times from {len(available_races)} race(s)")

        return combined_lap_times

    def _load_lap_times_for_race(self, track, race):
        """
        Load lap times for a specific race.

        Parameters:
        -----------
        track : str
            Track name
        race : str
            Race identifier

        Returns:
        --------
        pd.DataFrame
            Lap times for this race
        """
        try:
            # Load lap start and end files separately
            import glob
            lap_times_dir = self.data_dir / track / race / 'lap_times'

            # Find lap_start and lap_end files
            lap_start_files = list(lap_times_dir.glob('*_lap_start.csv'))
            lap_end_files = list(lap_times_dir.glob('*_lap_end.csv'))

            if not lap_start_files or not lap_end_files:
                return None

            # Load and combine all lap start files
            start_dfs = []
            for file in lap_start_files:
                df = pd.read_csv(file, low_memory=False)
                df['source_file'] = file.stem
                start_dfs.append(df)
            lap_start_df = pd.concat(start_dfs, ignore_index=True)

            # Load and combine all lap end files
            end_dfs = []
            for file in lap_end_files:
                df = pd.read_csv(file, low_memory=False)
                df['source_file'] = file.stem
                end_dfs.append(df)
            lap_end_df = pd.concat(end_dfs, ignore_index=True)

            # Merge on vehicle_number and lap to get start and end timestamps
            lap_times_df = lap_start_df[['vehicle_number', 'lap', 'timestamp']].merge(
                lap_end_df[['vehicle_number', 'lap', 'timestamp']],
                on=['vehicle_number', 'lap'],
                suffixes=('_start', '_end'),
                how='inner'
            )

            # Convert ISO 8601 timestamps to datetime objects
            lap_times_df['timestamp_start'] = pd.to_datetime(lap_times_df['timestamp_start'], errors='coerce')
            lap_times_df['timestamp_end'] = pd.to_datetime(lap_times_df['timestamp_end'], errors='coerce')

            # Drop rows with NaT timestamps
            lap_times_df = lap_times_df.dropna(subset=['timestamp_start', 'timestamp_end'])

            # Calculate lap duration in seconds (timedelta to seconds)
            lap_times_df['lap_time'] = (lap_times_df['timestamp_end'] - lap_times_df['timestamp_start']).dt.total_seconds()

            # Keep only valid lap times (realistic racing lap times)
            # Filter out outlaps, inlaps, incomplete laps, and data errors
            lap_times_df = lap_times_df[
                (lap_times_df['lap_time'] >= 60) &   # Min reasonable lap time
                (lap_times_df['lap_time'] <= 200)    # Max reasonable lap time for these tracks
            ].copy()

            # Select final columns
            lap_times_df = lap_times_df[['vehicle_number', 'lap', 'lap_time']]

            if len(lap_times_df) == 0:
                print(f"✗ No valid lap times computed for {track}")
                return None

            print(f"✓ Computed {len(lap_times_df)} lap times for {track}")
            print(f"  Lap time range: {lap_times_df['lap_time'].min():.2f}s - {lap_times_df['lap_time'].max():.2f}s")
            return lap_times_df

        except Exception as e:
            print(f"✗ Error loading lap times for {track}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def prepare_training_data(self, tracks=None, sample_size=None, max_chunks=10):
        """
        Prepare training dataset by extracting features and merging with lap times.

        Parameters:
        -----------
        tracks : list, optional
            List of track names to process. If None, processes all tracks.
        sample_size : int, optional
            Number of laps per track to sample (for testing)
        max_chunks : int, optional
            Maximum number of telemetry chunks to process per track (default: 10)

        Returns:
        --------
        pd.DataFrame
            Combined features and lap times ready for training
        """
        if tracks is None:
            tracks = self.loader.list_tracks()
            # Filter out README
            tracks = [t for t in tracks if t != 'README.md']

        print(f"\n{'='*60}")
        print(f"PREPARING TRAINING DATA")
        print(f"{'='*60}")
        print(f"Tracks to process: {tracks}")
        print(f"Sample size: {sample_size or 'All laps'}")
        print(f"Max chunks per track: {max_chunks}")

        all_data = []

        for track in tracks:
            print(f"\n--- Processing {track} ---")

            # Extract features
            features_df = self.extract_features_for_track(track, sample_size=sample_size, max_chunks=max_chunks)
            if features_df is None:
                continue

            # Load lap times
            lap_times_df = self.load_lap_times_for_track(track)
            if lap_times_df is None:
                continue

            # Rename 'lap' to 'lap_number' to match features_df
            lap_times_df = lap_times_df.rename(columns={'lap': 'lap_number'})

            # Merge features with lap times (on vehicle, lap, and race)
            merged = features_df.merge(
                lap_times_df,
                on=['vehicle_number', 'lap_number', 'race'],
                how='inner'
            )

            print(f"✓ Merged {len(merged)} laps with lap times")

            if len(merged) == 0:
                print(f"✗ No matching laps found for {track}")
                continue

            # Add track identifier for stratification
            merged['track'] = track
            all_data.append(merged)

        if not all_data:
            raise ValueError("No data could be processed from any track")

        # Combine all tracks
        combined_df = pd.concat(all_data, ignore_index=True)

        print(f"\n{'='*60}")
        print(f"DATA PREPARATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total laps: {len(combined_df):,}")
        print(f"Total tracks: {combined_df['track'].nunique()}")
        print(f"Total vehicles: {combined_df['vehicle_number'].nunique()}")
        print(f"Feature count: {len([c for c in combined_df.columns if c not in ['vehicle_number', 'lap', 'track', 'lap_time']])}")

        # Save combined dataset
        combined_file = self.processed_dir / 'combined_features.parquet'
        combined_df.to_parquet(combined_file, index=False)
        print(f"✓ Saved combined dataset to {combined_file}")

        return combined_df

    def train(self, df, test_size=0.2, random_state=42):
        """
        Train LightGBM model with track-stratified split.

        Parameters:
        -----------
        df : pd.DataFrame
            Training data with features and lap_time target
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed for reproducibility

        Returns:
        --------
        dict
            Training results including metrics and model
        """
        print(f"\n{'='*60}")
        print(f"TRAINING LIGHTGBM MODEL")
        print(f"{'='*60}")

        # Prepare features and target
        feature_cols = [c for c in df.columns if c not in ['vehicle_number', 'lap', 'track', 'race', 'lap_time']]
        X = df[feature_cols]
        y = df['lap_time']
        tracks = df['track']

        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X):,}")
        print(f"Target range: {y.min():.2f}s - {y.max():.2f}s")

        # Handle missing values
        X = X.fillna(0)

        # Track-stratified split (or use all data if test_size=0)
        if test_size > 0:
            X_train, X_test, y_train, y_test, tracks_train, tracks_test = train_test_split(
                X, y, tracks, test_size=test_size, random_state=random_state, stratify=tracks
            )
        else:
            # Train on all data
            X_train, y_train, tracks_train = X, y, tracks
            X_test, y_test, tracks_test = X.iloc[:0], y.iloc[:0], tracks.iloc[:0]  # Empty test set

        print(f"\nTrain set: {len(X_train):,} laps")
        if test_size > 0:
            print(f"Test set: {len(X_test):,} laps")
            print(f"Train tracks: {tracks_train.unique()}")
            print(f"Test tracks: {tracks_test.unique()}")
        else:
            print(f"Test set: None (training on all data)")

        # Train LightGBM
        print(f"\nTraining LightGBM...")
        start_time = time.time()

        # Define parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state
        }

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)

        # Train
        if test_size > 0:
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[train_data, test_data],
                valid_names=['train', 'test'],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
            )
        else:
            # Train without validation set
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=500,  # Fixed number since no early stopping
                valid_sets=[train_data],
                valid_names=['train'],
                callbacks=[lgb.log_evaluation(period=100)]
            )

        train_time = time.time() - start_time
        print(f"✓ Training completed in {train_time:.1f}s")

        # Predictions
        y_train_pred = self.model.predict(X_train)

        # Evaluate
        train_metrics = {
            'mae': mean_absolute_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'r2': r2_score(y_train, y_train_pred)
        }

        print(f"\n{'='*60}")
        print(f"MODEL PERFORMANCE")
        print(f"{'='*60}")
        print(f"\nTrain Metrics:")
        print(f"  MAE:  {train_metrics['mae']:.3f}s")
        print(f"  RMSE: {train_metrics['rmse']:.3f}s")
        print(f"  R²:   {train_metrics['r2']:.4f}")

        if test_size > 0:
            y_test_pred = self.model.predict(X_test)
            test_metrics = {
                'mae': mean_absolute_error(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'r2': r2_score(y_test, y_test_pred)
            }

            print(f"\nTest Metrics:")
            print(f"  MAE:  {test_metrics['mae']:.3f}s")
            print(f"  RMSE: {test_metrics['rmse']:.3f}s")
            print(f"  R²:   {test_metrics['r2']:.4f}")

            # Check success criteria
            success = test_metrics['r2'] > 0.90 and test_metrics['mae'] < 0.6
            status = "✓ SUCCESS" if success else "✗ NEEDS IMPROVEMENT"
            print(f"\nTarget Achievement: {status}")
            print(f"  R² > 0.90: {'✓' if test_metrics['r2'] > 0.90 else '✗'} ({test_metrics['r2']:.4f})")
            print(f"  MAE < 0.6s: {'✓' if test_metrics['mae'] < 0.6 else '✗'} ({test_metrics['mae']:.3f}s)")
        else:
            y_test_pred = None
            test_metrics = None
            print(f"\nNo test set (trained on all data)")

        return {
            'model': self.model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_cols': feature_cols,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'training_time': train_time
        }

    def save_model(self, filename='lightgbm_baseline.pkl'):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        model_path = self.models_dir / filename
        joblib.dump(self.model, model_path)
        print(f"✓ Model saved to {model_path}")
        return model_path

    def cross_validate(self, df, n_folds=5, random_state=42):
        """
        Perform k-fold cross-validation with track-aware splits.

        Parameters:
        -----------
        df : pd.DataFrame
            Training data with features and lap_time target
        n_folds : int
            Number of cross-validation folds
        random_state : int
            Random seed for reproducibility

        Returns:
        --------
        dict
            Cross-validation results including per-fold metrics
        """
        from sklearn.model_selection import StratifiedKFold

        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION ({n_folds} FOLDS)")
        print(f"{'='*60}")

        # Prepare features and target
        feature_cols = [c for c in df.columns if c not in ['vehicle_number', 'lap_number', 'track', 'lap_time']]
        X = df[feature_cols].fillna(0)
        y = df['lap_time']
        tracks = df['track']

        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X):,}")
        print(f"Tracks: {tracks.nunique()}")

        # Use stratified k-fold based on tracks
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        track_labels = le.fit_transform(tracks)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, track_labels), 1):
            print(f"\nFold {fold_idx}/{n_folds}:")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(f"  Train: {len(X_train)} laps | Test: {len(X_test)} laps")

            # Train model
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': random_state
            }

            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[test_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
            )

            # Predictions
            y_pred = model.predict(X_test)

            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"  MAE: {mae:.3f}s | RMSE: {rmse:.3f}s | R²: {r2:.4f}")

            fold_results.append({
                'fold': fold_idx,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'n_train': len(X_train),
                'n_test': len(X_test)
            })

        # Aggregate results
        avg_mae = np.mean([r['mae'] for r in fold_results])
        avg_rmse = np.mean([r['rmse'] for r in fold_results])
        avg_r2 = np.mean([r['r2'] for r in fold_results])

        std_mae = np.std([r['mae'] for r in fold_results])
        std_rmse = np.std([r['rmse'] for r in fold_results])
        std_r2 = np.std([r['r2'] for r in fold_results])

        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Average MAE:  {avg_mae:.3f} ± {std_mae:.3f}s")
        print(f"Average RMSE: {avg_rmse:.3f} ± {std_rmse:.3f}s")
        print(f"Average R²:   {avg_r2:.4f} ± {std_r2:.4f}")

        # Check success criteria
        success = avg_r2 > 0.90 and avg_mae < 0.6
        status = "✓ SUCCESS" if success else "✗ NEEDS IMPROVEMENT"
        print(f"\nTarget Achievement: {status}")
        print(f"  R² > 0.90: {'✓' if avg_r2 > 0.90 else '✗'} ({avg_r2:.4f})")
        print(f"  MAE < 0.6s: {'✓' if avg_mae < 0.6 else '✗'} ({avg_mae:.3f}s)")

        return {
            'fold_results': fold_results,
            'avg_mae': avg_mae,
            'avg_rmse': avg_rmse,
            'avg_r2': avg_r2,
            'std_mae': std_mae,
            'std_rmse': std_rmse,
            'std_r2': std_r2
        }

    def plot_feature_importance(self, results, top_n=20, save=True):
        """
        Plot feature importance from trained model.

        Parameters:
        -----------
        results : dict
            Training results from train()
        top_n : int
            Number of top features to display
        save : bool
            Whether to save plot to disk
        """
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = results['feature_cols']

        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Plot top N features
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance (Gain)')
        plt.title(f'Top {top_n} Feature Importance - LightGBM')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save:
            plot_path = self.models_dir / 'feature_importance.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance plot saved to {plot_path}")

        plt.show()

        return importance_df

    def plot_predictions(self, results, save=True):
        """
        Plot predicted vs actual lap times.

        Parameters:
        -----------
        results : dict
            Training results from train()
        save : bool
            Whether to save plot to disk
        """
        # Guard for empty test set (e.g., when test_size=0)
        if len(results.get('y_test', [])) == 0:
            print("⚠ Skipping predictions plot (no test data available)")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Train set
        axes[0].scatter(results['y_train'], results['y_train_pred'], alpha=0.5, s=10)
        axes[0].plot([results['y_train'].min(), results['y_train'].max()],
                     [results['y_train'].min(), results['y_train'].max()],
                     'r--', lw=2)
        axes[0].set_xlabel('Actual Lap Time (s)')
        axes[0].set_ylabel('Predicted Lap Time (s)')
        axes[0].set_title(f"Train Set (R²={results['train_metrics']['r2']:.4f})")
        axes[0].grid(True, alpha=0.3)

        # Test set
        axes[1].scatter(results['y_test'], results['y_test_pred'], alpha=0.5, s=10)
        axes[1].plot([results['y_test'].min(), results['y_test'].max()],
                     [results['y_test'].min(), results['y_test'].max()],
                     'r--', lw=2)
        axes[1].set_xlabel('Actual Lap Time (s)')
        axes[1].set_ylabel('Predicted Lap Time (s)')
        axes[1].set_title(f"Test Set (R²={results['test_metrics']['r2']:.4f})")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plot_path = self.models_dir / 'predictions_vs_actual.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Predictions plot saved to {plot_path}")

        plt.show()


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train baseline LightGBM model')
    parser.add_argument('--tracks', nargs='+', default=None,
                        help='Tracks to process (default: all)')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Number of laps per track to sample (default: all)')
    parser.add_argument('--max-chunks', type=int, default=10,
                        help='Maximum telemetry chunks to process per track (default: 10)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion (default: 0.2)')
    parser.add_argument('--cv', action='store_true',
                        help='Run cross-validation instead of single train/test split')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained model to disk')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate visualizations')

    args = parser.parse_args()

    # Initialize trainer
    trainer = LightGBMTrainer()

    # Prepare data
    print(f"\n{'#'*60}")
    print(f"# LIGHTGBM BASELINE TRAINING PIPELINE")
    print(f"{'#'*60}\n")

    df = trainer.prepare_training_data(tracks=args.tracks, sample_size=args.sample_size, max_chunks=args.max_chunks)

    # Cross-validation or single train/test
    if args.cv:
        cv_results = trainer.cross_validate(df, n_folds=args.cv_folds)
        # Train final model on all data for saving
        if args.save_model:
            print(f"\nTraining final model on all data...")
            results = trainer.train(df, test_size=0.0)  # Train on all data
    else:
        # Train model with train/test split
        results = trainer.train(df, test_size=args.test_size)

    # Save model
    if args.save_model:
        trainer.save_model()

    # Generate plots
    if args.plot:
        print(f"\nGenerating visualizations...")
        trainer.plot_feature_importance(results)

        # Only plot predictions if test data exists
        if len(results.get('y_test', [])) > 0:
            trainer.plot_predictions(results)
        else:
            print("⚠ Skipping predictions plot (trained on full dataset with test_size=0)")

    print(f"\n{'='*60}")
    print(f"TRAINING PIPELINE COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
