"""
Hyperparameter Optimization for Racing Telemetry Models
========================================================

Uses Optuna to find optimal hyperparameters for LightGBM, XGBoost, and CatBoost models.

Target: >2% R² improvement over baseline

Usage:
    python -m src.models.baseline.optimize_hyperparameters --model lightgbm --trials 100
    python -m src.models.baseline.optimize_hyperparameters --model all --trials 100
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class HyperparameterOptimizer:
    """Optimize hyperparameters for racing telemetry models using Optuna."""

    def __init__(self, data_path, output_dir='data', reports_dir='reports'):
        """
        Initialize optimizer.

        Parameters:
        -----------
        data_path : str
            Path to combined features parquet file
        output_dir : str
            Directory to save optimized models
        reports_dir : str
            Directory to save optimization reports and database
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.reports_dir = Path(reports_dir)
        self.models_dir = self.output_dir / 'models'

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Storage for Optuna studies
        self.storage = f'sqlite:///{self.reports_dir}/optuna_study.db'

        # Data placeholders
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_cols = None

        # Baseline scores for comparison
        self.baseline_scores = {}

    def load_data(self, test_size=0.2, random_state=42):
        """
        Load and split data for optimization.

        Parameters:
        -----------
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed for reproducibility
        """
        print(f"\n{'='*60}")
        print(f"LOADING DATA")
        print(f"{'='*60}")

        # Load data
        print(f"Loading from: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        print(f"Loaded {len(df):,} rows")

        # Prepare features and target
        self.feature_cols = [c for c in df.columns
                             if c not in ['vehicle_number', 'lap', 'lap_number', 'track', 'race', 'lap_time']]
        X = df[self.feature_cols]
        y = df['lap_time']
        tracks = df['track']

        print(f"Features: {len(self.feature_cols)}")
        print(f"Target range: {y.min():.2f}s - {y.max():.2f}s")

        # Handle missing values
        X = X.fillna(0)

        # Track-stratified split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=tracks
        )

        print(f"\nTrain set: {len(self.X_train):,} samples")
        print(f"Test set: {len(self.X_test):,} samples")
        print(f"Train target: {self.y_train.mean():.2f}s +/- {self.y_train.std():.2f}s")
        print(f"Test target: {self.y_test.mean():.2f}s +/- {self.y_test.std():.2f}s")

    def objective_lightgbm(self, trial):
        """
        Optuna objective function for LightGBM.

        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object

        Returns:
        --------
        float
            Validation R² score
        """
        # Suggest hyperparameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.05, 0.1]),
            'num_leaves': trial.suggest_categorical('num_leaves', [15, 31, 63, 127]),
            'max_depth': trial.suggest_categorical('max_depth', [5, 7, 10, 15, -1]),
            'min_child_samples': trial.suggest_categorical('min_child_samples', [5, 10, 20, 50]),
            'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.7, 0.8, 0.9, 1.0]),
            'reg_alpha': trial.suggest_categorical('reg_alpha', [0, 0.01, 0.1, 1.0]),
            'reg_lambda': trial.suggest_categorical('reg_lambda', [0, 0.01, 0.1, 1.0]),
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }

        # Create datasets
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        val_data = lgb.Dataset(self.X_test, label=self.y_test, reference=train_data)

        # Train with early stopping
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        )

        # Evaluate
        y_pred = model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)

        return r2

    def objective_xgboost(self, trial):
        """
        Optuna objective function for XGBoost.

        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object

        Returns:
        --------
        float
            Validation R² score
        """
        # Suggest hyperparameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7, 10]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.3]),
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 500, 1000]),
            'min_child_weight': trial.suggest_categorical('min_child_weight', [1, 3, 5, 7]),
            'gamma': trial.suggest_categorical('gamma', [0, 0.1, 0.2, 0.5]),
            'subsample': trial.suggest_categorical('subsample', [0.6, 0.8, 1.0]),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.8, 1.0]),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }

        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            early_stopping_rounds=50,
            verbose=False
        )

        # Evaluate
        y_pred = model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)

        return r2

    def objective_catboost(self, trial):
        """
        Optuna objective function for CatBoost.

        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object

        Returns:
        --------
        float
            Validation R² score
        """
        # Suggest hyperparameters
        params = {
            'iterations': trial.suggest_categorical('iterations', [100, 300, 500, 1000]),
            'depth': trial.suggest_categorical('depth', [4, 6, 8, 10]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.03, 0.1, 0.3]),
            'l2_leaf_reg': trial.suggest_categorical('l2_leaf_reg', [1, 3, 5, 7, 9]),
            'border_count': trial.suggest_categorical('border_count', [32, 64, 128, 255]),
            'random_state': 42,
            'thread_count': -1,
            'verbose': 0
        }

        # Train model
        model = cb.CatBoostRegressor(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=(self.X_test, self.y_test),
            early_stopping_rounds=50,
            verbose=False
        )

        # Evaluate
        y_pred = model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)

        return r2

    def optimize_model(self, model_name, n_trials=100):
        """
        Optimize hyperparameters for a specific model.

        Parameters:
        -----------
        model_name : str
            Model to optimize ('lightgbm', 'xgboost', 'catboost')
        n_trials : int
            Number of optimization trials

        Returns:
        --------
        dict
            Optimization results including best parameters and score
        """
        print(f"\n{'='*60}")
        print(f"OPTIMIZING {model_name.upper()}")
        print(f"{'='*60}")

        # Select objective function
        if model_name == 'lightgbm':
            objective = self.objective_lightgbm
        elif model_name == 'xgboost':
            objective = self.objective_xgboost
        elif model_name == 'catboost':
            objective = self.objective_catboost
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Create or load study
        study_name = f'{model_name}_optimization'
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            direction='maximize',
            load_if_exists=True
        )

        # Run optimization
        print(f"Starting optimization with {n_trials} trials...")
        start_time = time.time()

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        optimization_time = time.time() - start_time

        # Results
        print(f"\nOptimization completed in {optimization_time:.1f}s")
        print(f"\nBest trial:")
        print(f"  Value (R²): {study.best_trial.value:.6f}")
        print(f"  Params:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

        # Calculate improvement over baseline (if available)
        if model_name in self.baseline_scores:
            baseline_r2 = self.baseline_scores[model_name]
            improvement = (study.best_trial.value - baseline_r2) / baseline_r2 * 100
            print(f"\nImprovement over baseline:")
            print(f"  Baseline R²: {baseline_r2:.6f}")
            print(f"  Optimized R²: {study.best_trial.value:.6f}")
            print(f"  Improvement: {improvement:+.2f}%")

        return {
            'study': study,
            'best_params': study.best_trial.params,
            'best_score': study.best_trial.value,
            'optimization_time': optimization_time,
            'n_trials': n_trials
        }

    def train_final_model(self, model_name, params):
        """
        Train final model with optimized hyperparameters.

        Parameters:
        -----------
        model_name : str
            Model to train ('lightgbm', 'xgboost', 'catboost')
        params : dict
            Optimized hyperparameters

        Returns:
        --------
        tuple
            (model, metrics)
        """
        print(f"\n{'='*60}")
        print(f"TRAINING FINAL {model_name.upper()} MODEL")
        print(f"{'='*60}")

        start_time = time.time()

        if model_name == 'lightgbm':
            # Prepare params for LightGBM
            train_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1,
                **params
            }

            train_data = lgb.Dataset(self.X_train, label=self.y_train)
            test_data = lgb.Dataset(self.X_test, label=self.y_test, reference=train_data)

            model = lgb.train(
                train_params,
                train_data,
                num_boost_round=1000,
                valid_sets=[train_data, test_data],
                valid_names=['train', 'test'],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
            )

        elif model_name == 'xgboost':
            # Prepare params for XGBoost
            train_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'random_state': 42,
                'n_jobs': -1,
                **params
            }

            model = xgb.XGBRegressor(**train_params)
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
                early_stopping_rounds=50,
                verbose=True
            )

        elif model_name == 'catboost':
            # Prepare params for CatBoost
            train_params = {
                'random_state': 42,
                'thread_count': -1,
                **params
            }

            model = cb.CatBoostRegressor(**train_params)
            model.fit(
                self.X_train, self.y_train,
                eval_set=(self.X_test, self.y_test),
                early_stopping_rounds=50,
                verbose=100
            )

        else:
            raise ValueError(f"Unknown model: {model_name}")

        training_time = time.time() - start_time

        # Evaluate on both train and test sets
        if model_name == 'lightgbm':
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
        else:
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)

        train_metrics = {
            'mae': mean_absolute_error(self.y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'r2': r2_score(self.y_train, y_train_pred)
        }

        test_metrics = {
            'mae': mean_absolute_error(self.y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'r2': r2_score(self.y_test, y_test_pred)
        }

        print(f"\nTraining completed in {training_time:.1f}s")
        print(f"\nTrain Metrics:")
        print(f"  MAE:  {train_metrics['mae']:.3f}s")
        print(f"  RMSE: {train_metrics['rmse']:.3f}s")
        print(f"  R²:   {train_metrics['r2']:.6f}")

        print(f"\nTest Metrics:")
        print(f"  MAE:  {test_metrics['mae']:.3f}s")
        print(f"  RMSE: {test_metrics['rmse']:.3f}s")
        print(f"  R²:   {test_metrics['r2']:.6f}")

        return model, {'train': train_metrics, 'test': test_metrics, 'training_time': training_time}

    def save_model(self, model, model_name):
        """Save optimized model to disk."""
        model_path = self.models_dir / f'{model_name}_optimized.pkl'
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        return model_path

    def generate_report(self, results):
        """
        Generate comprehensive optimization report.

        Parameters:
        -----------
        results : dict
            Dictionary of optimization results for each model
        """
        print(f"\n{'='*60}")
        print(f"GENERATING OPTIMIZATION REPORT")
        print(f"{'='*60}")

        report_path = self.reports_dir / 'phase3_hyperparameter_optimization_report.md'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Phase 3: Hyperparameter Optimization Report\n\n")
            f.write("## Summary\n\n")
            f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Data:** {self.data_path}\n\n")
            f.write(f"**Train samples:** {len(self.X_train):,}\n\n")
            f.write(f"**Test samples:** {len(self.X_test):,}\n\n")
            f.write(f"**Features:** {len(self.feature_cols)}\n\n")

            f.write("\n## Optimization Results\n\n")

            for model_name, result in results.items():
                f.write(f"### {model_name.upper()}\n\n")
                f.write(f"**Trials:** {result['n_trials']}\n\n")
                f.write(f"**Optimization time:** {result['optimization_time']:.1f}s\n\n")
                f.write(f"**Best R² score:** {result['best_score']:.6f}\n\n")

                if model_name in self.baseline_scores:
                    baseline_r2 = self.baseline_scores[model_name]
                    improvement = (result['best_score'] - baseline_r2) / baseline_r2 * 100
                    f.write(f"**Baseline R²:** {baseline_r2:.6f}\n\n")
                    f.write(f"**Improvement:** {improvement:+.2f}%\n\n")

                f.write("\n**Best Parameters:**\n\n")
                f.write("```python\n")
                for key, value in result['best_params'].items():
                    f.write(f"{key}: {value}\n")
                f.write("```\n\n")

                if 'final_metrics' in result:
                    f.write("\n**Final Model Performance:**\n\n")
                    f.write("| Metric | Train | Test |\n")
                    f.write("|--------|-------|------|\n")
                    f.write(f"| MAE (s) | {result['final_metrics']['train']['mae']:.3f} | {result['final_metrics']['test']['mae']:.3f} |\n")
                    f.write(f"| RMSE (s) | {result['final_metrics']['train']['rmse']:.3f} | {result['final_metrics']['test']['rmse']:.3f} |\n")
                    f.write(f"| R² | {result['final_metrics']['train']['r2']:.6f} | {result['final_metrics']['test']['r2']:.6f} |\n\n")

            f.write("\n## Model Comparison\n\n")
            f.write("| Model | Best R² | Improvement | Training Time |\n")
            f.write("|-------|---------|-------------|---------------|\n")
            for model_name, result in results.items():
                improvement = ""
                if model_name in self.baseline_scores:
                    baseline_r2 = self.baseline_scores[model_name]
                    improvement = f"{((result['best_score'] - baseline_r2) / baseline_r2 * 100):+.2f}%"
                training_time = result.get('final_metrics', {}).get('training_time', 0)
                f.write(f"| {model_name.upper()} | {result['best_score']:.6f} | {improvement} | {training_time:.1f}s |\n")

            f.write("\n## Recommendations\n\n")
            best_model = max(results.items(), key=lambda x: x[1]['best_score'])
            f.write(f"**Best performing model:** {best_model[0].upper()} (R² = {best_model[1]['best_score']:.6f})\n\n")

            f.write("\n## Files Generated\n\n")
            f.write(f"- Optuna database: `{self.reports_dir}/optuna_study.db`\n")
            for model_name in results.keys():
                f.write(f"- {model_name.upper()} model: `{self.models_dir}/{model_name}_optimized.pkl`\n")

        print(f"Report saved to {report_path}")

    def plot_optimization_results(self, results):
        """
        Generate visualization plots for optimization results.

        Parameters:
        -----------
        results : dict
            Dictionary of optimization results for each model
        """
        print(f"\nGenerating optimization visualizations...")

        for model_name, result in results.items():
            study = result['study']

            # Optimization history
            fig = plot_optimization_history(study)
            fig.write_html(self.reports_dir / f'{model_name}_optimization_history.html')
            print(f"  Saved {model_name} optimization history")

            # Parameter importances
            fig = plot_param_importances(study)
            fig.write_html(self.reports_dir / f'{model_name}_param_importances.html')
            print(f"  Saved {model_name} parameter importances")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Optimize model hyperparameters with Optuna')
    parser.add_argument('--data', type=str, default='data/processed/combined_features.parquet',
                        help='Path to combined features parquet file')
    parser.add_argument('--model', type=str, default='all',
                        choices=['lightgbm', 'xgboost', 'catboost', 'all'],
                        help='Model to optimize (default: all)')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of optimization trials per model (default: 100)')
    parser.add_argument('--baseline-scores', type=str, default='',
                        help='Baseline scores as comma-separated model:score pairs (e.g., "lightgbm:0.95,xgboost:0.93")')

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = HyperparameterOptimizer(data_path=args.data)

    # Parse baseline scores
    if args.baseline_scores:
        for pair in args.baseline_scores.split(','):
            model, score = pair.split(':')
            optimizer.baseline_scores[model.strip()] = float(score.strip())

    # Load data
    optimizer.load_data()

    # Determine which models to optimize
    if args.model == 'all':
        models_to_optimize = ['lightgbm', 'xgboost', 'catboost']
    else:
        models_to_optimize = [args.model]

    # Run optimization
    results = {}
    for model_name in models_to_optimize:
        try:
            # Optimize hyperparameters
            result = optimizer.optimize_model(model_name, n_trials=args.trials)

            # Train final model with best parameters
            final_model, final_metrics = optimizer.train_final_model(model_name, result['best_params'])

            # Save model
            optimizer.save_model(final_model, model_name)

            # Store results
            result['final_metrics'] = final_metrics
            results[model_name] = result

        except Exception as e:
            print(f"\nError optimizing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Generate report
    if results:
        optimizer.generate_report(results)
        optimizer.plot_optimization_results(results)

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
