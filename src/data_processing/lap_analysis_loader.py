"""
Lap Analysis Data Loader for Alkamel Racing Data

This module loads and processes lap analysis data with sector times (S1, S2, S3)
from Alkamel timing system CSV files.

Usage:
    from src.data_processing.lap_analysis_loader import LapAnalysisLoader

    loader = LapAnalysisLoader()
    laps_df = loader.load_all_lap_data()

    # Get sector statistics
    sector_stats = loader.get_sector_statistics(laps_df)
"""

import pandas as pd
import glob
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class LapAnalysisLoader:
    """
    Loads and processes lap analysis data with sector times.

    Lap analysis files contain:
    - NUMBER: Car number
    - DRIVER_NAME: Driver name
    - LAP_NUMBER: Lap number in session
    - LAP_TIME: Total lap time (formatted string)
    - S1, S2, S3: Sector times (formatted strings)
    - S1_SECONDS, S2_SECONDS, S3_SECONDS: Sector times in seconds (numeric)
    - CLASS: Racing class (SRO3, GT4, etc.)
    - TEAM: Team name
    - TOP_SPEED: Top speed in session
    - Intermediate times (IM1a, IM1, IM2a, IM2, IM3a)
    """

    def __init__(self, data_dir: str = "new_data/alkamel_downloads"):
        """
        Initialize the lap analysis loader.

        Args:
            data_dir: Directory containing Alkamel CSV files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        logger.info(f"Initialized LapAnalysisLoader with data_dir: {self.data_dir}")

    def find_lap_analysis_files(self) -> List[Path]:
        """
        Find all lap analysis CSV files in the data directory.

        Returns:
            List of Path objects for lap analysis files
        """
        # Pattern matches files like: "23_AnalysisEnduranceWithSections_Race 1.CSV"
        analysis_files = list(self.data_dir.glob("*AnalysisEndurance*.CSV"))
        analysis_files.extend(list(self.data_dir.glob("*AnalysisEndurance*.csv")))

        logger.info(f"Found {len(analysis_files)} lap analysis files")
        return analysis_files

    def parse_lap_time(self, time_str: str) -> Optional[float]:
        """
        Parse lap time string to seconds.

        Args:
            time_str: Time string (e.g., "1:23.456" or "5:23.915")

        Returns:
            Time in seconds, or None if parsing fails
        """
        if pd.isna(time_str):
            return None

        try:
            time_str = str(time_str).strip()

            # Handle format: "M:SS.mmm" or "MM:SS.mmm"
            if ':' in time_str:
                parts = time_str.split(':')
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                # Already in seconds
                return float(time_str)
        except:
            return None

    def load_lap_analysis_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single lap analysis CSV file.

        Args:
            file_path: Path to lap analysis CSV file

        Returns:
            DataFrame with lap analysis data
        """
        try:
            # Alkamel CSVs use semicolon delimiter
            df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')

            # Clean column names (remove leading/trailing spaces)
            df.columns = df.columns.str.strip()

            # Extract session name from filename
            filename = file_path.name
            session = filename.split('_AnalysisEndurance')[-1].replace('WithSections_', '').replace('.CSV', '').replace('.csv', '')
            df['session'] = session
            df['filename'] = filename

            # Ensure numeric columns are properly typed
            numeric_cols = ['NUMBER', 'DRIVER_NUMBER', 'LAP_NUMBER',
                          'S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS',
                          'KPH', 'TOP_SPEED']

            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Calculate total lap time in seconds if not already present
            if 'LAP_TIME_SECONDS' not in df.columns:
                df['LAP_TIME_SECONDS'] = df['S1_SECONDS'] + df['S2_SECONDS'] + df['S3_SECONDS']

            # Remove outlier laps (likely pit laps or issues)
            # Filter out laps where any sector is abnormally long (>300 seconds = 5 minutes)
            if 'S1_SECONDS' in df.columns:
                df = df[
                    (df['S1_SECONDS'] < 300) &
                    (df['S2_SECONDS'] < 300) &
                    (df['S3_SECONDS'] < 300)
                ]

            logger.debug(f"Loaded {len(df)} valid laps from {filename}")
            return df

        except Exception as e:
            logger.error(f"Error loading lap analysis file {file_path}: {e}")
            raise

    def load_all_lap_data(self) -> pd.DataFrame:
        """
        Load all lap analysis files and combine into a single DataFrame.

        Returns:
            Combined DataFrame with all lap data
        """
        analysis_files = self.find_lap_analysis_files()

        if not analysis_files:
            logger.warning("No lap analysis files found!")
            return pd.DataFrame()

        all_laps = []
        for file_path in analysis_files:
            try:
                df = self.load_lap_analysis_file(file_path)
                all_laps.append(df)
            except Exception as e:
                logger.warning(f"Skipping file {file_path.name} due to error: {e}")
                continue

        if not all_laps:
            logger.warning("No lap data could be loaded!")
            return pd.DataFrame()

        # Combine all dataframes
        combined = pd.concat(all_laps, ignore_index=True)

        logger.info(f"Loaded {len(combined)} total laps from {len(all_laps)} files")
        return combined

    def get_sector_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate sector time statistics.

        Args:
            df: DataFrame with lap data

        Returns:
            Dictionary with sector statistics
        """
        if df.empty:
            return {}

        stats = {}

        for sector in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
            if sector not in df.columns:
                continue

            sector_name = sector.replace('_SECONDS', '')
            valid_times = df[sector].dropna()

            if len(valid_times) == 0:
                continue

            stats[sector_name] = {
                'min': valid_times.min(),
                'max': valid_times.max(),
                'mean': valid_times.mean(),
                'median': valid_times.median(),
                'std': valid_times.std(),
                'p10': valid_times.quantile(0.10),
                'p25': valid_times.quantile(0.25),
                'p50': valid_times.quantile(0.50),
                'p75': valid_times.quantile(0.75),
                'p90': valid_times.quantile(0.90),
                'count': len(valid_times)
            }

        return stats

    def get_driver_sector_performance(self, df: pd.DataFrame, driver_name: str) -> Dict[str, float]:
        """
        Get sector performance for a specific driver.

        Args:
            df: DataFrame with lap data
            driver_name: Name of the driver

        Returns:
            Dictionary with driver's sector times
        """
        driver_laps = df[df['DRIVER_NAME'] == driver_name]

        if driver_laps.empty:
            return {}

        performance = {
            'S1_best': driver_laps['S1_SECONDS'].min(),
            'S2_best': driver_laps['S2_SECONDS'].min(),
            'S3_best': driver_laps['S3_SECONDS'].min(),
            'S1_avg': driver_laps['S1_SECONDS'].mean(),
            'S2_avg': driver_laps['S2_SECONDS'].mean(),
            'S3_avg': driver_laps['S3_SECONDS'].mean(),
            'total_laps': len(driver_laps),
            'best_lap': driver_laps['LAP_TIME_SECONDS'].min() if 'LAP_TIME_SECONDS' in driver_laps.columns else None
        }

        return performance

    def calculate_percentile_rank(self, df: pd.DataFrame, driver_name: str,
                                 sector: str = 'S1_SECONDS') -> Optional[float]:
        """
        Calculate driver's percentile rank for a sector.

        Args:
            df: DataFrame with lap data
            driver_name: Name of the driver
            sector: Sector column name (S1_SECONDS, S2_SECONDS, or S3_SECONDS)

        Returns:
            Percentile rank (0-100), where 100 is the best
        """
        if sector not in df.columns:
            return None

        # Get driver's best time in this sector
        driver_laps = df[df['DRIVER_NAME'] == driver_name]
        if driver_laps.empty:
            return None

        driver_best = driver_laps[sector].min()

        # Get all sector times
        all_times = df[sector].dropna()

        if len(all_times) == 0:
            return None

        # Calculate percentile (lower time = better = higher percentile)
        # Count how many times are worse (slower) than driver's best
        worse_count = (all_times > driver_best).sum()
        percentile = (worse_count / len(all_times)) * 100

        return percentile

    def get_best_theoretical_lap(self, df: pd.DataFrame, driver_name: str) -> Dict[str, float]:
        """
        Calculate best theoretical lap time by combining best sectors.

        Args:
            df: DataFrame with lap data
            driver_name: Name of the driver

        Returns:
            Dictionary with theoretical lap time and sector times
        """
        driver_laps = df[df['DRIVER_NAME'] == driver_name]

        if driver_laps.empty:
            return {}

        best_s1 = driver_laps['S1_SECONDS'].min()
        best_s2 = driver_laps['S2_SECONDS'].min()
        best_s3 = driver_laps['S3_SECONDS'].min()

        theoretical_time = best_s1 + best_s2 + best_s3
        actual_best = driver_laps['LAP_TIME_SECONDS'].min() if 'LAP_TIME_SECONDS' in driver_laps.columns else None

        result = {
            'theoretical_lap_time': theoretical_time,
            'actual_best_lap': actual_best,
            'improvement_potential': actual_best - theoretical_time if actual_best else None,
            'best_S1': best_s1,
            'best_S2': best_s2,
            'best_S3': best_s3
        }

        return result

    def get_available_sessions(self) -> List[str]:
        """
        Get list of available session names.

        Returns:
            List of unique session names
        """
        all_laps = self.load_all_lap_data()

        if all_laps.empty or 'session' not in all_laps.columns:
            return []

        sessions = sorted(all_laps['session'].unique())
        logger.info(f"Found {len(sessions)} unique sessions")
        return sessions

    def get_available_drivers(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of available driver names.

        Args:
            df: DataFrame with lap data

        Returns:
            List of unique driver names
        """
        if df.empty or 'DRIVER_NAME' not in df.columns:
            return []

        drivers = sorted(df['DRIVER_NAME'].dropna().unique())
        return drivers


def main():
    """Example usage and testing."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize loader
    loader = LapAnalysisLoader()

    # Load all lap data
    print("=== Loading All Lap Data ===")
    laps_df = loader.load_all_lap_data()
    print(f"Total laps loaded: {len(laps_df)}")
    print(f"\nColumns: {laps_df.columns.tolist()}")

    # Get sector statistics
    print("\n=== Sector Statistics (Overall) ===")
    stats = loader.get_sector_statistics(laps_df)
    for sector, sector_stats in stats.items():
        print(f"\n{sector}:")
        print(f"  Best (min): {sector_stats['min']:.3f}s")
        print(f"  Average: {sector_stats['mean']:.3f}s")
        print(f"  90th percentile: {sector_stats['p90']:.3f}s")

    # Get available drivers
    drivers = loader.get_available_drivers(laps_df)
    print(f"\n=== Available Drivers ({len(drivers)}) ===")
    print(drivers[:10] if len(drivers) > 10 else drivers)

    # Test with first driver
    if drivers:
        test_driver = drivers[0]
        print(f"\n=== Driver Performance: {test_driver} ===")
        perf = loader.get_driver_sector_performance(laps_df, test_driver)
        for key, value in perf.items():
            print(f"  {key}: {value}")

        # Calculate percentile ranks
        print(f"\n=== Percentile Ranks for {test_driver} ===")
        for sector in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
            percentile = loader.calculate_percentile_rank(laps_df, test_driver, sector)
            if percentile is not None:
                print(f"  {sector.replace('_SECONDS', '')}: {percentile:.1f}th percentile")

        # Get theoretical best lap
        print(f"\n=== Best Theoretical Lap for {test_driver} ===")
        theoretical = loader.get_best_theoretical_lap(laps_df, test_driver)
        for key, value in theoretical.items():
            if value is not None:
                print(f"  {key}: {value:.3f}s")


if __name__ == "__main__":
    main()
