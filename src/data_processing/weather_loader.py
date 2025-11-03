"""
Weather Data Loader for Alkamel Racing Data

This module loads and processes weather data from Alkamel timing system CSV files.
Weather data includes air temperature, track temperature, humidity, wind, and rain conditions.

Usage:
    from src.data_processing.weather_loader import WeatherDataLoader

    loader = WeatherDataLoader()
    weather_df = loader.load_all_weather_data()

    # Or load specific session
    race_weather = loader.load_weather_by_session('Race 1')
"""

import pandas as pd
import glob
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WeatherDataLoader:
    """
    Loads and processes weather data from Alkamel CSV files.

    Weather files contain:
    - TIME_UTC_SECONDS: Unix timestamp
    - TIME_UTC_STR: Human-readable timestamp
    - AIR_TEMP: Air temperature (°C)
    - TRACK_TEMP: Track surface temperature (°C)
    - HUMIDITY: Relative humidity (%)
    - PRESSURE: Atmospheric pressure (mbar)
    - WIND_SPEED: Wind speed (km/h or mph)
    - WIND_DIRECTION: Wind direction (degrees)
    - RAIN: Rain indicator (0/1)
    """

    def __init__(self, data_dir: str = "new_data/alkamel_downloads"):
        """
        Initialize the weather data loader.

        Args:
            data_dir: Directory containing Alkamel CSV files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        logger.info(f"Initialized WeatherDataLoader with data_dir: {self.data_dir}")

    def find_weather_files(self) -> List[Path]:
        """
        Find all weather CSV files in the data directory.

        Returns:
            List of Path objects for weather files
        """
        # Pattern matches files like: "26_Weather_Race 1.CSV"
        weather_files = list(self.data_dir.glob("*Weather*.CSV"))
        weather_files.extend(list(self.data_dir.glob("*Weather*.csv")))

        logger.info(f"Found {len(weather_files)} weather files")
        return weather_files

    def load_weather_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single weather CSV file.

        Args:
            file_path: Path to weather CSV file

        Returns:
            DataFrame with weather data
        """
        try:
            # Alkamel CSVs use semicolon delimiter
            df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')

            # Clean column names (remove leading/trailing spaces and semicolons)
            df.columns = df.columns.str.strip().str.strip(';')

            # Extract session name from filename
            # Example: "26_Weather_Race 1.CSV" -> "Race 1"
            filename = file_path.name
            session = filename.split('_Weather_')[-1].replace('.CSV', '').replace('.csv', '')
            df['session'] = session
            df['filename'] = filename

            # Convert timestamp to datetime
            if 'TIME_UTC_SECONDS' in df.columns:
                df['timestamp'] = pd.to_datetime(df['TIME_UTC_SECONDS'], unit='s', errors='coerce')

            # Ensure numeric columns are properly typed
            numeric_cols = ['AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'PRESSURE',
                          'WIND_SPEED', 'WIND_DIRECTION', 'RAIN']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            logger.debug(f"Loaded {len(df)} weather records from {filename}")
            return df

        except Exception as e:
            logger.error(f"Error loading weather file {file_path}: {e}")
            raise

    def load_all_weather_data(self) -> pd.DataFrame:
        """
        Load all weather files and combine into a single DataFrame.

        Returns:
            Combined DataFrame with all weather data
        """
        weather_files = self.find_weather_files()

        if not weather_files:
            logger.warning("No weather files found!")
            return pd.DataFrame()

        all_weather = []
        for file_path in weather_files:
            try:
                df = self.load_weather_file(file_path)
                all_weather.append(df)
            except Exception as e:
                logger.warning(f"Skipping file {file_path.name} due to error: {e}")
                continue

        if not all_weather:
            logger.warning("No weather data could be loaded!")
            return pd.DataFrame()

        # Combine all dataframes
        combined = pd.concat(all_weather, ignore_index=True)

        # Sort by timestamp
        if 'timestamp' in combined.columns:
            combined = combined.sort_values('timestamp')

        logger.info(f"Loaded {len(combined)} total weather records from {len(all_weather)} files")
        return combined

    def load_weather_by_session(self, session_pattern: str) -> pd.DataFrame:
        """
        Load weather data for sessions matching a pattern.

        Args:
            session_pattern: Pattern to match session names (e.g., "Race 1", "Practice")

        Returns:
            DataFrame with weather data for matching sessions
        """
        all_weather = self.load_all_weather_data()

        if all_weather.empty:
            return all_weather

        # Filter by session pattern (case-insensitive)
        mask = all_weather['session'].str.contains(session_pattern, case=False, na=False)
        filtered = all_weather[mask]

        logger.info(f"Found {len(filtered)} weather records for session pattern '{session_pattern}'")
        return filtered

    def get_session_summary(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get summary statistics for weather data.

        Args:
            df: DataFrame with weather data

        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}

        summary = {
            'avg_air_temp': df['AIR_TEMP'].mean() if 'AIR_TEMP' in df.columns else None,
            'avg_track_temp': df['TRACK_TEMP'].mean() if 'TRACK_TEMP' in df.columns else None,
            'avg_humidity': df['HUMIDITY'].mean() if 'HUMIDITY' in df.columns else None,
            'avg_pressure': df['PRESSURE'].mean() if 'PRESSURE' in df.columns else None,
            'avg_wind_speed': df['WIND_SPEED'].mean() if 'WIND_SPEED' in df.columns else None,
            'max_air_temp': df['AIR_TEMP'].max() if 'AIR_TEMP' in df.columns else None,
            'max_track_temp': df['TRACK_TEMP'].max() if 'TRACK_TEMP' in df.columns else None,
            'min_air_temp': df['AIR_TEMP'].min() if 'AIR_TEMP' in df.columns else None,
            'min_track_temp': df['TRACK_TEMP'].min() if 'TRACK_TEMP' in df.columns else None,
            'rain_detected': bool(df['RAIN'].max() if 'RAIN' in df.columns else False),
            'num_readings': len(df)
        }

        return {k: v for k, v in summary.items() if v is not None}

    def get_available_sessions(self) -> List[str]:
        """
        Get list of available session names.

        Returns:
            List of unique session names
        """
        all_weather = self.load_all_weather_data()

        if all_weather.empty or 'session' not in all_weather.columns:
            return []

        sessions = sorted(all_weather['session'].unique())
        logger.info(f"Found {len(sessions)} unique sessions: {sessions}")
        return sessions


def main():
    """Example usage and testing."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize loader
    loader = WeatherDataLoader()

    # Get available sessions
    print("=== Available Sessions ===")
    sessions = loader.get_available_sessions()
    for session in sessions:
        print(f"  - {session}")

    # Load all weather data
    print("\n=== Loading All Weather Data ===")
    weather_df = loader.load_all_weather_data()
    print(f"Total weather records: {len(weather_df)}")
    print(f"\nColumns: {weather_df.columns.tolist()}")
    print(f"\nFirst 5 records:")
    print(weather_df.head())

    # Get summary for a specific session
    if sessions:
        test_session = sessions[0]
        print(f"\n=== Weather Summary for '{test_session}' ===")
        session_weather = loader.load_weather_by_session(test_session)
        summary = loader.get_session_summary(session_weather)
        for key, value in summary.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
