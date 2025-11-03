"""
Championship Data Loader for Alkamel Racing Data

This module loads and processes championship standings data from Alkamel timing system.
Includes driver/team championships with track-by-track points breakdown.

Usage:
    from src.data_processing.championship_loader import ChampionshipLoader

    loader = ChampionshipLoader()
    standings = loader.load_championship('GR Cup', 'Drivers')
"""

import pandas as pd
import glob
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class ChampionshipLoader:
    """
    Loads and processes championship standings data.

    Championship files contain:
    - Pos: Current championship position
    - Participant: Driver/Team name
    - Points: Total championship points
    - Track-by-track points breakdown (Race 1, Race 2 per track)
    - Pole points, Fastest lap points per race
    """

    def __init__(self, data_dir: str = "new_data/alkamel_downloads"):
        """
        Initialize the championship loader.

        Args:
            data_dir: Directory containing Alkamel CSV files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        logger.info(f"Initialized ChampionshipLoader with data_dir: {self.data_dir}")

    def find_championship_files(self) -> List[Path]:
        """
        Find all championship CSV files.

        Returns:
            List of Path objects for championship files
        """
        files = list(self.data_dir.glob("*Championship*.csv"))
        files.extend(list(self.data_dir.glob("*Championship*.CSV")))
        files.extend(list(self.data_dir.glob("*Trophy*.csv")))

        logger.info(f"Found {len(files)} championship files")
        return files

    def load_championship_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single championship CSV file.

        Args:
            file_path: Path to championship CSV file

        Returns:
            DataFrame with championship data
        """
        try:
            df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
            df.columns = df.columns.str.strip()

            # Extract championship info from filename
            filename = file_path.name
            df['championship_file'] = filename

            logger.debug(f"Loaded {len(df)} entries from {filename}")
            return df

        except Exception as e:
            logger.error(f"Error loading championship file {file_path}: {e}")
            raise

    def get_available_championships(self) -> List[Dict[str, str]]:
        """
        Get list of available championships.

        Returns:
            List of dicts with championship info
        """
        files = self.find_championship_files()
        championships = []

        for file_path in files:
            filename = file_path.name
            # Parse championship name from filename
            championships.append({
                'name': filename.replace('.csv', '').replace('.CSV', '').replace('%20', ' '),
                'file': filename,
                'path': str(file_path)
            })

        return championships

    def load_championship(self, championship_name: str) -> Optional[pd.DataFrame]:
        """
        Load a specific championship by name.

        Args:
            championship_name: Name or pattern to match championship

        Returns:
            DataFrame with championship standings
        """
        import urllib.parse
        files = self.find_championship_files()

        # Find matching file (handle URL-encoded filenames)
        for file_path in files:
            # URL-decode the filename for comparison
            decoded_name = urllib.parse.unquote(file_path.name)
            if championship_name.lower() in decoded_name.lower():
                return self.load_championship_file(file_path)

        logger.warning(f"No championship found matching '{championship_name}'")
        return None

    def extract_track_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract track-by-track points into a long format DataFrame.

        Args:
            df: Championship DataFrame

        Returns:
            Long format DataFrame with track points
        """
        # Find all track columns (format: TrackName_Race X_Points)
        track_cols = [col for col in df.columns if '_Race' in col and '_Points' in col and 'Extra' not in col]

        # Extract unique tracks
        tracks = set()
        for col in track_cols:
            track_name = col.split('_Race')[0]
            tracks.add(track_name)

        # Build long format data
        long_data = []

        for idx, row in df.iterrows():
            participant = row.get('Participant', 'Unknown')
            total_points = row.get('Points', 0)

            for track in tracks:
                # Get points for Race 1 and Race 2
                race1_col = f"{track}_Race 1_Points"
                race2_col = f"{track}_Race 2_Points"

                race1_points = row.get(race1_col, 0) if race1_col in df.columns else 0
                race2_points = row.get(race2_col, 0) if race2_col in df.columns else 0

                # Convert to numeric, handling NaN
                # Fix for Issue #003: Use specific exception types instead of bare except
                try:
                    race1_points = float(race1_points) if pd.notna(race1_points) else 0
                    race2_points = float(race2_points) if pd.notna(race2_points) else 0
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid points value for {participant} at {track}: {e}")
                    race1_points = 0
                    race2_points = 0

                if race1_points > 0 or race2_points > 0:
                    long_data.append({
                        'Participant': participant,
                        'Track': track,
                        'Race 1 Points': race1_points,
                        'Race 2 Points': race2_points,
                        'Track Total': race1_points + race2_points
                    })

        return pd.DataFrame(long_data)

    def get_points_progression(self, df: pd.DataFrame, participant_name: str) -> pd.DataFrame:
        """
        Get cumulative points progression for a participant.

        Args:
            df: Championship DataFrame
            participant_name: Name of participant

        Returns:
            DataFrame with cumulative points by track
        """
        track_points = self.extract_track_points(df)

        # Filter for participant
        participant_data = track_points[track_points['Participant'] == participant_name]

        if participant_data.empty:
            return pd.DataFrame()

        # Sort by track (this is simplified - ideally would sort by race order)
        participant_data = participant_data.sort_values('Track')

        # Calculate cumulative points
        participant_data['Cumulative Points'] = participant_data['Track Total'].cumsum()

        return participant_data

    def get_top_n(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Get top N participants from championship.

        Args:
            df: Championship DataFrame
            n: Number of top participants

        Returns:
            DataFrame with top N participants
        """
        if df.empty:
            return df

        # Sort by position or points
        if 'Pos' in df.columns:
            df_sorted = df.sort_values('Pos')
        elif 'Points' in df.columns:
            df_sorted = df.sort_values('Points', ascending=False)
        else:
            df_sorted = df

        return df_sorted.head(n)

    def load_all_championships(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available championships into a dictionary.

        Returns:
            Dict mapping championship names to DataFrames

        Example:
            loader = ChampionshipLoader()
            all_champs = loader.load_all_championships()
            for name, df in all_champs.items():
                print(f"{name}: {len(df)} entries")
        """
        championships = {}
        available = self.get_available_championships()

        logger.info(f"Loading {len(available)} championships...")

        for champ in available:
            try:
                df = self.load_championship(champ['name'])
                if df is not None and not df.empty:
                    championships[champ['name']] = df
                    logger.debug(f"Loaded championship '{champ['name']}' with {len(df)} entries")
                else:
                    logger.warning(f"Championship '{champ['name']}' returned empty DataFrame")
            except Exception as e:
                logger.warning(f"Failed to load championship '{champ['name']}': {e}")
                # Continue loading other championships

        logger.info(f"Successfully loaded {len(championships)} championships")
        return championships


def main():
    """Example usage and testing."""
    logging.basicConfig(level=logging.INFO)

    loader = ChampionshipLoader()

    # Get available championships
    print("=== Available Championships ===")
    championships = loader.get_available_championships()
    for champ in championships[:10]:
        print(f"  - {champ['name']}")

    # Load first championship
    if championships:
        print(f"\n=== Loading: {championships[0]['name']} ===")
        df = loader.load_championship(championships[0]['name'])

        if df is not None:
            print(f"Total entries: {len(df)}")
            print(f"\nTop 10:")
            top10 = loader.get_top_n(df, 10)
            print(top10[['Pos', 'Participant', 'Points', 'TEAM']].to_string())

            # Track points
            print("\n=== Track-by-Track Points ===")
            track_points = loader.extract_track_points(df)
            print(f"Total track entries: {len(track_points)}")
            print(track_points.head(10))

            # Points progression for top driver
            if not df.empty:
                top_driver = df.iloc[0]['Participant']
                print(f"\n=== Points Progression: {top_driver} ===")
                progression = loader.get_points_progression(df, top_driver)
                print(progression)


if __name__ == "__main__":
    main()
