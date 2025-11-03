#!/usr/bin/env python3
"""
Helper script for loading and working with organized racing data.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List


class RacingDataLoader:
    """Utility class for loading racing data from the organized structure."""

    def __init__(self, base_dir: str = "organized_data"):
        """
        Initialize the data loader.

        Args:
            base_dir: Path to the organized_data directory
        """
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Organized data directory not found: {base_dir}")

    def list_tracks(self) -> List[str]:
        """List all available tracks."""
        tracks = [d.name for d in self.base_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]
        return sorted(tracks)

    def list_races(self, track: str) -> List[str]:
        """List all races for a given track."""
        track_dir = self.base_dir / track
        if not track_dir.exists():
            raise ValueError(f"Track not found: {track}")

        races = [d.name for d in track_dir.iterdir() if d.is_dir()]
        return sorted(races)

    def list_categories(self, track: str, race: str) -> List[str]:
        """List all data categories for a given track and race."""
        race_dir = self.base_dir / track / race
        if not race_dir.exists():
            raise ValueError(f"Race not found: {track}/{race}")

        categories = [d.name for d in race_dir.iterdir() if d.is_dir()]
        return sorted(categories)

    def load_data(
        self,
        track: str,
        race: str,
        category: str,
        file_pattern: Optional[str] = None,
        combine_chunks: bool = True
    ) -> pd.DataFrame:
        """
        Load data from the organized structure.

        Args:
            track: Track name (e.g., 'barber-motorsports-park')
            race: Race identifier (e.g., 'race_1', 'race_unknown')
            category: Data category (e.g., 'telemetry', 'lap_times', 'weather')
            file_pattern: Optional file pattern to match (e.g., '*barber*R1*')
            combine_chunks: If True, automatically combine chunked files

        Returns:
            DataFrame containing the loaded data
        """
        data_dir = self.base_dir / track / race / category

        if not data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")

        # Find files matching the pattern
        if file_pattern:
            files = list(data_dir.glob(file_pattern))
        else:
            files = list(data_dir.glob("*.csv"))

        # Filter out metadata files
        files = [f for f in files if not f.name.endswith('_metadata.txt')]

        if not files:
            raise ValueError(f"No data files found in {data_dir}")

        # Check if these are chunked files
        chunk_files = [f for f in files if '_chunk_' in f.name]

        if chunk_files and combine_chunks:
            # Group chunks by base filename
            chunks_by_base = {}
            for chunk_file in chunk_files:
                base_name = chunk_file.name.rsplit('_chunk_', 1)[0]
                if base_name not in chunks_by_base:
                    chunks_by_base[base_name] = []
                chunks_by_base[base_name].append(chunk_file)

            # Load and combine all chunks
            all_dfs = []
            for base_name, chunks in sorted(chunks_by_base.items()):
                print(f"Loading {len(chunks)} chunks for {base_name}...")
                sorted_chunks = sorted(chunks, key=lambda x: x.name)

                chunk_dfs = []
                for i, chunk_file in enumerate(sorted_chunks, 1):
                    if i % 10 == 0:
                        print(f"  Loading chunk {i}/{len(sorted_chunks)}...")
                    chunk_dfs.append(pd.read_csv(chunk_file, low_memory=False))

                df = pd.concat(chunk_dfs, ignore_index=True)
                del chunk_dfs  # Free memory immediately
                all_dfs.append(df)
                print(f"  Loaded {len(df):,} rows for {base_name}")

            # Combine all dataframes
            if len(all_dfs) == 0:
                raise ValueError(f"No valid data found in {data_dir}")
            elif len(all_dfs) > 1:
                print(f"\nCombining {len(all_dfs)} files...")
                result = pd.concat(all_dfs, ignore_index=True)
            else:
                result = all_dfs[0]

            return result
        else:
            # Load regular files
            dfs = []
            for file in files:
                print(f"Loading {file.name}...")
                dfs.append(pd.read_csv(file, low_memory=False))

            if len(dfs) > 1:
                return pd.concat(dfs, ignore_index=True)
            else:
                return dfs[0]

    def load_single_chunk(self, track: str, race: str, category: str, chunk_num: int = 1) -> pd.DataFrame:
        """
        Load a single chunk file for quick exploration.

        Args:
            track: Track name
            race: Race identifier
            category: Data category
            chunk_num: Chunk number to load (default: 1)

        Returns:
            DataFrame containing the chunk data
        """
        data_dir = self.base_dir / track / race / category
        chunk_pattern = f"*_chunk_{chunk_num:03d}.csv"

        chunk_files = list(data_dir.glob(chunk_pattern))

        if not chunk_files:
            raise ValueError(f"No chunk file found matching {chunk_pattern} in {data_dir}")

        chunk_file = chunk_files[0]
        print(f"Loading {chunk_file.name}...")
        return pd.read_csv(chunk_file, low_memory=False)

    def get_file_info(self, track: str, race: str, category: str) -> dict:
        """
        Get information about files in a category.

        Args:
            track: Track name
            race: Race identifier
            category: Data category

        Returns:
            Dictionary with file information
        """
        data_dir = self.base_dir / track / race / category

        if not data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")

        # Get metadata files
        metadata_files = list(data_dir.glob("*_metadata.txt"))

        info = {
            'directory': str(data_dir),
            'files': [],
            'chunked_files': []
        }

        # Process metadata files
        for meta_file in metadata_files:
            with open(meta_file, 'r') as f:
                content = f.read()
                info['chunked_files'].append({
                    'metadata_file': meta_file.name,
                    'content': content
                })

        # Get regular files
        csv_files = list(data_dir.glob("*.csv"))
        csv_files = [f for f in csv_files if '_chunk_' not in f.name]

        for csv_file in csv_files:
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            info['files'].append({
                'filename': csv_file.name,
                'size_mb': round(size_mb, 2)
            })

        return info


# Example usage functions
def example_load_lap_times():
    """Example: Load lap times data."""
    loader = RacingDataLoader()

    # Load lap times for a specific track and race
    df = loader.load_data(
        track='barber-motorsports-park',
        race='race_unknown',
        category='lap_times'
    )

    print(f"Loaded {len(df):,} rows of lap time data")
    print(f"Columns: {df.columns.tolist()}")
    return df


def example_load_single_telemetry_chunk():
    """Example: Load a single telemetry chunk for quick exploration."""
    loader = RacingDataLoader()

    # Load just the first chunk of telemetry data
    df = loader.load_single_chunk(
        track='barber-motorsports-park',
        race='race_unknown',
        category='telemetry',
        chunk_num=1
    )

    print(f"Loaded {len(df):,} rows from chunk 1")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    return df


def example_explore_data():
    """Example: Explore available data."""
    loader = RacingDataLoader()

    print("Available tracks:")
    for track in loader.list_tracks():
        print(f"  - {track}")
        races = loader.list_races(track)
        for race in races:
            categories = loader.list_categories(track, race)
            print(f"    - {race}: {', '.join(categories)}")


if __name__ == "__main__":
    print("=" * 80)
    print("Racing Data Loader - Example Usage")
    print("=" * 80)

    # Example 1: Explore available data
    print("\n1. Exploring available data:")
    print("-" * 80)
    example_explore_data()

    # Example 2: Load a single chunk
    print("\n2. Loading a single telemetry chunk:")
    print("-" * 80)
    df_chunk = example_load_single_telemetry_chunk()

    print("\n" + "=" * 80)
    print("Data loader ready! Import this module to use in your analysis.")
    print("=" * 80)
