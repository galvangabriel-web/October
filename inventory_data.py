#!/usr/bin/env python3
"""
Quick script to inventory all data files in the project
"""
import os
from pathlib import Path
import pandas as pd

def get_dir_size(path):
    """Get total size of directory in MB"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except (OSError, PermissionError) as e:
        import warnings
        warnings.warn(f"Cannot access {path}: {e}")
    return total / (1024 * 1024)  # Convert to MB

def inventory_organized_data():
    """Inventory all organized data"""
    base_dir = Path('organized_data')

    inventory = []

    # Iterate through tracks
    for track_dir in sorted(base_dir.iterdir()):
        if not track_dir.is_dir():
            continue

        track_name = track_dir.name

        # Iterate through races (could be race directories or direct CSV files)
        for item in sorted(track_dir.iterdir()):
            if item.is_dir() and item.name.startswith('race'):
                # This is a race directory
                race_name = item.name

                # Check categories
                for category_dir in sorted(item.iterdir()):
                    if not category_dir.is_dir():
                        continue

                    category = category_dir.name

                    # Count files
                    csv_files = list(category_dir.glob('*.csv'))
                    chunk_files = [f for f in csv_files if '_chunk_' in f.name]
                    regular_files = [f for f in csv_files if '_chunk_' not in f.name]
                    metadata_files = list(category_dir.glob('*_metadata.txt'))

                    # Calculate size
                    size_mb = get_dir_size(category_dir)

                    inventory.append({
                        'track': track_name,
                        'race': race_name,
                        'category': category,
                        'regular_files': len(regular_files),
                        'chunked_files': len(chunk_files),
                        'metadata_files': len(metadata_files),
                        'size_mb': round(size_mb, 2)
                    })

            elif item.is_file() and item.suffix == '.csv':
                # Direct CSV files in track directory
                size_mb = item.stat().st_size / (1024 * 1024)
                inventory.append({
                    'track': track_name,
                    'race': 'direct_files',
                    'category': 'unknown',
                    'regular_files': 1,
                    'chunked_files': 0,
                    'metadata_files': 0,
                    'size_mb': round(size_mb, 2)
                })

    return pd.DataFrame(inventory)

if __name__ == '__main__':
    print("=" * 80)
    print("DATA INVENTORY FOR RACING TELEMETRY PROJECT")
    print("=" * 80)

    df = inventory_organized_data()

    print("\n1. SUMMARY BY TRACK:")
    print("-" * 80)
    track_summary = df.groupby('track').agg({
        'size_mb': 'sum',
        'regular_files': 'sum',
        'chunked_files': 'sum'
    }).round(2)
    print(track_summary)
    print(f"\nTotal size: {df['size_mb'].sum():.2f} MB ({df['size_mb'].sum()/1024:.2f} GB)")

    print("\n2. SUMMARY BY CATEGORY:")
    print("-" * 80)
    category_summary = df.groupby('category').agg({
        'size_mb': 'sum',
        'regular_files': 'sum',
        'chunked_files': 'sum'
    }).round(2)
    print(category_summary)

    print("\n3. DETAILED INVENTORY:")
    print("-" * 80)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False))

    print("\n4. CHUNKED FILES (Large datasets >50MB):")
    print("-" * 80)
    chunked = df[df['chunked_files'] > 0][['track', 'race', 'category', 'chunked_files', 'size_mb']]
    if len(chunked) > 0:
        print(chunked.to_string(index=False))
    else:
        print("No chunked files found")

    print("\n" + "=" * 80)
    print("Inventory complete!")
    print("=" * 80)
