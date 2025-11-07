"""
Test script for SimplePostRacePredictor
"""

import pandas as pd
import sys
from pathlib import Path

# Fix Windows console encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.inference.simple_post_race_predictor import SimplePostRacePredictor

def test_simple_predictor():
    """Test the simple predictor with template data"""

    print("\n" + "="*70)
    print("TESTING SIMPLE POST-RACE PREDICTOR")
    print("="*70 + "\n")

    # Load template CSV
    template_path = Path("post_race_sample_template.csv")
    if not template_path.exists():
        print(f"❌ Template not found: {template_path}")
        return False

    print(f"[1] Loading template CSV: {template_path}")
    df = pd.read_csv(template_path)
    print(f"  ✓ Loaded {len(df)} rows")
    print(f"  ✓ Columns: {df.columns.tolist()}")
    print(f"  ✓ Unique sensors: {df['telemetry_name'].unique().tolist()}")
    print(f"  ✓ Laps: {sorted(df['lap'].unique())}")
    print(f"  ✓ Vehicles: {sorted(df['vehicle_number'].unique())}")

    # Calculate lap times
    print("\n[2] Calculating lap times...")
    lap_times_list = []
    for (vehicle, lap), group in df.groupby(['vehicle_number', 'lap']):
        timestamps = pd.to_datetime(group['timestamp'])
        lap_duration = (timestamps.max() - timestamps.min()).total_seconds()

        lap_times_list.append({
            'vehicle_number': vehicle,
            'lap_number': int(lap),
            'lap_time': lap_duration,
            'track': group['track'].iloc[0],
            'race': group['race'].iloc[0]
        })

    lap_times_df = pd.DataFrame(lap_times_list)
    print(f"  ✓ Calculated {len(lap_times_df)} lap times")
    print(f"  ✓ Lap time range: {lap_times_df['lap_time'].min():.1f}s - {lap_times_df['lap_time'].max():.1f}s")

    # Initialize predictor
    print("\n[3] Initializing SimplePostRacePredictor...")
    try:
        predictor = SimplePostRacePredictor()
        print("  ✓ Predictor initialized")
    except Exception as e:
        print(f"  ❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Make predictions
    print("\n[4] Making predictions...")
    try:
        results = predictor.predict_session(df, lap_times_df)
        print(f"  ✓ Predictions successful!")
        print(f"\n  Results summary:")
        print(f"    Laps predicted: {len(results)}")
        print(f"    Average error: {results['abs_error'].mean():.2f}s")
        print(f"    Max error: {results['abs_error'].max():.2f}s")
        print(f"    Min error: {results['abs_error'].min():.2f}s")

        print(f"\n  Sample predictions:")
        print(results[['lap_number', 'actual', 'predicted', 'error']].to_string())

        print("\n" + "="*70)
        print("✅ TEST PASSED - SimplePostRacePredictor is working!")
        print("="*70 + "\n")
        return True

    except Exception as e:
        print(f"  ❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simple_predictor()
    sys.exit(0 if success else 1)
