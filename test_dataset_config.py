#!/usr/bin/env python3
"""
Test script for dataset configuration system.

Tests:
1. Configuration loading
2. Path resolution (tab-specific and global)
3. Dataset loading with validation
4. Fallback behavior
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import DatasetConfig, DatasetLoader, DatasetValidationError


def test_config_loading():
    """Test 1: Configuration loading"""
    print("=" * 60)
    print("TEST 1: Configuration Loading")
    print("=" * 60)

    try:
        config = DatasetConfig()
        print("✅ Configuration loaded successfully")
        print(f"   Config path: {config.config_path}")
        print(f"   Global dataset: {config.config['global']['dataset_path']}")
        print(f"   Tabs configured: {list(config.config.get('tabs', {}).keys())}")
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def test_path_resolution():
    """Test 2: Path resolution"""
    print("\n" + "=" * 60)
    print("TEST 2: Path Resolution")
    print("=" * 60)

    try:
        config = DatasetConfig()

        # Test global path
        global_path = config.get_dataset_path()
        print(f"✅ Global dataset path: {global_path}")

        # Test Winning Edge tab path
        winning_edge_path = config.get_dataset_path("winning_edge")
        print(f"✅ Winning Edge dataset path: {winning_edge_path}")

        # Verify file exists
        if Path(winning_edge_path).exists():
            file_size = Path(winning_edge_path).stat().st_size / (1024 * 1024)
            print(f"   File exists: {file_size:.1f} MB")
        else:
            print(f"   ⚠️  File not found: {winning_edge_path}")

        # Test fallback for unconfigured tab
        fallback_path = config.get_dataset_path("nonexistent_tab")
        print(f"✅ Fallback path for unconfigured tab: {fallback_path}")

        return True

    except Exception as e:
        print(f"❌ Path resolution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test 3: Dataset loading with validation"""
    print("\n" + "=" * 60)
    print("TEST 3: Dataset Loading & Validation")
    print("=" * 60)

    try:
        loader = DatasetLoader()

        # Try to load Winning Edge dataset
        print("Loading Winning Edge dataset...")
        df = loader.load_dataset("winning_edge", validate=True)

        print(f"✅ Dataset loaded successfully")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.1f} MB")

        # Check data characteristics
        if 'vehicle_number' in df.columns:
            vehicles = df['vehicle_number'].nunique()
            print(f"   Vehicles: {vehicles}")

        if 'lap' in df.columns:
            laps = df['lap'].nunique()
            print(f"   Laps: {laps}")

        if 'telemetry_name' in df.columns:
            sensors = df['telemetry_name'].nunique()
            print(f"   Sensors: {sensors}")
            print(f"   Sensor list: {sorted(df['telemetry_name'].unique())}")

        return True

    except DatasetValidationError as e:
        print(f"❌ Dataset validation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_info():
    """Test 4: Dataset metadata"""
    print("\n" + "=" * 60)
    print("TEST 4: Dataset Metadata")
    print("=" * 60)

    try:
        loader = DatasetLoader()

        # Get metadata without loading
        info = loader.get_dataset_info("winning_edge")

        print("✅ Dataset metadata retrieved")
        for key, value in info.items():
            print(f"   {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ Metadata retrieval failed: {e}")
        return False


def test_fallback_behavior():
    """Test 5: Fallback behavior"""
    print("\n" + "=" * 60)
    print("TEST 5: Fallback Behavior")
    print("=" * 60)

    try:
        config = DatasetConfig()

        # Test with disabled tab (should fallback to global)
        post_race_path = config.get_dataset_path("post_race")
        print(f"✅ Post-Race tab (disabled) falls back to: {post_race_path}")

        # Test with non-existent tab (should fallback to global)
        unknown_path = config.get_dataset_path("unknown_tab")
        print(f"✅ Unknown tab falls back to: {unknown_path}")

        return True

    except Exception as e:
        print(f"❌ Fallback behavior test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("DATASET CONFIGURATION SYSTEM TEST SUITE")
    print("=" * 60)

    results = {
        "Configuration Loading": test_config_loading(),
        "Path Resolution": test_path_resolution(),
        "Dataset Loading": test_dataset_loading(),
        "Dataset Metadata": test_dataset_info(),
        "Fallback Behavior": test_fallback_behavior(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {test_name}")

    print(f"\nTotal: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✅ ALL TESTS PASSED - Configuration system ready for production!")
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
