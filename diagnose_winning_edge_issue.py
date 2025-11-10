#!/usr/bin/env python3
"""
Winning Edge Dashboard Diagnostic Script
=========================================

This script diagnoses why plots are not showing in the Winning Edge dashboard.

It checks:
1. Dataset files existence and format
2. Data loading via DatasetLoader
3. Data structure and required columns
4. Corner detection logic
5. Data processing helper functions
6. Callback simulation

Usage:
    python diagnose_winning_edge_issue.py
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Colors.RESET}\n")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")

# ============================================================================
# TEST 1: Check Dataset Files
# ============================================================================

def test_dataset_files():
    """Check if dataset files exist and are accessible."""
    print_header("TEST 1: Dataset File Existence Check")

    dataset_paths = [
        "data/winning_edge_dataset.csv",
        "data/barber_winning_edge_dataset.csv",
        "master_racing_data.csv",
        "master_racing_data_production.csv"
    ]

    results = {}
    for path_str in dataset_paths:
        path = Path(path_str)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print_success(f"Found: {path} ({size_mb:.1f} MB)")
            results[path_str] = {'exists': True, 'size_mb': size_mb, 'path': path}
        else:
            print_error(f"Missing: {path}")
            results[path_str] = {'exists': False}

    return results

# ============================================================================
# TEST 2: Load Dataset via DatasetLoader
# ============================================================================

def test_dataset_loader():
    """Test loading dataset using DatasetLoader."""
    print_header("TEST 2: DatasetLoader Configuration Check")

    try:
        from src.config import DatasetLoader, DatasetConfig
        print_success("DatasetLoader imported successfully")

        # Check configuration
        config = DatasetConfig()
        print_info(f"Config file: {config.config_path}")

        # Get winning edge dataset path
        we_path = config.get_dataset_path("winning_edge")
        print_info(f"Winning Edge dataset path: {we_path}")

        # Try to load
        loader = DatasetLoader()
        print_info("Attempting to load winning_edge dataset...")
        df = loader.load_dataset(tab_name="winning_edge", validate=True, use_cache=False)

        print_success(f"Dataset loaded: {len(df):,} rows")
        return df, loader

    except Exception as e:
        print_error(f"Failed to load via DatasetLoader: {e}")
        logger.exception("DatasetLoader error:")
        return None, None

# ============================================================================
# TEST 3: Validate Dataset Structure
# ============================================================================

def test_dataset_structure(df: pd.DataFrame):
    """Validate that the dataset has required columns and format."""
    print_header("TEST 3: Dataset Structure Validation")

    if df is None:
        print_error("No dataframe to validate")
        return False

    # Check required columns
    required_columns = [
        'vehicle_number', 'lap', 'timestamp',
        'telemetry_name', 'telemetry_value'
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print_error(f"Missing required columns: {missing_cols}")
        return False
    else:
        print_success("All required columns present")

    # Print column info
    print_info(f"Available columns: {list(df.columns)}")

    # Check data format (long format)
    if 'telemetry_name' in df.columns:
        sensors = df['telemetry_name'].unique()
        print_info(f"Found {len(sensors)} unique sensors")
        print_info(f"Sample sensors: {list(sensors[:10])}")

        # Check for required sensors
        required_sensors = ['speed', 'pbrake_f', 'pbrake_r', 'aps', 'Steering_Angle']
        found_sensors = [s for s in required_sensors if s in sensors]
        missing_sensors = [s for s in required_sensors if s not in sensors]

        if found_sensors:
            print_success(f"Found required sensors: {found_sensors}")
        if missing_sensors:
            print_warning(f"Missing recommended sensors: {missing_sensors}")

    # Check vehicles and laps
    vehicles = sorted(df['vehicle_number'].unique())
    print_info(f"Vehicles: {vehicles}")

    for vehicle in vehicles[:3]:  # Check first 3 vehicles
        vehicle_data = df[df['vehicle_number'] == vehicle]
        laps = sorted(vehicle_data['lap'].unique())
        print_info(f"  Vehicle {vehicle}: {len(laps)} laps ({min(laps)}-{max(laps)})")

    return True

# ============================================================================
# TEST 4: Test Corner Detection
# ============================================================================

def test_corner_detection(df: pd.DataFrame):
    """Test if corner detection logic works on the dataset."""
    print_header("TEST 4: Corner Detection Logic")

    if df is None:
        print_error("No dataframe to test")
        return False

    # Test with first vehicle
    vehicles = df['vehicle_number'].unique()
    if len(vehicles) == 0:
        print_error("No vehicles found in dataset")
        return False

    test_vehicle = vehicles[0]
    print_info(f"Testing with vehicle: {test_vehicle}")

    # Get telemetry for this vehicle
    vehicle_df = df[df['vehicle_number'] == test_vehicle].copy()
    print_info(f"Vehicle has {len(vehicle_df):,} telemetry samples")

    # Check if we have speed data
    speed_data = vehicle_df[vehicle_df['telemetry_name'] == 'speed']
    if len(speed_data) == 0:
        print_error("No speed telemetry found for corner detection")
        return False

    print_success(f"Found {len(speed_data):,} speed samples")

    # Analyze speed distribution
    speed_values = speed_data['telemetry_value'].values
    print_info(f"Speed range: {speed_values.min():.1f} - {speed_values.max():.1f} km/h")
    print_info(f"Speed mean: {speed_values.mean():.1f} km/h")
    print_info(f"Speed std: {speed_values.std():.1f} km/h")

    # Simple corner detection test (speed drops below threshold)
    corner_threshold = speed_values.mean() - 0.5 * speed_values.std()
    potential_corners = speed_values < corner_threshold
    num_potential_corners = np.sum(potential_corners)

    print_info(f"Potential corner points (speed < {corner_threshold:.1f}): {num_potential_corners}")

    return True

# ============================================================================
# TEST 5: Test Data Processing Helpers
# ============================================================================

def test_callback_helpers(df: pd.DataFrame):
    """Test the callback helper functions that process data for visualizations."""
    print_header("TEST 5: Callback Helper Functions")

    if df is None:
        print_error("No dataframe to test")
        return False

    try:
        from src.dashboard.winning_edge_callback_helpers import (
            process_corner_data_for_heatmap,
            process_speed_gap_data_for_spider,
            load_telemetry_from_json
        )
        print_success("Callback helpers imported successfully")

        # Test data conversion
        print_info("Testing JSON conversion...")
        data_json = df.to_json(date_format='iso', orient='split')
        print_success(f"Converted to JSON ({len(data_json):,} bytes)")

        # Test loading from JSON
        print_info("Testing load_telemetry_from_json...")
        loaded_df = load_telemetry_from_json(data_json)
        if loaded_df is not None:
            print_success(f"Loaded dataframe: {len(loaded_df):,} rows")
        else:
            print_error("Failed to load dataframe from JSON")
            return False

        # Test corner data processing
        vehicles = df['vehicle_number'].unique()
        if len(vehicles) > 0:
            test_vehicle = vehicles[0]
            print_info(f"Testing corner data processing for vehicle {test_vehicle}...")

            corner_data = process_corner_data_for_heatmap(loaded_df, test_vehicle)
            if corner_data:
                print_success(f"Processed corner data: {len(corner_data)} corners detected")
                for corner, data in list(corner_data.items())[:3]:
                    print_info(f"  {corner}: time_loss={data.get('time_loss', 'N/A'):.3f}s, "
                             f"pct={data.get('pct_of_total', 'N/A'):.1f}%")
            else:
                print_error("No corner data returned (empty dict)")
                return False

            # Test speed gap processing
            print_info(f"Testing speed gap processing for vehicle {test_vehicle}...")
            speed_gaps = process_speed_gap_data_for_spider(loaded_df, test_vehicle)
            if speed_gaps:
                print_success(f"Processed speed gaps: {len(speed_gaps)} corners")
            else:
                print_error("No speed gap data returned (empty dict)")
                return False

        return True

    except ImportError as e:
        print_error(f"Failed to import callback helpers: {e}")
        logger.exception("Import error:")
        return False
    except Exception as e:
        print_error(f"Error testing callback helpers: {e}")
        logger.exception("Processing error:")
        return False

# ============================================================================
# TEST 6: Simulate Dashboard Flow
# ============================================================================

def test_dashboard_flow(df: pd.DataFrame):
    """Simulate the complete dashboard data flow."""
    print_header("TEST 6: Dashboard Flow Simulation")

    if df is None:
        print_error("No dataframe to test")
        return False

    try:
        # Simulate what happens in app.py
        print_info("Step 1: Convert dataframe to JSON (simulating dcc.Store)...")
        data_json = df.to_json(date_format='iso', orient='split')
        print_success(f"JSON created: {len(data_json):,} characters")

        # Simulate what happens in callback
        print_info("Step 2: Load JSON in callback (simulating callback input)...")
        import io
        loaded_df = pd.read_json(io.StringIO(data_json), orient='split')
        print_success(f"Loaded in callback: {len(loaded_df):,} rows")

        # Check vehicle dropdown population
        print_info("Step 3: Populate vehicle dropdown...")
        vehicles = sorted(loaded_df['vehicle_number'].unique())
        vehicle_options = [{'label': f'Vehicle #{v}', 'value': v} for v in vehicles]
        print_success(f"Vehicle options: {vehicle_options}")

        # Simulate selecting a vehicle
        if len(vehicles) > 0:
            selected_vehicle = vehicles[0]
            print_info(f"Step 4: User selects vehicle {selected_vehicle}...")

            # Simulate heatmap callback
            print_info("Step 5: Generate heatmap...")
            from src.dashboard.winning_edge_callback_helpers import (
                process_corner_data_for_heatmap,
                load_telemetry_from_json
            )

            telemetry_df = load_telemetry_from_json(data_json)
            corner_data = process_corner_data_for_heatmap(telemetry_df, selected_vehicle)

            if corner_data:
                print_success(f"Heatmap data ready: {len(corner_data)} corners")

                # Try to create the actual figure
                from src.dashboard.winning_edge_widget import create_time_loss_heatmap
                fig = create_time_loss_heatmap(corner_data)

                if fig and hasattr(fig, 'data') and len(fig.data) > 0:
                    print_success("Heatmap figure created successfully!")
                    print_info(f"  Figure has {len(fig.data)} traces")
                else:
                    print_error("Heatmap figure is empty or invalid")
                    return False
            else:
                print_error("No corner data - heatmap will be empty")
                return False

        return True

    except Exception as e:
        print_error(f"Dashboard flow simulation failed: {e}")
        logger.exception("Simulation error:")
        return False

# ============================================================================
# TEST 7: Check Dataset Configuration
# ============================================================================

def test_dataset_config():
    """Check the dataset_config.yaml settings."""
    print_header("TEST 7: Dataset Configuration YAML")

    try:
        from src.config import DatasetConfig
        import yaml

        config = DatasetConfig()
        yaml_path = config.config_path

        print_info(f"Config file: {yaml_path}")

        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)

            print_success("YAML file loaded successfully")
            print_info("Configuration:")
            print(json.dumps(config_data, indent=2))

            # Check winning edge tab config
            if 'tabs' in config_data and 'winning_edge' in config_data['tabs']:
                we_config = config_data['tabs']['winning_edge']
                print_success(f"Winning Edge tab configuration found")
                print_info(f"  Dataset path: {we_config.get('dataset_path')}")
                print_info(f"  Enabled: {we_config.get('enabled')}")
                print_info(f"  Description: {we_config.get('description')}")
            else:
                print_error("No winning_edge tab configuration in YAML")
                return False
        else:
            print_error(f"Config file not found: {yaml_path}")
            return False

        return True

    except Exception as e:
        print_error(f"Failed to check config: {e}")
        logger.exception("Config error:")
        return False

# ============================================================================
# MAIN DIAGNOSTIC RUNNER
# ============================================================================

def run_all_diagnostics():
    """Run all diagnostic tests."""
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*18 + "WINNING EDGE DIAGNOSTIC TOOL" + " "*22 + "║")
    print("║" + " "*68 + "║")
    print("║" + "  This script will diagnose why plots are not showing" + " "*14 + "║")
    print("║" + "  in the Winning Edge dashboard." + " "*36 + "║")
    print("╚" + "═"*68 + "╝")
    print(Colors.RESET)

    results = {
        'files': False,
        'loader': False,
        'structure': False,
        'corners': False,
        'helpers': False,
        'flow': False,
        'config': False
    }

    # Test 1: Files
    dataset_files = test_dataset_files()
    results['files'] = any(info.get('exists') for info in dataset_files.values())

    # Test 7: Config (moved up to understand dataset loading)
    results['config'] = test_dataset_config()

    # Test 2: Loader
    df, loader = test_dataset_loader()
    results['loader'] = df is not None

    if df is not None:
        # Test 3: Structure
        results['structure'] = test_dataset_structure(df)

        # Test 4: Corners
        results['corners'] = test_corner_detection(df)

        # Test 5: Helpers
        results['helpers'] = test_callback_helpers(df)

        # Test 6: Flow
        results['flow'] = test_dashboard_flow(df)

    # Print summary
    print_header("DIAGNOSTIC SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        color = Colors.GREEN if passed else Colors.RED
        print(f"{color}{test_name.upper():20s}: {status}{Colors.RESET}")

    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.RESET}")

    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All diagnostics passed! The dashboard should be working.{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Some diagnostics failed. See errors above for details.{Colors.RESET}")
        print(f"{Colors.YELLOW}Recommendation: Review failed tests and check troubleshooting plan.{Colors.RESET}")

    return results

if __name__ == "__main__":
    try:
        results = run_all_diagnostics()

        # Exit with error code if any test failed
        exit_code = 0 if all(results.values()) else 1
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Diagnostic interrupted by user.{Colors.RESET}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.RESET}")
        logger.exception("Fatal error:")
        sys.exit(1)
