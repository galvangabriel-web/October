#!/usr/bin/env python3
"""
Dashboard Upload Diagnostic Script
===================================

This script tests the exact upload logic that the dashboard uses
to find WHERE the bug was introduced.

Run on server: python3 diagnose_dashboard.py
"""

import pandas as pd
import io
import base64
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, '/home/tactical/racing-dashboard')

def test_file_upload(csv_path):
    """Test the exact upload logic from the dashboard"""
    print("=" * 80)
    print("DASHBOARD UPLOAD DIAGNOSTIC")
    print("=" * 80)
    print()

    # Read the file
    print(f"📁 Reading file: {csv_path}")
    with open(csv_path, 'rb') as f:
        file_content = f.read()

    file_size_mb = len(file_content) / (1024 * 1024)
    print(f"   File size: {file_size_mb:.2f} MB")
    print()

    # Simulate the upload process (this is what the dashboard does)
    print("🔍 Step 1: Decoding file (simulating browser upload)...")
    try:
        # The dashboard receives base64 encoded data from browser
        # Let's simulate that
        encoded = base64.b64encode(file_content).decode('utf-8')
        content_string = f"data:text/csv;base64,{encoded}"

        # This is what the dashboard callback does
        content_type, content_string = content_string.split(',')
        decoded = base64.b64decode(content_string)
        print("   ✓ File decoding successful")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False
    print()

    # Step 2: Parse CSV
    print("🔍 Step 2: Parsing CSV...")
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        print(f"   ✓ CSV parsed: {len(df)} rows")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False
    print()

    # Step 3: Check columns
    print("🔍 Step 3: Checking columns...")
    print(f"   Columns found: {df.columns.tolist()}")
    print()

    # Step 4: Check for vehicle_number column (THIS IS WHERE IT MIGHT FAIL)
    print("🔍 Step 4: Looking for vehicle_number column...")
    if 'vehicle_number' in df.columns:
        print("   ✓ 'vehicle_number' column FOUND")
        vehicles = sorted(df['vehicle_number'].unique())
        print(f"   ✓ Found {len(vehicles)} vehicles: {vehicles[:10]}")  # Show first 10
    else:
        print("   ✗ 'vehicle_number' column NOT FOUND")
        print("   Available columns:")
        for col in df.columns:
            print(f"      - {col}")
        return False
    print()

    # Step 5: Check other expected columns
    print("🔍 Step 5: Checking other columns...")
    expected_cols = ['lap', 'timestamp', 'telemetry_name', 'telemetry_value']
    for col in expected_cols:
        if col in df.columns:
            print(f"   ✓ '{col}' found")
        else:
            print(f"   ⚠ '{col}' missing (optional)")
    print()

    # Step 6: Create vehicle dropdown options (what dashboard does)
    print("🔍 Step 6: Creating vehicle dropdown options...")
    try:
        vehicle_options = [{'label': f'Vehicle #{v}', 'value': v} for v in vehicles]
        print(f"   ✓ Created {len(vehicle_options)} dropdown options")
        print(f"   Sample: {vehicle_options[:3]}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False
    print()

    # Step 7: Get statistics
    print("🔍 Step 7: Calculating statistics...")
    num_samples = len(df)
    num_vehicles = len(vehicles)
    num_laps = df['lap'].nunique() if 'lap' in df.columns else 0
    print(f"   Samples: {num_samples:,}")
    print(f"   Vehicles: {num_vehicles}")
    print(f"   Laps: {num_laps}")
    print()

    # Step 8: Test JSON conversion (dashboard stores data as JSON)
    print("🔍 Step 8: Testing JSON conversion...")
    try:
        data_json = df.to_json(date_format='iso', orient='split')
        json_size_mb = len(data_json) / (1024 * 1024)
        print(f"   ✓ JSON conversion successful ({json_size_mb:.2f} MB)")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False
    print()

    print("=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("The file should work with the dashboard.")
    print("If upload still fails, the issue is in the dashboard callback registration.")
    print()

    return True

def check_dashboard_code():
    """Check if the dashboard code has the upload callback"""
    print("=" * 80)
    print("CHECKING DASHBOARD CODE")
    print("=" * 80)
    print()

    app_file = '/home/tactical/racing-dashboard/src/dashboard/app.py'
    print(f"📄 Reading: {app_file}")
    print()

    with open(app_file, 'r') as f:
        content = f.read()

    # Check for critical components
    checks = [
        ("handle_upload function", "def handle_upload(contents, filename):"),
        ("upload-telemetry component", "id='upload-telemetry'"),
        ("vehicle-dropdown component", "id='vehicle-dropdown'"),
        ("upload callback decorator", "@app.callback"),
        ("Output('vehicle-dropdown', 'options')", "Output('vehicle-dropdown', 'options')"),
    ]

    all_good = True
    for check_name, check_string in checks:
        if check_string in content:
            print(f"   ✓ {check_name} found")
        else:
            print(f"   ✗ {check_name} NOT FOUND - THIS IS THE PROBLEM!")
            all_good = False

    print()

    # Check for the specific line that extracts vehicles
    if "'vehicle_number' in df.columns" in content:
        print("   ✓ Vehicle column check found")
    else:
        print("   ✗ Vehicle column check NOT FOUND or MODIFIED")
        all_good = False

    print()

    if all_good:
        print("✅ Dashboard code looks correct")
    else:
        print("❌ Dashboard code has issues - see above")

    print()
    return all_good

def find_callback_issue():
    """Find if there's a callback registration issue"""
    print("=" * 80)
    print("CHECKING CALLBACK REGISTRATION")
    print("=" * 80)
    print()

    app_file = '/home/tactical/racing-dashboard/src/dashboard/app.py'

    with open(app_file, 'r') as f:
        lines = f.readlines()

    # Find the callback that handles upload
    in_callback = False
    callback_start = 0
    outputs = []

    for i, line in enumerate(lines, 1):
        if '@app.callback' in line:
            in_callback = True
            callback_start = i
            outputs = []
        elif in_callback and 'Output(' in line:
            outputs.append(line.strip())
        elif in_callback and ('def ' in line or i - callback_start > 20):
            if 'upload' in ' '.join(outputs).lower():
                print(f"📍 Upload callback found at line {callback_start}")
                print(f"   Outputs ({len(outputs)}):")
                for out in outputs:
                    print(f"      {out}")
                print()

                # Check if vehicle-dropdown options is in outputs
                if any('vehicle-dropdown' in out and 'options' in out for out in outputs):
                    print("   ✓ vehicle-dropdown options output found")
                else:
                    print("   ✗ vehicle-dropdown options output MISSING!")
                    print("   THIS IS THE BUG!")

                break
            in_callback = False

    print()

if __name__ == '__main__':
    # Test with actual master file if available
    test_file = '/home/tactical/master_racing_data.csv'

    if not Path(test_file).exists():
        print(f"Test file not found: {test_file}")
        print("Please upload master_racing_data.csv to /home/tactical/")
        print()
        test_file = None

    # Check dashboard code structure
    code_ok = check_dashboard_code()

    # Check callback registration
    find_callback_issue()

    # Test file if available
    if test_file:
        test_ok = test_file_upload(test_file)

    print()
    print("=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
