#!/usr/bin/env python3
"""
Fix Dashboard Upload Issues
============================

This script patches the dashboard upload handler to:
1. Handle multiple column name variations
2. Provide better error messages
3. Log upload attempts for debugging

Run on server: python fix_dashboard_upload.py
"""

import re

APP_FILE = '/home/tactical/racing-dashboard/src/dashboard/app.py'
BACKUP_FILE = '/home/tactical/racing-dashboard/src/dashboard/app.py.backup'

# The improved upload handling code
UPLOAD_FIX = '''
def handle_upload(contents, filename):
    """Handle telemetry file upload with improved column detection"""
    if contents is None:
        return ("", "", [], "0", "0", "0", "--", True, False)  # Stay on upload page

    try:
        # Decode uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        # Check file size (limit to 500MB for web upload - server has 3GB RAM!)
        file_size_mb = len(decoded) / (1024 * 1024)
        if file_size_mb > 50:  # Reduced from 500MB to 50MB for 3GB RAM server
            error_msg = dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                html.Strong("File too large! "),
                f"({file_size_mb:.1f}MB) ",
                html.Br(),
                f"Maximum file size for this server is 50MB (3GB RAM limitation). ",
                html.Br(),
                html.Small("For larger files, upgrade server RAM to 8GB+", className="d-block mt-2"),
            ], color="danger", className="mb-0")
            return (error_msg, "", [], "0", "0", "0", "--", True, False)

        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        # Log columns for debugging
        logger.info(f"Uploaded file '{filename}' columns: {df.columns.tolist()}")
        logger.info(f"File size: {file_size_mb:.2f}MB, Rows: {len(df)}")

        # Store data as JSON
        data_json = df.to_json(date_format='iso', orient='split')

        # Get statistics
        num_samples = len(df)

        # IMPROVED: Flexible vehicle column detection
        def find_vehicle_column(df):
            """Find vehicle column with flexible naming"""
            possible_names = [
                'vehicle_number', 'Vehicle', 'VehicleNumber', 'vehicle',
                'Car', 'CarNumber', 'car_number', 'car', 'VEHICLE_NUMBER',
                'VEHICLE', 'CAR', 'car_id', 'vehicle_id'
            ]
            for col in possible_names:
                if col in df.columns:
                    return col
            return None

        vehicle_col = find_vehicle_column(df)

        if vehicle_col:
            vehicles = sorted(df[vehicle_col].unique())
            num_vehicles = len(vehicles)
            logger.info(f"Found {num_vehicles} vehicles in column '{vehicle_col}': {vehicles}")
        else:
            # No vehicle column found - show helpful error
            error_msg = dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                html.Strong("No vehicle column found!"),
                html.Br(),
                html.Br(),
                f"Columns in your file: {', '.join(df.columns.tolist())}",
                html.Br(),
                html.Br(),
                "Expected one of: vehicle_number, Vehicle, VehicleNumber, Car, CarNumber",
                html.Br(),
                html.Br(),
                html.Small("Please add a vehicle identifier column to your CSV.", className="text-muted")
            ], color="warning", className="mb-0")
            return (error_msg, "", [], f"{num_samples:,}", "0", "0", "--", True, False)

        # Get lap count (flexible column naming)
        lap_col_names = ['lap', 'Lap', 'LAP', 'lap_number', 'LapNumber']
        num_laps = 0
        for lap_col in lap_col_names:
            if lap_col in df.columns:
                num_laps = df[lap_col].nunique()
                break

        # Calculate average lap time if available
        avg_time = "--"

        # Create vehicle dropdown options
        vehicle_options = [{'label': f'Vehicle #{v}', 'value': v} for v in vehicles]

        # Success message
        status_msg = dbc.Alert([
            html.I(className="fas fa-check-circle me-2"),
            f"Successfully loaded {filename}",
            html.Br(),
            html.Small(f"{num_samples:,} samples, {num_vehicles} vehicles", className="text-muted")
        ], color="success", className="mb-0")

        logger.info(f"Upload successful: {filename}, {num_samples} samples, {num_vehicles} vehicles")

        return (
            status_msg,
            data_json,
            vehicle_options,
            f"{num_samples:,}",
            str(num_vehicles),
            str(num_laps),
            avg_time,
            False,  # Enable analyze button
            True    # Switch to dashboard page
        )

    except Exception as e:
        logger.error(f"Upload error for {filename}: {str(e)}", exc_info=True)
        error_msg = dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Error loading file: {str(e)}",
            html.Br(),
            html.Br(),
            html.Small("Check dashboard logs for details: sudo journalctl -u racing-dashboard -n 50", className="text-muted")
        ], color="danger", className="mb-0")
        return (error_msg, "", [], "0", "0", "0", "--", True, False)
'''

def apply_fix():
    """Apply the upload fix to app.py"""
    print("Reading app.py...")
    with open(APP_FILE, 'r') as f:
        content = f.read()

    # Backup original
    print(f"Creating backup at {BACKUP_FILE}...")
    with open(BACKUP_FILE, 'w') as f:
        f.write(content)

    # Find and replace the handle_upload function
    pattern = r'def handle_upload\(contents, filename\):.*?(?=\n@app\.callback|\nif __name__|$)'

    if not re.search(pattern, content, re.DOTALL):
        print("ERROR: Could not find handle_upload function!")
        return False

    # Replace the function
    new_content = re.sub(pattern, UPLOAD_FIX.strip(), content, flags=re.DOTALL)

    # Write back
    print("Writing patched app.py...")
    with open(APP_FILE, 'w') as f:
        f.write(new_content)

    print("✓ Patch applied successfully!")
    print("")
    print("Next steps:")
    print("  1. Restart dashboard: sudo systemctl restart racing-dashboard")
    print("  2. Test upload at: http://200.58.107.214")
    print("  3. Check logs if issues: sudo journalctl -u racing-dashboard -f")
    print("")
    print(f"Backup saved at: {BACKUP_FILE}")
    return True

if __name__ == '__main__':
    apply_fix()
