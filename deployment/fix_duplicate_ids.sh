#!/bin/bash
################################################################################
# Fix Duplicate Upload Component IDs
# THIS IS THE BUG: Three components with id='upload-telemetry'
################################################################################

set -e

APP_FILE="/home/tactical/racing-dashboard/src/dashboard/app.py"
BACKUP_FILE="/home/tactical/racing-dashboard/src/dashboard/app.py.backup_duplicate_fix"

echo "=================================="
echo "Fixing Duplicate Component IDs"
echo "=================================="
echo ""

# Backup
echo "Creating backup..."
cp "$APP_FILE" "$BACKUP_FILE"
echo "Backup saved: $BACKUP_FILE"
echo ""

echo "The problem:"
echo "  - Line 144: Old legacy upload component"
echo "  - Line 237: Upload page component (correct)"
echo "  - Line 469: Footer upload component (correct)"
echo ""
echo "Solution: Comment out the old legacy upload section (lines 140-167)"
echo ""

# Comment out lines 140-167 (the old legacy upload section)
sed -i '140,167s/^/# /' "$APP_FILE"

echo "✓ Fixed: Old legacy upload section commented out"
echo ""

# Verify the fix
echo "Verifying fix..."
UPLOAD_COUNT=$(grep -c "id='upload-telemetry'" "$APP_FILE" || true)
echo "  Upload components after fix: $UPLOAD_COUNT (should be 2)"
echo ""

if [ "$UPLOAD_COUNT" -eq 2 ]; then
    echo "✅ FIX SUCCESSFUL!"
    echo ""
    echo "Next steps:"
    echo "  1. Restart dashboard: sudo systemctl restart racing-dashboard"
    echo "  2. Test upload at: http://200.58.107.214"
    echo "  3. Upload master_racing_data.csv"
    echo "  4. Vehicles should now appear in dropdown!"
else
    echo "⚠ Still $UPLOAD_COUNT components - manual fix needed"
fi

echo ""
echo "To revert if needed:"
echo "  cp $BACKUP_FILE $APP_FILE"
echo "  sudo systemctl restart racing-dashboard"
