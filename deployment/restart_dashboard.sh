#!/bin/bash
################################################################################
# Restart Racing Dashboard with Updated Code
################################################################################

set -e

DASHBOARD_DIR="/home/tactical/racing-dashboard"

echo "========================================"
echo "Restarting Racing Dashboard"
echo "========================================"
echo ""

cd "$DASHBOARD_DIR"

echo "1. Stopping old dashboard process..."
pkill -f "src/dashboard/app.py" || echo "   (no existing process found)"
sleep 2

echo ""
echo "2. Starting new dashboard..."
nohup venv/bin/python src/dashboard/app.py > /tmp/dashboard.log 2>&1 &

# Give it time to start
sleep 5

echo ""
echo "3. Checking dashboard status..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8050/ || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ Dashboard is running! (HTTP $HTTP_CODE)"
else
    echo "   ⚠ Dashboard may not be ready yet (HTTP $HTTP_CODE)"
    echo "   Check logs: tail -20 /tmp/dashboard.log"
fi

echo ""
echo "========================================"
echo "✅ Dashboard Restart Complete!"
echo "========================================"
echo ""
echo "Dashboard URL: http://200.58.107.214"
echo ""
echo "To view logs:"
echo "  tail -f /tmp/dashboard.log"
echo ""
echo "To stop dashboard:"
echo "  pkill -f 'src/dashboard/app.py'"
