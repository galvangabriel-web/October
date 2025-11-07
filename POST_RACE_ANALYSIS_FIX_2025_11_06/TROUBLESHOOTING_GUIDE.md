# Post-Race Analysis - Comprehensive Troubleshooting Guide

**Version:** 2.0
**Last Updated:** 2025-11-06
**Status:** Active fix deployed

---

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Errors](#common-errors)
3. [System Health Checks](#system-health-checks)
4. [Recovery Procedures](#recovery-procedures)
5. [Prevention Strategies](#prevention-strategies)
6. [Advanced Debugging](#advanced-debugging)

---

## Quick Diagnostics

### Is the Dashboard Running?

```bash
# Method 1: Check HTTP response
curl -s http://200.58.107.214:8050 | head -1
# Expected: <!DOCTYPE html>

# Method 2: Check process count
cd data_analisys_car
venv/Scripts/python.exe ssh_helper.py "pgrep -f 'python.*dashboard' | wc -l"
# Expected: 1-4 (number of workers)

# Method 3: Check port
venv/Scripts/python.exe ssh_helper.py "netstat -tlnp | grep 8050"
# Expected: Shows process listening on port 8050
```

**If NOT running:**
```bash
cd data_analisys_car
venv/Scripts/python.exe deploy_post_race_fix.py
```

---

## Common Errors

### Error 1: "147 vs 108 Features" ❌

**Full Error:**
```
Error: The number of features in data (108) is not the same as it was in training data (147).
You can set `predict_disable_shape_check=true` to discard this error, but please be aware what you are doing.
```

**Cause:**
- Using old PostRacePredictor instead of SimplePostRacePredictor
- SimplePostRacePredictor not deployed or failed to load

**Diagnosis:**
```bash
# Check if SimplePostRacePredictor exists on server
cd data_analisys_car
venv/Scripts/python.exe ssh_helper.py "ls -la /home/tactical/racing_analytics/src/models/inference/simple_post_race_predictor.py"

# Check dashboard logs for which predictor is being used
venv/Scripts/python.exe ssh_helper.py "grep -E 'SimplePostRacePredictor|PostRacePredictor' /home/tactical/racing_analytics/dashboard.log | tail -10"
```

**Solution:**
```bash
# Redeploy the fix
cd data_analisys_car
venv/Scripts/python.exe deploy_post_race_fix.py

# Clear browser cache
# Press Ctrl+F5 in browser

# Verify deployment
venv/Scripts/python.exe ssh_helper.py "grep -n 'SimplePostRacePredictor' /home/tactical/racing_analytics/src/dashboard/post_race_widget.py"
# Should show line numbers where it's imported and used
```

---

### Error 2: "No Module Named 'lightgbm'" ❌

**Full Error:**
```
ModuleNotFoundError: No module named 'lightgbm'
```

**Cause:**
- Running with wrong Python interpreter (not using venv)
- LightGBM not installed in virtualenv

**Diagnosis:**
```bash
# Check which Python is being used
cd data_analisys_car
venv/Scripts/python.exe ssh_helper.py "which python"
# Expected: /home/tactical/racing_analytics/venv/bin/python

# Check if lightgbm is installed
venv/Scripts/python.exe ssh_helper.py "cd /home/tactical/racing_analytics && venv/bin/python -c 'import lightgbm; print(lightgbm.__version__)'"
# Expected: Version number (e.g., 3.3.2)
```

**Solution:**
```bash
# Install lightgbm on server
cd data_analisys_car
venv/Scripts/python.exe ssh_helper.py "cd /home/tactical/racing_analytics && venv/bin/pip install lightgbm"

# Restart dashboard
venv/Scripts/python.exe ssh_helper.py "pkill -f dashboard && cd /home/tactical/racing_analytics && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &"
```

---

### Error 3: "No Valid Laps Found" ❌

**Full Error:**
```
❌ No valid laps found. Check data format.
```

**Cause:**
- Lap times outside 60-300 second range
- Missing or incorrect timestamp format
- Incorrect lap numbering

**Diagnosis:**
```python
# Check your CSV file
import pandas as pd

df = pd.read_csv('your_file.csv')

# Check lap durations
for (vehicle, lap), group in df.groupby(['vehicle_number', 'lap']):
    timestamps = pd.to_datetime(group['timestamp'])
    duration = (timestamps.max() - timestamps.min()).total_seconds()
    print(f"Vehicle {vehicle}, Lap {lap}: {duration:.1f}s")

# Valid range: 60-300 seconds
```

**Solution:**
```bash
# Fix 1: Check timestamp format
# Timestamps must be ISO 8601: 2025-11-06T10:00:00.000Z

# Fix 2: Verify lap numbers are consecutive (1, 2, 3...)
cut -d',' -f2 your_file.csv | sort -u
# Should show: lap, 1, 2, 3...

# Fix 3: Check for enough data per lap
# Each lap needs ~600-2000 sensor readings (60-200s × 10 Hz × 9 sensors)
```

---

### Error 4: "Feature Extraction Failed" ❌

**Full Error:**
```
Feature extraction failed - no laps processed. Check telemetry data format.
```

**Cause:**
- Missing required sensors
- Wrong telemetry format (wide instead of long)
- Sensor names don't match expected values

**Diagnosis:**
```bash
# Check unique sensors in your CSV
cut -d',' -f4 your_file.csv | sort | uniq

# Expected 9 sensors:
# speed, pbrake_f, pbrake_r, ath, accx_can, accy_can, Steering_Angle, gear, nmot

# Check format (should be long, not wide)
head -20 your_file.csv
# Each timestamp should have 9 rows (one per sensor)
```

**Solution:**
```csv
# Correct format (long):
timestamp,lap,vehicle_number,telemetry_name,telemetry_value,track,race,source_file
2025-11-06T10:00:00.000Z,1,5,speed,145.2,circuit-of-the-americas,race_1,cota
2025-11-06T10:00:00.000Z,1,5,pbrake_f,12.5,circuit-of-the-americas,race_1,cota
2025-11-06T10:00:00.000Z,1,5,pbrake_r,8.3,circuit-of-the-americas,race_1,cota
...

# Wrong format (wide) - DO NOT USE:
timestamp,lap,vehicle_number,speed,pbrake_f,pbrake_r,ath...
2025-11-06T10:00:00.000Z,1,5,145.2,12.5,8.3,85.0...
```

---

### Error 5: "Dashboard Not Loading" ❌

**Symptoms:**
- Page shows "Connection refused"
- Page shows "Timeout"
- Blank white page

**Diagnosis:**
```bash
cd data_analisys_car

# Check 1: Is server accessible?
ping 200.58.107.214
# Should respond

# Check 2: Is dashboard process running?
venv/Scripts/python.exe ssh_helper.py "pgrep -f dashboard"
# Should show process IDs

# Check 3: Is port open?
telnet 200.58.107.214 8050
# Should connect

# Check 4: Check logs for crashes
venv/Scripts/python.exe ssh_helper.py "tail -100 /home/tactical/racing_analytics/dashboard.log | grep -E 'ERROR|Exception|Traceback'"
```

**Solution:**
```bash
# Restart dashboard
cd data_analisys_car
venv/Scripts/python.exe ssh_helper.py "pkill -f dashboard"
sleep 3
venv/Scripts/python.exe ssh_helper.py "cd /home/tactical/racing_analytics && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &"

# Wait 5 seconds then test
curl http://200.58.107.214:8050
```

---

## System Health Checks

### Daily Health Check

```bash
#!/bin/bash
# Save as: check_dashboard_health.sh

echo "=== Dashboard Health Check ==="
echo ""

# 1. Process Check
echo "[1] Process Status:"
ssh tactical@200.58.107.214 -p 5197 "pgrep -fa dashboard" && echo "✓ Running" || echo "✗ Not running"

# 2. HTTP Check
echo ""
echo "[2] HTTP Response:"
curl -s -o /dev/null -w "%{http_code}" http://200.58.107.214:8050
echo " (200 = OK)"

# 3. Recent Errors
echo ""
echo "[3] Recent Errors (last 24h):"
ssh tactical@200.58.107.214 -p 5197 "find /home/tactical/racing_analytics -name 'dashboard.log' -mtime -1 -exec grep -c ERROR {} \;"

# 4. Disk Space
echo ""
echo "[4] Disk Space:"
ssh tactical@200.58.107.214 -p 5197 "df -h /home/tactical/racing_analytics | tail -1"

echo ""
echo "=== End Health Check ==="
```

---

## Recovery Procedures

### Full Reset Procedure

If dashboard is completely broken:

```bash
cd data_analisys_car

# Step 1: Kill all dashboard processes
venv/Scripts/python.exe ssh_helper.py "pkill -9 -f dashboard"

# Step 2: Clear Python cache
venv/Scripts/python.exe ssh_helper.py "cd /home/tactical/racing_analytics && find src -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true"

# Step 3: Redeploy all fixes
venv/Scripts/python.exe deploy_post_race_fix.py

# Step 4: Wait for startup
sleep 10

# Step 5: Verify
curl http://200.58.107.214:8050
```

### Rollback to Previous Version

If new fix causes issues:

```bash
cd data_analisys_car

# Option 1: Use git to revert
venv/Scripts/python.exe ssh_helper.py "cd /home/tactical/racing_analytics && git checkout HEAD~1 src/dashboard/post_race_widget.py"

# Option 2: Remove SimplePostRacePredictor
venv/Scripts/python.exe ssh_helper.py "rm /home/tactical/racing_analytics/src/models/inference/simple_post_race_predictor.py"

# Option 3: Edit widget to use old predictor
venv/Scripts/python.exe ssh_helper.py "sed -i 's/SimplePostRacePredictor/PostRacePredictor/g' /home/tactical/racing_analytics/src/dashboard/post_race_widget.py"

# Then restart
venv/Scripts/python.exe ssh_helper.py "pkill -f dashboard && cd /home/tactical/racing_analytics && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &"
```

---

## Prevention Strategies

### Before Making Changes

1. **Always test locally first:**
```bash
cd data_analisys_car
venv/Scripts/python.exe test_simple_predictor.py
# Should pass before deploying
```

2. **Backup current dashboard:**
```bash
venv/Scripts/python.exe ssh_helper.py "cd /home/tactical/racing_analytics && cp -r src/dashboard src/dashboard.backup.$(date +%Y%m%d)"
```

3. **Check dashboard logs before deployment:**
```bash
venv/Scripts/python.exe ssh_helper.py "tail -50 /home/tactical/racing_analytics/dashboard.log"
# Verify no existing errors
```

### After Deployment

1. **Monitor logs for 5 minutes:**
```bash
# Watch for errors
venv/Scripts/python.exe ssh_helper.py "tail -f /home/tactical/racing_analytics/dashboard.log"
# Press Ctrl+C after 5 minutes if no errors
```

2. **Test with template CSV:**
```
Upload: post_race_sample_template.csv
Expected: Success without errors
```

3. **Check predictor being used:**
```bash
venv/Scripts/python.exe ssh_helper.py "grep 'Using.*Predictor' /home/tactical/racing_analytics/dashboard.log | tail -1"
# Should show: "Using SimplePostRacePredictor (basic features only)"
```

---

## Advanced Debugging

### Enable Debug Logging

Add to `src/dashboard/post_race_widget.py` (line 396):

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Redeploy and check logs.

### Test Predictor Directly

```python
# test_predictor_direct.py
import sys
sys.path.insert(0, '/home/tactical/racing_analytics')

from src.models.inference.simple_post_race_predictor import SimplePostRacePredictor
import pandas as pd

# Load test data
df = pd.read_csv('post_race_sample_template.csv')

# Initialize predictor
predictor = SimplePostRacePredictor()
print(f"Model loaded: {predictor.model_path}")

# Calculate lap times
lap_times = []
for (v, l), g in df.groupby(['vehicle_number', 'lap']):
    ts = pd.to_datetime(g['timestamp'])
    duration = (ts.max() - ts.min()).total_seconds()
    lap_times.append({'vehicle_number': v, 'lap_number': l, 'lap_time': duration,
                     'track': g['track'].iloc[0], 'race': g['race'].iloc[0]})

lap_times_df = pd.DataFrame(lap_times)

# Predict
results = predictor.predict_session(df, lap_times_df)
print(results)
```

### Check Model File

```bash
cd data_analisys_car
venv/Scripts/python.exe ssh_helper.py "python3 << 'PYEOF'
import joblib
import sys
sys.path.insert(0, '/home/tactical/racing_analytics')

model = joblib.load('/home/tactical/racing_analytics/data/models/lightgbm_baseline.pkl')
print(f'Model type: {type(model)}')
print(f'Expected features: {model.n_features_}')
print(f'Model params: {model.get_params()}')
PYEOF
"
```

---

## Error Decision Tree

```
Error Occurred
│
├─ "147 vs 108 features"
│  └─> Redeploy simple_post_race_predictor.py
│
├─ "No valid laps found"
│  └─> Check CSV format (lap times 60-300s)
│
├─ "Feature extraction failed"
│  └─> Verify 9 sensors present, long format
│
├─ "No module named X"
│  └─> Install module in venv on server
│
├─ "Connection refused"
│  └─> Restart dashboard
│
└─ Other
   └─> Check dashboard.log for details
```

---

## Contact Decision Matrix

| Issue Severity | Action | Timeline |
|----------------|--------|----------|
| **Critical** (Dashboard down) | Immediate redeploy | 5 min |
| **High** (Tab not working) | Follow troubleshooting guide | 30 min |
| **Medium** (Slow performance) | Monitor logs, check resources | 1 hour |
| **Low** (Cosmetic issues) | Log for future fix | Next sprint |

---

**For quick fixes, see [QUICK_REFERENCE.md](QUICK_REFERENCE.md)**
