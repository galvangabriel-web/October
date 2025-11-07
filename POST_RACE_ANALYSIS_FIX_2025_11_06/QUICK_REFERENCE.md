# Post-Race Analysis - Quick Reference Card

**⚡ Fast problem identification and resolution**

---

## 🚨 Common Error → Quick Fix

### Error: "147 vs 108 features"
```bash
cd data_analisys_car && venv/Scripts/python.exe deploy_post_race_fix.py
```

### Dashboard Down
```bash
cd data_analisys_car && venv/Scripts/python.exe ssh_helper.py "pkill -f dashboard && cd /home/tactical/racing_analytics && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &"
```

### No Valid Laps
Check lap times are 60-300 seconds, timestamps in ISO 8601 format

### Feature Extraction Failed
Verify CSV has 9 sensors in long format (see template)

---

## 📋 Essential Commands

### Check Dashboard Status
```bash
curl -s http://200.58.107.214:8050 | head -1
```
✓ Should return: `<!DOCTYPE html>`

### View Recent Logs
```bash
cd data_analisys_car
venv/Scripts/python.exe ssh_helper.py "tail -50 /home/tactical/racing_analytics/dashboard.log"
```

### Redeploy Fix
```bash
cd data_analisys_car
venv/Scripts/python.exe deploy_post_race_fix.py
```

### Restart Dashboard
```bash
cd data_analisys_car
venv/Scripts/python.exe ssh_helper.py "pkill -f dashboard && cd /home/tactical/racing_analytics && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &"
```

---

## 📊 CSV Requirements Checklist

```csv
✓ Header: timestamp,lap,vehicle_number,telemetry_name,telemetry_value,track,race,source_file
✓ Format: Long (one row per sensor reading)
✓ Sensors: 9 minimum (speed, pbrake_f, pbrake_r, ath, accx_can, accy_can, Steering_Angle, gear, nmot)
✓ Timestamps: ISO 8601 (2025-11-06T10:00:00.000Z)
✓ Lap times: 60-300 seconds
✓ Track names: lowercase-with-hyphens (circuit-of-the-americas)
```

---

## 🔍 Diagnostic Flowchart

```
Problem?
│
├─ Dashboard won't load
│  └─> Check: curl http://200.58.107.214:8050
│     ├─ No response → Restart dashboard
│     └─ Error 500 → Check logs
│
├─ Post-Race tab broken
│  └─> Upload test CSV (post_race_sample_template.csv)
│     ├─ "147 vs 108" → Redeploy fix
│     ├─ "No laps" → Check CSV format
│     └─ "Feature extraction" → Verify sensors
│
└─ Predictions wrong
   └─> Check data quality, sensor accuracy
```

---

## 🎯 Test Procedure (30 seconds)

1. **Navigate to:** http://200.58.107.214:8050
2. **Go to tab:** Post-Race Analysis (Tab 5/8)
3. **Upload:** `post_race_sample_template.csv`
4. **Click:** "Analyze Session"
5. **Expected:** ✓ Success, charts displayed

If fails → See [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)

---

## 📁 Critical File Locations

### Production Server
```
/home/tactical/racing_analytics/
├── src/models/inference/simple_post_race_predictor.py  ← Core fix
├── src/dashboard/post_race_widget.py                   ← Modified
├── data/models/lightgbm_baseline.pkl                   ← Model
└── dashboard.log                                       ← Errors here
```

### Local Repository
```
data_analisys_car/
├── POST_RACE_ANALYSIS_FIX_2025_11_06/  ← This folder
├── post_race_sample_template.csv        ← Test file
├── deploy_post_race_fix.py              ← Deploy script
└── test_simple_predictor.py             ← Test script
```

---

## ⚡ One-Liner Fixes

```bash
# Full redeploy
cd data_analisys_car && venv/Scripts/python.exe deploy_post_race_fix.py

# Restart only
cd data_analisys_car && venv/Scripts/python.exe ssh_helper.py "pkill -f dashboard && cd /home/tactical/racing_analytics && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &"

# Check logs
cd data_analisys_car && venv/Scripts/python.exe ssh_helper.py "tail -100 /home/tactical/racing_analytics/dashboard.log | grep ERROR"

# Test predictor
cd data_analisys_car && venv/Scripts/python.exe test_simple_predictor.py

# Verify files
cd data_analisys_car && venv/Scripts/python.exe ssh_helper.py "ls -la /home/tactical/racing_analytics/src/models/inference/simple_post_race_predictor.py"
```

---

## 🔧 Quick Verification

### Is Fix Deployed?
```bash
cd data_analisys_car
venv/Scripts/python.exe ssh_helper.py "grep -n SimplePostRacePredictor /home/tactical/racing_analytics/src/dashboard/post_race_widget.py"
```
✓ Should show line numbers (396, 397)

### Which Predictor is Running?
```bash
cd data_analisys_car
venv/Scripts/python.exe ssh_helper.py "grep 'Using.*Predictor' /home/tactical/racing_analytics/dashboard.log | tail -1"
```
✓ Should show: "Using SimplePostRacePredictor"

### Model File Exists?
```bash
cd data_analisys_car
venv/Scripts/python.exe ssh_helper.py "ls -lh /home/tactical/racing_analytics/data/models/lightgbm_baseline.pkl"
```
✓ Should show: ~42K file

---

## 📞 Support Priority

| Issue | Action | See |
|-------|--------|-----|
| Dashboard down | Restart immediately | Commands above |
| Error 147/108 | Redeploy fix | Deploy command above |
| CSV not working | Check format | CSV Requirements section |
| Other errors | Check logs | Troubleshooting Guide |

---

## 💾 Backup Commands

### Before Making Changes
```bash
cd data_analisys_car
venv/Scripts/python.exe ssh_helper.py "cd /home/tactical/racing_analytics && tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz src/dashboard/post_race_widget.py src/models/inference/"
```

### Restore from Backup
```bash
cd data_analisys_car
venv/Scripts/python.exe ssh_helper.py "cd /home/tactical/racing_analytics && tar -xzf backup_YYYYMMDD_HHMMSS.tar.gz"
```

---

## 🎯 Success Indicators

✅ Dashboard loads (HTTP 200)
✅ Post-Race tab accessible
✅ CSV upload works
✅ Predictions complete
✅ Charts display
✅ No "147 vs 108" error
✅ Logs show "SimplePostRacePredictor"

**All checked? You're good! 🎉**

---

**For detailed information:**
- Full troubleshooting: [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
- Technical solution: [SOLUTION_DOCUMENTATION.md](SOLUTION_DOCUMENTATION.md)
- Maintenance: [MAINTENANCE_GUIDE.md](MAINTENANCE_GUIDE.md)
