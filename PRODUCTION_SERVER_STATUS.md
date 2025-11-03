# Production Server Status Report
**Date:** November 3, 2025
**Server:** 200.58.107.214:8050 (Dashboard), 200.58.107.214:8000 (API)

## ✅ COMPLETED

### 1. CLAUDE.md Documentation Update
- **Added** Production Server Directory Structure (lines 243-314)
  - Complete directory tree for `/home/tactical/racing_analytics`
  - Key file locations (app.py, api.py, logs, data files)
  - Process management information

- **Added** Common Linux Commands (lines 316-515)
  - 10 comprehensive categories of commands
  - Service status checks, log management, restart procedures
  - Disk management, Python package management, cache clearing
  - Troubleshooting guides, performance optimization
  - All commands use `ssh_helper.py` (no password prompts)

- **Committed** to git (commit 79e21f9)
  ```
  docs: Add production server directory structure and Linux commands to CLAUDE.md
  +274 lines
  ```

### 2. Issues Discovered

#### ⚠️ CRITICAL: Directory Path Change
- **OLD (deleted):** `/home/tactical/racing-dashboard/`
- **NEW (current):** `/home/tactical/racing_analytics/`

**Evidence:**
```
ls -l /proc/555/cwd
lrwxrwxrwx 1 tactical tactical 0 Nov  3 10:46 /proc/555/cwd -> /home/tactical/racing-dashboard (deleted)
```

The old directory was deleted/renamed, but process PID 555 (old API) was still running from the deleted path.

#### ⚠️ API ImportError
**Error from api.log:**
```
ImportError: cannot import name 'DriverProfiler' from 'src.insights.driver_profiler'
(/home/tactical/racing_analytics/src/insights/driver_profiler.py)
```

**Root Cause:**
- The `DriverProfiler` class import is failing
- Likely due to stale Python cache (`__pycache__`)
- Or missing/outdated src/insights files on production server

#### ⚠️ Dashboard Not Running
- No dashboard process found on production server
- Only API (PID 555) was running from deleted directory
- Dashboard needs to be started from correct path: `/home/tactical/racing_analytics/`

##  BLOCKED: SSH Authentication Failure

**All SSH attempts now failing with:**
```
Permission denied, please try again.
Connection closed by 200.58.107.214 port 5197
```

**This prevents:**
- Uploading src/insights files
- Clearing Python cache
- Restarting API/Dashboard
- Any remote diagnostics

## 📋 TODO: Required Actions

### 1. Fix SSH Authentication
**User must do this first:**
```bash
# Option A: Update .env file with correct password
# Edit .env and update SSH_PASSWORD

# Option B: Test SSH manually
ssh -p 5197 tactical@200.58.107.214
# If this works, update .env with the password you used
```

### 2. Upload Insights Files
**After fixing SSH, run:**
```bash
python fix_production_issues.py
```

**Or manually:**
```bash
# Upload all src/insights/*.py files
python deploy_insights.py
```

### 3. Clear Python Cache
```bash
python ssh_helper.py "cd /home/tactical/racing_analytics && find src -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true"
```

### 4. Stop Old Processes
```bash
# Kill any old API processes
python ssh_helper.py "pkill -f 'uvicorn.*src.api.main'"

# Kill any old dashboard processes
python ssh_helper.py "pkill -f 'dashboard/app.py'"
```

### 5. Start API
```bash
python ssh_helper.py "cd /home/tactical/racing_analytics && nohup venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 30 > api.log 2>&1 &"
```

### 6. Start Dashboard
```bash
python ssh_helper.py "cd /home/tactical/racing_analytics && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &"
```

### 7. Verify Services
```bash
# Check API is running
python ssh_helper.py "ps aux | grep '[u]vicorn.*src.api.main'"

# Check dashboard is running
python ssh_helper.py "ps aux | grep '[p]ython.*dashboard/app.py'"

# Check API logs for errors
python ssh_helper.py "cd /home/tactical/racing_analytics && tail -50 api.log"

# Check dashboard logs
python ssh_helper.py "cd /home/tactical/racing_analytics && tail -50 dashboard.log"
```

### 8. Test Production URLs
```
Dashboard: http://200.58.107.214:8050
API Docs: http://200.58.107.214:8000/docs
API Health: http://200.58.107.214:8000/
```

## 🔧 Files Created for Production Fixes

### fix_production_issues.py
**Purpose:** Comprehensive fix script that:
1. Uploads all src/insights/*.py files
2. Clears Python cache
3. Stops old processes
4. Starts new API and Dashboard
5. Verifies services are running

**Usage:**
```bash
python fix_production_issues.py
```

**Status:** ⚠️ Blocked by SSH authentication issues

### map_server_structure_fixed.py
**Purpose:** Maps production server directory structure

**Usage:**
```bash
python map_server_structure_fixed.py
```

**Output:** `server_structure.txt` with complete directory tree

## 📊 Current Server State (Last Known)

### Processes Running:
```
tactical   555  - Old API from deleted directory (needs to be killed)
tactical   581  - RMM API (unrelated)
[No dashboard process found]
```

### Directories:
```
/home/tactical/racing_analytics/        # Current (correct path)
/home/tactical/racing-dashboard/        # Deleted (old path)
```

### Python Environment:
```
/home/tactical/racing_analytics/venv/bin/python  -> /usr/bin/python3 (Python 3.11.2)
```

### Log Files:
```
/home/tactical/racing_analytics/api.log
/home/tactical/racing_analytics/dashboard.log
```

## 🎯 Quick Fix Once SSH Works

**Single command to fix everything:**
```bash
python fix_production_issues.py
```

This will automatically:
- Upload insights files
- Clear cache
- Restart services
- Verify everything is running

## 📝 Notes

1. **PyWavelets:** Already installed previously (1.9.0)
2. **master_racing_data.csv:** Should be at `/home/tactical/racing_analytics/master_racing_data.csv` (71,000 samples)
3. **Auto-load feature:** Dashboard will auto-load data on Linux in production mode
4. **Port forwarding:** If ports 8050/8000 are blocked externally, may need to configure firewall/nginx

## 🆘 If Still Having Issues

### Check if server is accessible:
```bash
ping 200.58.107.214
telnet 200.58.107.214 5197
```

### Check if ports are open:
```bash
telnet 200.58.107.214 8050  # Dashboard
telnet 200.58.107.214 8000  # API
```

### Contact server administrator if:
- SSH port 5197 is blocked
- Firewall is blocking ports 8050/8000
- SSH password has changed and .env needs updating
- Server has been rebooted and processes didn't auto-start
