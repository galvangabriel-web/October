# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**GR Cup Racing Analytics Platform** - A production-deployed machine learning system for professional racing telemetry analysis.

- **Domain:** Racing telemetry analysis for Toyota GR Cup competition
- **Model:** Sequential LightGBM achieving 97.49% R² accuracy for lap time prediction
- **Data Scale:** 18.5GB telemetry data across 6 tracks, 4,881 laps, 20 vehicles, 12 sensors @ 10Hz
- **Production Status:** Live at http://200.58.107.214:8050 (Linux server)

---

## Quick Start Commands

### Essential Development Commands

```bash
# Module execution (CRITICAL - always use -m flag)
python -m src.models.baseline.train_lightgbm    # ✅ Correct
python src/models/baseline/train_lightgbm.py    # ❌ Import errors

# Start dashboard locally
python src/dashboard/app.py                     # http://localhost:8050

# Start API locally (required for Tabs 1 & 3)
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Deploy to production
python quick_upload_app.py                      # Deploy dashboard changes
python deploy_insights.py                       # Deploy src/insights changes

# SSH to production (uses .env credentials automatically)
python ssh_helper.py "command"                  # ✅ Reads .env
ssh tactical@200.58.107.214                     # ❌ Asks for password

# Run tests
pytest tests/ -v
pytest tests/insights/ --cov=src.insights
```

### Environment Setup

```bash
# Windows (primary development platform)
setup.bat
venv\Scripts\activate

# Linux/Mac
./setup.sh
source venv/bin/activate

# Verify installation
python --version  # Should be 3.10+
python -c "import pandas, lightgbm, dash; print('OK')"
```

---

## Architecture Overview

### System Architecture

```
Raw Telemetry Data (18.5GB in organized_data/)
           ↓
    data_loader.py (RacingDataLoader - chunked loading)
           ↓
    Feature Engineering Pipeline
    ├─ TelemetryFeatureEngineer (100+ features)
    └─ AdvancedFeatureEngineer (FFT, wavelets)
           ↓
    Sequential LightGBM Model (97.49% R²)
           ↓
    ┌──────────────────┬──────────────────┐
    │  FastAPI (8000)  │  Dash (8050)     │
    │  Backend API     │  10-Tab Dashboard│
    └──────────────────┴──────────────────┘
```

### Key Module Structure

```
src/
├── api/                     # FastAPI backend
│   └── main.py              # REST API endpoints
├── dashboard/               # 10-tab Dash frontend
│   ├── app.py               # Main dashboard application
│   ├── tour/                # Welcome tour system
│   └── *_widget.py          # Individual tab widgets
├── data_processing/
│   ├── feature_engineering.py          # Core feature extraction
│   └── advanced_feature_engineering.py # FFT/wavelet features
├── insights/                # Production-grade analysis (strict types, 198 tests)
│   ├── driver_profiler.py
│   ├── corner_analyzer.py
│   ├── consistency_tracker.py
│   └── config.py            # InsightsConfig
├── models/baseline/
│   ├── train_lightgbm.py              # Main model training
│   └── optimize_hyperparameters.py    # Hyperparameter tuning
└── services/
    └── telemetry_analyzer.py # Business logic (pattern detection, cube analysis)

data_loader.py               # ROOT LEVEL - Data access layer (use for ALL data loading)
```

---

## Critical Patterns & Rules

### 1. Module Execution Pattern (CRITICAL - 90% of import errors)

**Always use `-m` flag when running Python modules:**

```bash
# ✅ CORRECT - Adds project root to sys.path
python -m src.models.baseline.train_lightgbm --tracks all

# ❌ WRONG - Causes ModuleNotFoundError
python src/models/baseline/train_lightgbm.py
```

**Why:** The `-m` flag ensures project root is in `sys.path`, enabling imports like `from data_loader import RacingDataLoader`.

### 2. Long-Format Telemetry Pattern (CRITICAL)

Telemetry data has **one row per sensor reading** (not one row per timestamp). Each timestamp has ~12 rows with different `telemetry_name` values.

```python
# ✅ CORRECT: Filter by sensor FIRST, then analyze
speed_data = df[df['telemetry_name'] == 'speed']
avg_speed = speed_data['telemetry_value'].mean()  # All values are km/h

# ❌ WRONG: Meaningless - mixes km/h, bar, %, degrees, etc.
avg_all = df['telemetry_value'].mean()
```

**Available sensors:** speed, pbrake_f, pbrake_r, aps, accx_can, accy_can, Steering_Angle, gear, nmot, gps_lat, gps_long, gps_alt

### 3. Chunked Memory Management

Telemetry files are 50MB+ each and split into 100k-row chunks to prevent memory issues.

```python
from data_loader import RacingDataLoader

loader = RacingDataLoader()

# ✅ RECOMMENDED: Fast prototyping (single chunk, ~5MB)
df = loader.load_single_chunk('barber-motorsports-park', 'race_unknown', 'telemetry', chunk_num=1)

# ⚠️ Use with caution: Full dataset (requires 16GB+ RAM)
df_full = loader.load_data('barber-motorsports-park', 'race_unknown', 'telemetry', combine_chunks=True)
```

**Always start with `load_single_chunk()` for prototyping!**

### 4. Track Name Format

Use lowercase-with-hyphens matching directory structure:
- ✅ `'barber-motorsports-park'`, `'circuit-of-the-americas'`
- ❌ `'Barber Motorsports Park'`, `'COTA'`

**Available tracks:** barber-motorsports-park, circuit-of-the-americas, road-america, sebring, sonoma, virginia-international-raceway

### 5. Production vs. Development Environments

| Environment | URL | Platform | Data Loading | Issue Tracking |
|-------------|-----|----------|--------------|----------------|
| **Production** | http://200.58.107.214:8050 | Linux (Ubuntu) | Auto-loads master_racing_data.csv | **YES - All user issues are here** ✅ |
| **Development** | http://localhost:8050 | Windows 11 | Manual upload required | NO ❌ |

**When users report dashboard issues, they mean PRODUCTION.**

Fix workflow: Test locally → Deploy via scripts → Verify on production

### 6. Post-Race Analysis Tab (Tab 5)

**Current Implementation (Nov 2025):**
- Uses `SimplePostRacePredictor` with baseline model (40 features) for reliability
- Multi-track support: COTA, Road America, Sonoma, VIR
- Radio button UI for track selection (vertical layout)
- Template CSV files in `post_race_templates/` directory
- Works with minimal 9-sensor telemetry (no GPS required)

**Key Files:**
- `src/models/inference/simple_post_race_predictor.py` - Baseline predictor (RECOMMENDED)
- `src/models/inference/post_race_predictor.py` - Advanced predictor (147 features, requires full telemetry)
- `src/dashboard/post_race_widget.py` - Dashboard widget
- `post_race_sample_template.csv` - Working test data example

**Documentation:**
- `POST_RACE_ANALYSIS_FIX_2025_11_06/` - Complete fix documentation package
- See `POST_RACE_ANALYSIS_FIX_2025_11_06/INDEX.md` for navigation

---

## Production Deployment

### Server Configuration

- **Host:** 200.58.107.214
- **SSH Port:** 5197
- **User:** tactical
- **Base Path:** /home/tactical/racing_analytics
- **Python:** 3.11.2 (venv)
- **Dashboard Port:** 8050
- **API Port:** 8000
- **Credentials:** Stored in `.env` file (auto-read by scripts)

### Deployment Commands

```bash
# Deploy dashboard changes
python quick_upload_app.py

# Deploy insights module
python deploy_insights.py

# Run any SSH command (uses .env credentials via paramiko)
python ssh_helper.py "ps aux | grep dashboard"
python ssh_helper.py "tail -30 /home/tactical/racing_analytics/dashboard.log"

# Restart services
python ssh_helper.py "pkill -f dashboard && cd /home/tactical/racing_analytics && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &"

# Clear Python cache (CRITICAL after deployment)
python ssh_helper.py "cd /home/tactical/racing_analytics && find src -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true"
```

**NEVER use raw `ssh` or `scp` commands - they will prompt for passwords. Always use `ssh_helper.py` or deployment scripts which automatically read `.env` credentials using `paramiko` (cross-platform Python SSH library).**

### CRITICAL: Deployment Verification Protocol

**⚠️ LESSON LEARNED (2025-11-07): Never claim deployment success without verification**

After ANY production deployment, you MUST verify changes are actually visible:

```bash
# 1. Verify file was uploaded and has correct size/timestamp
python ssh_helper.py "ls -lh /home/tactical/racing_analytics/src/dashboard/post_race_widget.py"

# 2. Verify dashboard process is running
python ssh_helper.py "ps aux | grep 'python.*src/dashboard/app.py' | grep -v grep"

# 3. Verify port is listening
python ssh_helper.py "netstat -tulpn 2>/dev/null | grep 8050"

# 4. Check for errors in logs
python ssh_helper.py "tail -50 /home/tactical/racing_analytics/dashboard.log | grep -i 'error\|exception\|traceback'"

# 5. MOST IMPORTANT: Ask user to verify changes in browser
# "Please refresh the page (Ctrl+F5) and confirm the changes are visible"
```

**Why This Matters:**
- `quick_upload_app.py` doesn't always restart the dashboard correctly
- Browser cache can show old versions even after deployment
- Python import cache can load old code even after file updates
- **Trust but verify** - the user's eyes are the final test

**What Went Wrong (Nov 7, 2025):**
- I claimed radio button UI was deployed and enhanced
- User verified it was unchanged (still inline, no styling)
- Root cause: `post_race_widget.py` was never actually uploaded
- I reported success based on script output, not actual verification
- **This undermines trust and wastes the user's time**

**Correct Workflow:**
1. Make changes locally
2. Test locally (actually run dashboard and check browser)
3. Deploy to production using proper script
4. Verify files uploaded (check size/timestamp)
5. Verify dashboard restarted
6. **Ask user to verify changes in browser**
7. Only then claim success

**Golden Rule:** Never say "deployed successfully" or "changes are live" unless you've confirmed the changes are actually visible. When in doubt, say "Changes uploaded, please verify in your browser."

---

## Code Standards

### Production Code (src/insights/)

The `src/insights/` module is production-grade with strict standards:
- ✅ Full type annotations (mypy strict mode)
- ✅ 198 tests with >90% coverage
- ✅ All public methods documented
- ✅ Pydantic models only (no raw dicts)
- ✅ Structured exceptions
- ❌ No `type: ignore` comments

### Other Modules

`src/dashboard/`, `src/api/`, and root-level scripts are more lenient - prioritize functionality.

### Code Patterns

```python
# Configuration-driven (no magic numbers)
from src.insights import InsightsConfig

config = InsightsConfig(
    hard_brake_threshold=110.0,      # bar
    full_throttle_threshold=95.0,    # %
    high_lateral_g_threshold=1.2,    # g
    log_level='INFO'
)

# Driver analysis
from src.insights import DriverProfiler

profiler = DriverProfiler(config=config)
profile = profiler.analyze_driver_performance(telemetry, vehicle_number=5)

# Feature engineering
from src.data_processing.feature_engineering import TelemetryFeatureEngineer

engineer = TelemetryFeatureEngineer()
features = engineer.extract_features(telemetry_df)  # Returns 100+ features
```

---

## Domain-Specific Rules

### Never Remove Statistical Outliers

In racing data, extreme values are **features, not errors**:
- High brake pressure (13% "outliers") = hard braking zones
- High lateral g (10.7% "outliers") = fast corners at grip limits
- Aggressive steering = quick corrections

**Statistical outlier removal degrades model performance.** Use domain knowledge validation instead.

### Data Frequency and Sensors

- **Frequency:** 10 Hz (10 samples per second)
- **Sensors:** 12 channels per vehicle
- **Typical lap:** 60-200 seconds = 600-2000 data points per sensor per lap
- **Units:** Speed (km/h), Brake (bar), Throttle (%), G-forces (g), Steering (deg), RPM (rpm), GPS (deg/m)

---

## Common Pitfalls

### Top 5 Critical Mistakes

1. **Running scripts without `-m` flag** → Import errors
   - Fix: Always use `python -m src.module.script`

2. **Analyzing telemetry without filtering by sensor** → Meaningless results
   - Fix: `df[df['telemetry_name'] == 'speed']` BEFORE operations

3. **Loading full telemetry without chunking first** → Memory crashes
   - Fix: Start with `load_single_chunk()` for prototyping

4. **Using raw SSH commands** → Password prompts
   - Fix: Use `python ssh_helper.py "command"`

5. **Testing dashboard without API running** → Tab 1 & 3 fail
   - Fix: Run API in separate terminal: `python -m uvicorn src.api.main:app --port 8000 --reload`

---

## Dashboard Structure

### 10-Tab Dashboard

1. **Tab 1: Enhanced Driver Insights** - Performance metrics, strengths/weaknesses
2. **Tab 2: Telemetry Comparison** - Multi-lap synchronized comparison
3. **Tab 3: Model Predictions** - Feature importance, pattern analysis, corner insights
4. **Tab 4: Coaching Insights** - Actionable improvement recommendations
5. **Tab 5: Post-Race Analysis** - Comprehensive race reports with PDF export
6. **Tab 6: Track Animation** - GPS-based lap visualization
7. **Tab 7: Track Maps** - Interactive track layouts with sector markers
8. **Tab 8: Weather Analysis** - Weather impact on performance
9. **Tab 9: Sector Benchmarking** - Sector-by-sector driver comparison
10. **Tab 10: Championships** - Championship standings and rankings

**Tabs 1-5:** Require telemetry upload or auto-loaded production data
**Tabs 6:** Requires GPS telemetry
**Tabs 7-10:** Static or use Week 1 data

### Tour System

**Phase 1 (COMPLETE - v3.1.0):**
- Welcome modal with data confirmation (71,000 samples, 5 vehicles, 10 laps)
- Quick stats display and feature highlights
- "Don't show again" preference tracking
- Start Tour / Skip Tour buttons
- Files: `src/dashboard/tour/welcome_modal.py`, `src/dashboard/assets/tour.css`

**Phase 2 (PLANNED):**
- Interactive 25-step guided tour overlay
- See: `DASHBOARD_TOUR_IMPLEMENTATION_GUIDE.md`

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src.insights --cov-report=html

# Test specific module
pytest tests/insights/test_driver_profiler.py -v

# Type checking (strict for src/insights/)
mypy src/insights/

# Full quality check
pytest tests/ --cov=src.insights && mypy src/insights/
```

---

## Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'data_loader'`
**Solution:** Use `-m` flag: `python -m src.models.baseline.train_lightgbm`

**Error:** `No module named 'src'`
**Solution:** Run from project root (`C:\project\data_analisys_car` or `/home/tactical/racing_analytics`)

### Memory Issues

**Error:** MemoryError or system slowdown
**Solution:** Use `load_single_chunk()` instead of `combine_chunks=True`

### Production Dashboard Not Working

```bash
# 1. Check if running
python ssh_helper.py "ps aux | grep dashboard | grep -v grep"

# 2. Check logs
python ssh_helper.py "tail -50 /home/tactical/racing_analytics/dashboard.log"

# 3. Restart service
python ssh_helper.py "pkill -f dashboard && cd /home/tactical/racing_analytics && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &"
```

### Changes Not Appearing After Deployment

1. Clear browser cache: Ctrl+F5
2. Verify upload: `python ssh_helper.py "ls -la /home/tactical/racing_analytics/src/dashboard/app.py"`
3. Clear Python cache: See command in Production Deployment section above
4. Restart service

---

## Model Information

### Sequential LightGBM Model

- **Accuracy:** 97.49% R² (top 1% of ML models)
- **Error:** ±1.73 seconds MAE on ~120s laps (1.44% relative error)
- **Features:** 147 total (45 basic + 89 advanced + 13 sequential)
- **Training Time:** <5 minutes (CPU-only)
- **Tracks:** COTA, Road America, Sonoma, Virginia International Raceway (Barber & Sebring data available)

### Key Sequential Features

The breakthrough from 91.57% to 97.49% R² came from adding 13 sequential features:
- **Lag features:** Previous lap times (t-1, t-2, t-3)
- **Rolling statistics:** 3-lap and 5-lap moving averages
- **Context features:** Gap to best lap, best lap so far, consistency
- **Cumulative features:** Laps in stint, fuel burned

These capture lap-to-lap dependencies like tire warm-up, fuel burn, and driver rhythm.

---

## Git Workflow

```bash
# Check status
git status
git log --oneline -10

# Commit conventions
git commit -m "feat: New feature description"
git commit -m "fix: Bug fix description"
git commit -m "docs: Documentation update"

# NEVER commit (see .gitignore):
# - organized_data/ (18.5GB telemetry)
# - venv/, myenv/
# - data/models/
# - __pycache__/, *.pyc
# - *.csv, *.parquet, *.pkl (except small test files)
```

---

## Performance Specifications

- **Memory:** 8GB minimum, 16GB recommended
- **CPU:** All code is CPU-only (no GPU required)
- **Training:** <5 minutes for full model
- **API Response:** <200ms for predictions, <1s for features
- **Dashboard:** Auto-loads 71,000 samples in production
- **Chunk Size:** 100k rows (configurable)

---

## Key Documentation Files

- **`README.md`** - Project overview, model performance, deployment status
- **`CLAUDE.md`** (this file) - Development guidance and architecture
- **`QUICK_RESUME_CARD.txt`** - Current sprint status and immediate next steps (CHECK THIS FIRST when resuming work)
- **`.env`** - SSH credentials and configuration (never commit)
- **`requirements.txt`** - Python dependencies

---

## Recent Updates & Changes

**November 2025 - Tour System & Auto-Load**
- ✅ **Tour System Phase 1:** Welcome modal with data confirmation (v3.1.0-tour-system-mvp)
- ✅ **Auto-Load Feature:** Production mode auto-loads 71,000 samples on startup
- ✅ **Post-Race Analysis:** Multi-track upgrade with radio buttons and SimplePostRacePredictor
- ⚠️ **Deployment Lesson:** Always verify changes are visible in browser before claiming success

**Check QUICK_RESUME_CARD.txt first when resuming work** - it contains current sprint status and immediate next steps.

## Version

CLAUDE.md Version: 2.3
Last Updated: 2025-11-07
Dashboard Version: 3.1.0-tour-system-mvp
Production Status: Live at http://200.58.107.214:8050
