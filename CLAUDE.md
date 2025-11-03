# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 CRITICAL: PRODUCTION SERVER CONTEXT

**THE DASHBOARD IS DEPLOYED ON PRODUCTION SERVER: http://200.58.107.214:8050/**

**IMPORTANT:** When the user reports issues, they are referring to the **PRODUCTION SERVER (Linux)**, NOT local development (Windows)!

### Production vs Development Environments

| Aspect | Production (Linux Server) | Development (Windows Local) |
|--------|--------------------------|---------------------------|
| **URL** | http://200.58.107.214:8050 | http://localhost:8050 |
| **Platform** | Linux (Ubuntu) | Windows 11 |
| **Data Loading** | Auto-loads master_racing_data.csv | Manual upload required |
| **Path** | /home/tactical/racing_analytics | C:\project\data_analisys_car |
| **Python** | System Python 3.11.2 | venv Python |
| **Issues Reported** | **YES - User's issues are HERE** | NO - Only for local testing |

### To Fix Production Issues:

1. **Test locally first** (if possible to replicate)
2. **Deploy changes via SSH automation:**
   ```bash
   # Quick fix for single file
   python quick_upload_app.py

   # Full deployment
   python deployment/deploy.py
   ```
3. **Verify on production:** http://200.58.107.214:8050
4. **Never assume local fixes work on production** - environments differ!

## ⚠️ CRITICAL TESTING REQUIREMENTS

**BEFORE claiming ANY feature works, you MUST:**

1. **Actually test the feature** - Don't assume CSS/code changes work without verification
2. **Run the application** - Start the dashboard/API and verify in browser
3. **Verify the specific behavior** - Check that the claimed functionality actually works as described
4. **Document test results** - Report what you see, not what you expect to see
5. **If you cannot test visually** - Explicitly state "I cannot verify this visually" and ask user to test

**NEVER say something "works" or "is ready" unless you have:**
- ✅ Started the application
- ✅ Opened it in a browser or tested the specific feature
- ✅ Confirmed the behavior matches the requirement
- ✅ OR explicitly asked the user to test and confirm

**Violations of this rule waste user time and break trust.**

## 🧭 Quick Navigation (Start Here!)

**First time in this codebase?**
→ Read [Project Overview](#project-overview) + [Day 1 Quick Start](#day-1-quick-start)

**Resuming existing work?**
→ **Check `QUICK_RESUME_CARD.txt` FIRST** (current Sprint status, next steps)

**Need to understand architecture?**
→ Read [Critical Architecture Patterns](#critical-architecture-patterns) + [High-Level Architecture](#high-level-architecture)

**Ready to code?**
→ Jump to [Common Commands](#common-commands) + [Code Patterns Quick Reference](#code-patterns-quick-reference)

**Hitting errors?**
→ Check [Common Pitfalls](#common-pitfalls-to-avoid) + [Troubleshooting](#troubleshooting)

## Project Overview

Racing telemetry analysis for GR Cup competition. Processes 18.5GB of professional racing data from 6 tracks: telemetry (12 sensors @ high frequency), lap times, weather, and race results (4,881 laps, 20 vehicles).

**Tech Stack:** pandas/polars (data), LightGBM/XGBoost/CatBoost (ML), FastAPI (backend), Dash/Plotly (frontend), pytest (198 tests, >90% coverage in src/insights/), mypy (type checking).

**Current Development:** **Always check `QUICK_RESUME_CARD.txt` FIRST** before starting any work to see current sprint status and what tasks are in progress.

**Model Performance:** R² = 97.49% on lap time predictions (Sequential LightGBM with temporal features)

**Production Status:** Fully deployed on Linux server (200.58.107.214:8050) with auto-load, welcome tour, and 71,000 telemetry samples
**⚠️ User Issues:** All reported issues relate to the PRODUCTION SERVER, not local development!

## Quick Reference

```bash
# SSH to production server - NEVER use raw SSH, use automation!
python ssh_helper.py "command"                  # ✅ Reads .env, no password prompt
ssh tactical@200.58.107.214                     # ❌ Will ask for password!

# Deploy to production
python quick_upload_app.py                      # ✅ Deploy app.py changes
python deploy_insights.py                       # ✅ Deploy src/insights changes

# Most common mistake - ALWAYS use -m flag for src/ modules
python -m src.models.baseline.train_lightgbm    # ✅ Correct
python src/models/baseline/train_lightgbm.py    # ❌ Wrong

# Start dashboard (local development)
python src/dashboard/app.py                     # http://localhost:8050

# Load data correctly (filter by sensor BEFORE analyzing)
speed_data = df[df['telemetry_name'] == 'speed']
avg_speed = speed_data['telemetry_value'].mean()

# Run tests
pytest tests/ -v
```

## 🔐 SSH Automation & Production Deployment

**CRITICAL CONTEXT:** The dashboard is LIVE on production (http://200.58.107.214:8050). All user-reported issues are about THIS SERVER!

### ⚠️ CRITICAL: SSH Password Policy

**NEVER ask the user for SSH passwords! ALL credentials are in `.env` file.**

**❌ DO NOT USE these commands (they will prompt for password):**
```bash
ssh tactical@200.58.107.214          # ❌ Will ask for password
scp file.py tactical@...             # ❌ Will ask for password
git push/pull via SSH                # ❌ Will ask for password
```

**✅ ALWAYS USE Python automation scripts instead:**
```bash
# For running SSH commands
python ssh_helper.py "ls -la"                    # ✅ Uses .env credentials
venv/Scripts/python.exe ssh_helper.py "command"  # ✅ Full path version

# For deploying files
python quick_upload_app.py                       # ✅ Deploy single file
python deploy_insights.py                        # ✅ Deploy src/insights
```

**Why Git Bash asks for password:**
- Git for Windows uses its own SSH client, which doesn't read `.env`
- The `.env` file is ONLY used by our Python scripts
- Use `ssh_helper.py` for ALL SSH operations to avoid password prompts

### Production Server Credentials

**Location:** All credentials are in `.env` file at project root

```bash
# Production Linux Server (DO NOT ask user for these!)
SSH_HOST=200.58.107.214
SSH_PORT=5197
SSH_USER=tactical
SSH_PASSWORD=[automatically read from .env by Python scripts]
DEPLOY_PATH=/home/tactical/racing_analytics

# URLs
Dashboard: http://200.58.107.214:8050
API: http://200.58.107.214:8000
```

### Workflow for Fixing Production Issues

1. **Understand the issue** - User reports issue on http://200.58.107.214:8050
2. **Reproduce locally** (if possible) - Test on Windows development
3. **Fix the code** - Make changes in local codebase
4. **Test locally** - Verify fix works on http://localhost:8050
5. **Deploy to production** - Use Python automation scripts below
6. **Verify on production** - Check http://200.58.107.214:8050
7. **Confirm with user** - "The fix has been deployed to production"

### SSH Helper Script (ssh_helper.py)

**This is THE tool for all SSH operations. Use it instead of raw SSH commands.**

```bash
# Run any SSH command (reads .env automatically)
python ssh_helper.py "ps aux | grep python"
python ssh_helper.py "tail -30 /home/tactical/racing_analytics/api.log"
python ssh_helper.py "cd /home/tactical/racing_analytics && ls -la"

# Full diagnostic (checks dashboard, API, ports, firewall)
python ssh_helper.py

# Examples from this session:
python ssh_helper.py "netstat -tlnp | grep 8000"  # Check if API port is open
python ssh_helper.py "pkill -f uvicorn"           # Kill API process
```

### Deployment Scripts

```bash
# Quick single-file upload (e.g., after fixing app.py)
python quick_upload_app.py
# - Uploads src/dashboard/app.py
# - Restarts dashboard automatically
# - Verifies it's running

# Deploy entire src/insights directory
python deploy_insights.py
# - Uploads all 13 Python files in src/insights/
# - Clears Python cache files
# - Restarts API automatically
# - Verifies API is running on port 8000

# Full deployment (if deployment/deploy.py exists)
python deployment/deploy.py
```

### Common SSH Operations

```bash
# Check dashboard status
python ssh_helper.py "ps aux | grep dashboard"

# Check API status
python ssh_helper.py "ps aux | grep uvicorn"

# View logs
python ssh_helper.py "tail -50 /home/tactical/racing_analytics/api.log"
python ssh_helper.py "tail -50 /home/tactical/racing_analytics/dashboard.log"

# Check listening ports
python ssh_helper.py "netstat -tlnp | grep -E '(8000|8050)'"

# Restart services
python ssh_helper.py "pkill -f dashboard && cd /home/tactical/racing_analytics && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &"

# Clear Python cache (important after deploying new .py files!)
python ssh_helper.py "cd /home/tactical/racing_analytics && find src -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true"
```

### Troubleshooting SSH Issues

**Problem: "Git Bash keeps asking for SSH password"**
- **Solution:** Don't use Git Bash SSH! Use `python ssh_helper.py "command"` instead
- Git Bash doesn't know about `.env` file
- Only Python scripts with `python-dotenv` can read `.env`

**Problem: "ssh command not working"**
- **Solution:** Use `venv/Scripts/python.exe ssh_helper.py "command"`
- Make sure you're in project root: `C:\project\data_analisys_car`

**Problem: "paramiko module not found"**
- **Solution:** Install dependencies: `pip install paramiko python-dotenv`

## Tour System & Auto-Load Features

### Welcome Tour System (Phase 1 MVP - November 3, 2025)
The dashboard includes an interactive welcome tour system to onboard new users:

**Welcome Modal Features:**
- Auto-appears on first dashboard load when data is present
- Data confirmation: "71,000 telemetry samples loaded"
- Quick stats display (5 vehicles, 10 laps, 9 analysis tabs)
- Feature highlights (AI Pattern Detection, Track Maps, Telemetry, Weather)
- Professional gradient styling (purple/blue)
- Start Tour / Skip Tour buttons
- "Don't show again" preference tracking

**Implementation Files:**
- `src/dashboard/tour/welcome_modal.py` - Modal component
- `src/dashboard/assets/tour.css` - Gradient styling & animations
- `src/dashboard/app.py` - Integration & callbacks

**Testing the Welcome Modal:**
```bash
# Local testing
python src/dashboard/app.py
# Open: http://localhost:8050

# Production testing
# Open: http://200.58.107.214:8050

# Clear browser cache to see modal again
# Chrome/Edge: Ctrl+F5
# Clear localStorage: F12 → Application → Local Storage → Clear
```

**Documentation:**
- `TOUR_SYSTEM_DEPLOYMENT_COMPLETE.txt` - Full technical documentation
- `TOUR_SYSTEM_QUICK_START.txt` - Quick reference guide
- `DASHBOARD_TOUR_IMPLEMENTATION_GUIDE.md` - Phase 1 & 2 implementation
- `PRODUCT_TOUR_RESEARCH_REPORT.md` - Library comparison research

### Auto-Load Feature (November 3, 2025)
Dashboard v3.0.0+ includes automatic data loading in production:

**Features:**
- Auto-loads master_racing_data.csv from server path on startup
- Platform detection: Windows (dev) uses local path, Linux (prod) uses server path
- No upload page in production mode - dashboard displays immediately
- Stats populated automatically: 71,000 samples, 5 vehicles, 10 laps
- Welcome modal appears after data is loaded

**Implementation:**
```python
# Platform detection in app.py
import platform
PRODUCTION_MODE = platform.system() == 'Linux'

if PRODUCTION_MODE:
    auto_load_data_on_startup()
```

## Day 1 Quick Start

If this is your first time working with this codebase:
```bash
# 1. Check project status
cat QUICK_RESUME_CARD.txt

# 2. Setup environment
setup.bat                      # Windows
source venv/Scripts/activate   # Activate

# 3. Test the dashboard
python src/dashboard/app.py    # http://localhost:8050
```

### Dashboard Two-Page Flow

**Development Mode (Windows):**
- **Page 1 (Upload Page):** Clean interface with purple gradient, drag & drop upload
- **Page 2 (Dashboard):** Full analytics interface with all tabs after successful upload

**Production Mode (Linux):**
- Skips upload page entirely, auto-loads master_racing_data.csv
- Welcome modal appears with data confirmation
- Dashboard is immediately accessible

**Technical details:** See `TWO_PAGE_FLOW_IMPLEMENTATION.md`

## Resuming Work

**CRITICAL:** Before starting ANY task, ALWAYS check these files in order:
1. `QUICK_RESUME_CARD.txt` - **START HERE** - Current development status, active Sprint info, next steps
2. `HOW_TO_RESUME_*.md` - Specific resume guides for ongoing work (e.g., HOW_TO_RESUME_SPRINT_3.md)
3. Recent commit messages - `git log --oneline -10` - See what was recently completed

This project uses **Agile Sprint-based development**. Each Sprint has:
- Planning documents (`SPRINT_*_PLAN.md`)
- Task completion reports (`SPRINT_*_TASK_*_COMPLETE.md`)
- Testing checklists (`SPRINT_*_TESTING_CHECKLIST.md`)

**Never start work without checking current Sprint status first.** You might be in the middle of a Sprint with partially completed tasks.

## Critical Architecture Patterns

### 1. Module Execution Pattern
**ALWAYS use `-m` flag for src/ modules** - this is the #1 source of import errors:
```bash
# ✅ Correct
python -m src.models.baseline.train_lightgbm --tracks all

# ❌ Wrong - causes ModuleNotFoundError
python src/models/baseline/train_lightgbm.py
```
The `-m` flag adds project root to `sys.path`, enabling imports like `from data_loader import RacingDataLoader`.

### 2. Long-Format Telemetry Pattern
Telemetry data has **one row per sensor reading** (not one row per timestamp). Each timestamp has ~12 rows with different `telemetry_name` values. **Must filter by sensor type before analyzing values**:
```python
# ✅ Correct: Filter by sensor first
speed_data = df[df['telemetry_name'] == 'speed']
avg_speed = speed_data['telemetry_value'].mean()  # Now all values are km/h

# ❌ Wrong: Operating on mixed units
df['telemetry_value'].mean()  # Meaningless! Mixes km/h, bar, %, degrees
```

### 3. Chunked Memory Pattern
Telemetry files are 50MB+ each, split into 100k-row chunks. **Always start with `load_single_chunk()`** for prototyping:
```python
loader = RacingDataLoader()
# Fast prototyping (single chunk)
df = loader.load_single_chunk('barber-motorsports-park', 'race_unknown', 'telemetry', chunk_num=1)

# Production (all chunks - requires 16GB+ RAM)
df_full = loader.load_data('barber-motorsports-park', 'race_unknown', 'telemetry', combine_chunks=True)
```

### 4. Dashboard Widget Pattern
Dashboard tabs are self-contained widgets in `src/dashboard/`. Each widget file exports:
- `create_*_layout()` - Returns Dash layout components (HTML/Dash components)
- `create_*_callbacks(app)` - Registers callbacks with app instance (Input/Output/State)

Main `app.py` imports widgets, assembles tabs, and registers all callbacks. See `weather_widget.py` or `sector_widget.py` for examples.

**Widget Implementation Flow:**
1. Create layout function with unique IDs (prefix with widget name to avoid conflicts)
2. Create callbacks function that uses `@app.callback` decorator
3. Import in `app.py` and add to tabs list
4. Call `create_*_callbacks(app)` in main callback registration section

**Callback Registration:** Use `callback_context` to handle multiple triggers:
```python
from dash import callback_context

@app.callback(
    Output('modal-id', 'is_open'),
    [Input('button-1', 'n_clicks'), Input('button-2', 'n_clicks')],
    [State('modal-id', 'is_open')]
)
def toggle_modal(n1, n2, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return False
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # Handle different buttons...
    return not is_open if (n1 or n2) else is_open
```

**Global State Management:** For sharing data between callbacks, store in global dictionaries:
```python
_telemetry_cache = {}
_analysis_results_cache = {}

# In callback: access cached data by session/vehicle ID
results = _analysis_results_cache.get(cache_key, {})
```

### 5. Configuration-Driven Pattern
No magic numbers or strings in production code:
- **Thresholds:** `InsightsConfig` class (e.g., `hard_brake_threshold=110.0`)
- **Sensor names:** Enums from `src.insights.constants` (e.g., `TelemetrySensor.SPEED`)
- **Validation:** Decorators from `src.insights.validation` at entry points
- **Logging:** Automatic performance tracking via decorators (loguru, 10MB rotation)

### 6. Production Code Standards (src/insights/ only)
The `src/insights/` module follows stricter standards than the rest of the codebase:
- **Type checking:** Strict mypy with no `type: ignore` allowed
- **Testing:** 198 tests, >90% coverage required
- **Documentation:** All public methods require docstrings
- **Return types:** Pydantic models only (no raw dicts)
- **Error handling:** Structured exceptions from `src.insights.exceptions`

Other modules (`src/dashboard/`, `src/api/`, root scripts) are more lenient.

### 7. Intelligent Caching Pattern
For expensive operations (cube analysis, corner detection), use MD5 hash-based caching:
```python
import hashlib
import json

# Generate cache key from telemetry data
def _generate_cache_key(df, vehicle_number):
    data_hash = hashlib.md5(df.to_csv(index=False).encode()).hexdigest()[:16]
    return f"v{vehicle_number}_{data_hash}"

# Global cache dictionaries
_analysis_cache = {}

# Check cache before expensive operations
cache_key = _generate_cache_key(telemetry_df, vehicle_number)
if cache_key in _analysis_cache:
    return _analysis_cache[cache_key]

# Perform analysis and cache result
result = expensive_analysis(telemetry_df)
_analysis_cache[cache_key] = result
return result
```
Used in: `TelemetryAnalyzer` (pattern detection, corner analysis), dashboard callbacks.

## Domain-Specific Rules

### Never Remove "Outliers" in Racing Data
Extreme values are **features, not errors**:
- High brake pressure (13% "outliers") = hard braking zones at turn entry
- High lateral g (10.7% "outliers") = fast corners showing grip limits
- Aggressive steering (2.7% "outliers") = quick corrections

Statistical outlier removal will degrade model performance. Use domain knowledge validation instead.

### Track Names
Use lowercase-with-hyphens matching directory structure:
- ✅ `'barber-motorsports-park'`
- ❌ `'Barber Motorsports Park'`

Available: barber-motorsports-park, circuit-of-the-americas, road-america, sebring, sonoma, virginia-international-raceway

## Common Commands

### Environment Setup
```bash
# Automated setup (creates venv/, installs deps, creates data/processed/ & data/models/)
setup.bat         # Windows (RECOMMENDED - primary development platform)
./setup.sh        # Linux/Mac

# Activate environment
venv\Scripts\activate          # Windows Command Prompt
venv\Scripts\Activate.ps1      # Windows PowerShell
source venv/Scripts/activate   # Git Bash on Windows
source venv/bin/activate       # Linux/Mac

# Verify installation
python --version               # Should be 3.10+
python -c "import pandas, lightgbm, dash; print('OK')"
```
**Note:** Repo has both `venv/` and `myenv/` directories. **Always use `venv/`** (current standard). `myenv/` is legacy and should be ignored.

### Dashboard (Current version: 3.1.0-tour-system-mvp)

**⚠️ REMEMBER: User issues are about PRODUCTION (http://200.58.107.214:8050), not local!**

```bash
# LOCAL TESTING (Windows development)
python src/dashboard/app.py    # http://localhost:8050
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload  # API if needed

# PRODUCTION SERVER (Linux - where user issues occur!)
# URL: http://200.58.107.214:8050
# Auto-loads 71,000 samples, no upload page
# To fix issues here, use SSH deployment scripts!

# Deploy fixes to production:
python quick_upload_app.py     # Single file update
python deployment/deploy.py    # Full deployment
```

**Production Features:**
- Auto-loads master_racing_data.csv on startup
- Welcome modal appears on first visit
- No upload page in production mode
- Linux environment (different from Windows dev!)

**Dashboard Tabs:**
- Tab 1 (Enhanced Driver Insights): Requires telemetry upload
- Tab 2 (Telemetry Comparison): Requires telemetry upload
- Tab 3 (Model Predictions): **[SPRINT 2 COMPLETE]** Requires API + telemetry upload
  - ✅ Feature categorization (10 categories, accordion UI) - Sprint 1
  - ✅ Pattern analysis with real cube analysis (4 driving patterns detected)
  - ✅ Corner analysis modal (Speed/Braking/Cornering buttons with real data)
  - ✅ Importance-based filtering (All/Critical/Important/Advanced)
  - ✅ Track intelligence summary (6 tracks, key corners, recommendations)
  - ✅ Intelligent caching (MD5 hash-based, prevents re-analysis)
- Tab 4 (Coaching Insights): Requires telemetry upload
- Tab 5 (Post-Race Analysis): Requires telemetry upload
- Tab 6 (Track Animation): Requires telemetry with GPS data
- Tab 7 (Track Maps): Static track visualization
- Tab 8 (Weather Analysis): Uses Week 1 data
- Tab 9 (Sector Benchmarking): Uses Week 1 data
- Tab 10 (Championships): Uses Week 1 data

### Data Analysis
```bash
# Multi-track analysis (recommended)
python analyze_all_data.py

# Data inventory
python inventory_data.py

# Advanced features (FFT, wavelets)
python extract_advanced_features_efficient.py
```

### Machine Learning
```bash
# Train LightGBM baseline (ALWAYS use -m flag!)
python -m src.models.baseline.train_lightgbm --tracks all --save-model

# Hyperparameter optimization with Optuna (target: >2% R² improvement)
python -m src.models.baseline.optimize_hyperparameters --model lightgbm --trials 100

# Ensemble models
python train_ensemble_models.py
```

### Git Workflow
```bash
# Check status and branch
git status
git branch

# Add specific files (avoid adding generated data)
git add src/
git add *.md
git add QUICK_RESUME_CARD.txt  # Always update when completing tasks

# Create descriptive commits with Sprint context
git commit -m "feat: Sprint 3 Task 1 - PDF report generator"
git commit -m "fix: Modal callback issue in Model Predictions tab"
git commit -m "docs: Update QUICK_RESUME_CARD.txt with Sprint 3 status"

# Push to remote
git push origin main

# View recent changes
git log --oneline -10
git diff HEAD~1  # Compare with previous commit

# NEVER commit these directories (see .gitignore):
# - organized_data/ (18.5GB telemetry data)
# - venv/, myenv/ (virtual environments)
# - data/models/ (trained model binaries)
# - __pycache__/, *.pyc (Python cache)
# - catboost_info/ (training artifacts)
# - csv_testing/, output/, reports/ (generated test results)
# - *.csv, *.parquet, *.pkl (data files)
```

**Commit Message Conventions:**
Use conventional commits with Sprint context when applicable:
- `feat:` - New features (e.g., "feat: Sprint 3 Task 1 - Add PDF export")
- `fix:` - Bug fixes (e.g., "fix: Modal callback in corner analysis")
- `docs:` - Documentation updates (e.g., "docs: Update Sprint 3 status")
- `refactor:` - Code refactoring (no functionality change)
- `test:` - Test additions/updates
- `chore:` - Maintenance tasks (e.g., dependency updates)
- `style:` - Code formatting only (whitespace, missing semicolons, etc.)

### Testing
```bash
# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=src.insights --cov-report=html

# Specific module
pytest tests/insights/test_driver_profiler.py -v

# Single test
pytest tests/insights/test_driver_profiler.py::test_analyze_driver_performance -v

# By name pattern
pytest tests/ -k "test_analyze_driver" -v

# Type checking (strict for src/insights/)
mypy src/insights/

# Code formatting
black src/
black --check src/

# Full quality check
pytest tests/ --cov=src.insights && mypy src/insights/ && black --check src/
```

**Test Structure:**
- Fixtures: `tests/conftest.py` (sample telemetry, mock configs)
- Organization: `tests/insights/` mirrors `src/insights/` structure
- Coverage target: >90% for `src/insights/`
- Count: 198 test cases across 7 files

## High-Level Architecture

### Data Flow
```
organized_data/ (18.5GB)
    ↓
RacingDataLoader (chunked loading)
    ↓
TelemetryFeatureEngineer (100+ features)
AdvancedFeatureEngineer (60-80 FFT/wavelet features)
    ↓
LightGBMTrainer / Ensemble Models
    ↓
FastAPI (/extract-features, /predict, /insights) → Dashboard Tabs 1-4

new_data/alkamel_downloads/ (Week 1)
    ↓
WeatherDataLoader / LapAnalysisLoader / ChampionshipLoader
    ↓
Dashboard Tabs 5-7 (weather, sectors, championships)
```

### Module Dependencies
```
data_loader.py (Core - use for ALL data access)
├── src.data_processing.feature_engineering (TelemetryFeatureEngineer)
├── src.data_processing.advanced_feature_engineering (AdvancedFeatureEngineer)
├── src.insights.* (Production driver analysis - 198 tests, >90% coverage)
│   ├── driver_profiler (performance metrics, strengths/weaknesses)
│   ├── corner_analyzer (corner-by-corner analysis)
│   ├── consistency_tracker (lap consistency)
│   ├── config (InsightsConfig)
│   ├── constants (TelemetrySensor enums)
│   ├── validation (input validation decorators)
│   └── models (Pydantic models)
├── src.services.* (Business logic layer)
│   └── telemetry_analyzer (pattern detection, cube analysis)
├── src.models.baseline.train_lightgbm (LightGBMTrainer class)
├── src.api.main (FastAPI endpoints)
├── src.dashboard.app (10-tab unified dashboard)
│   ├── enhanced_driver_insights_widget
│   ├── telemetry_comparison_charts
│   ├── model_predictions_widget (Sprint 1-2: categories, patterns, track intelligence)
│   ├── pattern_analysis_widget (Sprint 2: 4 driving patterns with coaching)
│   ├── corner_analysis_widget (Sprint 2: corner modals with real metrics)
│   ├── post_race_widget
│   ├── animation_widget
│   ├── weather_widget
│   └── sector_widget
└── src.track_data.* (metadata, images, visualizations)
```

### Key Directories
```
src/
├── api/                 # FastAPI backend
├── dashboard/           # 10-tab dashboard (widget pattern)
│   ├── tour/            # Welcome tour system (Phase 1 MVP)
│   └── assets/          # CSS, images, tour.css
├── data_processing/     # Feature engineers + data loaders
├── insights/            # Production driver analysis (strict mypy, 198 tests)
├── models/baseline/     # LightGBMTrainer, hyperparameter optimization
├── services/            # Business logic layer (telemetry analyzer, pattern detection)
└── track_data/          # Track metadata, images

organized_data/          # Primary: 18.5GB telemetry (6 tracks)
│   └── [track]/         # e.g., barber-motorsports-park/
│       └── race_unknown/
│           ├── telemetry/      # *_chunk_*.csv files (50MB+ each)
│           ├── lap_times/      # Lap timing data
│           ├── weather/        # Weather conditions
│           └── results/        # Race results

new_data/alkamel_downloads/  # Week 1: weather, lap analysis, championships
tests/insights/          # 198 test cases, fixtures in conftest.py
data/                    # Auto-created: processed/, models/
```

**IMPORTANT:** Never use `find`, `ls`, or manual iteration to explore `organized_data/`. Use `RacingDataLoader.list_tracks()`, `list_races()`, `list_categories()` instead.

### Root Scripts

**Active Scripts (Use These):**
- `analyze_all_data.py` - Multi-track analysis
- `data_loader.py` - RacingDataLoader class
- `python -m src.models.baseline.train_lightgbm` - Train models
- `python src/dashboard/app.py` - Launch dashboard

**Legacy Scripts (Avoid):**
- `*_retrain*.py` - Old training (use `src.models.baseline.train_lightgbm`)
- `*_scraper*.py` - Experimental (use `RacingDataLoader`)
- `test_*.py` in root - Ad-hoc testing (use `pytest tests/`)
- `deep_data_analysis.py` - Replaced by `analyze_all_data.py`
- `*_dashboard_standalone.py` - Replaced by unified `src/dashboard/app.py`
- `train_*.py` in root - Legacy training scripts

### Documentation Files

**Focus on these core documents:**
- `QUICK_RESUME_CARD.txt` - **Check this FIRST** for current status
- `CLAUDE.md` (this file) - Architecture guide
- `README.md` - Quick start overview
- `SETUP.md` - Installation guide
- `HOW_TO_RESUME_*.md` - Context-specific resume guides

**Ignore status reports** (`*_SUMMARY.md`, `*_REPORT.md`, `*_COMPLETE.md`) unless debugging historical issues. These are project history documentation, not needed for current development.

## Code Patterns Quick Reference

### Load Data
```python
from data_loader import RacingDataLoader

loader = RacingDataLoader()
df = loader.load_single_chunk('barber-motorsports-park', 'race_unknown', 'telemetry')
```

### Feature Engineering
```python
from src.data_processing.feature_engineering import TelemetryFeatureEngineer
from src.data_processing.advanced_feature_engineering import AdvancedFeatureEngineer

engineer = TelemetryFeatureEngineer()
features = engineer.extract_features(telemetry_df)  # 100+ features

advanced = AdvancedFeatureEngineer()
advanced_features = advanced.extract_features(telemetry_df)  # FFT, wavelets
```

### Driver Insights
```python
from src.insights import DriverProfiler, InsightsConfig

config = InsightsConfig(hard_brake_threshold=110.0)
profiler = DriverProfiler(config=config)
profile = profiler.analyze_driver_performance(telemetry, vehicle_number=5)
```

### Week 1 Data
```python
from src.data_processing.weather_loader import WeatherDataLoader
from src.data_processing.lap_analysis_loader import LapAnalysisLoader
from src.data_processing.championship_loader import ChampionshipLoader

weather = WeatherDataLoader().load_weather_data()  # 518 readings, 11 sessions
laps = LapAnalysisLoader().load_lap_analysis()  # 4,468 laps, 66 drivers
champs = ChampionshipLoader().load_championship('GR Cup - Drivers')  # 16 championships
```

### Telemetry Analysis (Pattern Detection & Corner Analysis)
```python
from src.services.telemetry_analyzer import TelemetryAnalyzer

analyzer = TelemetryAnalyzer()
# Full analysis with intelligent caching (MD5 hash-based)
results = analyzer.analyze_telemetry_file(telemetry_df, vehicle_number=2)
# Returns: patterns detected, corner analysis, driving insights

# Corner-specific analysis (integrated with CubeAnalysisEngine)
corner_analysis = analyzer.analyze_corners(telemetry_df, vehicle_number=2, track_name='circuit-of-the-americas')
# Returns: corner metrics (entry/apex speeds, brake pressure), coaching per corner
```

### API Endpoints
```python
# FastAPI endpoints (port 8000)
POST /extract-features    # Extract 100+ features from telemetry CSV
POST /predict             # Make lap time predictions
POST /insights            # Generate driver performance insights
POST /pattern-analysis    # Detect driving patterns (Sprint 2)
POST /corner-analysis     # Analyze corner performance (Sprint 2)
```

## Common Pitfalls to Avoid

### Critical Mistakes That Break the Codebase
1. **Running scripts without `-m` flag** - Causes 90% of import errors
   - ❌ `python src/models/baseline/train_lightgbm.py`
   - ✅ `python -m src.models.baseline.train_lightgbm`

2. **Analyzing telemetry without filtering by sensor** - Results are meaningless
   - ❌ `df['telemetry_value'].mean()` (mixes km/h, bar, %, degrees)
   - ✅ `df[df['telemetry_name'] == 'speed']['telemetry_value'].mean()`

3. **Loading full telemetry without chunking** - Causes memory crashes
   - ❌ `load_data(..., combine_chunks=True)` on first try
   - ✅ `load_single_chunk(...)` for prototyping, then scale up

4. **Starting work without checking Sprint status** - Duplicates work or breaks ongoing tasks
   - ❌ Diving straight into coding
   - ✅ Read `QUICK_RESUME_CARD.txt` first, every time

5. **Using duplicate callback IDs in dashboard** - Causes mysterious callback errors
   - ❌ Multiple widgets using ID 'submit-button'
   - ✅ Prefix IDs with widget name: 'weather-submit-button', 'sector-submit-button'

6. **Committing large data files** - Repository becomes unmaintainable
   - ❌ `git add organized_data/`
   - ✅ Check `.gitignore` before committing anything

7. **Removing "outliers" from racing data** - Destroys valuable features
   - ❌ Statistical outlier removal (z-score > 3)
   - ✅ Keep all data, validate with domain knowledge instead

8. **Testing dashboard without starting API** - Tab 1 & 3 will fail
   - ❌ Running only `python src/dashboard/app.py`
   - ✅ Also run `python -m uvicorn src.api.main:app --port 8000` in another terminal

9. **Modifying src/insights/ without tests** - Breaks production standards
   - ❌ Quick fixes without test coverage
   - ✅ Always add tests - this module requires >90% coverage

## Troubleshooting

### Production Server Issues (http://200.58.107.214:8050)

**Dashboard not accessible:**
1. Check server status: `python ssh_helper.py "systemctl status racing-dashboard"`
2. Restart if needed: `python ssh_helper.py "sudo systemctl restart racing-dashboard"`
3. Check logs: `python ssh_helper.py "tail -n 50 /home/tactical/racing_analytics/logs/dashboard.log"`

**Changes not appearing after deployment:**
1. Clear browser cache: Ctrl+F5 (force refresh)
2. Verify file was uploaded: `python ssh_helper.py "ls -la /home/tactical/racing_analytics/src/dashboard/"`
3. Restart dashboard service: `python ssh_helper.py "sudo systemctl restart racing-dashboard"`

**Data loading issues:**
1. Check if master_racing_data.csv exists: `python ssh_helper.py "ls -la /home/tactical/racing_analytics/master_racing_data.csv"`
2. Check file permissions: `python ssh_helper.py "ls -la /home/tactical/racing_analytics/"`
3. Review dashboard logs for loading errors

### Import Errors
**Error:** `ModuleNotFoundError: No module named 'data_loader'`
**Solution:** Use `-m` flag: `python -m src.models.baseline.train_lightgbm`

**Error:** `No module named 'src'`
**Solution:** Run from project root: `cd C:\project\data_analisys_car`

### Memory Issues
**Error:** MemoryError or system slowdown
**Solution:** Use `load_single_chunk()` instead of `load_data(..., combine_chunks=True)`

### Dashboard Issues
**Tab 1 shows "API connection failed"**
**Solution:** Start API: `python -m uvicorn src.api.main:app --port 8000 --reload`
**Note:** Tabs 2-7 work without API (Tab 2-4 need telemetry upload, Tab 5-7 work immediately)

**Tabs 5-7 show "Week 1 features not available"**
**Solution:** Download data: `python download_alkamel_files.py`
Verify: `dir new_data\alkamel_downloads\` should show weather/, lap_analysis/, championships/

### Windows-Specific Issues

**Console encoding errors (Unicode characters)**
**Solution:** Scripts automatically set UTF-8 encoding. If issues persist:
```bash
chcp 65001  # Set console to UTF-8
```

**Path issues with backslashes**
**Solution:** Use raw strings or Path objects:
```python
from pathlib import Path
data_path = Path("organized_data/barber-motorsports-park")  # Works on all platforms
```

**PowerShell execution policy blocks scripts**
**Solution:** Run PowerShell as Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Git Bash Python path issues**
**Solution:** Use Windows-style paths or `winpty`:
```bash
winpty python src/dashboard/app.py  # If regular python command fails in Git Bash
```

### Quick Debugging Commands
```bash
# Check Python version (need 3.10+)
python --version

# Verify environment
python -c "import pandas, lightgbm, dash; print('OK')"

# Check data
python -c "from data_loader import RacingDataLoader; loader = RacingDataLoader(); print('Tracks:', loader.list_tracks())"

# Test dashboard
python src/dashboard/app.py
```

## Sensor Types & Units

| Sensor | Description | Units | Range |
|--------|-------------|-------|-------|
| speed | Vehicle speed | km/h | 66-190 |
| pbrake_f | Front brake pressure | bar | 0-153 |
| pbrake_r | Rear brake pressure | bar | 0-153 |
| aps | Throttle position | % | 0-100 |
| accx_can | Longitudinal acceleration | g | -1.5 to 1.5 |
| accy_can | Lateral acceleration | g | -2.0 to 2.0 |
| Steering_Angle | Steering wheel angle | deg | -540 to 540 |
| gear | Current gear | int | 1-6 |
| nmot | Engine RPM | rpm | 1000-8000 |
| gps_lat | GPS latitude | deg | varies |
| gps_long | GPS longitude | deg | varies |
| gps_alt | GPS altitude | m | varies |

## Environment Variables

### API Configuration (src/api/main.py)
```bash
export MAX_FILE_SIZE=524288000  # 500MB upload limit
export ALLOWED_ORIGINS="http://example.com"  # Production CORS
export ENVIRONMENT=production
```

### Insights Configuration (src/insights/config.py)
```python
from src.insights import InsightsConfig

config = InsightsConfig(
    hard_brake_threshold=110.0,      # bar (default: 100.0)
    full_throttle_threshold=95.0,    # % (default: 95.0)
    high_lateral_g_threshold=1.2,    # g (default: 1.0)
    log_level='INFO'                 # DEBUG|INFO|WARNING|ERROR
)
```

## Performance Notes

- **Memory:** 8GB min, 16GB recommended (for `combine_chunks=True`)
- **CPU:** All code CPU-only, no GPU required
- **Training:** LightGBM <5 minutes on full dataset
- **API Response:** <1s features, <2s insights
- **Ports:** Dashboard 8050, API 8000
- **Chunk Size:** 100k rows (configurable in `organize_and_chunk_data.py`)

## Sprint Development Process

This project follows Agile Sprint methodology. Each Sprint has a clear lifecycle:

### Sprint Lifecycle
1. **Planning** - Create `SPRINT_N_PLAN.md` with tasks, timelines, success criteria
2. **Implementation** - Complete tasks, updating `QUICK_RESUME_CARD.txt` after each task
3. **Testing** - Use `SPRINT_N_TESTING_CHECKLIST.md` for QA
4. **Documentation** - Create task completion reports (`SPRINT_N_TASK_M_COMPLETE.md`)
5. **Sprint Closure** - Create final report (`SPRINT_N_COMPLETE_FINAL_REPORT.md`)

### Working on Sprint Tasks
When asked to "start Sprint N Task M":
1. Read the Sprint plan: `SPRINT_N_PLAN.md`
2. Check task dependencies (must previous tasks be complete?)
3. Create/update implementation files
4. Test the feature thoroughly
5. Update `QUICK_RESUME_CARD.txt` with task status
6. Create task completion report if it's a major task
7. Commit changes with clear Sprint context in message

### Sprint Status Tracking
**Current Sprint status is ALWAYS in `QUICK_RESUME_CARD.txt`**
- Sprint number and phase
- Completed tasks (✅)
- In-progress tasks (🔄)
- Pending tasks (❌)
- Next recommended action

## Common Development Patterns

### Adding a New Dashboard Widget
1. Create widget file in `src/dashboard/` with `create_*_layout()` and `create_*_callbacks(app)` functions
2. Import in `src/dashboard/app.py`
3. Add to tabs list in app layout
4. Register callbacks using `create_*_callbacks(app)` in main callback section
5. Test by running dashboard and navigating to new tab

### Adding a New API Endpoint
1. Add route to `src/api/main.py`
2. Use Pydantic models for request/response validation
3. Enable CORS for dashboard access
4. Test with `test_api.py` or curl commands
5. Document endpoint in API docstring

### Adding a New Analysis Feature
1. If production-grade: Add to `src/insights/` with full typing, tests, docstrings
2. If experimental: Add to root-level script
3. Use `RacingDataLoader` for all data access
4. Filter telemetry by sensor type BEFORE analysis
5. Return Pydantic models (insights) or DataFrame (data processing)

### Testing Dashboard Changes
```bash
# Terminal 1: Start API (if needed)
python -m uvicorn src.api.main:app --reload

# Terminal 2: Start dashboard
python src/dashboard/app.py

# Test with sample data
# Upload: master_racing_data.csv or TEST_*.csv files in root
```

## Current Project Status

**Always check `QUICK_RESUME_CARD.txt` for the latest status.**

**🚨 PRODUCTION CONTEXT:** Dashboard is LIVE at http://200.58.107.214:8050
- **User reports issues about THIS SERVER** (not local development)
- **To fix issues:** Code locally → Test → Deploy via SSH → Verify on production
- **Platform:** Linux (production) vs Windows (development) - environments differ!

The project follows Agile Sprint methodology with production deployment on Linux server (200.58.107.214:8050).

**Recent Completions:**
- Sprint 1-2: Complete (feature categorization, pattern analysis, corner analysis)
- Tour System Phase 1: Complete (welcome modal)
- Auto-Load Feature: Complete (production mode)
- Dashboard v3.1.0: Deployed

**See `QUICK_RESUME_CARD.txt` for:**
- Current Sprint/Task status
- Next recommended actions
- In-progress work
- Testing requirements

## Reference Documentation

### Essential Documents
- `QUICK_RESUME_CARD.txt` - Current status (check first!)
- `CLAUDE.md` (this file) - Architecture guide
- `README.md` - Quick start overview
- `SETUP.md` - Installation guide

### Recent Features (November 2025)
- `TOUR_SYSTEM_DEPLOYMENT_COMPLETE.txt` - Welcome modal technical docs
- `TOUR_SYSTEM_QUICK_START.txt` - Quick reference guide
- `DASHBOARD_TOUR_IMPLEMENTATION_GUIDE.md` - Phase 1 & 2 implementation
- `DEPLOYMENT_COMPLETE_SUMMARY.txt` - Auto-load feature deployment

### Sprint Documentation
- `SPRINT_3_PLAN.md` - Sprint 3 planning (export & persistence)
- `HOW_TO_RESUME_SPRINT_3.md` - Sprint 3 resume guide
- `SPRINT_2_COMPLETE_FINAL_REPORT.md` - Sprint 2 summary

### Technical Guides
- `QUICK_START_GUIDE.md` - Hackathon examples
- `TWO_PAGE_FLOW_IMPLEMENTATION.md` - Two-page flow details
- `TESTING_AND_QA_SUMMARY.md` - Test coverage metrics