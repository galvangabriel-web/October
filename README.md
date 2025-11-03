# GR Cup Racing Analytics

**Professional Racing Telemetry Analysis System**

![Model Performance](https://img.shields.io/badge/R²-97.49%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Production-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)

---

## 🏁 View Full Documentation

**[📖 Open Complete Technical Documentation (index.html)](index.html)**

The comprehensive documentation includes:
- **97.49% R² Lap Time Prediction Model** (Sequential LightGBM)
- **10-Tab Interactive Dashboard** (Pattern Analysis, Corner Intelligence, Track Maps)
- **Real-time FastAPI Prediction System** (<200ms latency)
- **Full Production Deployment Guide** (Linux server ready)
- **147 Advanced Features** (Telemetry, FFT, Sequential, Domain-Specific)
- **Sprint 2 Achievements** (Feature Categorization, Driving Patterns, AI Coaching)

---

## Quick Start

```bash
# 1. Setup environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start dashboard
python src/dashboard/app.py
# Access at http://localhost:8050

# 4. Start API (optional)
python -m uvicorn src.api.main:app --port 8000
```

---

## Project Structure

```
src/
├── dashboard/          # 10-tab Dash application
├── api/               # FastAPI prediction endpoints
├── insights/          # Production analysis modules (198 tests, >90% coverage)
├── models/            # LightGBM trainers and optimizers
├── data_processing/   # Feature engineering (147 features)
├── services/          # Business logic (pattern detection, cube analysis)
└── track_data/        # Track metadata and visualizations

data/models/           # Trained models (included in repo)
tests/                 # Test suite (198 test cases)
deployment/            # Production deployment scripts
```

---

## Key Features

### Machine Learning
- **Sequential LightGBM**: 97.49% R², 1.733s MAE
- **147 Features**: Telemetry (45) + Advanced (89) + Sequential (13)
- **Real-time Predictions**: <200ms API response time
- **Multi-track Support**: 6 professional racing circuits

### Dashboard Intelligence
- **Pattern Analysis**: 4 driving patterns with AI coaching
- **Corner Intelligence**: Speed/braking/cornering metrics per corner
- **Feature Categorization**: 10 categories with importance filtering
- **Track Intelligence**: Circuit-specific recommendations
- **Live Telemetry**: Real-time visualization and comparison

### Production Ready
- **Deployed**: November 2025 (Linux server, full capabilities)
- **Tested**: 198 test cases, >90% coverage (src/insights/)
- **Documented**: Type hints, docstrings, API schemas
- **Monitored**: Structured logging, performance tracking

---

## Technology Stack

**Backend**: Python 3.10+, FastAPI, LightGBM, pandas/polars
**Frontend**: Dash/Plotly (10 interactive tabs)
**ML**: LightGBM, XGBoost, CatBoost, scikit-learn
**Testing**: pytest (198 tests), mypy (strict typing)
**Deployment**: Linux server, systemd services, SSH automation

---

**Built for GR Cup Racing Competition** | Production Deployment: November 2025
