"""
Racing Analytics API - Main Application
========================================

FastAPI application providing endpoints for:
- Lap time prediction
- Feature engineering
- Driver coaching insights
- Model explanations
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import io
import logging
import os
import time
import gc  # Fix for Issue #002: Memory leak prevention

# Import project modules
from src.data_processing.feature_engineering import TelemetryFeatureEngineer
from src.insights import DriverProfiler, CornerAnalyzer, ConsistencyTracker
from src.insights import InsightsConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 500 * 1024 * 1024))  # Default: 500MB
# Note: For files >500MB, use the data_loader.py directly or chunk the data first
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8050,http://localhost:3000").split(",")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Initialize FastAPI app
app = FastAPI(
    title="Racing Analytics API",
    description="Professional racing telemetry analysis and lap time prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for dashboard integration
# Production: Set ALLOWED_ORIGINS env var to specific domains
if ENVIRONMENT == "production":
    logger.info(f"🔒 Production mode: CORS restricted to {ALLOWED_ORIGINS}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )
else:
    logger.warning("⚠️ Development mode: CORS allows all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Required CSV columns
REQUIRED_COLUMNS = {'telemetry_name', 'telemetry_value', 'vehicle_number', 'timestamp', 'lap'}

# Global model storage
MODEL_PATH = Path("data/models/lightgbm_multi_track.pkl")
model = None
feature_engineer = TelemetryFeatureEngineer()
insights_config = InsightsConfig()

# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    model_loaded: bool

class PredictionRequest(BaseModel):
    """Request model for lap time prediction"""
    features: Dict[str, float]

    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        return v

class PredictionResponse(BaseModel):
    """Response model for lap time prediction"""
    predicted_lap_time: float = Field(..., description="Predicted lap time in seconds")
    confidence_interval: Optional[List[float]] = Field(None, description="95% confidence interval [lower, upper]")
    top_features: List[Dict[str, Any]] = Field(..., description="Top contributing features")

class DriverInsightsResponse(BaseModel):
    """Response model for driver insights"""
    vehicle_number: int
    performance_summary: Dict[str, Any]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]

class FeaturesResponse(BaseModel):
    """Response model for feature engineering"""
    num_laps: int
    num_features: int
    feature_names: List[str]
    sample_features: Optional[Dict[str, Any]]

# ============================================================================
# Middleware
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests with timing"""
    start_time = time.time()

    # Log request
    logger.info(f"→ {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Log response
    duration = time.time() - start_time
    logger.info(
        f"← {request.method} {request.url.path} "
        f"[{response.status_code}] {duration:.3f}s"
    )

    return response

# ============================================================================
# Helper Functions
# ============================================================================

async def validate_csv_file(file: UploadFile) -> bytes:
    """
    Validate uploaded CSV file for size and content

    Returns: File contents as bytes
    Raises: HTTPException if validation fails
    """
    # Read file contents
    contents = await file.read()

    # Check file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
        )

    # Check file extension
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed"
        )

    # Validate it's not empty
    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    return contents

def validate_telemetry_dataframe(df: pd.DataFrame) -> None:
    """
    Validate telemetry DataFrame has required structure

    Raises: HTTPException if validation fails
    """
    # Check not empty
    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file contains no data rows"
        )

    # Check minimum rows for meaningful analysis
    if len(df) < 100:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for analysis (minimum 100 rows required, got {len(df)})"
        )

    # Check required columns
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing_cols)}"
        )

    # Validate data types
    if not pd.api.types.is_numeric_dtype(df['telemetry_value']):
        raise HTTPException(
            status_code=400,
            detail="Column 'telemetry_value' must contain numeric data"
        )

    if not pd.api.types.is_numeric_dtype(df['vehicle_number']):
        raise HTTPException(
            status_code=400,
            detail="Column 'vehicle_number' must contain numeric data"
        )

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def load_model():
    """Load ML model on startup"""
    global model
    try:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"✅ Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"⚠️ Model not found at {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        model = None

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Racing Analytics API")

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="online",
        version="1.0.0",
        model_loaded=model is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_lap_time(request: PredictionRequest):
    """
    Predict lap time from engineered features

    **Example Request:**
    ```json
    {
        "features": {
            "avg_speed": 150.5,
            "max_lateral_g": 1.8,
            "traction_circle_utilization": 0.85,
            ...
        }
    }
    ```
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])

        # Get feature names from model (LightGBM compatibility)
        expected_features = getattr(model, 'feature_name_', getattr(model, 'feature_names_in_', list(request.features.keys())))

        # Check for missing features
        missing_features = set(expected_features) - set(features_df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {list(missing_features)[:5]}..."
            )

        # Reorder columns to match training
        features_df = features_df[expected_features]

        # Make prediction
        prediction = model.predict(features_df)[0]

        # Get feature importance (try different attributes for compatibility)
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'feature_importance'):
            feature_importance = model.feature_importance()
        else:
            # No feature importance available
            feature_importance = np.ones(len(expected_features))

        top_indices = np.argsort(feature_importance)[-5:][::-1]
        top_features = [
            {
                "name": expected_features[i],
                "value": float(features_df.iloc[0, i]),
                "importance": float(feature_importance[i])
            }
            for i in top_indices
        ]

        return PredictionResponse(
            predicted_lap_time=float(prediction),
            top_features=top_features
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction. Please contact support."
        )

@app.post("/extract-features", response_model=FeaturesResponse)
async def extract_features(file: UploadFile = File(...)):
    """
    Extract engineered features from raw telemetry CSV

    **Upload:** CSV file with telemetry data
    **Returns:** Engineered features ready for prediction
    """
    telemetry_df = None
    features_df = None
    try:
        # Validate and read CSV file
        contents = await validate_csv_file(file)
        telemetry_df = pd.read_csv(io.BytesIO(contents))

        logger.info(f"📁 Received telemetry file: {len(telemetry_df)} rows")

        # Validate DataFrame structure (but filter out unknown sensors instead of failing)
        try:
            validate_telemetry_dataframe(telemetry_df)
        except Exception as e:
            logger.warning(f"Validation warning: {e}")
            # Filter to only known sensors
            from src.insights.constants import TelemetrySensor
            known_sensors = TelemetrySensor.all_sensors()
            if 'telemetry_name' in telemetry_df.columns:
                telemetry_df = telemetry_df[telemetry_df['telemetry_name'].isin(known_sensors)]
                logger.info(f"Filtered to {len(telemetry_df)} rows with known sensors")

        # Extract features using the same approach as Post-Race Analysis
        # This extracts 130+ features (44 basic + 86+ advanced)
        from src.data_processing.advanced_feature_engineering import AdvancedFeatureEngineer

        # Extract track name for advanced features
        track_name = telemetry_df['track'].iloc[0] if 'track' in telemetry_df.columns and len(telemetry_df) > 0 else 'unknown'

        # Step 1: Extract basic features (44) using process_session
        logger.info("Extracting basic features...")
        basic_features = feature_engineer.process_session(telemetry_df)

        if len(basic_features) == 0:
            raise HTTPException(
                status_code=400,
                detail="Feature extraction failed - no laps processed. Check telemetry data format."
            )

        logger.info(f"Basic features: {len(basic_features)} laps, {len(basic_features.columns)} columns")

        # Step 2: Extract advanced features (86+)
        logger.info(f"Extracting advanced features for track: {track_name}...")
        advanced_engineer = AdvancedFeatureEngineer()

        try:
            # Use process_session_advanced which returns combined DataFrame
            features_df = advanced_engineer.process_session_advanced(
                telemetry_df=telemetry_df,
                base_features_df=basic_features,
                track_name=track_name
            )

            logger.info(f"Combined features: {len(features_df)} laps, {len(features_df.columns)} columns")

        except Exception as e:
            logger.warning(f"Advanced feature extraction failed: {e}. Using basic features only.")
            features_df = basic_features

        logger.info(f"✅ Extracted {len(features_df)} laps with {len(features_df.columns)} features")

        # Get sample features (first lap)
        sample_features = None
        if len(features_df) > 0:
            sample_features = features_df.iloc[0].to_dict()
            # Convert numpy types to Python types
            sample_features = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                             for k, v in sample_features.items()}

        return FeaturesResponse(
            num_laps=len(features_df),
            num_features=len(features_df.columns),
            feature_names=list(features_df.columns),
            sample_features=sample_features
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature extraction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during feature extraction. Please contact support."
        )
    finally:
        # Fix for Issue #002: Explicitly release memory from large DataFrames
        if telemetry_df is not None:
            del telemetry_df
        if features_df is not None:
            del features_df
        gc.collect()  # Force garbage collection for large objects

@app.post("/driver-insights", response_model=DriverInsightsResponse)
async def get_driver_insights(
    file: UploadFile = File(...),
    vehicle_number: int = Query(0, ge=0, le=20, description="Vehicle number (0-20)")
):
    """
    Generate comprehensive driver insights from telemetry - SIMPLIFIED VERSION

    **Upload:** CSV file with telemetry data in long format
    **Query Param:** vehicle_number (default: 0, range: 0-20)
    **Returns:** Performance analysis, strengths, weaknesses, recommendations
    """
    telemetry_df = None
    try:
        # Read CSV file
        contents = await file.read()
        telemetry_df = pd.read_csv(io.BytesIO(contents))

        logger.info(f"📊 Analyzing driver #{vehicle_number} from {len(telemetry_df)} telemetry rows")

        # Validate required columns
        required_cols = ['telemetry_name', 'telemetry_value', 'vehicle_number', 'lap']
        missing_cols = [col for col in required_cols if col not in telemetry_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )

        # Filter data for the requested vehicle
        vehicle_data = telemetry_df[telemetry_df['vehicle_number'] == vehicle_number].copy()

        if len(vehicle_data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for vehicle #{vehicle_number}. Available vehicles: {sorted(telemetry_df['vehicle_number'].unique().tolist())}"
            )

        logger.info(f"Found {len(vehicle_data)} rows for vehicle #{vehicle_number}")

        # Calculate simple metrics by filtering telemetry_name
        # Speed metrics
        speed_data = vehicle_data[vehicle_data['telemetry_name'] == 'speed']['telemetry_value']
        avg_speed = float(speed_data.mean()) if len(speed_data) > 0 else 0.0
        max_speed = float(speed_data.max()) if len(speed_data) > 0 else 0.0

        # Throttle metrics (aps = throttle position %)
        throttle_data = vehicle_data[vehicle_data['telemetry_name'] == 'aps']['telemetry_value']
        avg_throttle = float(throttle_data.mean()) if len(throttle_data) > 0 else 0.0

        # Brake metrics
        brake_data = vehicle_data[vehicle_data['telemetry_name'] == 'pbrake_f']['telemetry_value']
        avg_brake = float(brake_data.mean()) if len(brake_data) > 0 else 0.0
        max_brake = float(brake_data.max()) if len(brake_data) > 0 else 0.0

        # Lateral G metrics
        lat_g_data = vehicle_data[vehicle_data['telemetry_name'] == 'accy_can']['telemetry_value']
        max_lat_g = float(lat_g_data.abs().max()) if len(lat_g_data) > 0 else 0.0

        # Lap count
        total_laps = int(vehicle_data['lap'].nunique())

        # Calculate simple scores for dashboard
        # Consistency: based on data quality (higher is better, 0-100 scale)
        consistency_score = min(100, (total_laps * 20) + (len(vehicle_data) / 1000))

        # Aggression: based on braking intensity (0-100 scale)
        aggression_index = min(100, (max_brake / 1.5) + (avg_throttle / 2))

        # Smoothness: inverse of throttle variance (0-100 scale, simplified)
        smoothness_rating = min(100, 100 - (throttle_data.std() if len(throttle_data) > 0 else 50))

        # Build performance summary matching dashboard expectations
        performance_summary = {
            "consistency_score": round(consistency_score, 1),
            "aggression_index": round(aggression_index, 1),
            "smoothness_rating": round(smoothness_rating, 1),
            "avg_speed": round(avg_speed, 1),
            "max_speed": round(max_speed, 1),
            "avg_brake_pressure": round(avg_brake, 1),
            "max_brake_pressure": round(max_brake, 1)
        }

        # Generate insights based on metrics
        strengths = []
        weaknesses = []
        recommendations = []

        # Speed analysis
        if max_speed > 180:
            strengths.append(f"High top speed achieved: {max_speed:.1f} km/h")
        elif max_speed < 140:
            weaknesses.append(f"Low top speed: {max_speed:.1f} km/h")
            recommendations.append("Work on carrying more speed through corners and on straights")

        # Throttle analysis
        if avg_throttle > 60:
            strengths.append(f"Good throttle application: {avg_throttle:.1f}% average")
        elif avg_throttle < 40:
            weaknesses.append(f"Conservative throttle use: {avg_throttle:.1f}% average")
            recommendations.append("Build confidence with earlier and more aggressive throttle application")

        # Braking analysis
        if max_brake > 100:
            strengths.append(f"Strong braking: {max_brake:.1f} bar peak pressure")
        elif max_brake < 60:
            weaknesses.append(f"Light braking: {max_brake:.1f} bar peak pressure")
            recommendations.append("Practice threshold braking to maximize braking potential")

        # Cornering analysis
        if max_lat_g > 1.2:
            strengths.append(f"Excellent cornering forces: {max_lat_g:.2f}g")
        elif max_lat_g > 0.8:
            strengths.append(f"Good cornering capability: {max_lat_g:.2f}g")
        else:
            weaknesses.append(f"Conservative cornering: {max_lat_g:.2f}g")
            recommendations.append("Increase corner entry speed and use more of the track width")

        # Data quality check
        if total_laps < 3:
            recommendations.append(f"Limited data available ({total_laps} laps). Upload more laps for detailed analysis.")

        # Default messages if lists are empty
        if not strengths:
            strengths.append("Baseline performance established")

        if not recommendations:
            recommendations.append("Continue practicing to build consistency and confidence")

        logger.info(f"✅ Analysis complete for vehicle #{vehicle_number}")

        return DriverInsightsResponse(
            vehicle_number=vehicle_number,
            performance_summary=performance_summary,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )

    except HTTPException:
        raise
    except Exception as e:
        # Simple error handling
        logger.error(f"Driver insights error: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
    finally:
        # Clean up memory
        if telemetry_df is not None:
            del telemetry_df
        gc.collect()

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get feature names (LightGBM compatibility)
        if hasattr(model, 'feature_name_'):
            feature_names = model.feature_name_
        elif hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = [f"feature_{i}" for i in range(model.num_feature())]

        # Get feature importance (LightGBM compatibility)
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'feature_importance'):
            feature_importance = model.feature_importance(importance_type='gain')
        else:
            feature_importance = np.ones(len(feature_names))

        top_indices = np.argsort(feature_importance)[-10:][::-1]

        top_features = [
            {
                "rank": i + 1,
                "name": feature_names[idx],
                "importance": float(feature_importance[idx])
            }
            for i, idx in enumerate(top_indices)
        ]

        return {
            "model_type": type(model).__name__,
            "num_features": len(feature_names),
            "feature_names": feature_names,
            "top_10_features": top_features,
            "model_path": str(MODEL_PATH)
        }
    except Exception as e:
        logger.error(f"Model info error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while retrieving model information. Please contact support."
        )

# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
