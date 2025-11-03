"""
Type-Safe Data Models for Racing Insights Module

Production-grade Pydantic models providing:
- Strongly-typed return structures
- Automatic validation
- JSON serialization
- IDE autocomplete support
- Clear API contracts

Design Pattern: Data Transfer Object (DTO) Pattern
- Immutable data structures
- Validated at creation
- Clear field contracts
- Self-documenting

Replaces generic Dict[str, any] with strongly-typed models.

Author: Production Engineering Team
Version: 1.0.0
License: GR Cup 2025 Hackathon
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


# ==============================================================================
# ENUMERATIONS FOR TYPE SAFETY
# ==============================================================================

class DrivingStyleEnum(str, Enum):
    """Driving style classifications."""
    AGGRESSIVE = "Aggressive"
    SMOOTH = "Smooth"
    CONSERVATIVE = "Conservative"
    BALANCED = "Balanced"


class PerformanceTrendEnum(str, Enum):
    """Performance trajectory classifications."""
    IMPROVING = "Improving"
    STABLE = "Stable"
    DECLINING = "Declining"
    INCONSISTENT = "Inconsistent"


class OutlierTypeEnum(str, Enum):
    """Outlier classification types."""
    EXCEPTIONALLY_FAST = "Exceptionally Fast"
    EXCEPTIONALLY_SLOW = "Exceptionally Slow"


# ==============================================================================
# SUB-MODELS FOR NESTED STRUCTURES
# ==============================================================================

class BrakingProfile(BaseModel):
    """Braking characteristics profile."""
    max_brake_pressure: float = Field(..., description="Maximum brake pressure (bar)")
    avg_brake_pressure: float = Field(..., description="Average brake pressure (bar)")
    brake_consistency: float = Field(..., description="Brake pressure consistency score")
    brake_variance: float = Field(..., description="Brake pressure variance (bar)")
    hard_braking_events: int = Field(..., description="Count of hard braking events", ge=0)

    class Config:
        frozen = True  # Immutable


class ThrottleProfile(BaseModel):
    """Throttle application characteristics."""
    full_throttle_percentage: float = Field(..., description="Percentage of lap at full throttle", ge=0, le=100)
    avg_throttle: float = Field(..., description="Average throttle position (%)", ge=0, le=100)
    throttle_smoothness: float = Field(..., description="Throttle smoothness score (0-100)", ge=0, le=100)
    partial_throttle_percentage: float = Field(..., description="Percentage at partial throttle", ge=0, le=100)

    class Config:
        frozen = True


class CorneringProfile(BaseModel):
    """Cornering performance characteristics."""
    max_lateral_g: float = Field(..., description="Maximum lateral g-force")
    avg_lateral_g: float = Field(..., description="Average lateral g-force in corners")
    avg_cornering_speed: float = Field(..., description="Average cornering speed (km/h)", ge=0)
    cornering_speed_index: Optional[float] = Field(None, description="Cornering speed performance index")

    class Config:
        frozen = True


# ==============================================================================
# DRIVER PROFILE MODEL
# ==============================================================================

class DriverProfile(BaseModel):
    """
    Comprehensive driver performance profile.

    Complete type-safe structure for DriverProfiler.create_profile() output.
    """
    vehicle_number: int = Field(..., description="Vehicle/driver identifier", ge=0, le=19)
    total_laps: int = Field(..., description="Total number of laps analyzed", ge=0)

    # Performance Metrics
    driving_style: DrivingStyleEnum = Field(..., description="Classified driving style")
    consistency_score: float = Field(..., description="Lap time consistency score (0-100)", ge=0, le=100)
    aggression_index: float = Field(..., description="Driving aggression index (0-100)", ge=0, le=100)
    smoothness_index: float = Field(..., description="Input smoothness index (0-100)", ge=0, le=100)

    # Analysis Components
    braking_profile: BrakingProfile = Field(..., description="Braking characteristics")
    throttle_profile: ThrottleProfile = Field(..., description="Throttle characteristics")
    cornering_profile: CorneringProfile = Field(..., description="Cornering characteristics")

    # Insights
    strengths: List[str] = Field(default_factory=list, description="Identified strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Identified weaknesses")
    recommendations: Optional[List[str]] = Field(None, description="Training recommendations")

    class Config:
        frozen = True

    def to_summary_dict(self) -> Dict[str, Any]:
        """Get summary dictionary for quick display."""
        return {
            "vehicle_number": self.vehicle_number,
            "total_laps": self.total_laps,
            "driving_style": self.driving_style.value,
            "consistency": f"{self.consistency_score:.1f}/100",
            "aggression": f"{self.aggression_index:.1f}/100",
            "smoothness": f"{self.smoothness_index:.1f}/100",
            "strengths": self.strengths[:3],  # Top 3
            "weaknesses": self.weaknesses[:3]  # Top 3
        }


# ==============================================================================
# CORNER ANALYSIS MODELS
# ==============================================================================

class CornerData(BaseModel):
    """Individual corner detection result."""
    corner_id: int = Field(..., description="Corner identifier", ge=1)
    start_timestamp: int = Field(..., description="Corner entry timestamp (ms since epoch)")
    end_timestamp: int = Field(..., description="Corner exit timestamp (ms since epoch)")
    duration_ms: int = Field(..., description="Corner duration in milliseconds", ge=0)
    min_speed: float = Field(..., description="Minimum speed in corner (km/h)", ge=0)
    entry_speed: Optional[float] = Field(None, description="Speed at corner entry (km/h)")
    apex_speed: Optional[float] = Field(None, description="Speed at apex (km/h)")
    exit_speed: Optional[float] = Field(None, description="Speed at corner exit (km/h)")

    class Config:
        frozen = True


class CornerPerformance(BaseModel):
    """Corner performance analysis result."""
    corner_id: int = Field(..., description="Corner identifier")
    entry_speed: float = Field(..., description="Entry speed (km/h)")
    apex_speed: float = Field(..., description="Apex speed (km/h)")
    exit_speed: float = Field(..., description="Exit speed (km/h)")
    braking_point_timestamp: Optional[int] = Field(None, description="Braking point timestamp")
    throttle_point_timestamp: Optional[int] = Field(None, description="Throttle application point timestamp")
    max_lateral_g: Optional[float] = Field(None, description="Maximum lateral g-force")
    max_longitudinal_g: Optional[float] = Field(None, description="Maximum longitudinal g-force")

    class Config:
        frozen = True


class OptimalLine(BaseModel):
    """Optimal racing line for a corner."""
    corner_id: int = Field(..., description="Corner identifier")
    optimal_entry_speed: float = Field(..., description="Optimal entry speed (km/h)")
    optimal_apex_speed: float = Field(..., description="Optimal apex speed (km/h)")
    optimal_exit_speed: float = Field(..., description="Optimal exit speed (km/h)")
    based_on_lap: int = Field(..., description="Lap number this optimal line is based on")
    lap_time: float = Field(..., description="Lap time of reference lap (seconds)")

    class Config:
        frozen = True


# ==============================================================================
# CONSISTENCY TRACKING MODELS
# ==============================================================================

class SessionMetrics(BaseModel):
    """Session-level consistency metrics."""
    session_id: str = Field(..., description="Unique session identifier")
    vehicle_number: int = Field(..., description="Vehicle/driver identifier", ge=0, le=19)
    total_laps: int = Field(..., description="Total laps in session", ge=0)
    fastest_lap: float = Field(..., description="Fastest lap time (seconds)", gt=0)
    average_lap: float = Field(..., description="Average lap time (seconds)", gt=0)
    lap_time_std: float = Field(..., description="Lap time standard deviation", ge=0)
    consistency_score: float = Field(..., description="Consistency score (0-100)", ge=0, le=100)
    outlier_count: int = Field(..., description="Number of outlier laps", ge=0)

    class Config:
        frozen = True


class PerformanceTrend(BaseModel):
    """Performance trend analysis result."""
    vehicle_number: int = Field(..., description="Vehicle/driver identifier")
    trend: PerformanceTrendEnum = Field(..., description="Overall performance trend")
    avg_change_per_session: float = Field(..., description="Average change per session (seconds)")
    total_improvement: float = Field(..., description="Total improvement from first to last session (seconds)")
    slope: float = Field(..., description="Linear regression slope")
    confidence: float = Field(..., description="Trend confidence (0-1)", ge=0, le=1)
    sessions_analyzed: int = Field(..., description="Number of sessions in analysis", ge=2)

    class Config:
        frozen = True


class OutlierLap(BaseModel):
    """Outlier lap detection result."""
    lap: int = Field(..., description="Lap number")
    lap_duration: float = Field(..., description="Lap duration (seconds)")
    z_score: float = Field(..., description="Z-score relative to session mean")
    outlier_type: OutlierTypeEnum = Field(..., description="Type of outlier")
    deviation_seconds: float = Field(..., description="Deviation from mean (seconds)")

    class Config:
        frozen = True


# ==============================================================================
# COMPARISON MODELS
# ==============================================================================

class DriverComparison(BaseModel):
    """Multi-driver comparison result."""
    vehicle_numbers: List[int] = Field(..., description="Vehicles being compared")
    fastest_driver: int = Field(..., description="Vehicle number of fastest driver")
    most_consistent: int = Field(..., description="Vehicle number of most consistent driver")
    most_aggressive: int = Field(..., description="Vehicle number of most aggressive driver")
    profiles: Dict[int, DriverProfile] = Field(..., description="Individual driver profiles")

    class Config:
        frozen = True


# ==============================================================================
# ERROR RESPONSE MODEL
# ==============================================================================

class ErrorResponse(BaseModel):
    """
    Standardized error response.

    Used when analysis fails or data is invalid.
    """
    error: bool = Field(True, description="Error flag (always True)")
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")

    class Config:
        frozen = True


# ==============================================================================
# PROGRESS TRACKING MODEL
# ==============================================================================

class AnalysisProgress(BaseModel):
    """Progress tracking for long-running analysis."""
    total_steps: int = Field(..., description="Total number of analysis steps", ge=1)
    current_step: int = Field(..., description="Current step number", ge=0)
    step_name: str = Field(..., description="Current step description")
    percent_complete: float = Field(..., description="Percentage complete (0-100)", ge=0, le=100)
    estimated_time_remaining_seconds: Optional[float] = Field(None, description="Estimated time remaining")

    @validator('percent_complete', pre=True, always=True)
    def calculate_percent(cls, v, values):
        """Auto-calculate percentage if not provided."""
        if v is None and 'current_step' in values and 'total_steps' in values:
            return (values['current_step'] / values['total_steps']) * 100
        return v

    class Config:
        frozen = True


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def driver_profile_from_dict(data: dict) -> DriverProfile:
    """
    Create DriverProfile from dictionary (e.g., from old code).

    Provides migration path from dict-based to model-based returns.

    Parameters:
        data: Dictionary with driver profile data

    Returns:
        DriverProfile instance

    Example:
        old_dict = profiler.create_profile_old(...)
        profile = driver_profile_from_dict(old_dict)
    """
    # Extract braking profile
    braking = BrakingProfile(
        max_brake_pressure=data.get('max_brake_pressure', 0.0),
        avg_brake_pressure=data.get('avg_brake_pressure', 0.0),
        brake_consistency=data.get('brake_consistency', 0.0),
        brake_variance=data.get('brake_variance', 0.0),
        hard_braking_events=data.get('hard_braking_events', 0)
    )

    # Extract throttle profile
    throttle = ThrottleProfile(
        full_throttle_percentage=data.get('full_throttle_percentage', 0.0),
        avg_throttle=data.get('avg_throttle', 0.0),
        throttle_smoothness=data.get('throttle_smoothness', 0.0),
        partial_throttle_percentage=data.get('partial_throttle_percentage', 0.0)
    )

    # Extract cornering profile
    cornering = CorneringProfile(
        max_lateral_g=data.get('max_lateral_g', 0.0),
        avg_lateral_g=data.get('avg_lateral_g', 0.0),
        avg_cornering_speed=data.get('avg_cornering_speed', 0.0),
        cornering_speed_index=data.get('cornering_speed_index')
    )

    return DriverProfile(
        vehicle_number=data['vehicle_number'],
        total_laps=data['total_laps'],
        driving_style=DrivingStyleEnum(data['driving_style']),
        consistency_score=data['consistency_score'],
        aggression_index=data['aggression_index'],
        smoothness_index=data['smoothness_index'],
        braking_profile=braking,
        throttle_profile=throttle,
        cornering_profile=cornering,
        strengths=data.get('strengths', []),
        weaknesses=data.get('weaknesses', []),
        recommendations=data.get('recommendations')
    )
