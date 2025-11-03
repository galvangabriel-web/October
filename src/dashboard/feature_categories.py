"""
Feature Categorization System for Model Predictions Tab
========================================================

Organizes 138+ telemetry features into professional racing categories
for intuitive exploration and analysis.

Categories (10):
1. Speed & Acceleration
2. Braking Performance
3. Throttle Management
4. Cornering Dynamics
5. Steering Control
6. Powertrain & Gear Management
7. Composite Performance Metrics
8. FFT Frequency Analysis
9. Wavelet Pattern Analysis
10. Lap Segmentation & Timing

Each category includes:
- Display name
- Icon (Font Awesome)
- Description
- Color theme
- Feature patterns (regex/keywords)
- Importance level (Critical/Important/Advanced)
"""

from typing import Dict, List, Set, Tuple
from enum import Enum
import re


class FeatureImportance(Enum):
    """Importance levels for feature categories"""
    CRITICAL = "critical"      # Essential for lap time
    IMPORTANT = "important"    # Significant impact
    ADVANCED = "advanced"      # Deep analysis


class FeatureCategory:
    """Represents a feature category with metadata"""

    def __init__(
        self,
        id: str,
        name: str,
        icon: str,
        description: str,
        color: str,
        importance: FeatureImportance,
        patterns: List[str]
    ):
        self.id = id
        self.name = name
        self.icon = icon
        self.description = description
        self.color = color
        self.importance = importance
        self.patterns = patterns  # List of regex patterns or keywords

    def matches_feature(self, feature_name: str) -> bool:
        """Check if a feature name matches this category"""
        feature_lower = feature_name.lower()

        for pattern in self.patterns:
            # Try as regex first
            try:
                if re.search(pattern, feature_lower, re.IGNORECASE):
                    return True
            except re.error:
                # Fall back to substring match
                if pattern.lower() in feature_lower:
                    return True

        return False


# ============================================================================
# CATEGORY DEFINITIONS
# ============================================================================

FEATURE_CATEGORIES = [
    FeatureCategory(
        id="speed",
        name="Speed & Acceleration",
        icon="fa-tachometer-alt",
        description="Vehicle speed metrics, acceleration, and velocity analysis",
        color="#3498db",  # Blue
        importance=FeatureImportance.CRITICAL,
        patterns=[
            r"^speed",
            r"velocity",
            r"accx",
            r"acceleration",
            r"max_speed",
            r"min_speed",
            r"avg_speed",
            r"speed_range",
            r"speed_std",
            r"speed_cv",
        ]
    ),

    FeatureCategory(
        id="braking",
        name="Braking Performance",
        icon="fa-stop-circle",
        description="Brake pressure, timing, consistency, and trail braking analysis",
        color="#e74c3c",  # Red
        importance=FeatureImportance.CRITICAL,
        patterns=[
            r"brake",
            r"pbrake",
            r"braking",
            r"decel",
        ]
    ),

    FeatureCategory(
        id="throttle",
        name="Throttle Management",
        icon="fa-bolt",
        description="Throttle application, modulation, and full-throttle usage",
        color="#2ecc71",  # Green
        importance=FeatureImportance.CRITICAL,
        patterns=[
            r"^aps",
            r"throttle",
            r"full_throttle",
            r"part_throttle",
        ]
    ),

    FeatureCategory(
        id="cornering",
        name="Cornering Dynamics",
        icon="fa-sync-alt",
        description="Lateral G-forces, cornering speed, and turn performance",
        color="#9b59b6",  # Purple
        importance=FeatureImportance.CRITICAL,
        patterns=[
            r"accy",
            r"lateral",
            r"cornering",
            r"turn",
            r"g_force",
            r"lat_g",
        ]
    ),

    FeatureCategory(
        id="steering",
        name="Steering Control",
        icon="fa-life-ring",
        description="Steering angle, rate of change, smoothness, and corrections",
        color="#f39c12",  # Orange
        importance=FeatureImportance.IMPORTANT,
        patterns=[
            r"steering",
            r"steer",
            r"angle",
            r"turn_in",
        ]
    ),

    FeatureCategory(
        id="powertrain",
        name="Powertrain & Gear Management",
        icon="fa-cog",
        description="Engine RPM, gear selection, shift timing, and transmission metrics",
        color="#34495e",  # Dark gray
        importance=FeatureImportance.IMPORTANT,
        patterns=[
            r"^gear",
            r"rpm",
            r"nmot",
            r"engine",
            r"shift",
        ]
    ),

    FeatureCategory(
        id="composite",
        name="Composite Performance Metrics",
        icon="fa-layer-group",
        description="Combined metrics: track position, rolling statistics, aggregated performance",
        color="#16a085",  # Teal
        importance=FeatureImportance.IMPORTANT,
        patterns=[
            r"rolling",
            r"window",
            r"ratio",
            r"efficiency",
            r"balance",
            r"delta",
            r"avg_",
            r"max_",
            r"min_",
            r"std_",
            r"range_",
            r"pct_",
            r"count_",
        ]
    ),

    FeatureCategory(
        id="fft",
        name="FFT Frequency Analysis",
        icon="fa-wave-square",
        description="Frequency domain analysis: vibrations, oscillations, periodic patterns",
        color="#e67e22",  # Orange-red
        importance=FeatureImportance.ADVANCED,
        patterns=[
            r"fft",
            r"freq",
            r"frequency",
            r"hz_",
            r"power_",
            r"spectral",
        ]
    ),

    FeatureCategory(
        id="wavelet",
        name="Wavelet Pattern Analysis",
        icon="fa-chart-line",
        description="Time-frequency analysis: transient events, multi-scale patterns",
        color="#8e44ad",  # Deep purple
        importance=FeatureImportance.ADVANCED,
        patterns=[
            r"wavelet",
            r"cwt",
            r"dwt",
            r"scale_",
        ]
    ),

    FeatureCategory(
        id="lap_seg",
        name="Lap Segmentation & Timing",
        icon="fa-flag-checkered",
        description="Lap-based features: lap times, sector splits, consistency metrics",
        color="#c0392b",  # Dark red
        importance=FeatureImportance.CRITICAL,
        patterns=[
            r"^lap",
            r"sector",
            r"split",
            r"time_",
            r"duration",
            r"segment",
        ]
    ),
]


# ============================================================================
# CATEGORIZATION FUNCTIONS
# ============================================================================

def categorize_features(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Categorize a list of feature names into defined categories

    Args:
        feature_names: List of feature names to categorize

    Returns:
        Dictionary mapping category IDs to lists of matching feature names
    """
    categorized = {cat.id: [] for cat in FEATURE_CATEGORIES}
    uncategorized = []

    for feature_name in feature_names:
        matched = False

        for category in FEATURE_CATEGORIES:
            if category.matches_feature(feature_name):
                categorized[category.id].append(feature_name)
                matched = True
                break  # Only assign to first matching category

        if not matched:
            uncategorized.append(feature_name)

    # Add uncategorized to composite if not empty
    if uncategorized:
        categorized['uncategorized'] = uncategorized

    return categorized


def get_category_by_id(category_id: str) -> FeatureCategory:
    """Get category object by ID"""
    for cat in FEATURE_CATEGORIES:
        if cat.id == category_id:
            return cat
    return None


def get_category_summary(categorized_features: Dict[str, List[str]]) -> List[Dict]:
    """
    Generate summary statistics for categorized features

    Args:
        categorized_features: Output from categorize_features()

    Returns:
        List of category summaries with counts and percentages
    """
    total_features = sum(len(features) for features in categorized_features.values())

    summary = []
    for category in FEATURE_CATEGORIES:
        features = categorized_features.get(category.id, [])
        count = len(features)
        percentage = (count / total_features * 100) if total_features > 0 else 0

        summary.append({
            'id': category.id,
            'name': category.name,
            'icon': category.icon,
            'description': category.description,
            'color': category.color,
            'importance': category.importance.value,
            'count': count,
            'percentage': percentage,
            'features': features
        })

    # Add uncategorized if present
    if 'uncategorized' in categorized_features:
        features = categorized_features['uncategorized']
        count = len(features)
        percentage = (count / total_features * 100) if total_features > 0 else 0

        summary.append({
            'id': 'uncategorized',
            'name': 'Other Metrics',
            'icon': 'fa-question-circle',
            'description': 'Uncategorized features',
            'color': '#95a5a6',
            'importance': 'important',
            'count': count,
            'percentage': percentage,
            'features': features
        })

    # Sort by importance and count
    importance_order = {
        FeatureImportance.CRITICAL.value: 0,
        FeatureImportance.IMPORTANT.value: 1,
        FeatureImportance.ADVANCED.value: 2,
    }

    summary.sort(key=lambda x: (importance_order.get(x['importance'], 3), -x['count']))

    return summary


def detect_available_categories(feature_names: List[str]) -> Set[str]:
    """
    Detect which categories have available features

    Args:
        feature_names: List of feature names

    Returns:
        Set of category IDs that have at least one feature
    """
    categorized = categorize_features(feature_names)
    return {cat_id for cat_id, features in categorized.items() if len(features) > 0}


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

def validate_categorization(feature_names: List[str]) -> Dict[str, any]:
    """
    Validate that all features are properly categorized

    Returns:
        Dictionary with validation results
    """
    categorized = categorize_features(feature_names)
    uncategorized = categorized.get('uncategorized', [])

    total_categorized = sum(len(features) for cat_id, features in categorized.items() if cat_id != 'uncategorized')
    total_features = len(feature_names)

    return {
        'total_features': total_features,
        'categorized': total_categorized,
        'uncategorized_count': len(uncategorized),
        'uncategorized_names': uncategorized,
        'coverage_percentage': (total_categorized / total_features * 100) if total_features > 0 else 0,
        'categories': get_category_summary(categorized)
    }


if __name__ == '__main__':
    # Test with sample feature names
    sample_features = [
        'speed_mean', 'speed_max', 'speed_std',
        'pbrake_f_max', 'pbrake_r_mean',
        'aps_mean', 'aps_pct_full_throttle',
        'accy_can_max', 'accy_can_std',
        'Steering_Angle_mean', 'Steering_Angle_range',
        'gear_changes', 'nmot_mean',
        'speed_fft_0.5hz', 'speed_fft_1.0hz',
        'speed_wavelet_scale1', 'speed_wavelet_scale2',
        'lap_time', 'sector_1_time'
    ]

    print("Feature Categorization Test")
    print("=" * 80)

    categorized = categorize_features(sample_features)

    for cat in FEATURE_CATEGORIES:
        features = categorized.get(cat.id, [])
        if features:
            print(f"\n{cat.name} ({len(features)} features):")
            for feat in features:
                print(f"  - {feat}")

    if 'uncategorized' in categorized:
        print(f"\nUncategorized ({len(categorized['uncategorized'])} features):")
        for feat in categorized['uncategorized']:
            print(f"  - {feat}")

    print("\n" + "=" * 80)
    validation = validate_categorization(sample_features)
    print(f"\nCoverage: {validation['coverage_percentage']:.1f}%")
    print(f"Categorized: {validation['categorized']} / {validation['total_features']}")
