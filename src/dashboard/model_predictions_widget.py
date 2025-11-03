"""
Enhanced Model Predictions Widget - Sprint 1
===========================================

Professional racing analytics interface with:
- 10 categorized feature groups
- Expandable accordion UI
- Category filtering
- Feature availability detection
- Color-coded importance levels
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
from typing import Dict, List, Optional
import logging

from src.dashboard.feature_categories import (
    categorize_features,
    get_category_summary,
    get_category_by_id,
    FEATURE_CATEGORIES,
    FeatureImportance
)

logger = logging.getLogger(__name__)


# Track metadata (hardcoded for simplicity - can be extended later)
TRACK_METADATA = {
    'circuit-of-the-americas': {
        'name': 'Circuit of the Americas',
        'type': 'High-speed road course',
        'turns': 20,
        'length_miles': 3.41,
        'elevation_ft': 133,
        'description': 'Technical track with elevation changes and high-speed sections'
    },
    'barber-motorsports-park': {
        'name': 'Barber Motorsports Park',
        'type': 'Technical road course',
        'turns': 17,
        'length_miles': 2.38,
        'elevation_ft': 80,
        'description': 'Challenging track with tight hairpins and flowing sections'
    },
    'road-america': {
        'name': 'Road America',
        'type': 'High-speed road course',
        'turns': 14,
        'length_miles': 4.05,
        'elevation_ft': 160,
        'description': 'Fast, flowing circuit with long straights'
    },
    'sebring': {
        'name': 'Sebring International Raceway',
        'type': 'Bumpy road course',
        'turns': 17,
        'length_miles': 3.74,
        'elevation_ft': 10,
        'description': 'Historic track with rough surface and challenging corners'
    },
    'sonoma': {
        'name': 'Sonoma Raceway',
        'type': 'Hilly road course',
        'turns': 12,
        'length_miles': 2.52,
        'elevation_ft': 160,
        'description': 'Technical track with significant elevation changes'
    },
    'virginia-international-raceway': {
        'name': 'Virginia International Raceway',
        'type': 'Fast road course',
        'turns': 17,
        'length_miles': 3.27,
        'elevation_ft': 100,
        'description': 'Challenging track with mix of technical and high-speed sections'
    }
}


def create_track_intelligence_section(
    track_name: str = 'circuit-of-the-americas',
    corner_analyses: Optional[List[Dict]] = None,
    patterns_data: Optional[List[Dict]] = None
) -> dbc.Card:
    """
    Create track intelligence summary section (Sprint 2 Task 6)

    Args:
        track_name: Track identifier (e.g., 'circuit-of-the-americas')
        corner_analyses: List of corner analysis dictionaries from TelemetryAnalyzer
        patterns_data: List of detected patterns from cube analysis

    Returns:
        Dash Bootstrap Card with track intelligence summary
    """
    # Get track metadata
    track_info = TRACK_METADATA.get(track_name, {
        'name': 'Unknown Track',
        'type': 'Road course',
        'turns': 0,
        'length_miles': 0.0,
        'elevation_ft': 0,
        'description': 'No track information available'
    })

    # Get top 3-5 corners by time delta (highest opportunity first)
    key_corners = []
    if corner_analyses and len(corner_analyses) > 0:
        # Sort by time_delta descending (higher = more opportunity)
        sorted_corners = sorted(
            corner_analyses,
            key=lambda x: abs(x.get('time_delta', 0)),
            reverse=True
        )
        key_corners = sorted_corners[:5]

    # Get recommendations from patterns
    recommendations = []
    if patterns_data and len(patterns_data) > 0:
        # Extract top 3 patterns by impact
        sorted_patterns = sorted(
            patterns_data,
            key=lambda x: x.get('impact_seconds', 0),
            reverse=True
        )
        for pattern in sorted_patterns[:3]:
            recommendations.append({
                'category': pattern.get('category', 'Performance'),
                'issue': pattern.get('name', 'Unknown'),
                'impact': pattern.get('impact_seconds', 0.0)
            })

    # Build UI components
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-flag-checkered me-2", style={'color': '#e74c3c'}),
                "Track Intelligence Summary"
            ], className="mb-0")
        ], style={'backgroundColor': '#f8f9fa', 'borderBottom': '2px solid #dee2e6'}),
        dbc.CardBody([
            # Track Characteristics
            html.Div([
                html.H6([
                    html.I(className="fas fa-road me-2", style={'color': '#3498db'}),
                    "Track Characteristics"
                ], className="mb-3"),
                html.Div([
                    html.Strong(track_info['name'], className="d-block mb-2", style={'fontSize': '1.1rem'}),
                    html.Div([
                        html.Span([
                            html.I(className="fas fa-tag me-1"),
                            f"Type: {track_info['type']}"
                        ], className="me-3 text-muted"),
                        html.Span([
                            html.I(className="fas fa-arrows-left-right me-1"),
                            f"{track_info['length_miles']} miles"
                        ], className="me-3 text-muted"),
                        html.Span([
                            html.I(className="fas fa-chart-line me-1"),
                            f"{track_info['turns']} turns"
                        ], className="me-3 text-muted"),
                        html.Span([
                            html.I(className="fas fa-mountain me-1"),
                            f"{track_info['elevation_ft']} ft elevation"
                        ], className="text-muted"),
                    ], className="mb-2"),
                    html.P(track_info['description'], className="text-muted small mb-0")
                ])
            ], className="mb-4"),

            # Key Corners
            html.Div([
                html.H6([
                    html.I(className="fas fa-bullseye me-2", style={'color': '#e74c3c'}),
                    "Key Corners for Lap Time"
                ], className="mb-3"),
                html.Div([
                    html.Ul([
                        html.Li([
                            html.Strong(corner.get('corner_name', f"Corner {corner.get('corner_number', i+1)}")),
                            html.Span(f" - {abs(corner.get('time_delta', 0)):.2f}s opportunity",
                                     className="text-danger ms-2 small")
                        ], className="mb-2") if abs(corner.get('time_delta', 0)) > 0.01 else None
                        for i, corner in enumerate(key_corners)
                    ], className="mb-0") if key_corners else html.P("Upload telemetry to analyze corner performance",
                                                                     className="text-muted small mb-0")
                ])
            ], className="mb-4") if key_corners else None,

            # Recommended Focus Areas
            html.Div([
                html.H6([
                    html.I(className="fas fa-lightbulb me-2", style={'color': '#f39c12'}),
                    "Recommended Focus Areas"
                ], className="mb-3"),
                html.Div([
                    html.Ul([
                        html.Li([
                            html.Strong(f"{rec['category']}: "),
                            html.Span(rec['issue'], className="text-muted me-2"),
                            dbc.Badge(f"{rec['impact']:.2f}s", color="warning", className="small")
                        ], className="mb-2")
                        for rec in recommendations
                    ], className="mb-0") if recommendations else html.P("Upload telemetry to get personalized recommendations",
                                                                        className="text-muted small mb-0")
                ])
            ], className="mb-0")
        ])
    ], className="shadow-sm mb-4")


def create_importance_badge(importance: str) -> dbc.Badge:
    """Create a badge indicating feature importance level"""
    badge_config = {
        'critical': {'text': 'CRITICAL', 'color': 'danger'},
        'important': {'text': 'IMPORTANT', 'color': 'warning'},
        'advanced': {'text': 'ADVANCED', 'color': 'info'}
    }

    config = badge_config.get(importance, {'text': 'STANDARD', 'color': 'secondary'})

    return dbc.Badge(
        config['text'],
        color=config['color'],
        className="ms-2",
        style={'fontSize': '0.65rem', 'fontWeight': '600'}
    )


def create_feature_badge(feature_name: str, is_available: bool = True) -> dbc.Badge:
    """Create a badge for an individual feature"""
    return dbc.Badge(
        feature_name,
        color="light" if is_available else "secondary",
        text_color="dark" if is_available else "light",
        className="me-2 mb-2",
        style={
            'fontSize': '0.75rem',
            'fontWeight': '400',
            'padding': '0.4rem 0.7rem',
            'borderRadius': '0.25rem',
            'border': '1px solid' + ('#dee2e6' if is_available else '#6c757d'),
            'opacity': '1' if is_available else '0.5'
        }
    )


def create_category_header(category_summary: Dict, is_expanded: bool = False) -> dbc.CardHeader:
    """Create the header for a category accordion item"""
    category = get_category_by_id(category_summary['id'])

    # Icon
    icon = html.I(
        className=f"fas {category_summary['icon']} me-3",
        style={'color': category_summary['color'], 'fontSize': '1.2rem'}
    )

    # Category name and count
    title = html.Span([
        html.Strong(category_summary['name']),
        html.Span(
            f" ({category_summary['count']} features)",
            className="text-muted ms-2",
            style={'fontSize': '0.9rem'}
        )
    ])

    # Importance badge
    importance_badge = create_importance_badge(category_summary['importance'])

    # Add "Analyze" button for categories with data (October 2025 update - user request)
    # Originally only 3 categories (speed, braking, cornering) had buttons in Sprint 2
    # Now 9 categories have analyze buttons for complete data visibility
    # Note: FFT and Wavelet categories are not displayed in current feature set
    analyze_button = None
    analyzable_categories = ['speed', 'braking', 'cornering', 'throttle', 'steering',
                            'powertrain', 'composite', 'lap_seg', 'uncategorized']
    if category_summary['id'] in analyzable_categories:
        analyze_button = dbc.Button(
            [
                html.I(className="fas fa-chart-line me-1"),
                "Analyze"
            ],
            id=f"analyze-btn-{category_summary['id']}",
            color="primary",
            size="sm",
            outline=True,
            className="ms-2",
            style={'fontSize': '0.75rem', 'padding': '0.25rem 0.5rem'}
        )

    # Percentage bar
    percentage_bar = dbc.Progress(
        value=category_summary['percentage'],
        className="mt-2",
        style={'height': '4px'},
        bar=True,
        color="primary"  # Use standard bootstrap color
    )

    # Expand/collapse indicator
    expand_icon = html.I(
        className=f"fas fa-chevron-{'up' if is_expanded else 'down'} ms-auto",
        style={'fontSize': '0.8rem', 'color': '#6c757d'}
    )

    # Build header row with optional analyze button
    header_items = [icon, title, importance_badge]
    if analyze_button:
        header_items.append(analyze_button)
    header_items.append(expand_icon)

    return dbc.CardHeader([
        html.Div([
            html.Div(header_items, style={'display': 'flex', 'alignItems': 'center'}),
            percentage_bar
        ]),
    ],
    id=f"header-{category_summary['id']}",  # Add ID for click handler
    style={
        'cursor': 'pointer',
        'backgroundColor': '#f8f9fa',
        'borderLeft': f'4px solid {category_summary["color"]}',
        'padding': '1rem'
    })


def create_category_body(category_summary: Dict, all_features: List[str]) -> dbc.CardBody:
    """Create the body content for a category accordion item"""
    features = category_summary['features']

    if len(features) == 0:
        return dbc.CardBody([
            html.P("No features in this category", className="text-muted small mb-0")
        ])

    # Description
    description = html.P(
        category_summary['description'],
        className="text-muted mb-3",
        style={'fontSize': '0.9rem', 'fontStyle': 'italic'}
    )

    # Feature badges
    feature_badges = html.Div([
        create_feature_badge(feat, is_available=True)
        for feat in sorted(features)
    ], style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'gap': '4px'
    })

    # Statistics
    stats = dbc.Row([
        dbc.Col([
            html.Small([
                html.I(className="fas fa-check-circle me-1 text-success"),
                f"{len(features)} features available"
            ], className="text-muted")
        ], width="auto"),
        dbc.Col([
            html.Small([
                html.I(className="fas fa-percent me-1 text-info"),
                f"{category_summary['percentage']:.1f}% of total"
            ], className="text-muted")
        ], width="auto"),
    ], className="mt-3")

    return dbc.CardBody([
        description,
        feature_badges,
        stats
    ], style={'backgroundColor': '#ffffff'})


def create_model_predictions_layout(
    features_data: Dict,
    prediction_result: Optional[Dict] = None,
    patterns_data: Optional[List[Dict]] = None,
    corner_analyses: Optional[List[Dict]] = None,
    track_name: str = 'circuit-of-the-americas'
) -> dbc.Container:
    """
    Create the enhanced Model Predictions tab layout (Sprint 1 + Sprint 2)

    Args:
        features_data: Dictionary with num_features, num_laps, feature_names
        prediction_result: Optional prediction result from API
        patterns_data: Optional list of detected patterns from cube analysis
        corner_analyses: Optional list of corner analysis dictionaries (Sprint 2 Task 6)
        track_name: Track identifier for track intelligence (Sprint 2 Task 6)

    Returns:
        Dash Bootstrap Container with full layout
    """
    components = []

    # Import pattern widget for Sprint 2
    try:
        from src.dashboard.pattern_analysis_widget import create_patterns_section, DEMO_PATTERNS
    except ImportError:
        logger.warning("Pattern analysis widget not available")
        patterns_data = None

    # Import corner analysis widget for Sprint 2 Task 3
    try:
        from src.dashboard.corner_analysis_widget import create_corner_analysis_modal, DEMO_CORNER_ANALYSES
        corner_analysis_available = True
    except ImportError:
        logger.warning("Corner analysis widget not available")
        corner_analysis_available = False

    # ========================================================================
    # SECTION 1: Summary Statistics
    # ========================================================================

    num_features = features_data.get('num_features', 0)
    num_laps = features_data.get('num_laps', 0)
    feature_names = features_data.get('feature_names', [])

    # Categorize features
    categorized = categorize_features(feature_names)
    category_summaries = get_category_summary(categorized)

    # Count categories with features
    active_categories = sum(1 for cat in category_summaries if cat['count'] > 0)

    # Summary cards
    summary_row = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-layer-group fa-2x mb-2", style={'color': '#3498db'}),
                        html.H3(str(num_features), className="mb-0"),
                        html.P("Total Features", className="text-muted small mb-0")
                    ], className="text-center")
                ])
            ], className="shadow-sm h-100")
        ], md=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-folder-open fa-2x mb-2", style={'color': '#2ecc71'}),
                        html.H3(str(active_categories), className="mb-0"),
                        html.P("Active Categories", className="text-muted small mb-0")
                    ], className="text-center")
                ])
            ], className="shadow-sm h-100")
        ], md=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-flag-checkered fa-2x mb-2", style={'color': '#e74c3c'}),
                        html.H3(str(num_laps), className="mb-0"),
                        html.P("Laps Analyzed", className="text-muted small mb-0")
                    ], className="text-center")
                ])
            ], className="shadow-sm h-100")
        ], md=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-check-circle fa-2x mb-2", style={'color': '#f39c12'}),
                        html.H3("100%", className="mb-0"),
                        html.P("Coverage", className="text-muted small mb-0")
                    ], className="text-center")
                ])
            ], className="shadow-sm h-100")
        ], md=3),
    ], className="mb-4")

    components.append(summary_row)

    # ========================================================================
    # SECTION 2: Prediction Result (if available)
    # ========================================================================

    if prediction_result:
        pred_time = prediction_result['predicted_lap_time']
        top_features = prediction_result.get('top_features', [])

        prediction_card = dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-stopwatch me-2"),
                    "Predicted Lap Time"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H1(f"{pred_time:.3f}s", className="text-primary text-center mb-0"),
                        html.P("Based on extracted features", className="text-muted text-center small")
                    ], md=6),
                    dbc.Col([
                        html.H6("Top Contributing Features:", className="mb-3"),
                        html.Div([
                            html.Div([
                                dbc.Badge(f"#{i+1}", color="primary", className="me-2"),
                                html.Span(feat['name'], className="me-2"),
                                dbc.Badge(f"{feat['importance']:.0f}", color="secondary", className="me-2"),
                            ], className="mb-2") for i, feat in enumerate(top_features[:5])
                        ])
                    ], md=6),
                ])
            ])
        ], className="shadow-sm mb-4")

        components.append(prediction_card)

    # ========================================================================
    # SECTION 2.5: Driving Patterns Analysis (Sprint 2)
    # ========================================================================

    # Show patterns if provided, otherwise use demo patterns for testing
    if patterns_data is not None:
        try:
            patterns_section = create_patterns_section(patterns_data)
            components.append(patterns_section)
        except Exception as e:
            logger.error(f"Error creating patterns section: {e}")
    else:
        # Use demo patterns for Sprint 2 testing
        try:
            patterns_section = create_patterns_section(DEMO_PATTERNS)
            components.append(patterns_section)
            logger.info("Using demo patterns for display")
        except Exception as e:
            logger.warning(f"Could not display patterns: {e}")

    # ========================================================================
    # SECTION 2.6: Track Intelligence Summary (Sprint 2 Task 6)
    # ========================================================================

    try:
        track_intelligence_section = create_track_intelligence_section(
            track_name=track_name,
            corner_analyses=corner_analyses,
            patterns_data=patterns_data
        )
        components.append(track_intelligence_section)
        logger.info(f"Track intelligence section created for {track_name}")
    except Exception as e:
        logger.warning(f"Could not create track intelligence section: {e}")

    # ========================================================================
    # SECTION 3: Category Filters (Sprint 1)
    # ========================================================================

    # Filter buttons - Sprint 2 Task 4
    filter_section = dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="fas fa-filter me-2"),
                html.Strong("Quick Filters: "),
                dbc.ButtonGroup([
                    dbc.Button("All", id="filter-btn-all", size="sm", color="primary", outline=True, className="me-1"),
                    dbc.Button("Critical", id="filter-btn-critical", size="sm", color="danger", outline=True, className="me-1"),
                    dbc.Button("Important", id="filter-btn-important", size="sm", color="warning", outline=True, className="me-1"),
                    dbc.Button("Advanced", id="filter-btn-advanced", size="sm", color="info", outline=True, className="me-1"),
                ], className="ms-2")
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], style={'padding': '0.75rem'})
    ], className="mb-3")

    components.append(filter_section)

    # ========================================================================
    # SECTION 4: Categorized Features Accordion
    # ========================================================================

    accordion_items = []

    for i, category_summary in enumerate(category_summaries):
        if category_summary['count'] == 0:
            continue  # Skip empty categories

        item = dbc.AccordionItem(
            [
                create_category_body(category_summary, feature_names)
            ],
            title="",  # We'll use custom header
            item_id=f"category-{category_summary['id']}",
        )

        # Customize the accordion item with our header
        # Note: We'll need to style this with custom CSS
        # Add ID and data-importance for filtering (Sprint 2 Task 4)
        accordion_items.append(
            html.Div([
                create_category_header(category_summary, is_expanded=True),
                dbc.Collapse(
                    create_category_body(category_summary, feature_names),
                    id=f"collapse-{category_summary['id']}",
                    is_open=True,  # All categories expanded by default
                )
            ],
            id=f"category-container-{category_summary['id']}",
            className="mb-2 category-item",
            **{'data-importance': category_summary['importance']},
            style={
                'border': '1px solid #dee2e6',
                'borderRadius': '0.25rem',
                'overflow': 'hidden'
            })
        )

    features_section = dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-list-alt me-2"),
                "Feature Categories"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            html.P([
                "Features organized into ",
                html.Strong(f"{active_categories} professional categories"),
                ". Click to expand and explore."
            ], className="text-muted mb-3"),
            html.Div(accordion_items)
        ])
    ], className="shadow-sm")

    components.append(features_section)

    # ========================================================================
    # SECTION 5: Legend
    # ========================================================================

    legend = dbc.Card([
        dbc.CardBody([
            html.Small([
                html.I(className="fas fa-info-circle me-2 text-info"),
                html.Strong("Importance Levels: "),
                create_importance_badge('critical'),
                html.Span(" - Essential for lap time performance", className="me-3"),
                create_importance_badge('important'),
                html.Span(" - Significant impact", className="me-3"),
                create_importance_badge('advanced'),
                html.Span(" - Deep analysis metrics", className="me-3"),
            ], className="text-muted")
        ], style={'padding': '0.75rem', 'backgroundColor': '#f8f9fa'})
    ], className="mt-3")

    components.append(legend)

    # ========================================================================
    # SECTION 6: Corner Analysis Modal (Sprint 2 Task 3)
    # ========================================================================

    # NOTE: Modal is now in base app.layout (app.py) for proper callback registration.
    # The modal's children are updated via callback when Analyze buttons are clicked.
    # No need to add modal here - it exists globally in the app.

    # ========================================================================
    # SECTION 7: Export Analysis (Sprint 3 Task 1)
    # ========================================================================

    export_section = dbc.Card([
        dbc.CardBody([
            html.H5([
                html.I(className="fas fa-file-download me-2", style={'color': '#3498db'}),
                "Export Analysis"
            ], className="mb-3"),
            html.P(
                "Export this analysis to PDF report or CSV data for offline review, team meetings, or external analysis.",
                className="text-muted mb-3"
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [html.I(className="fas fa-file-pdf me-2"), "Export PDF Report"],
                        id="export-pdf-btn",
                        color="danger",
                        className="w-100",
                        size="lg"
                    ),
                    html.Small(
                        "Professional report with patterns, corners, and track intelligence",
                        className="text-muted d-block mt-2 text-center"
                    )
                ], md=6),
                dbc.Col([
                    dbc.Button(
                        [html.I(className="fas fa-file-csv me-2"), "Export CSV Data"],
                        id="export-csv-btn",
                        color="success",
                        className="w-100",
                        size="lg"
                    ),
                    html.Small(
                        "Raw data for Excel, Python, or R analysis",
                        className="text-muted d-block mt-2 text-center"
                    )
                ], md=6)
            ]),
            # Download components (hidden, triggered by callbacks)
            dcc.Download(id="download-pdf"),
            dcc.Download(id="download-csv")
        ])
    ], className="mt-4 shadow-sm")

    components.append(export_section)

    return dbc.Container(components, fluid=True)


def create_model_predictions_error(error_type: str, message: str) -> dbc.Container:
    """Create error message for Model Predictions tab"""
    icon_map = {
        'connection': 'fa-exclamation-triangle',
        'timeout': 'fa-clock',
        'extraction': 'fa-exclamation-circle',
        'model': 'fa-robot'
    }

    color_map = {
        'connection': 'danger',
        'timeout': 'warning',
        'extraction': 'danger',
        'model': 'warning'
    }

    icon = icon_map.get(error_type, 'fa-exclamation-circle')
    color = color_map.get(error_type, 'warning')

    return dbc.Container([
        dbc.Alert([
            html.I(className=f"fas {icon} me-2"),
            message
        ], color=color)
    ], fluid=True)


# ============================================================================
# HELPER FUNCTIONS FOR CALLBACKS (Future implementation)
# ============================================================================

def create_model_predictions_callbacks(app):
    """
    Create callbacks for Model Predictions tab

    Future Sprint 1 implementation:
    - Category expand/collapse
    - Filter by importance
    - Search features
    - Export feature list
    """
    # TODO: Implement in next phase
    pass
