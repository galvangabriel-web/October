"""
Pattern Analysis Widget - Sprint 2
===================================

Displays detected driving patterns from cube analysis with:
- Severity indicators (High/Medium/Low)
- Impact estimates (seconds per lap)
- WHAT × WHERE × WHEN breakdown
- Specific coaching recommendations
"""

import dash_bootstrap_components as dbc
from dash import html
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def create_severity_badge(severity: str) -> dbc.Badge:
    """Create severity badge with color coding"""
    badge_config = {
        'High': {'color': 'danger', 'icon': 'fa-exclamation-circle'},
        'Medium': {'color': 'warning', 'icon': 'fa-exclamation-triangle'},
        'Low': {'color': 'info', 'icon': 'fa-info-circle'}
    }

    config = badge_config.get(severity, {'color': 'secondary', 'icon': 'fa-question'})

    return dbc.Badge([
        html.I(className=f"fas {config['icon']} me-1"),
        severity.upper()
    ], color=config['color'], className="me-2", style={'fontSize': '0.75rem', 'fontWeight': '600'})


def create_impact_badge(impact_seconds: float) -> dbc.Badge:
    """Create impact badge showing time loss"""
    color = 'danger' if impact_seconds >= 0.4 else 'warning' if impact_seconds >= 0.2 else 'info'

    return dbc.Badge([
        html.I(className="fas fa-clock me-1"),
        f"{impact_seconds:.2f}s/lap"
    ], color=color, className="ms-2", style={'fontSize': '0.75rem'})


def create_dimension_breakdown(pattern: Dict) -> html.Div:
    """Create WHAT × WHERE × WHEN breakdown display"""

    what_metrics = pattern.get('what_metrics', [])
    where_corners = pattern.get('where_corners', [])
    when_laps = pattern.get('when_laps', [])

    return html.Div([
        # WHAT dimension
        html.Div([
            html.Strong([
                html.I(className="fas fa-chart-bar me-2", style={'color': '#3498db'}),
                "WHAT: "
            ]),
            html.Span(
                ', '.join(what_metrics) if len(what_metrics) <= 3 else f"{', '.join(what_metrics[:3])} +{len(what_metrics)-3} more",
                className="text-muted"
            )
        ], className="mb-2"),

        # WHERE dimension
        html.Div([
            html.Strong([
                html.I(className="fas fa-map-marker-alt me-2", style={'color': '#e74c3c'}),
                "WHERE: "
            ]),
            html.Span(
                f"{len(where_corners)} corners affected" if len(where_corners) > 3 else f"Corners {', '.join(map(str, where_corners))}",
                className="text-muted"
            )
        ], className="mb-2"),

        # WHEN dimension
        html.Div([
            html.Strong([
                html.I(className="fas fa-clock me-2", style={'color': '#2ecc71'}),
                "WHEN: "
            ]),
            html.Span(
                f"All {len(when_laps)} laps analyzed",
                className="text-muted"
            )
        ], className="mb-0"),
    ], style={
        'backgroundColor': '#f8f9fa',
        'padding': '1rem',
        'borderRadius': '0.25rem',
        'border': '1px solid #dee2e6'
    })


def create_pattern_card(pattern: Dict, index: int) -> dbc.Card:
    """
    Create individual pattern card with full details

    Args:
        pattern: Dictionary with pattern data
        index: Pattern number (1-based)

    Returns:
        Dash Bootstrap Card component
    """
    pattern_name = pattern.get('pattern_name', 'Unknown Pattern')
    severity = pattern.get('severity', 'Medium')
    impact = pattern.get('impact_seconds', 0.0)
    coaching = pattern.get('coaching', 'No coaching available')

    # Severity color for left border
    border_colors = {
        'High': '#dc3545',
        'Medium': '#ffc107',
        'Low': '#17a2b8'
    }
    border_color = border_colors.get(severity, '#6c757d')

    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.Div([
                    html.H6([
                        html.Span(f"#{index}", className="badge bg-dark me-2", style={'fontSize': '0.8rem'}),
                        html.Strong(pattern_name),
                        create_severity_badge(severity),
                        create_impact_badge(impact)
                    ], className="mb-0")
                ]),
            ])
        ], style={
            'backgroundColor': '#ffffff',
            'borderLeft': f'4px solid {border_color}'
        }),

        dbc.CardBody([
            # Dimension breakdown
            html.Div([
                html.H6([
                    html.I(className="fas fa-cube me-2"),
                    "Multi-Dimensional Analysis"
                ], className="mb-3", style={'fontSize': '0.9rem', 'color': '#6c757d'}),
                create_dimension_breakdown(pattern)
            ], className="mb-3"),

            # Coaching section
            html.Div([
                html.H6([
                    html.I(className="fas fa-lightbulb me-2", style={'color': '#f39c12'}),
                    "Coaching Recommendation"
                ], className="mb-2", style={'fontSize': '0.9rem'}),
                html.P(
                    coaching,
                    className="mb-0",
                    style={
                        'fontSize': '0.95rem',
                        'lineHeight': '1.6',
                        'color': '#2c3e50',
                        'fontStyle': 'italic',
                        'padding': '0.75rem',
                        'backgroundColor': '#fff9e6',
                        'borderLeft': '3px solid #f39c12',
                        'borderRadius': '0.25rem'
                    }
                )
            ])
        ])
    ], className="shadow-sm mb-3", style={'border': '1px solid #dee2e6'})


def create_patterns_summary_card(patterns: List[Dict]) -> dbc.Card:
    """
    Create summary card showing total impact

    Args:
        patterns: List of pattern dictionaries

    Returns:
        Summary card component
    """
    total_impact = sum(p.get('impact_seconds', 0) for p in patterns)
    high_severity = sum(1 for p in patterns if p.get('severity') == 'High')
    medium_severity = sum(1 for p in patterns if p.get('severity') == 'Medium')
    low_severity = sum(1 for p in patterns if p.get('severity') == 'Low')

    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-flag-checkered fa-2x mb-2", style={'color': '#3498db'}),
                        html.H3(f"{total_impact:.2f}s", className="mb-0"),
                        html.P("Total Time Savings", className="text-muted small mb-0")
                    ], className="text-center")
                ], md=3),

                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-exclamation-circle fa-2x mb-2", style={'color': '#dc3545'}),
                        html.H3(str(high_severity), className="mb-0"),
                        html.P("High Severity", className="text-muted small mb-0")
                    ], className="text-center")
                ], md=3),

                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle fa-2x mb-2", style={'color': '#ffc107'}),
                        html.H3(str(medium_severity), className="mb-0"),
                        html.P("Medium Severity", className="text-muted small mb-0")
                    ], className="text-center")
                ], md=3),

                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-info-circle fa-2x mb-2", style={'color': '#17a2b8'}),
                        html.H3(str(low_severity), className="mb-0"),
                        html.P("Low Severity", className="text-muted small mb-0")
                    ], className="text-center")
                ], md=3),
            ])
        ])
    ], className="shadow-sm mb-4", style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'color': 'white'
    })


def create_patterns_section(patterns: List[Dict]) -> html.Div:
    """
    Create complete patterns analysis section

    Args:
        patterns: List of detected pattern dictionaries

    Returns:
        Complete section with header, summary, and pattern cards
    """
    if not patterns or len(patterns) == 0:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                "No driving patterns detected. Upload telemetry data to analyze your driving."
            ], color="info")
        ])

    # Sort patterns by severity and impact
    severity_order = {'High': 0, 'Medium': 1, 'Low': 2}
    sorted_patterns = sorted(
        patterns,
        key=lambda p: (severity_order.get(p.get('severity', 'Medium'), 3), -p.get('impact_seconds', 0))
    )

    pattern_cards = [
        create_pattern_card(pattern, i+1)
        for i, pattern in enumerate(sorted_patterns)
    ]

    return html.Div([
        # Section header
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-brain me-2", style={'color': '#667eea'}),
                    f"Detected Driving Patterns ({len(patterns)} patterns found)"
                ], className="mb-0")
            ], style={'backgroundColor': '#f8f9fa'})
        ], className="mb-3"),

        # Summary card
        create_patterns_summary_card(sorted_patterns),

        # Info banner
        dbc.Alert([
            html.I(className="fas fa-info-circle me-2"),
            html.Strong("Multi-Dimensional Analysis: "),
            "Each pattern shows WHAT metrics are affected, WHERE on track it occurs, and WHEN during your session. ",
            "Focus on high-severity patterns first for maximum lap time improvement."
        ], color="info", className="mb-3"),

        # Pattern cards
        html.Div(pattern_cards),

        # Footer with total
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-calculator me-2"),
                    html.Strong("Total Potential Improvement: "),
                    html.Span(
                        f"{sum(p.get('impact_seconds', 0) for p in patterns):.2f} seconds per lap",
                        style={'fontSize': '1.1rem', 'color': '#2ecc71', 'fontWeight': 'bold'}
                    )
                ], className="text-center")
            ], style={'padding': '1rem', 'backgroundColor': '#f8f9fa'})
        ], className="shadow-sm")
    ])


def create_patterns_from_analysis_result(analysis_data: Dict) -> html.Div:
    """
    Create patterns section from cube analysis results

    Args:
        analysis_data: Dictionary from CUBE_ANALYSIS_DEMO_DATA.json

    Returns:
        Complete patterns section
    """
    patterns = analysis_data.get('patterns', [])

    if not patterns:
        logger.warning("No patterns found in analysis data")
        return html.Div([
            dbc.Alert("No patterns detected in analysis", color="warning")
        ])

    return create_patterns_section(patterns)


# ============================================================================
# DEMO DATA (for testing without cube analysis)
# ============================================================================

DEMO_PATTERNS = [
    {
        'pattern_name': 'Underutilizing Brake Pressure',
        'severity': 'High',
        'impact_seconds': 0.50,
        'what_metrics': ['pbrake_f', 'pbrake_r'],
        'where_corners': [1, 2, 3, 4, 5],
        'when_laps': list(range(1, 11)),
        'coaching': "You're only using an average of 13.1 bar when you've shown you can brake at 163.2 bar. Commit to later, harder braking in slow corners to reduce lap times."
    },
    {
        'pattern_name': 'Conservative Throttle Application',
        'severity': 'Medium',
        'impact_seconds': 0.30,
        'what_metrics': ['aps'],
        'where_corners': [2, 3, 5],
        'when_laps': list(range(1, 11)),
        'coaching': "You're at full throttle only 15.6% of the time. Work on earlier, more aggressive throttle application on corner exits to carry more speed down straights."
    },
    {
        'pattern_name': 'Speed Inconsistency',
        'severity': 'Medium',
        'impact_seconds': 0.20,
        'what_metrics': ['speed'],
        'where_corners': [1, 2, 3, 4, 5],
        'when_laps': list(range(1, 11)),
        'coaching': "Speed variation is 20.1 km/h between laps. Focus on consistent brake points and throttle application to improve lap time consistency and tire management."
    },
    {
        'pattern_name': 'Aggressive Steering Inputs',
        'severity': 'Low',
        'impact_seconds': 0.10,
        'what_metrics': ['Steering_Angle'],
        'where_corners': [1, 3, 4],
        'when_laps': list(range(1, 11)),
        'coaching': "Average steering change is 13.1°. Smoother inputs will improve tire grip and consistency through corners. Practice progressive steering application."
    }
]


def create_demo_patterns_section() -> html.Div:
    """
    Create patterns section with demo data for testing

    Returns:
        Patterns section with hardcoded demo patterns
    """
    return create_patterns_section(DEMO_PATTERNS)


if __name__ == '__main__':
    # Test the widget components
    print("Pattern Analysis Widget - Component Test")
    print("=" * 60)

    print("\nDemo patterns loaded:")
    for i, pattern in enumerate(DEMO_PATTERNS, 1):
        print(f"  {i}. {pattern['pattern_name']} - {pattern['severity']} ({pattern['impact_seconds']}s)")

    total = sum(p['impact_seconds'] for p in DEMO_PATTERNS)
    print(f"\nTotal impact: {total:.2f} seconds/lap")
    print("\nWidget components ready for integration!")
