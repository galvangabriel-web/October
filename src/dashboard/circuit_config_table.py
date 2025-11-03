"""
Circuit Configuration Table Component
======================================

Creates a beautiful Bootstrap table displaying track circuit configuration data.
"""

import dash_bootstrap_components as dbc
from dash import html
from typing import Dict, Optional


def create_circuit_config_table(track_metadata: Dict) -> html.Div:
    """
    Create a formatted circuit configuration table from track metadata

    Args:
        track_metadata: Dictionary containing track information

    Returns:
        Dash HTML Div containing the formatted table
    """
    if not track_metadata:
        return html.Div([
            dbc.Alert("No track data available", color="warning")
        ])

    # Extract key information
    name = track_metadata.get('name', 'Unknown Track')
    location = track_metadata.get('location', 'Unknown Location')
    length_miles = track_metadata.get('length_miles', 0)
    length_meters = track_metadata.get('length_meters', 0)
    elevation_ft = track_metadata.get('elevation_ft', 0)
    elevation_m = track_metadata.get('elevation_m', 0)

    gps_finish = track_metadata.get('gps_finish', {})
    gps_pit_in = track_metadata.get('gps_pit_in', {})
    gps_pit_out = track_metadata.get('gps_pit_out', {})

    sectors = track_metadata.get('sectors', {})
    turns = track_metadata.get('turns', {})
    pit_lane = track_metadata.get('pit_lane', {})
    speed_trap = track_metadata.get('speed_trap', {})

    # Create the table
    return html.Div([
        # Header
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-info-circle me-2"),
                    "Circuit Configuration"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                # Track Specifications Table
                html.H6([
                    html.I(className="fas fa-ruler me-2"),
                    "Track Specifications"
                ], className="mb-3"),
                dbc.Table([
                    html.Tbody([
                        html.Tr([
                            html.Td("Circuit Name", className="fw-bold", style={'width': '30%'}),
                            html.Td(name, colSpan=3)
                        ]),
                        html.Tr([
                            html.Td("Location", className="fw-bold"),
                            html.Td(location, colSpan=3)
                        ]),
                        html.Tr([
                            html.Td("Circuit Length", className="fw-bold"),
                            html.Td(f"{length_miles:.3f} Miles"),
                            html.Td(f"{length_meters:.1f} Meters", colSpan=2)
                        ]),
                        html.Tr([
                            html.Td("Elevation @ Finish Line", className="fw-bold"),
                            html.Td(f"{elevation_ft}' / {elevation_m}m", colSpan=3)
                        ]),
                    ])
                ], bordered=True, hover=True, striped=True, size="sm", className="mb-4"),

                # GPS Coordinates Table
                html.H6([
                    html.I(className="fas fa-map-marker-alt me-2"),
                    "GPS Coordinates"
                ], className="mb-3"),
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Location", style={'width': '30%'}),
                            html.Th("Latitude"),
                            html.Th("Longitude")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td("Finish Line & Timing", className="fw-bold"),
                            html.Td(f"{gps_finish.get('lat', 0):.7f}° N"),
                            html.Td(f"{gps_finish.get('lon', 0):.7f}° W" if gps_finish.get('lon', 0) < 0 else f"{gps_finish.get('lon', 0):.7f}° E")
                        ]),
                        html.Tr([
                            html.Td("GPS Pit In", className="fw-bold"),
                            html.Td(f"{gps_pit_in.get('lat', 0):.7f}° N"),
                            html.Td(f"{gps_pit_in.get('lon', 0):.7f}° W" if gps_pit_in.get('lon', 0) < 0 else f"{gps_pit_in.get('lon', 0):.7f}° E")
                        ]),
                        html.Tr([
                            html.Td("GPS Pit Out", className="fw-bold"),
                            html.Td(f"{gps_pit_out.get('lat', 0):.7f}° N"),
                            html.Td(f"{gps_pit_out.get('lon', 0):.7f}° W" if gps_pit_out.get('lon', 0) < 0 else f"{gps_pit_out.get('lon', 0):.7f}° E")
                        ]),
                    ])
                ], bordered=True, hover=True, striped=True, size="sm", className="mb-4"),

                # Sector Information
                html.H6([
                    html.I(className="fas fa-flag-checkered me-2"),
                    "Sector Information"
                ], className="mb-3"),
                dbc.Row([
                    # Sector lengths
                    dbc.Col([
                        dbc.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Sector"),
                                    html.Th("Length (m)"),
                                    html.Th("Length (mi)")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([
                                    html.Td(f"Sector {i+1} ({sector_name})", className="fw-bold"),
                                    html.Td(f"{sector_data.get('length_m', 0):.1f}m"),
                                    html.Td(f"{sector_data.get('length_m', 0) * 0.000621371:.3f}mi")
                                ]) for i, (sector_name, sector_data) in enumerate(sorted(sectors.items()))
                            ])
                        ], bordered=True, hover=True, size="sm")
                    ], md=6),

                    # Turn count per sector
                    dbc.Col([
                        dbc.Table([
                            html.Thead([
                                html.Tr([
                                    html.Th("Sector"),
                                    html.Th("Turns"),
                                    html.Th("Count")
                                ])
                            ]),
                            html.Tbody([
                                html.Tr([
                                    html.Td(f"Sector {i+1}", className="fw-bold"),
                                    html.Td(", ".join([
                                        str(turn_info.get('number', turn_name.replace('T', '')))
                                        for turn_name, turn_info in sorted(turns.items())
                                        if turn_info.get('sector') == sector_name
                                    ]) or "N/A", style={'fontSize': '0.85em'}),
                                    html.Td(str(len([
                                        t for t, ti in turns.items()
                                        if ti.get('sector') == sector_name
                                    ])))
                                ]) for i, sector_name in enumerate(['S1', 'S2', 'S3'])
                            ])
                        ], bordered=True, hover=True, size="sm")
                    ], md=6)
                ], className="mb-4"),

                # Pit Lane & Speed Trap
                dbc.Row([
                    dbc.Col([
                        html.H6([
                            html.I(className="fas fa-car me-2"),
                            "Pit Lane"
                        ], className="mb-3"),
                        dbc.Table([
                            html.Tbody([
                                html.Tr([
                                    html.Td("Time through pit lane @ 50 kph", className="fw-bold"),
                                    html.Td(f"{pit_lane.get('time_seconds', 0)} Seconds")
                                ]),
                                html.Tr([
                                    html.Td("Pit In from S/F", className="fw-bold"),
                                    html.Td(f"{pit_lane.get('pit_in_from_sf_inches', 0):,}\"" if pit_lane.get('pit_in_from_sf_inches', 0) >= 0
                                            else f"{abs(pit_lane.get('pit_in_from_sf_inches', 0)):,}\" (before S/F)")
                                ]),
                                html.Tr([
                                    html.Td("Pit In to Pit Out", className="fw-bold"),
                                    html.Td(f"{pit_lane.get('pit_in_to_out_inches', 0):,}\"")
                                ]),
                            ])
                        ], bordered=True, hover=True, striped=True, size="sm")
                    ], md=6),

                    dbc.Col([
                        html.H6([
                            html.I(className="fas fa-tachometer-alt me-2"),
                            "Speed Trap"
                        ], className="mb-3"),
                        dbc.Table([
                            html.Tbody([
                                html.Tr([
                                    html.Td("Location from S/F", className="fw-bold"),
                                    html.Td(f"{speed_trap.get('location_inches', 0):,}\"" if speed_trap else "N/A")
                                ]),
                                html.Tr([
                                    html.Td("Location (meters)", className="fw-bold"),
                                    html.Td(f"{speed_trap.get('location_m', 0):.1f}m" if speed_trap else "N/A")
                                ]),
                                html.Tr([
                                    html.Td("Total Turns", className="fw-bold"),
                                    html.Td(f"{len(turns)} Turns")
                                ]),
                            ])
                        ], bordered=True, hover=True, striped=True, size="sm")
                    ], md=6)
                ])
            ])
        ], className="shadow-sm")
    ])


def create_circuit_summary_badge(track_metadata: Dict) -> html.Div:
    """
    Create a compact summary badge for quick reference

    Args:
        track_metadata: Dictionary containing track information

    Returns:
        Compact badge with key track info
    """
    if not track_metadata:
        return html.Div()

    length_miles = track_metadata.get('length_miles', 0)
    elevation_ft = track_metadata.get('elevation_ft', 0)
    turns_count = len(track_metadata.get('turns', {}))
    pit_time = track_metadata.get('pit_lane', {}).get('time_seconds', 0)

    return dbc.Alert([
        html.Strong([
            html.I(className="fas fa-info-circle me-2"),
            "Quick Stats: "
        ]),
        html.Span(f"{length_miles:.2f} mi", className="me-3"),
        html.Span("•", className="me-3"),
        html.Span(f"{elevation_ft}' elevation", className="me-3"),
        html.Span("•", className="me-3"),
        html.Span(f"{turns_count} turns", className="me-3"),
        html.Span("•", className="me-3"),
        html.Span(f"{pit_time}s pit lane")
    ], color="info", className="mb-3")
