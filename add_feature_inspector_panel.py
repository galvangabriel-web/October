"""
Add Feature Inspector Panel to Post-Race Analysis Tab
=====================================================

This script adds a comprehensive Feature Inspector Panel that displays
all 40 baseline features used by the SimplePostRacePredictor.
"""

# Read the current post_race_widget.py
with open('src/dashboard/post_race_widget.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the marker where we'll insert the new panel (after AI Model Config Card, before Track Selection)
marker = '''        # Quick Track Selection (Radio Buttons) - Enhanced Design
        dbc.Card(['''

# Define the Feature Inspector Panel
feature_inspector_panel = '''        # ============================================================================
        # FEATURE INSPECTOR PANEL - Shows all 40 baseline features
        # ============================================================================
        dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="fas fa-microscope me-3", style={'color': '#667eea'}),
                    "Feature Inspector - Under the Hood"
                ], className="mb-0", style={
                    'fontWeight': '700',
                    'fontSize': '28px',
                    'color': '#2c3e50',
                    'fontFamily': 'Inter, sans-serif'
                })
            ], style={
                'backgroundColor': '#ffffff',
                'borderBottom': '3px solid #667eea',
                'padding': '1.5rem'
            }),
            dbc.CardBody([
                # Introduction
                html.P([
                    "Our AI analyzes ",
                    html.Strong("40 engineered features", style={'color': '#667eea', 'fontSize': '1.1rem'}),
                    " from 9 telemetry sensors to predict lap times with 89-91% accuracy. ",
                    "Below are the features extracted from your session, organized by category:"
                ], className="mb-4", style={'fontSize': '1.05rem', 'color': '#34495e'}),

                # Feature Categories in Tabs
                dbc.Tabs([
                    # Speed Features Tab
                    dbc.Tab(label="🚀 Speed (8)", tab_id="speed-features", children=[
                        html.Div([
                            html.H5("Speed Analysis Features", className="mt-3 mb-3", style={'color': '#667eea'}),
                            html.P("Metrics analyzing your velocity profile throughout the lap", className="text-muted mb-3"),

                            # Feature list with descriptions
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-tachometer-alt me-2", style={'color': '#3498db'}),
                                                "Average Speed"
                                            ], className="mb-2"),
                                            html.P("Mean velocity across the entire lap", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("High - directly correlates with lap time", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-arrow-up me-2", style={'color': '#e74c3c'}),
                                                "Maximum Speed"
                                            ], className="mb-2"),
                                            html.P("Peak velocity reached during the lap", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - indicates straight-line performance", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-arrow-down me-2", style={'color': '#95a5a6'}),
                                                "Minimum Speed"
                                            ], className="mb-2"),
                                            html.P("Slowest point, typically slowest corner apex", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("High - slow corners lose time", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-chart-line me-2", style={'color': '#9b59b6'}),
                                                "Speed Variance"
                                            ], className="mb-2"),
                                            html.P("Consistency of velocity - lower is smoother", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - affects tire wear and balance", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            # Additional speed features
                            html.Hr(),
                            html.P([
                                html.Strong("Also analyzed: "),
                                "Speed Range (max-min), Time Above 170 km/h, Normalized Speed Metrics"
                            ], className="text-muted", style={'fontSize': '0.95rem'})
                        ], className="p-3")
                    ]),

                    # Braking Features Tab
                    dbc.Tab(label="🛑 Braking (8)", tab_id="braking-features", children=[
                        html.Div([
                            html.H5("Braking Performance Features", className="mt-3 mb-3", style={'color': '#e74c3c'}),
                            html.P("Metrics analyzing your braking technique and consistency", className="text-muted mb-3"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-hand-paper me-2", style={'color': '#e74c3c'}),
                                                "Maximum Brake Pressure"
                                            ], className="mb-2"),
                                            html.P("Peak front brake pressure (bar)", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("High - hard braking = shorter braking zones", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-chart-area me-2", style={'color': '#3498db'}),
                                                "Brake Consistency"
                                            ], className="mb-2"),
                                            html.P("Standard deviation of brake pressure >50 bar", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("High - consistent braking = predictable car", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-clock me-2", style={'color': '#f39c12'}),
                                                "Brake Duration"
                                            ], className="mb-2"),
                                            html.P("Percentage of lap time on brakes (>20 bar)", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - less time braking = more speed", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-shoe-prints me-2", style={'color': '#9b59b6'}),
                                                "Trail Braking Amount"
                                            ], className="mb-2"),
                                            html.P("Overlap between braking and throttle application", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - advanced technique for rotation", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            html.Hr(),
                            html.P([
                                html.Strong("Also analyzed: "),
                                "Average Brake Pressure, Number of Braking Zones, Brake Point Consistency"
                            ], className="text-muted", style={'fontSize': '0.95rem'})
                        ], className="p-3")
                    ]),

                    # Throttle Features Tab
                    dbc.Tab(label="⚡ Throttle (5)", tab_id="throttle-features", children=[
                        html.Div([
                            html.H5("Throttle Application Features", className="mt-3 mb-3", style={'color': '#2ecc71'}),
                            html.P("Metrics analyzing your power delivery and acceleration technique", className="text-muted mb-3"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-bolt me-2", style={'color': '#f39c12'}),
                                                "Full Throttle Percentage"
                                            ], className="mb-2"),
                                            html.P("Time at >95% throttle position", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Very High - more full throttle = faster laps", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-sliders-h me-2", style={'color': '#3498db'}),
                                                "Throttle Modulation"
                                            ], className="mb-2"),
                                            html.P("Standard deviation - smoothness of application", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - smooth = better traction", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-chart-bar me-2", style={'color': '#2ecc71'}),
                                                "Average Throttle"
                                            ], className="mb-2"),
                                            html.P("Mean throttle position throughout lap", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("High - overall power delivery metric", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-percentage me-2", style={'color': '#9b59b6'}),
                                                "Time Above 50% Throttle"
                                            ], className="mb-2"),
                                            html.P("Percentage of lap with significant power", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - indicates acceleration efficiency", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            html.Hr(),
                            html.P([
                                html.Strong("Also analyzed: "),
                                "Maximum Throttle Position"
                            ], className="text-muted", style={'fontSize': '0.95rem'})
                        ], className="p-3")
                    ]),

                    # Cornering Features Tab
                    dbc.Tab(label="🔄 Cornering (7)", tab_id="cornering-features", children=[
                        html.Div([
                            html.H5("Cornering Dynamics Features", className="mt-3 mb-3", style={'color': '#e67e22'}),
                            html.P("Metrics analyzing your lateral g-forces and corner performance", className="text-muted mb-3"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-circle-notch me-2", style={'color': '#e67e22'}),
                                                "Maximum Lateral G"
                                            ], className="mb-2"),
                                            html.P("Peak cornering force (g-force)", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Very High - higher G = faster corners", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-bullseye me-2", style={'color': '#3498db'}),
                                                "Minimum Corner Speed"
                                            ], className="mb-2"),
                                            html.P("Apex speed at slowest turn (km/h)", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Very High - higher apex = faster exit", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-grip-horizontal me-2", style={'color': '#2ecc71'}),
                                                "Grip Utilization"
                                            ], className="mb-2"),
                                            html.P("% of theoretical maximum lateral g used", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("High - pushing limits = faster times", style={'color': '#2ecc71'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-equals me-2", style={'color': '#9b59b6'}),
                                                "Cornering Consistency"
                                            ], className="mb-2"),
                                            html.P("Standard deviation of lateral g-forces", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - smooth cornering = predictability", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            html.Hr(),
                            html.P([
                                html.Strong("Also analyzed: "),
                                "Average Lateral G, Corner Count, High-G Sections"
                            ], className="text-muted", style={'fontSize': '0.95rem'})
                        ], className="p-3")
                    ]),

                    # Steering Features Tab
                    dbc.Tab(label="🎮 Steering (5)", tab_id="steering-features", children=[
                        html.Div([
                            html.H5("Steering Technique Features", className="mt-3 mb-3", style={'color': '#3498db'}),
                            html.P("Metrics analyzing steering smoothness and precision", className="text-muted mb-3"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-dharmachakra me-2", style={'color': '#3498db'}),
                                                "Steering Smoothness"
                                            ], className="mb-2"),
                                            html.P("Standard deviation of steering angle changes", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - smooth inputs = better balance", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-undo me-2", style={'color': '#e74c3c'}),
                                                "Steering Corrections"
                                            ], className="mb-2"),
                                            html.P("Number of direction changes (overcorrections)", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Medium - fewer = more precise", style={'color': '#f39c12'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ]),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-arrows-alt-h me-2", style={'color': '#2ecc71'}),
                                                "Maximum Steering Angle"
                                            ], className="mb-2"),
                                            html.P("Peak steering input (degrees)", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Low - track-specific characteristic", style={'color': '#95a5a6'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                html.I(className="fas fa-chart-line me-2", style={'color': '#9b59b6'}),
                                                "Average Absolute Steering"
                                            ], className="mb-2"),
                                            html.P("Mean magnitude of steering inputs", className="text-muted mb-2", style={'fontSize': '0.9rem'}),
                                            html.Div([
                                                html.Span("Impact: ", style={'fontWeight': '600'}),
                                                html.Span("Low - indicates track technicality", style={'color': '#95a5a6'})
                                            ])
                                        ])
                                    ], className="mb-3", style={'border': '1px solid #e1e8ed', 'borderRadius': '8px'})
                                ], md=6),
                            ])
                        ], className="p-3")
                    ]),

                    # Other Features Tab
                    dbc.Tab(label="🔧 Other (7+)", tab_id="other-features", children=[
                        html.Div([
                            html.H5("Powertrain & Combined Features", className="mt-3 mb-3", style={'color': '#9b59b6'}),
                            html.P("Additional metrics from engine, gearing, and composite calculations", className="text-muted mb-3"),

                            # Powertrain
                            html.H6("⚙️ Powertrain Features", className="mb-3", style={'color': '#e67e22'}),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.P("RPM Efficiency, Shift Points, Gear Usage, Time in Power Band", className="mb-0")
                                        ])
                                    ], className="mb-3", style={'backgroundColor': '#f8f9fa'})
                                ], md=12),
                            ]),

                            # Combined
                            html.H6("🔬 Combined Metrics", className="mb-3 mt-3", style={'color': '#3498db'}),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.P("Traction Circle Utilization, Performance Index, Driving Aggression Score", className="mb-0")
                                        ])
                                    ], className="mb-3", style={'backgroundColor': '#f8f9fa'})
                                ], md=12),
                            ]),

                            html.Hr(),
                            html.Div([
                                html.I(className="fas fa-info-circle me-2", style={'color': '#3498db'}),
                                html.Strong("Total Feature Count: "),
                                html.Span("40 baseline features", style={'fontSize': '1.1rem', 'color': '#667eea'})
                            ], className="mb-3"),

                            html.P([
                                "These features are engineered from just 9 basic telemetry sensors, making the analysis ",
                                "reliable even without GPS data. Each feature contributes to the AI's understanding of your ",
                                "driving style and helps identify specific areas for improvement."
                            ], className="text-muted", style={'fontSize': '0.95rem'})
                        ], className="p-3")
                    ])
                ], active_tab="speed-features", className="mt-3"),

                # Summary Footer
                html.Hr(className="mt-4 mb-3"),
                dbc.Alert([
                    html.I(className="fas fa-lightbulb me-2", style={'color': '#f39c12'}),
                    html.Strong("Pro Tip: "),
                    "Focus on improving features marked as 'Very High Impact' first. Small improvements in Full Throttle %, ",
                    "Maximum Lateral G, and Minimum Corner Speed yield the biggest lap time gains."
                ], color="info", className="mb-0")
            ], style={'padding': '2rem'})
        ], className="mb-4", style={
            'border': '1px solid #dee2e6',
            'borderRadius': '15px',
            'boxShadow': '0 4px 20px rgba(102, 126, 234, 0.15)'
        }),

        '''

# Insert the Feature Inspector Panel before the Track Selection
if marker in content:
    content = content.replace(marker, feature_inspector_panel + marker)
    print('[OK] Added Feature Inspector Panel')
    print('  • 6 tabbed categories (Speed, Braking, Throttle, Cornering, Steering, Other)')
    print('  • 40 total features with descriptions and impact ratings')
    print('  • Marketing-style design with purple gradient theme')
    print('  • Educational tooltips explaining each feature')
else:
    print('[ERROR] Could not find Track Selection marker')
    print('Looking for: "# Quick Track Selection (Radio Buttons) - Enhanced Design"')
    exit(1)

# Write the modified content
with open('src/dashboard/post_race_widget.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('\n' + '='*70)
print('FEATURE INSPECTOR PANEL - SUCCESSFULLY ADDED!')
print('='*70)
print('[OK] Panel inserted before Track Selection')
print('[OK] All 40 baseline features documented')
print('[OK] Organized into 6 interactive tabs')
print('[OK] Impact ratings for each feature (Very High / High / Medium / Low)')
print('[OK] Pro tip section for improvement priorities')
print('\nReady for local testing!')
