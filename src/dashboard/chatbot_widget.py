"""
GR Cup Racing Dashboard Chatbot Widget
========================================

Interactive chatbot widget powered by Google Gemini AI for natural language
queries about telemetry data, performance metrics, and coaching insights.

Features:
- Natural language query interface
- Real-time responses from Gemini API
- Context-aware answers using dashboard data
- Racing-specific query validation (200+ keywords)
- Smart caching for performance
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import requests
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Backend API configuration - Auto-detect environment
# In production, use same host as dashboard on port 3000
# In development, use localhost:3000
CHATBOT_API_BASE = os.getenv("CHATBOT_API_URL", "http://localhost:3000")
CHATBOT_API_URL = f"{CHATBOT_API_BASE}/api/chat"
CHATBOT_HEALTH_URL = f"{CHATBOT_API_BASE}/api/health"


def create_chatbot_layout():
    """
    Create the chatbot widget layout with improved typography and draggable/resizable features

    Returns:
        dbc.Card: Chatbot interface card
    """
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.I(className="fas fa-robot me-2"),
                html.Span("AI Racing Assistant", className="fw-bold", style={'fontSize': '1.1rem'}),
                html.Span(
                    id="chatbot-status-indicator",
                    className="badge bg-secondary ms-2",
                    children="Connecting..."
                ),
                html.I(className="fas fa-arrows-alt ms-auto", title="Drag to move",
                       style={'cursor': 'move', 'opacity': '0.6'}, id="chatbot-drag-handle")
            ], className="d-flex align-items-center", style={'padding': '0.75rem 1rem'})
        ]),
        dbc.CardBody([
            # Chat messages container
            html.Div(
                id="chatbot-messages",
                className="chatbot-messages-container mb-3",
                children=[
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-robot me-2"),
                            html.Span("AI Assistant", className="fw-bold text-primary")
                        ], className="mb-1"),
                        html.P([
                            "Hello! I'm your GR Cup Racing AI assistant. ",
                            "Ask me about your telemetry data, performance metrics, or request coaching advice!"
                        ], className="mb-1"),
                        html.P([
                            html.Strong("Try asking:"),
                            html.Br(),
                            "• What was my best lap time?",
                            html.Br(),
                            "• Compare braking between lap 3 and lap 4",
                            html.Br(),
                            "• How can I improve my corner exit speed?"
                        ], className="small text-muted mb-0")
                    ], className="message assistant-message")
                ],
                style={
                    "height": "400px",
                    "overflowY": "auto",
                    "border": "1px solid #dee2e6",
                    "borderRadius": "0.25rem",
                    "padding": "1rem",
                    "backgroundColor": "#f8f9fa"
                }
            ),

            # Input form
            dbc.Row([
                dbc.Col([
                    dbc.Input(
                        id="chatbot-input",
                        type="text",
                        placeholder="Ask me anything about your racing data...",
                        className="mb-2"
                    )
                ], width=10),
                dbc.Col([
                    dbc.Button(
                        html.I(className="fas fa-paper-plane"),
                        id="chatbot-send-button",
                        color="primary",
                        className="w-100"
                    )
                ], width=2)
            ]),

            # Example queries
            html.Div([
                html.Small("Quick queries: ", className="text-muted me-2"),
                dbc.ButtonGroup([
                    dbc.Button(
                        "Best Lap",
                        id="chatbot-query-best-lap",
                        size="sm",
                        color="link",
                        className="text-decoration-none"
                    ),
                    dbc.Button(
                        "Throttle Analysis",
                        id="chatbot-query-throttle",
                        size="sm",
                        color="link",
                        className="text-decoration-none"
                    ),
                    dbc.Button(
                        "Coaching Tips",
                        id="chatbot-query-coaching",
                        size="sm",
                        color="link",
                        className="text-decoration-none"
                    )
                ], size="sm", className="flex-wrap")
            ], className="mt-2")
        ]),

        # Hidden stores
        dcc.Store(id="chatbot-conversation-history", data=[]),
        dcc.Interval(id="chatbot-health-check", interval=30000, n_intervals=0)  # Check every 30s
    ], className="mb-3", id="chatbot-card")


def create_chatbot_callbacks(app):
    """
    Register chatbot callbacks

    Args:
        app: Dash application instance
    """

    # Simplified health check callback (no model info)
    @app.callback(
        [Output("chatbot-status-indicator", "children"),
         Output("chatbot-status-indicator", "className")],
        Input("chatbot-health-check", "n_intervals")
    )
    def update_health_status(n):
        """Check backend API health"""
        try:
            response = requests.get(CHATBOT_HEALTH_URL, timeout=5)
            if response.status_code == 200:
                health = response.json()
                if health.get("status") == "healthy":
                    return "Connected", "badge bg-success ms-2"
                else:
                    return "Degraded", "badge bg-warning ms-2"
        except Exception as e:
            logger.error(f"Health check failed: {e}")

        return "Offline", "badge bg-danger ms-2"

    # Send message callback
    @app.callback(
        Output("chatbot-messages", "children"),
        Output("chatbot-input", "value"),
        Output("chatbot-conversation-history", "data"),
        Input("chatbot-send-button", "n_clicks"),
        Input("chatbot-query-best-lap", "n_clicks"),
        Input("chatbot-query-throttle", "n_clicks"),
        Input("chatbot-query-coaching", "n_clicks"),
        Input("chatbot-input", "n_submit"),
        State("chatbot-input", "value"),
        State("chatbot-messages", "children"),
        State("chatbot-conversation-history", "data"),
        State("vehicle-dropdown", "value"),
        State("tabs", "active_tab"),
        prevent_initial_call=True
    )
    def handle_chat_message(send_clicks, best_lap_clicks, throttle_clicks, coaching_clicks,
                           n_submit, user_input, current_messages, conversation_history,
                           vehicle_number, active_tab):
        """Handle user messages and API responses"""

        # Determine which input triggered the callback
        ctx = callback_context
        if not ctx.triggered:
            return current_messages, "", conversation_history

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Set query based on trigger
        if trigger_id == "chatbot-query-best-lap":
            query = "What was my best lap time?"
        elif trigger_id == "chatbot-query-throttle":
            query = "Analyze my throttle usage patterns"
        elif trigger_id == "chatbot-query-coaching":
            query = "How can I improve my corner exit speed?"
        else:
            query = user_input

        # Validate query
        if not query or not query.strip():
            return current_messages, "", conversation_history

        # Use vehicle number from dropdown (default to 2 if not selected)
        if not vehicle_number:
            vehicle_number = 2

        # Track is auto-detected by backend from CSV data
        track = None

        # Extract tab name from active_tab (e.g., "tab-insights" -> "insights")
        tab = None
        if active_tab and active_tab.startswith("tab-"):
            tab = active_tab.replace("tab-", "")

        # Add user message to UI
        timestamp = datetime.now().strftime("%H:%M")
        user_message = html.Div([
            html.Div([
                html.I(className="fas fa-user me-2"),
                html.Span("You", className="fw-bold text-secondary"),
                html.Span(f" · {timestamp}", className="small text-muted ms-2")
            ], className="mb-1"),
            html.P(query, className="mb-0")
        ], className="message user-message")

        # Show loading indicator
        loading_message = html.Div([
            html.Div([
                html.I(className="fas fa-robot me-2"),
                html.Span("AI Assistant", className="fw-bold text-primary"),
            ], className="mb-1"),
            html.Div([
                dbc.Spinner(size="sm", color="primary", spinner_class_name="me-2"),
                html.Span("Thinking...", className="text-muted")
            ])
        ], className="message assistant-message")

        updated_messages = current_messages + [user_message, loading_message]

        # Call chatbot API
        try:
            payload = {
                "query": query,
                "vehicle_number": vehicle_number
            }
            if track:
                payload["track"] = track
            if tab:
                payload["tab"] = tab

            response = requests.post(
                CHATBOT_API_URL,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No response received.")
                metadata = data.get("metadata", {})

                # Format response with metadata
                intent_badge = ""
                if "intent" in metadata:
                    intent_colors = {
                        "fact_retrieval": "info",
                        "comparison": "primary",
                        "analysis": "success",
                        "coaching": "warning",
                        "navigation": "secondary"
                    }
                    intent = metadata["intent"]
                    color = intent_colors.get(intent, "secondary")
                    intent_badge = html.Span(
                        intent.replace("_", " ").title(),
                        className=f"badge bg-{color} ms-2"
                    )

                # Create assistant response
                assistant_message = html.Div([
                    html.Div([
                        html.I(className="fas fa-robot me-2"),
                        html.Span("AI Assistant", className="fw-bold text-primary"),
                        intent_badge,
                        html.Span(f" · {timestamp}", className="small text-muted ms-2")
                    ], className="mb-2"),
                    dcc.Markdown(answer, className="mb-0")
                ], className="message assistant-message")

                # Remove loading, add response
                updated_messages = current_messages + [user_message, assistant_message]

                # Update conversation history
                if not conversation_history:
                    conversation_history = []
                conversation_history.append({
                    "role": "user",
                    "content": query,
                    "timestamp": timestamp
                })
                conversation_history.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": metadata,
                    "timestamp": timestamp
                })

            else:
                error_message = html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle me-2 text-danger"),
                        html.Span("Error", className="fw-bold text-danger")
                    ], className="mb-1"),
                    html.P(
                        f"Sorry, I encountered an error (Status {response.status_code}). Please try again.",
                        className="mb-0"
                    )
                ], className="message assistant-message")

                updated_messages = current_messages + [user_message, error_message]

        except requests.exceptions.Timeout:
            timeout_message = html.Div([
                html.Div([
                    html.I(className="fas fa-clock me-2 text-warning"),
                    html.Span("Timeout", className="fw-bold text-warning")
                ], className="mb-1"),
                html.P(
                    "The request timed out. The backend may be processing a complex query. Please try again.",
                    className="mb-0"
                )
            ], className="message assistant-message")

            updated_messages = current_messages + [user_message, timeout_message]

        except Exception as e:
            logger.error(f"Chatbot error: {e}")
            error_message = html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle me-2 text-danger"),
                    html.Span("Error", className="fw-bold text-danger")
                ], className="mb-1"),
                html.P(
                    "Sorry, I'm having trouble connecting to the backend. Please ensure the chatbot API is running.",
                    className="mb-0"
                )
            ], className="message assistant-message")

            updated_messages = current_messages + [user_message, error_message]

        return updated_messages, "", conversation_history
