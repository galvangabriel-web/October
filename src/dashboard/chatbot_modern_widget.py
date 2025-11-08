"""
Modern GR Cup Racing Dashboard Chatbot Widget
=============================================

Beautiful glass-morphism design with gradient effects, smooth animations,
and enhanced drag-and-drop functionality.

Features:
- Glass morphism design with gradients
- Chat bubble style messages
- User and AI avatars
- Smooth animations and transitions
- Minimize/maximize/close controls
- Edge snapping and position memory
- Typing indicators
- Quick action chips
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
import requests
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Backend API configuration
CHATBOT_API_BASE = os.getenv("CHATBOT_API_URL", "http://localhost:3000")
CHATBOT_API_URL = f"{CHATBOT_API_BASE}/api/chat"
CHATBOT_HEALTH_URL = f"{CHATBOT_API_BASE}/api/health"


def create_modern_chatbot_layout():
    """
    Create the modern chatbot widget layout with glass-morphism design

    Returns:
        html.Div: Modern chatbot interface container
    """
    return html.Div([
        # Floating container
        html.Div([
            # Glass morphism card
            html.Div([
                # Beautiful gradient header
                html.Div([
                    # Left side - Avatar and info
                    html.Div([
                        # AI Avatar
                        html.Div([
                            html.I(className="fas fa-robot")
                        ], className="chatbot-ai-avatar"),

                        # Header info
                        html.Div([
                            html.H3("AI Racing Assistant", className="chatbot-header-title"),
                            html.Div([
                                html.Span(id="chatbot-status-dot", className="chatbot-status-dot"),
                                html.Span("Online", id="chatbot-status-label", className="chatbot-status-text")
                            ])
                        ], className="chatbot-header-info")
                    ], className="chatbot-header-left"),

                    # Right side - Window controls
                    html.Div([
                        html.Button([
                            html.I(className="fas fa-minus")
                        ], className="chatbot-control-btn chatbot-minimize-btn", title="Minimize"),

                        html.Button([
                            html.I(className="fas fa-times")
                        ], className="chatbot-control-btn chatbot-close-btn", title="Close")
                    ], className="chatbot-controls")
                ], className="chatbot-modern-header"),

                # Messages container with smooth scroll
                html.Div(
                    id="chatbot-modern-messages",
                    className="chatbot-messages-modern",
                    children=[
                        # Welcome message with assistant avatar
                        create_message_bubble(
                            "Hello! I'm your AI Racing Assistant. I can help you analyze telemetry data, compare lap times, and provide coaching tips. How can I help you today?",
                            is_user=False,
                            timestamp=datetime.now().strftime("%H:%M")
                        )
                    ]
                ),

                # Input area with gradient send button
                html.Div([
                    # Input wrapper with focus effect
                    html.Div([
                        dcc.Input(
                            id="chatbot-modern-input",
                            type="text",
                            placeholder="Type your message...",
                            className="chatbot-modern-input",
                            n_submit=0
                        ),

                        html.Button([
                            html.I(className="fas fa-paper-plane")
                        ], id="chatbot-modern-send", className="chatbot-send-modern")
                    ], className="chatbot-input-wrapper"),

                    # Quick action chips
                    html.Div([
                        html.Span("Best Lap", className="quick-action-chip", id={"type": "quick-action", "index": 0}),
                        html.Span("Compare Laps", className="quick-action-chip", id={"type": "quick-action", "index": 1}),
                        html.Span("Coaching Tips", className="quick-action-chip", id={"type": "quick-action", "index": 2}),
                        html.Span("Track Analysis", className="quick-action-chip", id={"type": "quick-action", "index": 3})
                    ], className="chatbot-quick-actions")
                ], className="chatbot-input-area"),

                # Resize handle
                html.Div(className="chatbot-resize-handle"),

                # Minimized state icon
                html.I(className="fas fa-comment-alt chatbot-fab-icon")

            ], className="chatbot-glass-card")
        ], className="chatbot-floating-container", id="chatbot-modern-container"),

        # Hidden stores
        dcc.Store(id="chatbot-modern-history", data=[]),
        dcc.Store(id="chatbot-modern-state", data={"minimized": False}),
        dcc.Interval(id="chatbot-modern-health", interval=30000, n_intervals=0),

        # Load modern CSS
        html.Link(
            rel='stylesheet',
            href='/assets/chatbot-modern.css'
        ),

        # Load modern JavaScript
        html.Script(src='/assets/chatbot-modern.js')
    ], id="chatbot-modern-wrapper")


def create_message_bubble(content, is_user=False, timestamp=None, show_typing=False):
    """
    Create a message bubble with avatar

    Args:
        content: Message content
        is_user: Whether this is a user message
        timestamp: Message timestamp
        show_typing: Whether to show typing indicator

    Returns:
        html.Div: Message bubble container
    """
    if show_typing:
        return html.Div([
            # AI Avatar
            html.Div([
                html.I(className="fas fa-robot")
            ], className="message-avatar ai-avatar"),

            # Typing indicator
            html.Div([
                html.Div(className="typing-dot"),
                html.Div(className="typing-dot"),
                html.Div(className="typing-dot")
            ], className="typing-indicator")
        ], className="message-bubble-container assistant")

    if is_user:
        return html.Div([
            # Message bubble
            html.Div([
                html.P(content, className="mb-0"),
                html.Div(timestamp or datetime.now().strftime("%H:%M"), className="message-timestamp")
            ], className="message-bubble user-bubble"),

            # User Avatar
            html.Div([
                html.I(className="fas fa-user")
            ], className="message-avatar user-avatar")
        ], className="message-bubble-container user")
    else:
        return html.Div([
            # AI Avatar
            html.Div([
                html.I(className="fas fa-robot")
            ], className="message-avatar ai-avatar"),

            # Message bubble with markdown support
            html.Div([
                dcc.Markdown(content, className="mb-0"),
                html.Div(timestamp or datetime.now().strftime("%H:%M"), className="message-timestamp")
            ], className="message-bubble assistant-bubble")
        ], className="message-bubble-container assistant")


def create_modern_chatbot_callbacks(app):
    """
    Register modern chatbot callbacks

    Args:
        app: Dash application instance
    """

    # Health check callback
    @app.callback(
        [Output("chatbot-status-label", "children"),
         Output("chatbot-status-dot", "style")],
        Input("chatbot-modern-health", "n_intervals")
    )
    def update_health_status(n):
        """Check backend API health with visual indicators"""
        try:
            response = requests.get(CHATBOT_HEALTH_URL, timeout=5)
            if response.status_code == 200:
                health = response.json()
                if health.get("status") == "healthy":
                    return "Online", {"background": "#4ade80"}
                else:
                    return "Degraded", {"background": "#fbbf24"}
        except Exception as e:
            logger.error(f"Health check failed: {e}")

        return "Offline", {"background": "#ef4444"}

    # Message handling callback
    @app.callback(
        [Output("chatbot-modern-messages", "children"),
         Output("chatbot-modern-input", "value"),
         Output("chatbot-modern-history", "data")],
        [Input("chatbot-modern-send", "n_clicks"),
         Input("chatbot-modern-input", "n_submit"),
         Input({"type": "quick-action", "index": ALL}, "n_clicks")],
        [State("chatbot-modern-input", "value"),
         State("chatbot-modern-messages", "children"),
         State("chatbot-modern-history", "data"),
         State("vehicle-dropdown", "value"),
         State("tabs", "active_tab")],
        prevent_initial_call=True
    )
    def handle_modern_message(send_clicks, n_submit, quick_clicks, user_input, messages, history,
                            vehicle_number, active_tab):
        """Handle chat messages with beautiful UI updates"""

        # Determine trigger
        ctx = callback_context
        if not ctx.triggered:
            return messages, "", history

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Handle quick actions
        query = user_input
        if "quick-action" in trigger_id:
            import json
            trigger_dict = json.loads(trigger_id)
            action_index = trigger_dict.get("index", 0)

            quick_queries = [
                "What was my best lap time and how does it compare to the session average?",
                "Compare my last 3 laps and show me where I'm losing time",
                "Give me 3 specific tips to improve my corner exit speed",
                "Analyze my performance on the most challenging sections of this track"
            ]
            query = quick_queries[action_index] if action_index < len(quick_queries) else user_input

        # Validate query
        if not query or not query.strip():
            return messages, "", history

        # Add user message
        timestamp = datetime.now().strftime("%H:%M")
        messages.append(create_message_bubble(query, is_user=True, timestamp=timestamp))

        # Add typing indicator
        messages.append(create_message_bubble("", show_typing=True))

        # Store temporarily for API call
        temp_messages = messages[:-1]  # Remove typing indicator

        # Call API
        try:
            payload = {
                "query": query,
                "vehicle_number": vehicle_number or 2
            }

            if active_tab:
                payload["tab"] = active_tab.replace("tab-", "")

            response = requests.post(CHATBOT_API_URL, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "I couldn't process that request.")

                # Remove typing indicator and add actual response
                messages = temp_messages
                messages.append(create_message_bubble(answer, is_user=False, timestamp=timestamp))

                # Update history
                if not history:
                    history = []
                history.extend([
                    {"role": "user", "content": query, "timestamp": timestamp},
                    {"role": "assistant", "content": answer, "timestamp": timestamp}
                ])
            else:
                # Error message
                messages = temp_messages
                error_msg = f"Sorry, I encountered an error. Please try again."
                messages.append(create_message_bubble(error_msg, is_user=False, timestamp=timestamp))

        except requests.exceptions.Timeout:
            messages = temp_messages
            timeout_msg = "The request timed out. Please try again with a simpler query."
            messages.append(create_message_bubble(timeout_msg, is_user=False, timestamp=timestamp))

        except Exception as e:
            logger.error(f"Chatbot error: {e}")
            messages = temp_messages
            error_msg = "I'm having trouble connecting to the backend. Please ensure the API is running."
            messages.append(create_message_bubble(error_msg, is_user=False, timestamp=timestamp))

        return messages, "", history