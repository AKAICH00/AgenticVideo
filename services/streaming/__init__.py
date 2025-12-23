"""
SSE Progress Streaming Service

Provides real-time visibility into video generation progress via
Server-Sent Events (SSE). Designed for CLI consumption.

Usage:
    # Start server
    from services.streaming import SSEServer
    server = SSEServer(host="0.0.0.0", port=8765)
    await server.start()

    # In CLI
    curl -N http://localhost:8765/stream/campaign-123

    # In orchestrator
    async def on_progress(event):
        await server.broadcast(campaign_id, event)
"""

from .sse_server import SSEServer, SSEClient
from .progress_tracker import ProgressTracker, ProgressEvent, EventType

__all__ = [
    "SSEServer",
    "SSEClient",
    "ProgressTracker",
    "ProgressEvent",
    "EventType",
]
