"""
SSE Server for Video Generation Progress Streaming

Provides real-time progress updates via Server-Sent Events.
Designed for CLI consumption with support for multiple concurrent campaigns.

Features:
- Multi-campaign support (one stream per campaign)
- Automatic client cleanup on disconnect
- Event replay for late-joining clients
- Heartbeat to keep connections alive
- JSON-formatted events for easy parsing

Usage:
    # Start server
    server = SSEServer(host="0.0.0.0", port=8765)
    await server.start()

    # From CLI
    curl -N http://localhost:8765/stream/campaign-123

    # Or with Python
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{url}/stream/{campaign_id}") as resp:
            async for line in resp.content:
                event = parse_sse(line)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from aiohttp import web

from .progress_tracker import ProgressEvent, EventType, ProgressTracker

logger = logging.getLogger(__name__)


@dataclass
class SSEClient:
    """Represents a connected SSE client."""

    client_id: str = field(default_factory=lambda: str(uuid4()))
    campaign_id: str = ""
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_event_id: Optional[str] = None
    user_agent: str = ""


class SSEServer:
    """
    Server-Sent Events server for streaming video generation progress.

    Supports multiple concurrent campaigns with dedicated streams.
    Clients connect to /stream/{campaign_id} to receive events.

    Usage:
        server = SSEServer()
        await server.start()

        # Send event to all clients watching a campaign
        await server.broadcast(campaign_id, event)

        # Send to specific client
        await server.send(client_id, event)
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        heartbeat_interval: int = 30,
        event_history_size: int = 100,
    ):
        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.event_history_size = event_history_size

        # Client tracking
        self._clients: dict[str, SSEClient] = {}  # client_id -> client
        self._campaign_clients: dict[str, set[str]] = {}  # campaign_id -> client_ids

        # Event history for replay
        self._event_history: dict[str, list[ProgressEvent]] = {}  # campaign_id -> events

        # Server state
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the SSE server."""
        self._app = web.Application()
        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/stream/{campaign_id}", self._handle_stream)
        self._app.router.add_get("/monitor/{campaign_id}", self._handle_stream)  # Alias for /stream
        self._app.router.add_get("/status", self._handle_status)
        self._app.router.add_get("/status/{campaign_id}", self._handle_campaign_status)
        self._app.router.add_post("/generate", self._handle_generate)

        # Track running generations
        self._running_tasks: dict[str, asyncio.Task] = {}

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()

        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(f"SSE server started at http://{self.host}:{self.port}")

    async def stop(self):
        """Stop the SSE server."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Disconnect all clients
        for client_id in list(self._clients.keys()):
            await self._disconnect_client(client_id)

        if self._runner:
            await self._runner.cleanup()

        logger.info("SSE server stopped")

    async def _handle_index(self, request: web.Request) -> web.Response:
        """Handle index route - show usage info."""
        return web.Response(
            text="""
AgenticVideo SSE Progress Server

Endpoints:
  POST /generate               - Start video generation
  GET /stream/{campaign_id}    - SSE stream for campaign progress
  GET /monitor/{campaign_id}   - Alias for /stream
  GET /status                  - Server status and connected clients
  GET /status/{campaign_id}    - Campaign-specific status
  GET /health                  - Health check

Generate Video:
  curl -X POST http://localhost:8765/generate \\
    -H "Content-Type: application/json" \\
    -d '{"topic": "10 AI Tools You Need", "niche": "tech", "quality_tier": "bulk"}'

  Response:
    {"campaign_id": "uuid", "status": "started", "stream_url": "/stream/uuid"}

Monitor Progress:
  curl -N http://localhost:8765/stream/{campaign_id}

Events are JSON formatted with the following structure:
  {
    "id": "event-uuid",
    "campaign_id": "campaign-uuid",
    "type": "progress|phase_started|decision|completed|failed",
    "timestamp": "2024-01-15T10:30:00Z",
    "phase": "script|visual|compose|...",
    "phase_number": 2,
    "total_phases": 8,
    "progress_percent": 45.5,
    "current_step": "Generating visuals",
    "message": "Human readable message"
  }
            """,
            content_type="text/plain",
        )

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "connected_clients": len(self._clients),
            "active_campaigns": len(self._campaign_clients),
        })

    async def _handle_status(self, request: web.Request) -> web.Response:
        """Server status endpoint."""
        return web.json_response({
            "server": {
                "host": self.host,
                "port": self.port,
                "uptime_seconds": 0,  # TODO: Track uptime
            },
            "clients": {
                "total": len(self._clients),
                "by_campaign": {
                    cid: len(clients)
                    for cid, clients in self._campaign_clients.items()
                },
            },
            "campaigns": {
                cid: {
                    "clients": len(self._campaign_clients.get(cid, set())),
                    "events": len(self._event_history.get(cid, [])),
                }
                for cid in self._event_history.keys()
            },
        })

    async def _handle_campaign_status(self, request: web.Request) -> web.Response:
        """Campaign-specific status endpoint."""
        campaign_id = request.match_info["campaign_id"]

        events = self._event_history.get(campaign_id, [])
        clients = self._campaign_clients.get(campaign_id, set())

        # Get last event of key types
        last_events = {}
        for event in reversed(events):
            event_type = event.event_type.value
            if event_type not in last_events:
                last_events[event_type] = {
                    "timestamp": event.timestamp.isoformat(),
                    "message": event.message,
                }

        return web.json_response({
            "campaign_id": campaign_id,
            "clients": len(clients),
            "total_events": len(events),
            "current_phase": events[-1].phase if events else None,
            "progress_percent": events[-1].progress_percent if events else 0,
            "last_events": last_events,
        })

    async def _handle_generate(self, request: web.Request) -> web.Response:
        """Handle video generation request - triggers VideoGraph orchestrator."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {"error": "Invalid JSON body"},
                status=400
            )

        # Demo mode - simulate generation without actual API calls
        if data.get("demo") is True or data.get("mode") == "demo":
            return await self._handle_demo_generate(data)

        # Extract parameters
        topic = data.get("topic", "")
        niche = data.get("niche", "general")
        quality_tier = data.get("quality_tier", "bulk")
        reference_video_url = data.get("reference_video_url")
        campaign_id = data.get("campaign_id", str(uuid4()))

        if not topic:
            return web.json_response(
                {"error": "topic is required"},
                status=400
            )

        logger.info(f"Starting generation for campaign {campaign_id}: {topic}")

        # Create progress tracker and callback
        tracker = ProgressTracker(campaign_id=campaign_id)

        async def progress_callback(event: ProgressEvent):
            """Broadcast progress events to SSE clients."""
            await self.broadcast(campaign_id, event)

        tracker.on_event(lambda e: asyncio.create_task(progress_callback(e)))

        # Start generation in background
        task = asyncio.create_task(
            self._run_generation(
                campaign_id=campaign_id,
                topic=topic,
                niche=niche,
                quality_tier=quality_tier,
                reference_video_url=reference_video_url,
                tracker=tracker,
            )
        )
        self._running_tasks[campaign_id] = task

        # Emit started event
        tracker.started(f"Starting video generation: {topic}")

        return web.json_response({
            "campaign_id": campaign_id,
            "status": "started",
            "stream_url": f"/stream/{campaign_id}",
            "message": f"Video generation started for: {topic}",
        })

    async def _run_generation(
        self,
        campaign_id: str,
        topic: str,
        niche: str,
        quality_tier: str,
        reference_video_url: Optional[str],
        tracker: ProgressTracker,
    ):
        """Run VideoGraph generation in background."""
        try:
            # Import here to avoid circular imports
            from services.orchestrator.graph import VideoGraph
            from services.orchestrator.state import VideoState, GenerationPhase

            # Create state
            state = VideoState(
                campaign_id=campaign_id,
                topic=topic,
                niche=niche,
                quality_tier=quality_tier,
                reference_video_url=reference_video_url,
            )

            # Create progress callback for VideoGraph
            async def on_progress(video_state: VideoState):
                """Convert VideoState to progress events."""
                phase = video_state.phase.value if hasattr(video_state.phase, 'value') else str(video_state.phase)
                phase_number = {
                    "planning": 1, "script": 2, "storyboard": 3, "motion": 4,
                    "visual_generation": 5, "quality_review": 6, "composition": 7,
                    "repurposing": 8, "complete": 8,
                }.get(phase, 0)

                progress_percent = video_state.progress_percent if hasattr(video_state, 'progress_percent') else (phase_number / 8 * 100)

                event = ProgressEvent(
                    campaign_id=campaign_id,
                    event_type=EventType.PROGRESS,
                    phase=phase,
                    phase_number=phase_number,
                    total_phases=8,
                    progress_percent=progress_percent,
                    current_step=video_state.current_step if hasattr(video_state, 'current_step') else "",
                    message=video_state.message if hasattr(video_state, 'message') else f"Phase: {phase}",
                )
                await self.broadcast(campaign_id, event)

            # Create and run graph
            graph = VideoGraph(on_progress=on_progress)
            final_state = await graph.run(state)

            # Emit completion
            if final_state.phase == GenerationPhase.COMPLETE:
                tracker.completed(
                    message="Video generation completed successfully",
                    data={
                        "long_form_video_url": final_state.long_form_video_url,
                        "short_form_clips": len(final_state.short_form_clips) if final_state.short_form_clips else 0,
                    }
                )
            else:
                tracker.failed(
                    message=f"Generation ended in phase: {final_state.phase.value}",
                    error=final_state.error if hasattr(final_state, 'error') else None,
                )

        except Exception as e:
            logger.exception(f"Generation failed for campaign {campaign_id}: {e}")
            tracker.failed(message=f"Generation failed: {str(e)}", error=str(e))
        finally:
            # Cleanup task reference
            self._running_tasks.pop(campaign_id, None)

    async def _handle_demo_generate(self, data: dict) -> web.Response:
        """Demo mode - simulate video generation with progress events."""
        topic = data.get("topic", "Demo Video Topic")
        campaign_id = data.get("campaign_id", str(uuid4()))

        logger.info(f"Starting DEMO generation for campaign {campaign_id}: {topic}")

        # Create progress tracker
        tracker = ProgressTracker(campaign_id=campaign_id)

        async def progress_callback(event: ProgressEvent):
            await self.broadcast(campaign_id, event)

        tracker.on_event(lambda e: asyncio.create_task(progress_callback(e)))

        # Start demo simulation in background
        task = asyncio.create_task(self._run_demo_generation(campaign_id, topic, tracker))
        self._running_tasks[campaign_id] = task

        # Emit started event
        tracker.started(f"Starting DEMO video generation: {topic}")

        return web.json_response({
            "campaign_id": campaign_id,
            "status": "started",
            "stream_url": f"/stream/{campaign_id}",
            "message": f"[DEMO MODE] Video generation started for: {topic}",
            "mode": "demo",
        })

    async def _run_demo_generation(
        self,
        campaign_id: str,
        topic: str,
        tracker: ProgressTracker,
    ):
        """Simulate video generation with realistic progress events."""
        import random

        phases = [
            ("planning", "Analyzing topic and planning video structure", 5),
            ("script", "Generating video script with AI", 12),
            ("storyboard", "Creating visual storyboard", 8),
            ("motion", "Extracting motion reference data", 6),
            ("visual_generation", "Generating video scenes", 35),
            ("quality_review", "Reviewing and validating quality", 8),
            ("composition", "Compositing final video", 15),
            ("repurposing", "Creating short-form clips", 8),
        ]

        try:
            total_progress = 0
            for i, (phase, description, weight) in enumerate(phases, 1):
                # Phase start
                event = ProgressEvent(
                    campaign_id=campaign_id,
                    event_type=EventType.PHASE_STARTED,
                    phase=phase,
                    phase_number=i,
                    total_phases=8,
                    progress_percent=total_progress,
                    current_step=description,
                    message=f"Starting: {description}",
                )
                await self.broadcast(campaign_id, event)
                await asyncio.sleep(0.5)

                # Simulate steps within phase
                steps = random.randint(3, 6)
                for step in range(steps):
                    step_progress = total_progress + (weight * (step + 1) / steps)

                    # Generate realistic step messages
                    if phase == "visual_generation":
                        message = f"Generating scene {step + 1}/{steps}"
                    elif phase == "script":
                        message = f"Writing section {step + 1}/{steps}"
                    elif phase == "composition":
                        message = f"Rendering segment {step + 1}/{steps}"
                    else:
                        message = f"Processing step {step + 1}/{steps}"

                    event = ProgressEvent(
                        campaign_id=campaign_id,
                        event_type=EventType.PROGRESS,
                        phase=phase,
                        phase_number=i,
                        total_phases=8,
                        progress_percent=min(step_progress, 99),
                        current_step=message,
                        message=message,
                    )
                    await self.broadcast(campaign_id, event)
                    await asyncio.sleep(random.uniform(0.3, 1.0))

                total_progress += weight

            # Complete
            tracker.completed(
                message="[DEMO] Video generation completed successfully!",
                data={
                    "long_form_video_url": "https://example.com/demo-video.mp4",
                    "short_form_clips": 3,
                    "note": "This was a demo generation - no actual video was created"
                }
            )

        except Exception as e:
            logger.exception(f"Demo generation failed for campaign {campaign_id}: {e}")
            tracker.failed(message=f"Demo failed: {str(e)}", error=str(e))
        finally:
            self._running_tasks.pop(campaign_id, None)

    async def _handle_stream(self, request: web.Request) -> web.StreamResponse:
        """Handle SSE stream connection."""
        campaign_id = request.match_info["campaign_id"]
        last_event_id = request.headers.get("Last-Event-ID")

        # Create client
        client = SSEClient(
            campaign_id=campaign_id,
            last_event_id=last_event_id,
            user_agent=request.headers.get("User-Agent", ""),
        )

        # Register client
        self._clients[client.client_id] = client
        if campaign_id not in self._campaign_clients:
            self._campaign_clients[campaign_id] = set()
        self._campaign_clients[campaign_id].add(client.client_id)

        logger.info(f"Client {client.client_id} connected for campaign {campaign_id}")

        # Create response
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
        await response.prepare(request)

        try:
            # Send connection event
            connect_event = ProgressEvent(
                campaign_id=campaign_id,
                event_type=EventType.INFO,
                message="Connected to progress stream",
                data={"client_id": client.client_id},
            )
            await response.write(connect_event.to_sse().encode())

            # Replay missed events if Last-Event-ID provided
            if last_event_id and campaign_id in self._event_history:
                await self._replay_events(response, campaign_id, last_event_id)

            # Stream events
            while True:
                try:
                    # Wait for event with timeout for heartbeat
                    event = await asyncio.wait_for(
                        client.queue.get(),
                        timeout=self.heartbeat_interval,
                    )

                    # Check for disconnect signal
                    if event is None:
                        break

                    # Send event
                    await response.write(event.to_sse().encode())
                    client.last_event_id = event.event_id

                except asyncio.TimeoutError:
                    # Send heartbeat comment
                    await response.write(b": heartbeat\n\n")

        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            await self._disconnect_client(client.client_id)

        return response

    async def _replay_events(
        self,
        response: web.StreamResponse,
        campaign_id: str,
        last_event_id: str,
    ):
        """Replay events after the given event ID."""
        events = self._event_history.get(campaign_id, [])

        # Find position of last event
        replay_from = 0
        for i, event in enumerate(events):
            if event.event_id == last_event_id:
                replay_from = i + 1
                break

        # Replay missed events
        if replay_from < len(events):
            for event in events[replay_from:]:
                await response.write(event.to_sse().encode())

            logger.debug(
                f"Replayed {len(events) - replay_from} events for campaign {campaign_id}"
            )

    async def _disconnect_client(self, client_id: str):
        """Disconnect and cleanup client."""
        client = self._clients.pop(client_id, None)
        if client:
            campaign_id = client.campaign_id
            if campaign_id in self._campaign_clients:
                self._campaign_clients[campaign_id].discard(client_id)
                if not self._campaign_clients[campaign_id]:
                    del self._campaign_clients[campaign_id]

            logger.info(f"Client {client_id} disconnected")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to all clients."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                # Heartbeats are sent in the stream handler via timeout
            except asyncio.CancelledError:
                break

    async def broadcast(self, campaign_id: str, event: ProgressEvent):
        """
        Broadcast event to all clients watching a campaign.

        Args:
            campaign_id: Campaign ID
            event: Event to broadcast
        """
        # Store in history
        if campaign_id not in self._event_history:
            self._event_history[campaign_id] = []
        self._event_history[campaign_id].append(event)

        # Trim history if needed
        if len(self._event_history[campaign_id]) > self.event_history_size:
            self._event_history[campaign_id] = self._event_history[campaign_id][
                -self.event_history_size:
            ]

        # Send to all clients
        client_ids = self._campaign_clients.get(campaign_id, set())
        for client_id in client_ids:
            client = self._clients.get(client_id)
            if client:
                try:
                    await client.queue.put(event)
                except Exception as e:
                    logger.error(f"Failed to send to client {client_id}: {e}")

    async def send(self, client_id: str, event: ProgressEvent):
        """
        Send event to specific client.

        Args:
            client_id: Client ID
            event: Event to send
        """
        client = self._clients.get(client_id)
        if client:
            try:
                await client.queue.put(event)
            except Exception as e:
                logger.error(f"Failed to send to client {client_id}: {e}")

    def get_client_count(self, campaign_id: str = None) -> int:
        """Get number of connected clients."""
        if campaign_id:
            return len(self._campaign_clients.get(campaign_id, set()))
        return len(self._clients)

    def clear_history(self, campaign_id: str):
        """Clear event history for a campaign."""
        if campaign_id in self._event_history:
            del self._event_history[campaign_id]


def create_progress_callback(
    server: SSEServer,
    campaign_id: str,
) -> callable:
    """
    Create a progress callback function for the orchestrator.

    This bridges the orchestrator's progress events to the SSE server.

    Usage:
        callback = create_progress_callback(server, campaign_id)
        graph = VideoGraph(config=config, on_progress=callback)
    """

    async def callback(event_data: dict[str, Any]):
        """Convert orchestrator event to ProgressEvent and broadcast."""
        event = ProgressEvent(
            campaign_id=campaign_id,
            event_type=EventType(event_data.get("type", "info")),
            phase=event_data.get("phase", ""),
            phase_number=event_data.get("phase_number", 0),
            message=event_data.get("message", ""),
            current_step=event_data.get("step", ""),
            data=event_data.get("data", {}),
            agent=event_data.get("agent"),
            confidence=event_data.get("confidence"),
            reasoning=event_data.get("reasoning"),
        )
        await server.broadcast(campaign_id, event)

    return callback
