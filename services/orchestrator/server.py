"""
Video Orchestrator HTTP + SSE Server (V2)

FastAPI server that provides:
- POST /generate - Start video generation (V2 isolated)
- GET /monitor/{campaign_id} - SSE stream for progress
- GET /status/{campaign_id} - Get campaign status
- GET /health - Health check
- GET /migration - Migration status
- GET /v2/isolation - V2 isolation status check
- GET /v2/pipeline - V2 pipeline status

CRITICAL: This server ONLY creates V2 campaigns with status='v2_pending'.
OLD agents (which poll for status='new') will NEVER see these campaigns.

Usage:
    # Start server
    python -m uvicorn services.orchestrator.server:app --host 0.0.0.0 --port 8765

    # Or via main.py
    python main.py server
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.feature_flags import get_processor_mode, get_migration_status, ProcessorMode, should_use_new_orchestrator
from services.router import CampaignRouter
import asyncio
from services.streaming.sse_server import SSEServer, create_progress_callback
from services.streaming.progress_tracker import ProgressEvent, EventType
from .graph import VideoGraph
from .state import VideoState, GenerationPhase
from .db import get_v2_db, V2Database

logger = logging.getLogger(__name__)

# Global SSE server instance
_sse_server: Optional[SSEServer] = None

# Database persistence flag (set False for testing without DB)
USE_DATABASE = os.getenv("V2_USE_DATABASE", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global _sse_server

    # Startup
    logger.info("Starting Video Orchestrator server...")
    _sse_server = SSEServer(host="0.0.0.0", port=8766)  # Different port for internal SSE
    # Note: Not starting SSE server in lifespan since we'll use FastAPI's own SSE

    # ========================================================================
    # RECOVERY: Resume incomplete campaigns from previous pod instance
    # ========================================================================
    if USE_DATABASE:
        try:
            recovered = await _recover_incomplete_campaigns()
            if recovered > 0:
                logger.info(f"Recovered {recovered} incomplete campaigns from database")
        except Exception as e:
            logger.error(f"Failed to recover campaigns: {e}")
            # Don't fail startup - just log the error

    yield

    # Shutdown
    logger.info("Shutting down Video Orchestrator server...")
    if _sse_server:
        await _sse_server.stop()


async def _recover_incomplete_campaigns() -> int:
    """
    Recover incomplete V2 campaigns after pod restart.

    Loads state from database snapshots and resumes generation.
    Returns the number of campaigns recovered.
    """
    v2_db = get_v2_db()
    incomplete = await v2_db.get_incomplete_v2_campaigns(limit=20)

    if not incomplete:
        logger.info("No incomplete V2 campaigns to recover")
        return 0

    logger.info(f"Found {len(incomplete)} incomplete V2 campaigns to recover")
    recovered = 0

    for row in incomplete:
        campaign_id = row["id"]
        has_state = row.get("has_state_snapshot", False)

        if not has_state:
            # No state snapshot - mark as orphaned
            logger.warning(
                f"Campaign {campaign_id} has no state snapshot - marking as orphaned"
            )
            await v2_db.mark_v2_campaign_orphaned(
                campaign_id,
                reason="No state snapshot available for recovery"
            )
            continue

        try:
            # Load state from database
            state = await v2_db.load_v2_state(campaign_id)
            if not state:
                logger.warning(f"Failed to load state for campaign {campaign_id}")
                await v2_db.mark_v2_campaign_orphaned(
                    campaign_id,
                    reason="Failed to deserialize state snapshot"
                )
                continue

            # Generate a unique recovery campaign_id to avoid conflicts
            recovery_campaign_id = f"{state.campaign_id}_recovered_{campaign_id}"
            state.meta["recovered_from_db_id"] = campaign_id
            state.meta["recovered_at"] = datetime.utcnow().isoformat()

            # Store in memory
            _campaigns[recovery_campaign_id] = state

            # Resume generation in background
            asyncio.create_task(
                _run_generation(
                    campaign_id=recovery_campaign_id,
                    state=state,
                    db_campaign_id=campaign_id,
                )
            )

            logger.info(
                f"Recovered campaign {campaign_id} as {recovery_campaign_id} "
                f"at phase {state.phase.value}"
            )
            recovered += 1

        except Exception as e:
            logger.error(f"Failed to recover campaign {campaign_id}: {e}")
            await v2_db.mark_v2_campaign_orphaned(
                campaign_id,
                reason=f"Recovery failed: {e}"
            )

    return recovered


app = FastAPI(
    title="Video Orchestrator API",
    description="Agentic video generation with real-time progress streaming",
    version="2.0.0",
    lifespan=lifespan,
)


# Request/Response Models
class GenerateRequest(BaseModel):
    """Request to generate a video."""
    campaign_id: str
    topic: str
    niche: str
    quality_tier: str = "bulk"
    target_duration_seconds: int = 60
    reference_video_url: Optional[str] = None
    style_reference: str = "viral"
    target_audience: str = "general"
    output_formats: list[str] = ["16:9", "9:16"]


class GenerateResponse(BaseModel):
    """Response from generate endpoint."""
    campaign_id: str
    status: str
    db_status: str  # V2 database status (v2_pending, etc.)
    processor: str
    message: str
    monitor_url: str
    v2_session_id: Optional[str] = None  # V2 tracking ID
    db_campaign_id: Optional[int] = None  # Database ID (if persisted)


class CampaignStatus(BaseModel):
    """Campaign status response."""
    campaign_id: str
    phase: str
    progress_percent: int
    current_step: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    errors: list[dict] = []
    long_form_video_url: Optional[str] = None
    short_form_clips: int = 0


# In-memory campaign state tracking (for demo - use Redis in production)
_campaigns: Dict[str, VideoState] = {}

# Lock to prevent race conditions on campaign creation
_campaign_locks: Dict[str, asyncio.Lock] = {}
_global_lock = asyncio.Lock()  # For creating per-campaign locks

# ============================================================================
# EVENT-DRIVEN SSE SYSTEM
# ============================================================================
# Event queues for pushing updates to SSE clients instead of polling
_campaign_event_queues: Dict[str, list[asyncio.Queue]] = {}
_event_queue_lock = asyncio.Lock()


async def _create_event_queue(campaign_id: str) -> asyncio.Queue:
    """Create an event queue for a campaign's SSE subscribers."""
    async with _event_queue_lock:
        if campaign_id not in _campaign_event_queues:
            _campaign_event_queues[campaign_id] = []
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        _campaign_event_queues[campaign_id].append(queue)
        return queue


async def _remove_event_queue(campaign_id: str, queue: asyncio.Queue):
    """Remove an event queue when SSE client disconnects."""
    async with _event_queue_lock:
        if campaign_id in _campaign_event_queues:
            try:
                _campaign_event_queues[campaign_id].remove(queue)
                if not _campaign_event_queues[campaign_id]:
                    del _campaign_event_queues[campaign_id]
            except ValueError:
                pass  # Queue already removed


async def _emit_event(campaign_id: str, event: dict):
    """Emit an event to all SSE subscribers for a campaign."""
    async with _event_queue_lock:
        if campaign_id in _campaign_event_queues:
            for queue in _campaign_event_queues[campaign_id]:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    # Queue full - skip this event (client is too slow)
                    logger.warning(f"Event queue full for campaign {campaign_id}")

# ============================================================================
# CHECKPOINT SYSTEM
# ============================================================================
# Storage for pending checkpoints: checkpoint_id -> (campaign_id, checkpoint, event, resolution)
_pending_checkpoints: Dict[str, Dict[str, Any]] = {}
_checkpoint_lock = asyncio.Lock()


class CheckpointResolution(BaseModel):
    """Resolution for a checkpoint."""
    resolution: str  # "approved", "rejected", "modified"
    feedback: Optional[str] = None


async def _register_checkpoint(
    campaign_id: str,
    checkpoint: Any,  # Checkpoint dataclass
) -> asyncio.Event:
    """
    Register a checkpoint and return an event to wait on.

    Returns an asyncio.Event that will be set when the checkpoint is resolved.
    """
    async with _checkpoint_lock:
        event = asyncio.Event()
        _pending_checkpoints[checkpoint.checkpoint_id] = {
            "campaign_id": campaign_id,
            "checkpoint": checkpoint,
            "event": event,
            "resolution": None,
            "feedback": None,
            "created_at": datetime.utcnow().isoformat(),
        }
        logger.info(
            f"Checkpoint registered: {checkpoint.checkpoint_id} "
            f"(type={checkpoint.checkpoint_type.value}, campaign={campaign_id})"
        )
        return event


async def _resolve_checkpoint_by_id(
    checkpoint_id: str,
    resolution: str,
    feedback: Optional[str] = None,
) -> bool:
    """
    Resolve a checkpoint by its ID.

    Returns True if checkpoint was found and resolved, False otherwise.
    """
    async with _checkpoint_lock:
        if checkpoint_id not in _pending_checkpoints:
            logger.warning(f"Checkpoint not found: {checkpoint_id}")
            return False

        entry = _pending_checkpoints[checkpoint_id]
        entry["resolution"] = resolution
        entry["feedback"] = feedback
        entry["resolved_at"] = datetime.utcnow().isoformat()

        # Signal the waiting coroutine
        entry["event"].set()

        logger.info(f"Checkpoint resolved: {checkpoint_id} -> {resolution}")
        return True


async def _wait_for_checkpoint(checkpoint_id: str) -> Optional[tuple[str, Optional[str]]]:
    """
    Wait for a checkpoint to be resolved.

    Returns (resolution, feedback) tuple or None if checkpoint not found.
    """
    async with _checkpoint_lock:
        if checkpoint_id not in _pending_checkpoints:
            return None
        event = _pending_checkpoints[checkpoint_id]["event"]

    # Wait outside lock to avoid deadlock
    await event.wait()

    # Get resolution
    async with _checkpoint_lock:
        if checkpoint_id in _pending_checkpoints:
            entry = _pending_checkpoints[checkpoint_id]
            resolution = entry.get("resolution")
            feedback = entry.get("feedback")

            # Clean up
            del _pending_checkpoints[checkpoint_id]

            return (resolution, feedback)

    return None


def _create_checkpoint_resolver():
    """
    Create a checkpoint resolver function for VideoGraph.

    Returns an async function that waits for checkpoint resolution.
    """
    async def resolver(checkpoint_id: str) -> Optional[tuple[str, Optional[str]]]:
        return await _wait_for_checkpoint(checkpoint_id)

    return resolver


async def _on_checkpoint_created(state: VideoState, checkpoint: Any):
    """
    Callback when a checkpoint is created.

    Registers the checkpoint, emits SSE event, and logs.
    """
    await _register_checkpoint(state.campaign_id, checkpoint)

    # Emit checkpoint event (event-driven SSE)
    await _emit_event(state.campaign_id, {
        "type": "checkpoint",
        "campaign_id": state.campaign_id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "checkpoint_type": checkpoint.checkpoint_type.value,
        "phase": checkpoint.phase.value if checkpoint.phase else "unknown",
        "proposal": checkpoint.proposal,
        "reasoning": checkpoint.reasoning,
        "alternatives": checkpoint.alternatives,
        "risks": checkpoint.risks,
        "resolve_url": f"/v2/checkpoints/{checkpoint.checkpoint_id}/resolve",
        "message": "Human checkpoint pending - awaiting resolution",
        "timestamp": datetime.utcnow().isoformat(),
    })


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Video Orchestrator",
        "version": "2.0.0",
        "endpoints": {
            "POST /generate": "Start video generation",
            "GET /monitor/{campaign_id}": "SSE progress stream (includes checkpoint events)",
            "GET /status/{campaign_id}": "Campaign status",
            "GET /health": "Health check",
            "GET /migration": "Migration status",
            "GET /v2/checkpoints": "List pending human checkpoints",
            "GET /v2/checkpoints/{id}": "Get checkpoint details",
            "POST /v2/checkpoints/{id}/resolve": "Approve/reject checkpoint",
            "GET /v2/checkpoints/campaign/{id}": "Get checkpoints for campaign",
            "GET /v2/isolation": "V2 isolation status",
            "GET /v2/pipeline": "V2 pipeline status",
            "GET /v2/info": "V2 orchestrator info",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mode": get_processor_mode().value,
        "active_campaigns": len(_campaigns),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/migration")
async def migration_status():
    """Get V2 migration status."""
    return get_migration_status()


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Start video generation (V2 isolated).

    CRITICAL: This endpoint respects PROCESSOR_MODE feature flag:
    - OLD mode: Returns 503 (use legacy agents instead)
    - SPLIT mode: Only premium tier uses V2, bulk returns 503
    - NEW mode: All campaigns use V2

    CRITICAL: This endpoint ONLY creates V2 campaigns with status='v2_pending'.
    OLD agents (which poll for status='new') will NEVER see these campaigns.

    Returns immediately with campaign info. Use /monitor/{campaign_id}
    to track progress via SSE.
    """
    campaign_id = request.campaign_id

    # ============================================================
    # FEATURE FLAG CHECK - Route to appropriate processor
    # ============================================================
    processor_mode = get_processor_mode()

    if processor_mode == ProcessorMode.OLD:
        # V2 orchestrator disabled - campaigns should use OLD agents via DB polling
        raise HTTPException(
            status_code=503,
            detail="V2 orchestrator disabled (PROCESSOR_MODE=old). "
                   "Create campaigns via NocoDB - OLD agents will poll for them."
        )

    if not should_use_new_orchestrator(campaign_id, request.quality_tier):
        # SPLIT mode and this campaign should go to OLD agents
        raise HTTPException(
            status_code=503,
            detail=f"Campaign quality_tier='{request.quality_tier}' routed to OLD agents "
                   f"(PROCESSOR_MODE=split). Create via NocoDB for OLD agent processing."
        )

    # ============================================================
    # RACE CONDITION PROTECTION - Use per-campaign lock
    # ============================================================
    async with _global_lock:
        if campaign_id not in _campaign_locks:
            _campaign_locks[campaign_id] = asyncio.Lock()
        lock = _campaign_locks[campaign_id]

    async with lock:
        # Check if already processing (now protected by lock)
        if campaign_id in _campaigns:
            existing = _campaigns[campaign_id]
            if existing.phase not in [GenerationPhase.COMPLETE, GenerationPhase.FAILED]:
                raise HTTPException(
                    status_code=409,
                    detail=f"Campaign {campaign_id} is already processing"
                )

        # V2 Database integration (if enabled)
        db_campaign_id = None
        v2_session_id = None

        if USE_DATABASE:
            try:
                v2_db = get_v2_db()
                db_result = await v2_db.create_v2_campaign(
                    topic=request.topic,
                    niche=request.niche,
                    quality_tier=request.quality_tier,
                    target_duration_seconds=request.target_duration_seconds,
                    reference_video_url=request.reference_video_url,
                    style_reference=request.style_reference,
                    target_audience=request.target_audience,
                )
                db_campaign_id = db_result["id"]
                v2_session_id = db_result["v2_session_id"]
                logger.info(
                    f"V2 campaign created in database: "
                    f"id={db_campaign_id}, session={v2_session_id}, "
                    f"status={db_result['status']}, processor={db_result['processor']}"
                )
            except Exception as e:
                logger.error(f"Failed to create V2 campaign in database: {e}")
                # Continue without database - graceful degradation

        # Create initial state with V2 isolation
        state = VideoState(
            campaign_id=campaign_id,
            session_id=v2_session_id or f"v2_memory_{campaign_id}",
            topic=request.topic,
            niche=request.niche,
            quality_tier=request.quality_tier,
            target_duration_seconds=request.target_duration_seconds,
            reference_video_url=request.reference_video_url,
            style_reference=request.style_reference,
            target_audience=request.target_audience,
            output_formats=request.output_formats,
            started_at=datetime.utcnow(),
            phase=GenerationPhase.PENDING,  # Will transition to v2_pending in DB
        )

        # Store database ID in meta for later reference
        if db_campaign_id:
            state.meta["db_campaign_id"] = db_campaign_id
            state.meta["v2_session_id"] = v2_session_id

        # Register campaign in-memory (protected by lock)
        _campaigns[campaign_id] = state

    # Start generation in background (outside lock - safe since campaign is registered)
    background_tasks.add_task(
        _run_generation,
        campaign_id=campaign_id,
        state=state,
        db_campaign_id=db_campaign_id,
    )

    # Get the V2 database status
    db_status = state.phase.to_db_status()

    return GenerateResponse(
        campaign_id=campaign_id,
        status="started",
        db_status=db_status,  # v2_pending
        processor="new",  # Always "new" for V2
        message="V2 video generation started. Monitor progress via SSE.",
        monitor_url=f"/monitor/{campaign_id}",
        v2_session_id=v2_session_id,
        db_campaign_id=db_campaign_id,
    )


async def _run_generation(
    campaign_id: str,
    state: VideoState,
    db_campaign_id: Optional[int] = None,
):
    """
    Background task to run video generation.

    Updates database status using V2-isolated status values.
    Provides progress callback to VideoGraph for real-time DB updates.
    """
    v2_db = get_v2_db() if USE_DATABASE and db_campaign_id else None

    # Track last phase for state persistence (only save on phase change)
    last_saved_phase = [None]  # Use list to allow mutation in closure

    # Progress callback that updates database, in-memory state, and emits SSE events
    async def on_progress(updated_state: VideoState):
        """Update database, in-memory state, and emit SSE event on each phase change."""
        # Update in-memory state
        _campaigns[campaign_id] = updated_state

        # Emit event-driven SSE update (no polling needed!)
        await _emit_event(campaign_id, {
            "type": "progress",
            "campaign_id": campaign_id,
            "phase": updated_state.phase.value,
            "progress_percent": updated_state.progress_percent,
            "current_step": updated_state.current_step,
            "message": updated_state.progress_message,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Update database status and optionally save full state
        if v2_db and db_campaign_id:
            try:
                # Always update status
                await v2_db.update_v2_status(
                    campaign_id=db_campaign_id,
                    phase=updated_state.phase,
                    current_step=updated_state.current_step or updated_state.phase.value,
                    progress_percent=updated_state.progress_percent,
                )

                # Save full state snapshot on phase change (for pod restart recovery)
                if updated_state.phase != last_saved_phase[0]:
                    await v2_db.save_v2_state(
                        campaign_id=db_campaign_id,
                        state=updated_state,
                    )
                    last_saved_phase[0] = updated_state.phase
                    logger.debug(
                        f"V2 state snapshot saved: campaign={db_campaign_id}, "
                        f"phase={updated_state.phase.value}"
                    )
                else:
                    logger.debug(
                        f"V2 DB status updated: campaign={db_campaign_id}, "
                        f"phase={updated_state.phase.value}, progress={updated_state.progress_percent}%"
                    )
            except Exception as e:
                logger.warning(f"Failed to update DB: {e}")
                # Don't fail the generation for DB update failures

    try:
        # Create graph with progress callback and checkpoint support
        graph = VideoGraph(
            on_progress=on_progress,
            on_checkpoint=_on_checkpoint_created,
            checkpoint_resolver=_create_checkpoint_resolver(),
            checkpoint_timeout=3600,  # 1 hour timeout for human checkpoints
        )
        result = await graph.run(state)
        _campaigns[campaign_id] = result
        logger.info(f"V2 campaign {campaign_id} completed: {result.phase.value}")

        # Emit completion event (event-driven SSE)
        await _emit_event(campaign_id, {
            "type": "complete" if result.phase == GenerationPhase.COMPLETE else "failed",
            "campaign_id": campaign_id,
            "phase": result.phase.value,
            "long_form_video_url": result.long_form_video_url,
            "short_form_clips": len(result.short_form_clips),
            "errors": result.errors if result.errors else None,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Update database on completion
        if v2_db and db_campaign_id:
            success = result.phase == GenerationPhase.COMPLETE
            await v2_db.complete_v2_campaign(
                campaign_id=db_campaign_id,
                success=success,
                final_video_url=result.long_form_video_url,
                error_log=str(result.errors) if result.errors else None,
            )

    except Exception as e:
        logger.error(f"V2 campaign {campaign_id} failed: {e}")
        state.phase = GenerationPhase.FAILED
        state.add_error("orchestrator", str(e), recoverable=False)
        _campaigns[campaign_id] = state

        # Emit failure event (event-driven SSE)
        await _emit_event(campaign_id, {
            "type": "failed",
            "campaign_id": campaign_id,
            "phase": GenerationPhase.FAILED.value,
            "errors": state.errors,
            "error_message": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Update database on failure
        if v2_db and db_campaign_id:
            await v2_db.complete_v2_campaign(
                campaign_id=db_campaign_id,
                success=False,
                error_log=str(e),
            )


@app.get("/status/{campaign_id}", response_model=CampaignStatus)
async def get_status(campaign_id: str):
    """Get campaign status."""
    if campaign_id not in _campaigns:
        raise HTTPException(status_code=404, detail="Campaign not found")

    state = _campaigns[campaign_id]

    return CampaignStatus(
        campaign_id=campaign_id,
        phase=state.phase.value,
        progress_percent=state.progress_percent,
        current_step=state.current_step,
        started_at=state.started_at.isoformat() if state.started_at else None,
        completed_at=state.completed_at.isoformat() if state.completed_at else None,
        errors=state.errors,
        long_form_video_url=state.long_form_video_url,
        short_form_clips=len(state.short_form_clips),
    )


@app.get("/monitor/{campaign_id}")
async def monitor(campaign_id: str):
    """
    SSE endpoint for real-time progress monitoring.

    EVENT-DRIVEN: Uses asyncio.Queue for instant updates instead of polling.
    Events are pushed to the queue by callbacks (on_progress, checkpoints, etc.)
    and consumed here with zero latency.

    Event types:
    - connected: Initial connection established
    - progress: Phase/progress updates
    - checkpoint: Human checkpoint pending (requires resolution)
    - checkpoint_resolved: Checkpoint was resolved
    - complete: Generation finished successfully
    - failed: Generation failed
    - heartbeat: Keep-alive (every 30s)

    Usage:
        curl -N http://localhost:8765/monitor/campaign-123
    """
    async def event_stream():
        """Generate SSE events for campaign progress (event-driven, no polling)."""
        # Create event queue for this subscriber
        queue = await _create_event_queue(campaign_id)

        try:
            # Send initial connection event
            yield _format_sse({
                "type": "connected",
                "campaign_id": campaign_id,
                "message": "Connected to progress stream (event-driven)",
                "timestamp": datetime.utcnow().isoformat(),
            })

            # Send current state immediately if campaign exists
            if campaign_id in _campaigns:
                state = _campaigns[campaign_id]
                yield _format_sse({
                    "type": "progress",
                    "campaign_id": campaign_id,
                    "phase": state.phase.value,
                    "progress_percent": state.progress_percent,
                    "current_step": state.current_step,
                    "message": state.progress_message,
                    "timestamp": datetime.utcnow().isoformat(),
                })

                # Check for any pending checkpoints
                async with _checkpoint_lock:
                    for checkpoint_id, entry in _pending_checkpoints.items():
                        if entry["campaign_id"] == campaign_id:
                            checkpoint = entry["checkpoint"]
                            yield _format_sse({
                                "type": "checkpoint",
                                "campaign_id": campaign_id,
                                "checkpoint_id": checkpoint.checkpoint_id,
                                "checkpoint_type": checkpoint.checkpoint_type.value,
                                "phase": checkpoint.phase.value if checkpoint.phase else "unknown",
                                "proposal": checkpoint.proposal,
                                "reasoning": checkpoint.reasoning,
                                "alternatives": checkpoint.alternatives,
                                "risks": checkpoint.risks,
                                "resolve_url": f"/v2/checkpoints/{checkpoint.checkpoint_id}/resolve",
                                "message": "Human checkpoint pending - awaiting resolution",
                                "timestamp": datetime.utcnow().isoformat(),
                            })

                # If already complete, send final event
                if state.phase in [GenerationPhase.COMPLETE, GenerationPhase.FAILED]:
                    yield _format_sse({
                        "type": "complete" if state.phase == GenerationPhase.COMPLETE else "failed",
                        "campaign_id": campaign_id,
                        "phase": state.phase.value,
                        "long_form_video_url": state.long_form_video_url,
                        "short_form_clips": len(state.short_form_clips),
                        "errors": state.errors if state.errors else None,
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                    return

            # Event-driven loop: wait for events instead of polling
            while True:
                try:
                    # Wait for next event with 30-second timeout for heartbeat
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)

                    # Yield the event
                    yield _format_sse(event)

                    # Check if this is a terminal event
                    if event.get("type") in ["complete", "failed", "error"]:
                        break

                except asyncio.TimeoutError:
                    # No events for 30 seconds - send heartbeat
                    yield ": heartbeat\n\n"

                    # Check if campaign still exists (cleanup if abandoned)
                    if campaign_id not in _campaigns:
                        yield _format_sse({
                            "type": "error",
                            "message": "Campaign not found or expired",
                            "campaign_id": campaign_id,
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                        break

        finally:
            # Clean up event queue when client disconnects
            await _remove_event_queue(campaign_id, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _format_sse(data: dict) -> str:
    """Format data as SSE event."""
    import json
    return f"data: {json.dumps(data)}\n\n"


@app.get("/campaigns")
async def list_campaigns():
    """List all tracked campaigns."""
    return {
        "total": len(_campaigns),
        "campaigns": [
            {
                "campaign_id": cid,
                "phase": state.phase.value,
                "progress": state.progress_percent,
                "topic": state.topic,
            }
            for cid, state in _campaigns.items()
        ],
    }


@app.delete("/campaigns/{campaign_id}")
async def delete_campaign(campaign_id: str):
    """Remove campaign from tracking."""
    if campaign_id not in _campaigns:
        raise HTTPException(status_code=404, detail="Campaign not found")

    state = _campaigns[campaign_id]
    if state.phase not in [GenerationPhase.COMPLETE, GenerationPhase.FAILED]:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete campaign that is still processing"
        )

    del _campaigns[campaign_id]
    return {"status": "deleted", "campaign_id": campaign_id}


# ============================================================================
# V2 ISOLATION ENDPOINTS
# ============================================================================


@app.get("/v2/isolation")
async def v2_isolation_check():
    """
    Check V2 isolation integrity.

    Returns any campaigns that violate isolation rules:
    - V2 processor with OLD status
    - OLD processor with V2 status
    - V2 status without processor set

    SHOULD ALWAYS RETURN isolated=True WITH 0 VIOLATIONS.
    """
    if not USE_DATABASE:
        return {
            "isolated": True,
            "message": "Database not enabled - isolation check skipped",
            "database_enabled": False,
        }

    try:
        v2_db = get_v2_db()
        result = await v2_db.check_isolation()
        result["database_enabled"] = True
        return result
    except Exception as e:
        logger.error(f"Isolation check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Isolation check failed: {e}"
        )


@app.get("/v2/pipeline")
async def v2_pipeline_status():
    """
    Get V2 pipeline status summary.

    Shows counts of campaigns in each V2 status.
    """
    if not USE_DATABASE:
        # Return in-memory status
        in_memory_status = {}
        for cid, state in _campaigns.items():
            db_status = state.phase.to_db_status()
            in_memory_status[db_status] = in_memory_status.get(db_status, 0) + 1

        return {
            "source": "in_memory",
            "database_enabled": False,
            "statuses": [
                {"status": s, "count": c} for s, c in in_memory_status.items()
            ],
            "total_v2_campaigns": len(_campaigns),
            "checked_at": datetime.utcnow().isoformat(),
        }

    try:
        v2_db = get_v2_db()
        result = await v2_db.get_v2_pipeline_status()
        result["source"] = "database"
        result["database_enabled"] = True
        return result
    except Exception as e:
        logger.error(f"Pipeline status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline status check failed: {e}"
        )


@app.get("/v2/info")
async def v2_info():
    """
    Get V2 orchestrator information.

    Shows configuration, isolation status, and version info.
    """
    return {
        "version": "2.0.0",
        "processor": "new",
        "isolation": {
            "strategy": "status_based",
            "description": "V2 uses v2_* status values, OLD uses 'new' status",
            "v2_statuses": [
                "v2_pending", "v2_planning", "v2_scripting", "v2_storyboarding",
                "v2_motion", "v2_visual", "v2_quality", "v2_composing", "v2_repurposing",
            ],
            "shared_final_statuses": ["published", "failed"],
            "old_statuses": [
                "new", "in_scripting", "script_approved",
                "generating_visuals", "rendering",
            ],
        },
        "database_enabled": USE_DATABASE,
        "mode": get_processor_mode().value,
        "active_campaigns": len(_campaigns),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/v2/recover")
async def v2_recover():
    """
    Manually trigger recovery of incomplete campaigns.

    Use this if automatic startup recovery failed or to re-check for
    orphaned campaigns after a crash.
    """
    if not USE_DATABASE:
        raise HTTPException(
            status_code=400,
            detail="Database not enabled - recovery requires V2_USE_DATABASE=true"
        )

    try:
        recovered = await _recover_incomplete_campaigns()
        return {
            "status": "success",
            "recovered_count": recovered,
            "active_campaigns": len(_campaigns),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Manual recovery failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Recovery failed: {e}"
        )


@app.get("/v2/recovery-status")
async def v2_recovery_status():
    """
    Get status of campaigns that could be recovered.

    Shows incomplete campaigns with their state snapshot availability.
    """
    if not USE_DATABASE:
        return {
            "database_enabled": False,
            "message": "Recovery requires V2_USE_DATABASE=true",
        }

    try:
        v2_db = get_v2_db()
        incomplete = await v2_db.get_incomplete_v2_campaigns(limit=100)

        return {
            "database_enabled": True,
            "incomplete_campaigns": len(incomplete),
            "campaigns": [
                {
                    "id": row["id"],
                    "topic": row.get("topic", ""),
                    "status": row["status"],
                    "v2_session_id": row.get("v2_session_id"),
                    "has_state_snapshot": row.get("has_state_snapshot", False),
                    "started_at": row.get("v2_started_at").isoformat() if row.get("v2_started_at") else None,
                    "last_updated": row.get("updated_at").isoformat() if row.get("updated_at") else None,
                }
                for row in incomplete
            ],
            "active_in_memory": list(_campaigns.keys()),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Recovery status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Recovery status check failed: {e}"
        )


# ============================================================================
# CHECKPOINT API ENDPOINTS
# ============================================================================


class CheckpointInfo(BaseModel):
    """Checkpoint information for API responses."""
    checkpoint_id: str
    checkpoint_type: str
    campaign_id: str
    phase: str
    proposal: str
    reasoning: str
    alternatives: list[str]
    risks: list[str]
    timeout_seconds: int
    created_at: str


class CheckpointListResponse(BaseModel):
    """Response for listing checkpoints."""
    total: int
    checkpoints: list[CheckpointInfo]


@app.get("/v2/checkpoints", response_model=CheckpointListResponse)
async def list_checkpoints():
    """
    List all pending checkpoints awaiting human resolution.

    Returns checkpoints with their proposals, reasoning, and alternatives.
    Use POST /v2/checkpoints/{checkpoint_id}/resolve to approve/reject.
    """
    async with _checkpoint_lock:
        checkpoints = []
        for checkpoint_id, entry in _pending_checkpoints.items():
            checkpoint = entry["checkpoint"]
            checkpoints.append(CheckpointInfo(
                checkpoint_id=checkpoint.checkpoint_id,
                checkpoint_type=checkpoint.checkpoint_type.value,
                campaign_id=entry["campaign_id"],
                phase=checkpoint.phase.value if checkpoint.phase else "unknown",
                proposal=checkpoint.proposal,
                reasoning=checkpoint.reasoning,
                alternatives=checkpoint.alternatives,
                risks=checkpoint.risks,
                timeout_seconds=checkpoint.timeout_seconds,
                created_at=entry["created_at"],
            ))

        return CheckpointListResponse(
            total=len(checkpoints),
            checkpoints=checkpoints,
        )


@app.get("/v2/checkpoints/{checkpoint_id}", response_model=CheckpointInfo)
async def get_checkpoint(checkpoint_id: str):
    """
    Get details of a specific checkpoint.

    Returns the checkpoint's proposal, reasoning, alternatives, and risks.
    """
    async with _checkpoint_lock:
        if checkpoint_id not in _pending_checkpoints:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        entry = _pending_checkpoints[checkpoint_id]
        checkpoint = entry["checkpoint"]

        return CheckpointInfo(
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_type=checkpoint.checkpoint_type.value,
            campaign_id=entry["campaign_id"],
            phase=checkpoint.phase.value if checkpoint.phase else "unknown",
            proposal=checkpoint.proposal,
            reasoning=checkpoint.reasoning,
            alternatives=checkpoint.alternatives,
            risks=checkpoint.risks,
            timeout_seconds=checkpoint.timeout_seconds,
            created_at=entry["created_at"],
        )


@app.post("/v2/checkpoints/{checkpoint_id}/resolve")
async def resolve_checkpoint(checkpoint_id: str, resolution: CheckpointResolution):
    """
    Resolve a checkpoint with approval, rejection, or modification.

    Args:
        checkpoint_id: The checkpoint to resolve
        resolution: The resolution with feedback
            - resolution: "approved", "rejected", or "modified"
            - feedback: Optional feedback text (required for "rejected" or "modified")

    Returns:
        Resolution status

    Example:
        POST /v2/checkpoints/chk-123/resolve
        {"resolution": "approved"}

        POST /v2/checkpoints/chk-123/resolve
        {"resolution": "rejected", "feedback": "The hook needs to be more engaging"}
    """
    # Validate resolution value
    valid_resolutions = ["approved", "rejected", "modified"]
    if resolution.resolution not in valid_resolutions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid resolution. Must be one of: {valid_resolutions}"
        )

    # Require feedback for rejection/modification
    if resolution.resolution in ["rejected", "modified"] and not resolution.feedback:
        raise HTTPException(
            status_code=400,
            detail=f"Feedback is required when resolution is '{resolution.resolution}'"
        )

    # Resolve the checkpoint
    success = await _resolve_checkpoint_by_id(
        checkpoint_id=checkpoint_id,
        resolution=resolution.resolution,
        feedback=resolution.feedback,
    )

    if not success:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    return {
        "status": "resolved",
        "checkpoint_id": checkpoint_id,
        "resolution": resolution.resolution,
        "feedback": resolution.feedback,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/v2/checkpoints/campaign/{campaign_id}")
async def get_campaign_checkpoints(campaign_id: str):
    """
    Get all pending checkpoints for a specific campaign.

    Returns:
        List of checkpoints for the specified campaign.
    """
    async with _checkpoint_lock:
        checkpoints = []
        for checkpoint_id, entry in _pending_checkpoints.items():
            if entry["campaign_id"] == campaign_id:
                checkpoint = entry["checkpoint"]
                checkpoints.append({
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "checkpoint_type": checkpoint.checkpoint_type.value,
                    "phase": checkpoint.phase.value if checkpoint.phase else "unknown",
                    "proposal": checkpoint.proposal,
                    "created_at": entry["created_at"],
                })

        return {
            "campaign_id": campaign_id,
            "pending_checkpoints": len(checkpoints),
            "checkpoints": checkpoints,
        }


# Module-level run function for main.py
def run_server(host: str = "0.0.0.0", port: int = 8765):
    """Run the server using uvicorn."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
