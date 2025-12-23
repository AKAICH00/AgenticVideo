"""
Progress Tracker for Video Generation Pipeline

Tracks progress across all stages and formats events for SSE streaming.
Provides structured events for CLI rendering.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of progress events."""

    # Lifecycle events
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    # Phase events
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    PHASE_FAILED = "phase_failed"

    # Progress events
    PROGRESS = "progress"
    STEP = "step"
    SUBSTEP = "substep"

    # Agent events
    DECISION = "decision"
    RETRY = "retry"
    QUALITY_CHECK = "quality_check"

    # Checkpoint events
    CHECKPOINT_WAITING = "checkpoint_waiting"
    CHECKPOINT_APPROVED = "checkpoint_approved"
    CHECKPOINT_REJECTED = "checkpoint_rejected"

    # Resource events
    ASSET_GENERATED = "asset_generated"
    ASSET_UPLOADED = "asset_uploaded"

    # Info events
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"


@dataclass
class ProgressEvent:
    """A progress event for SSE streaming."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    campaign_id: str = ""
    event_type: EventType = EventType.INFO
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Phase tracking
    phase: str = ""
    phase_number: int = 0
    total_phases: int = 8

    # Progress tracking
    progress_percent: float = 0.0
    current_step: str = ""
    substep: Optional[str] = None

    # Message and data
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    # Timing
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None

    # For decisions/quality
    agent: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None

    def to_sse(self) -> str:
        """Format as SSE message."""
        event_data = {
            "id": self.event_id,
            "campaign_id": self.campaign_id,
            "type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "phase": self.phase,
            "phase_number": self.phase_number,
            "total_phases": self.total_phases,
            "progress_percent": round(self.progress_percent, 1),
            "current_step": self.current_step,
            "message": self.message,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }

        # Add optional fields
        if self.substep:
            event_data["substep"] = self.substep
        if self.estimated_remaining_seconds:
            event_data["eta_seconds"] = round(self.estimated_remaining_seconds, 0)
        if self.agent:
            event_data["agent"] = self.agent
        if self.confidence is not None:
            event_data["confidence"] = round(self.confidence, 2)
        if self.reasoning:
            event_data["reasoning"] = self.reasoning
        if self.data:
            event_data["data"] = self.data

        # Format as SSE
        json_data = json.dumps(event_data)
        return f"id: {self.event_id}\nevent: {self.event_type.value}\ndata: {json_data}\n\n"

    def to_cli_line(self) -> str:
        """Format as single CLI line with colors/formatting."""
        # Phase indicator
        phase_indicator = f"[{self.phase_number}/{self.total_phases}]" if self.phase else ""

        # Progress bar
        bar_width = 20
        filled = int(self.progress_percent / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Time info
        elapsed = f"{int(self.elapsed_seconds)}s"
        eta = f" ETA:{int(self.estimated_remaining_seconds)}s" if self.estimated_remaining_seconds else ""

        # Icon based on event type
        icons = {
            EventType.STARTED: "🚀",
            EventType.COMPLETED: "✅",
            EventType.FAILED: "❌",
            EventType.PHASE_STARTED: "▶️",
            EventType.PHASE_COMPLETED: "✔️",
            EventType.PROGRESS: "⏳",
            EventType.DECISION: "🤔",
            EventType.RETRY: "🔄",
            EventType.QUALITY_CHECK: "🔍",
            EventType.CHECKPOINT_WAITING: "⏸️",
            EventType.ASSET_GENERATED: "🎬",
            EventType.WARNING: "⚠️",
            EventType.ERROR: "🔴",
            EventType.INFO: "ℹ️",
        }
        icon = icons.get(self.event_type, "•")

        # Build line
        if self.event_type in (EventType.PROGRESS, EventType.STEP):
            return f"{icon} {phase_indicator} [{bar}] {self.progress_percent:.0f}% | {self.current_step} | {elapsed}{eta}"
        elif self.event_type == EventType.DECISION:
            conf = f" ({self.confidence:.0%})" if self.confidence else ""
            return f"{icon} {self.agent}: {self.message}{conf}"
        else:
            return f"{icon} {self.message}"


class ProgressTracker:
    """
    Tracks progress of video generation and emits SSE events.

    Integrates with the VideoGraph orchestrator to provide
    real-time visibility into generation progress.

    Usage:
        tracker = ProgressTracker(campaign_id="abc123")

        # Register callback for SSE streaming
        tracker.on_event(lambda e: server.send(e))

        # Track progress
        tracker.phase_started("script", 1)
        tracker.progress(25, "Generating hook")
        tracker.decision("script_agent", "Added viral hook", confidence=0.9)
        tracker.phase_completed("script")
    """

    # Phase weights for overall progress calculation
    PHASE_WEIGHTS = {
        "planning": 5,
        "script": 15,
        "storyboard": 10,
        "motion": 15,
        "visual": 30,
        "compose": 15,
        "quality": 5,
        "repurpose": 5,
    }

    def __init__(
        self,
        campaign_id: str,
        total_phases: int = 8,
    ):
        self.campaign_id = campaign_id
        self.total_phases = total_phases

        self._start_time = datetime.utcnow()
        self._phase_start_time: Optional[datetime] = None
        self._current_phase: str = ""
        self._current_phase_number: int = 0
        self._phase_progress: float = 0.0
        self._completed_phases: set[str] = set()

        self._callbacks: list[Callable[[ProgressEvent], None]] = []
        self._event_history: list[ProgressEvent] = []

    def on_event(self, callback: Callable[[ProgressEvent], None]):
        """Register callback for progress events."""
        self._callbacks.append(callback)

    def _emit(self, event: ProgressEvent):
        """Emit event to all callbacks."""
        # Calculate overall progress
        event.progress_percent = self._calculate_overall_progress()
        event.elapsed_seconds = (datetime.utcnow() - self._start_time).total_seconds()

        # Store in history
        self._event_history.append(event)

        # Emit to callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def _calculate_overall_progress(self) -> float:
        """Calculate overall progress across all phases."""
        total_weight = sum(self.PHASE_WEIGHTS.values())

        # Progress from completed phases
        completed_weight = sum(
            self.PHASE_WEIGHTS.get(p, 0) for p in self._completed_phases
        )

        # Progress from current phase
        current_phase_weight = self.PHASE_WEIGHTS.get(self._current_phase, 0)
        current_contribution = (current_phase_weight * self._phase_progress / 100)

        total_progress = ((completed_weight + current_contribution) / total_weight) * 100
        return min(100.0, total_progress)

    def started(self, message: str = "Video generation started"):
        """Emit generation started event."""
        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.STARTED,
            message=message,
            total_phases=self.total_phases,
        ))

    def completed(self, message: str = "Video generation completed", data: dict = None):
        """Emit generation completed event."""
        self._phase_progress = 100
        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.COMPLETED,
            message=message,
            progress_percent=100,
            data=data or {},
            total_phases=self.total_phases,
        ))

    def failed(self, message: str, error: Optional[str] = None):
        """Emit generation failed event."""
        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.FAILED,
            message=message,
            data={"error": error} if error else {},
            phase=self._current_phase,
            phase_number=self._current_phase_number,
            total_phases=self.total_phases,
        ))

    def phase_started(self, phase: str, phase_number: int, message: str = None):
        """Emit phase started event."""
        self._current_phase = phase
        self._current_phase_number = phase_number
        self._phase_progress = 0
        self._phase_start_time = datetime.utcnow()

        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.PHASE_STARTED,
            phase=phase,
            phase_number=phase_number,
            total_phases=self.total_phases,
            message=message or f"Starting {phase} phase",
            current_step=phase,
        ))

    def phase_completed(self, phase: str, message: str = None, data: dict = None):
        """Emit phase completed event."""
        self._phase_progress = 100
        self._completed_phases.add(phase)

        phase_duration = 0
        if self._phase_start_time:
            phase_duration = (datetime.utcnow() - self._phase_start_time).total_seconds()

        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.PHASE_COMPLETED,
            phase=phase,
            phase_number=self._current_phase_number,
            total_phases=self.total_phases,
            message=message or f"Completed {phase} phase",
            current_step=phase,
            data={
                **(data or {}),
                "phase_duration_seconds": round(phase_duration, 1),
            },
        ))

    def phase_failed(self, phase: str, message: str, error: Optional[str] = None):
        """Emit phase failed event."""
        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.PHASE_FAILED,
            phase=phase,
            phase_number=self._current_phase_number,
            total_phases=self.total_phases,
            message=message,
            data={"error": error} if error else {},
        ))

    def progress(
        self,
        percent: float,
        step: str,
        substep: Optional[str] = None,
        eta_seconds: Optional[float] = None,
    ):
        """Emit progress update within current phase."""
        self._phase_progress = percent

        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.PROGRESS,
            phase=self._current_phase,
            phase_number=self._current_phase_number,
            total_phases=self.total_phases,
            current_step=step,
            substep=substep,
            message=f"{step}: {substep}" if substep else step,
            estimated_remaining_seconds=eta_seconds,
        ))

    def step(self, step: str, message: str = None):
        """Emit step event (major step within phase)."""
        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.STEP,
            phase=self._current_phase,
            phase_number=self._current_phase_number,
            total_phases=self.total_phases,
            current_step=step,
            message=message or step,
        ))

    def decision(
        self,
        agent: str,
        message: str,
        confidence: float = None,
        reasoning: str = None,
        data: dict = None,
    ):
        """Emit agent decision event."""
        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.DECISION,
            phase=self._current_phase,
            phase_number=self._current_phase_number,
            total_phases=self.total_phases,
            agent=agent,
            message=message,
            confidence=confidence,
            reasoning=reasoning,
            data=data or {},
        ))

    def retry(self, attempt: int, max_attempts: int, reason: str):
        """Emit retry event."""
        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.RETRY,
            phase=self._current_phase,
            phase_number=self._current_phase_number,
            total_phases=self.total_phases,
            message=f"Retry {attempt}/{max_attempts}: {reason}",
            data={"attempt": attempt, "max_attempts": max_attempts},
        ))

    def quality_check(self, passed: bool, score: float, issues: list[str] = None):
        """Emit quality check event."""
        status = "passed" if passed else "failed"
        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.QUALITY_CHECK,
            phase=self._current_phase,
            phase_number=self._current_phase_number,
            total_phases=self.total_phases,
            message=f"Quality check {status} (score: {score:.1%})",
            confidence=score,
            data={"passed": passed, "issues": issues or []},
        ))

    def checkpoint_waiting(self, checkpoint_type: str, proposal: str):
        """Emit checkpoint waiting for human approval."""
        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.CHECKPOINT_WAITING,
            phase=self._current_phase,
            phase_number=self._current_phase_number,
            total_phases=self.total_phases,
            message=f"Waiting for approval: {checkpoint_type}",
            data={"checkpoint_type": checkpoint_type, "proposal": proposal},
        ))

    def asset_generated(self, asset_type: str, url: str = None, data: dict = None):
        """Emit asset generated event."""
        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.ASSET_GENERATED,
            phase=self._current_phase,
            phase_number=self._current_phase_number,
            total_phases=self.total_phases,
            message=f"Generated {asset_type}",
            data={"asset_type": asset_type, "url": url, **(data or {})},
        ))

    def info(self, message: str, data: dict = None):
        """Emit info event."""
        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.INFO,
            phase=self._current_phase,
            phase_number=self._current_phase_number,
            total_phases=self.total_phases,
            message=message,
            data=data or {},
        ))

    def warning(self, message: str, data: dict = None):
        """Emit warning event."""
        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.WARNING,
            phase=self._current_phase,
            phase_number=self._current_phase_number,
            total_phases=self.total_phases,
            message=message,
            data=data or {},
        ))

    def error(self, message: str, error: str = None):
        """Emit error event."""
        self._emit(ProgressEvent(
            campaign_id=self.campaign_id,
            event_type=EventType.ERROR,
            phase=self._current_phase,
            phase_number=self._current_phase_number,
            total_phases=self.total_phases,
            message=message,
            data={"error": error} if error else {},
        ))

    def get_history(self) -> list[ProgressEvent]:
        """Get all events emitted so far."""
        return self._event_history.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get summary of current progress state."""
        return {
            "campaign_id": self.campaign_id,
            "current_phase": self._current_phase,
            "phase_number": self._current_phase_number,
            "total_phases": self.total_phases,
            "overall_progress": self._calculate_overall_progress(),
            "phase_progress": self._phase_progress,
            "completed_phases": list(self._completed_phases),
            "elapsed_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
            "event_count": len(self._event_history),
        }
