"""
Video Generation State

Defines the state schema for the LangGraph workflow.
All agents read from and write to this shared state.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID


class GenerationPhase(str, Enum):
    """Main phases of video generation."""
    PENDING = "pending"
    PLANNING = "planning"
    SCRIPTING = "scripting"
    STORYBOARDING = "storyboarding"
    MOTION_EXTRACTION = "motion_extraction"
    VISUAL_GENERATION = "visual_generation"
    COMPOSITION = "composition"
    QUALITY_CHECK = "quality_check"
    REPURPOSING = "repurposing"
    PUBLISHING = "publishing"
    COMPLETE = "complete"
    FAILED = "failed"

    def to_db_status(self) -> str:
        """
        Convert internal phase to V2 database status.

        CRITICAL: This ensures V2 campaigns use v2_* statuses
        which are completely isolated from OLD agent statuses.

        OLD agents poll for: status='new', 'in_scripting', etc.
        V2 uses: 'v2_pending', 'v2_planning', etc.

        This isolation prevents ANY cross-contamination between systems.
        """
        V2_STATUS_MAP = {
            GenerationPhase.PENDING: "v2_pending",
            GenerationPhase.PLANNING: "v2_planning",
            GenerationPhase.SCRIPTING: "v2_scripting",
            GenerationPhase.STORYBOARDING: "v2_storyboarding",
            GenerationPhase.MOTION_EXTRACTION: "v2_motion",
            GenerationPhase.VISUAL_GENERATION: "v2_visual",
            GenerationPhase.COMPOSITION: "v2_composing",
            GenerationPhase.QUALITY_CHECK: "v2_quality",
            GenerationPhase.REPURPOSING: "v2_repurposing",
            GenerationPhase.PUBLISHING: "ready_to_publish",
            # Final states are shared between OLD and V2
            GenerationPhase.COMPLETE: "published",
            GenerationPhase.FAILED: "failed",
        }
        return V2_STATUS_MAP.get(self, "v2_pending")

    @classmethod
    def from_db_status(cls, db_status: str) -> "GenerationPhase":
        """Convert V2 database status back to internal phase."""
        DB_TO_PHASE = {
            "v2_pending": cls.PENDING,
            "v2_planning": cls.PLANNING,
            "v2_scripting": cls.SCRIPTING,
            "v2_storyboarding": cls.STORYBOARDING,
            "v2_motion": cls.MOTION_EXTRACTION,
            "v2_visual": cls.VISUAL_GENERATION,
            "v2_composing": cls.COMPOSITION,
            "v2_quality": cls.QUALITY_CHECK,
            "v2_repurposing": cls.REPURPOSING,
            "ready_to_publish": cls.PUBLISHING,
            "published": cls.COMPLETE,
            "failed": cls.FAILED,
        }
        return DB_TO_PHASE.get(db_status, cls.PENDING)


class CheckpointType(str, Enum):
    """Types of human checkpoints."""
    APPROVAL = "approval"  # Agent proposes, human approves
    INPUT_REQUEST = "input_request"  # Agent needs info from human
    VALIDATION = "validation"  # Agent has result, human verifies
    DECISION = "decision"  # Agent offers multiple options


@dataclass
class ScriptData:
    """Generated script content."""
    hook: str = ""
    body: str = ""
    call_to_action: str = ""
    estimated_duration_seconds: int = 0
    key_points: list[str] = field(default_factory=list)
    tone: str = "engaging"
    variations: list[dict] = field(default_factory=list)  # A/B testing


@dataclass
class Scene:
    """A single scene in the storyboard."""
    scene_id: str = ""
    scene_number: int = 0
    description: str = ""
    visual_prompt: str = ""
    duration_seconds: float = 0
    camera_motion: str = ""  # "static", "pan_left", "zoom_in", etc.
    transition_in: str = "cut"
    transition_out: str = "cut"
    audio_cue: str = ""

    # Generation state
    status: str = "pending"
    video_url: Optional[str] = None
    local_path: Optional[str] = None  # Local file path after download
    thumbnail_url: Optional[str] = None
    generation_job_id: Optional[str] = None
    retry_count: int = 0


@dataclass
class StoryboardData:
    """Full storyboard with all scenes."""
    scenes: list[Scene] = field(default_factory=list)
    total_duration_seconds: float = 0
    style_notes: str = ""
    color_palette: list[str] = field(default_factory=list)
    character_descriptions: dict[str, str] = field(default_factory=dict)


@dataclass
class MotionData:
    """Extracted motion data from reference video."""
    reference_video_url: Optional[str] = None
    pose_keypoints: list[dict] = field(default_factory=list)
    camera_trajectory: list[dict] = field(default_factory=list)
    transitions: list[dict] = field(default_factory=list)
    beat_sync: list[dict] = field(default_factory=list)
    style_embedding: Optional[bytes] = None
    style_tags: list[str] = field(default_factory=list)
    extraction_complete: bool = False


@dataclass
class QualityMetrics:
    """Quality assessment of generated content."""
    overall_score: float = 0.0  # 0-1
    visual_quality: float = 0.0
    motion_smoothness: float = 0.0
    audio_sync: float = 0.0
    consistency: float = 0.0  # Between scenes
    hook_strength: float = 0.0

    feedback: list[str] = field(default_factory=list)
    requires_regeneration: list[str] = field(default_factory=list)  # Scene IDs


@dataclass
class ShortFormClip:
    """A short-form clip generated from the long-form video."""
    clip_id: str = ""
    start_time: float = 0
    end_time: float = 0
    moment_score: float = 0.0
    hook_text: str = ""
    target_platform: str = "tiktok"
    clip_url: Optional[str] = None
    status: str = "pending"


@dataclass
class Checkpoint:
    """A human checkpoint in the workflow."""
    checkpoint_id: str = ""
    checkpoint_type: CheckpointType = CheckpointType.APPROVAL
    phase: GenerationPhase = GenerationPhase.PENDING
    proposal: str = ""
    reasoning: str = ""
    alternatives: list[dict] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    timeout_seconds: int = 3600  # 1 hour default
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None  # "approved", "rejected", "modified"
    human_feedback: Optional[str] = None


@dataclass
class AgentDecision:
    """Record of an agent's decision for transparency."""
    agent_name: str = ""
    decision: str = ""
    reasoning: str = ""
    confidence: float = 0.0  # 0-1
    alternatives_considered: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VideoState:
    """
    Complete state for video generation workflow.

    This is passed through all LangGraph nodes and represents
    the full context of the video being created.
    """

    # Identity
    campaign_id: str = ""
    session_id: str = ""

    # Input
    topic: str = ""
    niche: str = ""
    target_audience: str = ""
    style_reference: str = ""  # "educational", "viral", "cinematic", etc.
    reference_video_url: Optional[str] = None

    # Quality settings
    quality_tier: str = "premium"  # "premium" or "bulk"
    target_duration_seconds: int = 60
    output_formats: list[str] = field(default_factory=lambda: ["16:9", "9:16"])

    # Current state
    phase: GenerationPhase = GenerationPhase.PENDING
    current_step: str = ""
    progress_percent: int = 0
    progress_message: str = ""

    # Generated content
    script: ScriptData = field(default_factory=ScriptData)
    storyboard: StoryboardData = field(default_factory=StoryboardData)
    motion_data: MotionData = field(default_factory=MotionData)
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)

    # Short-form repurposing
    short_form_clips: list[ShortFormClip] = field(default_factory=list)

    # Final outputs
    long_form_video_url: Optional[str] = None
    long_form_thumbnail_url: Optional[str] = None

    # Human interaction
    checkpoints: list[Checkpoint] = field(default_factory=list)
    pending_checkpoint: Optional[Checkpoint] = None

    # Agent reasoning trail
    decisions: list[AgentDecision] = field(default_factory=list)

    # Error handling
    errors: list[dict] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Cost tracking
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0

    # Metadata from intelligence layer
    meta: dict[str, Any] = field(default_factory=dict)

    def add_decision(
        self,
        agent_name: str,
        decision: str,
        reasoning: str,
        confidence: float = 0.8,
        alternatives: Optional[list[dict]] = None,
    ):
        """Record an agent decision for transparency."""
        self.decisions.append(
            AgentDecision(
                agent_name=agent_name,
                decision=decision,
                reasoning=reasoning,
                confidence=confidence,
                alternatives_considered=alternatives or [],
            )
        )

    def add_error(self, phase: str, error: str, recoverable: bool = True):
        """Record an error."""
        self.errors.append({
            "phase": phase,
            "error": error,
            "recoverable": recoverable,
            "timestamp": datetime.utcnow().isoformat(),
            "retry_count": self.retry_count,
        })

    def request_checkpoint(
        self,
        checkpoint_type: CheckpointType,
        proposal: str,
        reasoning: str,
        alternatives: Optional[list[dict]] = None,
        risks: Optional[list[str]] = None,
    ):
        """Request a human checkpoint."""
        import uuid

        self.pending_checkpoint = Checkpoint(
            checkpoint_id=str(uuid.uuid4()),
            checkpoint_type=checkpoint_type,
            phase=self.phase,
            proposal=proposal,
            reasoning=reasoning,
            alternatives=alternatives or [],
            risks=risks or [],
        )

    def resolve_checkpoint(
        self,
        resolution: str,
        human_feedback: Optional[str] = None,
    ):
        """Resolve the pending checkpoint."""
        if self.pending_checkpoint:
            self.pending_checkpoint.resolved_at = datetime.utcnow()
            self.pending_checkpoint.resolution = resolution
            self.pending_checkpoint.human_feedback = human_feedback
            self.checkpoints.append(self.pending_checkpoint)
            self.pending_checkpoint = None

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "campaign_id": self.campaign_id,
            "session_id": self.session_id,
            "topic": self.topic,
            "niche": self.niche,
            "phase": self.phase.value,
            "current_step": self.current_step,
            "progress_percent": self.progress_percent,
            "progress_message": self.progress_message,
            "script": {
                "hook": self.script.hook,
                "body": self.script.body,
                "estimated_duration": self.script.estimated_duration_seconds,
            },
            "storyboard": {
                "scene_count": len(self.storyboard.scenes),
                "total_duration": self.storyboard.total_duration_seconds,
            },
            "motion_data": {
                "has_reference": self.motion_data.reference_video_url is not None,
                "extraction_complete": self.motion_data.extraction_complete,
            },
            "quality_metrics": {
                "overall_score": self.quality_metrics.overall_score,
                "requires_regeneration": self.quality_metrics.requires_regeneration,
            },
            "short_form_clips": len(self.short_form_clips),
            "long_form_video_url": self.long_form_video_url,
            "errors": self.errors,
            "estimated_cost_usd": self.estimated_cost_usd,
            "actual_cost_usd": self.actual_cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoState":
        """Create state from dictionary."""
        state = cls()
        state.campaign_id = data.get("campaign_id", "")
        state.session_id = data.get("session_id", "")
        state.topic = data.get("topic", "")
        state.niche = data.get("niche", "")
        state.phase = GenerationPhase(data.get("phase", "pending"))
        # ... additional deserialization as needed
        return state
