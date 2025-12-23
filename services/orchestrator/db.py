"""
V2 Orchestrator Database Layer

CRITICAL: This module enforces V2 isolation from OLD agents.

All database operations here ONLY use v2_* status values.
This ensures OLD agents (which poll for status='new') NEVER see V2 campaigns.

Usage:
    from services.orchestrator.db import V2Database

    v2_db = V2Database()
    campaign = await v2_db.create_v2_campaign(
        topic="Test Topic",
        niche="test",
        quality_tier="premium"
    )
"""

import os
import uuid
import logging
from typing import Optional, Any
from datetime import datetime

# Import the shared Database class
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from agents.shared.db import Database, get_db

from .state import VideoState, GenerationPhase

logger = logging.getLogger(__name__)


class V2Database:
    """
    V2-isolated database operations.

    CRITICAL ISOLATION RULES:
    - All campaigns created here use status='v2_pending' (NEVER 'new')
    - All status updates use v2_* values via GenerationPhase.to_db_status()
    - processor='new' is ALWAYS set to identify V2 campaigns
    - V2 session IDs are tracked for debugging

    This ensures OLD agents (polling for status='new') NEVER pick up V2 campaigns.
    """

    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_db()

    async def create_v2_campaign(
        self,
        topic: str,
        niche: str,
        quality_tier: str = "bulk",
        target_duration_seconds: int = 60,
        reference_video_url: Optional[str] = None,
        style_reference: str = "viral",
        target_audience: str = "general",
        meta: Optional[dict] = None,
    ) -> dict:
        """
        Create a V2 campaign with proper isolation.

        CRITICAL: Uses status='v2_pending' (NOT 'new') to prevent OLD agents
        from ever picking up this campaign.
        """
        # Generate V2 session ID for tracking
        v2_session_id = f"v2_{uuid.uuid4().hex[:16]}"

        # Build meta JSONB
        campaign_meta = {
            "target_duration_seconds": target_duration_seconds,
            "reference_video_url": reference_video_url,
            "style_reference": style_reference,
            "target_audience": target_audience,
            "created_via": "v2_orchestrator",
            "v2_version": "2.0.0",
            **(meta or {}),
        }

        # CRITICAL: status='v2_pending' and processor='new' for isolation
        query = """
            INSERT INTO video_campaigns (
                topic,
                niche,
                quality_tier,
                status,
                processor,
                v2_session_id,
                v2_started_at,
                v2_node_history,
                meta,
                created_at,
                updated_at
            ) VALUES (
                %s, %s, %s,
                'v2_pending',  -- CRITICAL: V2 status, not 'new'
                'new',         -- CRITICAL: processor='new' for V2
                %s, %s, '[]'::jsonb, %s::jsonb,
                NOW(), NOW()
            )
            RETURNING id, v2_session_id, status, processor, created_at
        """

        import json
        result = await self.db.fetch_one(
            query,
            [
                topic,
                niche,
                quality_tier,
                v2_session_id,
                datetime.utcnow(),
                json.dumps(campaign_meta),
            ]
        )

        logger.info(
            f"V2 campaign created: id={result['id']}, "
            f"session={result['v2_session_id']}, "
            f"status={result['status']}, "
            f"processor={result['processor']}"
        )

        return result

    async def update_v2_status(
        self,
        campaign_id: int,
        phase: GenerationPhase,
        current_step: str = "",
        progress_percent: int = 0,
        progress_message: str = "",
    ) -> None:
        """
        Update V2 campaign status using V2-isolated status values.

        CRITICAL: Uses GenerationPhase.to_db_status() to ensure
        we only write v2_* status values (or published/failed).
        """
        # Get V2-isolated database status
        db_status = phase.to_db_status()

        # Validate it's a V2 status
        if not db_status.startswith("v2_") and db_status not in ("published", "failed"):
            raise ValueError(
                f"ISOLATION VIOLATION: Attempted to set non-V2 status '{db_status}'. "
                f"V2 campaigns must use v2_* statuses."
            )

        # Update campaign
        query = """
            UPDATE video_campaigns
            SET
                status = %s,
                updated_at = NOW()
            WHERE id = %s AND processor = 'new'
            RETURNING id, status
        """

        result = await self.db.fetch_one(query, [db_status, campaign_id])

        if not result:
            raise ValueError(
                f"Campaign {campaign_id} not found or is not a V2 campaign (processor != 'new')"
            )

        logger.debug(f"V2 campaign {campaign_id} status updated to: {db_status}")

    async def record_v2_node_execution(
        self,
        campaign_id: int,
        node_name: str,
        status: str,
        started_at: datetime,
        completed_at: Optional[datetime] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Record a node execution in the V2 history.

        This provides detailed visibility into graph execution.
        """
        import json

        node_record = {
            "node": node_name,
            "status": status,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat() if completed_at else None,
            "error": error,
        }

        query = """
            UPDATE video_campaigns
            SET
                v2_node_history = v2_node_history || %s::jsonb,
                updated_at = NOW()
            WHERE id = %s AND processor = 'new'
        """

        await self.db.execute(query, [json.dumps([node_record]), campaign_id])

    async def complete_v2_campaign(
        self,
        campaign_id: int,
        success: bool,
        final_video_url: Optional[str] = None,
        error_log: Optional[str] = None,
    ) -> None:
        """
        Mark a V2 campaign as complete.

        Final statuses ('published', 'failed') are shared between V2 and OLD,
        but we keep processor='new' for analytics.
        """
        status = "published" if success else "failed"

        query = """
            UPDATE video_campaigns
            SET
                status = %s,
                final_video_url = %s,
                error_log = %s,
                v2_completed_at = NOW(),
                updated_at = NOW()
            WHERE id = %s AND processor = 'new'
            RETURNING id, status, processor
        """

        result = await self.db.fetch_one(
            query,
            [status, final_video_url, error_log, campaign_id]
        )

        if not result:
            raise ValueError(f"Campaign {campaign_id} not found or is not a V2 campaign")

        logger.info(f"V2 campaign {campaign_id} completed: {status}")

    async def get_v2_campaign(self, campaign_id: int) -> Optional[dict]:
        """
        Get a V2 campaign by ID.

        Only returns campaigns with processor='new' (V2 campaigns).
        """
        query = """
            SELECT *
            FROM video_campaigns
            WHERE id = %s AND processor = 'new'
        """

        return await self.db.fetch_one(query, [campaign_id])

    async def get_v2_pending_campaigns(self, limit: int = 10) -> list:
        """
        Get pending V2 campaigns.

        Only returns campaigns with status='v2_pending'.
        """
        query = """
            SELECT *
            FROM video_campaigns
            WHERE status = 'v2_pending'
            AND processor = 'new'
            ORDER BY created_at ASC
            LIMIT %s
        """

        return await self.db.fetch_all(query, [limit])

    async def check_isolation(self) -> dict:
        """
        Check V2 isolation integrity.

        Returns any campaigns that violate isolation rules:
        - V2 processor with OLD status
        - OLD processor with V2 status
        - V2 status without processor set
        """
        query = """
            SELECT id, topic, status, processor,
                   CASE
                       WHEN processor = 'new' AND status IN ('new', 'in_scripting', 'script_approved', 'generating_visuals', 'rendering')
                           THEN 'V2 processor with OLD status'
                       WHEN processor = 'old' AND status LIKE 'v2_%'
                           THEN 'OLD processor with V2 status'
                       WHEN status LIKE 'v2_%' AND processor IS NULL
                           THEN 'V2 status without processor'
                       ELSE 'unknown'
                   END as violation_type
            FROM video_campaigns
            WHERE
                (processor = 'new' AND status IN ('new', 'in_scripting', 'script_approved', 'generating_visuals', 'rendering'))
                OR (processor = 'old' AND status LIKE 'v2_%%')
                OR (status LIKE 'v2_%%' AND processor IS NULL)
        """

        violations = await self.db.fetch_all(query, [])

        return {
            "isolated": len(violations) == 0,
            "violation_count": len(violations),
            "violations": violations,
            "checked_at": datetime.utcnow().isoformat(),
        }

    async def get_v2_pipeline_status(self) -> dict:
        """
        Get V2 pipeline status summary.
        """
        query = """
            SELECT
                status,
                COUNT(*) as count,
                MIN(created_at) as oldest,
                MAX(created_at) as newest
            FROM video_campaigns
            WHERE processor = 'new' OR status LIKE 'v2_%%'
            GROUP BY status
            ORDER BY status
        """

        rows = await self.db.fetch_all(query, [])

        return {
            "statuses": [
                {
                    "status": row["status"],
                    "count": row["count"],
                    "oldest": row["oldest"].isoformat() if row["oldest"] else None,
                    "newest": row["newest"].isoformat() if row["newest"] else None,
                }
                for row in rows
            ],
            "total_v2_campaigns": sum(row["count"] for row in rows),
            "checked_at": datetime.utcnow().isoformat(),
        }

    # ========================================================================
    # STATE PERSISTENCE (for pod restart recovery)
    # ========================================================================

    async def save_v2_state(
        self,
        campaign_id: int,
        state: VideoState,
    ) -> None:
        """
        Persist full VideoState to database for recovery after pod restart.

        Called at phase boundaries to checkpoint state.
        """
        import json

        # Serialize VideoState to JSON (excluding non-serializable fields)
        state_dict = {
            "campaign_id": state.campaign_id,
            "session_id": state.session_id,
            "topic": state.topic,
            "niche": state.niche,
            "quality_tier": state.quality_tier,
            "target_duration_seconds": state.target_duration_seconds,
            "reference_video_url": state.reference_video_url,
            "style_reference": state.style_reference,
            "target_audience": state.target_audience,
            "output_formats": state.output_formats,
            "phase": state.phase.value,
            "progress_percent": state.progress_percent,
            "current_step": state.current_step,
            "progress_message": state.progress_message,
            "retry_count": state.retry_count,
            "max_retries": state.max_retries,
            "errors": state.errors,
            "long_form_video_url": state.long_form_video_url,
            "short_form_clips": state.short_form_clips,
            "started_at": state.started_at.isoformat() if state.started_at else None,
            "completed_at": state.completed_at.isoformat() if state.completed_at else None,
            "meta": state.meta,
            # Script/storyboard data
            "script": state.script,
            "storyboard": state.storyboard.model_dump() if state.storyboard else None,
            "scenes": state.scenes,
            "motion_blueprint": state.motion_blueprint.model_dump() if state.motion_blueprint else None,
        }

        query = """
            UPDATE video_campaigns
            SET
                v2_state_snapshot = %s::jsonb,
                v2_state_saved_at = NOW(),
                updated_at = NOW()
            WHERE id = %s AND processor = 'new'
        """

        await self.db.execute(query, [json.dumps(state_dict), campaign_id])
        logger.debug(f"V2 state saved for campaign {campaign_id} at phase {state.phase.value}")

    async def load_v2_state(self, campaign_id: int) -> Optional[VideoState]:
        """
        Load VideoState from database snapshot.

        Used for recovery after pod restart.
        """
        import json

        query = """
            SELECT v2_state_snapshot, v2_session_id, topic, niche, quality_tier
            FROM video_campaigns
            WHERE id = %s AND processor = 'new'
            AND v2_state_snapshot IS NOT NULL
        """

        result = await self.db.fetch_one(query, [campaign_id])

        if not result or not result.get("v2_state_snapshot"):
            return None

        state_dict = result["v2_state_snapshot"]
        if isinstance(state_dict, str):
            state_dict = json.loads(state_dict)

        # Reconstruct VideoState from dict
        from .state import Storyboard, MotionBlueprint

        state = VideoState(
            campaign_id=state_dict.get("campaign_id", ""),
            session_id=state_dict.get("session_id", result.get("v2_session_id", "")),
            topic=state_dict.get("topic", result.get("topic", "")),
            niche=state_dict.get("niche", result.get("niche", "")),
            quality_tier=state_dict.get("quality_tier", result.get("quality_tier", "bulk")),
            target_duration_seconds=state_dict.get("target_duration_seconds", 60),
            reference_video_url=state_dict.get("reference_video_url"),
            style_reference=state_dict.get("style_reference", "viral"),
            target_audience=state_dict.get("target_audience", "general"),
            output_formats=state_dict.get("output_formats", ["16:9", "9:16"]),
            phase=GenerationPhase(state_dict.get("phase", "pending")),
            progress_percent=state_dict.get("progress_percent", 0),
            current_step=state_dict.get("current_step", ""),
            progress_message=state_dict.get("progress_message", ""),
            retry_count=state_dict.get("retry_count", 0),
            max_retries=state_dict.get("max_retries", 3),
            errors=state_dict.get("errors", []),
            long_form_video_url=state_dict.get("long_form_video_url"),
            short_form_clips=state_dict.get("short_form_clips", []),
            meta=state_dict.get("meta", {}),
            script=state_dict.get("script"),
            scenes=state_dict.get("scenes", []),
        )

        # Parse dates
        if state_dict.get("started_at"):
            state.started_at = datetime.fromisoformat(state_dict["started_at"])
        if state_dict.get("completed_at"):
            state.completed_at = datetime.fromisoformat(state_dict["completed_at"])

        # Reconstruct complex objects if present
        if state_dict.get("storyboard"):
            try:
                state.storyboard = Storyboard(**state_dict["storyboard"])
            except Exception as e:
                logger.warning(f"Failed to reconstruct storyboard: {e}")

        if state_dict.get("motion_blueprint"):
            try:
                state.motion_blueprint = MotionBlueprint(**state_dict["motion_blueprint"])
            except Exception as e:
                logger.warning(f"Failed to reconstruct motion_blueprint: {e}")

        logger.info(f"V2 state loaded for campaign {campaign_id} at phase {state.phase.value}")
        return state

    async def get_incomplete_v2_campaigns(self, limit: int = 50) -> list:
        """
        Get V2 campaigns that are in progress (not published/failed).

        Used for recovery after pod restart - these campaigns should be resumed.
        """
        query = """
            SELECT id, v2_session_id, topic, niche, quality_tier, status,
                   v2_state_snapshot IS NOT NULL as has_state_snapshot,
                   v2_started_at, updated_at
            FROM video_campaigns
            WHERE processor = 'new'
            AND status LIKE 'v2_%%'
            AND status NOT IN ('published', 'failed')
            ORDER BY v2_started_at ASC
            LIMIT %s
        """

        return await self.db.fetch_all(query, [limit])

    async def mark_v2_campaign_orphaned(
        self,
        campaign_id: int,
        reason: str = "Pod restart - recovery failed",
    ) -> None:
        """
        Mark a V2 campaign as orphaned when recovery fails.

        This prevents the campaign from being picked up again but keeps the data.
        """
        query = """
            UPDATE video_campaigns
            SET
                status = 'failed',
                error_log = COALESCE(error_log, '') || %s,
                v2_completed_at = NOW(),
                updated_at = NOW()
            WHERE id = %s AND processor = 'new'
        """

        error_msg = f"\n[ORPHANED] {datetime.utcnow().isoformat()}: {reason}"
        await self.db.execute(query, [error_msg, campaign_id])


# Singleton instance
_v2_db: Optional[V2Database] = None


def get_v2_db() -> V2Database:
    """Get the singleton V2Database instance."""
    global _v2_db
    if _v2_db is None:
        _v2_db = V2Database()
    return _v2_db
