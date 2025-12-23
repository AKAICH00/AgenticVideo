"""
Campaign Router

Routes campaigns to either OLD daemons or NEW orchestrator based on feature flags.

Usage:
    from services.router import CampaignRouter

    router = CampaignRouter()
    result = await router.process_campaign(
        campaign_id="abc-123",
        topic="AI Trends 2025",
        niche="tech",
        quality_tier="premium",
    )
"""

import logging
from typing import Optional, Dict, Any

from core.feature_flags import should_use_new_orchestrator, get_migration_status
from services.orchestrator.graph import VideoGraph
from services.orchestrator.state import VideoState, GenerationPhase

logger = logging.getLogger(__name__)


class CampaignRouter:
    """
    Routes campaigns to the appropriate processor based on feature flags.

    In SPLIT mode:
    - Premium campaigns → VideoGraph (new orchestrator)
    - Bulk campaigns → Database queue (old daemons pick up)
    """

    def __init__(self):
        self.orchestrator = VideoGraph()

    async def process_campaign(
        self,
        campaign_id: str,
        topic: str,
        niche: str,
        quality_tier: str = "bulk",
        target_duration_seconds: int = 60,
        reference_video_url: Optional[str] = None,
        style_reference: str = "viral",
        override_processor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Route and process a campaign.

        Args:
            campaign_id: Unique campaign identifier
            topic: Video topic
            niche: Content niche
            quality_tier: "premium" or "bulk"
            target_duration_seconds: Target video length
            reference_video_url: Optional TikTok/reference video URL
            style_reference: Style (viral, educational, cinematic)
            override_processor: Force "old" or "new" processor

        Returns:
            Dict with processor used and result
        """
        use_new = should_use_new_orchestrator(
            campaign_id=campaign_id,
            quality_tier=quality_tier,
            override=override_processor,
        )

        if use_new:
            logger.info(f"Routing campaign {campaign_id} to NEW orchestrator")
            return await self._process_with_orchestrator(
                campaign_id=campaign_id,
                topic=topic,
                niche=niche,
                quality_tier=quality_tier,
                target_duration_seconds=target_duration_seconds,
                reference_video_url=reference_video_url,
                style_reference=style_reference,
            )
        else:
            logger.info(f"Routing campaign {campaign_id} to OLD daemons")
            return await self._queue_for_daemons(
                campaign_id=campaign_id,
                topic=topic,
                niche=niche,
                quality_tier=quality_tier,
            )

    async def _process_with_orchestrator(
        self,
        campaign_id: str,
        topic: str,
        niche: str,
        quality_tier: str,
        target_duration_seconds: int,
        reference_video_url: Optional[str],
        style_reference: str,
    ) -> Dict[str, Any]:
        """Process campaign using the new VideoGraph orchestrator."""
        state = VideoState(
            campaign_id=campaign_id,
            topic=topic,
            niche=niche,
            quality_tier=quality_tier,
            target_duration_seconds=target_duration_seconds,
            reference_video_url=reference_video_url,
            style_reference=style_reference,
        )

        try:
            result = await self.orchestrator.run(state)
            return {
                "processor": "new",
                "campaign_id": campaign_id,
                "phase": result.phase.value,
                "progress": result.progress_percent,
                "long_form_video_url": result.long_form_video_url,
                "short_form_clips": len(result.short_form_clips),
                "success": result.phase == GenerationPhase.COMPLETE,
            }
        except Exception as e:
            logger.error(f"Orchestrator error for {campaign_id}: {e}")
            return {
                "processor": "new",
                "campaign_id": campaign_id,
                "phase": "failed",
                "error": str(e),
                "success": False,
            }

    async def _queue_for_daemons(
        self,
        campaign_id: str,
        topic: str,
        niche: str,
        quality_tier: str,
    ) -> Dict[str, Any]:
        """Queue campaign for OLD polling daemons."""
        # In production, this would update the database
        # to trigger the director daemon
        # For now, return a placeholder indicating queued status
        return {
            "processor": "old",
            "campaign_id": campaign_id,
            "status": "queued",
            "message": "Campaign queued for polling daemons. Director will pick up.",
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current router and migration status."""
        return {
            "migration": get_migration_status(),
            "orchestrator_ready": self.orchestrator is not None,
        }


# Convenience function
async def route_campaign(**kwargs) -> Dict[str, Any]:
    """Convenience function to route a campaign."""
    router = CampaignRouter()
    return await router.process_campaign(**kwargs)
