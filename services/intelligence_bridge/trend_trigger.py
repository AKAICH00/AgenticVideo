"""
Trend Trigger - Automated campaign creation based on trend detection.

Monitors trends and automatically triggers video generation when:
1. New high-velocity trend detected
2. Trend score exceeds threshold
3. We haven't covered the topic recently

This enables the "24-hour trend-to-publish" pipeline goal.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Optional, Any
import asyncpg

from services.orchestrator.state import VideoState
from services.orchestrator.graph import VideoGraph
from .campaign_factory import CampaignFactory

logger = logging.getLogger(__name__)


class TrendTrigger:
    """
    Monitors trends and triggers video generation automatically.

    The trigger runs as a background service that:
    1. Polls for new trends periodically
    2. Filters to high-potential opportunities
    3. Checks we haven't already covered the topic
    4. Creates and runs video generation campaigns

    Can run in two modes:
    - Passive: Detects and queues campaigns for manual approval
    - Active: Automatically starts generation (with rate limits)
    """

    def __init__(
        self,
        db_pool: Optional[asyncpg.Pool] = None,
        campaign_factory: Optional[CampaignFactory] = None,
        video_graph: Optional[VideoGraph] = None,
        on_campaign_created: Optional[Callable[[VideoState], None]] = None,
        on_video_complete: Optional[Callable[[VideoState], None]] = None,
    ):
        self.db_pool = db_pool
        self.factory = campaign_factory or CampaignFactory(db_pool)
        self.graph = video_graph or VideoGraph()
        self.on_campaign_created = on_campaign_created
        self.on_video_complete = on_video_complete

        # Configuration
        self.trend_score_threshold = 60.0  # Minimum trend score to trigger
        self.velocity_threshold = 1000.0   # Minimum views/hour
        self.cooldown_hours = 24           # Don't re-cover same topic within this period
        self.max_daily_campaigns = 10      # Rate limit per niche per day
        self.check_interval_seconds = 1800 # 30 minutes

        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._campaigns_created_today: dict[str, int] = {}

    async def connect_db(self):
        """Initialize database connection."""
        if not self.db_pool:
            import os
            self.db_pool = await asyncpg.create_pool(
                os.getenv("DATABASE_URL"),
                min_size=2,
                max_size=10
            )
            self.factory.db_pool = self.db_pool

    async def start(self, niches: list[str], mode: str = "passive"):
        """
        Start monitoring trends for specified niches.

        Args:
            niches: List of niches to monitor
            mode: "passive" (queue only) or "active" (auto-generate)
        """
        self._running = True
        logger.info(f"Starting trend trigger for niches: {niches} (mode: {mode})")

        self._task = asyncio.create_task(
            self._monitor_loop(niches, mode)
        )

    async def stop(self):
        """Stop the trend monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Trend trigger stopped")

    async def _monitor_loop(self, niches: list[str], mode: str):
        """Main monitoring loop."""
        await self.connect_db()

        while self._running:
            try:
                for niche in niches:
                    await self._check_niche(niche, mode)

                await asyncio.sleep(self.check_interval_seconds)

            except Exception as e:
                logger.error(f"Error in trend monitoring loop: {e}")
                await asyncio.sleep(60)  # Brief pause on error

    async def _check_niche(self, niche: str, mode: str):
        """Check trends for a single niche and trigger campaigns."""
        logger.debug(f"Checking trends for niche: {niche}")

        # Get recent trends
        trends = await self._get_actionable_trends(niche)

        for trend in trends:
            # Check rate limit
            if not self._can_create_campaign(niche):
                logger.info(f"Rate limit reached for {niche}, skipping")
                break

            # Check if already covered
            if await self._recently_covered(trend["topic"], niche):
                logger.debug(f"Topic already covered: {trend['topic']}")
                continue

            # Create campaign
            try:
                state = await self.factory.create_from_trend(
                    niche=niche,
                    quality_tier="premium" if mode == "active" else "bulk",
                    format_type="short",  # Shorts for fast turnaround
                )

                self._increment_daily_count(niche)

                if self.on_campaign_created:
                    self.on_campaign_created(state)

                # Auto-generate in active mode
                if mode == "active":
                    asyncio.create_task(self._run_generation(state))

                logger.info(f"Created campaign for trend: {trend['topic']}")

            except Exception as e:
                logger.error(f"Failed to create campaign for {trend['topic']}: {e}")

    async def _get_actionable_trends(self, niche: str) -> list[dict]:
        """Get trends that meet our threshold criteria."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT topic, trend_score, velocity, source_data, detected_at
                FROM trending_topics
                WHERE niche = $1
                  AND detected_at > NOW() - INTERVAL '12 hours'
                  AND trend_score >= $2
                  AND velocity >= $3
                ORDER BY trend_score DESC
                LIMIT 10
                """,
                niche,
                self.trend_score_threshold,
                self.velocity_threshold,
            )
            return [dict(row) for row in rows]

    async def _recently_covered(self, topic: str, niche: str) -> bool:
        """Check if we've already made a video on this topic recently."""
        async with self.db_pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM video_campaigns
                WHERE niche = $1
                  AND LOWER(topic) LIKE LOWER($2)
                  AND created_at > NOW() - make_interval(hours => $3)
                """,
                niche,
                f"%{topic}%",
                self.cooldown_hours,
            )
            return count > 0

    def _can_create_campaign(self, niche: str) -> bool:
        """Check if we can create more campaigns today."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        key = f"{niche}:{today}"
        count = self._campaigns_created_today.get(key, 0)
        return count < self.max_daily_campaigns

    def _increment_daily_count(self, niche: str):
        """Increment the daily campaign counter."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        key = f"{niche}:{today}"
        self._campaigns_created_today[key] = self._campaigns_created_today.get(key, 0) + 1

    async def _run_generation(self, state: VideoState):
        """Run the video generation pipeline for a campaign."""
        try:
            logger.info(f"Starting generation for campaign: {state.campaign_id}")

            final_state = await self.graph.run(state)

            if self.on_video_complete:
                self.on_video_complete(final_state)

            logger.info(f"Generation complete: {state.campaign_id}")

        except Exception as e:
            logger.error(f"Generation failed for {state.campaign_id}: {e}")

    async def trigger_manual(
        self,
        niche: str,
        topic: Optional[str] = None,
        quality_tier: str = "premium",
        format_type: str = "long",
    ) -> VideoState:
        """
        Manually trigger a campaign (bypasses thresholds).

        Args:
            niche: Content niche
            topic: Specific topic (optional, uses top trend if None)
            quality_tier: Quality tier
            format_type: Format type

        Returns:
            Created VideoState
        """
        await self.connect_db()

        if topic:
            # Create brief for specific topic
            brief = {
                "topic": topic,
                "angle": f"Deep dive into {topic}",
                "hooks": [f"Everything you need to know about {topic}"],
                "recommended_format": format_type,
            }
            state = await self.factory.create_from_brief(brief, niche, quality_tier)
        else:
            state = await self.factory.create_from_trend(
                niche=niche,
                quality_tier=quality_tier,
                format_type=format_type,
            )

        return state


class TrendScheduler:
    """
    Schedule trend analysis and campaign creation at optimal times.

    Timing matters for viral content:
    - Peak engagement hours vary by niche
    - Trending topics peak at specific times
    - Publishing too early/late misses the wave
    """

    PEAK_HOURS = {
        "tech": [9, 10, 14, 15, 20, 21],      # Workday peaks + evening
        "finance": [7, 8, 9, 16, 17],          # Market hours
        "gaming": [18, 19, 20, 21, 22],        # After work/school
        "lifestyle": [7, 8, 12, 18, 19, 20],   # Morning + lunch + evening
        "education": [10, 11, 14, 15, 16],     # School hours
    }

    def __init__(self, trigger: TrendTrigger):
        self.trigger = trigger

    def get_next_optimal_time(self, niche: str) -> datetime:
        """Get the next optimal time to publish for a niche."""
        now = datetime.utcnow()
        peak_hours = self.PEAK_HOURS.get(niche, [10, 14, 18, 20])

        for hour in sorted(peak_hours):
            if hour > now.hour:
                return now.replace(hour=hour, minute=0, second=0, microsecond=0)

        # Next day first peak
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=peak_hours[0], minute=0, second=0, microsecond=0)

    def should_analyze_now(self, niche: str) -> bool:
        """Check if now is a good time to analyze trends for a niche."""
        # Analyze 2-4 hours before peak to allow production time
        peak_hours = self.PEAK_HOURS.get(niche, [10, 14, 18, 20])
        current_hour = datetime.utcnow().hour

        for peak in peak_hours:
            lead_time = peak - current_hour
            if 2 <= lead_time <= 4:
                return True

        return False
