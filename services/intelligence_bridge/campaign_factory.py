"""
Campaign Factory - Creates video campaigns from intelligence data.

Takes trend analysis + content briefs and creates ready-to-generate VideoState instances.
This is the core integration between intelligence layer and generation pipeline.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4
import asyncpg

from services.orchestrator.state import VideoState, GenerationPhase
from services.youtube_intelligence.trend_analyzer import TrendAnalyzer
from services.strategy_engine.content_planner import ContentPlanner, VIRAL_PARAMS

logger = logging.getLogger(__name__)


class CampaignFactory:
    """
    Factory that creates video campaigns from trend intelligence.

    The factory:
    1. Queries trend analyzer for hot topics
    2. Gets content briefs from strategy engine
    3. Creates VideoState with pre-populated intelligence
    4. Optionally stores campaign in database for tracking
    """

    def __init__(
        self,
        db_pool: Optional[asyncpg.Pool] = None,
        trend_analyzer: Optional[TrendAnalyzer] = None,
        content_planner: Optional[ContentPlanner] = None,
    ):
        self.db_pool = db_pool
        self.trend_analyzer = trend_analyzer or TrendAnalyzer(db_pool)
        self.content_planner = content_planner or ContentPlanner(db_pool)

    async def connect_db(self):
        """Initialize database connection if not provided."""
        if not self.db_pool:
            import os
            self.db_pool = await asyncpg.create_pool(
                os.getenv("DATABASE_URL"),
                min_size=2,
                max_size=10
            )
            self.trend_analyzer.db_pool = self.db_pool
            self.content_planner.db_pool = self.db_pool

    async def create_from_trend(
        self,
        niche: str,
        quality_tier: str = "premium",
        format_type: str = "long",  # "long", "short", or "both"
        auto_select_best: bool = True,
    ) -> VideoState:
        """
        Create a campaign from the current top trend in a niche.

        This is the main entry point for trend-driven content.

        Args:
            niche: Content niche (tech, finance, gaming, etc.)
            quality_tier: "premium" (Runway/Sora) or "bulk" (local Wan 2.1)
            format_type: Target format - affects duration and style
            auto_select_best: If True, picks highest-priority brief automatically

        Returns:
            VideoState ready for generation pipeline
        """
        await self.connect_db()

        logger.info(f"Creating trend-driven campaign for niche: {niche}")

        # Step 1: Analyze current trends
        trends = await self.trend_analyzer.analyze_niche_trends(niche, max_results=30)

        if not trends:
            logger.warning(f"No trends found for niche: {niche}")
            raise ValueError(f"No trending topics found for {niche}")

        top_trend = trends[0]
        logger.info(f"Top trend: {top_trend['topic']} (score: {top_trend['trend_score']})")

        # Step 2: Generate content briefs
        format_pref = "short" if format_type == "short" else "long" if format_type == "long" else "both"
        briefs = await self.content_planner.generate_content_plan(
            niche=niche,
            num_recommendations=5,
            format_preference=format_pref,
        )

        if not briefs:
            logger.warning("No content briefs generated, using trend data directly")
            brief = self._create_fallback_brief(top_trend, format_type)
        else:
            brief = briefs[0] if auto_select_best else briefs

        # Step 3: Determine video parameters based on format
        params = self._get_format_params(format_type, quality_tier)

        # Step 4: Create VideoState with intelligence context
        state = VideoState(
            campaign_id=str(uuid4()),
            topic=brief.get("topic", top_trend["topic"]),
            niche=niche,
            target_duration_seconds=params["duration"],
            quality_tier=quality_tier,
            style_reference=brief.get("format_reasoning", "engaging, high-energy"),
            target_audience=f"{niche} enthusiasts",
            phase=GenerationPhase.PENDING,

            # Intelligence-injected metadata
            meta={
                "intelligence_source": "trend_analyzer",
                "trend_score": top_trend.get("trend_score", 0),
                "trend_velocity": top_trend.get("velocity", 0),
                "opportunity_type": brief.get("opportunity_type", "trending_gap"),
                "brief": brief,
                "hooks": brief.get("hooks", []),
                "title_options": brief.get("title_options", []),
                "thumbnail_concept": brief.get("thumbnail_concept", ""),
                "estimated_virality": brief.get("estimated_virality", 5),
                "platform_params": params,
            }
        )

        # Step 5: Store campaign in database
        await self._store_campaign(state)

        logger.info(f"Created campaign {state.campaign_id} for topic: {state.topic}")
        return state

    async def create_from_brief(
        self,
        brief: dict,
        niche: str,
        quality_tier: str = "premium",
    ) -> VideoState:
        """
        Create a campaign from a specific content brief.

        Use when you want to generate a specific piece of content
        rather than auto-selecting from trends.

        Args:
            brief: Content brief dict from ContentPlanner
            niche: Content niche
            quality_tier: Quality tier for generation

        Returns:
            VideoState ready for generation
        """
        await self.connect_db()

        format_type = brief.get("recommended_format", "long")
        params = self._get_format_params(format_type, quality_tier)

        state = VideoState(
            campaign_id=str(uuid4()),
            topic=brief["topic"],
            niche=niche,
            target_duration_seconds=params["duration"],
            quality_tier=quality_tier,
            style_reference=brief.get("format_reasoning", ""),
            phase=GenerationPhase.PENDING,
            meta={
                "intelligence_source": "content_planner",
                "brief": brief,
                "hooks": brief.get("hooks", []),
                "title_options": brief.get("title_options", []),
                "thumbnail_concept": brief.get("thumbnail_concept", ""),
                "platform_params": params,
            }
        )

        await self._store_campaign(state)
        return state

    async def create_batch_from_trends(
        self,
        niche: str,
        count: int = 3,
        quality_tier: str = "bulk",
        format_type: str = "short",
    ) -> list[VideoState]:
        """
        Create multiple campaigns from top trends.

        Ideal for bulk content production with local GPU.

        Args:
            niche: Content niche
            count: Number of campaigns to create
            quality_tier: Quality tier (bulk recommended for volume)
            format_type: short/long/both

        Returns:
            List of VideoState instances
        """
        await self.connect_db()

        briefs = await self.content_planner.generate_content_plan(
            niche=niche,
            num_recommendations=count,
            format_preference=format_type if format_type != "both" else "both",
        )

        campaigns = []
        for brief in briefs[:count]:
            state = await self.create_from_brief(brief, niche, quality_tier)
            campaigns.append(state)

        logger.info(f"Created batch of {len(campaigns)} campaigns for {niche}")
        return campaigns

    def _get_format_params(self, format_type: str, quality_tier: str) -> dict:
        """Get platform-optimized parameters for video format."""
        if format_type == "short":
            params = VIRAL_PARAMS.get("youtube_shorts", {})
            duration_range = params.get("optimal_duration_seconds", (45, 58))
            return {
                "duration": duration_range[1],  # Use max for shorts
                "hook_window": params.get("hook_window_seconds", 1.5),
                "vertical": True,
                "platform": "shorts",
            }
        else:
            params = VIRAL_PARAMS.get("youtube_long", {})
            duration_range = params.get("optimal_duration_seconds", (480, 720))
            # For premium, use longer; for bulk, use shorter
            duration = duration_range[1] if quality_tier == "premium" else duration_range[0]
            return {
                "duration": duration,
                "hook_window": params.get("hook_window_seconds", 8),
                "vertical": False,
                "platform": "youtube",
                "pattern_interrupts": params.get("pattern_interrupts_per_minute", 2),
            }

    def _create_fallback_brief(self, trend: dict, format_type: str) -> dict:
        """Create a basic brief when content planner returns nothing."""
        return {
            "topic": trend["topic"],
            "angle": f"Breaking down {trend['topic']}",
            "hooks": [
                f"You won't believe what's happening with {trend['topic']}",
                f"Here's what everyone's getting wrong about {trend['topic']}",
                f"The truth about {trend['topic']} that nobody talks about",
            ],
            "recommended_format": format_type,
            "format_reasoning": f"Trending topic with velocity {trend.get('velocity', 0)} views/hour",
            "title_options": [
                f"What's Really Happening with {trend['topic']}",
                f"The {trend['topic']} Situation Explained",
                f"{trend['topic']} - Everything You Need to Know",
            ],
            "priority_score": trend.get("trend_score", 50),
        }

    async def _store_campaign(self, state: VideoState):
        """Store campaign in database for tracking."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO video_campaigns
                (id, topic, niche, target_duration, quality_tier, status, meta, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    meta = EXCLUDED.meta
                """,
                state.campaign_id,
                state.topic,
                state.niche,
                state.target_duration_seconds,
                state.quality_tier,
                state.phase.value,
                state.meta,
                datetime.utcnow(),
            )
