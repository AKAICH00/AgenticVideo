"""
Feedback Loop - Closes the intelligence cycle by learning from performance.

After videos are published, this service:
1. Tracks video performance metrics (views, CTR, retention)
2. Correlates performance with content characteristics
3. Updates strategy engine with learnings
4. Adjusts trend scoring based on what works

This enables continuous optimization of content strategy.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
import asyncpg
import aiohttp

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """
    Collects performance data and feeds learnings back to strategy.

    The loop analyzes:
    - Which topics performed well vs predicted
    - What hooks drove highest retention
    - Which thumbnails got best CTR
    - What content patterns we should repeat

    These insights improve future content recommendations.
    """

    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self._session: Optional[aiohttp.ClientSession] = None

        # YouTube API config
        self.youtube_api_key = None
        self.youtube_api_base = "https://www.googleapis.com/youtube/v3"

    async def connect_db(self):
        """Initialize database connection."""
        if not self.db_pool:
            import os
            self.db_pool = await asyncpg.create_pool(
                os.getenv("DATABASE_URL"),
                min_size=2,
                max_size=10
            )
            self.youtube_api_key = os.getenv("GOOGLE_API_KEY")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close resources."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def collect_performance(
        self,
        video_ids: list[str],
        youtube_channel_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Collect performance data for published videos.

        Args:
            video_ids: YouTube video IDs to check
            youtube_channel_id: Channel ID for analytics access

        Returns:
            List of performance metrics per video
        """
        await self.connect_db()
        session = await self._get_session()

        results = []

        # Batch videos into groups of 50 (API limit)
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i + 50]

            params = {
                "part": "statistics,snippet",
                "id": ",".join(batch),
                "key": self.youtube_api_key,
            }

            try:
                async with session.get(
                    f"{self.youtube_api_base}/videos",
                    params=params
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"YouTube API error: {resp.status}")
                        continue

                    data = await resp.json()

                    for item in data.get("items", []):
                        stats = item.get("statistics", {})
                        results.append({
                            "video_id": item["id"],
                            "title": item["snippet"]["title"],
                            "published_at": item["snippet"]["publishedAt"],
                            "view_count": int(stats.get("viewCount", 0)),
                            "like_count": int(stats.get("likeCount", 0)),
                            "comment_count": int(stats.get("commentCount", 0)),
                            "collected_at": datetime.utcnow().isoformat(),
                        })

            except Exception as e:
                logger.error(f"Failed to fetch video stats: {e}")

        return results

    async def update_campaign_performance(self, campaign_id: str, video_id: str):
        """
        Update a campaign with its video's performance data.

        This links our internal campaign tracking with actual YouTube metrics.
        """
        await self.connect_db()

        perf_data = await self.collect_performance([video_id])
        if not perf_data:
            logger.warning(f"No performance data found for video: {video_id}")
            return

        perf = perf_data[0]

        # Calculate engagement rate
        views = perf["view_count"]
        engagement = (perf["like_count"] + perf["comment_count"]) / max(views, 1)

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO video_performance
                (campaign_id, platform, external_video_id, youtube_video_id, view_count,
                 like_count, comment_count, collected_at, engagement_rate)
                VALUES ($1, 'youtube', $2, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (campaign_id, platform, collected_at)
                DO UPDATE SET
                    view_count = EXCLUDED.view_count,
                    like_count = EXCLUDED.like_count,
                    comment_count = EXCLUDED.comment_count,
                    engagement_rate = EXCLUDED.engagement_rate
                """,
                campaign_id,
                video_id,
                perf["view_count"],
                perf["like_count"],
                perf["comment_count"],
                datetime.utcnow(),
                engagement,
            )

        logger.info(f"Updated performance for campaign {campaign_id}: {views} views")

    async def analyze_content_patterns(self, niche: str, days: int = 30) -> dict:
        """
        Analyze what content patterns worked best in a niche.

        Returns insights on:
        - Best performing topics
        - Optimal video duration
        - Hook patterns that worked
        - Thumbnail styles that got clicks

        These insights inform future content strategy.
        """
        await self.connect_db()

        async with self.db_pool.acquire() as conn:
            # Get top performing campaigns
            top_performers = await conn.fetch(
                """
                SELECT
                    c.id,
                    c.topic,
                    c.meta,
                    c.target_duration,
                    p.view_count,
                    p.engagement_rate,
                    g.api_cost_usd
                FROM video_campaigns c
                JOIN video_performance p ON p.campaign_id = c.id
                LEFT JOIN generation_costs g ON g.campaign_id = c.id
                WHERE c.niche = $1
                  AND c.created_at > NOW() - make_interval(days => $2)
                  AND p.view_count > 0
                ORDER BY p.view_count DESC
                LIMIT 20
                """,
                niche,
                days,
            )

            # Get average performance
            avg_performance = await conn.fetchrow(
                """
                SELECT
                    AVG(p.view_count) as avg_views,
                    AVG(p.engagement_rate) as avg_engagement,
                    AVG(g.api_cost_usd) as avg_cost
                FROM video_campaigns c
                JOIN video_performance p ON p.campaign_id = c.id
                LEFT JOIN generation_costs g ON g.campaign_id = c.id
                WHERE c.niche = $1
                  AND c.created_at > NOW() - make_interval(days => $2)
                """,
                niche,
                days,
            )

        # Analyze patterns
        analysis = {
            "niche": niche,
            "period_days": days,
            "total_campaigns": len(top_performers),
            "avg_views": avg_performance["avg_views"] or 0,
            "avg_engagement": avg_performance["avg_engagement"] or 0,
            "avg_cost": avg_performance["avg_cost"] or 0,
            "top_topics": [],
            "successful_patterns": [],
            "recommendations": [],
        }

        # Extract patterns from top performers
        for row in top_performers[:5]:
            analysis["top_topics"].append({
                "topic": row["topic"],
                "views": row["view_count"],
                "engagement": row["engagement_rate"],
                "roi": row["view_count"] / max(row["api_cost_usd"] or 0.01, 0.01),
            })

            # Extract patterns from meta
            meta = row["meta"] or {}
            if "hooks" in meta and meta["hooks"]:
                analysis["successful_patterns"].extend(meta["hooks"][:1])

        return analysis

    async def generate_strategy_update(self, niche: str) -> dict:
        """
        Generate updated strategy recommendations based on performance data.

        This is the key feedback output - it tells the strategy engine
        what content to prioritize based on what actually performed.
        """
        await self.connect_db()

        analysis = await self.analyze_content_patterns(niche)

        # Calculate performance benchmarks
        async with self.db_pool.acquire() as conn:
            # Get underperforming topics to avoid
            underperformers = await conn.fetch(
                """
                SELECT c.topic, AVG(p.view_count) as avg_views
                FROM video_campaigns c
                JOIN video_performance p ON p.campaign_id = c.id
                WHERE c.niche = $1
                  AND c.created_at > NOW() - INTERVAL '30 days'
                GROUP BY c.topic
                HAVING AVG(p.view_count) < 1000
                ORDER BY avg_views ASC
                LIMIT 10
                """,
                niche,
            )

            # Get high-ROI content
            high_roi = await conn.fetch(
                """
                SELECT
                    c.topic,
                    p.view_count,
                    g.api_cost_usd,
                    p.view_count / NULLIF(g.api_cost_usd, 0) as roi
                FROM video_campaigns c
                JOIN video_performance p ON p.campaign_id = c.id
                JOIN generation_costs g ON g.campaign_id = c.id
                WHERE c.niche = $1
                  AND c.created_at > NOW() - INTERVAL '30 days'
                  AND g.api_cost_usd > 0
                ORDER BY roi DESC
                LIMIT 10
                """,
                niche,
            )

        strategy_update = {
            "niche": niche,
            "generated_at": datetime.utcnow().isoformat(),

            # Topics to prioritize
            "prioritize_topics": [
                t["topic"] for t in analysis["top_topics"][:5]
            ],

            # Topics to avoid
            "avoid_topics": [
                row["topic"] for row in underperformers
            ],

            # Pattern recommendations
            "recommended_patterns": analysis["successful_patterns"],

            # Performance benchmarks
            "benchmarks": {
                "target_views": analysis["avg_views"] * 1.2,  # 20% above average
                "target_engagement": analysis["avg_engagement"] * 1.1,
                "max_cost_per_video": analysis["avg_cost"] * 0.9,  # 10% below average
            },

            # High-ROI insights
            "high_roi_patterns": [
                {"topic": row["topic"], "roi": row["roi"]}
                for row in high_roi[:5]
            ],
        }

        # Store strategy update
        await self._store_strategy_update(niche, strategy_update)

        return strategy_update

    async def _store_strategy_update(self, niche: str, update: dict):
        """Store strategy update for the content planner to use."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO content_recommendations
                (niche, recommended_topic, reasoning, priority_score)
                VALUES ($1, $2, $3, $4)
                """,
                niche,
                f"Strategy Update: {datetime.utcnow().strftime('%Y-%m-%d')}",
                str(update),
                100,  # High priority for strategy updates
            )

    async def run_feedback_cycle(self, niche: str):
        """
        Run a complete feedback cycle for a niche.

        1. Collect latest performance data
        2. Analyze patterns
        3. Generate strategy update
        4. Log insights
        """
        await self.connect_db()

        logger.info(f"Running feedback cycle for niche: {niche}")

        # Get campaigns needing performance update
        async with self.db_pool.acquire() as conn:
            campaigns = await conn.fetch(
                """
                SELECT c.id, c.meta->>'youtube_video_id' as video_id
                FROM video_campaigns c
                LEFT JOIN video_performance p ON p.campaign_id = c.id
                WHERE c.niche = $1
                  AND c.status = 'published'
                  AND (
                      p.collected_at IS NULL
                      OR p.collected_at < NOW() - INTERVAL '24 hours'
                  )
                LIMIT 50
                """,
                niche,
            )

        # Update performance for each
        for campaign in campaigns:
            if campaign["video_id"]:
                await self.update_campaign_performance(
                    campaign["id"],
                    campaign["video_id"]
                )

        # Generate strategy update
        strategy = await self.generate_strategy_update(niche)

        logger.info(
            f"Feedback cycle complete for {niche}: "
            f"{len(strategy['prioritize_topics'])} topics prioritized, "
            f"{len(strategy['avoid_topics'])} topics to avoid"
        )

        return strategy


class PerformanceTracker:
    """
    Background service that continuously tracks video performance.

    Runs as a scheduled job to:
    1. Poll YouTube API for performance updates
    2. Store metrics in time-series format
    3. Detect performance anomalies (viral moments, drops)
    4. Alert on significant events
    """

    def __init__(
        self,
        feedback_loop: FeedbackLoop,
        niches: list[str],
        poll_interval_seconds: int = 3600,  # 1 hour
    ):
        self.feedback = feedback_loop
        self.niches = niches
        self.poll_interval = poll_interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the performance tracking loop."""
        self._running = True
        self._task = asyncio.create_task(self._tracking_loop())
        logger.info(f"Performance tracker started for niches: {self.niches}")

    async def stop(self):
        """Stop the tracking loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _tracking_loop(self):
        """Main tracking loop."""
        while self._running:
            try:
                for niche in self.niches:
                    await self.feedback.run_feedback_cycle(niche)

                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Error in performance tracking: {e}")
                await asyncio.sleep(60)
