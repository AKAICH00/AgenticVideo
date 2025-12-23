"""
Trend Analyzer - Detects what's going viral NOW

Uses YouTube Data API v3 to:
1. Query trending videos in specific niches
2. Analyze common patterns (hooks, duration, thumbnails)
3. Calculate trend velocity (how fast something is growing)
4. Generate actionable content recommendations
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import asyncpg
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

# Niche-specific search queries for trend detection
NICHE_QUERIES = {
    "tech": ["tech news", "ai update", "gadget review", "coding tutorial"],
    "finance": ["crypto news", "stock market", "investing tips", "finance explained"],
    "gaming": ["gaming news", "game review", "gameplay", "esports"],
    "lifestyle": ["productivity", "morning routine", "life hacks", "minimalism"],
    "education": ["explained", "how to", "tutorial", "learn"],
}


class TrendAnalyzer:
    """
    Analyzes YouTube trends to inform content strategy.

    Key insight: YouTube IS Google. We leverage this ecosystem advantage:
    - Native API access (no scraping needed)
    - Real-time trending data
    - Direct integration with Vertex AI for pattern recognition
    """

    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def connect_db(self):
        """Initialize database connection pool."""
        if not self.db_pool:
            self.db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)

    async def analyze_niche_trends(self, niche: str, max_results: int = 50) -> list[dict]:
        """
        Analyze trending content in a specific niche.

        Returns list of trend insights with:
        - topic: What the trend is about
        - trend_score: How strong the trend is (0-100)
        - velocity: How fast it's growing
        - example_videos: Top performing examples
        - recommended_angle: Suggested content approach
        """
        await self.connect_db()

        queries = NICHE_QUERIES.get(niche, [niche])
        all_videos = []

        for query in queries:
            videos = await self._search_trending(query, max_results // len(queries))
            all_videos.extend(videos)

        # Cluster by topic and calculate trend scores
        trends = await self._cluster_and_score(all_videos, niche)

        # Store in database
        await self._store_trends(trends, niche)

        return trends

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _search_trending(self, query: str, max_results: int = 25) -> list[dict]:
        """Search YouTube for recent trending videos using async HTTP."""
        session = await self._get_session()

        # Search for videos
        search_params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "order": "viewCount",
            "publishedAfter": (datetime.utcnow() - timedelta(days=7)).isoformat() + "Z",
            "maxResults": max_results,
            "relevanceLanguage": "en",
            "key": GOOGLE_API_KEY,
        }

        async with session.get(f"{YOUTUBE_API_BASE}/search", params=search_params) as resp:
            if resp.status != 200:
                print(f"YouTube API error: {resp.status}")
                return []
            search_data = await resp.json()

        video_ids = [item["id"]["videoId"] for item in search_data.get("items", [])]
        if not video_ids:
            return []

        # Get video statistics
        stats_params = {
            "part": "statistics,contentDetails,snippet",
            "id": ",".join(video_ids),
            "key": GOOGLE_API_KEY,
        }

        async with session.get(f"{YOUTUBE_API_BASE}/videos", params=stats_params) as resp:
            if resp.status != 200:
                print(f"YouTube stats API error: {resp.status}")
                return []
            stats_data = await resp.json()

        videos = []
        for item in stats_data.get("items", []):
            videos.append({
                "video_id": item["id"],
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "channel_id": item["snippet"]["channelId"],
                "channel_title": item["snippet"]["channelTitle"],
                "published_at": item["snippet"]["publishedAt"],
                "thumbnail_url": item["snippet"]["thumbnails"]["high"]["url"],
                "tags": item["snippet"].get("tags", []),
                "view_count": int(item["statistics"].get("viewCount", 0)),
                "like_count": int(item["statistics"].get("likeCount", 0)),
                "comment_count": int(item["statistics"].get("commentCount", 0)),
                "duration": item["contentDetails"]["duration"],
            })

        return videos

    async def _cluster_and_score(self, videos: list[dict], niche: str) -> list[dict]:
        """
        Cluster videos by topic and calculate trend scores.

        Trend Score Formula:
        score = (views_velocity * 0.4) + (engagement_rate * 0.3) + (recency * 0.3)

        Where:
        - views_velocity = views / hours_since_published
        - engagement_rate = (likes + comments) / views
        - recency = exponential decay from publish time
        """
        from collections import defaultdict
        import re

        # Simple keyword extraction (replace with proper NLP in production)
        topic_videos = defaultdict(list)

        for video in videos:
            # Extract key phrases from title
            title_words = re.findall(r'\b\w{4,}\b', video["title"].lower())
            # Use first significant word as topic (simplified)
            topic = title_words[0] if title_words else "general"
            topic_videos[topic].append(video)

        trends = []
        for topic, vids in topic_videos.items():
            if len(vids) < 2:
                continue

            # Calculate aggregate metrics
            total_views = sum(v["view_count"] for v in vids)
            total_engagement = sum(v["like_count"] + v["comment_count"] for v in vids)
            avg_age_hours = sum(
                (datetime.utcnow() - datetime.fromisoformat(v["published_at"].replace("Z", "+00:00").replace("+00:00", ""))).total_seconds() / 3600
                for v in vids
            ) / len(vids)

            # Trend score calculation
            views_velocity = total_views / max(avg_age_hours, 1)
            engagement_rate = total_engagement / max(total_views, 1)
            recency_score = 100 / (1 + avg_age_hours / 24)  # Decay over days

            trend_score = (
                min(views_velocity / 10000, 100) * 0.4 +
                min(engagement_rate * 1000, 100) * 0.3 +
                recency_score * 0.3
            )

            trends.append({
                "topic": topic,
                "niche": niche,
                "trend_score": round(trend_score, 2),
                "velocity": round(views_velocity, 2),
                "video_count": len(vids),
                "total_views": total_views,
                "example_videos": sorted(vids, key=lambda x: x["view_count"], reverse=True)[:3],
                "detected_at": datetime.utcnow().isoformat(),
            })

        return sorted(trends, key=lambda x: x["trend_score"], reverse=True)

    async def _store_trends(self, trends: list[dict], niche: str):
        """Store detected trends in the database."""
        async with self.db_pool.acquire() as conn:
            for trend in trends:
                await conn.execute(
                    """
                    INSERT INTO trending_topics
                    (platform, niche, topic, trend_score, velocity, source_data)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (platform, niche, topic, (detected_at::date))
                    DO UPDATE SET
                        trend_score = EXCLUDED.trend_score,
                        velocity = EXCLUDED.velocity,
                        source_data = EXCLUDED.source_data
                    """,
                    "youtube",
                    niche,
                    trend["topic"],
                    trend["trend_score"],
                    trend["velocity"],
                    {
                        "video_count": trend["video_count"],
                        "total_views": trend["total_views"],
                        "examples": [v["video_id"] for v in trend["example_videos"]],
                    }
                )

    async def get_top_trends(self, niche: str, limit: int = 10) -> list[dict]:
        """Get top trends from database for a niche."""
        await self.connect_db()

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT topic, trend_score, velocity, source_data, detected_at
                FROM trending_topics
                WHERE platform = 'youtube'
                  AND niche = $1
                  AND detected_at > NOW() - INTERVAL '24 hours'
                ORDER BY trend_score DESC
                LIMIT $2
                """,
                niche,
                limit
            )
            return [dict(row) for row in rows]


async def main():
    """Test the trend analyzer."""
    analyzer = TrendAnalyzer()

    print("Analyzing tech niche trends...")
    trends = await analyzer.analyze_niche_trends("tech", max_results=30)

    print(f"\nFound {len(trends)} trending topics:")
    for i, trend in enumerate(trends[:5], 1):
        print(f"\n{i}. {trend['topic'].upper()}")
        print(f"   Score: {trend['trend_score']:.1f}")
        print(f"   Velocity: {trend['velocity']:.0f} views/hour")
        print(f"   Videos: {trend['video_count']}")
        print(f"   Top example: {trend['example_videos'][0]['title'][:50]}...")


if __name__ == "__main__":
    asyncio.run(main())
