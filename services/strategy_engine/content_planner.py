"""
Content Planner - Decides WHAT viral content to produce

Combines:
1. Trend data (what's hot right now)
2. Performance history (what worked for us)
3. Competitor gaps (what's missing in the market)
4. Resource constraints (what we can produce)

Output: Prioritized content recommendations with full briefs
"""

import os
import asyncio
import json
from datetime import datetime
from typing import Optional
import asyncpg
from google import genai
from google.genai import types

DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class ContentPlanner:
    """
    Intelligent content planning that maximizes viral potential.

    Key Insight: Viral content isn't random. It follows patterns:
    1. Timing: First to cover a trend wins
    2. Angle: Unique perspective on common topic
    3. Hook: First 3 seconds determine everything
    4. Value: Must deliver on the hook's promise
    """

    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    async def connect_db(self):
        if not self.db_pool:
            self.db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)

    async def generate_content_plan(
        self,
        niche: str,
        num_recommendations: int = 5,
        format_preference: str = "both"  # "short", "long", or "both"
    ) -> list[dict]:
        """
        Generate prioritized content recommendations.

        Returns list of content briefs with:
        - topic: What to cover
        - angle: Unique perspective
        - hook_suggestions: Opening line options
        - format: short-form or long-form
        - priority_score: How urgent/important
        - reasoning: Why this content
        """
        await self.connect_db()

        # 1. Get current trends
        trends = await self._get_current_trends(niche)

        # 2. Get our performance history
        performance = await self._get_performance_history(niche)

        # 3. Identify gaps and opportunities
        opportunities = await self._identify_opportunities(trends, performance)

        # 4. Generate detailed briefs with Claude
        briefs = await self._generate_briefs(opportunities, niche, format_preference)

        # 5. Store recommendations
        await self._store_recommendations(briefs, niche)

        return briefs[:num_recommendations]

    async def _get_current_trends(self, niche: str) -> list[dict]:
        """Fetch recent trends from database."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT topic, trend_score, velocity, source_data
                FROM trending_topics
                WHERE niche = $1
                  AND detected_at > NOW() - INTERVAL '48 hours'
                ORDER BY trend_score DESC
                LIMIT 20
                """,
                niche
            )
            return [dict(row) for row in rows]

    async def _get_performance_history(self, niche: str) -> list[dict]:
        """Get our best and worst performing content."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.topic,
                    c.meta,
                    p.view_count,
                    p.avg_view_percentage,
                    p.click_through_rate,
                    g.api_cost_usd
                FROM video_campaigns c
                JOIN video_performance p ON p.campaign_id = c.id
                LEFT JOIN generation_costs g ON g.campaign_id = c.id
                WHERE c.niche = $1
                  AND c.status = 'published'
                ORDER BY p.view_count DESC
                LIMIT 50
                """,
                niche
            )
            return [dict(row) for row in rows]

    async def _identify_opportunities(
        self,
        trends: list[dict],
        performance: list[dict]
    ) -> list[dict]:
        """
        Identify content opportunities by analyzing:
        1. High-trend topics we haven't covered
        2. Topics where our content outperformed
        3. Gaps in competitor coverage
        """
        opportunities = []

        # Topics we've already covered
        covered_topics = {p["topic"].lower() for p in performance}

        # High-performing patterns from our content
        high_performers = [p for p in performance if p.get("view_count", 0) > 10000]
        successful_patterns = self._extract_patterns(high_performers)

        for trend in trends:
            topic = trend["topic"]
            score = trend["trend_score"]

            # Opportunity: Trending topic we haven't covered
            if topic.lower() not in covered_topics:
                opportunities.append({
                    "topic": topic,
                    "opportunity_type": "trending_gap",
                    "trend_score": score,
                    "velocity": trend["velocity"],
                    "priority": score * 1.5,  # Boost for being new
                    "source_data": trend["source_data"],
                })

            # Opportunity: Trending topic that matches our successful patterns
            elif self._matches_patterns(topic, successful_patterns):
                opportunities.append({
                    "topic": topic,
                    "opportunity_type": "proven_success",
                    "trend_score": score,
                    "priority": score * 1.2,  # Boost for proven success
                    "matched_patterns": successful_patterns,
                })

        return sorted(opportunities, key=lambda x: x["priority"], reverse=True)

    def _extract_patterns(self, high_performers: list[dict]) -> list[str]:
        """Extract common patterns from high-performing content."""
        # Simplified - in production use NLP
        patterns = []
        for p in high_performers:
            topic = p.get("topic", "").lower()
            if "how to" in topic:
                patterns.append("how_to")
            if "vs" in topic or "versus" in topic:
                patterns.append("comparison")
            if any(w in topic for w in ["best", "top", "ultimate"]):
                patterns.append("listicle")
        return list(set(patterns))

    def _matches_patterns(self, topic: str, patterns: list[str]) -> bool:
        """Check if topic matches successful patterns."""
        topic_lower = topic.lower()
        for pattern in patterns:
            if pattern == "how_to" and "how" in topic_lower:
                return True
            if pattern == "comparison" and ("vs" in topic_lower or "compare" in topic_lower):
                return True
            if pattern == "listicle" and any(w in topic_lower for w in ["best", "top"]):
                return True
        return False

    async def _generate_briefs(
        self,
        opportunities: list[dict],
        niche: str,
        format_preference: str
    ) -> list[dict]:
        """Use Claude to generate detailed content briefs."""

        briefs = []

        for opp in opportunities[:10]:  # Process top 10 opportunities
            prompt = f"""You are a viral content strategist. Generate a detailed content brief for this opportunity:

NICHE: {niche}
TOPIC: {opp['topic']}
OPPORTUNITY TYPE: {opp['opportunity_type']}
TREND SCORE: {opp.get('trend_score', 'N/A')}
VELOCITY: {opp.get('velocity', 'N/A')} views/hour

Generate a brief with:
1. ANGLE: A unique perspective that differentiates from competitors
2. HOOK: 3 different opening lines (first 3 seconds are crucial)
3. KEY POINTS: 3-5 main points to cover
4. FORMAT: Recommend short-form (30-60s) or long-form (8-15min) with reasoning
5. THUMBNAIL CONCEPT: Visual description for the thumbnail
6. TITLE OPTIONS: 3 title options optimized for CTR

Respond in JSON format:
{{
    "topic": "...",
    "angle": "...",
    "hooks": ["...", "...", "..."],
    "key_points": ["...", "..."],
    "recommended_format": "short" or "long",
    "format_reasoning": "...",
    "thumbnail_concept": "...",
    "title_options": ["...", "...", "..."],
    "estimated_virality": 1-10,
    "production_notes": "..."
}}"""

            try:
                # Use Gemini for brief generation
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=1024,
                    )
                )

                content = response.text

                # Extract JSON from response
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0]
                else:
                    json_str = content

                brief = json.loads(json_str.strip())
                brief["priority_score"] = opp["priority"]
                brief["opportunity_type"] = opp["opportunity_type"]
                brief["trend_data"] = opp.get("source_data", {})

                # Filter by format preference
                if format_preference != "both":
                    if brief["recommended_format"] != format_preference:
                        continue

                briefs.append(brief)

            except Exception as e:
                print(f"Error generating brief for {opp['topic']}: {e}")
                continue

        return sorted(briefs, key=lambda x: x.get("priority_score", 0), reverse=True)

    async def _store_recommendations(self, briefs: list[dict], niche: str):
        """Store recommendations in database for tracking."""
        async with self.db_pool.acquire() as conn:
            for brief in briefs:
                await conn.execute(
                    """
                    INSERT INTO content_recommendations
                    (niche, recommended_topic, reasoning, priority_score)
                    VALUES ($1, $2, $3, $4)
                    """,
                    niche,
                    brief["topic"],
                    brief.get("format_reasoning", ""),
                    brief.get("priority_score", 0),
                )


# Viral Content Optimization Parameters
VIRAL_PARAMS = {
    "youtube_long": {
        "optimal_duration_seconds": (480, 720),  # 8-12 minutes
        "hook_window_seconds": 8,
        "pattern_interrupts_per_minute": 2,
        "cta_placement": 0.7,  # 70% through video
        "thumbnail_requirements": ["face", "contrast", "text_overlay"],
        "title_length_chars": (45, 60),
    },
    "youtube_shorts": {
        "optimal_duration_seconds": (45, 58),  # Just under 60s limit
        "hook_window_seconds": 1.5,
        "text_on_screen": True,
        "loop_compatibility": True,
        "vertical_only": True,
    },
    "tiktok": {
        "optimal_duration_seconds": (21, 34),  # Algorithm sweet spot
        "hook_window_seconds": 0.8,
        "trending_sound_recommended": True,
        "vertical_only": True,
    },
}


async def main():
    """Test the content planner."""
    planner = ContentPlanner()

    print("Generating content plan for tech niche...")
    plan = await planner.generate_content_plan("tech", num_recommendations=3)

    print(f"\nGenerated {len(plan)} content briefs:")
    for i, brief in enumerate(plan, 1):
        print(f"\n{'='*50}")
        print(f"BRIEF {i}: {brief['topic']}")
        print(f"{'='*50}")
        print(f"Angle: {brief['angle']}")
        print(f"Format: {brief['recommended_format']}")
        print(f"Priority: {brief['priority_score']:.1f}")
        print(f"\nHooks:")
        for hook in brief['hooks']:
            print(f"  - {hook}")
        print(f"\nTitle Options:")
        for title in brief['title_options']:
            print(f"  - {title}")


if __name__ == "__main__":
    asyncio.run(main())
