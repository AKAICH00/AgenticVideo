"""
Posting Scheduler

Calculates optimal posting times for YouTube videos based on:
- Niche-specific audience behavior
- Historical performance data
- Time zone targeting
- Platform-specific best practices
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Literal
from enum import Enum
import asyncpg

logger = logging.getLogger(__name__)


class DayOfWeek(Enum):
    """Days of the week."""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


@dataclass
class TimeSlot:
    """A time slot for posting."""
    day: DayOfWeek
    hour: int  # 0-23 in target timezone
    engagement_score: float = 0.0
    historical_views: int = 0
    competition_level: float = 0.0  # 0-1, lower is better

    @property
    def weighted_score(self) -> float:
        """Calculate weighted score for this slot."""
        # Higher engagement + lower competition = better
        return self.engagement_score * (1 - self.competition_level * 0.3)


@dataclass
class ScheduleRecommendation:
    """Recommended posting schedule."""
    scheduled_time: datetime
    timezone: str
    confidence_score: float
    reasoning: str
    alternative_times: list[datetime] = field(default_factory=list)
    niche_factors: dict = field(default_factory=dict)


# Research-backed optimal posting times by niche (UTC)
# Based on aggregate data from YouTube creator analytics
NICHE_OPTIMAL_TIMES = {
    "tech": {
        # Tech audience: professionals checking during work breaks
        "best_days": [DayOfWeek.TUESDAY, DayOfWeek.WEDNESDAY, DayOfWeek.THURSDAY],
        "best_hours": [9, 12, 17, 20],  # Morning, lunch, after work, evening
        "avoid_hours": [0, 1, 2, 3, 4, 5],  # Late night
        "weekend_modifier": 0.7,  # Lower engagement on weekends
    },
    "finance": {
        # Finance: market hours and early morning
        "best_days": [DayOfWeek.MONDAY, DayOfWeek.TUESDAY, DayOfWeek.WEDNESDAY],
        "best_hours": [7, 8, 12, 18],  # Pre-market, market open, lunch, close
        "avoid_hours": [22, 23, 0, 1, 2, 3, 4, 5],
        "weekend_modifier": 0.5,  # Much lower on weekends
    },
    "gaming": {
        # Gaming: evenings and weekends
        "best_days": [DayOfWeek.FRIDAY, DayOfWeek.SATURDAY, DayOfWeek.SUNDAY],
        "best_hours": [15, 16, 17, 18, 19, 20, 21],  # Afternoon to night
        "avoid_hours": [6, 7, 8, 9],  # Early morning
        "weekend_modifier": 1.3,  # Higher engagement on weekends
    },
    "education": {
        # Education: study times
        "best_days": [DayOfWeek.SUNDAY, DayOfWeek.MONDAY, DayOfWeek.TUESDAY],
        "best_hours": [10, 14, 15, 19, 20],  # Study breaks
        "avoid_hours": [0, 1, 2, 3, 4, 5, 23],
        "weekend_modifier": 1.1,
    },
    "entertainment": {
        # Entertainment: leisure time
        "best_days": [DayOfWeek.THURSDAY, DayOfWeek.FRIDAY, DayOfWeek.SATURDAY],
        "best_hours": [12, 17, 18, 19, 20, 21],
        "avoid_hours": [3, 4, 5, 6],
        "weekend_modifier": 1.2,
    },
    "default": {
        "best_days": [DayOfWeek.TUESDAY, DayOfWeek.THURSDAY, DayOfWeek.FRIDAY],
        "best_hours": [12, 15, 17, 20],
        "avoid_hours": [1, 2, 3, 4, 5],
        "weekend_modifier": 0.9,
    },
}

# Shorts-specific adjustments (higher engagement at different times)
SHORTS_ADJUSTMENTS = {
    "boost_hours": [7, 8, 12, 13, 18, 19, 21, 22],  # Commute, lunch, evening scroll
    "boost_factor": 1.2,
}


class PostingScheduler:
    """
    Calculates optimal posting times for YouTube videos.

    Uses a combination of:
    - Niche-specific research data
    - Historical performance from database
    - Time zone targeting
    - Competition analysis
    """

    def __init__(
        self,
        db_pool: Optional[asyncpg.Pool] = None,
        target_timezone: str = "America/New_York",
    ):
        self.db_pool = db_pool
        self.target_timezone = target_timezone

    async def get_optimal_time(
        self,
        niche: str,
        format_type: Literal["short", "long"] = "short",
        campaign_id: Optional[str] = None,
        min_hours_ahead: int = 1,
        max_hours_ahead: int = 168,  # 1 week
    ) -> ScheduleRecommendation:
        """
        Get the optimal posting time for a video.

        Args:
            niche: Content niche (tech, finance, gaming, etc.)
            format_type: Video format (short or long)
            campaign_id: Optional campaign ID for historical lookup
            min_hours_ahead: Minimum hours from now to schedule
            max_hours_ahead: Maximum hours from now to schedule

        Returns:
            ScheduleRecommendation with optimal time and alternatives
        """
        now = datetime.now(timezone.utc)
        min_time = now + timedelta(hours=min_hours_ahead)
        max_time = now + timedelta(hours=max_hours_ahead)

        # Get niche-specific patterns
        niche_patterns = NICHE_OPTIMAL_TIMES.get(niche, NICHE_OPTIMAL_TIMES["default"])

        # Generate candidate time slots
        candidates = self._generate_candidates(
            min_time, max_time, niche_patterns, format_type
        )

        # Score candidates based on multiple factors
        scored_candidates = await self._score_candidates(
            candidates, niche, campaign_id
        )

        # Sort by weighted score
        scored_candidates.sort(key=lambda x: x.weighted_score, reverse=True)

        if not scored_candidates:
            # Fallback to simple scheduling
            return self._fallback_schedule(min_time, niche)

        best_slot = scored_candidates[0]
        alternatives = [self._slot_to_datetime(s, min_time) for s in scored_candidates[1:4]]

        return ScheduleRecommendation(
            scheduled_time=self._slot_to_datetime(best_slot, min_time),
            timezone=self.target_timezone,
            confidence_score=min(best_slot.weighted_score / 10, 1.0),
            reasoning=self._generate_reasoning(best_slot, niche, format_type),
            alternative_times=alternatives,
            niche_factors={
                "best_days": [d.name for d in niche_patterns["best_days"]],
                "best_hours": niche_patterns["best_hours"],
                "weekend_modifier": niche_patterns["weekend_modifier"],
            },
        )

    def _generate_candidates(
        self,
        min_time: datetime,
        max_time: datetime,
        niche_patterns: dict,
        format_type: str,
    ) -> list[TimeSlot]:
        """Generate candidate time slots within the range."""
        candidates = []

        current = min_time.replace(minute=0, second=0, microsecond=0)

        while current <= max_time:
            day = DayOfWeek(current.weekday())
            hour = current.hour

            # Skip avoided hours
            if hour in niche_patterns.get("avoid_hours", []):
                current += timedelta(hours=1)
                continue

            # Calculate base engagement score
            score = 5.0  # Base score

            # Boost for best days
            if day in niche_patterns.get("best_days", []):
                score += 3.0

            # Boost for best hours
            if hour in niche_patterns.get("best_hours", []):
                score += 2.0

            # Weekend modifier
            if day in (DayOfWeek.SATURDAY, DayOfWeek.SUNDAY):
                score *= niche_patterns.get("weekend_modifier", 1.0)

            # Shorts-specific adjustments
            if format_type == "short":
                if hour in SHORTS_ADJUSTMENTS["boost_hours"]:
                    score *= SHORTS_ADJUSTMENTS["boost_factor"]

            candidates.append(TimeSlot(
                day=day,
                hour=hour,
                engagement_score=score,
            ))

            current += timedelta(hours=1)

        return candidates

    async def _score_candidates(
        self,
        candidates: list[TimeSlot],
        niche: str,
        campaign_id: Optional[str],
    ) -> list[TimeSlot]:
        """Score candidates using historical data if available."""
        if not self.db_pool:
            return candidates

        try:
            async with self.db_pool.acquire() as conn:
                # Get historical performance by hour/day for this niche
                historical = await conn.fetch(
                    """
                    SELECT
                        EXTRACT(DOW FROM created_at) as day_of_week,
                        EXTRACT(HOUR FROM created_at) as hour,
                        AVG(vp.view_count) as avg_views,
                        AVG(vp.engagement_rate) as avg_engagement
                    FROM video_campaigns vc
                    JOIN video_performance vp ON vc.id = vp.campaign_id
                    WHERE vc.niche = $1
                      AND vc.created_at > NOW() - INTERVAL '90 days'
                    GROUP BY 1, 2
                    """,
                    niche,
                )

                if historical:
                    perf_map = {
                        (int(r["day_of_week"]), int(r["hour"])): {
                            "views": r["avg_views"] or 0,
                            "engagement": r["avg_engagement"] or 0,
                        }
                        for r in historical
                    }

                    for candidate in candidates:
                        key = (candidate.day.value, candidate.hour)
                        if key in perf_map:
                            # Boost score based on historical performance
                            hist = perf_map[key]
                            candidate.historical_views = int(hist["views"])
                            if hist["engagement"]:
                                candidate.engagement_score *= (1 + hist["engagement"])

                # Check competition (videos posted at same time by others)
                competition = await conn.fetch(
                    """
                    SELECT
                        EXTRACT(DOW FROM created_at) as day_of_week,
                        EXTRACT(HOUR FROM created_at) as hour,
                        COUNT(*) as video_count
                    FROM video_campaigns
                    WHERE niche = $1
                      AND created_at > NOW() - INTERVAL '30 days'
                    GROUP BY 1, 2
                    """,
                    niche,
                )

                if competition:
                    max_count = max(r["video_count"] for r in competition)
                    comp_map = {
                        (int(r["day_of_week"]), int(r["hour"])): r["video_count"] / max_count
                        for r in competition
                    }

                    for candidate in candidates:
                        key = (candidate.day.value, candidate.hour)
                        candidate.competition_level = comp_map.get(key, 0.5)

        except Exception as e:
            logger.warning(f"Failed to get historical data: {e}")

        return candidates

    def _slot_to_datetime(self, slot: TimeSlot, reference: datetime) -> datetime:
        """Convert a TimeSlot to a specific datetime."""
        # Find the next occurrence of this day/hour
        current = reference.replace(minute=0, second=0, microsecond=0)

        # Move to the target hour
        current = current.replace(hour=slot.hour)

        # Find the next occurrence of the target day
        days_ahead = slot.day.value - current.weekday()
        if days_ahead < 0:
            days_ahead += 7
        if days_ahead == 0 and current <= reference:
            days_ahead += 7

        return current + timedelta(days=days_ahead)

    def _fallback_schedule(self, min_time: datetime, niche: str) -> ScheduleRecommendation:
        """Generate a simple fallback schedule."""
        patterns = NICHE_OPTIMAL_TIMES.get(niche, NICHE_OPTIMAL_TIMES["default"])

        # Find the next best hour
        best_hour = patterns["best_hours"][0] if patterns["best_hours"] else 12

        scheduled = min_time.replace(
            hour=best_hour, minute=0, second=0, microsecond=0
        )

        if scheduled <= min_time:
            scheduled += timedelta(days=1)

        return ScheduleRecommendation(
            scheduled_time=scheduled,
            timezone=self.target_timezone,
            confidence_score=0.5,
            reasoning=f"Scheduled for {best_hour}:00 based on {niche} niche patterns",
            alternative_times=[
                scheduled + timedelta(hours=h)
                for h in [3, 6, 24]
            ],
        )

    def _generate_reasoning(
        self,
        slot: TimeSlot,
        niche: str,
        format_type: str,
    ) -> str:
        """Generate human-readable reasoning for the schedule."""
        reasons = []

        day_name = slot.day.name.capitalize()
        hour_12 = slot.hour % 12 or 12
        am_pm = "AM" if slot.hour < 12 else "PM"

        reasons.append(f"{day_name} at {hour_12}:00 {am_pm}")

        patterns = NICHE_OPTIMAL_TIMES.get(niche, NICHE_OPTIMAL_TIMES["default"])

        if slot.day in patterns.get("best_days", []):
            reasons.append(f"peak day for {niche} content")

        if slot.hour in patterns.get("best_hours", []):
            reasons.append(f"optimal hour for {niche} audience")

        if format_type == "short" and slot.hour in SHORTS_ADJUSTMENTS["boost_hours"]:
            reasons.append("high Shorts engagement period")

        if slot.historical_views > 0:
            reasons.append(f"historically ~{slot.historical_views:,} avg views")

        if slot.competition_level < 0.3:
            reasons.append("low competition window")

        return " | ".join(reasons)

    async def get_queue_position(
        self,
        campaign_id: str,
    ) -> Optional[int]:
        """Get the queue position for a scheduled video."""
        if not self.db_pool:
            return None

        try:
            async with self.db_pool.acquire() as conn:
                position = await conn.fetchval(
                    """
                    SELECT COUNT(*) + 1
                    FROM video_campaigns
                    WHERE status = 'ready_to_publish'
                      AND scheduled_publish_at < (
                          SELECT scheduled_publish_at
                          FROM video_campaigns
                          WHERE id = $1
                      )
                    """,
                    campaign_id,
                )
                return position
        except Exception as e:
            logger.warning(f"Failed to get queue position: {e}")
            return None


# Convenience function for orchestrator integration
async def schedule_video(
    niche: str,
    format_type: str = "short",
    db_pool: Optional[asyncpg.Pool] = None,
    **kwargs,
) -> dict:
    """
    Get optimal posting schedule for a video.

    Returns:
        Dict with scheduled_time, confidence, reasoning, and alternatives
    """
    scheduler = PostingScheduler(db_pool=db_pool)

    recommendation = await scheduler.get_optimal_time(
        niche=niche,
        format_type=format_type,
        **kwargs,
    )

    return {
        "scheduled_time": recommendation.scheduled_time.isoformat(),
        "timezone": recommendation.timezone,
        "confidence_score": recommendation.confidence_score,
        "reasoning": recommendation.reasoning,
        "alternative_times": [t.isoformat() for t in recommendation.alternative_times],
        "niche_factors": recommendation.niche_factors,
    }
