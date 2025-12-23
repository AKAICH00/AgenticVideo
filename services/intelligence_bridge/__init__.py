"""
Intelligence Bridge - Connects trend analysis to video generation pipeline.

This service:
1. Fetches trending topics from YouTube Intelligence
2. Generates content briefs via Strategy Engine
3. Creates VideoState instances with trend-injected data
4. Triggers video generation with full context

The bridge enables "trend-to-video" automation:
Trend Detected → Brief Generated → Campaign Created → Video Produced
"""

from .campaign_factory import CampaignFactory
from .trend_trigger import TrendTrigger
from .feedback_loop import FeedbackLoop

__all__ = ["CampaignFactory", "TrendTrigger", "FeedbackLoop"]
