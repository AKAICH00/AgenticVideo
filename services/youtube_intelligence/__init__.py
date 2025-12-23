"""
YouTube Intelligence Service

Leverages Google's ecosystem for viral content optimization:
- YouTube Data API v3: Trending videos, competitor analysis
- YouTube Analytics API: Performance data for our videos
- Vertex AI: Pattern recognition and trend prediction
"""

from .trend_analyzer import TrendAnalyzer

__all__ = ["TrendAnalyzer"]
