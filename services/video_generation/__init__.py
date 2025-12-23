"""
Video Generation Service

Provides unified access to video generation through:
- Premium path: Kie AI / Fal AI aggregators (Runway, Sora, Kling)
- Bulk path: Self-hosted Wan 2.1

All generation goes through circuit breakers for resilience.
"""

from .client import (
    VideoGenerationClient,
    VideoResult,
    GenerationRequest,
    VideoModel,
    GenerationStatus,
)
from .job_tracker import JobTracker

__all__ = [
    "VideoGenerationClient",
    "VideoResult",
    "GenerationRequest",
    "VideoModel",
    "GenerationStatus",
    "JobTracker",
]
