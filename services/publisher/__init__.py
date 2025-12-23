"""
Publisher Services

Automated video distribution to YouTube and other platforms.
Supports direct YouTube API and Blotato multi-platform publishing.
"""

from .youtube_client import (
    YouTubeClient,
    VideoMetadata,
    UploadResult,
    YOUTUBE_CATEGORIES,
)

from .metadata_generator import (
    MetadataGenerator,
    SEOMetadata,
    VideoContext,
    generate_metadata,
    NICHE_CATEGORIES,
)

from .scheduler import (
    PostingScheduler,
    ScheduleRecommendation,
    TimeSlot,
    schedule_video,
)

from .blotato_client import (
    BlotaoClient,
    Platform,
    YouTubePrivacy,
    YouTubeTarget,
    TikTokTarget,
    LinkedInTarget,
    PublishResult,
    get_blotato_client,
    publish_video_simple,
)

__all__ = [
    # YouTube Client (direct API)
    "YouTubeClient",
    "VideoMetadata",
    "UploadResult",
    "YOUTUBE_CATEGORIES",
    # Blotato Client (multi-platform)
    "BlotaoClient",
    "Platform",
    "YouTubePrivacy",
    "YouTubeTarget",
    "TikTokTarget",
    "LinkedInTarget",
    "PublishResult",
    "get_blotato_client",
    "publish_video_simple",
    # Metadata Generator
    "MetadataGenerator",
    "SEOMetadata",
    "VideoContext",
    "generate_metadata",
    "NICHE_CATEGORIES",
    # Scheduler
    "PostingScheduler",
    "ScheduleRecommendation",
    "TimeSlot",
    "schedule_video",
]
