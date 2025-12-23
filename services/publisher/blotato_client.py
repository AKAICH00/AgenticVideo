"""
Blotato API Client

Multi-platform content publishing via Blotato API.
Supports: YouTube, TikTok, LinkedIn, Pinterest, Threads, Bluesky

API Docs: https://help.blotato.com/
Base URL: https://backend.blotato.com/v2
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


class Platform(str, Enum):
    """Supported publishing platforms."""
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    LINKEDIN = "linkedin"
    PINTEREST = "pinterest"
    THREADS = "threads"
    BLUESKY = "bluesky"


class YouTubePrivacy(str, Enum):
    """YouTube video privacy settings."""
    PUBLIC = "public"
    UNLISTED = "unlisted"
    PRIVATE = "private"


@dataclass
class YouTubeTarget:
    """YouTube-specific publishing target."""
    title: str
    privacy_status: YouTubePrivacy = YouTubePrivacy.PUBLIC
    should_notify_subscribers: bool = True
    is_made_for_kids: bool = False
    contains_synthetic_media: bool = True  # AI-generated content
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    playlist_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to Blotato API format."""
        return {
            "targetType": Platform.YOUTUBE.value,
            "title": self.title,
            "privacyStatus": self.privacy_status.value,
            "shouldNotifySubscribers": self.should_notify_subscribers,
            "isMadeForKids": self.is_made_for_kids,
            "containsSyntheticMedia": self.contains_synthetic_media,
            **({"description": self.description} if self.description else {}),
            **({"tags": self.tags} if self.tags else {}),
            **({"playlistId": self.playlist_id} if self.playlist_id else {}),
        }


@dataclass
class TikTokTarget:
    """TikTok-specific publishing target."""
    title: str
    privacy_level: str = "PUBLIC_TO_EVERYONE"  # PUBLIC_TO_EVERYONE, MUTUAL_FOLLOW_FRIENDS, FOLLOWER_OF_CREATOR, SELF_ONLY
    disable_duet: bool = False
    disable_stitch: bool = False
    disable_comment: bool = False
    brand_content_toggle: bool = False
    brand_organic_toggle: bool = False
    ai_generated_content: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to Blotato API format."""
        return {
            "targetType": Platform.TIKTOK.value,
            "title": self.title,
            "privacyLevel": self.privacy_level,
            "disableDuet": self.disable_duet,
            "disableStitch": self.disable_stitch,
            "disableComment": self.disable_comment,
            "brandContentToggle": self.brand_content_toggle,
            "brandOrganicToggle": self.brand_organic_toggle,
            "aiGeneratedContent": self.ai_generated_content,
        }


@dataclass
class LinkedInTarget:
    """LinkedIn-specific publishing target."""
    text: str
    visibility: str = "PUBLIC"  # PUBLIC, CONNECTIONS

    def to_dict(self) -> dict[str, Any]:
        """Convert to Blotato API format."""
        return {
            "targetType": Platform.LINKEDIN.value,
            "text": self.text,
            "visibility": self.visibility,
        }


@dataclass
class PublishResult:
    """Result from a publish operation."""
    success: bool
    post_id: Optional[str] = None
    platform: Optional[str] = None
    published_url: Optional[str] = None
    error: Optional[str] = None
    raw_response: Optional[dict] = None


class BlotaoClient:
    """
    Blotato API client for multi-platform publishing.

    Usage:
        client = BlotaoClient(api_key="your-api-key")

        # Publish to YouTube
        result = await client.publish_video(
            video_url="https://storage.example.com/video.mp4",
            targets=[
                YouTubeTarget(
                    title="5 AI Tools You Need in 2024",
                    description="Discover the best AI tools...",
                    tags=["AI", "productivity", "tech"],
                ),
            ],
        )

        # Publish to multiple platforms
        result = await client.publish_video(
            video_url="https://storage.example.com/video.mp4",
            targets=[
                YouTubeTarget(title="Amazing AI Tools"),
                TikTokTarget(title="AI Tools #tech #ai"),
            ],
        )
    """

    BASE_URL = "https://backend.blotato.com/v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 300,  # 5 minutes for video uploads
    ):
        """
        Initialize Blotato client.

        Args:
            api_key: Blotato API key (or set BLOTATO_API_KEY env var)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get("BLOTATO_API_KEY")
        if not self.api_key:
            raise ValueError("BLOTATO_API_KEY is required")

        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={
                    "blotato-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def upload_media(
        self,
        file_path: str,
        content_type: str = "video/mp4",
    ) -> Optional[str]:
        """
        Upload media file to Blotato.

        Note: This is optional - you can use public URLs directly.
        Use this only for local files or private storage.

        Args:
            file_path: Path to local file
            content_type: MIME type of the file

        Returns:
            Media URL if successful, None otherwise
        """
        session = await self._get_session()

        try:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field(
                    "file",
                    f,
                    filename=os.path.basename(file_path),
                    content_type=content_type,
                )

                async with session.post(
                    f"{self.BASE_URL}/media",
                    data=data,
                    headers={"Content-Type": None},  # Let aiohttp set multipart header
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("url")
                    else:
                        error = await response.text()
                        logger.error(f"Media upload failed: {response.status} - {error}")
                        return None

        except Exception as e:
            logger.error(f"Media upload error: {e}")
            return None

    async def publish_video(
        self,
        video_url: str,
        targets: list[YouTubeTarget | TikTokTarget | LinkedInTarget],
        thumbnail_url: Optional[str] = None,
        scheduled_time: Optional[datetime] = None,
    ) -> list[PublishResult]:
        """
        Publish video to one or more platforms.

        Args:
            video_url: Public URL to video file
            targets: List of platform-specific targets
            thumbnail_url: Optional thumbnail URL
            scheduled_time: Optional scheduled publish time (ISO format)

        Returns:
            List of results for each target platform
        """
        session = await self._get_session()
        results = []

        for target in targets:
            try:
                payload = {
                    "mediaUrls": [video_url],
                    **target.to_dict(),
                }

                if thumbnail_url:
                    payload["thumbnailUrl"] = thumbnail_url

                if scheduled_time:
                    payload["scheduledTime"] = scheduled_time.isoformat()

                logger.info(f"Publishing to {target.to_dict().get('targetType')}: {video_url}")

                async with session.post(
                    f"{self.BASE_URL}/posts",
                    json=payload,
                ) as response:
                    response_data = await response.json()

                    if response.status == 200:
                        results.append(PublishResult(
                            success=True,
                            post_id=response_data.get("id"),
                            platform=target.to_dict().get("targetType"),
                            published_url=response_data.get("url"),
                            raw_response=response_data,
                        ))
                        logger.info(f"Published successfully: {response_data.get('url')}")
                    else:
                        results.append(PublishResult(
                            success=False,
                            platform=target.to_dict().get("targetType"),
                            error=response_data.get("error", f"HTTP {response.status}"),
                            raw_response=response_data,
                        ))
                        logger.error(f"Publish failed: {response_data}")

            except Exception as e:
                results.append(PublishResult(
                    success=False,
                    platform=target.to_dict().get("targetType") if hasattr(target, "to_dict") else "unknown",
                    error=str(e),
                ))
                logger.error(f"Publish error: {e}")

            # Rate limiting: 30 requests/minute for posts
            await asyncio.sleep(2)

        return results

    async def publish_to_youtube(
        self,
        video_url: str,
        title: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        privacy: YouTubePrivacy = YouTubePrivacy.PUBLIC,
        thumbnail_url: Optional[str] = None,
        scheduled_time: Optional[datetime] = None,
        notify_subscribers: bool = True,
        is_made_for_kids: bool = False,
        contains_synthetic_media: bool = True,
    ) -> PublishResult:
        """
        Convenience method to publish to YouTube only.

        Args:
            video_url: Public URL to video file
            title: Video title
            description: Video description
            tags: Video tags
            privacy: Privacy setting
            thumbnail_url: Optional thumbnail URL
            scheduled_time: Optional scheduled time
            notify_subscribers: Whether to notify subscribers
            is_made_for_kids: COPPA compliance flag
            contains_synthetic_media: AI-generated content disclosure

        Returns:
            PublishResult with success/failure info
        """
        target = YouTubeTarget(
            title=title,
            description=description,
            tags=tags or [],
            privacy_status=privacy,
            should_notify_subscribers=notify_subscribers,
            is_made_for_kids=is_made_for_kids,
            contains_synthetic_media=contains_synthetic_media,
        )

        results = await self.publish_video(
            video_url=video_url,
            targets=[target],
            thumbnail_url=thumbnail_url,
            scheduled_time=scheduled_time,
        )

        return results[0] if results else PublishResult(
            success=False,
            error="No result returned",
        )

    async def publish_to_tiktok(
        self,
        video_url: str,
        title: str,
        privacy_level: str = "PUBLIC_TO_EVERYONE",
        ai_generated: bool = True,
    ) -> PublishResult:
        """
        Convenience method to publish to TikTok only.

        Args:
            video_url: Public URL to video file
            title: Video title/caption
            privacy_level: Privacy setting
            ai_generated: AI-generated content disclosure

        Returns:
            PublishResult with success/failure info
        """
        target = TikTokTarget(
            title=title,
            privacy_level=privacy_level,
            ai_generated_content=ai_generated,
        )

        results = await self.publish_video(
            video_url=video_url,
            targets=[target],
        )

        return results[0] if results else PublishResult(
            success=False,
            error="No result returned",
        )

    async def get_post_status(self, post_id: str) -> Optional[dict]:
        """
        Get status of a published post.

        Args:
            post_id: Post ID from publish result

        Returns:
            Post status dict or None
        """
        session = await self._get_session()

        try:
            async with session.get(
                f"{self.BASE_URL}/posts/{post_id}",
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return None
        except Exception as e:
            logger.error(f"Get post status error: {e}")
            return None


# Singleton instance for module-level usage
_client: Optional[BlotaoClient] = None


def get_blotato_client() -> BlotaoClient:
    """Get or create singleton Blotato client."""
    global _client
    if _client is None:
        _client = BlotaoClient()
    return _client


async def publish_video_simple(
    video_url: str,
    title: str,
    description: str = "",
    tags: list[str] = None,
    platforms: list[str] = None,
) -> list[PublishResult]:
    """
    Simple function to publish video to multiple platforms.

    Args:
        video_url: Public URL to video
        title: Video title
        description: Video description
        tags: Video tags
        platforms: List of platforms (default: ["youtube"])

    Returns:
        List of publish results
    """
    client = get_blotato_client()
    platforms = platforms or ["youtube"]
    tags = tags or []

    targets = []
    for platform in platforms:
        if platform == "youtube":
            targets.append(YouTubeTarget(
                title=title,
                description=description,
                tags=tags,
                contains_synthetic_media=True,
            ))
        elif platform == "tiktok":
            # TikTok title includes hashtags
            tiktok_title = f"{title} #{' #'.join(tags[:5])}" if tags else title
            targets.append(TikTokTarget(
                title=tiktok_title[:150],  # TikTok limit
                ai_generated_content=True,
            ))

    return await client.publish_video(video_url, targets)
