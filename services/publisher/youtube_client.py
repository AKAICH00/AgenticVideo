"""
YouTube Upload Client

Handles video uploads to YouTube using the YouTube Data API v3.
Supports:
- Video upload with resumable uploads
- Metadata setting (title, description, tags)
- Thumbnail upload
- Playlist management
- Privacy settings
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
from pathlib import Path
import aiohttp
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata for a YouTube video upload."""
    title: str
    description: str
    tags: list[str] = field(default_factory=list)
    category_id: str = "22"  # People & Blogs default
    privacy_status: Literal["public", "private", "unlisted"] = "private"
    made_for_kids: bool = False
    scheduled_publish_at: Optional[datetime] = None
    playlist_id: Optional[str] = None
    default_language: str = "en"
    thumbnail_path: Optional[str] = None


@dataclass
class UploadResult:
    """Result of a video upload."""
    success: bool
    video_id: Optional[str] = None
    video_url: Optional[str] = None
    error_message: Optional[str] = None
    upload_time_seconds: float = 0.0


# YouTube category IDs for reference
YOUTUBE_CATEGORIES = {
    "film_animation": "1",
    "autos_vehicles": "2",
    "music": "10",
    "pets_animals": "15",
    "sports": "17",
    "travel_events": "19",
    "gaming": "20",
    "people_blogs": "22",
    "comedy": "23",
    "entertainment": "24",
    "news_politics": "25",
    "howto_style": "26",
    "education": "27",
    "science_tech": "28",
    "nonprofits_activism": "29",
}


class YouTubeClient:
    """
    Client for YouTube Data API v3.
    
    Handles OAuth2 authentication and video uploads.
    """
    
    YOUTUBE_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
    YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
        access_token: Optional[str] = None,
    ):
        self.client_id = client_id or os.getenv("YOUTUBE_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("YOUTUBE_CLIENT_SECRET", "")
        self.refresh_token = refresh_token or os.getenv("YOUTUBE_REFRESH_TOKEN", "")
        self._access_token = access_token
        self._token_expires_at: Optional[datetime] = None
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _ensure_access_token(self):
        """Refresh access token if needed."""
        if self._access_token and self._token_expires_at:
            if datetime.utcnow() < self._token_expires_at:
                return  # Token still valid
        
        if not self.refresh_token:
            raise ValueError("No refresh token available. Run OAuth2 flow first.")
        
        session = await self._get_session()
        
        async with session.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "grant_type": "refresh_token",
            },
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise ValueError(f"Failed to refresh token: {error}")
            
            data = await resp.json()
            self._access_token = data["access_token"]
            expires_in = data.get("expires_in", 3600)
            self._token_expires_at = datetime.utcnow()
    
    async def _get_auth_headers(self) -> dict:
        """Get authorization headers."""
        await self._ensure_access_token()
        return {
            "Authorization": f"Bearer {self._access_token}",
        }
    
    async def upload_video(
        self,
        video_path: str,
        metadata: VideoMetadata,
        on_progress: Optional[callable] = None,
    ) -> UploadResult:
        """
        Upload a video to YouTube.
        
        Args:
            video_path: Path to the video file
            metadata: Video metadata
            on_progress: Optional callback for upload progress
            
        Returns:
            UploadResult with video ID and URL
        """
        start_time = datetime.utcnow()
        
        # Validate video file
        video_file = Path(video_path)
        if not video_file.exists():
            return UploadResult(
                success=False,
                error_message=f"Video file not found: {video_path}",
            )
        
        file_size = video_file.stat().st_size
        logger.info(f"Uploading video: {video_path} ({file_size / 1024 / 1024:.1f} MB)")
        
        try:
            session = await self._get_session()
            headers = await self._get_auth_headers()
            
            # Build video resource
            video_resource = {
                "snippet": {
                    "title": metadata.title[:100],  # YouTube limit
                    "description": metadata.description[:5000],  # YouTube limit
                    "tags": metadata.tags[:500],  # YouTube limit
                    "categoryId": metadata.category_id,
                    "defaultLanguage": metadata.default_language,
                },
                "status": {
                    "privacyStatus": metadata.privacy_status,
                    "selfDeclaredMadeForKids": metadata.made_for_kids,
                },
            }
            
            # Add scheduled publish time if set
            if metadata.scheduled_publish_at and metadata.privacy_status == "private":
                video_resource["status"]["publishAt"] = metadata.scheduled_publish_at.isoformat() + "Z"
            
            # Step 1: Initialize resumable upload
            init_headers = {
                **headers,
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Length": str(file_size),
                "X-Upload-Content-Type": "video/*",
            }
            
            async with session.post(
                f"{self.YOUTUBE_UPLOAD_URL}?uploadType=resumable&part=snippet,status",
                headers=init_headers,
                json=video_resource,
            ) as resp:
                if resp.status not in (200, 308):
                    error = await resp.text()
                    return UploadResult(
                        success=False,
                        error_message=f"Upload init failed: {error}",
                    )
                
                upload_url = resp.headers.get("Location")
                if not upload_url:
                    return UploadResult(
                        success=False,
                        error_message="No upload URL returned",
                    )
            
            # Step 2: Upload video data
            upload_headers = {
                "Content-Type": "video/*",
                "Content-Length": str(file_size),
            }
            
            async with aiofiles.open(video_path, "rb") as f:
                video_data = await f.read()
            
            async with session.put(
                upload_url,
                headers=upload_headers,
                data=video_data,
            ) as resp:
                if resp.status not in (200, 201):
                    error = await resp.text()
                    return UploadResult(
                        success=False,
                        error_message=f"Upload failed: {error}",
                    )
                
                result_data = await resp.json()
                video_id = result_data.get("id")
            
            # Step 3: Upload thumbnail if provided
            if metadata.thumbnail_path and video_id:
                await self._upload_thumbnail(video_id, metadata.thumbnail_path)
            
            # Step 4: Add to playlist if specified
            if metadata.playlist_id and video_id:
                await self._add_to_playlist(video_id, metadata.playlist_id)
            
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            
            return UploadResult(
                success=True,
                video_id=video_id,
                video_url=f"https://youtube.com/watch?v={video_id}",
                upload_time_seconds=elapsed,
            )
        
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return UploadResult(
                success=False,
                error_message=str(e),
            )
    
    async def _upload_thumbnail(self, video_id: str, thumbnail_path: str):
        """Upload a custom thumbnail for a video."""
        session = await self._get_session()
        headers = await self._get_auth_headers()
        
        async with aiofiles.open(thumbnail_path, "rb") as f:
            thumbnail_data = await f.read()
        
        async with session.post(
            f"{self.YOUTUBE_API_URL}/thumbnails/set",
            params={"videoId": video_id},
            headers={**headers, "Content-Type": "image/png"},
            data=thumbnail_data,
        ) as resp:
            if resp.status != 200:
                logger.warning(f"Thumbnail upload failed: {await resp.text()}")
    
    async def _add_to_playlist(self, video_id: str, playlist_id: str):
        """Add a video to a playlist."""
        session = await self._get_session()
        headers = await self._get_auth_headers()
        
        async with session.post(
            f"{self.YOUTUBE_API_URL}/playlistItems",
            params={"part": "snippet"},
            headers={**headers, "Content-Type": "application/json"},
            json={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": video_id,
                    },
                },
            },
        ) as resp:
            if resp.status != 200:
                logger.warning(f"Playlist add failed: {await resp.text()}")
    
    async def update_video_metadata(
        self,
        video_id: str,
        metadata: VideoMetadata,
    ) -> bool:
        """Update metadata for an existing video."""
        session = await self._get_session()
        headers = await self._get_auth_headers()
        
        video_resource = {
            "id": video_id,
            "snippet": {
                "title": metadata.title[:100],
                "description": metadata.description[:5000],
                "tags": metadata.tags[:500],
                "categoryId": metadata.category_id,
            },
        }
        
        async with session.put(
            f"{self.YOUTUBE_API_URL}/videos",
            params={"part": "snippet"},
            headers={**headers, "Content-Type": "application/json"},
            json=video_resource,
        ) as resp:
            return resp.status == 200
    
    async def set_video_privacy(
        self,
        video_id: str,
        privacy_status: Literal["public", "private", "unlisted"],
    ) -> bool:
        """Change the privacy status of a video."""
        session = await self._get_session()
        headers = await self._get_auth_headers()
        
        async with session.put(
            f"{self.YOUTUBE_API_URL}/videos",
            params={"part": "status"},
            headers={**headers, "Content-Type": "application/json"},
            json={
                "id": video_id,
                "status": {"privacyStatus": privacy_status},
            },
        ) as resp:
            return resp.status == 200
    
    async def get_channel_info(self) -> Optional[dict]:
        """Get information about the authenticated channel."""
        session = await self._get_session()
        headers = await self._get_auth_headers()
        
        async with session.get(
            f"{self.YOUTUBE_API_URL}/channels",
            params={"part": "snippet,statistics", "mine": "true"},
            headers=headers,
        ) as resp:
            if resp.status != 200:
                return None
            
            data = await resp.json()
            items = data.get("items", [])
            return items[0] if items else None
