"""
Remotion Rendering Service

Client for the Remotion video composition API.
Handles:
- Scene assembly into Remotion composition format
- Render job submission
- Progress tracking
- Output retrieval
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class RenderJob:
    """Tracks a Remotion render job."""
    job_id: str
    composition: str
    status: Literal["pending", "rendering", "completed", "failed"] = "pending"
    progress_percent: int = 0
    output_url: Optional[str] = None
    error_message: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


@dataclass
class SceneInput:
    """Scene data for Remotion composition."""
    order: int
    url: str
    type: Literal["video", "image"]
    duration_seconds: float = 5.0


@dataclass
class RenderRequest:
    """Request to render a video composition."""
    composition: Literal["ViralShort", "YouTubeVideo"]
    scenes: list[SceneInput]
    audio_url: Optional[str] = None
    subtitles: Optional[list[dict]] = None
    avatar_video_url: Optional[str] = None
    title: Optional[str] = None
    intro_url: Optional[str] = None
    outro_url: Optional[str] = None
    metadata: Optional[dict] = None


class RemotionClient:
    """
    Client for Remotion video rendering API.

    Connects to either:
    - Self-hosted Remotion server (api.cckeeper.dev)
    - Cloud Remotion Lambda
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.api_url = api_url or os.getenv("REMOTION_API_URL", "https://api.cckeeper.dev")
        self.api_key = api_key or os.getenv("REMOTION_API_KEY", "")
        self._session: Optional[aiohttp.ClientSession] = None
        self._active_jobs: dict[str, RenderJob] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def render(self, request: RenderRequest) -> RenderJob:
        """
        Submit a render job to Remotion.

        Args:
            request: Render request with scenes and options

        Returns:
            RenderJob with job_id for tracking
        """
        session = await self._get_session()

        # Build Remotion input props
        props = {
            "scenes": [
                {
                    "order": s.order,
                    "url": s.url,
                    "type": s.type,
                    "durationInSeconds": s.duration_seconds,
                }
                for s in request.scenes
            ],
            "audioUrl": request.audio_url or "",
            "subtitles": request.subtitles or [],
        }

        # Add composition-specific props
        if request.composition == "ViralShort":
            if request.avatar_video_url:
                props["avatarVideoUrl"] = request.avatar_video_url
        elif request.composition == "YouTubeVideo":
            if request.title:
                props["title"] = request.title
            if request.intro_url:
                props["introUrl"] = request.intro_url
            if request.outro_url:
                props["outroUrl"] = request.outro_url

        if request.metadata:
            props["metadata"] = request.metadata

        payload = {
            "composition": request.composition,
            "inputProps": props,
        }

        try:
            async with session.post(
                f"{self.api_url}/render",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"Remotion render failed: {error}")
                    return RenderJob(
                        job_id="",
                        composition=request.composition,
                        status="failed",
                        error_message=error,
                    )

                data = await resp.json()
                job = RenderJob(
                    job_id=data.get("jobId", ""),
                    composition=request.composition,
                    status="pending",
                )
                self._active_jobs[job.job_id] = job
                logger.info(f"Render job submitted: {job.job_id}")
                return job

        except Exception as e:
            logger.error(f"Failed to submit render job: {e}")
            return RenderJob(
                job_id="",
                composition=request.composition,
                status="failed",
                error_message=str(e),
            )

    async def get_status(self, job_id: str) -> RenderJob:
        """
        Check the status of a render job.

        Args:
            job_id: The render job ID

        Returns:
            Updated RenderJob with current status
        """
        session = await self._get_session()

        try:
            async with session.get(f"{self.api_url}/render/{job_id}") as resp:
                if resp.status != 200:
                    return self._active_jobs.get(job_id, RenderJob(
                        job_id=job_id,
                        composition="unknown",
                        status="failed",
                        error_message="Job not found",
                    ))

                data = await resp.json()
                job = self._active_jobs.get(job_id, RenderJob(
                    job_id=job_id,
                    composition=data.get("composition", "unknown"),
                ))

                job.status = data.get("status", "pending")
                job.progress_percent = data.get("progress", 0)

                if job.status == "completed":
                    job.output_url = data.get("outputUrl")
                    job.completed_at = datetime.utcnow()
                elif job.status == "failed":
                    job.error_message = data.get("error", "Unknown error")

                self._active_jobs[job_id] = job
                return job

        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return RenderJob(
                job_id=job_id,
                composition="unknown",
                status="failed",
                error_message=str(e),
            )

    async def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> RenderJob:
        """
        Wait for a render job to complete.

        Args:
            job_id: The render job ID
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait

        Returns:
            Final RenderJob with output URL or error
        """
        start_time = datetime.utcnow()

        while True:
            job = await self.get_status(job_id)

            if job.status in ("completed", "failed"):
                return job

            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > timeout:
                job.status = "failed"
                job.error_message = "Render timeout exceeded"
                return job

            await asyncio.sleep(poll_interval)

    async def render_and_wait(
        self,
        request: RenderRequest,
        timeout: float = 600.0,
    ) -> RenderJob:
        """
        Submit a render job and wait for completion.

        Convenience method that combines render() and wait_for_completion().

        Args:
            request: Render request
            timeout: Maximum seconds to wait

        Returns:
            Final RenderJob with output URL or error
        """
        job = await self.render(request)

        if job.status == "failed":
            return job

        return await self.wait_for_completion(job.job_id, timeout=timeout)


# Convenience function for orchestrator integration
async def compose_video(
    scenes: list[dict],
    composition: str = "ViralShort",
    audio_url: Optional[str] = None,
    **kwargs,
) -> tuple[Optional[str], Optional[str]]:
    """
    Compose a video from scenes using Remotion.

    Returns:
        Tuple of (output_url, error_message)
    """
    client = RemotionClient()

    try:
        scene_inputs = [
            SceneInput(
                order=s.get("order", i),
                url=s.get("url", s.get("video_url", "")),
                type=s.get("type", "video"),
                duration_seconds=s.get("duration_seconds", 5.0),
            )
            for i, s in enumerate(scenes)
        ]

        request = RenderRequest(
            composition=composition,
            scenes=scene_inputs,
            audio_url=audio_url,
            **kwargs,
        )

        job = await client.render_and_wait(request)

        if job.status == "completed":
            return job.output_url, None
        else:
            return None, job.error_message

    finally:
        await client.close()
