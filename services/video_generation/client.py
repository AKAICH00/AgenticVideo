"""
Unified Video Generation Client

Single interface for all video generation needs:
- Premium: Kie AI / Fal AI → Runway Gen-4.5, Sora 2, Kling 2.5
- Bulk: Self-hosted Wan 2.1

Features:
- Automatic routing based on quality tier
- Circuit breaker protection for all APIs
- Job tracking and cost estimation
- Progress callback support for SSE streaming
"""

import asyncio
import httpx
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from core.circuit_breaker import CircuitBreaker, CircuitBreakerOpen, get_video_api_breaker
from core.config import get_config

logger = logging.getLogger(__name__)


class VideoGenerationError(Exception):
    """Raised when video generation fails."""

    def __init__(self, message: str, error_code: str = None, provider: str = None):
        self.error_code = error_code
        self.provider = provider
        super().__init__(message)


class VideoModel(str, Enum):
    """Available video generation models."""
    # Premium models (via Kie.ai aggregator)
    VEO_3_1 = "veo-3.1"              # Google Veo 3.1 - best quality, with audio
    VEO_3_1_FAST = "veo-3.1-fast"    # Google Veo 3.1 Fast - faster, cheaper
    RUNWAY_ALEPH = "runway-aleph"    # Runway Aleph - advanced scene reasoning
    SORA_2 = "sora-2"                # OpenAI Sora 2 - realistic motion
    KLING_2_5 = "kling-2.5"          # Kling 2.5 - Chinese model (may render Chinese text)
    LUMA_DREAM = "luma-dream-machine"

    # Legacy aliases (mapped to newer models)
    RUNWAY_GEN4_5 = "runway-gen4.5"  # Maps to Runway Aleph

    # Bulk models (via Kie.ai or self-hosted)
    WAN_2_1 = "wan-2.1"              # Wan 2.1 - good balance of quality/cost
    WAN_T2V_1_3B = "wan-t2v-1.3b"    # Smaller, fits 8GB GPU


class GenerationStatus(str, Enum):
    """Status of a video generation job."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationRequest:
    """Request for video generation."""
    prompt: str
    duration_seconds: int = 5
    aspect_ratio: str = "16:9"
    model: VideoModel = VideoModel.RUNWAY_GEN4_5
    quality_tier: Literal["premium", "bulk"] = "premium"

    # Optional enhancements
    negative_prompt: Optional[str] = None
    reference_image_url: Optional[str] = None
    motion_data_id: Optional[str] = None  # For motion transfer

    # Scene context for consistency
    scene_id: Optional[str] = None
    campaign_id: Optional[str] = None

    # Request metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VideoResult:
    """Result from video generation."""
    request_id: str
    status: GenerationStatus
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    local_path: Optional[str] = None  # Local path after download

    # Provider details
    provider: Optional[str] = None
    model: Optional[str] = None
    external_job_id: Optional[str] = None

    # Timing
    processing_time_seconds: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Cost
    estimated_cost_usd: Optional[float] = None
    actual_cost_usd: Optional[float] = None

    # Error handling
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0


# Cost estimation per model (USD per second of video)
# Based on Kie.ai pricing: https://kie.ai/v3-api-pricing
MODEL_COSTS = {
    VideoModel.VEO_3_1: 0.25,         # $2/8s = $0.25/s (highest quality)
    VideoModel.VEO_3_1_FAST: 0.05,    # $0.40/8s = $0.05/s (faster, cheaper)
    VideoModel.RUNWAY_ALEPH: 0.15,    # Runway Aleph
    VideoModel.RUNWAY_GEN4_5: 0.15,   # Legacy alias
    VideoModel.SORA_2: 0.20,          # Sora 2
    VideoModel.KLING_2_5: 0.10,       # Kling (Chinese model)
    VideoModel.LUMA_DREAM: 0.08,      # Luma Dream Machine
    VideoModel.WAN_2_1: 0.02,         # Wan via Kie.ai
    VideoModel.WAN_T2V_1_3B: 0.002,   # Self-hosted GPU time only
}


class VideoGenerationClient:
    """
    Unified client for video generation.

    Usage:
        client = VideoGenerationClient()

        # Premium generation (uses Kie/Fal aggregator)
        result = await client.generate(
            prompt="A golden retriever running through a field",
            model=VideoModel.RUNWAY_GEN4_5,
            duration_seconds=10
        )

        # Bulk generation (uses local Wan 2.1)
        result = await client.generate(
            prompt="Product showcase with smooth camera movement",
            model=VideoModel.WAN_2_1,
            quality_tier="bulk",
            duration_seconds=5
        )
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        on_progress: Optional[Callable[[str, int, str], None]] = None,
    ):
        """
        Initialize the video generation client.

        Args:
            config: Optional config override
            on_progress: Callback for progress updates (request_id, percent, message)
        """
        self.config = config or get_config()
        self.on_progress = on_progress

        # HTTP client for API calls
        self._http_client: Optional[httpx.AsyncClient] = None

        # Circuit breakers for each provider
        self._breakers = {
            "kie": get_video_api_breaker("kie"),
            "fal": get_video_api_breaker("fal"),
            "runway": get_video_api_breaker("runway"),
            "sora": get_video_api_breaker("sora"),
            "kling": get_video_api_breaker("kling"),
            "wan_local": get_video_api_breaker("wan_local"),
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=600.0)  # 10 min timeout
        return self._http_client

    async def close(self):
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def download_video(
        self,
        video_url: str,
        output_dir: str = "output",
        filename: Optional[str] = None,
        campaign_id: Optional[str] = None,
        scene_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Download a video from a temporary URL to local storage.

        Args:
            video_url: The temporary video URL to download
            output_dir: Base output directory (default: "output")
            filename: Custom filename (auto-generated if not provided)
            campaign_id: Campaign ID for organizing files
            scene_id: Scene ID for filename

        Returns:
            Local path to the downloaded file, or None if failed
        """
        try:
            # Create output directory structure
            base_dir = Path(output_dir)
            if campaign_id:
                base_dir = base_dir / campaign_id
            base_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            if not filename:
                if scene_id:
                    filename = f"scene_{scene_id}.mp4"
                else:
                    filename = f"video_{uuid.uuid4().hex[:8]}.mp4"

            output_path = base_dir / filename

            # Download the video
            client = await self._get_client()
            response = await client.get(video_url, follow_redirects=True)
            response.raise_for_status()

            # Write to file
            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Video downloaded: {output_path} ({len(response.content) / 1024 / 1024:.1f} MB)")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to download video from {video_url}: {e}")
            return None

    def _emit_progress(self, request_id: str, percent: int, message: str):
        """Emit progress update via callback."""
        if self.on_progress:
            try:
                self.on_progress(request_id, percent, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def _estimate_cost(self, model: VideoModel, duration_seconds: int) -> float:
        """Estimate generation cost in USD."""
        cost_per_second = MODEL_COSTS.get(model, 0.10)
        return cost_per_second * duration_seconds

    def _get_provider_for_model(self, model: VideoModel) -> str:
        """Determine which provider handles this model."""
        if model in (VideoModel.WAN_2_1, VideoModel.WAN_T2V_1_3B):
            return "wan_local"

        # For premium models, use the configured aggregator
        return self.config.get_aggregator()

    async def generate(
        self,
        prompt: str,
        model: Optional[VideoModel] = None,
        duration_seconds: int = 5,
        aspect_ratio: str = "16:9",
        quality_tier: Literal["premium", "bulk"] = "premium",
        negative_prompt: Optional[str] = None,
        reference_image_url: Optional[str] = None,
        motion_data_id: Optional[str] = None,
        scene_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
    ) -> VideoResult:
        """
        Generate a video from a text prompt.

        Args:
            prompt: Text description of the video to generate
            model: Specific model to use (auto-selected if None)
            duration_seconds: Length of video (model limits apply)
            aspect_ratio: Output aspect ratio (16:9, 9:16, 1:1)
            quality_tier: "premium" uses APIs, "bulk" uses self-hosted
            negative_prompt: Things to avoid in generation
            reference_image_url: First frame / style reference
            motion_data_id: Motion data to apply
            scene_id: Scene this belongs to
            campaign_id: Campaign this belongs to

        Returns:
            VideoResult with status and URL (if successful)
        """
        # Auto-select model based on tier if not specified
        if model is None:
            model = (
                VideoModel(self.config.models.default_premium)
                if quality_tier == "premium"
                else VideoModel(self.config.models.default_bulk)
            )

        # Create request
        request = GenerationRequest(
            prompt=prompt,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
            model=model,
            quality_tier=quality_tier,
            negative_prompt=negative_prompt,
            reference_image_url=reference_image_url,
            motion_data_id=motion_data_id,
            scene_id=scene_id,
            campaign_id=campaign_id,
        )

        self._emit_progress(request.request_id, 0, "Starting generation")

        # Get the provider for this model
        provider = self._get_provider_for_model(model)
        breaker = self._breakers.get(provider)

        if breaker is None:
            return VideoResult(
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                error_code="UNKNOWN_PROVIDER",
                error_message=f"No circuit breaker for provider: {provider}",
            )

        # Route to appropriate generation method
        try:
            if provider == "wan_local":
                result = await breaker.call(
                    self._generate_wan_local, request
                )
            elif provider == "fal":
                result = await breaker.call(
                    self._generate_via_fal, request
                )
            else:  # kie
                result = await breaker.call(
                    self._generate_via_kie, request
                )

            self._emit_progress(request.request_id, 100, "Generation complete")
            return result

        except CircuitBreakerOpen as e:
            error_msg = f"Circuit breaker open for {e.service_name}, retry after {e.retry_after:.1f}s"
            logger.warning(error_msg)
            self._emit_progress(request.request_id, 100, error_msg)

            return VideoResult(
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                provider=provider,
                model=model.value,
                error_code="CIRCUIT_BREAKER_OPEN",
                error_message=error_msg,
            )

        except VideoGenerationError as e:
            error_msg = str(e)
            logger.error(f"Generation failed: {error_msg}")
            self._emit_progress(request.request_id, 100, f"Failed: {error_msg}")

            return VideoResult(
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                provider=e.provider or provider,
                model=model.value,
                error_code=e.error_code,
                error_message=error_msg,
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            logger.error(f"Generation failed: {error_msg}")
            self._emit_progress(request.request_id, 100, f"Failed: {error_msg}")

            return VideoResult(
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                provider=provider,
                model=model.value,
                error_message=error_msg,
            )

    async def _generate_via_kie(self, request: GenerationRequest) -> VideoResult:
        """Generate video via Kie AI Market API."""
        client = await self._get_client()
        started_at = datetime.utcnow()

        self._emit_progress(request.request_id, 10, "Submitting to Kie AI")

        # Map our model names to Kie Market API model names
        # See: https://kie.ai/market for available models
        # NOTE: Default to Veo 3.1 Fast or Wan to avoid Chinese text from Kling
        kie_model_map = {
            # Premium models
            VideoModel.VEO_3_1: "veo3/text-to-video",            # Google Veo 3.1 - best quality
            VideoModel.VEO_3_1_FAST: "veo3-fast/text-to-video",  # Google Veo 3.1 Fast
            VideoModel.RUNWAY_ALEPH: "runway-aleph/text-to-video",  # Runway Aleph
            VideoModel.RUNWAY_GEN4_5: "runway-aleph/text-to-video",  # Legacy alias → Runway Aleph
            VideoModel.SORA_2: "sora2/text-to-video",            # OpenAI Sora 2
            VideoModel.KLING_2_5: "kling-2.6/text-to-video",     # Kling (Chinese - use with caution)
            VideoModel.LUMA_DREAM: "hailuo-i2v/text-to-video",   # Hailuo as Luma alternative
            # Bulk/cost-effective models
            VideoModel.WAN_2_1: "wan-2.6/text-to-video",         # Wan 2.6 (non-Chinese)
        }

        # Default to Veo 3.1 Fast for best quality/cost ratio without Chinese text
        kie_model = kie_model_map.get(request.model, "veo3-fast/text-to-video")

        # Kie API only accepts duration of "5" or "10" seconds
        # Clamp to nearest valid value
        if request.duration_seconds <= 7:
            kie_duration = "5"
        else:
            kie_duration = "10"

        # Kie AI Market API request format
        payload = {
            "model": kie_model,
            "input": {
                "prompt": request.prompt,
                "duration": kie_duration,
                "aspect_ratio": request.aspect_ratio,
                "sound": False,
            },
        }

        if request.negative_prompt:
            payload["input"]["negative_prompt"] = request.negative_prompt
        if request.reference_image_url:
            payload["input"]["imageUrl"] = request.reference_image_url

        logger.info(f"Kie API request: model={kie_model}, prompt={request.prompt[:50]}...")

        headers = {
            "Authorization": f"Bearer {self.config.api.kie_api_key}",
            "Content-Type": "application/json",
        }

        try:
            # Submit job via Market API
            response = await client.post(
                f"{self.config.api.kie_api_base}/jobs/createTask",
                json=payload,
                headers=headers,
            )

            data = response.json()
            logger.info(f"Kie API response: {data}")

        except httpx.TimeoutException as e:
            raise VideoGenerationError(
                f"Kie API timeout: {type(e).__name__}",
                error_code="TIMEOUT",
                provider="kie",
            )
        except httpx.RequestError as e:
            raise VideoGenerationError(
                f"Kie API request failed: {type(e).__name__}: {e}",
                error_code="REQUEST_ERROR",
                provider="kie",
            )
        except Exception as e:
            raise VideoGenerationError(
                f"Kie API unexpected error: {type(e).__name__}: {e}",
                error_code="UNEXPECTED_ERROR",
                provider="kie",
            )

        if data.get("code") != 200:
            error_msg = data.get("msg", "Unknown error")
            logger.error(f"Kie API error: {data}")
            raise VideoGenerationError(
                f"Kie API error: {error_msg}",
                error_code=f"KIE_{data.get('code')}",
                provider="kie",
            )

        job_id = data.get("data", {}).get("taskId")
        logger.info(f"Kie task created: {job_id}")
        if not job_id:
            raise VideoGenerationError(
                "No taskId in Kie API response",
                error_code="NO_TASK_ID",
                provider="kie",
            )

        self._emit_progress(request.request_id, 20, f"Job queued: {job_id}")

        # Poll for completion
        result = await self._poll_kie_job(client, job_id, request, headers)

        # If polling returned a failed result, raise an exception
        if result.status == GenerationStatus.FAILED:
            raise VideoGenerationError(
                result.error_message or "Generation failed",
                error_code=result.error_code,
                provider="kie",
            )

        result.started_at = started_at
        result.completed_at = datetime.utcnow()
        result.processing_time_seconds = (
            result.completed_at - started_at
        ).total_seconds()
        result.estimated_cost_usd = self._estimate_cost(
            request.model, request.duration_seconds
        )

        return result

    async def _poll_kie_job(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        request: GenerationRequest,
        headers: dict,
    ) -> VideoResult:
        """Poll Kie AI Market API for job completion."""
        import json as json_module

        poll_interval = 5  # seconds
        max_polls = 180  # 15 minutes max (Kling takes 2-5 min per video)
        consecutive_errors = 0
        max_consecutive_errors = 10  # More tolerant of transient errors

        for i in range(max_polls):
            await asyncio.sleep(poll_interval)

            try:
                response = await client.get(
                    f"{self.config.api.kie_api_base}/jobs/recordInfo?taskId={task_id}",
                    headers=headers,
                )

                data = response.json()
                consecutive_errors = 0  # Reset on success

            except httpx.TimeoutException as e:
                consecutive_errors += 1
                logger.warning(f"Kie poll timeout (attempt {consecutive_errors}): {type(e).__name__}")
                if consecutive_errors >= max_consecutive_errors:
                    return VideoResult(
                        request_id=request.request_id,
                        status=GenerationStatus.FAILED,
                        provider="kie",
                        model=request.model.value,
                        external_job_id=task_id,
                        error_code="POLL_TIMEOUT",
                        error_message=f"Too many timeouts while polling: {type(e).__name__}",
                    )
                continue

            except httpx.RequestError as e:
                consecutive_errors += 1
                logger.warning(f"Kie poll error (attempt {consecutive_errors}): {type(e).__name__}: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    return VideoResult(
                        request_id=request.request_id,
                        status=GenerationStatus.FAILED,
                        provider="kie",
                        model=request.model.value,
                        external_job_id=task_id,
                        error_code="POLL_ERROR",
                        error_message=f"Network error while polling: {type(e).__name__}: {e}",
                    )
                continue

            except Exception as e:
                consecutive_errors += 1
                logger.warning(f"Kie poll unexpected error (attempt {consecutive_errors}): {type(e).__name__}: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    return VideoResult(
                        request_id=request.request_id,
                        status=GenerationStatus.FAILED,
                        provider="kie",
                        model=request.model.value,
                        external_job_id=task_id,
                        error_code="POLL_UNEXPECTED_ERROR",
                        error_message=f"Unexpected error while polling: {type(e).__name__}: {e}",
                    )
                continue

            if data.get("code") != 200:
                logger.warning(f"Kie poll API error: {data}")
                continue  # Retry on API error

            record = data.get("data", {})
            state = record.get("state", "").lower()

            # Update progress
            progress = min(20 + (i * 60 // max_polls), 90)
            self._emit_progress(
                request.request_id, progress, f"Processing: {state}"
            )

            if state == "success":
                # Parse resultJson to get output URLs (Kie API format)
                result_json_str = record.get("resultJson", "{}")
                try:
                    result_json = json_module.loads(result_json_str)
                    output_urls = result_json.get("resultUrls", [])
                except json_module.JSONDecodeError:
                    logger.warning(f"Failed to parse resultJson: {result_json_str[:100]}")
                    output_urls = []

                video_url = output_urls[0] if output_urls else None

                if not video_url:
                    logger.warning(f"No video URL in success response: {record}")

                return VideoResult(
                    request_id=request.request_id,
                    status=GenerationStatus.COMPLETED,
                    video_url=video_url,
                    thumbnail_url=None,
                    provider="kie",
                    model=request.model.value,
                    external_job_id=task_id,
                )

            if state == "failed":
                fail_msg = record.get("failMsg") or "Generation failed (no specific reason)"
                logger.error(f"Kie job failed: {fail_msg}")
                return VideoResult(
                    request_id=request.request_id,
                    status=GenerationStatus.FAILED,
                    provider="kie",
                    model=request.model.value,
                    external_job_id=task_id,
                    error_code="JOB_FAILED",
                    error_message=fail_msg,
                )

        return VideoResult(
            request_id=request.request_id,
            status=GenerationStatus.FAILED,
            provider="kie",
            model=request.model.value,
            external_job_id=task_id,
            error_code="TIMEOUT",
            error_message=f"Job did not complete within {poll_interval * max_polls} seconds",
        )

    async def _generate_via_fal(self, request: GenerationRequest) -> VideoResult:
        """Generate video via Fal AI aggregator."""
        client = await self._get_client()
        started_at = datetime.utcnow()

        self._emit_progress(request.request_id, 10, "Submitting to Fal AI")

        # Map our model names to Fal model IDs
        fal_model_map = {
            VideoModel.RUNWAY_GEN4_5: "fal-ai/runway-gen4.5",
            VideoModel.SORA_2: "fal-ai/sora",
            VideoModel.KLING_2_5: "fal-ai/kling-video/v2.5/pro",
            VideoModel.LUMA_DREAM: "fal-ai/luma-dream-machine",
        }

        fal_model = fal_model_map.get(request.model)
        if not fal_model:
            return VideoResult(
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                provider="fal",
                model=request.model.value,
                error_code="UNSUPPORTED_MODEL",
                error_message=f"Model {request.model.value} not available via Fal",
            )

        # Fal AI request format
        payload = {
            "prompt": request.prompt,
            "duration": str(request.duration_seconds),
            "aspect_ratio": request.aspect_ratio,
        }

        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        if request.reference_image_url:
            payload["image_url"] = request.reference_image_url

        headers = {
            "Authorization": f"Key {self.config.api.fal_api_key}",
            "Content-Type": "application/json",
        }

        # Fal uses queue-based API
        response = await client.post(
            f"{self.config.api.fal_api_base}/{fal_model}",
            json=payload,
            headers=headers,
        )

        if response.status_code not in (200, 201, 202):
            return VideoResult(
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                provider="fal",
                model=request.model.value,
                error_code=f"HTTP_{response.status_code}",
                error_message=response.text,
            )

        data = response.json()

        # Fal can return direct result or queue reference
        if "video" in data or "video_url" in data:
            # Direct result
            return VideoResult(
                request_id=request.request_id,
                status=GenerationStatus.COMPLETED,
                video_url=data.get("video", {}).get("url") or data.get("video_url"),
                provider="fal",
                model=request.model.value,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                estimated_cost_usd=self._estimate_cost(
                    request.model, request.duration_seconds
                ),
            )

        # Queue-based - poll for result
        request_id = data.get("request_id")
        self._emit_progress(request.request_id, 20, f"Queued: {request_id}")

        result = await self._poll_fal_job(
            client, fal_model, request_id, request, headers
        )

        result.started_at = started_at
        result.completed_at = datetime.utcnow()
        result.processing_time_seconds = (
            result.completed_at - started_at
        ).total_seconds()
        result.estimated_cost_usd = self._estimate_cost(
            request.model, request.duration_seconds
        )

        return result

    async def _poll_fal_job(
        self,
        client: httpx.AsyncClient,
        fal_model: str,
        fal_request_id: str,
        request: GenerationRequest,
        headers: dict,
    ) -> VideoResult:
        """Poll Fal AI for job completion."""
        poll_interval = 5
        max_polls = 120

        for i in range(max_polls):
            await asyncio.sleep(poll_interval)

            response = await client.get(
                f"{self.config.api.fal_api_base}/{fal_model}/requests/{fal_request_id}/status",
                headers=headers,
            )

            if response.status_code != 200:
                continue

            data = response.json()
            status = data.get("status", "").lower()

            progress = min(20 + (i * 60 // max_polls), 90)
            self._emit_progress(
                request.request_id, progress, f"Processing: {status}"
            )

            if status == "completed":
                # Fetch the actual result
                result_response = await client.get(
                    f"{self.config.api.fal_api_base}/{fal_model}/requests/{fal_request_id}",
                    headers=headers,
                )
                result_data = result_response.json()

                return VideoResult(
                    request_id=request.request_id,
                    status=GenerationStatus.COMPLETED,
                    video_url=(
                        result_data.get("video", {}).get("url") or
                        result_data.get("video_url")
                    ),
                    provider="fal",
                    model=request.model.value,
                    external_job_id=fal_request_id,
                )

            if status in ("failed", "error"):
                return VideoResult(
                    request_id=request.request_id,
                    status=GenerationStatus.FAILED,
                    provider="fal",
                    model=request.model.value,
                    external_job_id=fal_request_id,
                    error_message=data.get("error", "Unknown error"),
                )

        return VideoResult(
            request_id=request.request_id,
            status=GenerationStatus.FAILED,
            provider="fal",
            model=request.model.value,
            external_job_id=fal_request_id,
            error_code="TIMEOUT",
            error_message="Job did not complete within timeout",
        )

    async def _generate_wan_local(self, request: GenerationRequest) -> VideoResult:
        """Generate video via local Wan 2.1 installation."""
        started_at = datetime.utcnow()

        self._emit_progress(request.request_id, 10, "Starting local Wan 2.1 generation")

        if not self.config.gpu.enabled:
            return VideoResult(
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                provider="wan_local",
                model=request.model.value,
                error_code="GPU_DISABLED",
                error_message="Local GPU generation is not enabled",
            )

        # TODO: Implement actual Wan 2.1 integration
        # This will call the local GPU service running Wan 2.1
        # For now, return a placeholder that indicates the integration point

        client = await self._get_client()

        # Assuming local Wan service runs on a different port
        wan_base = "http://localhost:8081"

        payload = {
            "prompt": request.prompt,
            "duration_seconds": min(request.duration_seconds, 10),  # Wan 2.1 limit
            "aspect_ratio": request.aspect_ratio,
            "model": request.model.value,
        }

        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        if request.reference_image_url:
            payload["reference_image"] = request.reference_image_url
        if request.motion_data_id:
            payload["motion_data_id"] = request.motion_data_id

        try:
            response = await client.post(
                f"{wan_base}/generate",
                json=payload,
                timeout=600.0,  # 10 min for local generation
            )

            if response.status_code != 200:
                return VideoResult(
                    request_id=request.request_id,
                    status=GenerationStatus.FAILED,
                    provider="wan_local",
                    model=request.model.value,
                    error_code=f"HTTP_{response.status_code}",
                    error_message=response.text,
                )

            data = response.json()

            return VideoResult(
                request_id=request.request_id,
                status=GenerationStatus.COMPLETED,
                video_url=data.get("video_url"),
                thumbnail_url=data.get("thumbnail_url"),
                provider="wan_local",
                model=request.model.value,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                processing_time_seconds=(
                    datetime.utcnow() - started_at
                ).total_seconds(),
                estimated_cost_usd=self._estimate_cost(
                    request.model, request.duration_seconds
                ),
            )

        except httpx.ConnectError:
            return VideoResult(
                request_id=request.request_id,
                status=GenerationStatus.FAILED,
                provider="wan_local",
                model=request.model.value,
                error_code="CONNECTION_FAILED",
                error_message="Could not connect to local Wan 2.1 service",
            )

    async def get_status(self, job_id: str, provider: str) -> VideoResult:
        """
        Get the status of a generation job.

        Args:
            job_id: The external job ID from the provider
            provider: Which provider (kie, fal, wan_local)

        Returns:
            VideoResult with current status
        """
        client = await self._get_client()

        if provider == "kie":
            headers = {
                "Authorization": f"Bearer {self.config.api.kie_api_key}",
            }
            response = await client.get(
                f"{self.config.api.kie_api_base}/video/status/{job_id}",
                headers=headers,
            )
        elif provider == "fal":
            # For Fal, we'd need to know the model to construct the URL
            # This is a simplified version
            headers = {
                "Authorization": f"Key {self.config.api.fal_api_key}",
            }
            response = await client.get(
                f"{self.config.api.fal_api_base}/requests/{job_id}/status",
                headers=headers,
            )
        else:
            return VideoResult(
                request_id=job_id,
                status=GenerationStatus.FAILED,
                error_message=f"Unknown provider: {provider}",
            )

        if response.status_code != 200:
            return VideoResult(
                request_id=job_id,
                status=GenerationStatus.FAILED,
                error_code=f"HTTP_{response.status_code}",
                error_message=response.text,
            )

        data = response.json()
        status = data.get("status", "").lower()

        status_map = {
            "pending": GenerationStatus.PENDING,
            "queued": GenerationStatus.QUEUED,
            "processing": GenerationStatus.PROCESSING,
            "completed": GenerationStatus.COMPLETED,
            "failed": GenerationStatus.FAILED,
        }

        return VideoResult(
            request_id=job_id,
            status=status_map.get(status, GenerationStatus.PROCESSING),
            video_url=data.get("video_url"),
            provider=provider,
            external_job_id=job_id,
        )

    def get_circuit_breaker_status(self) -> dict[str, dict]:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }
