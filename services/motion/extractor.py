"""
Motion Extractor - Orchestrates all motion extraction from reference videos.

Combines multiple extraction methods:
1. Pose extraction (DWPose/OpenPose) for dance/movement
2. Camera tracking (CoTracker) for pan/tilt/zoom
3. Transition detection (PySceneDetect) for cuts
4. Audio analysis for beat sync
5. Style embedding (CLIP) for visual style matching
"""

import asyncio
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import httpx

from core.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class PoseKeypoint:
    """A single pose keypoint."""
    name: str
    x: float
    y: float
    confidence: float


@dataclass
class PoseFrame:
    """Pose data for a single frame."""
    frame_number: int
    timestamp_seconds: float
    keypoints: list[PoseKeypoint]
    overall_confidence: float


@dataclass
class CameraMotion:
    """Camera motion data for a single frame."""
    frame_number: int
    timestamp_seconds: float
    pan: float  # Horizontal movement (-1 to 1)
    tilt: float  # Vertical movement (-1 to 1)
    zoom: float  # Zoom factor (1.0 = no zoom)
    roll: float  # Rotation
    shake: float  # Camera shake intensity (0 to 1)


@dataclass
class Transition:
    """A detected transition/cut in the video."""
    frame_number: int
    timestamp_seconds: float
    transition_type: str  # "cut", "fade", "dissolve", "wipe"
    duration_frames: int
    confidence: float


@dataclass
class BeatSync:
    """Audio beat detection data."""
    frame_number: int
    timestamp_seconds: float
    beat_number: int
    bpm: float
    strength: float  # Beat intensity


@dataclass
class ExtractionResult:
    """Complete motion extraction result."""
    extraction_id: str = field(default_factory=lambda: str(uuid4()))
    reference_video_url: str = ""

    # Video metadata
    duration_seconds: float = 0
    fps: float = 0
    width: int = 0
    height: int = 0
    frame_count: int = 0

    # Extracted data
    poses: list[PoseFrame] = field(default_factory=list)
    camera_motion: list[CameraMotion] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    beat_sync: list[BeatSync] = field(default_factory=list)

    # Style analysis
    style_tags: list[str] = field(default_factory=list)
    color_palette: list[str] = field(default_factory=list)
    mood: str = ""
    energy_level: float = 0.0  # 0 to 1

    # Processing metadata
    pose_format: str = "dwpose"
    model_versions: dict[str, str] = field(default_factory=dict)
    processing_time_seconds: float = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "extraction_id": self.extraction_id,
            "reference_video_url": self.reference_video_url,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "pose_keypoints": [
                {
                    "frame": p.frame_number,
                    "timestamp": p.timestamp_seconds,
                    "keypoints": [
                        {"name": k.name, "x": k.x, "y": k.y, "confidence": k.confidence}
                        for k in p.keypoints
                    ],
                    "confidence": p.overall_confidence,
                }
                for p in self.poses
            ],
            "camera_motion": [
                {
                    "frame": c.frame_number,
                    "timestamp": c.timestamp_seconds,
                    "pan": c.pan,
                    "tilt": c.tilt,
                    "zoom": c.zoom,
                    "roll": c.roll,
                    "shake": c.shake,
                }
                for c in self.camera_motion
            ],
            "transitions": [
                {
                    "frame": t.frame_number,
                    "timestamp": t.timestamp_seconds,
                    "type": t.transition_type,
                    "duration_frames": t.duration_frames,
                    "confidence": t.confidence,
                }
                for t in self.transitions
            ],
            "beat_sync": [
                {
                    "frame": b.frame_number,
                    "timestamp": b.timestamp_seconds,
                    "beat": b.beat_number,
                    "bpm": b.bpm,
                    "strength": b.strength,
                }
                for b in self.beat_sync
            ],
            "style_tags": self.style_tags,
            "color_palette": self.color_palette,
            "mood": self.mood,
            "energy_level": self.energy_level,
            "pose_format": self.pose_format,
            "model_versions": self.model_versions,
            "processing_time_seconds": self.processing_time_seconds,
        }


class MotionExtractor:
    """
    Main motion extraction orchestrator.

    Coordinates all extraction methods and combines results.

    Usage:
        extractor = MotionExtractor()

        # Full extraction
        result = await extractor.extract(video_url)

        # Specific extractions
        poses = await extractor.extract_poses(video_url)
        camera = await extractor.extract_camera(video_url)
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()
        self._http_client: Optional[httpx.AsyncClient] = None

        # Lazy-loaded extractors
        self._pose_extractor = None
        self._camera_tracker = None
        self._transition_detector = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=300.0)
        return self._http_client

    async def close(self):
        """Close resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def download_video(self, url: str) -> Path:
        """
        Download video to temporary file.

        Args:
            url: Video URL (can be TikTok, YouTube, or direct URL)

        Returns:
            Path to downloaded video file
        """
        client = await self._get_client()

        # Handle different URL types
        if "tiktok.com" in url:
            download_url = await self._get_tiktok_download_url(url)
        elif "youtube.com" in url or "youtu.be" in url:
            download_url = await self._get_youtube_download_url(url)
        else:
            download_url = url

        # Download to temp file
        response = await client.get(download_url, follow_redirects=True)
        response.raise_for_status()

        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_file.write(response.content)
        temp_file.close()

        logger.info(f"Downloaded video to {temp_file.name}")
        return Path(temp_file.name)

    async def _get_tiktok_download_url(self, url: str) -> str:
        """Extract direct video URL from TikTok."""
        # TODO: Implement TikTok URL extraction
        # This could use a service like tikwm.com or self-hosted scraper
        logger.warning("TikTok extraction not implemented, using URL directly")
        return url

    async def _get_youtube_download_url(self, url: str) -> str:
        """Extract direct video URL from YouTube."""
        # TODO: Implement YouTube URL extraction via yt-dlp
        logger.warning("YouTube extraction not implemented, using URL directly")
        return url

    async def get_video_metadata(self, video_path: Path) -> dict:
        """
        Extract video metadata using ffprobe.

        Returns:
            Dict with duration, fps, width, height, frame_count
        """
        import subprocess
        import json

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)

            video_stream = next(
                (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
                {},
            )

            return {
                "duration_seconds": float(data.get("format", {}).get("duration", 0)),
                "fps": eval(video_stream.get("r_frame_rate", "30/1")),
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "frame_count": int(video_stream.get("nb_frames", 0)),
            }

        except Exception as e:
            logger.error(f"Failed to get video metadata: {e}")
            return {
                "duration_seconds": 0,
                "fps": 30,
                "width": 0,
                "height": 0,
                "frame_count": 0,
            }

    async def extract(
        self,
        video_url: str,
        extract_poses: bool = True,
        extract_camera: bool = True,
        extract_transitions: bool = True,
        extract_beats: bool = True,
        extract_style: bool = True,
    ) -> ExtractionResult:
        """
        Full motion extraction from a reference video.

        Args:
            video_url: URL of the reference video
            extract_poses: Whether to extract pose data
            extract_camera: Whether to extract camera motion
            extract_transitions: Whether to detect transitions
            extract_beats: Whether to analyze audio beats
            extract_style: Whether to analyze visual style

        Returns:
            ExtractionResult with all extracted data
        """
        start_time = datetime.utcnow()
        result = ExtractionResult(reference_video_url=video_url)

        logger.info(f"Starting motion extraction for: {video_url}")

        try:
            # Download video
            video_path = await self.download_video(video_url)

            # Get metadata
            metadata = await self.get_video_metadata(video_path)
            result.duration_seconds = metadata["duration_seconds"]
            result.fps = metadata["fps"]
            result.width = metadata["width"]
            result.height = metadata["height"]
            result.frame_count = metadata["frame_count"]

            # Run extractions in parallel
            tasks = []

            if extract_poses:
                tasks.append(self._extract_poses(video_path, result))
            if extract_camera:
                tasks.append(self._extract_camera(video_path, result))
            if extract_transitions:
                tasks.append(self._extract_transitions(video_path, result))
            if extract_beats:
                tasks.append(self._extract_beats(video_path, result))
            if extract_style:
                tasks.append(self._extract_style(video_path, result))

            await asyncio.gather(*tasks, return_exceptions=True)

            # Cleanup temp file
            video_path.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Motion extraction failed: {e}")
            raise

        result.processing_time_seconds = (
            datetime.utcnow() - start_time
        ).total_seconds()

        logger.info(
            f"Motion extraction complete: "
            f"{len(result.poses)} pose frames, "
            f"{len(result.camera_motion)} camera frames, "
            f"{len(result.transitions)} transitions, "
            f"{len(result.beat_sync)} beats"
        )

        return result

    async def _extract_poses(self, video_path: Path, result: ExtractionResult):
        """Extract pose keypoints from video."""
        from .pose import PoseExtractor

        if self._pose_extractor is None:
            self._pose_extractor = PoseExtractor(config=self.config)

        poses = await self._pose_extractor.extract(video_path)
        result.poses = poses
        result.model_versions["pose"] = self._pose_extractor.model_version

    async def _extract_camera(self, video_path: Path, result: ExtractionResult):
        """Extract camera motion from video."""
        from .camera import CameraTracker

        if self._camera_tracker is None:
            self._camera_tracker = CameraTracker(config=self.config)

        camera_motion = await self._camera_tracker.track(video_path)
        result.camera_motion = camera_motion
        result.model_versions["camera"] = self._camera_tracker.model_version

    async def _extract_transitions(self, video_path: Path, result: ExtractionResult):
        """Detect transitions and cuts in video."""
        from .transitions import TransitionDetector

        if self._transition_detector is None:
            self._transition_detector = TransitionDetector()

        transitions = await self._transition_detector.detect(video_path)
        result.transitions = transitions

    async def _extract_beats(self, video_path: Path, result: ExtractionResult):
        """Analyze audio for beat detection."""
        # TODO: Implement beat detection using librosa or similar
        # This would extract audio and find beat positions
        logger.info("Beat extraction placeholder")
        pass

    async def _extract_style(self, video_path: Path, result: ExtractionResult):
        """Analyze visual style using CLIP embeddings."""
        # TODO: Implement style analysis
        # This would:
        # 1. Sample key frames
        # 2. Generate CLIP embeddings
        # 3. Classify style characteristics
        result.style_tags = ["dynamic", "high-energy"]
        result.mood = "energetic"
        result.energy_level = 0.8
        logger.info("Style extraction placeholder")
