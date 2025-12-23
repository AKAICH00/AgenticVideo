"""
Motion Extraction Service

Extracts motion data from reference videos for motion transfer.

Capabilities:
- Pose extraction (DWPose/OpenPose)
- Camera motion tracking (CoTracker)
- Transition detection (PySceneDetect)
- Beat/rhythm analysis
- Style embedding (CLIP)

Usage:
    extractor = MotionExtractor()
    motion_data = await extractor.extract(video_url)

    # Apply motion to new content
    video_client = VideoGenerationClient()
    result = await video_client.generate(
        prompt="...",
        motion_data_id=motion_data.id,
    )
"""

from .extractor import MotionExtractor, ExtractionResult
from .pose import PoseExtractor
from .camera import CameraTracker
from .transitions import TransitionDetector

__all__ = [
    "MotionExtractor",
    "ExtractionResult",
    "PoseExtractor",
    "CameraTracker",
    "TransitionDetector",
]
