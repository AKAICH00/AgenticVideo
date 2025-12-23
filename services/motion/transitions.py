"""
Transition Detection using PySceneDetect

Detects scene cuts and transitions in video:
- Hard cuts
- Fade in/out
- Dissolves
- Wipes

Uses PySceneDetect for accurate cut detection.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from .extractor import Transition

logger = logging.getLogger(__name__)


class TransitionDetector:
    """
    Detects transitions and scene changes in video.

    Uses PySceneDetect with multiple detection methods:
    - ContentDetector: Detects cuts by analyzing content changes
    - ThresholdDetector: Detects fade-outs to black
    - AdaptiveDetector: Adapts to varying content

    Usage:
        detector = TransitionDetector()
        transitions = await detector.detect(video_path)

        for t in transitions:
            print(f"Frame {t.frame_number}: {t.transition_type}")
    """

    def __init__(
        self,
        threshold: float = 27.0,  # Content change threshold
        min_scene_len: int = 15,  # Minimum frames between cuts
    ):
        """
        Initialize transition detector.

        Args:
            threshold: Sensitivity threshold (lower = more sensitive)
            min_scene_len: Minimum scene length in frames
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    async def detect(
        self,
        video_path: Path,
        detect_fades: bool = True,
    ) -> list[Transition]:
        """
        Detect all transitions in video.

        Args:
            video_path: Path to video file
            detect_fades: Also detect fade transitions

        Returns:
            List of detected transitions
        """
        logger.info(f"Detecting transitions in {video_path}")

        transitions = []

        # Detect cuts using content analysis
        cuts = await self._detect_cuts(video_path)
        transitions.extend(cuts)

        # Detect fades
        if detect_fades:
            fades = await self._detect_fades(video_path)
            transitions.extend(fades)

        # Sort by frame number
        transitions.sort(key=lambda t: t.frame_number)

        logger.info(f"Detected {len(transitions)} transitions")
        return transitions

    async def _detect_cuts(self, video_path: Path) -> list[Transition]:
        """
        Detect hard cuts using content analysis.

        Uses PySceneDetect's ContentDetector.
        """
        try:
            # Try to use PySceneDetect
            from scenedetect import detect, ContentDetector

            scene_list = detect(str(video_path), ContentDetector(
                threshold=self.threshold,
                min_scene_len=self.min_scene_len,
            ))

            transitions = []
            for i, scene in enumerate(scene_list):
                if i > 0:  # Skip first scene (start)
                    transitions.append(Transition(
                        frame_number=scene[0].get_frames(),
                        timestamp_seconds=scene[0].get_seconds(),
                        transition_type="cut",
                        duration_frames=1,
                        confidence=0.9,
                    ))

            return transitions

        except ImportError:
            logger.warning("PySceneDetect not available, using fallback detection")
            return await self._detect_cuts_fallback(video_path)

    async def _detect_cuts_fallback(self, video_path: Path) -> list[Transition]:
        """
        Fallback cut detection using frame differencing.

        Less accurate but doesn't require PySceneDetect.
        """
        # TODO: Implement frame differencing cut detection
        # This would:
        # 1. Extract frames at intervals
        # 2. Calculate histogram difference between consecutive frames
        # 3. Mark large differences as cuts

        return []

    async def _detect_fades(self, video_path: Path) -> list[Transition]:
        """
        Detect fade transitions (fade to/from black).
        """
        try:
            from scenedetect import detect, ThresholdDetector

            # Threshold detector finds fades to/from black
            scene_list = detect(str(video_path), ThresholdDetector(
                threshold=12,
                min_scene_len=self.min_scene_len,
            ))

            transitions = []
            for i, scene in enumerate(scene_list):
                if i > 0:
                    transitions.append(Transition(
                        frame_number=scene[0].get_frames(),
                        timestamp_seconds=scene[0].get_seconds(),
                        transition_type="fade",
                        duration_frames=15,  # Typical fade duration
                        confidence=0.8,
                    ))

            return transitions

        except ImportError:
            return []

    async def detect_dissolves(self, video_path: Path) -> list[Transition]:
        """
        Detect dissolve transitions.

        Dissolves are harder to detect as they're gradual changes
        between two scenes. Uses motion analysis to find them.
        """
        # TODO: Implement dissolve detection
        # This would:
        # 1. Look for periods of high inter-frame blur
        # 2. Analyze luminance gradient patterns
        # 3. Detect overlapping content

        return []

    def analyze_pacing(self, transitions: list[Transition]) -> dict:
        """
        Analyze the pacing/rhythm of transitions.

        Returns statistics about cut frequency and timing patterns.
        """
        if len(transitions) < 2:
            return {
                "avg_scene_length_seconds": 0,
                "cuts_per_minute": 0,
                "pacing": "unknown",
            }

        # Calculate scene lengths
        scene_lengths = []
        for i in range(1, len(transitions)):
            length = (
                transitions[i].timestamp_seconds -
                transitions[i - 1].timestamp_seconds
            )
            scene_lengths.append(length)

        avg_length = sum(scene_lengths) / len(scene_lengths) if scene_lengths else 0
        cuts_per_minute = 60 / avg_length if avg_length > 0 else 0

        # Classify pacing
        if cuts_per_minute > 60:
            pacing = "very_fast"
        elif cuts_per_minute > 30:
            pacing = "fast"
        elif cuts_per_minute > 15:
            pacing = "medium"
        elif cuts_per_minute > 5:
            pacing = "slow"
        else:
            pacing = "very_slow"

        return {
            "avg_scene_length_seconds": avg_length,
            "cuts_per_minute": cuts_per_minute,
            "pacing": pacing,
            "total_cuts": len(transitions),
            "min_scene_length": min(scene_lengths) if scene_lengths else 0,
            "max_scene_length": max(scene_lengths) if scene_lengths else 0,
        }

    def match_to_beats(
        self,
        transitions: list[Transition],
        beat_timestamps: list[float],
        tolerance_seconds: float = 0.1,
    ) -> list[tuple[Transition, Optional[float]]]:
        """
        Match transitions to audio beats.

        Returns list of (transition, matched_beat) tuples.
        """
        matches = []

        for trans in transitions:
            # Find closest beat
            closest_beat = None
            min_diff = float('inf')

            for beat_time in beat_timestamps:
                diff = abs(trans.timestamp_seconds - beat_time)
                if diff < min_diff:
                    min_diff = diff
                    closest_beat = beat_time

            if closest_beat and min_diff <= tolerance_seconds:
                matches.append((trans, closest_beat))
            else:
                matches.append((trans, None))

        return matches
