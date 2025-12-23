"""
Video Generation Graph

LangGraph workflow definition for video generation.
Implements cyclic feedback loops and quality gates.

Graph Structure:
    START
      ↓
    PLANNER → SCRIPT → STORYBOARD
                           ↓
                  ┌──→ MOTION ──┐
                  │             ↓
                  │          VISUAL
                  │             ↓
                  │         QUALITY ──┐
                  │      (pass)  │    │(fail + retries left)
                  │         ↓    └────┘
                  │     COMPOSE
                  │         ↓
                  │    REPURPOSE
                  │         ↓
                  │     PUBLISH (SEO metadata + scheduling)
                  │         ↓
                  └───── COMPLETE

Key Features:
- Cyclic retry loop for quality failures (max 3 retries)
- Human checkpoints at key decision points
- SSE progress streaming
- Automated publishing with SEO optimization
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional, Union
from uuid import uuid4

from .state import VideoState, GenerationPhase, Checkpoint
from .nodes import (
    PlannerNode,
    ScriptNode,
    StoryboardNode,
    MotionNode,
    VisualNode,
    ComposeNode,
    QualityNode,
    RepurposeNode,
    PublishNode,
)

logger = logging.getLogger(__name__)


class VideoGraph:
    """
    LangGraph-style workflow for video generation.

    Usage:
        graph = VideoGraph(on_progress=my_callback)

        state = VideoState(
            topic="How AI is changing marketing",
            niche="tech",
            target_duration_seconds=60,
        )

        final_state = await graph.run(state)
        print(final_state.long_form_video_url)
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        on_progress: Optional[Callable[[VideoState], Union[None, Awaitable[None]]]] = None,
        on_checkpoint: Optional[Callable[[VideoState, "Checkpoint"], Union[None, Awaitable[None]]]] = None,
        checkpoint_resolver: Optional[Callable[[str], Awaitable[Optional[tuple[str, Optional[str]]]]]] = None,
        checkpoint_timeout: int = 3600,  # 1 hour default
    ):
        """
        Initialize the video generation graph.

        Args:
            config: Configuration object
            on_progress: Callback for progress updates (can be sync or async)
            on_checkpoint: Callback when checkpoint is created (for SSE notification)
            checkpoint_resolver: Async function that waits for checkpoint resolution.
                                Returns (resolution, feedback) tuple or None if timeout.
            checkpoint_timeout: Default timeout in seconds for checkpoints
        """
        self.config = config
        self.on_progress = on_progress
        self.on_checkpoint = on_checkpoint
        self.checkpoint_resolver = checkpoint_resolver
        self.checkpoint_timeout = checkpoint_timeout

        # Initialize nodes
        self.nodes = {
            "planner": PlannerNode(config),
            "script": ScriptNode(config),
            "storyboard": StoryboardNode(config),
            "motion": MotionNode(config),
            "visual": VisualNode(config),
            "quality": QualityNode(config),
            "compose": ComposeNode(config),
            "repurpose": RepurposeNode(config),
            "publish": PublishNode(config),
        }

        # Define edges (transitions)
        self.edges = {
            GenerationPhase.PENDING: "planner",
            GenerationPhase.PLANNING: "script",
            GenerationPhase.SCRIPTING: "storyboard",
            GenerationPhase.STORYBOARDING: "motion",
            GenerationPhase.MOTION_EXTRACTION: "visual",
            GenerationPhase.VISUAL_GENERATION: "quality",
            GenerationPhase.QUALITY_CHECK: self._route_after_quality,
            GenerationPhase.COMPOSITION: "repurpose",
            GenerationPhase.REPURPOSING: "publish",
            GenerationPhase.PUBLISHING: "publish",
            GenerationPhase.COMPLETE: None,  # Terminal
            GenerationPhase.FAILED: None,  # Terminal
        }

    def _route_after_quality(self, state: VideoState) -> Optional[str]:
        """Determine next node after quality check."""
        if state.phase == GenerationPhase.VISUAL_GENERATION:
            # Quality check failed, go back to visual
            return "visual"
        elif state.phase == GenerationPhase.REPURPOSING:
            # Quality check passed, proceed to compose
            return "compose"
        elif state.phase == GenerationPhase.COMPLETE:
            return None
        elif state.phase == GenerationPhase.FAILED:
            return None
        else:
            return "compose"

    async def _emit_progress(self, state: VideoState):
        """Emit progress update via callback (supports sync and async callbacks)."""
        if self.on_progress:
            try:
                result = self.on_progress(state)
                # If callback is async, await it
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    async def _emit_checkpoint(self, state: VideoState, checkpoint: Checkpoint):
        """Emit checkpoint notification via callback."""
        if self.on_checkpoint:
            try:
                result = self.on_checkpoint(state, checkpoint)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Checkpoint callback failed: {e}")

    async def _wait_for_checkpoint_resolution(
        self,
        checkpoint: Checkpoint,
        timeout: Optional[int] = None,
    ) -> Optional[tuple[str, Optional[str]]]:
        """
        Wait for checkpoint to be resolved by human.

        Args:
            checkpoint: The checkpoint to wait for
            timeout: Timeout in seconds (uses default if None)

        Returns:
            (resolution, feedback) tuple or None if timeout/no resolver
        """
        if not self.checkpoint_resolver:
            logger.warning("No checkpoint resolver configured, auto-approving")
            return ("approved", None)

        timeout = timeout or self.checkpoint_timeout

        try:
            # Wait for resolution with timeout
            result = await asyncio.wait_for(
                self.checkpoint_resolver(checkpoint.checkpoint_id),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Checkpoint {checkpoint.checkpoint_id} timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Checkpoint resolution failed: {e}")
            return None

    async def run(
        self,
        state: VideoState,
        start_from: Optional[str] = None,
    ) -> VideoState:
        """
        Run the video generation workflow.

        Args:
            state: Initial state
            start_from: Optional node to start from (for resumption)

        Returns:
            Final state with generated video
        """
        state.session_id = state.session_id or str(uuid4())
        logger.info(f"Starting video generation workflow: {state.session_id}")

        current_node = start_from

        while True:
            # Determine which node to run
            if current_node is None:
                edge = self.edges.get(state.phase)

                if edge is None:
                    # Terminal state
                    logger.info(f"Workflow complete: {state.phase.value}")
                    break

                if callable(edge):
                    current_node = edge(state)
                    if current_node is None:
                        break
                else:
                    current_node = edge

            # Execute the node
            node = self.nodes.get(current_node)
            if node is None:
                logger.error(f"Unknown node: {current_node}")
                state.add_error("graph", f"Unknown node: {current_node}", recoverable=False)
                state.phase = GenerationPhase.FAILED
                break

            try:
                logger.info(f"Executing node: {current_node}")
                state = await node.execute(state)
                await self._emit_progress(state)

                # Check for pending checkpoint
                if state.pending_checkpoint:
                    checkpoint = state.pending_checkpoint
                    logger.info(
                        f"Checkpoint pending: {checkpoint.checkpoint_type.value} "
                        f"(id={checkpoint.checkpoint_id})"
                    )

                    # Emit checkpoint notification (for SSE)
                    await self._emit_checkpoint(state, checkpoint)

                    # Wait for human resolution
                    resolution = await self._wait_for_checkpoint_resolution(checkpoint)

                    if resolution is None:
                        # Timeout or error - mark as rejected
                        logger.warning(f"Checkpoint {checkpoint.checkpoint_id} not resolved, rejecting")
                        state.resolve_checkpoint("timeout")
                        state.add_error(
                            current_node,
                            f"Checkpoint timeout: {checkpoint.proposal[:100]}",
                            recoverable=True,
                        )
                    else:
                        resolution_type, feedback = resolution
                        logger.info(f"Checkpoint resolved: {resolution_type}")
                        state.resolve_checkpoint(resolution_type, feedback)

                        # Handle rejection
                        if resolution_type == "rejected":
                            logger.info("Checkpoint rejected, workflow will retry or fail")
                            if state.retry_count < state.max_retries:
                                state.retry_count += 1
                                # Stay on current node to retry
                                continue
                            else:
                                state.phase = GenerationPhase.FAILED
                                state.add_error(
                                    current_node,
                                    f"Checkpoint rejected after {state.max_retries} retries",
                                    recoverable=False,
                                )
                                break

                    # Emit progress after checkpoint resolution
                    await self._emit_progress(state)

            except Exception as e:
                logger.error(f"Node {current_node} failed: {e}")
                state.add_error(current_node, str(e))

                # Check if recoverable
                if state.retry_count < state.max_retries:
                    state.retry_count += 1
                    logger.info(f"Retrying ({state.retry_count}/{state.max_retries})")
                    continue
                else:
                    state.phase = GenerationPhase.FAILED
                    break

            # Clear current node to let routing decide next
            current_node = None

        await self._emit_progress(state)
        return state

    async def run_single_node(
        self,
        node_name: str,
        state: VideoState,
    ) -> VideoState:
        """
        Run a single node (for testing/debugging).

        Args:
            node_name: Name of node to run
            state: Current state

        Returns:
            Updated state
        """
        node = self.nodes.get(node_name)
        if node is None:
            raise ValueError(f"Unknown node: {node_name}")

        return await node.execute(state)


def create_video_workflow(
    config: Optional[Any] = None,
    on_progress: Optional[Callable[[VideoState], None]] = None,
) -> VideoGraph:
    """
    Factory function to create a video generation workflow.

    Args:
        config: Optional configuration
        on_progress: Optional progress callback

    Returns:
        Configured VideoGraph instance
    """
    return VideoGraph(config=config, on_progress=on_progress)


# Convenience function for quick generation
async def generate_video(
    topic: str,
    niche: str = "general",
    target_duration_seconds: int = 60,
    quality_tier: str = "premium",
    reference_video_url: Optional[str] = None,
    on_progress: Optional[Callable[[VideoState], None]] = None,
) -> VideoState:
    """
    High-level function to generate a video.

    Args:
        topic: Video topic/title
        niche: Content niche
        target_duration_seconds: Target video length
        quality_tier: "premium" or "bulk"
        reference_video_url: Optional TikTok/reference for motion
        on_progress: Progress callback

    Returns:
        Final VideoState with generated content

    Example:
        result = await generate_video(
            topic="5 AI Tools for Productivity",
            niche="tech",
            target_duration_seconds=90,
            quality_tier="premium",
        )
        print(result.long_form_video_url)
    """
    state = VideoState(
        topic=topic,
        niche=niche,
        target_duration_seconds=target_duration_seconds,
        quality_tier=quality_tier,
        reference_video_url=reference_video_url,
    )

    graph = create_video_workflow(on_progress=on_progress)
    return await graph.run(state)
