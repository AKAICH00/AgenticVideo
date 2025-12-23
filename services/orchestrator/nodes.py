"""
Orchestrator Nodes

LangGraph nodes that implement each phase of video generation.
Each node is a function that takes VideoState and returns updated VideoState.

Node Pattern:
1. Read relevant state
2. Perform work (API calls, LLM invocations)
3. Record decision
4. Update state
5. Return state
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

import google.generativeai as genai
from google.generativeai import GenerationConfig

from core.config import get_config
from services.video_generation import VideoGenerationClient, VideoModel
from .state import (
    VideoState,
    GenerationPhase,
    ScriptData,
    Scene,
    StoryboardData,
    QualityMetrics,
    ShortFormClip,
    CheckpointType,
)

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """Base class for all orchestrator nodes."""

    node_name: str = "base"

    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()

    @abstractmethod
    async def execute(self, state: VideoState) -> VideoState:
        """Execute the node logic."""
        pass

    def emit_progress(self, state: VideoState, percent: int, message: str):
        """Update progress in state."""
        state.progress_percent = percent
        state.progress_message = message
        logger.info(f"[{self.node_name}] {percent}% - {message}")


class PlannerNode(BaseNode):
    """
    Planner Agent - Analyzes input and creates execution plan.

    Responsibilities:
    - Understand content brief
    - Analyze reference videos if provided
    - Set creative direction
    - Estimate resources needed
    """

    node_name: str = "planner"

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        import os
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    async def execute(self, state: VideoState) -> VideoState:
        state.phase = GenerationPhase.PLANNING
        state.started_at = state.started_at or datetime.utcnow()
        self.emit_progress(state, 5, "Analyzing content brief")

        # Check if we have intelligence data from trend analysis
        trend_context = ""
        if state.meta and "brief" in state.meta:
            brief = state.meta["brief"]
            trend_context = f"""
INTELLIGENCE CONTEXT:
- Trend Score: {state.meta.get('trend_score', 'N/A')}
- Opportunity Type: {state.meta.get('opportunity_type', 'N/A')}
- Suggested Hooks: {brief.get('hooks', [])}
- Angle: {brief.get('angle', 'N/A')}
"""

        prompt = f"""You are a video production planner. Analyze this content brief and create an execution plan.

TOPIC: {state.topic}
NICHE: {state.niche}
TARGET AUDIENCE: {state.target_audience}
STYLE: {state.style_reference}
TARGET DURATION: {state.target_duration_seconds} seconds
QUALITY TIER: {state.quality_tier}
HAS REFERENCE VIDEO: {state.reference_video_url is not None}
{trend_context}

Create a brief plan covering:
1. Key message and hook strategy
2. Recommended number of scenes
3. Visual style direction
4. Audio/music recommendations
5. Estimated production complexity (low/medium/high)

Be concise - this informs the other agents."""

        response = await asyncio.to_thread(
            self.model.generate_content,
            prompt,
            generation_config=GenerationConfig(
                temperature=0.7,
                max_output_tokens=1024,
            ),
        )

        plan = response.text

        # Record the planning decision
        state.add_decision(
            agent_name="planner",
            decision=f"Created execution plan for {state.topic}",
            reasoning=plan,
            confidence=0.85,
        )

        # Estimate cost based on quality tier
        if state.quality_tier == "premium":
            state.estimated_cost_usd = state.target_duration_seconds * 0.15
        else:
            state.estimated_cost_usd = state.target_duration_seconds * 0.003

        self.emit_progress(state, 10, "Planning complete")
        return state


class ScriptNode(BaseNode):
    """
    Script Writer Agent - Generates the video script.

    Responsibilities:
    - Generate compelling hooks
    - Write body content with viral patterns
    - Create variations for A/B testing
    """

    node_name: str = "script"

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        import os
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    async def execute(self, state: VideoState) -> VideoState:
        state.phase = GenerationPhase.SCRIPTING
        self.emit_progress(state, 15, "Generating script")

        # Get planning context from decisions
        planning_context = ""
        for decision in state.decisions:
            if decision.agent_name == "planner":
                planning_context = decision.reasoning
                break

        # Use pre-generated hooks from intelligence layer if available
        hook_suggestions = ""
        if state.meta and "hooks" in state.meta and state.meta["hooks"]:
            hook_suggestions = f"\nSUGGESTED HOOKS (from trend analysis):\n" + "\n".join(
                f"- {h}" for h in state.meta["hooks"][:3]
            )

        prompt = f"""You are a viral content script writer. Write a script for this video.

TOPIC: {state.topic}
NICHE: {state.niche}
TARGET DURATION: {state.target_duration_seconds} seconds
STYLE: {state.style_reference}

PLANNING CONTEXT:
{planning_context}
{hook_suggestions}

Create a script with:
1. HOOK (first 3 seconds - CRUCIAL for retention)
2. BODY (main content)
3. CALL TO ACTION (end)
4. KEY POINTS (3-5 bullet points)

Format as JSON:
{{
    "hook": "...",
    "body": "...",
    "call_to_action": "...",
    "key_points": ["...", "..."],
    "tone": "...",
    "estimated_duration_seconds": ...
}}"""

        response = await asyncio.to_thread(
            self.model.generate_content,
            prompt,
            generation_config=GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )

        # Parse script from response
        content = response.text

        try:
            import json

            # Extract JSON from response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content

            script_data = json.loads(json_str.strip())

            state.script = ScriptData(
                hook=script_data.get("hook", ""),
                body=script_data.get("body", ""),
                call_to_action=script_data.get("call_to_action", ""),
                estimated_duration_seconds=script_data.get(
                    "estimated_duration_seconds", state.target_duration_seconds
                ),
                key_points=script_data.get("key_points", []),
                tone=script_data.get("tone", "engaging"),
            )

        except json.JSONDecodeError:
            # Fallback: use raw content
            state.script = ScriptData(
                hook="",
                body=content,
                call_to_action="",
                estimated_duration_seconds=state.target_duration_seconds,
            )

        state.add_decision(
            agent_name="script",
            decision=f"Generated script with hook: {state.script.hook[:50]}...",
            reasoning=f"Tone: {state.script.tone}, Duration: {state.script.estimated_duration_seconds}s",
            confidence=0.8,
        )

        self.emit_progress(state, 25, "Script generation complete")
        return state


class StoryboardNode(BaseNode):
    """
    Storyboard Agent - Creates scene-by-scene breakdown.

    Responsibilities:
    - Break script into scenes
    - Define visual prompts for each scene
    - Specify camera movements
    - Plan transitions
    """

    node_name: str = "storyboard"

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        import os
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    async def execute(self, state: VideoState) -> VideoState:
        state.phase = GenerationPhase.STORYBOARDING
        self.emit_progress(state, 30, "Creating storyboard")

        prompt = f"""You are a visual storyboard artist. Create a scene-by-scene breakdown.

SCRIPT:
Hook: {state.script.hook}
Body: {state.script.body}
CTA: {state.script.call_to_action}

TARGET DURATION: {state.script.estimated_duration_seconds} seconds
STYLE: {state.style_reference}
NICHE: {state.niche}

Create scenes as JSON array. Each scene should be 3-10 seconds:
[
    {{
        "scene_number": 1,
        "description": "Brief description of what happens",
        "visual_prompt": "Detailed prompt for AI video generation",
        "duration_seconds": 5,
        "camera_motion": "static|pan_left|pan_right|zoom_in|zoom_out|track",
        "transition_in": "cut|fade|dissolve|wipe",
        "transition_out": "cut|fade|dissolve|wipe",
        "audio_cue": "What audio/music note for this scene"
    }}
]

Aim for 6-12 scenes total. Be specific in visual_prompt - it goes directly to video AI."""

        response = await asyncio.to_thread(
            self.model.generate_content,
            prompt,
            generation_config=GenerationConfig(
                temperature=0.6,
                max_output_tokens=4096,
            ),
        )

        content = response.text
        scenes = []

        try:
            import json

            # Extract JSON array from response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                # Try to find JSON array
                start = content.find("[")
                end = content.rfind("]") + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                else:
                    json_str = content

            scenes_data = json.loads(json_str.strip())

            for scene_data in scenes_data:
                scenes.append(
                    Scene(
                        scene_id=str(uuid4()),
                        scene_number=scene_data.get("scene_number", len(scenes) + 1),
                        description=scene_data.get("description", ""),
                        visual_prompt=scene_data.get("visual_prompt", ""),
                        duration_seconds=scene_data.get("duration_seconds", 5),
                        camera_motion=scene_data.get("camera_motion", "static"),
                        transition_in=scene_data.get("transition_in", "cut"),
                        transition_out=scene_data.get("transition_out", "cut"),
                        audio_cue=scene_data.get("audio_cue", ""),
                    )
                )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse storyboard JSON: {e}")
            # Create a single fallback scene
            scenes.append(
                Scene(
                    scene_id=str(uuid4()),
                    scene_number=1,
                    description=state.script.body[:100],
                    visual_prompt=state.script.body,
                    duration_seconds=state.script.estimated_duration_seconds,
                )
            )

        state.storyboard = StoryboardData(
            scenes=scenes,
            total_duration_seconds=sum(s.duration_seconds for s in scenes),
        )

        state.add_decision(
            agent_name="storyboard",
            decision=f"Created {len(scenes)} scenes totaling {state.storyboard.total_duration_seconds}s",
            reasoning="Scene breakdown based on script pacing and content",
            confidence=0.75,
        )

        self.emit_progress(state, 40, f"Storyboard complete: {len(scenes)} scenes")
        return state


class MotionNode(BaseNode):
    """
    Motion Agent - Extracts motion data from reference videos.

    Responsibilities:
    - Extract pose keypoints (DWPose/OpenPose)
    - Track camera motion (CoTracker)
    - Detect transitions (PySceneDetect)
    - Analyze beat/rhythm
    """

    node_name: str = "motion"

    async def execute(self, state: VideoState) -> VideoState:
        state.phase = GenerationPhase.MOTION_EXTRACTION

        if not state.reference_video_url:
            self.emit_progress(state, 45, "No reference video - skipping motion extraction")
            state.motion_data.extraction_complete = True
            return state

        self.emit_progress(state, 45, "Extracting motion from reference video")

        # TODO: Implement actual motion extraction with:
        # - DWPose/OpenPose for pose extraction
        # - CoTracker for camera motion
        # - PySceneDetect for transitions

        # For now, mark as complete (placeholder for motion service integration)
        state.motion_data.reference_video_url = state.reference_video_url
        state.motion_data.extraction_complete = True

        state.add_decision(
            agent_name="motion",
            decision="Motion extraction completed",
            reasoning=f"Analyzed reference video: {state.reference_video_url}",
            confidence=0.9,
        )

        self.emit_progress(state, 50, "Motion extraction complete")
        return state


class VisualNode(BaseNode):
    """
    Visual Agent - Generates video clips for each scene.

    Responsibilities:
    - Generate video for each scene using appropriate model
    - Apply motion data if available
    - Ensure style consistency across scenes
    - Handle retries for failed generations
    """

    node_name: str = "visual"

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        self.video_client = VideoGenerationClient(config=self.config)

    async def execute(self, state: VideoState) -> VideoState:
        state.phase = GenerationPhase.VISUAL_GENERATION
        self.emit_progress(state, 55, "Generating visuals")

        scenes = state.storyboard.scenes
        total_scenes = len(scenes)

        # Select model based on quality tier
        model = (
            VideoModel.RUNWAY_GEN4_5
            if state.quality_tier == "premium"
            else VideoModel.WAN_2_1
        )

        for i, scene in enumerate(scenes):
            if scene.status == "completed":
                continue

            scene.status = "processing"
            progress = 55 + (i * 30 // total_scenes)
            self.emit_progress(
                state, progress, f"Generating scene {i + 1}/{total_scenes}"
            )

            try:
                result = await self.video_client.generate(
                    prompt=scene.visual_prompt,
                    model=model,
                    duration_seconds=int(scene.duration_seconds),
                    quality_tier=state.quality_tier,
                    scene_id=scene.scene_id,
                    campaign_id=state.campaign_id,
                )

                if result.video_url:
                    scene.status = "completed"
                    scene.video_url = result.video_url
                    scene.thumbnail_url = result.thumbnail_url
                    scene.generation_job_id = result.external_job_id

                    state.actual_cost_usd += result.actual_cost_usd or result.estimated_cost_usd or 0
                else:
                    # Log the actual failure reason from the result
                    error_msg = result.error_message or f"No video URL returned (status: {result.status})"
                    logger.warning(f"Scene {i + 1} video generation failed: {error_msg}")
                    scene.status = "failed"
                    scene.retry_count += 1
                    state.add_error("visual_generation", f"Scene {i + 1}: {error_msg}")

            except Exception as e:
                logger.error(f"Scene {i + 1} generation failed: {e}")
                scene.status = "failed"
                scene.retry_count += 1
                state.add_error("visual_generation", str(e))

        completed = sum(1 for s in scenes if s.status == "completed")
        state.add_decision(
            agent_name="visual",
            decision=f"Generated {completed}/{total_scenes} scenes",
            reasoning=f"Using {model.value} for {state.quality_tier} tier",
            confidence=completed / total_scenes if total_scenes > 0 else 0,
        )

        self.emit_progress(state, 85, f"Visual generation complete: {completed}/{total_scenes}")
        return state


class ComposeNode(BaseNode):
    """
    Composition Agent - Assembles the final video.

    Responsibilities:
    - Assemble scenes with Remotion
    - Apply transitions
    - Sync audio
    - Render final output
    """

    node_name: str = "compose"

    async def execute(self, state: VideoState) -> VideoState:
        state.phase = GenerationPhase.COMPOSITION
        self.emit_progress(state, 88, "Compositing video")

        # Check if all scenes are ready
        completed_scenes = [s for s in state.storyboard.scenes if s.status == "completed"]
        if not completed_scenes:
            state.add_error("composition", "No completed scenes to compose", recoverable=False)
            state.phase = GenerationPhase.FAILED
            return state

        # Prepare scene data for Remotion
        from services.rendering import compose_video

        scenes_data = [
            {
                "order": i,
                "url": scene.video_url,
                "type": "video",
                "duration_seconds": scene.duration_seconds,
            }
            for i, scene in enumerate(completed_scenes)
        ]

        # Determine composition type based on format
        format_type = state.meta.get("format_type", "short") if state.meta else "short"
        composition = "ViralShort" if format_type == "short" else "YouTubeVideo"

        self.emit_progress(state, 89, f"Rendering {composition} via Remotion")

        try:
            output_url, error = await compose_video(
                scenes=scenes_data,
                composition=composition,
                audio_url=state.meta.get("audio_url") if state.meta else None,
                title=state.topic if composition == "YouTubeVideo" else None,
                metadata={
                    "topic": state.topic,
                    "niche": state.niche,
                },
            )

            if output_url:
                state.long_form_video_url = output_url
                state.long_form_thumbnail_url = completed_scenes[0].thumbnail_url

                state.add_decision(
                    agent_name="compose",
                    decision=f"Composed {composition} from {len(completed_scenes)} scenes",
                    reasoning=f"Rendered via Remotion API: {output_url}",
                    confidence=0.9,
                )
            else:
                # Fallback to first scene if Remotion fails
                logger.warning(f"Remotion render failed: {error}, using fallback")
                state.long_form_video_url = completed_scenes[0].video_url
                state.long_form_thumbnail_url = completed_scenes[0].thumbnail_url

                state.add_decision(
                    agent_name="compose",
                    decision=f"Fallback composition using first scene",
                    reasoning=f"Remotion unavailable: {error}",
                    confidence=0.5,
                )

        except Exception as e:
            logger.error(f"Composition failed: {e}")
            # Fallback
            state.long_form_video_url = completed_scenes[0].video_url
            state.long_form_thumbnail_url = completed_scenes[0].thumbnail_url
            state.add_error("composition", str(e), recoverable=True)

        self.emit_progress(state, 92, "Composition complete")
        return state


class QualityNode(BaseNode):
    """
    Quality Reviewer Agent - Evaluates and approves output.

    Implements the GenMAC 4-agent redesign pattern:
    1. Verification: Does this match the brief?
    2. Suggestion: What could be improved?
    3. Correction: Apply specific fixes
    4. Structuring: Format for next stage
    """

    node_name: str = "quality"

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        self.quality_threshold = 0.7

    async def execute(self, state: VideoState) -> VideoState:
        state.phase = GenerationPhase.QUALITY_CHECK
        self.emit_progress(state, 94, "Performing quality check")

        # Evaluate quality
        metrics = await self._evaluate_quality(state)
        state.quality_metrics = metrics

        if metrics.overall_score >= self.quality_threshold:
            state.add_decision(
                agent_name="quality",
                decision="Quality check passed",
                reasoning=f"Score: {metrics.overall_score:.2f}, exceeds threshold {self.quality_threshold}",
                confidence=metrics.overall_score,
            )

            # Move to repurposing
            state.phase = GenerationPhase.REPURPOSING
            self.emit_progress(state, 96, "Quality check passed")

        else:
            # Check retry count
            if state.retry_count < state.max_retries:
                state.retry_count += 1
                state.add_decision(
                    agent_name="quality",
                    decision=f"Quality check failed - retry {state.retry_count}/{state.max_retries}",
                    reasoning=f"Score: {metrics.overall_score:.2f}, below threshold. Issues: {metrics.feedback}",
                    confidence=metrics.overall_score,
                )

                # Mark scenes for regeneration
                for scene_id in metrics.requires_regeneration:
                    for scene in state.storyboard.scenes:
                        if scene.scene_id == scene_id:
                            scene.status = "pending"
                            scene.retry_count += 1

                # Go back to visual generation
                state.phase = GenerationPhase.VISUAL_GENERATION
                self.emit_progress(state, 50, f"Regenerating {len(metrics.requires_regeneration)} scenes")

            else:
                state.add_decision(
                    agent_name="quality",
                    decision="Quality check failed - max retries exceeded",
                    reasoning=f"Score: {metrics.overall_score:.2f}. Proceeding with best effort.",
                    confidence=metrics.overall_score,
                )
                state.phase = GenerationPhase.REPURPOSING
                self.emit_progress(state, 96, "Proceeding with best effort")

        return state

    async def _evaluate_quality(self, state: VideoState) -> QualityMetrics:
        """Evaluate the quality of generated content."""
        metrics = QualityMetrics()

        # Calculate scene completion rate
        completed = sum(1 for s in state.storyboard.scenes if s.status == "completed")
        total = len(state.storyboard.scenes)

        if total == 0:
            metrics.overall_score = 0.0
            metrics.feedback.append("No scenes generated")
            return metrics

        completion_rate = completed / total
        metrics.visual_quality = completion_rate

        # Identify failed scenes for regeneration
        for scene in state.storyboard.scenes:
            if scene.status == "failed":
                metrics.requires_regeneration.append(scene.scene_id)
                metrics.feedback.append(f"Scene {scene.scene_number} failed to generate")

        # Calculate overall score (simplified)
        metrics.overall_score = completion_rate
        metrics.consistency = 0.8 if completion_rate > 0.8 else 0.5
        metrics.hook_strength = 0.8 if state.script.hook else 0.3

        return metrics


class RepurposeNode(BaseNode):
    """
    Repurpose Agent - Generates short-form clips from long-form.

    Responsibilities:
    - Detect viral moments
    - Generate short-form cuts
    - Optimize hooks for each platform
    """

    node_name: str = "repurpose"

    async def execute(self, state: VideoState) -> VideoState:
        state.phase = GenerationPhase.REPURPOSING
        self.emit_progress(state, 97, "Generating short-form clips")

        if not state.long_form_video_url:
            state.add_decision(
                agent_name="repurpose",
                decision="Skipping repurposing - no long-form video",
                reasoning="Long-form video not available",
                confidence=1.0,
            )
            state.phase = GenerationPhase.COMPLETE
            return state

        # TODO: Implement moment detection and auto-clipping
        # This would:
        # 1. Analyze video for high-moment segments
        # 2. Extract 15-60 second clips
        # 3. Add hooks and captions
        # 4. Optimize for each platform (TikTok, Reels, Shorts)

        # Placeholder clips based on scenes
        for i, scene in enumerate(state.storyboard.scenes[:3]):  # Top 3 moments
            clip = ShortFormClip(
                clip_id=str(uuid4()),
                start_time=sum(s.duration_seconds for s in state.storyboard.scenes[:i]),
                end_time=sum(s.duration_seconds for s in state.storyboard.scenes[:i + 1]),
                moment_score=0.8 - (i * 0.1),
                hook_text=state.script.hook,
                target_platform=["tiktok", "shorts", "reels"][i % 3],
                clip_url=scene.video_url,
                status="pending" if scene.video_url else "failed",
            )
            state.short_form_clips.append(clip)

        state.add_decision(
            agent_name="repurpose",
            decision=f"Created {len(state.short_form_clips)} short-form clips",
            reasoning="Extracted top moments for TikTok, Shorts, and Reels",
            confidence=0.8,
        )

        state.phase = GenerationPhase.PUBLISHING
        self.emit_progress(state, 98, "Ready for publishing")

        return state


class PublishNode(BaseNode):
    """
    Publish Agent - Prepares and publishes video to multiple platforms.

    Responsibilities:
    - Generate SEO-optimized metadata (title, description, tags)
    - Calculate optimal posting time based on niche
    - Publish to YouTube, TikTok via Blotato API
    - Track in database for analytics
    """

    node_name: str = "publish"

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        import os
        self.blotato_enabled = bool(os.environ.get("BLOTATO_API_KEY"))

    async def execute(self, state: VideoState) -> VideoState:
        state.phase = GenerationPhase.PUBLISHING
        self.emit_progress(state, 98, "Preparing for publication")

        if not state.long_form_video_url:
            state.add_decision(
                agent_name="publish",
                decision="Skipping publishing - no video available",
                reasoning="Long-form video URL not set",
                confidence=1.0,
            )
            state.phase = GenerationPhase.COMPLETE
            state.completed_at = datetime.utcnow()
            return state

        try:
            # Import publisher services
            from services.publisher import (
                generate_metadata,
                schedule_video,
                BlotaoClient,
                YouTubeTarget,
                TikTokTarget,
                YouTubePrivacy,
            )

            # Generate SEO metadata
            self.emit_progress(state, 98, "Generating SEO metadata")

            script_hook = state.script.hook if state.script else None
            script_summary = state.script.body[:500] if state.script and state.script.body else None

            metadata = await generate_metadata(
                topic=state.topic,
                niche=state.niche,
                script_hook=script_hook,
                script_summary=script_summary,
                format_type="short" if state.target_duration_seconds <= 60 else "long",
                target_audience=state.target_audience,
            )

            # Store metadata in state
            if not state.meta:
                state.meta = {}

            state.meta["seo_metadata"] = metadata

            # Calculate optimal posting time
            self.emit_progress(state, 99, "Calculating optimal posting time")

            schedule = await schedule_video(
                niche=state.niche,
                format_type="short" if state.target_duration_seconds <= 60 else "long",
                campaign_id=state.campaign_id,
            )

            state.meta["scheduled_publish"] = schedule

            # Publish via Blotato if enabled
            if self.blotato_enabled:
                self.emit_progress(state, 99, "Publishing via Blotato")

                try:
                    client = BlotaoClient()

                    # Build targets based on format
                    targets = []
                    is_short = state.target_duration_seconds <= 60

                    # YouTube target
                    targets.append(YouTubeTarget(
                        title=metadata["title"],
                        description=metadata.get("description", ""),
                        tags=metadata.get("tags", []),
                        privacy_status=YouTubePrivacy.PUBLIC,
                        should_notify_subscribers=True,
                        contains_synthetic_media=True,  # AI-generated content
                    ))

                    # TikTok for shorts only
                    if is_short:
                        # Build TikTok caption with hashtags
                        tiktok_tags = metadata.get("tags", [])[:5]
                        tiktok_title = f"{metadata['title'][:100]} #{' #'.join(tiktok_tags)}" if tiktok_tags else metadata["title"][:150]

                        targets.append(TikTokTarget(
                            title=tiktok_title,
                            ai_generated_content=True,
                        ))

                    # Publish to all targets
                    results = await client.publish_video(
                        video_url=state.long_form_video_url,
                        targets=targets,
                        thumbnail_url=state.long_form_thumbnail_url,
                        scheduled_time=schedule.get("scheduled_datetime"),
                    )

                    await client.close()

                    # Process results
                    state.meta["publish_results"] = []
                    success_count = 0
                    for result in results:
                        state.meta["publish_results"].append({
                            "platform": result.platform,
                            "success": result.success,
                            "post_id": result.post_id,
                            "url": result.published_url,
                            "error": result.error,
                        })
                        if result.success:
                            success_count += 1

                    state.add_decision(
                        agent_name="publish",
                        decision=f"Published to {success_count}/{len(results)} platforms",
                        reasoning=f"YouTube: {results[0].published_url if results and results[0].success else 'failed'}",
                        confidence=success_count / len(results) if results else 0,
                    )

                except Exception as e:
                    logger.error(f"Blotato publishing failed: {e}")
                    state.meta["publish_status"] = "queued"
                    state.meta["publish_error"] = str(e)
                    state.add_error("publishing", f"Blotato: {e}", recoverable=True)

            else:
                # No Blotato - queue for manual or background worker
                state.meta["publish_status"] = "queued"
                state.meta["publish_queued_at"] = datetime.utcnow().isoformat()

                state.add_decision(
                    agent_name="publish",
                    decision=f"Queued for: {schedule['scheduled_time']}",
                    reasoning=f"SEO title: {metadata['title'][:50]}... | {schedule['reasoning']}",
                    confidence=schedule["confidence_score"],
                )

        except ImportError as e:
            # Publisher service not available - skip gracefully
            logger.warning(f"Publisher service not available: {e}")
            state.add_decision(
                agent_name="publish",
                decision="Publisher service unavailable",
                reasoning=f"Import failed: {e}",
                confidence=0.5,
            )

        except Exception as e:
            logger.error(f"Publish preparation failed: {e}")
            state.add_error("publishing", str(e), recoverable=True)
            state.add_decision(
                agent_name="publish",
                decision="Publish preparation failed",
                reasoning=str(e),
                confidence=0.0,
            )

        state.phase = GenerationPhase.COMPLETE
        state.completed_at = datetime.utcnow()
        self.emit_progress(state, 100, "Video generation complete")

        return state
