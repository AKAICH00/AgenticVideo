"""
Visual Router - Routes visual generation tasks to optimal backend

Usage:
    from shared.visual_router import route_visual_task, generate_scene
    
    # Get recommended backend
    backend = route_visual_task("text_to_video", priority="quality")
    
    # Generate with automatic routing
    result = await generate_scene(scene, priority="cost")

Routing Logic:
    - "quality" priority: Use best available (Kie.ai Veo 3, Sync Labs)
    - "cost" priority: Use cheapest (ComfyUI local, Wav2Lip)
    - Automatic fallback if primary fails
"""

import os
from typing import Literal, Optional
from pydantic import BaseModel
# Langfuse OSS observability (optional)
try:
    from langfuse.decorators import observe
except ImportError:
    def observe(*args, **kwargs):
        def decorator(func): return func
        return decorator

from .kie_client import KieClient, GenerationResult as KieResult
from .lipsync_client import apply_lip_sync, LipSyncResult

# ============================================================
# Routing Configuration
# ============================================================

ROUTING_TABLE = {
    "text_to_video": {
        "quality": {"backend": "kie.ai", "model": "wan"},
        "balanced": {"backend": "kie.ai", "model": "kling"},
        "cost": {"backend": "kie.ai", "model": "hailuo"},
    },
    "text_to_image": {
        "quality": {"backend": "kie.ai", "model": "flux"},
        "balanced": {"backend": "kie.ai", "model": "seedream"},
        "cost": {"backend": "kie.ai", "model": "ideogram"},
    },
    "video_to_video": {
        "quality": {"backend": "kie.ai", "model": "kling_i2v"},
        "balanced": {"backend": "kie.ai", "model": "kling_i2v"},
        "cost": {"backend": "comfyui", "model": "controlnet"},
    },
    "lip_sync": {
        "quality": {"backend": "comfyui", "model": "wav2lip"},
        "balanced": {"backend": "comfyui", "model": "wav2lip"},
        "cost": {"backend": "comfyui", "model": "wav2lip"},
    },
    "image_generation": {
        "quality": {"backend": "kie.ai", "model": "flux"},
        "balanced": {"backend": "kie.ai", "model": "seedream"},
        "cost": {"backend": "kie.ai", "model": "ideogram"},
    },
    "music_generation": {
        "quality": {"backend": "kie.ai", "model": "elevenlabs"},
        "balanced": {"backend": "kie.ai", "model": "elevenlabs"},
        "cost": {"backend": "kie.ai", "model": "elevenlabs"},
    },
}

# Cost per second/image for estimation
COST_ESTIMATES = {
    # Video (per second)
    "veo3_quality": 0.40,
    "veo3_fast": 0.12,
    "kling": 0.10,
    "runway": 0.15,
    # Image (per image)
    "midjourney": 0.03,
    "flux_pro": 0.01,
    "flux_schnell": 0.002,
    # Lip sync (per second)
    "lipsync-2": 0.04,
    "lipsync-2-pro": 0.08,
    "wav2lip": 0.0,
    # ComfyUI (free)
    "controlnet": 0.0,
}


# ============================================================
# Router Functions
# ============================================================

def route_visual_task(
    task_type: str,
    priority: Literal["quality", "balanced", "cost"] = "balanced",
) -> dict:
    """
    Get the recommended backend and model for a task.
    
    Args:
        task_type: text_to_video, text_to_image, video_to_video, lip_sync
        priority: quality, balanced, or cost
    
    Returns:
        Dict with "backend" and "model" keys
    """
    task_config = ROUTING_TABLE.get(task_type, {})
    return task_config.get(priority, {"backend": "kie.ai", "model": "flux_pro"})


def estimate_cost(
    task_type: str,
    priority: str,
    duration_seconds: Optional[float] = None,
    image_count: int = 1,
) -> float:
    """
    Estimate cost for a visual generation task.
    
    Args:
        task_type: Type of task
        priority: Quality priority
        duration_seconds: For video/audio tasks
        image_count: For image tasks
    
    Returns:
        Estimated cost in dollars
    """
    route = route_visual_task(task_type, priority)
    model = route["model"]
    rate = COST_ESTIMATES.get(model, 0.0)
    
    if task_type in ["text_to_video", "video_to_video", "lip_sync", "music_generation"]:
        return rate * (duration_seconds or 5)
    else:
        return rate * image_count


# ============================================================
# Scene Generation
# ============================================================

class VisualScene(BaseModel):
    """A visual scene to generate"""
    scene_order: int
    visual_prompt: str
    scene_type: str
    duration_seconds: float = 5.0
    reference_video_url: Optional[str] = None  # For V2V
    

class SceneResult(BaseModel):
    """Result from scene generation"""
    scene_order: int
    backend: str
    model: str
    status: str
    output_url: Optional[str] = None
    cost: float = 0.0
    error_message: Optional[str] = None


@observe(name="generate_scene", run_type="tool")
async def generate_scene(
    scene: VisualScene,
    priority: Literal["quality", "balanced", "cost"] = "balanced",
    fallback_on_error: bool = True,
) -> SceneResult:
    """
    Generate a visual scene with automatic routing.
    
    Args:
        scene: The scene to generate
        priority: Quality priority
        fallback_on_error: Try cheaper option if primary fails
    
    Returns:
        SceneResult with output URL
    """
    # Determine task type
    if scene.reference_video_url:
        task_type = "video_to_video"
    else:
        task_type = "text_to_video"
    
    # Get route
    route = route_visual_task(task_type, priority)
    backend = route["backend"]
    model = route["model"]
    
    # Generate based on backend
    if backend == "kie.ai":
        client = KieClient()
        
        try:
            if scene.reference_video_url:
                result = await client.generate_video(
                    prompt=scene.visual_prompt,
                    model=model,
                    duration=int(scene.duration_seconds),
                    reference_video_url=scene.reference_video_url,
                )
            else:
                result = await client.generate_video(
                    prompt=scene.visual_prompt,
                    model=model,
                    duration=int(scene.duration_seconds),
                )
            
            if result.status == "completed":
                return SceneResult(
                    scene_order=scene.scene_order,
                    backend=backend,
                    model=model,
                    status="completed",
                    output_url=result.output_url,
                    cost=result.cost_credits or estimate_cost(task_type, priority, scene.duration_seconds),
                )
            else:
                error = result.error_message or "Unknown error"
                
                # Fallback to cheaper option
                if fallback_on_error and priority != "cost":
                    return await generate_scene(scene, priority="cost", fallback_on_error=False)
                
                return SceneResult(
                    scene_order=scene.scene_order,
                    backend=backend,
                    model=model,
                    status="failed",
                    error_message=error,
                )
                
        except Exception as e:
            if fallback_on_error and priority != "cost":
                return await generate_scene(scene, priority="cost", fallback_on_error=False)
            
            return SceneResult(
                scene_order=scene.scene_order,
                backend=backend,
                model=model,
                status="failed",
                error_message=str(e),
            )
    
    elif backend == "comfyui":
        # ComfyUI generation (to be implemented)
        return SceneResult(
            scene_order=scene.scene_order,
            backend=backend,
            model=model,
            status="failed",
            error_message="ComfyUI video generation not yet implemented",
        )
    
    return SceneResult(
        scene_order=scene.scene_order,
        backend=backend,
        model=model,
        status="failed",
        error_message=f"Unknown backend: {backend}",
    )


# ============================================================
# Batch Generation
# ============================================================

@observe(name="generate_all_scenes", run_type="chain")
async def generate_all_scenes(
    scenes: list[VisualScene],
    priority: Literal["quality", "balanced", "cost"] = "balanced",
) -> list[SceneResult]:
    """
    Generate all scenes for a video.
    
    Args:
        scenes: List of scenes to generate
        priority: Quality priority
    
    Returns:
        List of SceneResults
    """
    import asyncio
    
    tasks = [generate_scene(scene, priority) for scene in scenes]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert exceptions to failed results
    processed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed.append(SceneResult(
                scene_order=scenes[i].scene_order,
                backend="unknown",
                model="unknown",
                status="failed",
                error_message=str(result),
            ))
        else:
            processed.append(result)
    
    return processed


if __name__ == "__main__":
    print("Visual Router Configuration:")
    print("\nRouting Table:")
    for task, priorities in ROUTING_TABLE.items():
        print(f"\n  {task}:")
        for priority, config in priorities.items():
            cost = estimate_cost(task, priority, duration_seconds=5)
            print(f"    {priority}: {config['backend']}/{config['model']} (${cost:.3f}/5s)")
