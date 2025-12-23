"""
Lip Sync Client - Sync Labs + ComfyUI Wav2Lip

Usage:
    from shared.lipsync_client import apply_lip_sync
    
    result = await apply_lip_sync(
        avatar_url="https://r2.../character.png",
        audio_url="https://r2.../voiceover.mp3",
        provider="sync_labs"  # or "comfyui"
    )

Providers:
    - sync_labs: Cloud API, fast, $0.02-0.05/sec
    - comfyui: Local GPU (4060 via Tailscale), free but slower
"""

import os
import asyncio
import httpx
from typing import Optional, Literal
from pydantic import BaseModel, Field

# Langfuse OSS observability (optional)
try:
    from langfuse.decorators import observe
except ImportError:
    # Fallback: no-op decorator if langfuse not installed
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Environment
SYNC_LABS_API_KEY = os.getenv("SYNC_LABS_API_KEY")
SYNC_LABS_BASE_URL = "https://api.sync.so/v2"

COMFYUI_HOST = os.getenv("COMFYUI_HOST", "100.64.0.2")  # Tailscale IP
COMFYUI_PORT = os.getenv("COMFYUI_PORT", "8188")
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"


# ============================================================
# Models
# ============================================================

class LipSyncResult(BaseModel):
    """Result from lip sync generation"""
    provider: str
    status: Literal["completed", "failed", "processing"]
    output_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    cost: Optional[float] = None
    error_message: Optional[str] = None


# ============================================================
# Sync Labs Provider
# ============================================================

@observe(name="sync_labs_lipsync")
async def sync_labs_lipsync(
    video_or_image_url: str,
    audio_url: str,
    model: str = "lipsync-2",
    output_format: str = "mp4",
    webhook_url: Optional[str] = None,
) -> LipSyncResult:
    """
    Apply lip sync using Sync Labs cloud API.
    
    Args:
        video_or_image_url: URL of avatar image or video
        audio_url: URL of audio file
        model: lipsync-2 (standard) or lipsync-2-pro (4K, slower)
        output_format: mp4 or webm
        webhook_url: Optional webhook for async completion
    
    Returns:
        LipSyncResult with output URL
    """
    async with httpx.AsyncClient(timeout=300) as client:
        # Start generation
        response = await client.post(
            f"{SYNC_LABS_BASE_URL}/lipsync",
            headers={
                "Authorization": f"Bearer {SYNC_LABS_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "video_url": video_or_image_url,
                "audio_url": audio_url,
                "model": model,
                "output_format": output_format,
                **({"webhook_url": webhook_url} if webhook_url else {}),
            },
        )
        response.raise_for_status()
        data = response.json()
        
        job_id = data.get("id")
        
        # Poll for completion
        while True:
            status_response = await client.get(
                f"{SYNC_LABS_BASE_URL}/lipsync/{job_id}",
                headers={"Authorization": f"Bearer {SYNC_LABS_API_KEY}"},
            )
            status_response.raise_for_status()
            status_data = status_response.json()
            
            status = status_data.get("status")
            
            if status == "completed":
                # Estimate cost: $0.04/sec for lipsync-2
                duration = status_data.get("duration_seconds", 60)
                cost = duration * 0.04 if model == "lipsync-2" else duration * 0.08
                
                return LipSyncResult(
                    provider="sync_labs",
                    status="completed",
                    output_url=status_data.get("output_url"),
                    duration_seconds=duration,
                    cost=cost,
                )
            elif status == "failed":
                return LipSyncResult(
                    provider="sync_labs",
                    status="failed",
                    error_message=status_data.get("error", "Unknown error"),
                )
            
            await asyncio.sleep(5)


# ============================================================
# ComfyUI Wav2Lip Provider
# ============================================================

# Wav2Lip workflow template
WAV2LIP_WORKFLOW = {
    "1": {
        "class_type": "LoadImage",
        "inputs": {"image": "AVATAR_PATH"}
    },
    "2": {
        "class_type": "LoadAudio",
        "inputs": {"audio": "AUDIO_PATH"}
    },
    "3": {
        "class_type": "Wav2Lip",
        "inputs": {
            "images": ["1", 0],
            "audio": ["2", 0],
            "mode": "sequential",
            "face_detect_batch": 4
        }
    },
    "4": {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["3", 0],
            "audio": ["2", 0],
            "frame_rate": 30,
            "format": "video/h264-mp4"
        }
    }
}


@observe(name="comfyui_wav2lip")
async def comfyui_wav2lip(
    avatar_url: str,
    audio_url: str,
    frame_rate: int = 30,
) -> LipSyncResult:
    """
    Apply lip sync using local ComfyUI with Wav2Lip.
    
    Args:
        avatar_url: URL of avatar image
        audio_url: URL of audio file
        frame_rate: Output video frame rate
    
    Returns:
        LipSyncResult with output URL
    
    Note:
        Requires ComfyUI running on local machine with Wav2Lip node installed.
        Accessed via Tailscale at COMFYUI_HOST:COMFYUI_PORT
    """
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            # First, download and upload assets to ComfyUI
            # (In production, you'd upload to ComfyUI's input folder)
            
            # For now, assume assets are accessible via URL
            # Build the workflow with actual paths
            workflow = WAV2LIP_WORKFLOW.copy()
            workflow["1"]["inputs"]["image"] = avatar_url
            workflow["2"]["inputs"]["audio"] = audio_url
            workflow["4"]["inputs"]["frame_rate"] = frame_rate
            
            # Queue the workflow
            response = await client.post(
                f"{COMFYUI_URL}/prompt",
                json={"prompt": workflow},
            )
            response.raise_for_status()
            data = response.json()
            
            prompt_id = data.get("prompt_id")
            
            # Poll for completion
            max_wait = 600  # 10 minutes for local generation
            elapsed = 0
            
            while elapsed < max_wait:
                history_response = await client.get(
                    f"{COMFYUI_URL}/history/{prompt_id}"
                )
                history_response.raise_for_status()
                history = history_response.json()
                
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    
                    # Find video output
                    for node_id, node_output in outputs.items():
                        if "video" in node_output:
                            video_info = node_output["video"][0]
                            filename = video_info.get("filename")
                            
                            return LipSyncResult(
                                provider="comfyui",
                                status="completed",
                                output_url=f"{COMFYUI_URL}/view?filename={filename}",
                                cost=0.0,  # Free!
                            )
                
                await asyncio.sleep(5)
                elapsed += 5
            
            return LipSyncResult(
                provider="comfyui",
                status="failed",
                error_message=f"Timeout after {max_wait}s",
            )
            
    except Exception as e:
        return LipSyncResult(
            provider="comfyui",
            status="failed",
            error_message=str(e),
        )


# ============================================================
# Main Interface
# ============================================================

@observe(name="apply_lip_sync")
async def apply_lip_sync(
    avatar_url: str,
    audio_url: str,
    provider: Literal["sync_labs", "comfyui"] = "comfyui",  # Default to free local GPU
    **kwargs,
) -> LipSyncResult:
    """
    Apply lip sync to an avatar using the specified provider.
    
    Args:
        avatar_url: URL of avatar image or video
        audio_url: URL of audio file
        provider: "sync_labs" (cloud, paid) or "comfyui" (local, free)
        **kwargs: Additional provider-specific options
    
    Returns:
        LipSyncResult with output URL
    """
    if provider == "sync_labs":
        return await sync_labs_lipsync(avatar_url, audio_url, **kwargs)
    elif provider == "comfyui":
        return await comfyui_wav2lip(avatar_url, audio_url, **kwargs)
    else:
        return LipSyncResult(
            provider=provider,
            status="failed",
            error_message=f"Unknown provider: {provider}",
        )


def estimate_lipsync_cost(
    duration_seconds: float,
    provider: str = "sync_labs",
    model: str = "lipsync-2",
) -> float:
    """Estimate lip sync cost."""
    if provider == "comfyui":
        return 0.0
    elif provider == "sync_labs":
        rate = 0.04 if model == "lipsync-2" else 0.08
        return duration_seconds * rate
    return 0.0


if __name__ == "__main__":
    print("Lip sync client loaded.")
    print(f"Sync Labs API: {'configured' if SYNC_LABS_API_KEY else 'not configured'}")
    print(f"ComfyUI URL: {COMFYUI_URL}")
    print(f"\nCost for 60s video:")
    print(f"  - Sync Labs (lipsync-2): ${estimate_lipsync_cost(60, 'sync_labs', 'lipsync-2'):.2f}")
    print(f"  - Sync Labs (lipsync-2-pro): ${estimate_lipsync_cost(60, 'sync_labs', 'lipsync-2-pro'):.2f}")
    print(f"  - ComfyUI Wav2Lip: ${estimate_lipsync_cost(60, 'comfyui'):.2f}")
