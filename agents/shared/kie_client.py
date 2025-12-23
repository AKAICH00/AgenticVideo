"""
Kie.ai Client - Multi-model AI generation via Market API

Usage:
    from shared.kie_client import KieClient
    
    client = KieClient()
    result = await client.generate_video(prompt, model="kling")

Supported Models (Market API):
    Video: kling, wan, hailuo, sora2
    Image: flux, seedream, ideogram, recraft
    Audio: elevenlabs (via kie.ai proxy)

API Documentation: https://docs.kie.ai/market/quickstart
"""

import os
import json
import asyncio
import httpx
from typing import Optional, Dict, Any, Literal, List
from pydantic import BaseModel
# Langfuse OSS observability (optional)
try:
    from langfuse.decorators import observe
except ImportError:
    def observe(*args, **kwargs):
        def decorator(func): return func
        return decorator
from tenacity import retry, stop_after_attempt, wait_exponential

# Environment
KIE_API_KEY = os.getenv("KIE_API_KEY")
KIE_API_URL = os.getenv("KIE_API_URL", "https://api.kie.ai/api/v1")


# ============================================================
# Models
# ============================================================

class GenerationResult(BaseModel):
    """Result from a kie.ai generation"""
    task_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    output_url: Optional[str] = None
    output_urls: Optional[List[str]] = None
    cost_credits: Optional[float] = None
    generation_time_seconds: Optional[float] = None
    error_message: Optional[str] = None


# Model name mappings for Market API
MODEL_MAPPINGS: Dict[str, str] = {
    # Video models - mapped to Market API model names
    "kling": "kling-2.6/text-to-video",
    "kling_i2v": "kling-2.6/image-to-video",
    "wan": "wan-2.6/text-to-video",
    "hailuo": "hailuo-i2v/text-to-video",
    "sora2": "sora2/text-to-video",
    # Image models
    "flux": "flux-2/text-to-image",
    "seedream": "seedream/text-to-image",
    "ideogram": "ideogram-v2/text-to-image",
    "recraft": "recraft/text-to-image",
    "grok": "grok-imagine/text-to-image",
}

# Cost estimates per model (approximate)
MODEL_COSTS: Dict[str, float] = {
    "kling": 0.10,  # per 5s clip
    "wan": 0.15,
    "hailuo": 0.08,
    "flux": 0.01,  # per image
    "seedream": 0.02,
}


# ============================================================
# Kie.ai Client - Market API
# ============================================================

class KieClient:
    """Client for kie.ai Market API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or KIE_API_KEY
        self.base_url = KIE_API_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    async def _create_task(
        self,
        model: str,
        input_params: Dict[str, Any],
        callback_url: Optional[str] = None,
    ) -> str:
        """Create a generation task via Market API."""
        body = {
            "model": model,
            "input": input_params,
        }
        if callback_url:
            body["callBackUrl"] = callback_url
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/jobs/createTask",
                headers=self.headers,
                json=body,
            )
            data = response.json()
            
            if data.get("code") != 200:
                raise ValueError(f"Kie.ai API error: {data.get('msg', 'Unknown error')}")
            
            return data["data"]["taskId"]
    
    async def _poll_for_completion(
        self,
        task_id: str,
        max_wait: int = 300,
        poll_interval: int = 5,
    ) -> GenerationResult:
        """Poll for task completion via Market API."""
        elapsed = 0
        
        async with httpx.AsyncClient(timeout=30) as client:
            while elapsed < max_wait:
                response = await client.get(
                    f"{self.base_url}/jobs/recordInfo?taskId={task_id}",
                    headers=self.headers,
                )
                data = response.json()
                
                if data.get("code") != 200:
                    return GenerationResult(
                        task_id=task_id,
                        status="failed",
                        error_message=data.get("msg", "API error"),
                    )
                
                task_data = data.get("data", {})
                state = task_data.get("state", "unknown")
                
                if state == "success":
                    # Parse resultJson to get output URLs
                    result_json_str = task_data.get("resultJson", "{}")
                    try:
                        result_json = json.loads(result_json_str)
                        output_urls = result_json.get("resultUrls", [])
                    except json.JSONDecodeError:
                        output_urls = []
                    
                    return GenerationResult(
                        task_id=task_id,
                        status="completed",
                        output_url=output_urls[0] if output_urls else None,
                        output_urls=output_urls,
                        generation_time_seconds=task_data.get("costTime"),
                    )
                elif state == "failed":
                    return GenerationResult(
                        task_id=task_id,
                        status="failed",
                        error_message=task_data.get("failMsg") or "Generation failed",
                    )
                
                # Still processing
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
        
        return GenerationResult(
            task_id=task_id,
            status="failed",
            error_message=f"Timeout after {max_wait}s",
        )
    
    @observe(name="kie_generate_video", run_type="tool")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    async def generate_video(
        self,
        prompt: str,
        model: str = "kling",
        duration: int = 5,
        aspect_ratio: str = "16:9",
        sound: bool = False,
        image_url: Optional[str] = None,  # For I2V
    ) -> GenerationResult:
        """
        Generate a video clip using kie.ai Market API.
        
        Args:
            prompt: Text prompt for video generation
            model: Model to use (kling, wan, hailuo, sora2)
            duration: Duration in seconds (5 or 10)
            aspect_ratio: "16:9", "9:16", or "1:1"
            sound: Whether to generate sound
            image_url: Reference image for image-to-video
        
        Returns:
            GenerationResult with output URL
        """
        # Get the full model name for Market API
        model_name = MODEL_MAPPINGS.get(model, model)
        
        # Build input params
        input_params: Dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": str(duration),
            "sound": sound,
        }
        
        if image_url:
            input_params["imageUrl"] = image_url
        
        # Create task
        task_id = await self._create_task(model_name, input_params)
        print(f"[Kie.ai] Created task: {task_id}")
        
        # Poll for completion
        result = await self._poll_for_completion(task_id, max_wait=300)
        
        return result
    
    @observe(name="kie_generate_image", run_type="tool")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    async def generate_image(
        self,
        prompt: str,
        model: str = "flux",
        aspect_ratio: str = "16:9",
        negative_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate an image using kie.ai Market API.
        
        Args:
            prompt: Text prompt for image generation
            model: Model to use (flux, seedream, ideogram, recraft)
            aspect_ratio: Image aspect ratio
            negative_prompt: What to avoid
        
        Returns:
            GenerationResult with output URL
        """
        model_name = MODEL_MAPPINGS.get(model, model)
        
        input_params: Dict[str, Any] = {
            "prompt": prompt,
            "aspectRatio": aspect_ratio,
        }
        
        if negative_prompt:
            input_params["negativePrompt"] = negative_prompt
        
        task_id = await self._create_task(model_name, input_params)
        print(f"[Kie.ai] Created image task: {task_id}")
        
        result = await self._poll_for_completion(task_id, max_wait=120)
        
        return result
    
    def estimate_cost(self, model: str, duration: Optional[int] = None) -> float:
        """Estimate cost for a generation."""
        base_cost = MODEL_COSTS.get(model, 0.10)
        if duration and model in ["kling", "wan", "hailuo"]:
            return base_cost * (duration / 5)
        return base_cost


# ============================================================
# Convenience Functions
# ============================================================

_client: Optional[KieClient] = None

def get_client() -> KieClient:
    """Get or create singleton client."""
    global _client
    if _client is None:
        _client = KieClient()
    return _client


async def generate_video(prompt: str, **kwargs) -> GenerationResult:
    """Convenience function for video generation."""
    return await get_client().generate_video(prompt, **kwargs)


async def generate_image(prompt: str, **kwargs) -> GenerationResult:
    """Convenience function for image generation."""
    return await get_client().generate_image(prompt, **kwargs)


if __name__ == "__main__":
    async def test():
        client = KieClient()
        print(f"Estimated cost for 5s Kling: ${client.estimate_cost('kling', 5):.2f}")
        print(f"Estimated cost for Flux image: ${client.estimate_cost('flux'):.3f}")
        
        # Test video generation
        if os.getenv("KIE_API_KEY"):
            result = await client.generate_video(
                prompt="A beautiful sunset over the ocean, cinematic 4k",
                model="kling",
                duration=5,
            )
            print(f"Result: {result}")
    
    asyncio.run(test())
