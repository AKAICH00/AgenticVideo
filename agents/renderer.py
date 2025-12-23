"""
Renderer Agent - TTS, lip sync, and final video rendering

The Renderer agent is the final step of the pipeline:
1. Polls for campaigns with status='rendering'
2. Generates TTS audio with word-level timestamps
3. Applies lip sync if enabled (avatar talking head)
4. Calls Remotion SSR to compose final video
5. Uploads to R2 and updates campaign

Usage:
    python renderer.py  # Run as daemon
    
    # Or import and use directly
    from renderer import RendererAgent
    agent = RendererAgent()
    await agent.run_once()
"""

import os
import json
import asyncio
import httpx
import base64
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel
# Langfuse OSS observability (optional)
try:
    from langfuse.decorators import observe
except ImportError:
    def observe(*args, **kwargs):
        def decorator(func): return func
        return decorator

from shared.db import get_db, Database
from shared.tts_client import (
    generate_speech_with_timestamps,
    format_subtitles_for_remotion,
    TTSResult,
)
from shared.lipsync_client import apply_lip_sync, LipSyncResult

# Environment
POLL_INTERVAL = int(os.getenv("RENDERER_POLL_INTERVAL", "30"))
REMOTION_API_URL = os.getenv("REMOTION_API_URL", "https://api.cckeeper.dev")
REMOTION_COMPOSITION_ID = os.getenv("REMOTION_COMPOSITION_ID", "stoic-teaching")

R2_BUCKET_URL = os.getenv("R2_PUBLIC_URL", "https://r2.example.com")


# ============================================================
# Models
# ============================================================

class RenderJob(BaseModel):
    """A job ready for rendering"""
    campaign_id: Union[str, int]
    script_id: Union[str, int]
    topic: str
    niche: str
    hook_line: str
    body_text: str
    call_to_action: str
    lip_sync_enabled: bool = False
    lip_sync_provider: Optional[str] = None
    avatar_image_url: Optional[str] = None
    scenes: List[Dict[str, Any]] = []


class VisualScene(BaseModel):
    """A visual scene for rendering"""
    scene_order: int
    asset_url: str
    duration_seconds: float
    scene_type: str


# ============================================================
# Renderer Agent
# ============================================================

class RendererAgent:
    """
    The Renderer agent handles:
    - TTS generation with timestamps
    - Lip sync application
    - Remotion rendering
    - Final video upload
    """
    
    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_db()
    
    @observe(name="renderer_poll_jobs", run_type="chain")
    async def poll_render_jobs(self) -> List[RenderJob]:
        """Poll for campaigns ready for rendering."""
        query = """
            SELECT 
                c.id as campaign_id,
                s.id as script_id,
                c.topic, c.niche,
                s.hook_line, s.body_text, s.call_to_action,
                c.lip_sync_enabled, c.lip_sync_provider,
                c.avatar_image_url
            FROM video_campaigns c
            JOIN script_drafts s ON s.campaign_id = c.id AND s.is_approved = TRUE
            WHERE c.status = 'rendering'
            ORDER BY c.created_at ASC
            LIMIT 2
        """
        rows = await self.db.fetch_all(query)
        
        jobs = []
        for row in rows:
            # Get visual scenes
            scenes = await self._get_visual_scenes(row["script_id"])
            
            jobs.append(RenderJob(
                campaign_id=row["campaign_id"],
                script_id=row["script_id"],
                topic=row["topic"],
                niche=row["niche"],
                hook_line=row["hook_line"],
                body_text=row["body_text"],
                call_to_action=row["call_to_action"],
                lip_sync_enabled=row.get("lip_sync_enabled", False),
                lip_sync_provider=row.get("lip_sync_provider"),
                avatar_image_url=row.get("avatar_image_url"),
                scenes=scenes,
            ))
        
        return jobs
    
    async def _get_visual_scenes(self, script_id: str) -> List[Dict[str, Any]]:
        """Get completed visual scenes for a script."""
        query = """
            SELECT 
                scene_order, generated_asset_url,
                visual_prompt
            FROM visual_scenes
            WHERE script_id = %s
            AND generation_status = 'completed'
            ORDER BY scene_order ASC
        """
        rows = await self.db.fetch_all(query, [script_id])
        
        return [
            {
                "order": row["scene_order"],
                "url": row["generated_asset_url"],
                "prompt": row["visual_prompt"],
            }
            for row in rows
        ]
    
    @observe(name="renderer_generate_tts", run_type="chain")
    async def generate_tts(self, job: RenderJob) -> TTSResult:
        """Generate TTS audio with word-level timestamps."""
        await self._update_campaign_status(
            job.campaign_id,
            current_action="Generating voiceover..."
        )
        
        # Combine script text
        full_script = f"{job.hook_line}\n\n{job.body_text}\n\n{job.call_to_action}"
        
        # Generate TTS with timestamps
        result = await generate_speech_with_timestamps(
            text=full_script,
            voice="marcus",  # Default Stoic voice
        )
        
        # Save audio URL to script
        # In production, upload to R2 first
        await self._save_audio_to_script(job.script_id, result)
        
        return result
    
    async def _save_audio_to_script(self, script_id: str, tts_result: TTSResult):
        """Save TTS result to script_drafts table."""
        subtitles_json = format_subtitles_for_remotion(tts_result.subtitles)
        
        query = """
            UPDATE script_drafts
            SET audio_url = %s, subtitles = %s
            WHERE id = %s
        """
        # For now, store base64 audio URL placeholder
        # In production, upload to R2 and store URL
        audio_url = f"data:audio/mp3;base64,{tts_result.audio_base64[:100]}..."
        
        await self.db.execute(query, [
            audio_url,
            json.dumps(subtitles_json),
            script_id,
        ])
    
    @observe(name="renderer_apply_lipsync", run_type="chain")
    async def apply_lipsync(
        self,
        job: RenderJob,
        audio_url: str,
    ) -> Optional[LipSyncResult]:
        """Apply lip sync to avatar if enabled."""
        if not job.lip_sync_enabled or not job.avatar_image_url:
            return None
        
        await self._update_campaign_status(
            job.campaign_id,
            current_action="Applying lip sync to avatar..."
        )
        
        provider = job.lip_sync_provider or "sync_labs"
        
        result = await apply_lip_sync(
            avatar_url=job.avatar_image_url,
            audio_url=audio_url,
            provider=provider,
        )
        
        return result
    
    @observe(name="renderer_call_remotion", run_type="chain")
    async def render_video(
        self,
        job: RenderJob,
        tts_result: TTSResult,
        lipsync_result: Optional[LipSyncResult] = None,
    ) -> str:
        """Call Remotion SSR to render final video."""
        await self._update_campaign_status(
            job.campaign_id,
            current_action="Rendering final video..."
        )
        
        # Build Remotion props
        props = {
            "audioUrl": f"data:audio/mp3;base64,{tts_result.audio_base64}",
            "subtitles": format_subtitles_for_remotion(tts_result.subtitles),
            "scenes": [
                {
                    "order": scene["order"],
                    "url": scene["url"],
                    "type": "video",  # or "image"
                }
                for scene in job.scenes
            ],
            "metadata": {
                "topic": job.topic,
                "niche": job.niche,
            }
        }
        
        # Add lip-synced avatar if available
        if lipsync_result and lipsync_result.status == "completed":
            props["avatarVideoUrl"] = lipsync_result.output_url
        
        # Call Remotion API (Remotion Rig)
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(
                f"{REMOTION_API_URL}/render",
                json={
                    "compositionId": "stoic-teaching", # Or configured ID
                    "inputProps": props,
                    "outputFormat": "mp4",
                    "width": 1080,
                    "height": 1920,
                    "fps": 30,
                },
            )
            response.raise_for_status()
            data = response.json()
            
            # Remotion Rig (api-server) returns sync result by default (unless async worker disabled)
            # Response: { success: true, render: { url: '/renders/...' } }
            
            if data.get("success"):
                render_data = data.get("render", {})
                
                # Check if it's a relative URL (local renders) or absolute (R2)
                url = render_data.get("url")
                if url and url.startswith("/"):
                    # Construct absolute URL if relative
                    return f"{REMOTION_API_URL}{url}"
                return url
            else:
                raise Exception(f"Render failed: {data.get('error')}")
        
        # If async polling becomes necessary later, implement here.
        # For now, api-server handles it or returns directly.
        pass
    
    async def _poll_remotion_render(
        self,
        render_id: str,
        max_wait: int = 600,
    ) -> str:
        """Poll Remotion for render completion."""
        elapsed = 0
        
        async with httpx.AsyncClient(timeout=30) as client:
            while elapsed < max_wait:
                response = await client.get(
                    f"{REMOTION_API_URL}/render/{render_id}/status"
                )
                response.raise_for_status()
                data = response.json()
                
                status = data.get("status")
                
                if status == "done":
                    return data.get("outputUrl") or data.get("outputFile")
                elif status == "error":
                    raise Exception(f"Render failed: {data.get('error')}")
                
                await asyncio.sleep(10)
                elapsed += 10
        
        raise Exception(f"Render timeout after {max_wait}s")
    
    async def _update_campaign_status(
        self,
        campaign_id: str,
        status: Optional[str] = None,
        current_action: Optional[str] = None,
        final_video_url: Optional[str] = None,
    ):
        """Update campaign status and action."""
        updates = ["updated_at = NOW()"]
        params = []
        
        if status:
            updates.append("status = %s")
            params.append(status)
        
        if current_action:
            updates.append("current_action = %s")
            params.append(current_action)
        
        if final_video_url:
            updates.append("final_video_url = %s")
            params.append(final_video_url)
        
        params.append(campaign_id)
        
        query = f"""
            UPDATE video_campaigns
            SET {', '.join(updates)}
            WHERE id = %s
        """
        await self.db.execute(query, params)
    
    @observe(name="renderer_process_job", run_type="chain")
    async def process_render_job(self, job: RenderJob):
        """Process a single render job."""
        print(f"[Renderer] Processing job for: {job.topic}")
        
        try:
            # Step 1: Generate TTS
            tts_result = await self.generate_tts(job)
            print(f"[Renderer] TTS generated: {tts_result.duration_seconds:.1f}s")
            
            # Step 2: Apply lip sync (if enabled)
            lipsync_result = None
            if job.lip_sync_enabled:
                # In production, use actual audio URL from R2
                audio_url = f"{R2_BUCKET_URL}/{job.campaign_id}/voiceover.mp3"
                lipsync_result = await self.apply_lipsync(job, audio_url)
                
                if lipsync_result and lipsync_result.status == "completed":
                    print(f"[Renderer] Lip sync applied: {lipsync_result.output_url}")
                else:
                    print(f"[Renderer] Lip sync failed, continuing without avatar")
            
            # Step 3: Render final video
            output_url = await self.render_video(job, tts_result, lipsync_result)
            print(f"[Renderer] Video rendered: {output_url}")
            
            # Step 4: Update campaign as complete
            await self._update_campaign_status(
                job.campaign_id,
                status="published",
                current_action="Complete! Video ready for publishing.",
                final_video_url=output_url,
            )
            
        except Exception as e:
            print(f"[Renderer] Error: {e}")
            await self._update_campaign_status(
                job.campaign_id,
                status="failed",
                current_action=f"Render error: {str(e)[:200]}"
            )
            raise
    
    async def run_once(self):
        """Run one polling cycle."""
        jobs = await self.poll_render_jobs()
        
        if not jobs:
            print("[Renderer] No render jobs found.")
            return
        
        print(f"[Renderer] Found {len(jobs)} render jobs.")
        
        for job in jobs:
            await self.process_render_job(job)
    
    async def run_daemon(self):
        """Run as a daemon, polling continuously."""
        print(f"[Renderer] Starting daemon (poll interval: {POLL_INTERVAL}s)")
        
        while True:
            try:
                await self.run_once()
            except Exception as e:
                print(f"[Renderer] Daemon error: {e}")
            
            await asyncio.sleep(POLL_INTERVAL)


# ============================================================
# Main Entry Point
# ============================================================

async def main():
    agent = RendererAgent()
    await agent.run_daemon()


if __name__ == "__main__":
    asyncio.run(main())
