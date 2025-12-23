"""
Visualist Agent - Visual scene generation and asset creation

The Visualist agent handles:
1. Polls for approved scripts
2. Generates visual prompts from script
3. Routes to kie.ai or ComfyUI based on priority
4. Stores generated assets in R2
5. Updates visual_scenes table

Usage:
    python visualist.py  # Run as daemon
    
    # Or import and use directly
    from visualist import VisualistAgent
    agent = VisualistAgent()
    await agent.run_once()
"""

import os
import asyncio
from typing import Optional, List
from pydantic import BaseModel
# Langfuse OSS observability (optional)
try:
    from langfuse.decorators import observe
except ImportError:
    def observe(*args, **kwargs):
        def decorator(func): return func
        return decorator

from shared.db import get_db, Database
from shared.llm import generate_visual_prompts, VisualScene as LLMVisualScene
from shared.visual_router import (
    generate_scene,
    generate_all_scenes,
    VisualScene as RouterVisualScene,
    SceneResult,
)

# Environment
POLL_INTERVAL = int(os.getenv("VISUALIST_POLL_INTERVAL", "30"))
GENERATION_PRIORITY = os.getenv("VISUALIST_PRIORITY", "balanced")  # quality, balanced, cost


# ============================================================
# Models
# ============================================================

class ApprovedScript(BaseModel):
    """An approved script ready for visual generation"""
    id: int
    campaign_id: int
    hook_line: str
    body_text: str
    call_to_action: str
    niche: str
    format: str = "short"


# ============================================================
# Visualist Agent
# ============================================================

class VisualistAgent:
    """
    The Visualist agent handles:
    - Polling for approved scripts
    - Generating visual prompts
    - Creating visuals via kie.ai/ComfyUI
    - Storing assets in R2
    """
    
    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_db()
    
    @observe(name="visualist_poll_scripts", run_type="chain")
    async def poll_approved_scripts(self) -> List[ApprovedScript]:
        """Poll for approved scripts that need visuals."""
        query = """
            SELECT 
                s.id, s.campaign_id, s.hook_line, s.body_text, s.call_to_action,
                c.niche, c.meta
            FROM script_drafts s
            JOIN video_campaigns c ON s.campaign_id = c.id
            WHERE s.approved = TRUE
            AND c.status = 'script_approved'
            ORDER BY s.created_at ASC
            LIMIT 3
        """
        rows = await self.db.fetch_all(query)
        
        return [
            ApprovedScript(
                id=row["id"],
                campaign_id=row["campaign_id"],
                hook_line=row["hook_line"],
                body_text=row["body_text"],
                call_to_action=row["call_to_action"],
                niche=row["niche"],
                format=row.get("meta", {}).get("format", "short"),
            )
            for row in rows
        ]
    
    @observe(name="visualist_generate_prompts", run_type="chain")
    async def generate_scene_prompts(
        self,
        script: ApprovedScript,
    ) -> List[LLMVisualScene]:
        """Generate visual prompts from script."""
        # Update campaign status
        await self._update_campaign_status(
            script.campaign_id,
            status="generating_visuals",
            current_action="Generating visual prompts..."
        )
        
        # Combine script text
        full_script = f"""
HOOK: {script.hook_line}

{script.body_text}

CTA: {script.call_to_action}
"""
        
        # Duration based on format
        duration = 60 if script.format == "short" else 480
        
        # Generate prompts
        result = generate_visual_prompts(
            script_text=full_script,
            niche=script.niche,
            duration_seconds=duration,
        )
        
        return result.scenes
    
    @observe(name="visualist_create_visuals", run_type="chain")
    async def create_visuals(
        self,
        script: ApprovedScript,
        scenes: List[LLMVisualScene],
        priority: str = "balanced",
    ) -> List[SceneResult]:
        """Generate all visuals for the scenes."""
        await self._update_campaign_status(
            script.campaign_id,
            current_action=f"Generating {len(scenes)} visual scenes..."
        )
        
        # Convert to router format
        router_scenes = [
            RouterVisualScene(
                scene_order=scene.scene_order,
                visual_prompt=scene.visual_prompt,
                scene_type=scene.scene_type,
                duration_seconds=scene.timestamp_end - scene.timestamp_start,
            )
            for scene in scenes
        ]
        
        # Generate all scenes
        results = await generate_all_scenes(router_scenes, priority=priority)
        
        return results
    
    @observe(name="visualist_save_scenes", run_type="chain")
    async def save_visual_scenes(
        self,
        script: ApprovedScript,
        llm_scenes: List[LLMVisualScene],
        results: List[SceneResult],
    ) -> List[str]:
        """Save generated scenes to database."""
        scene_ids = []
        
        for llm_scene, result in zip(llm_scenes, results):
            query = """
                INSERT INTO visual_scenes (
                    script_id, scene_order, visual_prompt,
                    render_backend, generated_asset_url,
                    generation_status, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, NOW()
                )
                RETURNING id
            """
            
            status = "completed" if result.status == "completed" else "failed"
            
            db_result = await self.db.fetch_one(query, [
                script.id,
                llm_scene.scene_order,
                llm_scene.visual_prompt,
                f"{result.backend}/{result.model}",
                result.output_url,
                status,
            ])
            scene_ids.append(db_result["id"])
        
        # Count successes
        successes = sum(1 for r in results if r.status == "completed")
        
        await self._update_campaign_status(
            script.campaign_id,
            current_action=f"Generated {successes}/{len(results)} scenes."
        )
        
        return scene_ids
    
    async def _update_campaign_status(
        self,
        campaign_id: int,
        status: Optional[str] = None,
        current_action: Optional[str] = None,
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
        
        params.append(campaign_id)
        
        query = f"""
            UPDATE video_campaigns
            SET {', '.join(updates)}
            WHERE id = %s
        """
        await self.db.execute(query, params)
    
    @observe(name="visualist_process_script", run_type="chain")
    async def process_script(self, script: ApprovedScript):
        """Process a single approved script."""
        print(f"[Visualist] Processing script for campaign: {script.campaign_id}")
        
        try:
            # Generate prompts
            scenes = await self.generate_scene_prompts(script)
            print(f"[Visualist] Generated {len(scenes)} scene prompts")
            
            # Create visuals
            results = await self.create_visuals(
                script,
                scenes,
                priority=GENERATION_PRIORITY,
            )
            
            # Save to database
            scene_ids = await self.save_visual_scenes(script, scenes, results)
            print(f"[Visualist] Saved {len(scene_ids)} visual scenes")
            
            # Check if all completed
            all_complete = all(r.status == "completed" for r in results)
            
            if all_complete:
                await self._update_campaign_status(
                    script.campaign_id,
                    status="rendering",
                    current_action="All visuals ready. Queued for rendering."
                )
            else:
                failed = sum(1 for r in results if r.status != "completed")
                await self._update_campaign_status(
                    script.campaign_id,
                    current_action=f"{failed} scenes failed. Manual review needed."
                )
            
        except Exception as e:
            print(f"[Visualist] Error processing script: {e}")
            await self._update_campaign_status(
                script.campaign_id,
                status="failed",
                current_action=f"Visualist error: {str(e)[:200]}"
            )
            raise
    
    async def run_once(self):
        """Run one polling cycle."""
        scripts = await self.poll_approved_scripts()
        
        if not scripts:
            print("[Visualist] No approved scripts found.")
            return
        
        print(f"[Visualist] Found {len(scripts)} approved scripts.")
        
        for script in scripts:
            await self.process_script(script)
    
    async def run_daemon(self):
        """Run as a daemon, polling continuously."""
        print(f"[Visualist] Starting daemon (poll interval: {POLL_INTERVAL}s)")
        
        while True:
            try:
                await self.run_once()
            except Exception as e:
                print(f"[Visualist] Daemon error: {e}")
            
            await asyncio.sleep(POLL_INTERVAL)


# ============================================================
# Main Entry Point
# ============================================================

async def main():
    agent = VisualistAgent()
    await agent.run_daemon()


if __name__ == "__main__":
    asyncio.run(main())
