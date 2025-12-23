"""
Director Agent - Campaign polling, script generation, and concept creation

The Director agent is the entry point of the pipeline:
1. Polls NocoDB for new campaigns (status='new')
2. Generates script concepts using Gemini 2.0
3. Stores drafts in script_drafts table
4. Updates campaign status

Usage:
    python director.py  # Run as daemon
    
    # Or import and use directly
    from director import DirectorAgent
    agent = DirectorAgent()
    await agent.run_once()
"""

import os
import asyncio
from datetime import datetime
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
from shared.llm import generate_script, ScriptDraft

# Environment
POLL_INTERVAL = int(os.getenv("DIRECTOR_POLL_INTERVAL", "30"))
SCRIPTS_PER_CAMPAIGN = int(os.getenv("SCRIPTS_PER_CAMPAIGN", "2"))


# ============================================================
# Models
# ============================================================

class Campaign(BaseModel):
    """A video campaign from NocoDB"""
    id: int
    topic: str
    niche: str
    status: str
    lip_sync_enabled: bool = False
    lip_sync_provider: Optional[str] = None
    avatar_image_url: Optional[str] = None
    style_notes: Optional[str] = None
    format: str = "short"  # short (60s) or long (8min)


# ============================================================
# Director Agent
# ============================================================

class DirectorAgent:
    """
    The Director agent handles:
    - Polling for new campaigns
    - Generating script concepts
    - Quality critique (inline)
    - Updating campaign status
    """
    
    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_db()
    
    @observe(name="director_poll_campaigns", run_type="chain")
    async def poll_new_campaigns(self) -> List[Campaign]:
        """Poll NocoDB for campaigns with status='new'."""
        query = """
            SELECT 
                id, topic, niche, status,
                lip_sync_enabled, lip_sync_provider,
                avatar_image_url, meta
            FROM video_campaigns
            WHERE status = 'new'
            ORDER BY created_at ASC
            LIMIT 5
        """
        rows = await self.db.fetch_all(query)
        
        return [
            Campaign(
                id=row["id"],
                topic=row["topic"],
                niche=row["niche"],
                status=row["status"],
                lip_sync_enabled=row.get("lip_sync_enabled", False),
                lip_sync_provider=row.get("lip_sync_provider"),
                avatar_image_url=row.get("avatar_image_url"),
                style_notes=row.get("meta", {}).get("style_notes"),
                format=row.get("meta", {}).get("format", "short"),
            )
            for row in rows
        ]
    
    @observe(name="director_generate_scripts", run_type="chain")
    async def generate_scripts_for_campaign(
        self,
        campaign: Campaign,
        num_drafts: int = 2,
    ) -> List[ScriptDraft]:
        """Generate script drafts for a campaign."""
        # Update status
        await self._update_campaign_status(
            campaign.id,
            status="in_scripting",
            current_action=f"Generating {num_drafts} script concepts..."
        )
        
        drafts = []
        for i in range(num_drafts):
            # Add variation to each draft
            style_variation = None
            if i == 0:
                style_variation = "Focus on emotional storytelling and personal connection."
            elif i == 1:
                style_variation = "Focus on surprising facts and pattern interrupts."
            
            combined_style = f"{campaign.style_notes or ''} {style_variation}".strip()
            
            draft = generate_script(
                topic=campaign.topic,
                niche=campaign.niche,
                format=campaign.format,
                style_notes=combined_style if combined_style else None,
            )
            drafts.append(draft)
        
        return drafts
    
    @observe(name="director_save_drafts", run_type="chain")
    async def save_script_drafts(
        self,
        campaign: Campaign,
        drafts: List[ScriptDraft],
    ) -> List[str]:
        """Save script drafts to database."""
        draft_ids = []
        
        for i, draft in enumerate(drafts):
            query = """
                INSERT INTO script_drafts (
                    campaign_id, version,
                    hook_line, body_text, call_to_action,
                    approved, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, FALSE, NOW()
                )
                RETURNING id
            """
            result = await self.db.fetch_one(query, [
                campaign.id,
                i + 1,
                draft.hook_line,
                draft.body_text,
                draft.call_to_action,
            ])
            draft_ids.append(result["id"])
        
        # Update campaign status
        await self._update_campaign_status(
            campaign.id,
            status="in_scripting",
            current_action=f"Generated {len(drafts)} scripts. Awaiting approval."
        )
        
        return draft_ids
    
    async def _update_campaign_status(
        self,
        campaign_id: int,
        status: Optional[str] = None,
        current_action: Optional[str] = None,
    ):
        """Update campaign status and action."""
        updates = []
        params = []
        
        if status:
            updates.append("status = %s")
            params.append(status)
        
        if current_action:
            updates.append("current_action = %s")
            params.append(current_action)
        
        updates.append("updated_at = NOW()")
        params.append(campaign_id)
        
        query = f"""
            UPDATE video_campaigns
            SET {', '.join(updates)}
            WHERE id = %s
        """
        await self.db.execute(query, params)
    
    @observe(name="director_process_campaign", run_type="chain")
    async def process_campaign(self, campaign: Campaign):
        """Process a single campaign end-to-end."""
        print(f"[Director] Processing campaign: {campaign.topic}")
        
        try:
            # Generate scripts
            drafts = await self.generate_scripts_for_campaign(
                campaign,
                num_drafts=SCRIPTS_PER_CAMPAIGN,
            )
            
            # Save to database
            draft_ids = await self.save_script_drafts(campaign, drafts)
            
            print(f"[Director] Created {len(draft_ids)} drafts for {campaign.topic}")
            
        except Exception as e:
            print(f"[Director] Error processing {campaign.topic}: {e}")
            await self._update_campaign_status(
                campaign.id,
                status="failed",
                current_action=f"Error: {str(e)[:200]}"
            )
            raise
    
    async def run_once(self):
        """Run one polling cycle."""
        campaigns = await self.poll_new_campaigns()
        
        if not campaigns:
            print("[Director] No new campaigns found.")
            return
        
        print(f"[Director] Found {len(campaigns)} new campaigns.")
        
        for campaign in campaigns:
            await self.process_campaign(campaign)
    
    async def run_daemon(self):
        """Run as a daemon, polling continuously."""
        print(f"[Director] Starting daemon (poll interval: {POLL_INTERVAL}s)")
        
        while True:
            try:
                await self.run_once()
            except Exception as e:
                print(f"[Director] Daemon error: {e}")
            
            await asyncio.sleep(POLL_INTERVAL)


# ============================================================
# Main Entry Point
# ============================================================

async def main():
    agent = DirectorAgent()
    await agent.run_daemon()


if __name__ == "__main__":
    asyncio.run(main())
