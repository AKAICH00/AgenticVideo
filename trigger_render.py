import asyncio
from agents.shared.db import get_db

async def create_renderer_job():
    db = get_db()
    
    # Clean up old test jobs
    await db.execute("DELETE FROM video_campaigns WHERE topic = 'STOIC_TEST_RENDER'")
    
    # 1. Create Campaign
    await db.execute("""
        INSERT INTO video_campaigns (topic, niche, status, lip_sync_enabled)
        VALUES ('STOIC_TEST_RENDER', 'stoicism', 'rendering', true)
        RETURNING id
    """)
    row = await db.fetch_one("SELECT id FROM video_campaigns WHERE topic = 'STOIC_TEST_RENDER'")
    campaign_id = row['id']
    print(f"Created campaign {campaign_id}")
    
    # 2. Create Script Draft (Approved)
    await db.execute("""
        INSERT INTO script_drafts (campaign_id, hook_line, body_text, call_to_action, approved)
        VALUES (%s, 'Life is short.', 'Don''t waste it on nonsense.', 'Subscribe for wisdom.', true)
        RETURNING id
    """, [campaign_id])
    row = await db.fetch_one("SELECT id FROM script_drafts WHERE campaign_id = %s", [campaign_id])
    script_id = row['id']
    print(f"Created script {script_id}")
    
    # 3. Create Visual Scenes (Completed)
    scenes = [
        (1, "https://example.com/asset1.jpg", "A stoic statue"),
        (2, "https://example.com/asset2.jpg", "A sunset over rome")
    ]
    
    for order, url, prompt in scenes:
        await db.execute("""
            INSERT INTO visual_scenes (script_id, scene_order, generated_asset_url, visual_prompt, generation_status)
            VALUES (%s, %s, %s, %s, 'completed')
        """, [script_id, order, url, prompt])
        print(f"Created scene {order}")

    print("Job ready for pickup by RendererAgent")

if __name__ == "__main__":
    asyncio.run(create_renderer_job())
