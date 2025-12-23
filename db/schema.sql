-- Viral Video Agents - Database Schema
-- Run this on the PostgreSQL database accessible by agents

-- Video Campaigns table (main)
CREATE TABLE IF NOT EXISTS video_campaigns (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    niche VARCHAR(100) NOT NULL DEFAULT 'Stoic Philosophy',
    topic TEXT NOT NULL,
    format VARCHAR(20) DEFAULT 'short',  -- 'short' (60s) or 'long' (8min)
    status VARCHAR(50) DEFAULT 'new',
    -- Status flow: new -> scripting -> scripted -> visuals -> visual_ready -> rendering -> completed
    
    -- Script info
    approved_script_id INTEGER,
    
    -- Generation settings
    priority VARCHAR(20) DEFAULT 'balanced',  -- 'quality', 'balanced', 'cost'
    lip_sync_enabled BOOLEAN DEFAULT false,
    lip_sync_provider VARCHAR(50) DEFAULT 'comfyui',  -- 'comfyui' (free) or 'sync_labs'
    avatar_image_url TEXT,
    
    -- Agent visibility
    current_action TEXT,
    debug_url TEXT,  -- Langfuse trace URL
    
    -- Results
    final_video_url TEXT,
    error_log TEXT,
    estimated_cost DECIMAL(10, 4),
    actual_cost DECIMAL(10, 4),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Script Drafts table
CREATE TABLE IF NOT EXISTS script_drafts (
    id SERIAL PRIMARY KEY,
    campaign_id INTEGER REFERENCES video_campaigns(id) ON DELETE CASCADE,
    version INTEGER DEFAULT 1,
    
    hook_line TEXT NOT NULL,
    body_text TEXT NOT NULL,
    call_to_action TEXT,
    estimated_duration_seconds INTEGER,
    key_points JSONB,  -- Array of key points
    
    -- Critique
    approved BOOLEAN DEFAULT false,
    critique_score INTEGER,
    critique_notes TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW()
);

-- Visual Scenes table
CREATE TABLE IF NOT EXISTS visual_scenes (
    id SERIAL PRIMARY KEY,
    campaign_id INTEGER REFERENCES video_campaigns(id) ON DELETE CASCADE,
    scene_order INTEGER NOT NULL,
    
    -- Timing
    timestamp_start FLOAT,
    timestamp_end FLOAT,
    
    -- Generation
    visual_prompt TEXT NOT NULL,
    scene_type VARCHAR(50),  -- 'opening', 'nature', 'architecture', 'transition', 'closing'
    recommended_model VARCHAR(50),  -- 'veo3_quality', 'veo3_fast', 'kling', 'flux_pro'
    
    -- Results
    status VARCHAR(50) DEFAULT 'pending',  -- 'pending', 'generating', 'completed', 'failed'
    asset_url TEXT,
    asset_type VARCHAR(20),  -- 'image' or 'video'
    generation_cost DECIMAL(10, 4),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW()
);

-- Audio Assets table
CREATE TABLE IF NOT EXISTS audio_assets (
    id SERIAL PRIMARY KEY,
    campaign_id INTEGER REFERENCES video_campaigns(id) ON DELETE CASCADE,
    asset_type VARCHAR(50) NOT NULL,  -- 'voiceover', 'background_music', 'sfx'
    
    -- Voiceover info
    voice_id VARCHAR(100),
    voice_name VARCHAR(100),
    
    -- Content
    text_content TEXT,
    duration_seconds FLOAT,
    
    -- Timestamps (for subtitles)
    word_timestamps JSONB,  -- Array of {word, start, end}
    
    -- Storage
    audio_url TEXT,
    generation_cost DECIMAL(10, 4),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW()
);

-- Enable updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_video_campaigns_updated_at
    BEFORE UPDATE ON video_campaigns
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_campaigns_status ON video_campaigns(status);
CREATE INDEX IF NOT EXISTS idx_scenes_campaign ON visual_scenes(campaign_id);
CREATE INDEX IF NOT EXISTS idx_drafts_campaign ON script_drafts(campaign_id);

-- Insert a test campaign
INSERT INTO video_campaigns (title, topic, niche, format, priority)
VALUES (
    'Test Campaign - Stoic Anxiety',
    'Why you feel anxious according to Marcus Aurelius and how Stoic philosophy can help you find peace',
    'Stoic Philosophy',
    'short',
    'balanced'
) ON CONFLICT DO NOTHING;

SELECT 'Schema created successfully!' as result;
