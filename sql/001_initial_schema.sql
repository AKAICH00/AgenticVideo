-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enum for Campaign Status
CREATE TYPE campaign_status AS ENUM (
    'new', 
    'in_scripting', 
    'script_approved', 
    'generating_visuals', 
    'rendering', 
    'published', 
    'failed'
);

-- Video Campaigns Table (The Main Truth)
CREATE TABLE video_campaigns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    topic TEXT NOT NULL,
    niche TEXT NOT NULL,
    status campaign_status DEFAULT 'new',
    final_video_url TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    error_log TEXT,
    meta JSONB DEFAULT '{}'::jsonb -- For ad-hoc data
);

-- Script Drafts Table
CREATE TABLE script_drafts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES video_campaigns(id) ON DELETE CASCADE,
    script_version INT,
    hook_line TEXT,
    body_text TEXT,
    call_to_action TEXT,
    is_approved BOOLEAN DEFAULT FALSE,
    critic_notes TEXT,
    audio_mood TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Visual Scenes Table (The Storyboard)
CREATE TABLE visual_scenes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    script_id UUID REFERENCES script_drafts(id) ON DELETE CASCADE,
    scene_order INT,
    visual_prompt TEXT,
    reference_video_url TEXT, -- For Wan 2.6 Trend Mimicry
    generated_asset_url TEXT,
    is_regeneration_requested BOOLEAN DEFAULT FALSE,
    generation_status TEXT DEFAULT 'pending', -- pending, generating, completed, failed
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Updates Trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_video_campaigns_modtime
    BEFORE UPDATE ON video_campaigns
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
