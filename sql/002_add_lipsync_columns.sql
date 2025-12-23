-- Migration 002: Add Lip Sync and Visibility Columns
-- Run after 001_initial_schema.sql

-- Add lip sync provider enum
CREATE TYPE lip_sync_provider AS ENUM (
    'none', 'sync_labs', 'comfyui_wav2lip'
);

-- Add columns to video_campaigns
ALTER TABLE video_campaigns 
    ADD COLUMN lip_sync_enabled BOOLEAN DEFAULT FALSE,
    ADD COLUMN lip_sync_provider lip_sync_provider DEFAULT 'none',
    ADD COLUMN avatar_image_url TEXT,
    ADD COLUMN current_action TEXT,
    ADD COLUMN debug_url TEXT;

-- Add columns to script_drafts
ALTER TABLE script_drafts
    ADD COLUMN audio_url TEXT,
    ADD COLUMN subtitles JSONB;

-- Add columns to visual_scenes
ALTER TABLE visual_scenes
    ADD COLUMN reference_video_url TEXT,
    ADD COLUMN render_backend TEXT DEFAULT 'kie.ai';

-- Create index for faster polling
CREATE INDEX idx_campaigns_status ON video_campaigns(status);
CREATE INDEX idx_scripts_approved ON script_drafts(campaign_id, is_approved);
CREATE INDEX idx_scenes_status ON visual_scenes(script_id, generation_status);
