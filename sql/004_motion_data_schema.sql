-- Motion data and step-level tracking enhancement
-- Supports the motion transfer pipeline and detailed progress visibility

-- Enhanced campaign status with step-level granularity
CREATE TYPE generation_step AS ENUM (
    'pending',
    'analyzing_reference',
    'extracting_poses',
    'extracting_camera',
    'detecting_transitions',
    'generating_script',
    'generating_storyboard',
    'generating_visuals',
    'applying_motion',
    'compositing',
    'rendering_long_form',
    'clipping_short_form',
    'uploading',
    'complete',
    'failed'
);

-- Reference videos for motion extraction
CREATE TABLE reference_videos (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES video_campaigns(id) ON DELETE CASCADE,
    source_url TEXT NOT NULL,
    source_platform TEXT, -- 'tiktok', 'youtube', 'instagram', 'upload'
    local_path TEXT, -- Path after download
    duration_seconds FLOAT,
    fps FLOAT,
    resolution_width INT,
    resolution_height INT,
    analysis_status TEXT DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Extracted motion data from reference videos
CREATE TABLE motion_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    reference_video_id UUID REFERENCES reference_videos(id) ON DELETE CASCADE,

    -- Pose data (from DWPose/OpenPose)
    pose_keypoints JSONB, -- [{frame: 0, keypoints: [...], confidence: 0.95}, ...]
    pose_format TEXT DEFAULT 'dwpose', -- 'dwpose', 'openpose', 'smpl'

    -- Camera motion (from CoTracker)
    camera_motion JSONB, -- [{frame: 0, pan: 0.1, tilt: -0.05, zoom: 1.0, shake: 0.02}, ...]
    camera_trajectory_path TEXT, -- Path to trajectory file

    -- Transition/cut timing (from PySceneDetect)
    transitions JSONB, -- [{frame: 120, type: 'cut', duration_frames: 3}, ...]
    beat_sync JSONB, -- [{frame: 30, beat: 1, bpm: 128}, ...]

    -- Style embedding (from CLIP)
    style_embedding BYTEA, -- CLIP embedding for style matching
    style_tags JSONB, -- ['energetic', 'fast-paced', 'dance', ...]

    -- Processing metadata
    extraction_model_versions JSONB, -- {'dwpose': '1.0', 'cotracker': '2.0', ...}
    processing_time_seconds FLOAT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Step-level progress tracking
CREATE TABLE generation_steps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES video_campaigns(id) ON DELETE CASCADE,
    step_name generation_step NOT NULL,
    step_order INT NOT NULL,
    status TEXT DEFAULT 'pending', -- 'pending', 'in_progress', 'completed', 'failed', 'skipped'

    -- Progress within step
    progress_percent INT DEFAULT 0,
    progress_message TEXT,

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    estimated_remaining_seconds INT,

    -- Results
    output_data JSONB, -- Step-specific output
    error_message TEXT,
    retry_count INT DEFAULT 0,

    -- Agent decision log
    agent_reasoning TEXT, -- Why the agent made specific choices
    confidence_score FLOAT, -- Agent's confidence in this step (0-1)
    alternatives_considered JSONB, -- Other options the agent evaluated

    created_at TIMESTAMPTZ DEFAULT now()
);

-- Video generation jobs (for API tracking)
CREATE TABLE video_generation_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scene_id UUID REFERENCES visual_scenes(id) ON DELETE CASCADE,

    -- Provider info
    provider TEXT NOT NULL, -- 'kie', 'fal', 'runway', 'sora', 'kling', 'wan_local'
    model TEXT NOT NULL, -- 'runway-gen4.5', 'sora-2', 'kling-2.5', 'wan-2.1'
    quality_tier TEXT DEFAULT 'premium', -- 'premium', 'bulk'

    -- Request
    prompt TEXT NOT NULL,
    negative_prompt TEXT,
    duration_seconds INT,
    aspect_ratio TEXT DEFAULT '16:9',
    reference_image_url TEXT,
    motion_data_id UUID REFERENCES motion_data(id),

    -- API response
    external_job_id TEXT, -- Provider's job ID
    status TEXT DEFAULT 'pending', -- 'pending', 'queued', 'processing', 'completed', 'failed'
    result_url TEXT,

    -- Cost tracking
    estimated_cost_usd DECIMAL(10, 4),
    actual_cost_usd DECIMAL(10, 4),

    -- Timing
    submitted_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    processing_time_seconds FLOAT,

    -- Error handling
    error_code TEXT,
    error_message TEXT,
    retry_count INT DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT now()
);

-- Short-form clips generated from long-form
CREATE TABLE short_form_clips (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES video_campaigns(id) ON DELETE CASCADE,
    source_video_url TEXT NOT NULL, -- Long-form video

    -- Clip timing
    start_time_seconds FLOAT NOT NULL,
    end_time_seconds FLOAT NOT NULL,
    duration_seconds FLOAT GENERATED ALWAYS AS (end_time_seconds - start_time_seconds) STORED,

    -- Detection scores
    moment_score FLOAT, -- How "viral-worthy" this moment is
    audio_peak_detected BOOLEAN,
    visual_peak_detected BOOLEAN,
    hook_strength FLOAT, -- How strong the opening is

    -- Platform targeting
    target_platform TEXT, -- 'tiktok', 'shorts', 'reels'
    aspect_ratio TEXT DEFAULT '9:16',

    -- Processing
    processing_status TEXT DEFAULT 'pending',
    clip_url TEXT,
    thumbnail_url TEXT,

    -- Optimization
    optimized_hook_text TEXT, -- Generated hook for this clip
    suggested_hashtags JSONB,
    suggested_audio_id TEXT, -- Trending audio suggestion

    created_at TIMESTAMPTZ DEFAULT now()
);

-- Circuit breaker state for APIs
CREATE TABLE circuit_breaker_state (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name TEXT NOT NULL UNIQUE, -- 'runway', 'sora', 'kling', 'elevenlabs'
    state TEXT DEFAULT 'closed', -- 'closed', 'open', 'half_open'

    -- Failure tracking
    failure_count INT DEFAULT 0,
    success_count INT DEFAULT 0,
    last_failure_at TIMESTAMPTZ,
    last_success_at TIMESTAMPTZ,

    -- Configuration
    failure_threshold INT DEFAULT 5,
    recovery_timeout_seconds INT DEFAULT 30,
    half_open_max_calls INT DEFAULT 3,

    -- State change tracking
    state_changed_at TIMESTAMPTZ DEFAULT now(),

    CONSTRAINT valid_state CHECK (state IN ('closed', 'open', 'half_open'))
);

-- Indexes for performance
CREATE INDEX idx_motion_data_reference ON motion_data(reference_video_id);
CREATE INDEX idx_generation_steps_campaign ON generation_steps(campaign_id, step_order);
CREATE INDEX idx_generation_steps_status ON generation_steps(status) WHERE status = 'in_progress';
CREATE INDEX idx_video_jobs_status ON video_generation_jobs(status) WHERE status IN ('pending', 'queued', 'processing');
CREATE INDEX idx_short_form_campaign ON short_form_clips(campaign_id);
CREATE INDEX idx_circuit_breaker_service ON circuit_breaker_state(service_name);

-- Add motion_data reference to visual_scenes
ALTER TABLE visual_scenes ADD COLUMN IF NOT EXISTS motion_data_id UUID REFERENCES motion_data(id);
ALTER TABLE visual_scenes ADD COLUMN IF NOT EXISTS generation_job_id UUID REFERENCES video_generation_jobs(id);

-- Add step tracking to campaigns
ALTER TABLE video_campaigns ADD COLUMN IF NOT EXISTS current_step generation_step DEFAULT 'pending';
ALTER TABLE video_campaigns ADD COLUMN IF NOT EXISTS quality_tier TEXT DEFAULT 'premium';
ALTER TABLE video_campaigns ADD COLUMN IF NOT EXISTS reference_video_id UUID REFERENCES reference_videos(id);
