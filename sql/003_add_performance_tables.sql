-- Migration 003: Performance Tracking & Intelligence Layer
-- Run after 002_add_lipsync_columns.sql

-- Trend tracking for content intelligence
CREATE TABLE trending_topics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    platform TEXT NOT NULL, -- youtube, tiktok, instagram
    niche TEXT NOT NULL,
    topic TEXT NOT NULL,
    trend_score FLOAT,
    velocity FLOAT, -- how fast it's growing
    detected_at TIMESTAMPTZ DEFAULT now(),
    expires_at TIMESTAMPTZ,
    source_data JSONB DEFAULT '{}'::jsonb,
    UNIQUE(platform, niche, topic, detected_at::date)
);

-- Competitor analysis
CREATE TABLE competitor_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    platform TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    video_id TEXT UNIQUE NOT NULL,
    title TEXT,
    description TEXT,
    published_at TIMESTAMPTZ,
    view_count BIGINT,
    like_count BIGINT,
    comment_count BIGINT,
    duration_seconds INT,
    thumbnail_url TEXT,
    tags TEXT[],
    hook_analysis JSONB, -- AI analysis of first 10 seconds
    scraped_at TIMESTAMPTZ DEFAULT now()
);

-- Our video performance tracking
CREATE TABLE video_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES video_campaigns(id) ON DELETE CASCADE,
    platform TEXT NOT NULL,
    external_video_id TEXT NOT NULL,

    -- Snapshot metrics (updated periodically)
    snapshot_at TIMESTAMPTZ DEFAULT now(),
    view_count BIGINT DEFAULT 0,
    like_count BIGINT DEFAULT 0,
    dislike_count BIGINT DEFAULT 0,
    comment_count BIGINT DEFAULT 0,
    share_count BIGINT DEFAULT 0,

    -- Engagement metrics
    avg_view_duration_seconds FLOAT,
    avg_view_percentage FLOAT,
    click_through_rate FLOAT,

    -- Retention curve (percentage watching at each point)
    retention_curve JSONB, -- {"0": 100, "25": 80, "50": 60, "75": 40, "100": 20}

    -- Traffic sources
    traffic_sources JSONB, -- {"search": 30, "suggested": 50, "external": 20}

    UNIQUE(campaign_id, platform, snapshot_at)
);

-- A/B test tracking for thumbnails and titles
CREATE TABLE ab_tests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES video_campaigns(id) ON DELETE CASCADE,
    test_type TEXT NOT NULL, -- thumbnail, title, description
    variant_a JSONB NOT NULL,
    variant_b JSONB NOT NULL,
    started_at TIMESTAMPTZ DEFAULT now(),
    ended_at TIMESTAMPTZ,
    winner TEXT, -- 'a', 'b', or null if ongoing
    confidence FLOAT, -- statistical significance
    metrics JSONB DEFAULT '{}'::jsonb
);

-- Content generation cost tracking
CREATE TABLE generation_costs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES video_campaigns(id) ON DELETE CASCADE,
    service TEXT NOT NULL, -- claude, kie, elevenlabs, suno, remotion
    operation TEXT NOT NULL, -- script_gen, image_gen, voice_gen, etc
    tokens_used INT,
    api_cost_usd DECIMAL(10,4),
    compute_seconds FLOAT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Strategy recommendations (what to make next)
CREATE TABLE content_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    niche TEXT NOT NULL,
    recommended_topic TEXT NOT NULL,
    reasoning TEXT,
    priority_score FLOAT,
    based_on_trends UUID[], -- references trending_topics
    based_on_performance UUID[], -- references high-performing videos
    created_at TIMESTAMPTZ DEFAULT now(),
    expires_at TIMESTAMPTZ,
    was_used BOOLEAN DEFAULT FALSE
);

-- Indexes for efficient queries
CREATE INDEX idx_trending_platform_niche ON trending_topics(platform, niche);
CREATE INDEX idx_trending_score ON trending_topics(trend_score DESC);
CREATE INDEX idx_competitor_channel ON competitor_content(channel_id);
CREATE INDEX idx_competitor_views ON competitor_content(view_count DESC);
CREATE INDEX idx_performance_campaign ON video_performance(campaign_id);
CREATE INDEX idx_performance_time ON video_performance(snapshot_at DESC);
CREATE INDEX idx_costs_campaign ON generation_costs(campaign_id);
CREATE INDEX idx_recommendations_priority ON content_recommendations(priority_score DESC);

-- Time-series partitioning for video_performance (if using TimescaleDB)
-- SELECT create_hypertable('video_performance', 'snapshot_at', if_not_exists => TRUE);

-- View for campaign ROI calculation
CREATE VIEW campaign_roi AS
SELECT
    c.id as campaign_id,
    c.topic,
    c.niche,
    c.status,
    c.created_at,
    COALESCE(SUM(g.api_cost_usd), 0) as total_cost,
    MAX(p.view_count) as total_views,
    MAX(p.like_count) as total_likes,
    CASE
        WHEN COALESCE(SUM(g.api_cost_usd), 0) > 0
        THEN MAX(p.view_count) / SUM(g.api_cost_usd)
        ELSE 0
    END as views_per_dollar
FROM video_campaigns c
LEFT JOIN generation_costs g ON g.campaign_id = c.id
LEFT JOIN video_performance p ON p.campaign_id = c.id
GROUP BY c.id, c.topic, c.niche, c.status, c.created_at;
