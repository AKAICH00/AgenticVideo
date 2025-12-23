-- ============================================================================
-- Migration: V2 Pipeline Isolation
-- Created: 2025-12-22
-- Purpose: Complete isolation between OLD polling daemons and NEW V2 orchestrator
--
-- CRITICAL: This migration ensures ZERO conflict between systems:
-- - OLD agents poll for status='new' (unchanged)
-- - V2 orchestrator uses v2_* statuses (completely separate)
-- ============================================================================

-- ============================================================================
-- SECTION 1: V2 Status Values
-- ============================================================================

-- Add V2-specific status values to the campaigns table
-- These are COMPLETELY SEPARATE from OLD statuses
ALTER TABLE video_campaigns
DROP CONSTRAINT IF EXISTS video_campaigns_status_check;

ALTER TABLE video_campaigns
ADD CONSTRAINT video_campaigns_status_check CHECK (
    status IN (
        -- OLD Pipeline Statuses (unchanged, DO NOT MODIFY)
        'new',
        'in_scripting',
        'script_approved',
        'generating_visuals',
        'rendering',

        -- V2 Pipeline Statuses (NEW - completely isolated)
        'v2_pending',           -- Created via V2 API, waiting to be picked up
        'v2_planning',          -- PlannerNode running
        'v2_scripting',         -- ScriptNode running
        'v2_storyboarding',     -- StoryboardNode running
        'v2_motion',            -- MotionNode running
        'v2_visual',            -- VisualNode running
        'v2_quality',           -- QualityNode running
        'v2_composing',         -- ComposeNode running
        'v2_repurposing',       -- RepurposeNode running

        -- Shared Final States (both systems converge here)
        'published',
        'failed'
    )
);

COMMENT ON CONSTRAINT video_campaigns_status_check ON video_campaigns IS
    'Status values for both OLD (new, in_scripting, etc.) and V2 (v2_pending, v2_planning, etc.) pipelines';

-- ============================================================================
-- SECTION 2: Isolation Indexes
-- ============================================================================

-- Index for OLD agents (polls status='new')
-- This already exists but we document it for clarity
CREATE INDEX IF NOT EXISTS idx_campaigns_status_new
ON video_campaigns(status)
WHERE status = 'new';

COMMENT ON INDEX idx_campaigns_status_new IS
    'Used by OLD agents (director.py) to poll for new campaigns';

-- Index for V2 orchestrator (polls status='v2_pending')
CREATE INDEX IF NOT EXISTS idx_campaigns_v2_pending
ON video_campaigns(status)
WHERE status = 'v2_pending';

COMMENT ON INDEX idx_campaigns_v2_pending IS
    'Used by V2 orchestrator to poll for pending campaigns - ISOLATED from OLD';

-- Index for V2 in-progress campaigns
CREATE INDEX IF NOT EXISTS idx_campaigns_v2_processing
ON video_campaigns(status)
WHERE status LIKE 'v2_%' AND status != 'v2_pending';

COMMENT ON INDEX idx_campaigns_v2_processing IS
    'Used to find V2 campaigns that are currently processing';

-- Combined index for monitoring
CREATE INDEX IF NOT EXISTS idx_campaigns_processor_status
ON video_campaigns(processor, status);

-- ============================================================================
-- SECTION 3: V2 Tracking Columns (if not already added)
-- ============================================================================

-- Ensure processor column exists with proper default
ALTER TABLE video_campaigns
ADD COLUMN IF NOT EXISTS processor VARCHAR(10) DEFAULT NULL;

COMMENT ON COLUMN video_campaigns.processor IS
    'Which system processed: NULL=legacy, ''old''=polling daemons, ''new''=V2 orchestrator';

-- V2-specific tracking columns
ALTER TABLE video_campaigns
ADD COLUMN IF NOT EXISTS v2_session_id VARCHAR(64);

ALTER TABLE video_campaigns
ADD COLUMN IF NOT EXISTS v2_started_at TIMESTAMP;

ALTER TABLE video_campaigns
ADD COLUMN IF NOT EXISTS v2_completed_at TIMESTAMP;

ALTER TABLE video_campaigns
ADD COLUMN IF NOT EXISTS v2_node_history JSONB DEFAULT '[]'::jsonb;

COMMENT ON COLUMN video_campaigns.v2_session_id IS
    'Unique session ID for V2 orchestrator run';

COMMENT ON COLUMN video_campaigns.v2_node_history IS
    'Array of node executions: [{node, started, completed, status}]';

-- ============================================================================
-- SECTION 4: Monitoring Views
-- ============================================================================

-- View: Current pipeline status for both systems
CREATE OR REPLACE VIEW pipeline_status AS
SELECT
    CASE
        WHEN status LIKE 'v2_%' THEN 'V2'
        WHEN status IN ('new', 'in_scripting', 'script_approved', 'generating_visuals', 'rendering') THEN 'OLD'
        ELSE 'FINAL'
    END as pipeline,
    status,
    COUNT(*) as count,
    MIN(created_at) as oldest,
    MAX(created_at) as newest
FROM video_campaigns
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY 1, 2
ORDER BY 1, 2;

COMMENT ON VIEW pipeline_status IS
    'Shows campaign counts by pipeline (OLD vs V2) and status';

-- View: V2 pipeline detailed status
CREATE OR REPLACE VIEW v2_pipeline_status AS
SELECT
    id,
    topic,
    niche,
    quality_tier,
    status,
    processor,
    v2_session_id,
    v2_started_at,
    v2_completed_at,
    EXTRACT(EPOCH FROM (COALESCE(v2_completed_at, NOW()) - v2_started_at)) as duration_seconds,
    created_at,
    updated_at
FROM video_campaigns
WHERE status LIKE 'v2_%' OR (processor = 'new' AND status IN ('published', 'failed'))
ORDER BY created_at DESC;

COMMENT ON VIEW v2_pipeline_status IS
    'Detailed view of V2 pipeline campaigns only';

-- View: Processor comparison (for migration monitoring)
CREATE OR REPLACE VIEW processor_comparison AS
SELECT
    COALESCE(processor, 'legacy') as processor,
    COUNT(*) as total_campaigns,
    COUNT(*) FILTER (WHERE status = 'published') as completed,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    COUNT(*) FILTER (WHERE status NOT IN ('published', 'failed')) as in_progress,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE status = 'published') /
        NULLIF(COUNT(*) FILTER (WHERE status IN ('published', 'failed')), 0)
    , 2) as success_rate_pct,
    AVG(
        EXTRACT(EPOCH FROM (updated_at - created_at))
    ) FILTER (WHERE status = 'published') as avg_completion_seconds,
    MIN(created_at) as first_campaign,
    MAX(created_at) as last_campaign
FROM video_campaigns
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY COALESCE(processor, 'legacy');

COMMENT ON VIEW processor_comparison IS
    'Compare OLD vs V2 performance metrics - use for migration decisions';

-- View: Stuck campaigns alert (campaigns in processing state too long)
CREATE OR REPLACE VIEW stuck_campaigns AS
SELECT
    id,
    topic,
    status,
    processor,
    created_at,
    updated_at,
    EXTRACT(EPOCH FROM (NOW() - updated_at)) / 60 as minutes_since_update,
    CASE
        WHEN status LIKE 'v2_%' THEN 'V2'
        ELSE 'OLD'
    END as pipeline
FROM video_campaigns
WHERE
    status NOT IN ('published', 'failed')
    AND updated_at < NOW() - INTERVAL '30 minutes'
ORDER BY updated_at ASC;

COMMENT ON VIEW stuck_campaigns IS
    'Campaigns that may be stuck - not updated in 30+ minutes';

-- ============================================================================
-- SECTION 5: Rollback Procedure
-- ============================================================================

-- Function to manually rollback a V2 campaign to OLD pipeline
CREATE OR REPLACE FUNCTION rollback_v2_to_old(campaign_id_param INTEGER)
RETURNS TEXT AS $$
DECLARE
    current_status TEXT;
    current_processor TEXT;
BEGIN
    -- Get current state
    SELECT status, processor INTO current_status, current_processor
    FROM video_campaigns
    WHERE id = campaign_id_param;

    -- Validate it's a V2 campaign
    IF current_processor != 'new' AND NOT current_status LIKE 'v2_%' THEN
        RETURN 'ERROR: Campaign ' || campaign_id_param || ' is not a V2 campaign';
    END IF;

    -- Don't rollback completed campaigns
    IF current_status IN ('published', 'failed') THEN
        RETURN 'ERROR: Campaign ' || campaign_id_param || ' is already in final state: ' || current_status;
    END IF;

    -- Rollback to OLD pipeline
    UPDATE video_campaigns
    SET
        status = 'new',
        processor = 'old',
        v2_session_id = NULL,
        v2_started_at = NULL,
        v2_completed_at = NULL,
        v2_node_history = '[]'::jsonb,
        updated_at = NOW()
    WHERE id = campaign_id_param;

    RETURN 'SUCCESS: Campaign ' || campaign_id_param || ' rolled back to OLD pipeline (status=new, processor=old)';
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION rollback_v2_to_old IS
    'Manually roll back a V2 campaign to be processed by OLD pipeline';

-- ============================================================================
-- SECTION 6: V2 Campaign Creation Helper
-- ============================================================================

-- Function to create a V2 campaign with proper isolation
CREATE OR REPLACE FUNCTION create_v2_campaign(
    p_topic TEXT,
    p_niche TEXT,
    p_quality_tier TEXT DEFAULT 'bulk',
    p_target_duration INTEGER DEFAULT 60,
    p_reference_url TEXT DEFAULT NULL
)
RETURNS TABLE(campaign_id INTEGER, session_id TEXT) AS $$
DECLARE
    new_id INTEGER;
    new_session TEXT;
BEGIN
    new_session := 'v2_' || md5(random()::text || clock_timestamp()::text);

    INSERT INTO video_campaigns (
        topic,
        niche,
        quality_tier,
        status,
        processor,
        v2_session_id,
        created_at,
        updated_at,
        meta
    ) VALUES (
        p_topic,
        p_niche,
        p_quality_tier,
        'v2_pending',  -- CRITICAL: V2 status, not 'new'
        'new',         -- CRITICAL: processor='new' for V2
        new_session,
        NOW(),
        NOW(),
        jsonb_build_object(
            'target_duration_seconds', p_target_duration,
            'reference_video_url', p_reference_url,
            'created_via', 'v2_api'
        )
    )
    RETURNING id INTO new_id;

    RETURN QUERY SELECT new_id, new_session;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION create_v2_campaign IS
    'Create a campaign for V2 pipeline with proper isolation (status=v2_pending, processor=new)';

-- ============================================================================
-- SECTION 7: Isolation Verification
-- ============================================================================

-- This query should return 0 rows if isolation is working
-- Run periodically to verify no cross-contamination
CREATE OR REPLACE VIEW isolation_check AS
SELECT
    id,
    topic,
    status,
    processor,
    'POTENTIAL ISSUE' as alert
FROM video_campaigns
WHERE
    -- V2 processor but OLD status
    (processor = 'new' AND status IN ('new', 'in_scripting', 'script_approved', 'generating_visuals', 'rendering'))
    OR
    -- OLD processor but V2 status
    (processor = 'old' AND status LIKE 'v2_%')
    OR
    -- V2 status without processor set
    (status LIKE 'v2_%' AND processor IS NULL);

COMMENT ON VIEW isolation_check IS
    'Should always return 0 rows - any results indicate isolation breach';

-- ============================================================================
-- VERIFICATION QUERIES (Run after migration)
-- ============================================================================

-- Verify constraint added
-- SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid = 'video_campaigns'::regclass;

-- Verify indexes exist
-- SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'video_campaigns' AND indexname LIKE '%v2%';

-- Verify isolation
-- SELECT * FROM isolation_check;

-- Check current pipeline status
-- SELECT * FROM pipeline_status;
