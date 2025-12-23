-- Migration: Add processor tracking for V2 migration
-- Created: 2025-12-21
-- Purpose: Track which processor (old daemons vs new orchestrator) handled each campaign

-- Add processor column to video_campaigns
ALTER TABLE video_campaigns
ADD COLUMN IF NOT EXISTS processor VARCHAR(10) DEFAULT 'old';

-- Add comment explaining the column
COMMENT ON COLUMN video_campaigns.processor IS
    'Processor that handled this campaign: old (polling daemons) or new (orchestrator)';

-- Index for monitoring migration progress
CREATE INDEX IF NOT EXISTS idx_campaigns_processor
ON video_campaigns(processor);

-- Add processed_by_orchestrator timestamp for new system
ALTER TABLE video_campaigns
ADD COLUMN IF NOT EXISTS orchestrator_session_id VARCHAR(64);

-- Track orchestrator-specific timing
ALTER TABLE video_campaigns
ADD COLUMN IF NOT EXISTS orchestrator_started_at TIMESTAMP;

ALTER TABLE video_campaigns
ADD COLUMN IF NOT EXISTS orchestrator_completed_at TIMESTAMP;

-- View for migration monitoring
CREATE OR REPLACE VIEW v2_migration_status AS
SELECT
    processor,
    status,
    COUNT(*) as count,
    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_duration_seconds,
    COUNT(*) FILTER (WHERE status = 'published') as success_count,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_count,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'published') / NULLIF(COUNT(*), 0), 2) as success_rate_pct
FROM video_campaigns
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY processor, status
ORDER BY processor, status;

-- View for comparing old vs new processor performance
CREATE OR REPLACE VIEW processor_comparison AS
SELECT
    processor,
    COUNT(*) as total_campaigns,
    COUNT(*) FILTER (WHERE status = 'published') as completed,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'published') / NULLIF(COUNT(*), 0), 2) as success_rate,
    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) FILTER (WHERE status = 'published') as avg_completion_seconds,
    MIN(created_at) as first_campaign,
    MAX(created_at) as last_campaign
FROM video_campaigns
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY processor;

-- Add quality tier tracking for split routing
ALTER TABLE video_campaigns
ADD COLUMN IF NOT EXISTS quality_tier VARCHAR(20) DEFAULT 'bulk';

COMMENT ON COLUMN video_campaigns.quality_tier IS
    'Quality tier: premium (API-based, higher quality) or bulk (self-hosted, cost-effective)';

-- Add index for quality tier (used in split routing)
CREATE INDEX IF NOT EXISTS idx_campaigns_quality_tier
ON video_campaigns(quality_tier);

-- Combined index for routing queries
CREATE INDEX IF NOT EXISTS idx_campaigns_routing
ON video_campaigns(status, processor, quality_tier);
