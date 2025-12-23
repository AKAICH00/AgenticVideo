-- ============================================================================
-- V2 State Persistence Columns
--
-- Adds columns for checkpointing VideoState to survive pod restarts.
--
-- Usage:
--   psql $DATABASE_URL -f sql/008_v2_state_persistence.sql
-- ============================================================================

-- Add v2_state_snapshot column for full state persistence
ALTER TABLE video_campaigns
ADD COLUMN IF NOT EXISTS v2_state_snapshot JSONB;

-- Add v2_state_saved_at for tracking last checkpoint time
ALTER TABLE video_campaigns
ADD COLUMN IF NOT EXISTS v2_state_saved_at TIMESTAMP;

-- Index for finding incomplete V2 campaigns (for recovery)
CREATE INDEX IF NOT EXISTS idx_v2_incomplete_campaigns
ON video_campaigns (processor, status)
WHERE processor = 'new' AND status LIKE 'v2_%';

-- Index for finding campaigns with state snapshots
CREATE INDEX IF NOT EXISTS idx_v2_campaigns_with_state
ON video_campaigns (id)
WHERE v2_state_snapshot IS NOT NULL;

-- View for recovery monitoring
CREATE OR REPLACE VIEW v2_recovery_candidates AS
SELECT
    id,
    topic,
    status,
    v2_session_id,
    v2_state_snapshot IS NOT NULL as has_state,
    v2_state_saved_at,
    v2_started_at,
    updated_at,
    EXTRACT(EPOCH FROM (NOW() - updated_at)) / 60 as minutes_since_update
FROM video_campaigns
WHERE processor = 'new'
AND status LIKE 'v2_%'
AND status NOT IN ('published', 'failed')
ORDER BY v2_started_at ASC;

-- Comment on new columns
COMMENT ON COLUMN video_campaigns.v2_state_snapshot IS 'Full VideoState serialized as JSON for pod restart recovery';
COMMENT ON COLUMN video_campaigns.v2_state_saved_at IS 'Timestamp of last state checkpoint';
