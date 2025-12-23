-- Migration 005: Fix Intelligence Layer Column Mismatches
-- Run after 003_add_performance_tables.sql
--
-- This migration aligns the schema with intelligence layer code expectations.
-- Specifically fixes: feedback_loop.py column references

-- 1. Add engagement_rate to video_performance (feedback_loop.py:141)
ALTER TABLE video_performance
  ADD COLUMN IF NOT EXISTS engagement_rate FLOAT;

-- 2. Add youtube_video_id alias (feedback_loop.py:147)
-- Keeps external_video_id for backward compatibility
ALTER TABLE video_performance
  ADD COLUMN IF NOT EXISTS youtube_video_id TEXT;

-- 3. Rename snapshot_at → collected_at (feedback_loop.py:149,154)
-- Check if column exists before renaming to make migration idempotent
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'video_performance' AND column_name = 'snapshot_at'
    ) THEN
        ALTER TABLE video_performance RENAME COLUMN snapshot_at TO collected_at;
    END IF;
END $$;

-- 4. Update unique constraint for new column name
-- Drop old constraint if exists
ALTER TABLE video_performance
  DROP CONSTRAINT IF EXISTS video_performance_campaign_id_platform_snapshot_at_key;

-- Add new constraint with collected_at
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'video_performance_campaign_id_collected_at_key'
    ) THEN
        ALTER TABLE video_performance
          ADD CONSTRAINT video_performance_campaign_id_collected_at_key
          UNIQUE(campaign_id, platform, collected_at);
    END IF;
END $$;

-- 5. Add target_duration to video_campaigns (feedback_loop.py:190)
ALTER TABLE video_campaigns
  ADD COLUMN IF NOT EXISTS target_duration INT;

-- 6. Backfill youtube_video_id from external_video_id where not set
UPDATE video_performance
  SET youtube_video_id = external_video_id
  WHERE youtube_video_id IS NULL AND external_video_id IS NOT NULL;

-- 7. Create index for common query pattern (performance metrics by time)
CREATE INDEX IF NOT EXISTS idx_performance_collected
  ON video_performance(collected_at DESC);

-- 8. Create index for campaign lookups with collected_at
CREATE INDEX IF NOT EXISTS idx_performance_campaign_collected
  ON video_performance(campaign_id, collected_at DESC);

-- Verification query (run manually to check)
-- SELECT column_name, data_type
-- FROM information_schema.columns
-- WHERE table_name IN ('video_performance', 'video_campaigns')
-- ORDER BY table_name, ordinal_position;
