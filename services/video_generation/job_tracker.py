"""
Job Tracker - Database persistence for video generation jobs.

Tracks all generation requests for:
- Cost accounting
- Retry management
- Performance analytics
- Audit trail
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

import asyncpg

from .client import GenerationRequest, VideoResult, GenerationStatus

logger = logging.getLogger(__name__)


class JobTracker:
    """
    Persists video generation jobs to PostgreSQL.

    Usage:
        tracker = JobTracker(db_pool)

        # Record new job
        await tracker.create_job(request, result)

        # Update job status
        await tracker.update_job(job_id, result)

        # Get job history
        jobs = await tracker.get_jobs_by_campaign(campaign_id)
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def create_job(
        self,
        request: GenerationRequest,
        result: Optional[VideoResult] = None,
    ) -> UUID:
        """
        Create a new job record.

        Args:
            request: The generation request
            result: Optional initial result (if synchronous)

        Returns:
            The database job ID
        """
        async with self.db_pool.acquire() as conn:
            job_id = await conn.fetchval(
                """
                INSERT INTO video_generation_jobs (
                    scene_id,
                    provider,
                    model,
                    quality_tier,
                    prompt,
                    negative_prompt,
                    duration_seconds,
                    aspect_ratio,
                    reference_image_url,
                    motion_data_id,
                    external_job_id,
                    status,
                    estimated_cost_usd,
                    submitted_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                )
                RETURNING id
                """,
                request.scene_id and UUID(request.scene_id),
                result.provider if result else None,
                request.model.value,
                request.quality_tier,
                request.prompt,
                request.negative_prompt,
                request.duration_seconds,
                request.aspect_ratio,
                request.reference_image_url,
                request.motion_data_id and UUID(request.motion_data_id),
                result.external_job_id if result else None,
                (result.status.value if result else GenerationStatus.PENDING.value),
                result.estimated_cost_usd if result else None,
                datetime.utcnow(),
            )

            logger.info(f"Created job {job_id} for request {request.request_id}")
            return job_id

    async def update_job(
        self,
        job_id: UUID,
        result: VideoResult,
    ):
        """
        Update a job with new result data.

        Args:
            job_id: Database job ID
            result: Updated result
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE video_generation_jobs SET
                    status = $1,
                    result_url = $2,
                    external_job_id = COALESCE($3, external_job_id),
                    estimated_cost_usd = COALESCE($4, estimated_cost_usd),
                    actual_cost_usd = $5,
                    completed_at = $6,
                    processing_time_seconds = $7,
                    error_code = $8,
                    error_message = $9,
                    retry_count = $10
                WHERE id = $11
                """,
                result.status.value,
                result.video_url,
                result.external_job_id,
                result.estimated_cost_usd,
                result.actual_cost_usd,
                result.completed_at,
                result.processing_time_seconds,
                result.error_code,
                result.error_message,
                result.retry_count,
                job_id,
            )

            logger.info(f"Updated job {job_id} status to {result.status.value}")

    async def get_job(self, job_id: UUID) -> Optional[dict]:
        """Get a single job by ID."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM video_generation_jobs WHERE id = $1
                """,
                job_id,
            )
            return dict(row) if row else None

    async def get_jobs_by_campaign(
        self,
        campaign_id: UUID,
        status: Optional[GenerationStatus] = None,
    ) -> list[dict]:
        """
        Get all jobs for a campaign.

        Args:
            campaign_id: Campaign UUID
            status: Optional filter by status

        Returns:
            List of job records
        """
        async with self.db_pool.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    """
                    SELECT vgj.*
                    FROM video_generation_jobs vgj
                    JOIN visual_scenes vs ON vs.id = vgj.scene_id
                    WHERE vs.campaign_id = $1 AND vgj.status = $2
                    ORDER BY vgj.submitted_at DESC
                    """,
                    campaign_id,
                    status.value,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT vgj.*
                    FROM video_generation_jobs vgj
                    JOIN visual_scenes vs ON vs.id = vgj.scene_id
                    WHERE vs.campaign_id = $1
                    ORDER BY vgj.submitted_at DESC
                    """,
                    campaign_id,
                )

            return [dict(row) for row in rows]

    async def get_pending_jobs(self, limit: int = 100) -> list[dict]:
        """Get all pending jobs that need processing."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM video_generation_jobs
                WHERE status IN ('pending', 'queued', 'processing')
                ORDER BY submitted_at ASC
                LIMIT $1
                """,
                limit,
            )
            return [dict(row) for row in rows]

    async def get_failed_jobs(
        self,
        max_retries: int = 3,
        limit: int = 50,
    ) -> list[dict]:
        """Get failed jobs eligible for retry."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM video_generation_jobs
                WHERE status = 'failed'
                  AND retry_count < $1
                ORDER BY completed_at DESC
                LIMIT $2
                """,
                max_retries,
                limit,
            )
            return [dict(row) for row in rows]

    async def get_cost_summary(
        self,
        campaign_id: Optional[UUID] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict:
        """
        Get cost summary for jobs.

        Args:
            campaign_id: Optional filter by campaign
            start_date: Optional start of date range
            end_date: Optional end of date range

        Returns:
            Summary with total costs by provider/model
        """
        async with self.db_pool.acquire() as conn:
            base_query = """
                SELECT
                    provider,
                    model,
                    COUNT(*) as job_count,
                    SUM(COALESCE(actual_cost_usd, estimated_cost_usd)) as total_cost,
                    AVG(processing_time_seconds) as avg_processing_time,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed_count,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed_count
                FROM video_generation_jobs
                WHERE 1=1
            """
            params = []
            param_idx = 1

            if campaign_id:
                base_query += f"""
                    AND scene_id IN (
                        SELECT id FROM visual_scenes WHERE campaign_id = ${param_idx}
                    )
                """
                params.append(campaign_id)
                param_idx += 1

            if start_date:
                base_query += f" AND submitted_at >= ${param_idx}"
                params.append(start_date)
                param_idx += 1

            if end_date:
                base_query += f" AND submitted_at <= ${param_idx}"
                params.append(end_date)
                param_idx += 1

            base_query += " GROUP BY provider, model"

            rows = await conn.fetch(base_query, *params)

            # Also get totals
            totals = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_jobs,
                    SUM(COALESCE(actual_cost_usd, estimated_cost_usd)) as total_cost,
                    SUM(duration_seconds) as total_video_seconds
                FROM video_generation_jobs
                WHERE status = 'completed'
                """,
            )

            return {
                "by_provider_model": [dict(row) for row in rows],
                "totals": dict(totals) if totals else {},
            }

    async def increment_retry(self, job_id: UUID) -> int:
        """
        Increment retry count for a job.

        Returns:
            New retry count
        """
        async with self.db_pool.acquire() as conn:
            return await conn.fetchval(
                """
                UPDATE video_generation_jobs
                SET retry_count = retry_count + 1,
                    status = 'pending'
                WHERE id = $1
                RETURNING retry_count
                """,
                job_id,
            )
