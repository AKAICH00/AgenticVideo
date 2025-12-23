"""
Publisher Worker

Background service that processes the publish queue and handles
scheduled video uploads to YouTube.
"""

import asyncio
import logging
import os
import signal
from datetime import datetime, timezone
from typing import Optional
import asyncpg

from .youtube_client import YouTubeClient, VideoMetadata, UploadResult
from .metadata_generator import MetadataGenerator, VideoContext
from .scheduler import PostingScheduler

logger = logging.getLogger(__name__)


class PublisherWorker:
    """
    Background worker for processing video publish queue.

    Handles:
    - Monitoring publish queue for ready videos
    - Generating SEO-optimized metadata
    - Uploading to YouTube with retries
    - Updating campaign status in database
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        youtube_client: Optional[YouTubeClient] = None,
        metadata_generator: Optional[MetadataGenerator] = None,
        scheduler: Optional[PostingScheduler] = None,
    ):
        self.db_pool = db_pool
        self.youtube = youtube_client or YouTubeClient()
        self.metadata_gen = metadata_generator or MetadataGenerator()
        self.scheduler = scheduler or PostingScheduler(db_pool=db_pool)

        self._running = False
        self._current_task: Optional[asyncio.Task] = None

        # Configuration
        self.max_concurrent = int(os.getenv("MAX_CONCURRENT_UPLOADS", "2"))
        self.max_retries = int(os.getenv("UPLOAD_MAX_RETRIES", "3"))
        self.retry_delay = int(os.getenv("UPLOAD_RETRY_DELAY", "60"))
        self.poll_interval = 30  # seconds

    async def start(self):
        """Start the publisher worker."""
        self._running = True
        logger.info("Publisher worker starting...")

        while self._running:
            try:
                # Get videos ready to publish
                videos = await self._get_ready_videos()

                if videos:
                    logger.info(f"Found {len(videos)} videos ready to publish")

                    # Process in batches respecting max concurrent
                    for i in range(0, len(videos), self.max_concurrent):
                        batch = videos[i:i + self.max_concurrent]
                        await asyncio.gather(
                            *[self._publish_video(v) for v in batch],
                            return_exceptions=True
                        )

                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                logger.info("Publisher worker cancelled")
                break
            except Exception as e:
                logger.error(f"Publisher worker error: {e}")
                await asyncio.sleep(self.poll_interval)

    async def stop(self):
        """Stop the publisher worker."""
        self._running = False
        if self._current_task:
            self._current_task.cancel()

    async def _get_ready_videos(self) -> list[dict]:
        """Get videos ready to publish from the queue."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id,
                    topic,
                    niche,
                    target_duration,
                    final_video_url,
                    scheduled_publish_at,
                    retry_count,
                    meta
                FROM video_campaigns
                WHERE status = 'ready_to_publish'
                  AND final_video_url IS NOT NULL
                  AND (
                      scheduled_publish_at IS NULL
                      OR scheduled_publish_at <= NOW()
                  )
                  AND retry_count < $1
                ORDER BY scheduled_publish_at ASC NULLS LAST
                LIMIT $2
                """,
                self.max_retries,
                self.max_concurrent * 2,
            )
            return [dict(r) for r in rows]

    async def _publish_video(self, video: dict):
        """Publish a single video to YouTube."""
        campaign_id = video["id"]
        logger.info(f"Publishing video {campaign_id}: {video['topic']}")

        try:
            # Update status to publishing
            await self._update_status(campaign_id, "publishing")

            # Generate metadata
            context = VideoContext(
                topic=video["topic"],
                niche=video["niche"],
                video_duration_seconds=video.get("target_duration", 60),
                format_type="short" if video.get("target_duration", 60) <= 60 else "long",
            )

            meta = await self.metadata_gen.generate(context)

            # Build YouTube metadata
            youtube_meta = VideoMetadata(
                title=meta.title,
                description=meta.description,
                tags=meta.tags,
                category_id=meta.category_id,
                privacy_status=os.getenv("DEFAULT_PRIVACY_STATUS", "private"),
            )

            # Upload to YouTube
            result = await self.youtube.upload_video(
                video_path=video["final_video_url"],
                metadata=youtube_meta,
            )

            if result.success:
                # Success - update database
                await self._mark_published(
                    campaign_id,
                    result.video_id,
                    result.video_url,
                    meta,
                )
                logger.info(f"Published video {campaign_id}: {result.video_url}")
            else:
                # Failed - increment retry count
                await self._mark_failed(campaign_id, result.error_message)
                logger.error(f"Upload failed for {campaign_id}: {result.error_message}")

        except Exception as e:
            logger.error(f"Publish error for {campaign_id}: {e}")
            await self._mark_failed(campaign_id, str(e))

    async def _update_status(self, campaign_id: str, status: str):
        """Update campaign status in database."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE video_campaigns SET status = $1, updated_at = NOW() WHERE id = $2",
                status,
                campaign_id,
            )

    async def _mark_published(
        self,
        campaign_id: str,
        video_id: str,
        video_url: str,
        meta,
    ):
        """Mark campaign as successfully published."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE video_campaigns
                SET status = 'published',
                    youtube_video_id = $1,
                    published_url = $2,
                    published_at = NOW(),
                    seo_title = $3,
                    seo_description = $4,
                    updated_at = NOW()
                WHERE id = $5
                """,
                video_id,
                video_url,
                meta.title,
                meta.description,
                campaign_id,
            )

    async def _mark_failed(self, campaign_id: str, error: str):
        """Mark campaign as failed and increment retry count."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE video_campaigns
                SET retry_count = retry_count + 1,
                    last_error = $1,
                    status = CASE
                        WHEN retry_count + 1 >= $2 THEN 'failed'
                        ELSE 'ready_to_publish'
                    END,
                    updated_at = NOW()
                WHERE id = $3
                """,
                error,
                self.max_retries,
                campaign_id,
            )


async def process_scheduled():
    """
    Process scheduled videos that are due for publishing.

    Called by CronJob on a schedule.
    """
    logger.info("Processing scheduled videos...")

    db_pool = await asyncpg.create_pool(
        os.getenv("DATABASE_URL"),
        min_size=1,
        max_size=5,
    )

    try:
        async with db_pool.acquire() as conn:
            # Find videos due for publishing
            due_videos = await conn.fetch(
                """
                SELECT id
                FROM video_campaigns
                WHERE status = 'ready_to_publish'
                  AND scheduled_publish_at IS NOT NULL
                  AND scheduled_publish_at <= NOW()
                  AND final_video_url IS NOT NULL
                LIMIT 10
                """
            )

            if not due_videos:
                logger.info("No scheduled videos due for publishing")
                return

            logger.info(f"Found {len(due_videos)} scheduled videos to publish")

            # Create worker and process
            worker = PublisherWorker(db_pool)

            for video in due_videos:
                video_data = await conn.fetchrow(
                    """
                    SELECT id, topic, niche, target_duration, final_video_url,
                           scheduled_publish_at, retry_count, meta
                    FROM video_campaigns
                    WHERE id = $1
                    """,
                    video["id"],
                )

                if video_data:
                    await worker._publish_video(dict(video_data))

    finally:
        await db_pool.close()


async def main():
    """Main entry point for the publisher worker."""
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting publisher worker...")

    # Create database pool
    db_pool = await asyncpg.create_pool(
        os.getenv("DATABASE_URL"),
        min_size=2,
        max_size=10,
    )

    worker = PublisherWorker(db_pool)

    # Handle shutdown signals
    stop_event = asyncio.Event()

    def handle_signal():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal)

    # Start worker in background
    worker_task = asyncio.create_task(worker.start())

    # Wait for shutdown
    await stop_event.wait()

    # Cleanup
    await worker.stop()
    worker_task.cancel()

    try:
        await worker_task
    except asyncio.CancelledError:
        pass

    await worker.youtube.close()
    await worker.metadata_gen.close()
    await db_pool.close()

    logger.info("Publisher worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
