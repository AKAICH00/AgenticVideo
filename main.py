#!/usr/bin/env python3
"""
AgenticVideo - Main Entry Point

Starts the video generation system with SSE progress streaming.

Usage:
    # Start server mode (SSE + API)
    python main.py server

    # Generate a single video
    python main.py generate --topic "AI trends 2025" --reference https://tiktok.com/...

    # Monitor existing campaign
    python main.py monitor campaign-123
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional
from uuid import uuid4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agenticvideo")


async def start_server(host: str = "0.0.0.0", port: int = 8765):
    """Start the SSE server for progress streaming."""
    from services.streaming import SSEServer

    server = SSEServer(host=host, port=port)
    await server.start()

    logger.info(f"AgenticVideo server running at http://{host}:{port}")
    logger.info("Press Ctrl+C to stop")

    # Keep running until interrupted
    stop_event = asyncio.Event()

    def handle_signal():
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal)

    await stop_event.wait()
    await server.stop()

    logger.info("Server stopped")


async def generate_video(
    topic: str,
    reference_url: Optional[str] = None,
    quality_tier: str = "premium",
    output_dir: str = "./output",
    sse_server_url: Optional[str] = None,
):
    """
    Generate a video from topic and optional reference.

    Args:
        topic: Video topic/script idea
        reference_url: Optional TikTok/YouTube URL for motion reference
        quality_tier: "premium" or "bulk"
        output_dir: Directory for output files
        sse_server_url: SSE server URL for progress streaming
    """
    from core.config import get_config
    from services.orchestrator import VideoGraph, VideoState, GenerationPhase
    from services.streaming import ProgressTracker, SSEServer

    config = get_config()
    campaign_id = str(uuid4())

    logger.info(f"Starting video generation for campaign {campaign_id}")
    logger.info(f"Topic: {topic}")
    if reference_url:
        logger.info(f"Reference: {reference_url}")
    logger.info(f"Quality: {quality_tier}")

    # Create progress tracker
    tracker = ProgressTracker(campaign_id=campaign_id)

    # Console output callback
    def print_progress(event):
        print(event.to_cli_line())

    tracker.on_event(print_progress)

    # Optional: Connect to SSE server for remote monitoring
    sse_server = None
    if sse_server_url:
        import aiohttp
        # Would connect to existing server
        logger.info(f"Progress will be streamed to {sse_server_url}")

    # Create progress callback for orchestrator
    async def on_progress(event_data: dict):
        # Map orchestrator events to tracker methods
        event_type = event_data.get("type", "info")

        if event_type == "phase_started":
            tracker.phase_started(
                phase=event_data.get("phase", ""),
                phase_number=event_data.get("phase_number", 0),
                message=event_data.get("message"),
            )
        elif event_type == "phase_completed":
            tracker.phase_completed(
                phase=event_data.get("phase", ""),
                message=event_data.get("message"),
                data=event_data.get("data"),
            )
        elif event_type == "progress":
            tracker.progress(
                percent=event_data.get("percent", 0),
                step=event_data.get("step", ""),
                substep=event_data.get("substep"),
                eta_seconds=event_data.get("eta"),
            )
        elif event_type == "decision":
            tracker.decision(
                agent=event_data.get("agent", ""),
                message=event_data.get("message", ""),
                confidence=event_data.get("confidence"),
                reasoning=event_data.get("reasoning"),
            )
        elif event_type == "retry":
            tracker.retry(
                attempt=event_data.get("attempt", 0),
                max_attempts=event_data.get("max_attempts", 3),
                reason=event_data.get("reason", ""),
            )
        elif event_type == "quality_check":
            tracker.quality_check(
                passed=event_data.get("passed", False),
                score=event_data.get("score", 0),
                issues=event_data.get("issues", []),
            )
        elif event_type == "error":
            tracker.error(
                message=event_data.get("message", ""),
                error=event_data.get("error"),
            )
        else:
            tracker.info(
                message=event_data.get("message", ""),
                data=event_data.get("data", {}),
            )

    # Initialize state
    initial_state = VideoState(
        campaign_id=campaign_id,
        topic=topic,
        reference_video_url=reference_url or "",
        quality_tier=quality_tier,
    )

    # Create and run graph
    graph = VideoGraph(config=config, on_progress=on_progress)

    tracker.started(f"Generating video: {topic[:50]}...")

    try:
        final_state = await graph.run(initial_state)

        if final_state.phase == GenerationPhase.COMPLETED:
            output_url = final_state.output_video_url
            tracker.completed(
                message="Video generation completed!",
                data={
                    "output_url": output_url,
                    "duration_seconds": final_state.metadata.get("duration", 0),
                },
            )
            logger.info(f"Video ready: {output_url}")
            return output_url
        else:
            tracker.failed(
                message=f"Generation ended in phase: {final_state.phase.value}",
                error=final_state.error_message,
            )
            return None

    except Exception as e:
        tracker.failed(message="Generation failed", error=str(e))
        logger.exception("Video generation failed")
        raise


async def monitor_campaign(campaign_id: str, server_url: str = "http://localhost:8765"):
    """Monitor an existing campaign's progress."""
    from cli.progress_monitor import ProgressMonitor

    monitor = ProgressMonitor(campaign_id=campaign_id, server_url=server_url)
    await monitor.start()


def main():
    parser = argparse.ArgumentParser(
        description="AgenticVideo - AI Video Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start SSE server
    python main.py server

    # Generate a video
    python main.py generate --topic "Top 5 AI tools for productivity"

    # Generate with TikTok motion reference
    python main.py generate --topic "Dance tutorial" --reference https://tiktok.com/...

    # Monitor campaign progress
    python main.py monitor campaign-abc123
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start SSE server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    server_parser.add_argument("--port", type=int, default=8765, help="Port to bind")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate a video")
    gen_parser.add_argument("--topic", "-t", required=True, help="Video topic/idea")
    gen_parser.add_argument("--reference", "-r", help="Reference video URL")
    gen_parser.add_argument(
        "--quality",
        "-q",
        choices=["premium", "bulk"],
        default="premium",
        help="Quality tier",
    )
    gen_parser.add_argument("--output", "-o", default="./output", help="Output directory")
    gen_parser.add_argument("--sse-server", help="SSE server URL for streaming")

    # Monitor command
    mon_parser = subparsers.add_parser("monitor", help="Monitor campaign progress")
    mon_parser.add_argument("campaign_id", help="Campaign ID to monitor")
    mon_parser.add_argument(
        "--server",
        default="http://localhost:8765",
        help="SSE server URL",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Check server status")
    status_parser.add_argument(
        "--server",
        default="http://localhost:8765",
        help="SSE server URL",
    )

    # Intelligence command
    intel_parser = subparsers.add_parser(
        "intelligence",
        help="Start intelligence background services (trend monitoring + feedback)"
    )
    intel_parser.add_argument(
        "--niches",
        nargs="+",
        default=["tech", "finance"],
        help="Niches to monitor"
    )
    intel_parser.add_argument(
        "--mode",
        choices=["passive", "active"],
        default="passive",
        help="passive=queue only, active=auto-generate",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run appropriate command
    if args.command == "server":
        asyncio.run(start_server(host=args.host, port=args.port))

    elif args.command == "generate":
        result = asyncio.run(
            generate_video(
                topic=args.topic,
                reference_url=args.reference,
                quality_tier=args.quality,
                output_dir=args.output,
                sse_server_url=args.sse_server,
            )
        )
        sys.exit(0 if result else 1)

    elif args.command == "monitor":
        asyncio.run(monitor_campaign(args.campaign_id, args.server))

    elif args.command == "status":
        import aiohttp

        async def check_status():
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"{args.server}/status") as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            print(f"Server: {args.server}")
                            print(f"Status: Online")
                            print(f"Connected clients: {data['clients']['total']}")
                            print(f"Active campaigns: {len(data['campaigns'])}")
                            for cid, info in data["campaigns"].items():
                                print(f"  - {cid}: {info['events']} events")
                        else:
                            print(f"Server returned status {resp.status}")
                except aiohttp.ClientError as e:
                    print(f"Cannot connect to server: {e}")
                    sys.exit(1)

        asyncio.run(check_status())

    elif args.command == "intelligence":
        asyncio.run(start_intelligence(args.niches, args.mode))


async def start_intelligence(niches: list, mode: str):
    """
    Start intelligence background services.

    Runs TrendTrigger and PerformanceTracker in parallel:
    - TrendTrigger: Monitors trends every 30 minutes, creates campaigns
    - PerformanceTracker: Collects video performance data hourly
    """
    import asyncpg
    from core.config import get_config
    from services.intelligence_bridge.trend_trigger import TrendTrigger
    from services.intelligence_bridge.feedback_loop import FeedbackLoop, PerformanceTracker

    config = get_config()
    logger.info(f"Starting intelligence services for niches: {niches}")
    logger.info(f"Mode: {mode}")

    # Connect to database
    db_pool = await asyncpg.create_pool(
        config.database.url,
        min_size=2,
        max_size=10,
    )

    # Initialize services
    trigger = TrendTrigger(db_pool=db_pool)
    feedback = FeedbackLoop(db_pool=db_pool)
    tracker = PerformanceTracker(feedback, niches)

    # Handle shutdown
    stop_event = asyncio.Event()

    def handle_signal():
        logger.info("Shutting down intelligence services...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal)

    logger.info("Intelligence services running. Press Ctrl+C to stop.")

    # Start services in parallel
    trigger_task = asyncio.create_task(trigger.start(niches, mode))
    tracker_task = asyncio.create_task(tracker.start())

    # Wait for shutdown signal
    await stop_event.wait()

    # Cleanup
    logger.info("Stopping services...")
    await trigger.stop()
    await tracker.stop()
    await feedback.close()
    await db_pool.close()

    logger.info("Intelligence services stopped")


if __name__ == "__main__":
    main()
