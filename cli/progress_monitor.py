#!/usr/bin/env python3
"""
CLI Progress Monitor for Video Generation

Connects to SSE server and displays real-time progress with visual formatting.

Usage:
    python -m cli.progress_monitor campaign-123
    python -m cli.progress_monitor --server http://localhost:8765 campaign-123
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("Please install aiohttp: pip install aiohttp")
    sys.exit(1)


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

    # Clear line
    CLEAR_LINE = "\033[2K\r"


def colored(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.RESET}"


def progress_bar(percent: float, width: int = 30) -> str:
    """Create a visual progress bar."""
    filled = int(percent / 100 * width)
    bar = "█" * filled + "░" * (width - filled)

    # Color based on progress
    if percent >= 100:
        color = Colors.GREEN
    elif percent >= 50:
        color = Colors.CYAN
    elif percent >= 25:
        color = Colors.YELLOW
    else:
        color = Colors.WHITE

    return colored(f"[{bar}]", color) + f" {percent:5.1f}%"


def format_duration(seconds: float) -> str:
    """Format duration as HH:MM:SS or MM:SS."""
    if seconds < 0:
        return "--:--"

    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_event(event: dict) -> str:
    """Format event for display."""
    event_type = event.get("type", "info")
    message = event.get("message", "")
    phase = event.get("phase", "")
    phase_num = event.get("phase_number", 0)
    total_phases = event.get("total_phases", 8)
    progress = event.get("progress_percent", 0)
    elapsed = event.get("elapsed_seconds", 0)
    eta = event.get("eta_seconds")

    # Icons and colors for event types
    type_config = {
        "started": ("🚀", Colors.GREEN),
        "completed": ("✅", Colors.GREEN),
        "failed": ("❌", Colors.RED),
        "cancelled": ("⏹️", Colors.YELLOW),
        "phase_started": ("▶️", Colors.CYAN),
        "phase_completed": ("✔️", Colors.GREEN),
        "phase_failed": ("✖️", Colors.RED),
        "progress": ("⏳", Colors.DIM),
        "step": ("•", Colors.WHITE),
        "substep": ("  →", Colors.DIM),
        "decision": ("🤔", Colors.MAGENTA),
        "retry": ("🔄", Colors.YELLOW),
        "quality_check": ("🔍", Colors.BLUE),
        "checkpoint_waiting": ("⏸️", Colors.YELLOW + Colors.BOLD),
        "checkpoint_approved": ("✅", Colors.GREEN),
        "checkpoint_rejected": ("❌", Colors.RED),
        "asset_generated": ("🎬", Colors.CYAN),
        "asset_uploaded": ("☁️", Colors.CYAN),
        "info": ("ℹ️", Colors.BLUE),
        "warning": ("⚠️", Colors.YELLOW),
        "error": ("🔴", Colors.RED),
        "debug": ("🔧", Colors.DIM),
    }

    icon, color = type_config.get(event_type, ("•", Colors.WHITE))

    # Build output lines
    lines = []

    # For progress events, show progress bar on same line
    if event_type == "progress":
        phase_indicator = f"[{phase_num}/{total_phases}]"
        time_info = format_duration(elapsed)
        if eta:
            time_info += f" ETA: {format_duration(eta)}"

        line = (
            f"{Colors.CLEAR_LINE}"
            f"{icon} {colored(phase_indicator, Colors.DIM)} "
            f"{progress_bar(progress)} "
            f"{colored(message[:40], Colors.WHITE)} "
            f"{colored(time_info, Colors.DIM)}"
        )
        lines.append(line)

    # For phase events, show header
    elif event_type in ("phase_started", "phase_completed", "phase_failed"):
        phase_display = phase.upper() if phase else "UNKNOWN"
        lines.append("")
        lines.append(
            colored(f"═══ {icon} Phase {phase_num}/{total_phases}: {phase_display} ═══", color)
        )
        if message and message != f"Starting {phase} phase":
            lines.append(f"    {message}")

    # For decisions, show agent and reasoning
    elif event_type == "decision":
        agent = event.get("agent", "agent")
        confidence = event.get("confidence")
        reasoning = event.get("reasoning")

        conf_str = f" ({confidence:.0%})" if confidence else ""
        lines.append(f"{icon} {colored(agent, Colors.MAGENTA)}: {message}{conf_str}")
        if reasoning:
            lines.append(colored(f"    └─ {reasoning}", Colors.DIM))

    # For quality checks
    elif event_type == "quality_check":
        data = event.get("data", {})
        passed = data.get("passed", False)
        issues = data.get("issues", [])

        status = colored("PASSED", Colors.GREEN) if passed else colored("FAILED", Colors.RED)
        lines.append(f"{icon} Quality Check: {status} - {message}")
        for issue in issues[:3]:  # Show first 3 issues
            lines.append(colored(f"    • {issue}", Colors.YELLOW))

    # For checkpoints
    elif event_type == "checkpoint_waiting":
        lines.append("")
        lines.append(colored("════════════════════════════════════════", Colors.YELLOW))
        lines.append(f"{icon} {colored('HUMAN APPROVAL REQUIRED', Colors.YELLOW + Colors.BOLD)}")
        lines.append(f"    {message}")
        data = event.get("data", {})
        if "proposal" in data:
            lines.append(colored(f"    Proposal: {data['proposal'][:100]}...", Colors.DIM))
        lines.append(colored("════════════════════════════════════════", Colors.YELLOW))
        lines.append("")

    # For errors
    elif event_type in ("error", "failed"):
        lines.append(f"{icon} {colored(message, color)}")
        data = event.get("data", {})
        if "error" in data:
            lines.append(colored(f"    Error: {data['error']}", Colors.DIM))

    # For asset generation
    elif event_type == "asset_generated":
        data = event.get("data", {})
        asset_type = data.get("asset_type", "asset")
        url = data.get("url")
        lines.append(f"{icon} Generated: {colored(asset_type, Colors.CYAN)}")
        if url:
            lines.append(colored(f"    → {url}", Colors.DIM))

    # Default formatting
    else:
        lines.append(f"{icon} {colored(message, color)}")

    return "\n".join(lines)


class ProgressMonitor:
    """CLI progress monitor for video generation."""

    def __init__(
        self,
        campaign_id: str,
        server_url: str = "http://localhost:8765",
    ):
        self.campaign_id = campaign_id
        self.server_url = server_url.rstrip("/")
        self.stream_url = f"{self.server_url}/stream/{campaign_id}"

        self._running = False
        self._last_progress_percent = 0

    async def start(self):
        """Start monitoring progress."""
        self._running = True

        print(colored(f"\n╔═══════════════════════════════════════════╗", Colors.CYAN))
        print(colored(f"║  AgenticVideo Progress Monitor            ║", Colors.CYAN))
        print(colored(f"╚═══════════════════════════════════════════╝", Colors.CYAN))
        print(f"Campaign: {colored(self.campaign_id, Colors.BOLD)}")
        print(f"Server:   {colored(self.stream_url, Colors.DIM)}")
        print(colored("─" * 45, Colors.DIM))
        print()

        retry_count = 0
        max_retries = 5

        while self._running and retry_count < max_retries:
            try:
                await self._stream_events()
                break  # Clean exit
            except aiohttp.ClientError as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait = 2 ** retry_count
                    print(
                        colored(
                            f"\n⚠️ Connection lost. Retrying in {wait}s... ({retry_count}/{max_retries})",
                            Colors.YELLOW,
                        )
                    )
                    await asyncio.sleep(wait)
                else:
                    print(colored(f"\n❌ Failed to connect after {max_retries} attempts", Colors.RED))
            except asyncio.CancelledError:
                break

        print(colored("\n─" * 45, Colors.DIM))
        print(colored("Monitor stopped.", Colors.DIM))

    async def _stream_events(self):
        """Stream and display events."""
        async with aiohttp.ClientSession() as session:
            async with session.get(self.stream_url) as response:
                if response.status != 200:
                    raise aiohttp.ClientError(f"Server returned {response.status}")

                async for line in response.content:
                    if not self._running:
                        break

                    line = line.decode("utf-8").strip()

                    # Parse SSE format
                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                            self._handle_event(data)
                        except json.JSONDecodeError:
                            pass

    def _handle_event(self, event: dict):
        """Handle incoming event."""
        event_type = event.get("type", "")

        # For progress events, update in place
        if event_type == "progress":
            # Only update if progress changed significantly
            progress = event.get("progress_percent", 0)
            if abs(progress - self._last_progress_percent) >= 0.5:
                self._last_progress_percent = progress
                print(format_event(event), end="", flush=True)
        else:
            # For other events, print on new line
            if event_type != "progress":
                print()  # Clear progress line
            print(format_event(event))

        # Check for terminal events
        if event_type in ("completed", "failed", "cancelled"):
            self._running = False

    def stop(self):
        """Stop monitoring."""
        self._running = False


async def main():
    parser = argparse.ArgumentParser(
        description="Monitor video generation progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s campaign-123
    %(prog)s --server http://remote:8765 campaign-456
        """,
    )
    parser.add_argument(
        "campaign_id",
        help="Campaign ID to monitor",
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8765",
        help="SSE server URL (default: http://localhost:8765)",
    )

    args = parser.parse_args()

    monitor = ProgressMonitor(
        campaign_id=args.campaign_id,
        server_url=args.server,
    )

    try:
        await monitor.start()
    except KeyboardInterrupt:
        print(colored("\n\nInterrupted by user.", Colors.YELLOW))
        monitor.stop()


if __name__ == "__main__":
    asyncio.run(main())
