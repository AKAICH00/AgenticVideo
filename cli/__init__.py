"""
AgenticVideo CLI Tools

Command-line tools for interacting with the video generation system.

Tools:
- progress_monitor: Real-time progress visualization
- generate: Start video generation from CLI
- status: Check campaign status
"""

from .progress_monitor import ProgressMonitor

__all__ = ["ProgressMonitor"]
