"""
Feature Flags for V2 Migration

Enables gradual migration from OLD polling daemons to NEW orchestrator.

Environment variable PROCESSOR_MODE controls routing:
- "old": Use polling daemons (default for safety)
- "new": Use VideoGraph orchestrator
- "split": Route premium → new, bulk → old

Usage:
    from core.feature_flags import should_use_new_orchestrator

    if should_use_new_orchestrator(campaign_id, quality_tier):
        # Use VideoGraph
    else:
        # Let OLD daemons pick it up
"""

import os
from enum import Enum
from typing import Optional


class ProcessorMode(Enum):
    """Processor routing mode for gradual migration."""
    OLD = "old"      # Use polling daemons (agents/)
    NEW = "new"      # Use VideoGraph orchestrator (services/orchestrator/)
    SPLIT = "split"  # Route by quality tier: premium → new, bulk → old


def get_processor_mode() -> ProcessorMode:
    """Get the current processor mode from environment."""
    mode = os.getenv("PROCESSOR_MODE", "old").lower()
    try:
        return ProcessorMode(mode)
    except ValueError:
        # Default to OLD for safety
        return ProcessorMode.OLD


def should_use_new_orchestrator(
    campaign_id: str,
    quality_tier: str = "bulk",
    override: Optional[str] = None,
) -> bool:
    """
    Determine if a campaign should use the new orchestrator.

    Args:
        campaign_id: Campaign ID (for future per-campaign overrides)
        quality_tier: "premium" or "bulk"
        override: Optional explicit override ("old" or "new")

    Returns:
        True if should use new orchestrator, False for old daemons
    """
    # Allow explicit override for testing
    if override:
        return override.lower() == "new"

    mode = get_processor_mode()

    if mode == ProcessorMode.OLD:
        return False

    if mode == ProcessorMode.NEW:
        return True

    # SPLIT mode: premium campaigns use new, bulk uses old
    if mode == ProcessorMode.SPLIT:
        return quality_tier.lower() == "premium"

    # Default to old for safety
    return False


def get_migration_status() -> dict:
    """Get current migration configuration status."""
    mode = get_processor_mode()
    return {
        "mode": mode.value,
        "description": {
            "old": "All campaigns routed to polling daemons",
            "new": "All campaigns routed to VideoGraph orchestrator",
            "split": "Premium → new orchestrator, Bulk → old daemons",
        }.get(mode.value, "Unknown"),
        "env_var": "PROCESSOR_MODE",
        "current_value": os.getenv("PROCESSOR_MODE", "not set"),
    }
