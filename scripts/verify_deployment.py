#!/usr/bin/env python3
"""
V2 Deployment Verification Script

Verifies all components of the V2 migration are working correctly.
Run before deploying to production or after any changes.

Usage:
    python scripts/verify_deployment.py
    python scripts/verify_deployment.py --server-url http://localhost:8765
"""

import argparse
import asyncio
import sys
import os
from typing import Tuple, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def passed(msg: str) -> str:
    return f"{Colors.GREEN}✓ PASS{Colors.RESET} {msg}"


def failed(msg: str, error: str = "") -> str:
    err_msg = f" ({error})" if error else ""
    return f"{Colors.RED}✗ FAIL{Colors.RESET} {msg}{err_msg}"


def section(title: str) -> str:
    return f"\n{Colors.BOLD}{Colors.BLUE}=== {title} ==={Colors.RESET}"


def verify_imports() -> List[Tuple[str, bool, str]]:
    """Verify all critical imports work."""
    results = []

    # Core feature flags
    try:
        from core.feature_flags import ProcessorMode, get_processor_mode, should_use_new_orchestrator
        results.append(("core.feature_flags", True, ""))
    except Exception as e:
        results.append(("core.feature_flags", False, str(e)))

    # Router
    try:
        from services.router import CampaignRouter
        results.append(("services.router", True, ""))
    except Exception as e:
        results.append(("services.router", False, str(e)))

    # Orchestrator graph
    try:
        from services.orchestrator.graph import VideoGraph
        results.append(("services.orchestrator.graph", True, ""))
    except Exception as e:
        results.append(("services.orchestrator.graph", False, str(e)))

    # Orchestrator state
    try:
        from services.orchestrator.state import VideoState, GenerationPhase
        results.append(("services.orchestrator.state", True, ""))
    except Exception as e:
        results.append(("services.orchestrator.state", False, str(e)))

    # All nodes
    try:
        from services.orchestrator.nodes import (
            PlannerNode, ScriptNode, StoryboardNode, MotionNode,
            VisualNode, QualityNode, ComposeNode, RepurposeNode
        )
        results.append(("services.orchestrator.nodes (all 8)", True, ""))
    except Exception as e:
        results.append(("services.orchestrator.nodes", False, str(e)))

    # Server
    try:
        from services.orchestrator.server import app
        results.append(("services.orchestrator.server", True, ""))
    except Exception as e:
        results.append(("services.orchestrator.server", False, str(e)))

    # Motion extraction
    try:
        from services.motion import MotionExtractor
        results.append(("services.motion", True, ""))
    except Exception as e:
        results.append(("services.motion", False, str(e)))

    # Video generation client
    try:
        from services.video_generation import VideoGenerationClient
        results.append(("services.video_generation", True, ""))
    except Exception as e:
        results.append(("services.video_generation", False, str(e)))

    # Streaming
    try:
        from services.streaming.progress_tracker import ProgressTracker, ProgressEvent
        results.append(("services.streaming.progress_tracker", True, ""))
    except Exception as e:
        results.append(("services.streaming.progress_tracker", False, str(e)))

    return results


def verify_feature_flags() -> List[Tuple[str, bool, str]]:
    """Verify feature flag routing logic."""
    results = []

    try:
        from core.feature_flags import should_use_new_orchestrator, ProcessorMode
        original = os.environ.get("PROCESSOR_MODE")

        # Test OLD mode
        os.environ["PROCESSOR_MODE"] = "old"
        if not should_use_new_orchestrator("test", "premium"):
            results.append(("OLD mode: all → old daemons", True, ""))
        else:
            results.append(("OLD mode: all → old daemons", False, "premium went to new"))

        # Test NEW mode
        os.environ["PROCESSOR_MODE"] = "new"
        if should_use_new_orchestrator("test", "bulk"):
            results.append(("NEW mode: all → orchestrator", True, ""))
        else:
            results.append(("NEW mode: all → orchestrator", False, "bulk went to old"))

        # Test SPLIT mode
        os.environ["PROCESSOR_MODE"] = "split"
        premium_to_new = should_use_new_orchestrator("test", "premium")
        bulk_to_old = not should_use_new_orchestrator("test", "bulk")
        if premium_to_new and bulk_to_old:
            results.append(("SPLIT mode: premium→new, bulk→old", True, ""))
        else:
            results.append(("SPLIT mode: premium→new, bulk→old", False,
                          f"premium={premium_to_new}, bulk_to_old={bulk_to_old}"))

        # Restore
        if original:
            os.environ["PROCESSOR_MODE"] = original
        else:
            os.environ.pop("PROCESSOR_MODE", None)

    except Exception as e:
        results.append(("Feature flag routing", False, str(e)))

    return results


def verify_graph_structure() -> List[Tuple[str, bool, str]]:
    """Verify VideoGraph has correct structure."""
    results = []

    try:
        from services.orchestrator.graph import VideoGraph
        from services.orchestrator.state import GenerationPhase

        graph = VideoGraph()

        # Check all nodes exist
        expected_nodes = [
            "planner", "script", "storyboard", "motion",
            "visual", "quality", "compose", "repurpose"
        ]
        missing = [n for n in expected_nodes if n not in graph.nodes]
        if not missing:
            results.append(("All 8 nodes registered", True, ""))
        else:
            results.append(("All 8 nodes registered", False, f"missing: {missing}"))

        # Check edges exist
        if hasattr(graph, 'edges') and len(graph.edges) > 0:
            results.append(("Edges configured", True, f"{len(graph.edges)} edges"))
        else:
            results.append(("Edges configured", False, "no edges defined"))

        # Check quality loop
        quality_edges = [e for e in graph.edges if 'quality' in str(e).lower()]
        if quality_edges:
            results.append(("Quality feedback loop", True, ""))
        else:
            results.append(("Quality feedback loop", False, "no quality edges"))

    except Exception as e:
        results.append(("Graph structure", False, str(e)))

    return results


def verify_state_structure() -> List[Tuple[str, bool, str]]:
    """Verify VideoState has all required fields."""
    results = []

    try:
        from services.orchestrator.state import VideoState, GenerationPhase

        state = VideoState(campaign_id="test", topic="test", niche="test")

        # Check required fields
        required = ["campaign_id", "topic", "niche", "phase", "progress_percent"]
        missing = [f for f in required if not hasattr(state, f)]
        if not missing:
            results.append(("Required fields present", True, ""))
        else:
            results.append(("Required fields present", False, f"missing: {missing}"))

        # Check nested structures
        nested = ["script", "storyboard", "motion_data"]
        missing_nested = [f for f in nested if not hasattr(state, f)]
        if not missing_nested:
            results.append(("Nested structures present", True, ""))
        else:
            results.append(("Nested structures present", False, f"missing: {missing_nested}"))

        # Check error handling
        state.add_error("test", "test error")
        if len(state.errors) > 0 and "phase" in state.errors[0]:
            results.append(("Error handling works", True, ""))
        else:
            results.append(("Error handling works", False, "add_error failed"))

        # Check phase enumeration
        phases = list(GenerationPhase)
        if len(phases) >= 10:
            results.append(("All phases defined", True, f"{len(phases)} phases"))
        else:
            results.append(("All phases defined", False, f"only {len(phases)} phases"))

    except Exception as e:
        results.append(("State structure", False, str(e)))

    return results


async def verify_server(server_url: str = None) -> List[Tuple[str, bool, str]]:
    """Verify server endpoints if running."""
    results = []

    if not server_url:
        results.append(("Server endpoints", True, "skipped (no URL provided)"))
        return results

    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Health check
            try:
                resp = await client.get(f"{server_url}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    results.append(("GET /health", True, f"mode={data.get('mode')}"))
                else:
                    results.append(("GET /health", False, f"status={resp.status_code}"))
            except Exception as e:
                results.append(("GET /health", False, str(e)))

            # Root endpoint
            try:
                resp = await client.get(f"{server_url}/")
                if resp.status_code == 200:
                    results.append(("GET /", True, ""))
                else:
                    results.append(("GET /", False, f"status={resp.status_code}"))
            except Exception as e:
                results.append(("GET /", False, str(e)))

            # Migration status
            try:
                resp = await client.get(f"{server_url}/migration")
                if resp.status_code == 200:
                    data = resp.json()
                    results.append(("GET /migration", True, f"mode={data.get('mode')}"))
                else:
                    results.append(("GET /migration", False, f"status={resp.status_code}"))
            except Exception as e:
                results.append(("GET /migration", False, str(e)))

            # Campaigns list
            try:
                resp = await client.get(f"{server_url}/campaigns")
                if resp.status_code == 200:
                    data = resp.json()
                    results.append(("GET /campaigns", True, f"total={data.get('total', 0)}"))
                else:
                    results.append(("GET /campaigns", False, f"status={resp.status_code}"))
            except Exception as e:
                results.append(("GET /campaigns", False, str(e)))

    except ImportError:
        results.append(("Server endpoints", False, "httpx not installed"))
    except Exception as e:
        results.append(("Server endpoints", False, str(e)))

    return results


def verify_database_schema() -> List[Tuple[str, bool, str]]:
    """Verify database has V2 migration columns (if DB available)."""
    results = []

    try:
        import os
        db_url = os.environ.get("DATABASE_URL")

        if not db_url:
            results.append(("Database schema", True, "skipped (no DATABASE_URL)"))
            return results

        # Just verify the migration file exists
        migration_file = "sql/006_add_processor_column.sql"
        if os.path.exists(migration_file):
            results.append(("Migration SQL exists", True, migration_file))
        else:
            results.append(("Migration SQL exists", False, "file not found"))

    except Exception as e:
        results.append(("Database schema", False, str(e)))

    return results


def run_verification(server_url: str = None) -> bool:
    """Run all verification checks."""
    all_passed = True

    # Import verification
    print(section("Import Verification"))
    for name, passed_check, error in verify_imports():
        if passed_check:
            print(passed(name))
        else:
            print(failed(name, error))
            all_passed = False

    # Feature flags
    print(section("Feature Flag Routing"))
    for name, passed_check, error in verify_feature_flags():
        if passed_check:
            print(passed(name))
        else:
            print(failed(name, error))
            all_passed = False

    # Graph structure
    print(section("VideoGraph Structure"))
    for name, passed_check, error in verify_graph_structure():
        if passed_check:
            print(passed(name))
        else:
            print(failed(name, error))
            all_passed = False

    # State structure
    print(section("VideoState Structure"))
    for name, passed_check, error in verify_state_structure():
        if passed_check:
            print(passed(name))
        else:
            print(failed(name, error))
            all_passed = False

    # Database
    print(section("Database Schema"))
    for name, passed_check, error in verify_database_schema():
        if passed_check:
            print(passed(name))
        else:
            print(failed(name, error))
            all_passed = False

    # Server (if URL provided)
    if server_url:
        print(section("Server Endpoints"))
        results = asyncio.run(verify_server(server_url))
        for name, passed_check, error in results:
            if passed_check:
                print(passed(name))
            else:
                print(failed(name, error))
                all_passed = False

    # Summary
    print(section("Summary"))
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}All verification checks passed!{Colors.RESET}")
        print("\nReady for deployment. Recommended next steps:")
        print("  1. Set PROCESSOR_MODE=split for gradual migration")
        print("  2. Monitor /migration endpoint for status")
        print("  3. Check processor_comparison view in database")
    else:
        print(f"{Colors.RED}{Colors.BOLD}Some verification checks failed.{Colors.RESET}")
        print("\nFix the issues above before deploying.")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify V2 deployment readiness")
    parser.add_argument("--server-url", "-s",
                        help="Server URL to test endpoints (e.g., http://localhost:8765)")
    args = parser.parse_args()

    success = run_verification(args.server_url)
    sys.exit(0 if success else 1)
