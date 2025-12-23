#!/usr/bin/env python3
"""
V2 Isolation Monitor

Monitors the V2 orchestrator deployment for isolation violations and health.
Run periodically to ensure OLD agents never pick up V2 campaigns.

CRITICAL: This script should ALWAYS report "ISOLATION OK".
Any violations indicate a serious bug that must be fixed immediately.

Usage:
    # Basic check against running server
    python scripts/monitor_v2_isolation.py --server-url http://localhost:8765

    # Full check including database
    python scripts/monitor_v2_isolation.py --server-url http://localhost:8765 --database

    # Continuous monitoring (for cron/k8s)
    python scripts/monitor_v2_isolation.py --server-url http://localhost:8765 --continuous

    # Kubernetes readiness/liveness probe
    python scripts/monitor_v2_isolation.py --server-url http://video-orchestrator:8765 --probe
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def ok(msg: str) -> str:
    return f"{Colors.GREEN}✓ OK{Colors.RESET} {msg}"


def fail(msg: str, error: str = "") -> str:
    err_msg = f" ({error})" if error else ""
    return f"{Colors.RED}✗ FAIL{Colors.RESET} {msg}{err_msg}"


def warn(msg: str) -> str:
    return f"{Colors.YELLOW}⚠ WARN{Colors.RESET} {msg}"


def info(msg: str) -> str:
    return f"{Colors.BLUE}ℹ INFO{Colors.RESET} {msg}"


def section(title: str) -> str:
    return f"\n{Colors.BOLD}{Colors.CYAN}=== {title} ==={Colors.RESET}"


async def check_server_health(server_url: str) -> tuple[bool, dict]:
    """Check if the V2 server is healthy."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{server_url}/health")
            if resp.status_code == 200:
                data = resp.json()
                return True, data
            return False, {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


async def check_isolation(server_url: str) -> tuple[bool, dict]:
    """
    Check V2 isolation via /v2/isolation endpoint.

    CRITICAL: This MUST return isolated=True.
    Any violations mean OLD agents could pick up V2 campaigns.
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{server_url}/v2/isolation")
            if resp.status_code == 200:
                data = resp.json()
                isolated = data.get("isolated", False)
                return isolated, data
            return False, {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


async def check_pipeline(server_url: str) -> tuple[bool, dict]:
    """Check V2 pipeline status via /v2/pipeline endpoint."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{server_url}/v2/pipeline")
            if resp.status_code == 200:
                data = resp.json()
                return True, data
            return False, {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


async def check_v2_info(server_url: str) -> tuple[bool, dict]:
    """Get V2 orchestrator info."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{server_url}/v2/info")
            if resp.status_code == 200:
                data = resp.json()
                return True, data
            return False, {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


async def check_database_isolation() -> tuple[bool, dict]:
    """
    Direct database check for isolation violations.

    Checks for:
    - V2 processor with OLD status (processor='new' AND status IN ('new', 'in_scripting', ...))
    - OLD processor with V2 status (processor='old' AND status LIKE 'v2_%')
    - V2 status without processor set
    """
    try:
        from agents.shared.db import get_db
        db = get_db()

        query = """
            SELECT
                id,
                topic,
                status,
                processor,
                CASE
                    WHEN processor = 'new' AND status IN ('new', 'in_scripting', 'script_approved', 'generating_visuals', 'rendering')
                        THEN 'V2 processor with OLD status'
                    WHEN processor = 'old' AND status LIKE 'v2_%'
                        THEN 'OLD processor with V2 status'
                    WHEN status LIKE 'v2_%' AND processor IS NULL
                        THEN 'V2 status without processor'
                    ELSE 'unknown'
                END as violation_type
            FROM video_campaigns
            WHERE
                (processor = 'new' AND status IN ('new', 'in_scripting', 'script_approved', 'generating_visuals', 'rendering'))
                OR (processor = 'old' AND status LIKE 'v2_%%')
                OR (status LIKE 'v2_%%' AND processor IS NULL)
            LIMIT 10
        """

        violations = await db.fetch_all(query, [])

        return len(violations) == 0, {
            "isolated": len(violations) == 0,
            "violation_count": len(violations),
            "violations": violations[:5],  # Show first 5
            "checked_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return False, {"error": str(e), "database_available": False}


async def get_pipeline_stats() -> dict:
    """Get pipeline statistics from database."""
    try:
        from agents.shared.db import get_db
        db = get_db()

        # V2 campaigns
        v2_query = """
            SELECT
                status,
                COUNT(*) as count
            FROM video_campaigns
            WHERE processor = 'new' OR status LIKE 'v2_%%'
            GROUP BY status
            ORDER BY status
        """
        v2_stats = await db.fetch_all(v2_query, [])

        # OLD campaigns
        old_query = """
            SELECT
                status,
                COUNT(*) as count
            FROM video_campaigns
            WHERE processor = 'old' OR (processor IS NULL AND status NOT LIKE 'v2_%%')
            GROUP BY status
            ORDER BY status
        """
        old_stats = await db.fetch_all(old_query, [])

        return {
            "v2_pipeline": [{"status": r["status"], "count": r["count"]} for r in v2_stats],
            "old_pipeline": [{"status": r["status"], "count": r["count"]} for r in old_stats],
            "database_available": True,
        }
    except Exception as e:
        return {"error": str(e), "database_available": False}


def run_checks(
    server_url: str,
    check_db: bool = False,
    verbose: bool = True,
) -> bool:
    """Run all V2 isolation checks."""
    all_passed = True

    async def _run():
        nonlocal all_passed

        if verbose:
            print(section("V2 Isolation Monitor"))
            print(f"Server: {server_url}")
            print(f"Time: {datetime.utcnow().isoformat()}Z")

        # 1. Health check
        if verbose:
            print(section("Server Health"))
        healthy, health_data = await check_server_health(server_url)
        if healthy:
            if verbose:
                print(ok(f"Server healthy (mode={health_data.get('mode', 'unknown')})"))
        else:
            if verbose:
                print(fail("Server unhealthy", health_data.get("error", "")))
            all_passed = False
            return  # Can't continue without server

        # 2. Isolation check (CRITICAL)
        if verbose:
            print(section("V2 Isolation Check (CRITICAL)"))
        isolated, isolation_data = await check_isolation(server_url)
        if isolated:
            if verbose:
                print(ok("ISOLATION OK - No violations detected"))
                if isolation_data.get("database_enabled"):
                    print(ok("Database isolation enabled"))
                else:
                    print(warn("Database isolation disabled (in-memory only)"))
        else:
            if verbose:
                print(fail("ISOLATION VIOLATION DETECTED!", ""))
                if "violations" in isolation_data:
                    for v in isolation_data.get("violations", [])[:3]:
                        print(f"    Campaign {v.get('id')}: {v.get('violation_type')}")
            all_passed = False

        # 3. Pipeline status
        if verbose:
            print(section("V2 Pipeline Status"))
        pipe_ok, pipe_data = await check_pipeline(server_url)
        if pipe_ok:
            statuses = pipe_data.get("statuses", [])
            total = pipe_data.get("total_v2_campaigns", 0)
            if verbose:
                print(ok(f"Pipeline active ({total} V2 campaigns)"))
                for s in statuses:
                    print(f"    {s.get('status')}: {s.get('count')}")
        else:
            if verbose:
                print(warn("Could not get pipeline status", pipe_data.get("error", "")))

        # 4. V2 Info
        if verbose:
            print(section("V2 Orchestrator Info"))
        info_ok, info_data = await check_v2_info(server_url)
        if info_ok:
            if verbose:
                print(ok(f"Version: {info_data.get('version', 'unknown')}"))
                print(ok(f"Processor: {info_data.get('processor', 'unknown')}"))
                print(ok(f"Active campaigns: {info_data.get('active_campaigns', 0)}"))
        else:
            if verbose:
                print(warn("Could not get V2 info"))

        # 5. Database checks (optional)
        if check_db:
            if verbose:
                print(section("Database Isolation Check"))
            db_isolated, db_data = await check_database_isolation()
            if db_isolated:
                if verbose:
                    print(ok("Database isolation OK"))
            else:
                if db_data.get("database_available", True):
                    if verbose:
                        print(fail("Database isolation violation!", ""))
                        for v in db_data.get("violations", []):
                            print(f"    Campaign {v.get('id')}: {v.get('violation_type')}")
                    all_passed = False
                else:
                    if verbose:
                        print(warn("Database not available", db_data.get("error", "")))

            # Pipeline stats
            if verbose:
                print(section("Database Pipeline Stats"))
            stats = await get_pipeline_stats()
            if stats.get("database_available"):
                if verbose:
                    print(info("V2 Pipeline:"))
                    for s in stats.get("v2_pipeline", []):
                        print(f"    {s['status']}: {s['count']}")
                    print(info("OLD Pipeline:"))
                    for s in stats.get("old_pipeline", []):
                        print(f"    {s['status']}: {s['count']}")

        # Summary
        if verbose:
            print(section("Summary"))
        if all_passed:
            if verbose:
                print(f"{Colors.GREEN}{Colors.BOLD}All checks passed - V2 isolation OK{Colors.RESET}")
        else:
            if verbose:
                print(f"{Colors.RED}{Colors.BOLD}CHECKS FAILED - Investigate immediately!{Colors.RESET}")

    asyncio.run(_run())
    return all_passed


def run_continuous(server_url: str, interval: int = 60):
    """Run checks continuously with specified interval."""
    print(f"Starting continuous monitoring (interval: {interval}s)")
    print(f"Press Ctrl+C to stop\n")

    while True:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*60}")
        print(f"Check at {timestamp} UTC")
        print('='*60)

        passed = run_checks(server_url, check_db=True, verbose=True)

        if not passed:
            print(f"\n{Colors.RED}ALERT: Isolation check failed!{Colors.RESET}")
            # Could add alerting here (Slack, PagerDuty, etc.)

        time.sleep(interval)


def run_probe(server_url: str) -> int:
    """
    Run as a Kubernetes probe.

    Returns:
        0 if all checks pass (healthy)
        1 if any check fails (unhealthy)
    """
    passed = run_checks(server_url, check_db=False, verbose=False)

    if passed:
        print("OK")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor V2 orchestrator isolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic check
    python scripts/monitor_v2_isolation.py --server-url http://localhost:8765

    # With database verification
    python scripts/monitor_v2_isolation.py --server-url http://localhost:8765 --database

    # Continuous monitoring (every 60 seconds)
    python scripts/monitor_v2_isolation.py --server-url http://localhost:8765 --continuous

    # As K8s probe (exit 0/1)
    python scripts/monitor_v2_isolation.py --server-url http://localhost:8765 --probe
        """
    )
    parser.add_argument(
        "--server-url", "-s",
        default="http://localhost:8765",
        help="V2 orchestrator server URL"
    )
    parser.add_argument(
        "--database", "-d",
        action="store_true",
        help="Include direct database checks"
    )
    parser.add_argument(
        "--continuous", "-c",
        action="store_true",
        help="Run continuously"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=60,
        help="Interval between checks in continuous mode (seconds)"
    )
    parser.add_argument(
        "--probe", "-p",
        action="store_true",
        help="Run as K8s probe (minimal output, exit 0/1)"
    )

    args = parser.parse_args()

    if args.probe:
        sys.exit(run_probe(args.server_url))
    elif args.continuous:
        try:
            run_continuous(args.server_url, args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        passed = run_checks(args.server_url, check_db=args.database, verbose=True)
        sys.exit(0 if passed else 1)
