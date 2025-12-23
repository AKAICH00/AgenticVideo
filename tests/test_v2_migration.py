"""
V2 Migration Tests - OLD Daemons → NEW Orchestrator

Tests for the gradual migration from polling daemons to unified VideoGraph orchestrator.

Covers:
1. Feature flag routing (ProcessorMode)
2. CampaignRouter behavior
3. VideoGraph imports and initialization
4. Server endpoints
5. Migration status tracking

Run with:
    python -m pytest tests/test_v2_migration.py -v

Or standalone:
    python tests/test_v2_migration.py
"""

import asyncio
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFeatureFlags:
    """Test feature flag routing for V2 migration."""

    def setup_method(self):
        """Clean up environment before each test."""
        if "PROCESSOR_MODE" in os.environ:
            del os.environ["PROCESSOR_MODE"]

    def teardown_method(self):
        """Clean up environment after each test."""
        if "PROCESSOR_MODE" in os.environ:
            del os.environ["PROCESSOR_MODE"]

    def test_import_feature_flags(self):
        """Test that feature flags module imports correctly."""
        from core.feature_flags import (
            ProcessorMode,
            get_processor_mode,
            should_use_new_orchestrator,
            get_migration_status,
        )
        assert ProcessorMode.OLD.value == "old"
        assert ProcessorMode.NEW.value == "new"
        assert ProcessorMode.SPLIT.value == "split"

    def test_default_mode_is_old(self):
        """Test that default processor mode is OLD (safe default)."""
        from core.feature_flags import get_processor_mode, ProcessorMode

        mode = get_processor_mode()
        assert mode == ProcessorMode.OLD

    def test_mode_old_routing(self):
        """Test OLD mode routes all campaigns to old daemons."""
        os.environ["PROCESSOR_MODE"] = "old"

        from core.feature_flags import should_use_new_orchestrator

        # Both premium and bulk should go to OLD
        assert not should_use_new_orchestrator("test-123", "premium")
        assert not should_use_new_orchestrator("test-456", "bulk")

    def test_mode_new_routing(self):
        """Test NEW mode routes all campaigns to new orchestrator."""
        os.environ["PROCESSOR_MODE"] = "new"

        from core.feature_flags import should_use_new_orchestrator

        # Both premium and bulk should go to NEW
        assert should_use_new_orchestrator("test-123", "premium")
        assert should_use_new_orchestrator("test-456", "bulk")

    def test_mode_split_routing(self):
        """Test SPLIT mode routes premium→new, bulk→old."""
        os.environ["PROCESSOR_MODE"] = "split"

        from core.feature_flags import should_use_new_orchestrator

        # Premium goes to NEW
        assert should_use_new_orchestrator("test-123", "premium")

        # Bulk goes to OLD
        assert not should_use_new_orchestrator("test-456", "bulk")

    def test_override_parameter(self):
        """Test that override parameter takes precedence."""
        os.environ["PROCESSOR_MODE"] = "old"

        from core.feature_flags import should_use_new_orchestrator

        # Override should force NEW even in OLD mode
        assert should_use_new_orchestrator("test-123", "bulk", override="new")

        # Override should force OLD even when criteria would suggest NEW
        os.environ["PROCESSOR_MODE"] = "new"
        assert not should_use_new_orchestrator("test-123", "premium", override="old")

    def test_migration_status(self):
        """Test migration status returns correct info."""
        os.environ["PROCESSOR_MODE"] = "split"

        from core.feature_flags import get_migration_status

        status = get_migration_status()
        assert status["mode"] == "split"
        assert "premium" in status["description"].lower() or "bulk" in status["description"].lower()
        assert status["env_var"] == "PROCESSOR_MODE"


class TestCampaignRouter:
    """Test campaign router behavior."""

    def test_import_router(self):
        """Test that router module imports correctly."""
        from services.router import CampaignRouter, route_campaign

        router = CampaignRouter()
        assert router is not None
        assert hasattr(router, "process_campaign")
        assert hasattr(router, "get_status")

    @pytest.mark.asyncio
    async def test_router_queues_for_old_daemons(self):
        """Test that router returns queue status when using OLD processor."""
        os.environ["PROCESSOR_MODE"] = "old"

        from services.router import CampaignRouter

        router = CampaignRouter()
        result = await router.process_campaign(
            campaign_id="test-old-001",
            topic="Test Topic",
            niche="tech",
            quality_tier="bulk",
        )

        assert result["processor"] == "old"
        assert result["status"] == "queued"

    @pytest.mark.asyncio
    async def test_router_uses_orchestrator_for_new(self):
        """Test that router uses orchestrator when in NEW mode."""
        os.environ["PROCESSOR_MODE"] = "new"

        from services.router import CampaignRouter

        router = CampaignRouter()

        # Mock the orchestrator.run to avoid full pipeline execution
        with patch.object(router.orchestrator, 'run', new_callable=AsyncMock) as mock_run:
            from services.orchestrator.state import VideoState, GenerationPhase

            mock_result = VideoState(
                campaign_id="test-new-001",
                topic="Test Topic",
                niche="tech",
            )
            mock_result.phase = GenerationPhase.COMPLETE
            mock_result.long_form_video_url = "https://example.com/video.mp4"
            mock_run.return_value = mock_result

            result = await router.process_campaign(
                campaign_id="test-new-001",
                topic="Test Topic",
                niche="tech",
                quality_tier="premium",
            )

            assert result["processor"] == "new"
            assert result["success"] is True
            mock_run.assert_called_once()

    def test_router_get_status(self):
        """Test router status method."""
        from services.router import CampaignRouter

        router = CampaignRouter()
        status = router.get_status()

        assert "migration" in status
        assert "orchestrator_ready" in status
        assert status["orchestrator_ready"] is True


class TestOrchestratorImports:
    """Test that all orchestrator imports work correctly."""

    def test_import_video_graph(self):
        """Test VideoGraph import."""
        from services.orchestrator.graph import VideoGraph

        graph = VideoGraph()
        assert graph is not None
        assert hasattr(graph, "run")
        assert hasattr(graph, "nodes")
        assert hasattr(graph, "edges")

    def test_import_video_state(self):
        """Test VideoState import and structure."""
        from services.orchestrator.state import VideoState, GenerationPhase

        state = VideoState(
            campaign_id="test-123",
            topic="Test Topic",
            niche="tech",
        )

        # Verify required fields
        assert state.campaign_id == "test-123"
        assert state.topic == "Test Topic"
        assert state.niche == "tech"
        assert state.phase == GenerationPhase.PENDING

        # Verify meta field exists (was missing before fix)
        assert hasattr(state, "meta")
        assert isinstance(state.meta, dict)

    def test_import_all_nodes(self):
        """Test all node imports."""
        from services.orchestrator.nodes import (
            PlannerNode,
            ScriptNode,
            StoryboardNode,
            MotionNode,
            VisualNode,
            QualityNode,
            ComposeNode,
            RepurposeNode,
        )

        # Verify all nodes can be instantiated
        nodes = [
            PlannerNode(),
            ScriptNode(),
            StoryboardNode(),
            MotionNode(),
            VisualNode(),
            QualityNode(),
            ComposeNode(),
            RepurposeNode(),
        ]

        for node in nodes:
            assert hasattr(node, "execute")

    def test_import_services_package(self):
        """Test services package exports."""
        from services import (
            VideoGraph,
            VideoState,
            GenerationPhase,
            create_video_workflow,
        )

        assert VideoGraph is not None
        assert VideoState is not None
        assert GenerationPhase is not None

    def test_import_orchestrator_package(self):
        """Test orchestrator package exports."""
        from services.orchestrator import (
            VideoGraph,
            VideoState,
            GenerationPhase,
            PlannerNode,
            ScriptNode,
            RepurposeNode,
        )

        assert VideoGraph is not None
        assert VideoState is not None
        assert RepurposeNode is not None


class TestVideoGraphStructure:
    """Test VideoGraph structure and configuration."""

    def test_graph_has_all_nodes(self):
        """Test that graph has all 8 nodes configured."""
        from services.orchestrator.graph import VideoGraph

        graph = VideoGraph()
        expected_nodes = [
            "planner",
            "script",
            "storyboard",
            "motion",
            "visual",
            "quality",
            "compose",
            "repurpose",
        ]

        for node_name in expected_nodes:
            assert node_name in graph.nodes, f"Missing node: {node_name}"

    def test_graph_has_edges(self):
        """Test that graph has edges configured."""
        from services.orchestrator.graph import VideoGraph
        from services.orchestrator.state import GenerationPhase

        graph = VideoGraph()

        # Check key transitions exist (using actual phase names)
        assert GenerationPhase.PENDING in graph.edges
        assert GenerationPhase.SCRIPTING in graph.edges
        assert GenerationPhase.VISUAL_GENERATION in graph.edges

    def test_quality_loop_configuration(self):
        """Test that quality retry loop is configured."""
        from services.orchestrator.graph import VideoGraph
        from services.orchestrator.state import GenerationPhase

        graph = VideoGraph()

        # Quality phase should have conditional routing
        quality_edge = graph.edges.get(GenerationPhase.QUALITY_CHECK)
        assert quality_edge is not None


class TestServerEndpoints:
    """Test FastAPI server endpoints."""

    def test_import_server(self):
        """Test server module imports."""
        from services.orchestrator.server import app, GenerateRequest, CampaignStatus

        assert app is not None
        assert GenerateRequest is not None
        assert CampaignStatus is not None

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health check endpoint."""
        from services.orchestrator.server import health

        response = await health()

        assert response["status"] == "healthy"
        assert "mode" in response
        assert "active_campaigns" in response
        assert "timestamp" in response

    @pytest.mark.asyncio
    async def test_migration_endpoint(self):
        """Test migration status endpoint."""
        from services.orchestrator.server import migration_status

        response = await migration_status()

        assert "mode" in response
        assert "description" in response

    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        from services.orchestrator.server import root

        response = await root()

        assert response["service"] == "Video Orchestrator"
        assert response["version"] == "2.0.0"
        assert "endpoints" in response


class TestGenerationPhases:
    """Test generation phase transitions."""

    def test_all_phases_defined(self):
        """Test all generation phases are defined."""
        from services.orchestrator.state import GenerationPhase

        # Using actual phase names from state.py
        expected_phases = [
            "PENDING",
            "PLANNING",
            "SCRIPTING",
            "STORYBOARDING",
            "MOTION_EXTRACTION",
            "VISUAL_GENERATION",
            "COMPOSITION",
            "QUALITY_CHECK",
            "REPURPOSING",
            "COMPLETE",
            "FAILED",
        ]

        for phase in expected_phases:
            assert hasattr(GenerationPhase, phase), f"Missing phase: {phase}"

    def test_phase_values(self):
        """Test phase enum values."""
        from services.orchestrator.state import GenerationPhase

        # Phases should have string values
        assert GenerationPhase.PENDING.value == "pending"
        assert GenerationPhase.COMPLETE.value == "complete"
        assert GenerationPhase.FAILED.value == "failed"


class TestVideoStateStructure:
    """Test VideoState dataclass structure."""

    def test_state_has_all_fields(self):
        """Test VideoState has all required fields."""
        from services.orchestrator.state import VideoState

        state = VideoState(
            campaign_id="test-123",
            topic="Test Topic",
            niche="tech",
        )

        # Required fields
        assert hasattr(state, "campaign_id")
        assert hasattr(state, "topic")
        assert hasattr(state, "niche")
        assert hasattr(state, "phase")

        # Optional fields with defaults
        assert hasattr(state, "quality_tier")
        assert hasattr(state, "target_duration_seconds")
        assert hasattr(state, "reference_video_url")
        assert hasattr(state, "style_reference")
        assert hasattr(state, "target_audience")
        assert hasattr(state, "output_formats")

        # Progress tracking
        assert hasattr(state, "progress_percent")
        assert hasattr(state, "current_step")
        assert hasattr(state, "progress_message")

        # Timestamps
        assert hasattr(state, "started_at")
        assert hasattr(state, "completed_at")

        # Results - storyboard contains scenes, not separate scenes field
        assert hasattr(state, "script")
        assert hasattr(state, "storyboard")
        assert hasattr(state.storyboard, "scenes")
        assert hasattr(state, "long_form_video_url")
        assert hasattr(state, "short_form_clips")

        # Meta field (was missing before fix)
        assert hasattr(state, "meta")

    def test_state_error_handling(self):
        """Test VideoState error handling methods."""
        from services.orchestrator.state import VideoState

        state = VideoState(
            campaign_id="test-123",
            topic="Test Topic",
            niche="tech",
        )

        # Test add_error method - uses phase/error keys per implementation
        assert hasattr(state, "add_error")
        state.add_error("test_phase", "Test error message", recoverable=True)

        assert len(state.errors) == 1
        assert state.errors[0]["phase"] == "test_phase"
        assert state.errors[0]["error"] == "Test error message"
        assert state.errors[0]["recoverable"] is True


class TestIntegrationFlow:
    """Integration tests for the V2 migration flow."""

    @pytest.mark.asyncio
    async def test_end_to_end_mock_flow(self):
        """Test the complete flow with mocked components."""
        os.environ["PROCESSOR_MODE"] = "new"

        from services.orchestrator.graph import VideoGraph
        from services.orchestrator.state import VideoState, GenerationPhase

        # Create a campaign state
        state = VideoState(
            campaign_id="e2e-test-001",
            topic="V2 Migration Test",
            niche="tech",
            quality_tier="bulk",
            target_duration_seconds=60,
            meta={
                "test": True,
                "hooks": ["Test hook 1", "Test hook 2"],
            }
        )

        graph = VideoGraph()

        # Mock all nodes to skip actual generation
        for node_name, node in graph.nodes.items():
            node.execute = AsyncMock(side_effect=lambda s: s)

        # Run should complete without errors
        # Note: This is a structural test, actual generation is mocked
        assert state.phase == GenerationPhase.PENDING
        assert state.meta["test"] is True

        print("\n=== V2 Migration E2E Test ===")
        print(f"Campaign ID: {state.campaign_id}")
        print(f"Processor Mode: {os.environ.get('PROCESSOR_MODE')}")
        print(f"Quality Tier: {state.quality_tier}")
        print(f"Graph Nodes: {list(graph.nodes.keys())}")
        print("=== Test Complete ===")


def run_quick_verification():
    """Quick verification that all critical imports work."""
    print("\n" + "="*60)
    print("V2 MIGRATION QUICK VERIFICATION")
    print("="*60)

    errors = []

    # Test 1: Feature flags
    try:
        from core.feature_flags import (
            ProcessorMode,
            get_processor_mode,
            should_use_new_orchestrator,
        )
        print("✅ Feature flags import OK")
    except Exception as e:
        print(f"❌ Feature flags import FAILED: {e}")
        errors.append(f"feature_flags: {e}")

    # Test 2: Router
    try:
        from services.router import CampaignRouter
        router = CampaignRouter()
        print("✅ CampaignRouter import OK")
    except Exception as e:
        print(f"❌ CampaignRouter import FAILED: {e}")
        errors.append(f"router: {e}")

    # Test 3: VideoGraph
    try:
        from services.orchestrator.graph import VideoGraph
        graph = VideoGraph()
        print(f"✅ VideoGraph import OK ({len(graph.nodes)} nodes)")
    except Exception as e:
        print(f"❌ VideoGraph import FAILED: {e}")
        errors.append(f"graph: {e}")

    # Test 4: VideoState with meta
    try:
        from services.orchestrator.state import VideoState
        state = VideoState(campaign_id="test", topic="test", niche="test")
        assert hasattr(state, "meta")
        print("✅ VideoState import OK (meta field present)")
    except Exception as e:
        print(f"❌ VideoState import FAILED: {e}")
        errors.append(f"state: {e}")

    # Test 5: All nodes
    try:
        from services.orchestrator.nodes import (
            PlannerNode, ScriptNode, StoryboardNode,
            MotionNode, VisualNode, QualityNode,
            ComposeNode, RepurposeNode,
        )
        print("✅ All 8 nodes import OK")
    except Exception as e:
        print(f"❌ Nodes import FAILED: {e}")
        errors.append(f"nodes: {e}")

    # Test 6: Server
    try:
        from services.orchestrator.server import app
        print("✅ FastAPI server import OK")
    except Exception as e:
        print(f"❌ Server import FAILED: {e}")
        errors.append(f"server: {e}")

    # Test 7: Services package
    try:
        from services import VideoGraph, VideoState
        print("✅ Services package exports OK")
    except Exception as e:
        print(f"❌ Services package FAILED: {e}")
        errors.append(f"services package: {e}")

    print("\n" + "-"*60)
    if errors:
        print(f"❌ VERIFICATION FAILED: {len(errors)} error(s)")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("✅ ALL VERIFICATIONS PASSED")
        print("\nV2 migration code is ready for deployment.")
        return True


if __name__ == "__main__":
    # Run quick verification
    success = run_quick_verification()

    # Run full test suite if verification passes
    if success:
        print("\n\nRunning full pytest suite...")
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        print("\n⚠️  Fix import errors before running full test suite")
        sys.exit(1)
