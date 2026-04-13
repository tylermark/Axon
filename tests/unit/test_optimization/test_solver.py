"""Unit tests for the top-level optimization orchestrator."""

from __future__ import annotations

import pytest

from src.optimization.solver import OptimizationConfig


# ---------------------------------------------------------------------------
# OptimizationConfig tests
# ---------------------------------------------------------------------------


class TestOptimizationConfig:
    """Test configuration defaults and validation."""

    def test_default_config(self) -> None:
        config = OptimizationConfig()
        assert config.solver_backend == "cpsat"
        assert config.cpsat_time_limit_seconds == 30.0
        assert config.sku_minimization_weight == 0.1
        assert config.cost_weight == 0.05
        assert config.max_solutions_per_wall == 5
        assert config.drl_fallback_threshold == 500

    def test_drl_backend(self) -> None:
        config = OptimizationConfig(solver_backend="drl")
        assert config.solver_backend == "drl"

    def test_custom_config(self) -> None:
        config = OptimizationConfig(
            cpsat_time_limit_seconds=60.0,
            sku_minimization_weight=0.5,
            drl_fallback_threshold=100,
        )
        assert config.cpsat_time_limit_seconds == 60.0
        assert config.sku_minimization_weight == 0.5
        assert config.drl_fallback_threshold == 100


# ---------------------------------------------------------------------------
# Result builder tests
# ---------------------------------------------------------------------------


class TestResultBuilder:
    """Test the shared PanelizationResult builder."""

    def test_empty_assignments(self) -> None:
        """Empty assignments should produce a valid result with 0 coverage."""
        from unittest.mock import MagicMock

        import numpy as np

        from src.optimization.result_builder import build_panelization_result

        # Minimal mock of ClassifiedWallGraph
        mock_graph = MagicMock()
        mock_graph.graph.wall_segments = []
        mock_graph.graph.rooms = []
        mock_graph.graph.nodes = np.zeros((0, 2))

        result = build_panelization_result(
            classified_graph=mock_graph,
            wall_assignments={},
            room_assignments={},
        )

        assert result.coverage_percentage == 0.0
        assert result.waste_percentage == 0.0
        assert result.total_panel_count == 0
        assert result.policy_version == "or_cpsat"

    def test_solver_name_propagated(self) -> None:
        from unittest.mock import MagicMock

        import numpy as np

        from src.optimization.result_builder import build_panelization_result

        mock_graph = MagicMock()
        mock_graph.graph.wall_segments = []
        mock_graph.graph.rooms = []
        mock_graph.graph.nodes = np.zeros((0, 2))

        result = build_panelization_result(
            classified_graph=mock_graph,
            wall_assignments={},
            room_assignments={},
            solver_name="test_solver_v2",
        )

        assert result.policy_version == "test_solver_v2"
