"""Unit tests for the global CP-SAT coordination solver."""

from __future__ import annotations

import pytest

from src.knowledge_graph.schema import Panel, PanelType
from src.optimization.cutting_stock import CuttingStockSolution, WallCuttingResult
from src.optimization.global_cpsat import GlobalSolution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_panel(
    sku: str = "PNL-LB-16-6",
    gauge: int = 16,
    stud_depth: float = 6.0,
    panel_type: PanelType = PanelType.LOAD_BEARING,
) -> Panel:
    return Panel(
        sku=sku,
        name="Test Panel",
        panel_type=panel_type,
        gauge=gauge,
        stud_depth_inches=stud_depth,
        stud_spacing_inches=16.0,
        min_length_inches=24.0,
        max_length_inches=240.0,
        height_inches=96.0,
        fire_rating_hours=0.0,
        load_capacity_plf=2100.0,
        sheathing_type=None,
        sheathing_thickness_inches=None,
        insulation_type=None,
        insulation_r_value=None,
        weight_per_foot_lbs=7.2,
        unit_cost_per_foot=14.5,
        compatible_connections=[],
        fabricated_by=[],
    )


def _make_solution(
    edge_id: int,
    sku: str = "PNL-LB-16-6",
    gauge: int = 16,
    stud_depth: float = 6.0,
    waste: float = 2.0,
    cost: float = 50.0,
    cut_length: float = 120.0,
) -> CuttingStockSolution:
    panel = _make_panel(sku=sku, gauge=gauge, stud_depth=stud_depth)
    return CuttingStockSolution(
        wall_edge_id=edge_id,
        panel_sku=sku,
        panel=panel,
        assignments=[(sku, cut_length)],
        waste_inches=waste,
        waste_percentage=waste / cut_length if cut_length > 0 else 0.0,
        total_cost=cost,
        requires_splice=False,
        num_pieces=1,
        gauge=gauge,
        stud_depth_inches=stud_depth,
        score=0.8,
    )


def _make_wall_result(
    edge_id: int,
    solutions: list[CuttingStockSolution],
    length: float = 120.0,
) -> WallCuttingResult:
    return WallCuttingResult(
        wall_edge_id=edge_id,
        wall_length_inches=length,
        effective_length_inches=length,
        panel_type=PanelType.LOAD_BEARING,
        solutions=solutions,
        is_panelizable=bool(solutions),
    )


# ---------------------------------------------------------------------------
# GlobalSolution dataclass tests
# ---------------------------------------------------------------------------


class TestGlobalSolution:
    """Test GlobalSolution dataclass defaults."""

    def test_empty_solution(self) -> None:
        sol = GlobalSolution(
            wall_assignments={},
            room_assignments={},
            room_orientations={},
            solver_status="INFEASIBLE",
        )
        assert sol.solver_status == "INFEASIBLE"
        assert sol.total_waste_inches == 0.0
        assert sol.num_distinct_panel_skus == 0

    def test_populated_solution(self) -> None:
        sol = GlobalSolution(
            wall_assignments={0: [("SKU-A", 100.0)], 1: [("SKU-B", 96.0)]},
            room_assignments={0: "POD-BATH-001"},
            room_orientations={0: False},
            total_waste_inches=5.0,
            total_cost=250.0,
            num_distinct_panel_skus=2,
            num_distinct_pod_skus=1,
            solver_status="OPTIMAL",
            solve_time_seconds=0.5,
        )
        assert len(sol.wall_assignments) == 2
        assert sol.room_assignments[0] == "POD-BATH-001"
        assert not sol.room_orientations[0]


# ---------------------------------------------------------------------------
# WallCuttingResult interaction tests
# ---------------------------------------------------------------------------


class TestWallResultPreparation:
    """Test that wall results are correctly structured for CP-SAT."""

    def test_single_wall_single_solution(self) -> None:
        sol = _make_solution(edge_id=0)
        wr = _make_wall_result(edge_id=0, solutions=[sol])
        assert wr.is_panelizable
        assert len(wr.solutions) == 1
        assert wr.solutions[0].gauge == 16

    def test_wall_with_multiple_solutions(self) -> None:
        sol_a = _make_solution(edge_id=0, sku="A", gauge=16, waste=2.0)
        sol_b = _make_solution(edge_id=0, sku="B", gauge=18, waste=1.0)
        wr = _make_wall_result(edge_id=0, solutions=[sol_a, sol_b])
        assert len(wr.solutions) == 2
        # Different gauges
        assert wr.solutions[0].gauge != wr.solutions[1].gauge

    def test_unpanelizable_wall(self) -> None:
        wr = _make_wall_result(edge_id=0, solutions=[])
        assert not wr.is_panelizable

    def test_junction_gauge_compatibility_data(self) -> None:
        """Verify that solutions carry gauge info needed for junction constraints."""
        sol_16 = _make_solution(edge_id=0, gauge=16, stud_depth=6.0)
        sol_18 = _make_solution(edge_id=1, gauge=18, stud_depth=6.0)
        # These should be flagged as incompatible at a shared junction
        assert sol_16.gauge != sol_18.gauge
        assert sol_16.stud_depth_inches == sol_18.stud_depth_inches
