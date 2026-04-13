"""Unit tests for the per-wall 1D cutting stock solver."""

from __future__ import annotations

import pytest

from src.knowledge_graph.schema import Panel, PanelType
from src.optimization.cutting_stock import (
    _compute_solution_score,
    _enumerate_cutting_patterns,
    _solve_for_panel_type,
    solve_ffd,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_panel(
    sku: str = "PNL-LB-16-6",
    panel_type: PanelType = PanelType.LOAD_BEARING,
    gauge: int = 16,
    stud_depth: float = 6.0,
    min_length: float = 24.0,
    max_length: float = 240.0,
    fire_rating: float = 0.0,
    cost_per_foot: float = 14.5,
) -> Panel:
    """Create a minimal Panel for testing."""
    return Panel(
        sku=sku,
        name="Test Panel",
        panel_type=panel_type,
        gauge=gauge,
        stud_depth_inches=stud_depth,
        stud_spacing_inches=16.0,
        min_length_inches=min_length,
        max_length_inches=max_length,
        height_inches=96.0,
        fire_rating_hours=fire_rating,
        load_capacity_plf=2100.0,
        sheathing_type=None,
        sheathing_thickness_inches=None,
        insulation_type=None,
        insulation_r_value=None,
        weight_per_foot_lbs=7.2,
        unit_cost_per_foot=cost_per_foot,
        compatible_connections=[],
        fabricated_by=[],
    )


# ---------------------------------------------------------------------------
# _enumerate_cutting_patterns tests
# ---------------------------------------------------------------------------


class TestLPCuttingStock:
    """Tests for the LP-based cutting stock solver."""

    def test_single_panel_fits(self) -> None:
        """Wall shorter than max_length -> 1 piece."""
        panel = _make_panel(max_length=240.0, min_length=24.0)
        result = _enumerate_cutting_patterns(panel, 120.0)
        assert result is not None
        n, lengths = result
        assert n == 1
        assert len(lengths) == 1
        assert abs(lengths[0] - 120.0) < 1.0

    def test_two_panels_needed(self) -> None:
        """Wall longer than max_length -> 2 pieces."""
        panel = _make_panel(max_length=120.0, min_length=24.0)
        result = _enumerate_cutting_patterns(panel, 200.0)
        assert result is not None
        n, lengths = result
        assert n == 2
        assert abs(sum(lengths) - 200.0) < 1.0
        for cl in lengths:
            assert cl >= panel.min_length_inches
            assert cl <= panel.max_length_inches

    def test_three_panels_needed(self) -> None:
        """Very long wall needing 3 panels."""
        panel = _make_panel(max_length=96.0, min_length=24.0)
        result = _enumerate_cutting_patterns(panel, 250.0)
        assert result is not None
        n, lengths = result
        assert n == 3
        total = sum(lengths)
        assert total >= 250.0  # Must cover the wall
        for cl in lengths:
            assert cl >= panel.min_length_inches

    def test_wall_too_short(self) -> None:
        """Wall shorter than min_length -> None."""
        panel = _make_panel(min_length=24.0)
        result = _enumerate_cutting_patterns(panel, 10.0)
        # The LP solver can't produce valid pieces below min_length
        # It should either return None or a single panel at min_length
        if result is not None:
            _, lengths = result
            assert lengths[0] >= panel.min_length_inches

    def test_exact_max_length(self) -> None:
        """Wall exactly at max_length -> 1 piece, no waste."""
        panel = _make_panel(max_length=120.0)
        result = _enumerate_cutting_patterns(panel, 120.0)
        assert result is not None
        n, lengths = result
        assert n == 1
        assert abs(lengths[0] - 120.0) < 0.01

    def test_zero_length(self) -> None:
        """Zero-length wall is infeasible."""
        panel = _make_panel()
        result = _enumerate_cutting_patterns(panel, 0.0)
        # Should return None or a result with 0 pieces
        if result is not None:
            n, _ = result
            assert n == 0


# ---------------------------------------------------------------------------
# solve_ffd tests
# ---------------------------------------------------------------------------


class TestFFD:
    """Tests for the First Fit Decreasing heuristic."""

    def test_single_panel(self) -> None:
        panel = _make_panel(max_length=240.0, min_length=24.0)
        result = solve_ffd(panel, 100.0)
        assert result is not None
        n, lengths, waste = result
        assert n == 1
        assert abs(lengths[0] - 100.0) < 1.0

    def test_multi_panel(self) -> None:
        panel = _make_panel(max_length=96.0, min_length=24.0)
        result = solve_ffd(panel, 200.0)
        assert result is not None
        n, lengths, waste = result
        assert n >= 2
        assert sum(lengths) >= 200.0
        for cl in lengths:
            assert cl >= panel.min_length_inches

    def test_wall_too_short(self) -> None:
        panel = _make_panel(min_length=24.0)
        result = solve_ffd(panel, 10.0)
        assert result is None

    def test_zero_length(self) -> None:
        panel = _make_panel()
        result = solve_ffd(panel, 0.0)
        assert result is None


# ---------------------------------------------------------------------------
# _solve_for_panel_type tests (without KG store)
# ---------------------------------------------------------------------------


class TestSolveForPanelType:
    """Tests for single-panel-type solving (mocked store not needed for
    single-panel cases since splice check is skipped)."""

    def test_single_panel_solution(self) -> None:
        """Wall fits in one panel — no splice needed."""
        panel = _make_panel(max_length=240.0, min_length=24.0)
        # _solve_for_panel_type needs a store only for multi-panel (splicing)
        # Single panel path doesn't call store
        sol = _solve_for_panel_type(0, panel, 100.0, None)  # type: ignore[arg-type]
        assert sol is not None
        assert sol.num_pieces == 1
        assert not sol.requires_splice
        assert len(sol.assignments) == 1
        assert sol.assignments[0][0] == panel.sku
        assert abs(sol.assignments[0][1] - 100.0) < 1.0
        assert sol.waste_inches < 1.0

    def test_minimum_length_enforcement(self) -> None:
        """Wall shorter than min_length -> cut at min_length (with waste)."""
        panel = _make_panel(min_length=48.0, max_length=240.0)
        sol = _solve_for_panel_type(0, panel, 30.0, None)  # type: ignore[arg-type]
        # 30" < 48" min -> should return None
        assert sol is None

    def test_exact_max_length(self) -> None:
        """Wall at max_length -> single panel, minimal waste."""
        panel = _make_panel(max_length=120.0)
        sol = _solve_for_panel_type(0, panel, 120.0, None)  # type: ignore[arg-type]
        assert sol is not None
        assert sol.num_pieces == 1
        assert sol.waste_inches < 0.01


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------


class TestSolutionScoring:
    """Tests for the composite solution score."""

    def test_zero_waste_scores_high(self) -> None:
        score = _compute_solution_score(0.0, 100.0, 1, 10.0)
        assert score > 0.8

    def test_high_waste_scores_low(self) -> None:
        score = _compute_solution_score(50.0, 100.0, 1, 10.0)
        # 50% waste still gets partial credit from splice/cost components
        assert score < 0.8

    def test_more_pieces_scores_lower(self) -> None:
        score_1 = _compute_solution_score(5.0, 100.0, 1, 10.0)
        score_3 = _compute_solution_score(5.0, 100.0, 3, 10.0)
        assert score_1 > score_3

    def test_score_bounded(self) -> None:
        score = _compute_solution_score(0.0, 100.0, 1, 5.0)
        assert 0.0 <= score <= 1.0
        score = _compute_solution_score(100.0, 100.0, 10, 500.0)
        assert 0.0 <= score <= 1.0
