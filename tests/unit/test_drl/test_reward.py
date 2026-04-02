"""Unit tests for DRL reward function (DRL-004)."""

from __future__ import annotations

import numpy as np
import pytest

from src.drl.actions import PanelAction, PlacementAction
from src.drl.reward import (
    RewardBreakdown,
    RewardWeights,
    _compute_coverage_bonus,
    _compute_spur,
    _compute_waste_penalty,
    _detect_pod_violations,
    compute_reward,
)
from src.knowledge_graph.loader import load_knowledge_graph
from src.knowledge_graph.query import PanelRecommendation
from src.knowledge_graph.schema import Panel, PanelType, Pod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_panel(
    sku: str = "TEST-PNL-001",
    panel_type: PanelType = PanelType.LOAD_BEARING,
) -> Panel:
    """Create a minimal Panel for testing."""
    return Panel(
        sku=sku,
        name="Test Panel",
        panel_type=panel_type,
        gauge=16,
        stud_depth_inches=6.0,
        stud_spacing_inches=16.0,
        min_length_inches=24.0,
        max_length_inches=300.0,
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


def _make_pod(
    sku: str = "TEST-POD-001",
    width: float = 60.0,
    depth: float = 96.0,
    clearance: float = 3.0,
) -> Pod:
    """Create a minimal Pod for testing."""
    return Pod(
        sku=sku,
        name="Test Pod",
        pod_type="bathroom",
        width_inches=width,
        depth_inches=depth,
        height_inches=96.0,
        min_room_width_inches=width + 2 * clearance,
        min_room_depth_inches=depth + 2 * clearance,
        clearance_inches=clearance,
        included_trades=["plumbing", "electrical"],
        connection_type="clip_angle",
        weight_lbs=1800.0,
        unit_cost=12500.0,
        lead_time_days=21,
        compatible_panel_types=[PanelType.LOAD_BEARING, PanelType.PARTITION],
    )


def _make_recommendation(
    panel: Panel | None = None,
    cut_lengths: list[float] | None = None,
    waste: float = 0.0,
) -> PanelRecommendation:
    """Create a PanelRecommendation for testing."""
    if panel is None:
        panel = _make_panel()
    if cut_lengths is None:
        cut_lengths = [96.0]
    total_material = sum(cut_lengths)
    return PanelRecommendation(
        panel=panel,
        quantity=len(cut_lengths),
        cut_lengths_inches=cut_lengths,
        requires_splice=len(cut_lengths) > 1,
        splice_connections=[],
        total_material_cost=100.0,
        waste_inches=waste,
        waste_percentage=(waste / total_material * 100.0) if total_material > 0 else 0.0,
        score=0.85,
    )


def _make_panel_action(
    skip: bool = False,
    panel: Panel | None = None,
    cut_lengths: list[float] | None = None,
    waste: float = 0.0,
    edge_id: int = 0,
) -> PanelAction:
    """Create a PanelAction for testing."""
    if skip:
        return PanelAction(wall_edge_id=edge_id, skip=True)

    if panel is None:
        panel = _make_panel()
    rec = _make_recommendation(panel=panel, cut_lengths=cut_lengths, waste=waste)
    assignments = [(panel.sku, cl) for cl in rec.cut_lengths_inches]
    return PanelAction(
        wall_edge_id=edge_id,
        skip=False,
        panel=panel,
        recommendation=rec,
        panel_assignments=assignments,
    )


def _make_placement_action(
    skip: bool = False,
    pod: Pod | None = None,
    rotated: bool = False,
    room_id: int = 0,
) -> PlacementAction:
    """Create a PlacementAction for testing."""
    if skip:
        return PlacementAction(room_id=room_id, skip=True)
    if pod is None:
        pod = _make_pod()
    return PlacementAction(
        room_id=room_id,
        skip=False,
        pod=pod,
        rotated=rotated,
        position_x=50.0,
        position_y=75.0,
    )


@pytest.fixture(scope="module")
def kg_store():
    """Load the full KG once for all tests in this module."""
    return load_knowledge_graph()


# ---------------------------------------------------------------------------
# _compute_spur
# ---------------------------------------------------------------------------


class TestComputeSpur:
    """Tests for SPUR (Standard Panel Utilization Rate) scoring."""

    def test_skip_action_returns_zero(self):
        action = _make_panel_action(skip=True)
        assert _compute_spur(action) == 0.0

    def test_exact_standard_length_returns_one(self):
        """96 inches (8 ft) is a standard length; should get SPUR = 1.0."""
        action = _make_panel_action(cut_lengths=[96.0])
        spur = _compute_spur(action)
        assert spur == 1.0

    def test_standard_lengths_all_score_one(self):
        """All standard lengths should score exactly 1.0."""
        standard_lengths = [48.0, 72.0, 96.0, 120.0, 144.0, 192.0, 240.0]
        for std in standard_lengths:
            action = _make_panel_action(cut_lengths=[std])
            spur = _compute_spur(action)
            assert spur == 1.0, f"Standard length {std} did not score 1.0"

    def test_near_standard_length_within_tolerance(self):
        """96.4 inches is within 0.5 inch tolerance of 96; should score 1.0."""
        action = _make_panel_action(cut_lengths=[96.4])
        spur = _compute_spur(action)
        assert spur == 1.0

    def test_non_standard_length_scores_less(self):
        """A panel far from any standard length should score less than 1.0."""
        action = _make_panel_action(cut_lengths=[60.0])  # between 48 and 72
        spur = _compute_spur(action)
        assert 0.0 < spur < 1.0

    def test_multiple_pieces_averaged(self):
        """SPUR should be averaged across all pieces."""
        # One standard (96) and one non-standard (60)
        action = _make_panel_action(cut_lengths=[96.0, 60.0])
        spur = _compute_spur(action)
        # Standard piece contributes 1.0, non-standard less
        # Average should be between the two
        assert 0.5 < spur < 1.0

    def test_spur_in_valid_range(self):
        for lengths in [[50.0], [96.0], [150.0], [96.0, 96.0], [30.0]]:
            action = _make_panel_action(cut_lengths=lengths)
            spur = _compute_spur(action)
            assert 0.0 <= spur <= 1.0

    def test_no_recommendation_returns_zero(self):
        action = PanelAction(wall_edge_id=0, skip=False, panel=_make_panel())
        assert _compute_spur(action) == 0.0


# ---------------------------------------------------------------------------
# _compute_waste_penalty
# ---------------------------------------------------------------------------


class TestComputeWastePenalty:
    """Tests for waste penalty computation."""

    def test_skip_action_returns_zero(self):
        action = _make_panel_action(skip=True)
        assert _compute_waste_penalty(action) == 0.0

    def test_zero_waste_returns_zero(self):
        action = _make_panel_action(cut_lengths=[96.0], waste=0.0)
        assert _compute_waste_penalty(action) == 0.0

    def test_positive_waste_returns_negative(self):
        action = _make_panel_action(cut_lengths=[96.0], waste=10.0)
        penalty = _compute_waste_penalty(action)
        assert penalty < 0.0

    def test_higher_waste_more_penalty(self):
        action_low = _make_panel_action(cut_lengths=[96.0], waste=2.0)
        action_high = _make_panel_action(cut_lengths=[96.0], waste=20.0)
        penalty_low = _compute_waste_penalty(action_low)
        penalty_high = _compute_waste_penalty(action_high)
        assert penalty_high < penalty_low  # more negative

    def test_penalty_bounded(self):
        """Waste penalty should be in [-1.0, 0.0]."""
        action = _make_panel_action(cut_lengths=[96.0], waste=500.0)
        penalty = _compute_waste_penalty(action)
        assert -1.0 <= penalty <= 0.0


# ---------------------------------------------------------------------------
# _detect_pod_violations
# ---------------------------------------------------------------------------


class TestDetectPodViolations:
    """Tests for pod placement violation detection."""

    def test_skip_no_violations(self):
        action = _make_placement_action(skip=True)
        violations = _detect_pod_violations(action, 100.0, 100.0)
        assert violations == []

    def test_pod_fits_no_violations(self):
        pod = _make_pod(width=60.0, depth=96.0, clearance=3.0)
        action = _make_placement_action(pod=pod)
        violations = _detect_pod_violations(action, 200.0, 200.0)
        assert violations == []

    def test_pod_too_wide(self):
        pod = _make_pod(width=60.0, depth=96.0, clearance=3.0)
        action = _make_placement_action(pod=pod)
        # Room width 50 < pod width (60) + 2*clearance (6) = 66
        violations = _detect_pod_violations(action, 50.0, 200.0)
        assert len(violations) == 1
        assert "width" in violations[0].lower()

    def test_pod_too_deep(self):
        pod = _make_pod(width=60.0, depth=96.0, clearance=3.0)
        action = _make_placement_action(pod=pod)
        # Room depth 80 < pod depth (96) + 2*clearance (6) = 102
        violations = _detect_pod_violations(action, 200.0, 80.0)
        assert len(violations) == 1
        assert "depth" in violations[0].lower()

    def test_pod_too_wide_and_deep(self):
        pod = _make_pod(width=60.0, depth=96.0, clearance=3.0)
        action = _make_placement_action(pod=pod)
        violations = _detect_pod_violations(action, 50.0, 80.0)
        assert len(violations) == 2

    def test_rotated_pod_swaps_dimensions(self):
        pod = _make_pod(width=60.0, depth=96.0, clearance=3.0)
        action = _make_placement_action(pod=pod, rotated=True)
        # Rotated: pod_w = depth + 2*clearance = 102, pod_d = width + 2*clearance = 66
        # Room 110 x 70: width=110 >= 102 OK, depth=70 >= 66 OK
        violations = _detect_pod_violations(action, 110.0, 70.0)
        assert violations == []

    def test_no_pod_reports_violation(self):
        action = PlacementAction(room_id=0, skip=False, pod=None)
        violations = _detect_pod_violations(action, 100.0, 100.0)
        assert len(violations) == 1
        assert "no pod" in violations[0].lower()


# ---------------------------------------------------------------------------
# _compute_coverage_bonus
# ---------------------------------------------------------------------------


class TestComputeCoverageBonus:
    """Tests for incremental coverage bonus."""

    def test_zero_coverage(self):
        bonus = _compute_coverage_bonus(0, 10, 0, 5, is_terminal=False)
        assert bonus == 0.0

    def test_partial_coverage(self):
        bonus = _compute_coverage_bonus(5, 10, 2, 4, is_terminal=False)
        expected = (5 / 10 + 2 / 4) / 2.0  # (0.5 + 0.5) / 2 = 0.5
        assert abs(bonus - expected) < 1e-5

    def test_full_coverage(self):
        bonus = _compute_coverage_bonus(10, 10, 5, 5, is_terminal=False)
        expected = (1.0 + 1.0) / 2.0
        assert abs(bonus - expected) < 1e-5

    def test_no_walls_no_rooms(self):
        bonus = _compute_coverage_bonus(0, 0, 0, 0, is_terminal=False)
        assert bonus == 0.0


# ---------------------------------------------------------------------------
# compute_reward (integration)
# ---------------------------------------------------------------------------


class TestComputeReward:
    """Tests for the combined compute_reward function."""

    def test_skip_panel_action_neutral(self, kg_store):
        """Skipping a wall should give near-zero reward (no SPUR, no waste)."""
        action = _make_panel_action(skip=True)
        breakdown = compute_reward(
            panel_action=action,
            placement_action=None,
            wall_length_inches=100.0,
            wall_panel_type=PanelType.LOAD_BEARING,
            room_width_inches=0.0,
            room_depth_inches=0.0,
            store=kg_store,
            walls_assigned=0,
            total_walls=4,
            rooms_assigned=0,
            total_rooms=2,
            is_terminal=False,
        )
        assert isinstance(breakdown, RewardBreakdown)
        assert breakdown.spur == 0.0
        assert breakdown.waste == 0.0

    def test_returns_reward_breakdown(self, kg_store):
        action = _make_panel_action(cut_lengths=[96.0])
        breakdown = compute_reward(
            panel_action=action,
            placement_action=None,
            wall_length_inches=96.0,
            wall_panel_type=PanelType.LOAD_BEARING,
            room_width_inches=0.0,
            room_depth_inches=0.0,
            store=kg_store,
            walls_assigned=1,
            total_walls=4,
            rooms_assigned=0,
            total_rooms=2,
            is_terminal=False,
        )
        assert isinstance(breakdown, RewardBreakdown)

    def test_total_is_weighted_sum(self, kg_store):
        """Total should be the weighted sum of components."""
        weights = RewardWeights(spur=1.0, waste=0.5, violation=3.0, coverage=0.3)
        action = _make_panel_action(skip=True)
        breakdown = compute_reward(
            panel_action=action,
            placement_action=None,
            wall_length_inches=0.0,
            wall_panel_type=PanelType.LOAD_BEARING,
            room_width_inches=0.0,
            room_depth_inches=0.0,
            store=kg_store,
            walls_assigned=0,
            total_walls=4,
            rooms_assigned=0,
            total_rooms=2,
            is_terminal=False,
            weights=weights,
        )
        expected = (
            weights.spur * breakdown.spur
            + weights.waste * breakdown.waste
            + weights.violation * breakdown.violation
            + weights.coverage * breakdown.coverage
        )
        assert abs(breakdown.total - expected) < 1e-5

    def test_terminal_adds_completion_bonus(self, kg_store):
        """Terminal state should include a completion bonus."""
        action = _make_panel_action(skip=True)
        breakdown_term = compute_reward(
            panel_action=action,
            placement_action=None,
            wall_length_inches=0.0,
            wall_panel_type=PanelType.LOAD_BEARING,
            room_width_inches=0.0,
            room_depth_inches=0.0,
            store=kg_store,
            walls_assigned=4,
            total_walls=4,
            rooms_assigned=2,
            total_rooms=2,
            is_terminal=True,
        )
        breakdown_non = compute_reward(
            panel_action=action,
            placement_action=None,
            wall_length_inches=0.0,
            wall_panel_type=PanelType.LOAD_BEARING,
            room_width_inches=0.0,
            room_depth_inches=0.0,
            store=kg_store,
            walls_assigned=4,
            total_walls=4,
            rooms_assigned=2,
            total_rooms=2,
            is_terminal=False,
        )
        assert breakdown_term.total > breakdown_non.total
        assert "completion_bonus" in breakdown_term.info

    def test_placement_skip_neutral(self, kg_store):
        action = _make_placement_action(skip=True)
        breakdown = compute_reward(
            panel_action=None,
            placement_action=action,
            wall_length_inches=0.0,
            wall_panel_type=None,
            room_width_inches=100.0,
            room_depth_inches=100.0,
            store=kg_store,
            walls_assigned=4,
            total_walls=4,
            rooms_assigned=0,
            total_rooms=2,
            is_terminal=False,
        )
        assert breakdown.spur == 0.0
        assert breakdown.waste == 0.0

    def test_placement_with_pod_gives_spur(self, kg_store):
        pod = _make_pod(width=60.0, depth=96.0)
        action = _make_placement_action(pod=pod)
        breakdown = compute_reward(
            panel_action=None,
            placement_action=action,
            wall_length_inches=0.0,
            wall_panel_type=None,
            room_width_inches=100.0,
            room_depth_inches=150.0,
            store=kg_store,
            walls_assigned=4,
            total_walls=4,
            rooms_assigned=1,
            total_rooms=2,
            is_terminal=False,
        )
        # Pod area / room area utilization
        assert breakdown.spur > 0.0

    def test_placement_violation_gives_negative_violation(self, kg_store):
        """A pod that doesn't fit should generate violations."""
        pod = _make_pod(width=200.0, depth=200.0, clearance=3.0)
        action = _make_placement_action(pod=pod)
        breakdown = compute_reward(
            panel_action=None,
            placement_action=action,
            wall_length_inches=0.0,
            wall_panel_type=None,
            room_width_inches=50.0,  # too small for pod
            room_depth_inches=50.0,
            store=kg_store,
            walls_assigned=4,
            total_walls=4,
            rooms_assigned=1,
            total_rooms=2,
            is_terminal=False,
        )
        assert breakdown.violation < 0.0
        assert len(breakdown.violations) > 0

    def test_custom_weights_affect_total(self, kg_store):
        """Different weights should produce different totals for the same action."""
        action = _make_panel_action(cut_lengths=[96.0])
        w1 = RewardWeights(spur=2.0, waste=0.0, violation=0.0, coverage=0.0)
        w2 = RewardWeights(spur=0.5, waste=0.0, violation=0.0, coverage=0.0)
        b1 = compute_reward(
            panel_action=action, placement_action=None,
            wall_length_inches=96.0, wall_panel_type=PanelType.LOAD_BEARING,
            room_width_inches=0.0, room_depth_inches=0.0,
            store=kg_store,
            walls_assigned=1, total_walls=4,
            rooms_assigned=0, total_rooms=2,
            is_terminal=False, weights=w1,
        )
        b2 = compute_reward(
            panel_action=action, placement_action=None,
            wall_length_inches=96.0, wall_panel_type=PanelType.LOAD_BEARING,
            room_width_inches=0.0, room_depth_inches=0.0,
            store=kg_store,
            walls_assigned=1, total_walls=4,
            rooms_assigned=0, total_rooms=2,
            is_terminal=False, weights=w2,
        )
        # SPUR=1.0 for standard length: 2.0*1.0 vs 0.5*1.0
        assert b1.total != b2.total
