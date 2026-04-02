"""Unit tests for DRL action space (DRL-002, DRL-003)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from docs.interfaces.classified_wall_graph import (
    FireRating,
    WallClassification,
)
from docs.interfaces.graph_to_serializer import (
    Opening,
    OpeningType,
    WallSegment,
    WallType,
)
from src.drl.actions import (
    MAX_CANDIDATES,
    PANELIZATION_ACTION_SIZE,
    PLACEMENT_ACTION_SIZE,
    PanelAction,
    PlacementAction,
    compute_panel_action_mask,
    compute_placement_action_mask,
    decode_panel_action,
    decode_placement_action,
    get_panel_candidates,
    get_pod_candidates,
)
from src.knowledge_graph.loader import load_knowledge_graph
from src.knowledge_graph.query import PanelRecommendation
from src.knowledge_graph.schema import Connection, Panel, PanelType, Pod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(
    edge_id: int = 0,
    thickness: float = 6.0,
    length: float = 100.0,
    angle: float = 0.0,
) -> WallSegment:
    """Create a WallSegment with given properties."""
    return WallSegment(
        edge_id=edge_id,
        start_node=0,
        end_node=1,
        start_coord=np.array([0.0, 0.0]),
        end_coord=np.array([length, 0.0]),
        thickness=thickness,
        height=2700.0,
        wall_type=WallType.UNKNOWN,
        angle=angle,
        length=length,
        confidence=1.0,
    )


def _make_classification(
    edge_id: int = 0,
    wall_type: WallType = WallType.LOAD_BEARING,
    fire_rating: FireRating = FireRating.NONE,
    confidence: float = 0.9,
) -> WallClassification:
    """Create a WallClassification."""
    return WallClassification(
        edge_id=edge_id,
        wall_type=wall_type,
        fire_rating=fire_rating,
        confidence=confidence,
    )


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
    min_room_width: float = 66.0,
    min_room_depth: float = 102.0,
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
        min_room_width_inches=min_room_width,
        min_room_depth_inches=min_room_depth,
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
    """Create a minimal PanelRecommendation for testing."""
    if panel is None:
        panel = _make_panel()
    if cut_lengths is None:
        cut_lengths = [96.0]
    return PanelRecommendation(
        panel=panel,
        quantity=len(cut_lengths),
        cut_lengths_inches=cut_lengths,
        requires_splice=len(cut_lengths) > 1,
        splice_connections=[],
        total_material_cost=100.0,
        waste_inches=waste,
        waste_percentage=0.0,
        score=0.85,
    )


@pytest.fixture(scope="module")
def kg_store():
    """Load the full KG once for all tests in this module."""
    return load_knowledge_graph()


# ---------------------------------------------------------------------------
# Action space sizes
# ---------------------------------------------------------------------------


class TestActionSpaceSizes:
    """Tests for action space size constants."""

    def test_panelization_action_size(self):
        assert PANELIZATION_ACTION_SIZE == MAX_CANDIDATES + 1

    def test_placement_action_size(self):
        assert PLACEMENT_ACTION_SIZE == 2 * MAX_CANDIDATES + 1

    def test_placement_larger_than_panelization(self):
        assert PLACEMENT_ACTION_SIZE > PANELIZATION_ACTION_SIZE


# ---------------------------------------------------------------------------
# compute_panel_action_mask
# ---------------------------------------------------------------------------


class TestComputePanelActionMask:
    """Tests for compute_panel_action_mask."""

    def test_skip_always_valid(self):
        mask = compute_panel_action_mask(num_candidates=0)
        assert mask[0] == 1.0

    def test_shape(self):
        mask = compute_panel_action_mask(num_candidates=5)
        assert mask.shape == (PANELIZATION_ACTION_SIZE,)
        assert mask.dtype == np.float32

    def test_candidates_enabled(self):
        mask = compute_panel_action_mask(num_candidates=3)
        assert mask[0] == 1.0  # SKIP
        assert mask[1] == 1.0  # candidate 0
        assert mask[2] == 1.0  # candidate 1
        assert mask[3] == 1.0  # candidate 2
        assert mask[4] == 0.0  # beyond candidates

    def test_zero_candidates_only_skip(self):
        mask = compute_panel_action_mask(num_candidates=0)
        assert mask[0] == 1.0
        assert np.sum(mask) == 1.0

    def test_max_candidates_all_enabled(self):
        mask = compute_panel_action_mask(num_candidates=MAX_CANDIDATES)
        assert np.sum(mask) == MAX_CANDIDATES + 1  # skip + all candidates

    def test_beyond_max_candidates_capped(self):
        mask = compute_panel_action_mask(num_candidates=MAX_CANDIDATES + 10)
        assert np.sum(mask) == MAX_CANDIDATES + 1


# ---------------------------------------------------------------------------
# compute_placement_action_mask
# ---------------------------------------------------------------------------


class TestComputePlacementActionMask:
    """Tests for compute_placement_action_mask."""

    def test_skip_always_valid(self):
        mask = compute_placement_action_mask(
            num_candidates=0,
            room_width_inches=100.0,
            room_depth_inches=100.0,
            pods=[],
        )
        assert mask[0] == 1.0

    def test_shape(self):
        mask = compute_placement_action_mask(
            num_candidates=1,
            room_width_inches=100.0,
            room_depth_inches=100.0,
            pods=[_make_pod()],
        )
        assert mask.shape == (PLACEMENT_ACTION_SIZE,)
        assert mask.dtype == np.float32

    def test_normal_orientation_enabled_when_fits(self):
        pod = _make_pod(min_room_width=60.0, min_room_depth=80.0)
        mask = compute_placement_action_mask(
            num_candidates=1,
            room_width_inches=100.0,
            room_depth_inches=100.0,
            pods=[pod],
        )
        assert mask[1] == 1.0  # normal (action 1 = pod 0 normal)

    def test_rotated_orientation_enabled_when_fits(self):
        pod = _make_pod(min_room_width=60.0, min_room_depth=80.0)
        mask = compute_placement_action_mask(
            num_candidates=1,
            room_width_inches=100.0,
            room_depth_inches=100.0,
            pods=[pod],
        )
        assert mask[2] == 1.0  # rotated (action 2 = pod 0 rotated)

    def test_pod_too_wide_masks_normal(self):
        """Pod that is too wide for the room (normal orientation)."""
        pod = _make_pod(min_room_width=150.0, min_room_depth=80.0)
        mask = compute_placement_action_mask(
            num_candidates=1,
            room_width_inches=100.0,  # too narrow for 150
            room_depth_inches=200.0,
            pods=[pod],
        )
        assert mask[1] == 0.0  # normal masked out

    def test_pod_too_deep_masks_normal(self):
        """Pod that is too deep for the room (normal orientation)."""
        pod = _make_pod(min_room_width=60.0, min_room_depth=200.0)
        mask = compute_placement_action_mask(
            num_candidates=1,
            room_width_inches=100.0,
            room_depth_inches=100.0,  # too shallow for 200
            pods=[pod],
        )
        assert mask[1] == 0.0  # normal masked out

    def test_rotation_swaps_dimensions(self):
        """A pod that only fits when rotated."""
        # min_room_width=80, min_room_depth=60  (asymmetric)
        pod = _make_pod(min_room_width=80.0, min_room_depth=60.0)
        mask = compute_placement_action_mask(
            num_candidates=1,
            room_width_inches=70.0,  # too narrow for normal (needs 80)
            room_depth_inches=90.0,  # deep enough for rotated
            pods=[pod],
        )
        # Normal: width=70 < min_room_width=80 -> masked
        assert mask[1] == 0.0
        # Rotated: checks width>=min_room_depth(60) and depth>=min_room_width(80)
        assert mask[2] == 1.0

    def test_multiple_candidates(self):
        pods = [
            _make_pod(sku="POD-A", min_room_width=50.0, min_room_depth=50.0),
            _make_pod(sku="POD-B", min_room_width=50.0, min_room_depth=50.0),
        ]
        mask = compute_placement_action_mask(
            num_candidates=2,
            room_width_inches=100.0,
            room_depth_inches=100.0,
            pods=pods,
        )
        # POD-A: action 1 (normal), action 2 (rotated)
        # POD-B: action 3 (normal), action 4 (rotated)
        assert mask[0] == 1.0  # skip
        assert mask[1] == 1.0  # POD-A normal
        assert mask[2] == 1.0  # POD-A rotated
        assert mask[3] == 1.0  # POD-B normal
        assert mask[4] == 1.0  # POD-B rotated


# ---------------------------------------------------------------------------
# decode_panel_action
# ---------------------------------------------------------------------------


class TestDecodePanelAction:
    """Tests for decode_panel_action."""

    def test_skip_action(self):
        wall = _make_segment(edge_id=5)
        result = decode_panel_action(action=0, wall=wall, candidates=[])
        assert result.skip is True
        assert result.wall_edge_id == 5
        assert result.panel is None

    def test_valid_candidate_selection(self):
        wall = _make_segment(edge_id=3)
        panel = _make_panel(sku="PNL-ABC")
        rec = _make_recommendation(panel=panel, cut_lengths=[96.0, 48.0])
        result = decode_panel_action(action=1, wall=wall, candidates=[rec])
        assert result.skip is False
        assert result.wall_edge_id == 3
        assert result.panel is not None
        assert result.panel.sku == "PNL-ABC"
        assert len(result.panel_assignments) == 2

    def test_panel_assignments_format(self):
        wall = _make_segment()
        panel = _make_panel(sku="PNL-X")
        rec = _make_recommendation(panel=panel, cut_lengths=[120.0, 80.0])
        result = decode_panel_action(action=1, wall=wall, candidates=[rec])
        for sku, length in result.panel_assignments:
            assert sku == "PNL-X"
        assert result.panel_assignments[0][1] == 120.0
        assert result.panel_assignments[1][1] == 80.0

    def test_second_candidate_selected(self):
        wall = _make_segment()
        rec_a = _make_recommendation(panel=_make_panel(sku="PNL-A"))
        rec_b = _make_recommendation(panel=_make_panel(sku="PNL-B"))
        result = decode_panel_action(action=2, wall=wall, candidates=[rec_a, rec_b])
        assert result.panel.sku == "PNL-B"

    def test_out_of_range_raises(self):
        wall = _make_segment()
        with pytest.raises(ValueError):
            decode_panel_action(action=-1, wall=wall, candidates=[])
        with pytest.raises(ValueError):
            decode_panel_action(
                action=PANELIZATION_ACTION_SIZE, wall=wall, candidates=[],
            )

    def test_candidate_index_out_of_bounds_raises(self):
        wall = _make_segment()
        rec = _make_recommendation()
        with pytest.raises(ValueError, match="candidate index"):
            decode_panel_action(action=2, wall=wall, candidates=[rec])


# ---------------------------------------------------------------------------
# decode_placement_action
# ---------------------------------------------------------------------------


class TestDecodePlacementAction:
    """Tests for decode_placement_action."""

    def test_skip_action(self):
        result = decode_placement_action(
            action=0, room_id=7, candidates=[], room_centroid=(50.0, 75.0),
        )
        assert result.skip is True
        assert result.room_id == 7

    def test_normal_orientation(self):
        """Odd actions = normal orientation."""
        pod = _make_pod(sku="POD-BATH")
        result = decode_placement_action(
            action=1, room_id=0, candidates=[pod], room_centroid=(100.0, 200.0),
        )
        assert result.skip is False
        assert result.pod.sku == "POD-BATH"
        assert result.rotated is False
        assert result.position_x == 100.0
        assert result.position_y == 200.0

    def test_rotated_orientation(self):
        """Even actions = rotated 90 degrees."""
        pod = _make_pod(sku="POD-BATH")
        result = decode_placement_action(
            action=2, room_id=0, candidates=[pod], room_centroid=(100.0, 200.0),
        )
        assert result.skip is False
        assert result.pod.sku == "POD-BATH"
        assert result.rotated is True

    def test_action_index_mapping(self):
        """Verify the action-to-candidate index math."""
        pods = [_make_pod(sku=f"POD-{i}") for i in range(3)]
        # Action 1 -> candidate 0, normal
        r1 = decode_placement_action(action=1, room_id=0, candidates=pods, room_centroid=(0, 0))
        assert r1.pod.sku == "POD-0" and r1.rotated is False
        # Action 2 -> candidate 0, rotated
        r2 = decode_placement_action(action=2, room_id=0, candidates=pods, room_centroid=(0, 0))
        assert r2.pod.sku == "POD-0" and r2.rotated is True
        # Action 3 -> candidate 1, normal
        r3 = decode_placement_action(action=3, room_id=0, candidates=pods, room_centroid=(0, 0))
        assert r3.pod.sku == "POD-1" and r3.rotated is False
        # Action 4 -> candidate 1, rotated
        r4 = decode_placement_action(action=4, room_id=0, candidates=pods, room_centroid=(0, 0))
        assert r4.pod.sku == "POD-1" and r4.rotated is True
        # Action 5 -> candidate 2, normal
        r5 = decode_placement_action(action=5, room_id=0, candidates=pods, room_centroid=(0, 0))
        assert r5.pod.sku == "POD-2" and r5.rotated is False

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            decode_placement_action(
                action=-1, room_id=0, candidates=[], room_centroid=(0, 0),
            )
        with pytest.raises(ValueError):
            decode_placement_action(
                action=PLACEMENT_ACTION_SIZE,
                room_id=0, candidates=[], room_centroid=(0, 0),
            )

    def test_candidate_index_out_of_bounds_raises(self):
        pod = _make_pod()
        with pytest.raises(ValueError, match="candidate index"):
            decode_placement_action(
                action=3,  # candidate index 1, but only 1 candidate
                room_id=0,
                candidates=[pod],
                room_centroid=(0, 0),
            )


# ---------------------------------------------------------------------------
# get_panel_candidates (uses real KG)
# ---------------------------------------------------------------------------


class TestGetPanelCandidates:
    """Tests for KG-based panel candidate retrieval."""

    def test_returns_list(self, kg_store):
        wall = _make_segment(length=7200.0)  # 100 inches at scale=1.0 (1/72)
        cls = _make_classification(wall_type=WallType.LOAD_BEARING)
        result = get_panel_candidates(kg_store, wall, cls, [], scale=1.0)
        assert isinstance(result, list)

    def test_result_capped_at_max_candidates(self, kg_store):
        wall = _make_segment(length=7200.0)
        cls = _make_classification(wall_type=WallType.LOAD_BEARING)
        result = get_panel_candidates(kg_store, wall, cls, [], scale=1.0)
        assert len(result) <= MAX_CANDIDATES

    def test_zero_length_after_openings_returns_empty(self, kg_store):
        """If openings consume all panelizable length, return empty list."""
        wall = _make_segment(length=72.0)  # 1 inch
        cls = _make_classification()
        openings = [
            Opening(
                opening_type=OpeningType.DOOR,
                wall_edge_id=0,
                position_along_wall=0.5,
                width=72.0,  # 1 inch -- consumes entire wall
                height=80.0,
            ),
        ]
        result = get_panel_candidates(kg_store, wall, cls, openings, scale=1.0)
        assert result == []

    def test_fire_rated_wall_gets_fire_rated_panels(self, kg_store):
        wall = _make_segment(length=7200.0)
        cls = _make_classification(
            wall_type=WallType.LOAD_BEARING,
            fire_rating=FireRating.HOUR_1,
        )
        result = get_panel_candidates(kg_store, wall, cls, [], scale=1.0)
        # All returned panels should be fire-rated type
        for rec in result:
            assert rec.panel.panel_type == PanelType.FIRE_RATED

    def test_recommendations_have_panel_and_lengths(self, kg_store):
        wall = _make_segment(length=7200.0)
        cls = _make_classification(wall_type=WallType.LOAD_BEARING)
        result = get_panel_candidates(kg_store, wall, cls, [], scale=1.0)
        if result:
            rec = result[0]
            assert rec.panel is not None
            assert len(rec.cut_lengths_inches) > 0
            assert rec.quantity >= 1


# ---------------------------------------------------------------------------
# get_pod_candidates (uses real KG)
# ---------------------------------------------------------------------------


class TestGetPodCandidates:
    """Tests for KG-based pod candidate retrieval."""

    def test_returns_list(self, kg_store):
        result = get_pod_candidates(
            kg_store,
            room_width_inches=120.0,
            room_depth_inches=150.0,
            room_label="Bathroom",
        )
        assert isinstance(result, list)

    def test_result_capped_at_max_candidates(self, kg_store):
        result = get_pod_candidates(
            kg_store,
            room_width_inches=500.0,
            room_depth_inches=500.0,
            room_label="",
        )
        assert len(result) <= MAX_CANDIDATES

    def test_tiny_room_returns_empty(self, kg_store):
        """A very small room should have no fitting pods."""
        result = get_pod_candidates(
            kg_store,
            room_width_inches=10.0,
            room_depth_inches=10.0,
            room_label="Bathroom",
        )
        assert result == []

    def test_bathroom_returns_bathroom_pods(self, kg_store):
        result = get_pod_candidates(
            kg_store,
            room_width_inches=200.0,
            room_depth_inches=200.0,
            room_label="Bathroom",
        )
        for pod in result:
            assert pod.pod_type == "bathroom"

    def test_unknown_label_falls_back(self, kg_store):
        """Unknown room function should fall back to returning all fitting pods."""
        result = get_pod_candidates(
            kg_store,
            room_width_inches=500.0,
            room_depth_inches=500.0,
            room_label="observatory",
        )
        # Should return some pods (fallback to no function filter)
        # or empty if nothing fits -- just ensure it doesn't error
        assert isinstance(result, list)
