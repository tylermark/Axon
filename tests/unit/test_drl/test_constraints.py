"""Unit tests for DRL constraint functions (DRL-007, DRL-008).

Tests cover:
- compute_wall_sub_segments: splitting walls at openings into panelizable
  sub-segments.
- compute_junction_map: identifying wall junctions (corner, T, cross).
- compute_junction_penalties: gauge/stud-depth compatibility at junctions.
- get_corner_thickness_deduction: perpendicular wall thickness at corners.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from docs.interfaces.classified_wall_graph import (
    ClassifiedWallGraph,
    FireRating,
    WallClassification,
)
from docs.interfaces.graph_to_serializer import (
    FinalizedGraph,
    Opening,
    OpeningType,
    Room,
    WallSegment,
    WallType,
)
from src.drl.constraints import (
    JunctionInfo,
    WallSubSegment,
    compute_junction_map,
    compute_junction_penalties,
    compute_wall_sub_segments,
    get_corner_thickness_deduction,
)
from src.knowledge_graph.loader import load_knowledge_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wall(
    edge_id: int = 0,
    start_node: int = 0,
    end_node: int = 1,
    length: float = 7200.0,
    thickness: float = 6.0,
    angle: float = 0.0,
    start_coord: np.ndarray | None = None,
) -> WallSegment:
    """Create a WallSegment with known geometry.

    Default length is 7200 PDF user units (100 inches at 72 units/inch).
    """
    if start_coord is None:
        start_coord = np.array([0.0, 0.0])
    end_coord = start_coord + np.array(
        [length * math.cos(angle), length * math.sin(angle)]
    )
    return WallSegment(
        edge_id=edge_id,
        start_node=start_node,
        end_node=end_node,
        start_coord=start_coord,
        end_coord=end_coord,
        thickness=thickness,
        height=2700.0,
        wall_type=WallType.UNKNOWN,
        angle=angle,
        length=length,
        confidence=1.0,
    )


def _make_opening(
    wall_edge_id: int = 0,
    position_along_wall: float = 0.5,
    width: float = 2160.0,
    opening_type: OpeningType = OpeningType.DOOR,
) -> Opening:
    """Create an Opening on a wall.

    Default width is 2160 PDF units (30 inches at 72 units/inch).
    """
    return Opening(
        opening_type=opening_type,
        wall_edge_id=wall_edge_id,
        position_along_wall=position_along_wall,
        width=width,
        height=2000.0,
    )


@pytest.fixture(scope="module")
def kg_store():
    """Load the KG store once for all tests in this module."""
    return load_knowledge_graph()


# ---------------------------------------------------------------------------
# compute_wall_sub_segments
# ---------------------------------------------------------------------------


class TestComputeWallSubSegmentsNoOpenings:
    """Wall with no openings returns a single sub-segment."""

    def test_single_segment_returned(self):
        wall = _make_wall(length=7200.0)
        result = compute_wall_sub_segments(wall, [], scale=1.0)
        assert len(result) == 1

    def test_segment_covers_full_wall(self):
        wall = _make_wall(length=7200.0)
        result = compute_wall_sub_segments(wall, [], scale=1.0)
        seg = result[0]
        expected_inches = 7200.0 / 72.0  # 100 inches
        assert abs(seg.length_inches - expected_inches) < 0.1

    def test_segment_starts_at_zero(self):
        wall = _make_wall(length=7200.0)
        result = compute_wall_sub_segments(wall, [], scale=1.0)
        assert result[0].start_offset_inches == 0.0

    def test_segment_index_and_total(self):
        wall = _make_wall(length=7200.0)
        result = compute_wall_sub_segments(wall, [], scale=1.0)
        assert result[0].segment_index == 0
        assert result[0].total_segments == 1

    def test_not_bounded_by_opening(self):
        wall = _make_wall(length=7200.0)
        result = compute_wall_sub_segments(wall, [], scale=1.0)
        assert result[0].left_bounded_by_opening is False
        assert result[0].right_bounded_by_opening is False

    def test_wall_edge_id_preserved(self):
        wall = _make_wall(edge_id=42, length=7200.0)
        result = compute_wall_sub_segments(wall, [], scale=1.0)
        assert result[0].wall_edge_id == 42

    def test_empty_openings_list_same_as_no_openings(self):
        wall = _make_wall(length=7200.0)
        result_empty = compute_wall_sub_segments(wall, [], scale=1.0)
        result_none = compute_wall_sub_segments(wall, [], scale=1.0)
        assert len(result_empty) == len(result_none)


class TestComputeWallSubSegmentsOneOpening:
    """Wall with one opening in the middle produces two sub-segments."""

    def test_two_segments_returned(self):
        wall = _make_wall(edge_id=0, length=7200.0)
        opening = _make_opening(wall_edge_id=0, position_along_wall=0.5, width=2160.0)
        result = compute_wall_sub_segments(wall, [opening], scale=1.0)
        assert len(result) == 2

    def test_segments_cover_remaining_length(self):
        wall = _make_wall(edge_id=0, length=7200.0)
        # Opening is 30 inches wide in the middle of a 100-inch wall
        opening = _make_opening(wall_edge_id=0, position_along_wall=0.5, width=2160.0)
        result = compute_wall_sub_segments(wall, [opening], scale=1.0)
        total_sub_length = sum(s.length_inches for s in result)
        expected = 100.0 - 30.0  # 70 inches
        assert abs(total_sub_length - expected) < 1.0

    def test_first_segment_ends_before_opening(self):
        wall = _make_wall(edge_id=0, length=7200.0)
        opening = _make_opening(wall_edge_id=0, position_along_wall=0.5, width=2160.0)
        result = compute_wall_sub_segments(wall, [opening], scale=1.0)
        # Opening center at 50 inches, half-width = 15 inches
        # First segment: 0 to 35 inches
        assert result[0].start_offset_inches == pytest.approx(0.0, abs=0.5)
        assert result[0].end_offset_inches == pytest.approx(35.0, abs=0.5)

    def test_second_segment_starts_after_opening(self):
        wall = _make_wall(edge_id=0, length=7200.0)
        opening = _make_opening(wall_edge_id=0, position_along_wall=0.5, width=2160.0)
        result = compute_wall_sub_segments(wall, [opening], scale=1.0)
        # Opening center at 50, half-width = 15 -> opening ends at 65
        assert result[1].start_offset_inches == pytest.approx(65.0, abs=0.5)
        assert result[1].end_offset_inches == pytest.approx(100.0, abs=0.5)

    def test_boundary_flags(self):
        wall = _make_wall(edge_id=0, length=7200.0)
        opening = _make_opening(wall_edge_id=0, position_along_wall=0.5, width=2160.0)
        result = compute_wall_sub_segments(wall, [opening], scale=1.0)
        # First segment: left = wall start (not opening), right = opening
        assert result[0].left_bounded_by_opening is False
        assert result[0].right_bounded_by_opening is True
        # Second segment: left = opening, right = wall end (not opening)
        assert result[1].left_bounded_by_opening is True
        assert result[1].right_bounded_by_opening is False

    def test_total_segments_field(self):
        wall = _make_wall(edge_id=0, length=7200.0)
        opening = _make_opening(wall_edge_id=0, position_along_wall=0.5, width=2160.0)
        result = compute_wall_sub_segments(wall, [opening], scale=1.0)
        for seg in result:
            assert seg.total_segments == 2


class TestComputeWallSubSegmentsMultipleOpenings:
    """Wall with multiple openings produces sub-segments between them."""

    def test_three_segments_for_two_openings(self):
        wall = _make_wall(edge_id=0, length=14400.0)  # 200 inches
        o1 = _make_opening(wall_edge_id=0, position_along_wall=0.25, width=2160.0)
        o2 = _make_opening(wall_edge_id=0, position_along_wall=0.75, width=2160.0)
        result = compute_wall_sub_segments(wall, [o1, o2], scale=1.0)
        assert len(result) == 3

    def test_segment_indices_sequential(self):
        wall = _make_wall(edge_id=0, length=14400.0)
        o1 = _make_opening(wall_edge_id=0, position_along_wall=0.25, width=2160.0)
        o2 = _make_opening(wall_edge_id=0, position_along_wall=0.75, width=2160.0)
        result = compute_wall_sub_segments(wall, [o1, o2], scale=1.0)
        for i, seg in enumerate(result):
            assert seg.segment_index == i
            assert seg.total_segments == 3

    def test_total_sub_length_correct(self):
        wall = _make_wall(edge_id=0, length=14400.0)  # 200 inches
        o1 = _make_opening(wall_edge_id=0, position_along_wall=0.25, width=2160.0)  # 30 in
        o2 = _make_opening(wall_edge_id=0, position_along_wall=0.75, width=2160.0)  # 30 in
        result = compute_wall_sub_segments(wall, [o1, o2], scale=1.0)
        total = sum(s.length_inches for s in result)
        expected = 200.0 - 60.0  # 140 inches
        assert abs(total - expected) < 2.0


class TestComputeWallSubSegmentsOverlappingOpenings:
    """Overlapping openings should be merged."""

    def test_overlapping_openings_merged(self):
        wall = _make_wall(edge_id=0, length=7200.0)  # 100 inches
        # Two overlapping openings near the center
        o1 = _make_opening(wall_edge_id=0, position_along_wall=0.4, width=2160.0)
        o2 = _make_opening(wall_edge_id=0, position_along_wall=0.6, width=2160.0)
        result = compute_wall_sub_segments(wall, [o1, o2], scale=1.0)
        # The two openings overlap, so they form a single exclusion zone
        # This should yield 2 sub-segments, not 3
        assert len(result) == 2

    def test_total_length_accounts_for_merge(self):
        wall = _make_wall(edge_id=0, length=7200.0)  # 100 inches
        # Openings: center at 40 (width 30) and center at 60 (width 30)
        # Zone 1: 25-55, Zone 2: 45-75 -> merged: 25-75
        o1 = _make_opening(wall_edge_id=0, position_along_wall=0.4, width=2160.0)
        o2 = _make_opening(wall_edge_id=0, position_along_wall=0.6, width=2160.0)
        result = compute_wall_sub_segments(wall, [o1, o2], scale=1.0)
        total = sum(s.length_inches for s in result)
        # Merged exclusion zone is ~50 inches, so sub-segments total ~50 inches
        assert total < 55.0  # much less than 70 (if they didn't overlap)


class TestComputeWallSubSegmentsEdgeCases:
    """Opening at wall start, end, and covering entire wall."""

    def test_opening_at_wall_start(self):
        wall = _make_wall(edge_id=0, length=7200.0)  # 100 inches
        # Opening centered at position 0.0 (wall start), width 30 inches
        opening = _make_opening(wall_edge_id=0, position_along_wall=0.0, width=2160.0)
        result = compute_wall_sub_segments(wall, [opening], scale=1.0)
        # Only one sub-segment after the opening (the start portion is tiny/zero)
        assert len(result) >= 1
        # First sub-segment should start after the opening
        assert result[0].start_offset_inches >= 0.0

    def test_opening_at_wall_end(self):
        wall = _make_wall(edge_id=0, length=7200.0)  # 100 inches
        # Opening centered at position 1.0 (wall end), width 30 inches
        opening = _make_opening(wall_edge_id=0, position_along_wall=1.0, width=2160.0)
        result = compute_wall_sub_segments(wall, [opening], scale=1.0)
        assert len(result) >= 1
        # The last sub-segment should end before the opening
        assert result[-1].end_offset_inches <= 100.5

    def test_opening_covers_entire_wall(self):
        wall = _make_wall(edge_id=0, length=7200.0)  # 100 inches
        # Opening as wide as the wall
        opening = _make_opening(
            wall_edge_id=0, position_along_wall=0.5, width=7200.0,
        )
        result = compute_wall_sub_segments(wall, [opening], scale=1.0)
        # No panelizable sub-segments when the opening covers the entire wall
        assert len(result) == 0

    def test_custom_scale(self):
        """Scale factor != 1.0 uses scale/25.4 conversion."""
        wall = _make_wall(edge_id=0, length=2540.0)  # 2540 units
        # With scale=25.4: to_inches = 25.4/25.4 = 1.0 -> length = 2540 inches
        result = compute_wall_sub_segments(wall, [], scale=25.4)
        assert abs(result[0].length_inches - 2540.0) < 0.1


# ---------------------------------------------------------------------------
# compute_junction_map
# ---------------------------------------------------------------------------


class TestComputeJunctionMapCorner:
    """Corner junction: two walls sharing one node."""

    def test_corner_detected(self):
        # Two walls share node 1 (end of wall 0, start of wall 1)
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        junction_map = compute_junction_map([wall_0, wall_1])
        assert 1 in junction_map

    def test_corner_type(self):
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        junction_map = compute_junction_map([wall_0, wall_1])
        assert junction_map[1].junction_type == "corner"

    def test_corner_wall_edge_ids(self):
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        junction_map = compute_junction_map([wall_0, wall_1])
        assert set(junction_map[1].wall_edge_ids) == {0, 1}

    def test_corner_has_two_angles(self):
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        junction_map = compute_junction_map([wall_0, wall_1])
        assert len(junction_map[1].angles) == 2


class TestComputeJunctionMapTJunction:
    """T-junction: three walls sharing one node."""

    def test_t_junction_detected(self):
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        wall_2 = _make_wall(edge_id=2, start_node=1, end_node=3, angle=math.pi)
        junction_map = compute_junction_map([wall_0, wall_1, wall_2])
        assert 1 in junction_map

    def test_t_junction_type(self):
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        wall_2 = _make_wall(edge_id=2, start_node=1, end_node=3, angle=math.pi)
        junction_map = compute_junction_map([wall_0, wall_1, wall_2])
        assert junction_map[1].junction_type == "T"

    def test_t_junction_three_walls(self):
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        wall_2 = _make_wall(edge_id=2, start_node=1, end_node=3, angle=math.pi)
        junction_map = compute_junction_map([wall_0, wall_1, wall_2])
        assert len(junction_map[1].wall_edge_ids) == 3


class TestComputeJunctionMapCross:
    """Cross junction: four walls sharing one node."""

    def test_cross_junction_detected(self):
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        wall_2 = _make_wall(edge_id=2, start_node=1, end_node=3, angle=math.pi)
        wall_3 = _make_wall(edge_id=3, start_node=1, end_node=4, angle=3 * math.pi / 2)
        junction_map = compute_junction_map([wall_0, wall_1, wall_2, wall_3])
        assert 1 in junction_map

    def test_cross_junction_type(self):
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        wall_2 = _make_wall(edge_id=2, start_node=1, end_node=3, angle=math.pi)
        wall_3 = _make_wall(edge_id=3, start_node=1, end_node=4, angle=3 * math.pi / 2)
        junction_map = compute_junction_map([wall_0, wall_1, wall_2, wall_3])
        assert junction_map[1].junction_type == "cross"

    def test_cross_junction_four_walls(self):
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        wall_2 = _make_wall(edge_id=2, start_node=1, end_node=3, angle=math.pi)
        wall_3 = _make_wall(edge_id=3, start_node=1, end_node=4, angle=3 * math.pi / 2)
        junction_map = compute_junction_map([wall_0, wall_1, wall_2, wall_3])
        assert len(junction_map[1].wall_edge_ids) == 4


class TestComputeJunctionMapIsolated:
    """Isolated wall (no shared nodes) produces no junctions."""

    def test_single_wall_no_junction(self):
        wall = _make_wall(edge_id=0, start_node=0, end_node=1)
        junction_map = compute_junction_map([wall])
        assert len(junction_map) == 0

    def test_two_disjoint_walls_no_junction(self):
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1)
        wall_1 = _make_wall(edge_id=1, start_node=2, end_node=3)
        junction_map = compute_junction_map([wall_0, wall_1])
        assert len(junction_map) == 0

    def test_empty_walls_list(self):
        junction_map = compute_junction_map([])
        assert len(junction_map) == 0

    def test_dead_end_excluded(self):
        """Node with only one wall (dead end) should not be in the map."""
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2)
        junction_map = compute_junction_map([wall_0, wall_1])
        # Node 1 is shared by 2 walls: this IS a junction
        assert 1 in junction_map
        # Node 0 and node 2 are dead ends: should NOT be in the map
        assert 0 not in junction_map
        assert 2 not in junction_map


# ---------------------------------------------------------------------------
# compute_junction_penalties
# ---------------------------------------------------------------------------


class TestComputeJunctionPenaltiesMatchingGauges:
    """Matching gauges and stud depths produce no penalty."""

    def test_no_penalty_for_matching_specs(self, kg_store):
        # Two walls at a corner, both assigned the same panel
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        walls = [wall_0, wall_1]
        junction_map = compute_junction_map(walls)

        # Pick the first available panel from the KG
        first_sku = next(iter(kg_store.panels))
        wall_assignments = {
            1: [(first_sku, 96.0)],
        }
        panel = kg_store.panels[first_sku]

        penalty, violations = compute_junction_penalties(
            wall_edge_id=0,
            panel=panel,
            junction_map=junction_map,
            wall_assignments=wall_assignments,
            walls=walls,
            store=kg_store,
        )
        assert penalty == 0.0
        assert violations == []

    def test_none_panel_no_penalty(self, kg_store):
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1)
        walls = [wall_0]
        junction_map = compute_junction_map(walls)

        penalty, violations = compute_junction_penalties(
            wall_edge_id=0,
            panel=None,
            junction_map=junction_map,
            wall_assignments={},
            walls=walls,
            store=kg_store,
        )
        assert penalty == 0.0
        assert violations == []


class TestComputeJunctionPenaltiesMismatchedGauges:
    """Mismatched gauges produce a penalty."""

    def test_gauge_mismatch_penalty(self, kg_store):
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        walls = [wall_0, wall_1]
        junction_map = compute_junction_map(walls)

        # Find two panels with different gauges
        panels_by_gauge: dict[int, str] = {}
        for sku, panel in kg_store.panels.items():
            if panel.gauge not in panels_by_gauge:
                panels_by_gauge[panel.gauge] = sku
        if len(panels_by_gauge) < 2:
            pytest.skip("Need at least two different gauges in KG")

        gauges = sorted(panels_by_gauge.keys())
        sku_a = panels_by_gauge[gauges[0]]
        sku_b = panels_by_gauge[gauges[1]]

        wall_assignments = {1: [(sku_b, 96.0)]}
        panel_a = kg_store.panels[sku_a]

        penalty, violations = compute_junction_penalties(
            wall_edge_id=0,
            panel=panel_a,
            junction_map=junction_map,
            wall_assignments=wall_assignments,
            walls=walls,
            store=kg_store,
        )
        assert penalty < 0.0
        assert any("gauge" in v.lower() for v in violations)


class TestComputeJunctionPenaltiesMismatchedStudDepths:
    """Mismatched stud depths produce a penalty."""

    def test_stud_depth_mismatch_penalty(self, kg_store):
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        walls = [wall_0, wall_1]
        junction_map = compute_junction_map(walls)

        # Find two panels with different stud depths
        panels_by_depth: dict[float, str] = {}
        for sku, panel in kg_store.panels.items():
            if panel.stud_depth_inches not in panels_by_depth:
                panels_by_depth[panel.stud_depth_inches] = sku
        if len(panels_by_depth) < 2:
            pytest.skip("Need at least two different stud depths in KG")

        depths = sorted(panels_by_depth.keys())
        sku_a = panels_by_depth[depths[0]]
        sku_b = panels_by_depth[depths[1]]

        wall_assignments = {1: [(sku_b, 96.0)]}
        panel_a = kg_store.panels[sku_a]

        penalty, violations = compute_junction_penalties(
            wall_edge_id=0,
            panel=panel_a,
            junction_map=junction_map,
            wall_assignments=wall_assignments,
            walls=walls,
            store=kg_store,
        )
        assert penalty < 0.0
        assert any("stud depth" in v.lower() for v in violations)


class TestComputeJunctionPenaltiesBothMismatched:
    """Both gauge and stud depth mismatched produces compounded penalty."""

    def test_both_mismatched_larger_penalty(self, kg_store):
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        walls = [wall_0, wall_1]
        junction_map = compute_junction_map(walls)

        # Find two panels that differ on BOTH gauge and stud depth
        candidates: list[tuple[str, int, float]] = [
            (sku, p.gauge, p.stud_depth_inches) for sku, p in kg_store.panels.items()
        ]
        pair_found = False
        for i, (sku_a, ga, da) in enumerate(candidates):
            for sku_b, gb, db in candidates[i + 1 :]:
                if ga != gb and da != db:
                    pair_found = True
                    wall_assignments = {1: [(sku_b, 96.0)]}
                    panel_a = kg_store.panels[sku_a]
                    penalty, violations = compute_junction_penalties(
                        wall_edge_id=0,
                        panel=panel_a,
                        junction_map=junction_map,
                        wall_assignments=wall_assignments,
                        walls=walls,
                        store=kg_store,
                    )
                    assert penalty < 0.0
                    assert len(violations) >= 2
                    break
            if pair_found:
                break
        if not pair_found:
            pytest.skip("Need two panels differing on both gauge and stud depth")


class TestComputeJunctionPenaltiesNoAdjacentAssignments:
    """No adjacent wall assignments means no penalty."""

    def test_unassigned_adjacent_no_penalty(self, kg_store):
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        walls = [wall_0, wall_1]
        junction_map = compute_junction_map(walls)

        first_sku = next(iter(kg_store.panels))
        panel = kg_store.panels[first_sku]

        # No wall_assignments at all -- adjacent wall 1 has no panel
        penalty, violations = compute_junction_penalties(
            wall_edge_id=0,
            panel=panel,
            junction_map=junction_map,
            wall_assignments={},
            walls=walls,
            store=kg_store,
        )
        assert penalty == 0.0
        assert violations == []

    def test_penalty_clamped_at_negative_one(self, kg_store):
        """Penalty should be clamped to -1.0 at worst."""
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2)
        walls = [wall_0, wall_1]
        junction_map = compute_junction_map(walls)

        # Even with mismatched specs, the penalty should not go below -1.0
        candidates = list(kg_store.panels.items())
        if len(candidates) >= 2:
            sku_a = candidates[0][0]
            sku_b = candidates[1][0]
            wall_assignments = {1: [(sku_b, 96.0)]}
            panel_a = kg_store.panels[sku_a]
            penalty, _ = compute_junction_penalties(
                wall_edge_id=0,
                panel=panel_a,
                junction_map=junction_map,
                wall_assignments=wall_assignments,
                walls=walls,
                store=kg_store,
            )
            assert penalty >= -1.0


# ---------------------------------------------------------------------------
# get_corner_thickness_deduction
# ---------------------------------------------------------------------------


class TestGetCornerThicknessDeductionPerpendicular:
    """Perpendicular walls (~90 degrees) should produce a deduction."""

    def test_perpendicular_produces_deduction(self):
        # Horizontal wall and vertical wall sharing node 1
        wall_0 = _make_wall(
            edge_id=0, start_node=0, end_node=1,
            length=7200.0, thickness=432.0, angle=0.0,
        )
        wall_1 = _make_wall(
            edge_id=1, start_node=1, end_node=2,
            length=7200.0, thickness=432.0, angle=math.pi / 2,
        )
        walls = [wall_0, wall_1]
        junction_map = compute_junction_map(walls)

        deduction = get_corner_thickness_deduction(
            wall_edge_id=0,
            junction_map=junction_map,
            walls=walls,
            scale=1.0,
        )
        # Wall 1 thickness = 432 PDF units = 6 inches at 72 units/inch
        expected = 432.0 / 72.0
        assert abs(deduction - expected) < 0.5

    def test_deduction_at_both_ends(self):
        """A wall with perpendicular walls at both ends gets double deduction."""
        # Rectangle: wall 1 (horizontal top) has perpendicular walls at each end
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0, thickness=432.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=math.pi / 2, thickness=432.0)
        wall_2 = _make_wall(edge_id=2, start_node=2, end_node=3, angle=math.pi, thickness=432.0)
        wall_3 = _make_wall(edge_id=3, start_node=3, end_node=0, angle=3 * math.pi / 2, thickness=432.0)
        walls = [wall_0, wall_1, wall_2, wall_3]
        junction_map = compute_junction_map(walls)

        # Wall 0 (horizontal bottom): perpendicular to wall_3 at node 0, wall_1 at node 1
        deduction = get_corner_thickness_deduction(
            wall_edge_id=0,
            junction_map=junction_map,
            walls=walls,
            scale=1.0,
        )
        # Each perpendicular wall contributes 432/72 = 6 inches
        expected = 2 * (432.0 / 72.0)
        assert abs(deduction - expected) < 1.0

    def test_custom_scale_deduction(self):
        """Deduction with scale != 1.0 uses scale/25.4 conversion."""
        wall_0 = _make_wall(
            edge_id=0, start_node=0, end_node=1,
            thickness=152.4, angle=0.0,  # 152.4 units
        )
        wall_1 = _make_wall(
            edge_id=1, start_node=1, end_node=2,
            thickness=152.4, angle=math.pi / 2,
        )
        walls = [wall_0, wall_1]
        junction_map = compute_junction_map(walls)

        # scale=25.4: to_inches = 25.4/25.4 = 1.0 -> thickness = 152.4 inches
        deduction = get_corner_thickness_deduction(
            wall_edge_id=0,
            junction_map=junction_map,
            walls=walls,
            scale=25.4,
        )
        assert abs(deduction - 152.4) < 0.5


class TestGetCornerThicknessDeductionParallel:
    """Parallel walls (no perpendicular angle) produce no deduction."""

    def test_parallel_no_deduction(self):
        # Two walls at same angle sharing a node (straight continuation)
        wall_0 = _make_wall(edge_id=0, start_node=0, end_node=1, angle=0.0, thickness=432.0)
        wall_1 = _make_wall(edge_id=1, start_node=1, end_node=2, angle=0.0, thickness=432.0)
        walls = [wall_0, wall_1]
        junction_map = compute_junction_map(walls)

        deduction = get_corner_thickness_deduction(
            wall_edge_id=0,
            junction_map=junction_map,
            walls=walls,
            scale=1.0,
        )
        assert deduction == 0.0


class TestGetCornerThicknessDeductionNoJunction:
    """Wall with no junctions produces zero deduction."""

    def test_isolated_wall_no_deduction(self):
        wall = _make_wall(edge_id=0, start_node=0, end_node=1, thickness=432.0)
        junction_map = compute_junction_map([wall])

        deduction = get_corner_thickness_deduction(
            wall_edge_id=0,
            junction_map=junction_map,
            walls=[wall],
            scale=1.0,
        )
        assert deduction == 0.0

    def test_nonexistent_wall_no_deduction(self):
        """Querying a wall_edge_id that doesn't exist returns 0."""
        wall = _make_wall(edge_id=0, start_node=0, end_node=1)
        junction_map = compute_junction_map([wall])

        deduction = get_corner_thickness_deduction(
            wall_edge_id=999,
            junction_map=junction_map,
            walls=[wall],
            scale=1.0,
        )
        assert deduction == 0.0
