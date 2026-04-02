"""Unit tests for DRL state encoding (DRL-001: state representation)."""

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
from src.drl.state import (
    CANDIDATE_PANEL_DIM,
    CANDIDATE_POD_DIM,
    MAX_CANDIDATES,
    MAX_ROOMS,
    MAX_WALLS,
    PDF_UNITS_PER_INCH,
    ROOM_FEATURE_DIM,
    WALL_FEATURE_DIM,
    encode_observation,
    encode_panel_candidate,
    encode_pod_candidate,
    encode_room,
    encode_wall_segment,
    fire_rating_to_hours,
    wall_type_to_panel_type,
)
from src.knowledge_graph.schema import Panel, PanelType, Pod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(
    edge_id: int = 0,
    thickness: float = 6.0,
    length: float = 100.0,
    angle: float = 0.0,
    start_node: int = 0,
    end_node: int = 1,
    start_coord: np.ndarray | None = None,
    wall_type: WallType = WallType.UNKNOWN,
) -> WallSegment:
    """Create a WallSegment with given properties."""
    if start_coord is None:
        start_coord = np.array([0.0, 0.0])
    end_coord = start_coord + np.array([length * math.cos(angle), length * math.sin(angle)])
    return WallSegment(
        edge_id=edge_id,
        start_node=start_node,
        end_node=end_node,
        start_coord=start_coord,
        end_coord=end_coord,
        thickness=thickness,
        height=2700.0,
        wall_type=wall_type,
        angle=angle,
        length=length,
        confidence=1.0,
    )


def _make_classification(
    edge_id: int = 0,
    wall_type: WallType = WallType.LOAD_BEARING,
    fire_rating: FireRating = FireRating.NONE,
    confidence: float = 0.9,
    is_perimeter: bool = False,
) -> WallClassification:
    """Create a WallClassification with given properties."""
    return WallClassification(
        edge_id=edge_id,
        wall_type=wall_type,
        fire_rating=fire_rating,
        confidence=confidence,
        is_perimeter=is_perimeter,
    )


def _make_room(
    room_id: int = 0,
    boundary_edges: list[int] | None = None,
    boundary_nodes: list[int] | None = None,
    area: float = 50000.0,
    label: str = "Bedroom",
    is_exterior: bool = False,
) -> Room:
    """Create a Room with given properties."""
    return Room(
        room_id=room_id,
        boundary_edges=boundary_edges or [0, 1, 2, 3],
        boundary_nodes=boundary_nodes or [0, 1, 2, 3],
        area=area,
        label=label,
        is_exterior=is_exterior,
    )


def _make_panel(
    sku: str = "TEST-PNL-001",
    panel_type: PanelType = PanelType.LOAD_BEARING,
    gauge: int = 16,
    min_length: float = 24.0,
    max_length: float = 300.0,
    fire_rating_hours: float = 0.0,
) -> Panel:
    """Create a minimal Panel for testing."""
    return Panel(
        sku=sku,
        name="Test Panel",
        panel_type=panel_type,
        gauge=gauge,
        stud_depth_inches=6.0,
        stud_spacing_inches=16.0,
        min_length_inches=min_length,
        max_length_inches=max_length,
        height_inches=96.0,
        fire_rating_hours=fire_rating_hours,
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
    pod_type: str = "bathroom",
    width: float = 60.0,
    depth: float = 96.0,
    clearance: float = 3.0,
) -> Pod:
    """Create a minimal Pod for testing."""
    return Pod(
        sku=sku,
        name="Test Pod",
        pod_type=pod_type,
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


def _make_classified_graph(
    n_walls: int = 4,
    n_rooms: int = 1,
    scale_factor: float = 1.0,
) -> ClassifiedWallGraph:
    """Create a simple classified wall graph with a rectangular room."""
    nodes = np.array([
        [80, 100],
        [530, 100],
        [530, 600],
        [80, 600],
    ], dtype=np.float64)

    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
    ], dtype=np.int64)

    segments = []
    for i, (s, e) in enumerate(edges[:n_walls]):
        delta = nodes[e] - nodes[s]
        length = float(np.linalg.norm(delta))
        angle = float(np.arctan2(delta[1], delta[0]) % np.pi)
        segments.append(WallSegment(
            edge_id=i,
            start_node=int(s),
            end_node=int(e),
            start_coord=nodes[s].copy(),
            end_coord=nodes[e].copy(),
            thickness=6.0,
            height=2700.0,
            wall_type=WallType.UNKNOWN,
            angle=angle,
            length=length,
            confidence=1.0,
        ))

    rooms = []
    if n_rooms >= 1:
        rooms.append(Room(
            room_id=0,
            boundary_edges=[0, 1, 2, 3],
            boundary_nodes=[0, 1, 2, 3],
            area=225000.0,
            label="Bedroom",
            is_exterior=False,
        ))

    # Add exterior room (should be excluded by the env)
    rooms.append(Room(
        room_id=99,
        boundary_edges=[],
        boundary_nodes=[],
        area=0.0,
        label="",
        is_exterior=True,
    ))

    graph = FinalizedGraph(
        nodes=nodes,
        edges=edges[:n_walls],
        wall_segments=segments,
        openings=[],
        rooms=rooms,
        page_width=612.0,
        page_height=792.0,
        scale_factor=scale_factor,
    )

    classifications = [
        WallClassification(
            edge_id=seg.edge_id,
            wall_type=WallType.LOAD_BEARING,
            fire_rating=FireRating.NONE,
            confidence=0.9,
            is_perimeter=True,
        )
        for seg in segments
    ]

    return ClassifiedWallGraph(
        graph=graph,
        classifications=classifications,
    )


# ---------------------------------------------------------------------------
# wall_type_to_panel_type
# ---------------------------------------------------------------------------


class TestWallTypeToPanelType:
    """Tests for wall_type_to_panel_type mapping."""

    def test_load_bearing_maps_to_load_bearing(self):
        result = wall_type_to_panel_type(WallType.LOAD_BEARING, FireRating.NONE)
        assert result == PanelType.LOAD_BEARING

    def test_partition_maps_to_partition(self):
        result = wall_type_to_panel_type(WallType.PARTITION, FireRating.NONE)
        assert result == PanelType.PARTITION

    def test_shear_maps_to_shear(self):
        result = wall_type_to_panel_type(WallType.SHEAR, FireRating.NONE)
        assert result == PanelType.SHEAR

    def test_exterior_maps_to_envelope(self):
        result = wall_type_to_panel_type(WallType.EXTERIOR, FireRating.NONE)
        assert result == PanelType.ENVELOPE

    def test_curtain_maps_to_envelope(self):
        result = wall_type_to_panel_type(WallType.CURTAIN, FireRating.NONE)
        assert result == PanelType.ENVELOPE

    def test_unknown_defaults_to_partition(self):
        result = wall_type_to_panel_type(WallType.UNKNOWN, FireRating.NONE)
        assert result == PanelType.PARTITION

    def test_fire_rated_overrides_wall_type(self):
        """Fire-rated walls always map to FIRE_RATED regardless of wall type."""
        for wall_type in WallType:
            result = wall_type_to_panel_type(wall_type, FireRating.HOUR_1)
            assert result == PanelType.FIRE_RATED

    def test_fire_rating_none_does_not_override(self):
        result = wall_type_to_panel_type(WallType.LOAD_BEARING, FireRating.NONE)
        assert result == PanelType.LOAD_BEARING

    def test_fire_rating_unknown_does_not_override(self):
        result = wall_type_to_panel_type(WallType.LOAD_BEARING, FireRating.UNKNOWN)
        assert result == PanelType.LOAD_BEARING

    def test_all_fire_ratings_override(self):
        """All non-NONE/non-UNKNOWN fire ratings should map to FIRE_RATED."""
        for fr in [FireRating.HOUR_1, FireRating.HOUR_2, FireRating.HOUR_3, FireRating.HOUR_4]:
            result = wall_type_to_panel_type(WallType.PARTITION, fr)
            assert result == PanelType.FIRE_RATED


# ---------------------------------------------------------------------------
# fire_rating_to_hours
# ---------------------------------------------------------------------------


class TestFireRatingToHours:
    """Tests for fire_rating_to_hours conversion."""

    def test_none_returns_zero(self):
        assert fire_rating_to_hours(FireRating.NONE) == 0.0

    def test_unknown_returns_zero(self):
        assert fire_rating_to_hours(FireRating.UNKNOWN) == 0.0

    def test_hour_values(self):
        assert fire_rating_to_hours(FireRating.HOUR_1) == 1.0
        assert fire_rating_to_hours(FireRating.HOUR_2) == 2.0
        assert fire_rating_to_hours(FireRating.HOUR_3) == 3.0
        assert fire_rating_to_hours(FireRating.HOUR_4) == 4.0


# ---------------------------------------------------------------------------
# encode_wall_segment
# ---------------------------------------------------------------------------


class TestEncodeWallSegment:
    """Tests for encode_wall_segment feature vector."""

    def test_output_shape(self):
        wall = _make_segment(length=100.0)
        cls = _make_classification()
        result = encode_wall_segment(wall, cls, [], scale=1.0)
        assert result.shape == (WALL_FEATURE_DIM,)
        assert result.dtype == np.float32

    def test_length_conversion_default_scale(self):
        """With scale=1.0, length in PDF units converts via 1/72 (inches)."""
        wall = _make_segment(length=72.0)  # exactly 1 inch in PDF units
        cls = _make_classification()
        result = encode_wall_segment(wall, cls, [], scale=1.0)
        assert abs(result[0] - 1.0) < 1e-5

    def test_length_conversion_custom_scale(self):
        """With scale != 1.0, conversion is scale/25.4."""
        wall = _make_segment(length=100.0)
        cls = _make_classification()
        scale = 25.4  # 1 PDF unit = 25.4 mm = 1 inch
        result = encode_wall_segment(wall, cls, [], scale=scale)
        assert abs(result[0] - 100.0) < 1e-5

    def test_thickness_included(self):
        wall = _make_segment(thickness=10.0)
        cls = _make_classification()
        result = encode_wall_segment(wall, cls, [], scale=1.0)
        # Thickness should be in result[1], converted to inches
        expected_thickness = 10.0 / PDF_UNITS_PER_INCH
        assert abs(result[1] - expected_thickness) < 1e-5

    def test_angle_normalized(self):
        wall = _make_segment(angle=math.pi / 2)
        cls = _make_classification()
        result = encode_wall_segment(wall, cls, [], scale=1.0)
        expected_normalized = (math.pi / 2) / math.pi  # 0.5
        assert abs(result[2] - expected_normalized) < 1e-5

    def test_wall_type_encoded(self):
        wall = _make_segment()
        cls = _make_classification(wall_type=WallType.PARTITION)
        result = encode_wall_segment(wall, cls, [], scale=1.0)
        assert result[3] == 1.0  # PARTITION is encoded as 1

    def test_fire_rating_encoded(self):
        wall = _make_segment()
        cls = _make_classification(fire_rating=FireRating.HOUR_2)
        result = encode_wall_segment(wall, cls, [], scale=1.0)
        assert result[4] == 2.0

    def test_confidence_encoded(self):
        wall = _make_segment()
        cls = _make_classification(confidence=0.85)
        result = encode_wall_segment(wall, cls, [], scale=1.0)
        assert abs(result[5] - 0.85) < 1e-5

    def test_perimeter_flag(self):
        wall = _make_segment()
        cls_perimeter = _make_classification(is_perimeter=True)
        cls_interior = _make_classification(is_perimeter=False)
        result_p = encode_wall_segment(wall, cls_perimeter, [], scale=1.0)
        result_i = encode_wall_segment(wall, cls_interior, [], scale=1.0)
        assert result_p[6] == 1.0
        assert result_i[6] == 0.0

    def test_openings_count(self):
        wall = _make_segment(length=200.0)
        cls = _make_classification()
        openings = [
            Opening(
                opening_type=OpeningType.DOOR,
                wall_edge_id=0,
                position_along_wall=0.3,
                width=36.0,
                height=80.0,
            ),
            Opening(
                opening_type=OpeningType.WINDOW,
                wall_edge_id=0,
                position_along_wall=0.7,
                width=24.0,
                height=48.0,
            ),
        ]
        result = encode_wall_segment(wall, cls, openings, scale=1.0)
        assert result[7] == 2.0

    def test_opening_coverage_ratio(self):
        wall = _make_segment(length=144.0)  # 2 inches in PDF units
        cls = _make_classification()
        # Opening width = 72 PDF units = 1 inch, wall = 2 inches
        openings = [
            Opening(
                opening_type=OpeningType.DOOR,
                wall_edge_id=0,
                position_along_wall=0.5,
                width=72.0,
                height=80.0,
            ),
        ]
        result = encode_wall_segment(wall, cls, openings, scale=1.0)
        assert abs(result[8] - 0.5) < 1e-5

    def test_opening_coverage_clamped_to_one(self):
        """Coverage ratio should not exceed 1.0."""
        wall = _make_segment(length=72.0)  # 1 inch
        cls = _make_classification()
        # Opening wider than wall
        openings = [
            Opening(
                opening_type=OpeningType.DOOR,
                wall_edge_id=0,
                position_along_wall=0.5,
                width=144.0,  # 2 inches
                height=80.0,
            ),
        ]
        result = encode_wall_segment(wall, cls, openings, scale=1.0)
        assert result[8] <= 1.0

    def test_cos_angle(self):
        angle = math.pi / 3
        wall = _make_segment(angle=angle)
        cls = _make_classification()
        result = encode_wall_segment(wall, cls, [], scale=1.0)
        assert abs(result[11] - math.cos(angle)) < 1e-5

    def test_zero_length_wall(self):
        """A zero-length wall should not cause division by zero."""
        wall = _make_segment(length=0.0)
        cls = _make_classification()
        openings = [
            Opening(
                opening_type=OpeningType.DOOR,
                wall_edge_id=0,
                position_along_wall=0.0,
                width=36.0,
                height=80.0,
            ),
        ]
        result = encode_wall_segment(wall, cls, openings, scale=1.0)
        assert result[8] == 0.0  # coverage should be 0, not NaN


# ---------------------------------------------------------------------------
# encode_room
# ---------------------------------------------------------------------------


class TestEncodeRoom:
    """Tests for encode_room feature vector."""

    def test_output_shape(self):
        room = _make_room()
        nodes = np.array([
            [80, 100], [530, 100], [530, 600], [80, 600],
        ], dtype=np.float64)
        result = encode_room(room, nodes, scale=1.0)
        assert result.shape == (ROOM_FEATURE_DIM,)
        assert result.dtype == np.float32

    def test_area_conversion(self):
        """Area should be converted to square inches."""
        room = _make_room(area=72.0 * 72.0)  # 1 sq inch at 72 units/inch
        nodes = np.array([
            [0, 0], [72, 0], [72, 72], [0, 72],
        ], dtype=np.float64)
        result = encode_room(room, nodes, scale=1.0)
        assert abs(result[0] - 1.0) < 1e-3

    def test_width_and_depth(self):
        """Width and depth computed from bounding box of boundary nodes."""
        nodes = np.array([
            [0, 0], [144, 0], [144, 72], [0, 72],
        ], dtype=np.float64)
        room = _make_room(boundary_nodes=[0, 1, 2, 3])
        result = encode_room(room, nodes, scale=1.0)
        expected_width = 144.0 / PDF_UNITS_PER_INCH
        expected_depth = 72.0 / PDF_UNITS_PER_INCH
        assert abs(result[1] - expected_width) < 1e-5
        assert abs(result[2] - expected_depth) < 1e-5

    def test_aspect_ratio_clamped(self):
        """Aspect ratio should be clamped to [0.1, 10]."""
        # Very wide room
        nodes = np.array([
            [0, 0], [10000, 0], [10000, 1], [0, 1],
        ], dtype=np.float64)
        room = _make_room(boundary_nodes=[0, 1, 2, 3])
        result = encode_room(room, nodes, scale=1.0)
        assert result[3] <= 10.0

    def test_num_boundary_edges(self):
        room = _make_room(boundary_edges=[0, 1, 2, 3, 4])
        nodes = np.array([
            [0, 0], [100, 0], [100, 100], [50, 150], [0, 100],
        ], dtype=np.float64)
        result = encode_room(room, nodes, scale=1.0)
        assert result[4] == 5.0

    def test_label_encoding(self):
        """Known room labels should produce non-zero codes."""
        nodes = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
        room_bed = _make_room(label="Bedroom")
        room_bath = _make_room(label="Bathroom")
        room_kitchen = _make_room(label="Kitchen")
        result_bed = encode_room(room_bed, nodes, scale=1.0)
        result_bath = encode_room(room_bath, nodes, scale=1.0)
        result_kitchen = encode_room(room_kitchen, nodes, scale=1.0)
        assert result_bed[5] == 1.0   # bedroom -> 1
        assert result_bath[5] == 2.0  # bathroom -> 2
        assert result_kitchen[5] == 3.0  # kitchen -> 3

    def test_unknown_label_defaults_to_zero(self):
        nodes = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
        room = _make_room(label="observatory")  # not in mapping
        result = encode_room(room, nodes, scale=1.0)
        assert result[5] == 0.0

    def test_empty_boundary_nodes(self):
        """Room with no boundary nodes should produce zero width/depth."""
        room = Room(
            room_id=0,
            boundary_edges=[],
            boundary_nodes=[],
            area=50000.0,
            label="",
        )
        nodes = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
        result = encode_room(room, nodes, scale=1.0)
        # Width and depth should both be at minimum (0.01)
        assert result[1] == pytest.approx(0.01, abs=1e-5)
        assert result[2] == pytest.approx(0.01, abs=1e-5)


# ---------------------------------------------------------------------------
# encode_panel_candidate
# ---------------------------------------------------------------------------


class TestEncodePanelCandidate:
    """Tests for encode_panel_candidate feature vector."""

    def test_output_shape(self):
        panel = _make_panel()
        result = encode_panel_candidate(panel)
        assert result.shape == (CANDIDATE_PANEL_DIM,)
        assert result.dtype == np.float32

    def test_min_max_length(self):
        panel = _make_panel(min_length=24.0, max_length=300.0)
        result = encode_panel_candidate(panel)
        assert result[0] == 24.0
        assert result[1] == 300.0

    def test_gauge(self):
        panel = _make_panel(gauge=20)
        result = encode_panel_candidate(panel)
        assert result[2] == 20.0

    def test_fire_rating(self):
        panel = _make_panel(fire_rating_hours=2.0)
        result = encode_panel_candidate(panel)
        assert result[5] == 2.0

    def test_panel_type_encoded(self):
        """Panel type should be encoded as ordinal index in PanelType."""
        panel_lb = _make_panel(panel_type=PanelType.LOAD_BEARING)
        panel_part = _make_panel(panel_type=PanelType.PARTITION)
        result_lb = encode_panel_candidate(panel_lb)
        result_part = encode_panel_candidate(panel_part)
        # LOAD_BEARING and PARTITION should have different type codes
        assert result_lb[9] != result_part[9]


# ---------------------------------------------------------------------------
# encode_pod_candidate
# ---------------------------------------------------------------------------


class TestEncodePodCandidate:
    """Tests for encode_pod_candidate feature vector."""

    def test_output_shape(self):
        pod = _make_pod()
        result = encode_pod_candidate(pod)
        assert result.shape == (CANDIDATE_POD_DIM,)
        assert result.dtype == np.float32

    def test_dimensions(self):
        pod = _make_pod(width=60.0, depth=96.0)
        result = encode_pod_candidate(pod)
        assert result[0] == 60.0
        assert result[1] == 96.0

    def test_num_trades(self):
        pod = _make_pod()
        result = encode_pod_candidate(pod)
        assert result[6] == 2.0  # plumbing + electrical

    def test_unit_cost_normalized(self):
        """Unit cost is stored in thousands."""
        pod = _make_pod()
        result = encode_pod_candidate(pod)
        assert abs(result[8] - 12.5) < 1e-3  # 12500 / 1000

    def test_lead_time(self):
        pod = _make_pod()
        result = encode_pod_candidate(pod)
        assert result[9] == 21.0


# ---------------------------------------------------------------------------
# encode_observation
# ---------------------------------------------------------------------------


class TestEncodeObservation:
    """Tests for the full observation dictionary."""

    def test_returns_all_keys(self):
        cg = _make_classified_graph()
        obs = encode_observation(
            classified_graph=cg,
            wall_assignments={},
            room_assignments={},
            current_wall_idx=0,
            current_room_idx=None,
            panel_candidates=[],
            pod_candidates=[],
            phase="panelization",
        )
        expected_keys = {
            "wall_features", "room_features", "wall_assigned",
            "room_assigned", "current_target", "candidate_features",
            "candidate_mask", "phase", "progress",
        }
        assert set(obs.keys()) == expected_keys

    def test_wall_features_shape(self):
        cg = _make_classified_graph()
        obs = encode_observation(
            classified_graph=cg,
            wall_assignments={},
            room_assignments={},
            current_wall_idx=0,
            current_room_idx=None,
            panel_candidates=[],
            pod_candidates=[],
            phase="panelization",
        )
        assert obs["wall_features"].shape == (MAX_WALLS, WALL_FEATURE_DIM)

    def test_room_features_shape(self):
        cg = _make_classified_graph()
        obs = encode_observation(
            classified_graph=cg,
            wall_assignments={},
            room_assignments={},
            current_wall_idx=None,
            current_room_idx=0,
            panel_candidates=[],
            pod_candidates=[],
            phase="placement",
        )
        assert obs["room_features"].shape == (MAX_ROOMS, ROOM_FEATURE_DIM)

    def test_assignment_masks_initially_zero(self):
        cg = _make_classified_graph()
        obs = encode_observation(
            classified_graph=cg,
            wall_assignments={},
            room_assignments={},
            current_wall_idx=0,
            current_room_idx=None,
            panel_candidates=[],
            pod_candidates=[],
            phase="panelization",
        )
        assert np.all(obs["wall_assigned"] == 0.0)
        assert np.all(obs["room_assigned"] == 0.0)

    def test_wall_assigned_reflects_assignments(self):
        cg = _make_classified_graph()
        # Wall with edge_id=0 is assigned
        obs = encode_observation(
            classified_graph=cg,
            wall_assignments={0: [("SKU-001", 100.0)]},
            room_assignments={},
            current_wall_idx=1,
            current_room_idx=None,
            panel_candidates=[],
            pod_candidates=[],
            phase="panelization",
        )
        assert obs["wall_assigned"][0] == 1.0
        assert obs["wall_assigned"][1] == 0.0

    def test_phase_encoding_panelization(self):
        cg = _make_classified_graph()
        obs = encode_observation(
            classified_graph=cg,
            wall_assignments={},
            room_assignments={},
            current_wall_idx=0,
            current_room_idx=None,
            panel_candidates=[],
            pod_candidates=[],
            phase="panelization",
        )
        np.testing.assert_array_equal(obs["phase"], [1.0, 0.0])

    def test_phase_encoding_placement(self):
        cg = _make_classified_graph()
        obs = encode_observation(
            classified_graph=cg,
            wall_assignments={},
            room_assignments={},
            current_wall_idx=None,
            current_room_idx=0,
            panel_candidates=[],
            pod_candidates=[],
            phase="placement",
        )
        np.testing.assert_array_equal(obs["phase"], [0.0, 1.0])

    def test_progress_reflects_assignments(self):
        cg = _make_classified_graph(n_walls=4, n_rooms=1)
        # 2 of 4 walls assigned, 0 of 1 rooms
        obs = encode_observation(
            classified_graph=cg,
            wall_assignments={0: [("SKU", 100.0)], 1: [("SKU", 100.0)]},
            room_assignments={},
            current_wall_idx=2,
            current_room_idx=None,
            panel_candidates=[],
            pod_candidates=[],
            phase="panelization",
        )
        assert abs(obs["progress"][0] - 0.5) < 1e-5  # 2/4 walls
        assert obs["progress"][1] == 0.0  # 0/1 rooms

    def test_candidate_features_populated_for_panels(self):
        cg = _make_classified_graph()
        panels = [_make_panel(sku=f"PNL-{i}") for i in range(3)]
        obs = encode_observation(
            classified_graph=cg,
            wall_assignments={},
            room_assignments={},
            current_wall_idx=0,
            current_room_idx=None,
            panel_candidates=panels,
            pod_candidates=[],
            phase="panelization",
        )
        # First 3 candidates should be populated, rest zeros
        assert obs["candidate_mask"][0] == 1.0
        assert obs["candidate_mask"][1] == 1.0
        assert obs["candidate_mask"][2] == 1.0
        assert obs["candidate_mask"][3] == 0.0

    def test_candidate_features_capped_at_max(self):
        cg = _make_classified_graph()
        panels = [_make_panel(sku=f"PNL-{i}") for i in range(MAX_CANDIDATES + 5)]
        obs = encode_observation(
            classified_graph=cg,
            wall_assignments={},
            room_assignments={},
            current_wall_idx=0,
            current_room_idx=None,
            panel_candidates=panels,
            pod_candidates=[],
            phase="panelization",
        )
        # Only MAX_CANDIDATES should be populated
        assert np.sum(obs["candidate_mask"]) == MAX_CANDIDATES

    def test_current_target_populated_panelization(self):
        cg = _make_classified_graph()
        obs = encode_observation(
            classified_graph=cg,
            wall_assignments={},
            room_assignments={},
            current_wall_idx=0,
            current_room_idx=None,
            panel_candidates=[],
            pod_candidates=[],
            phase="panelization",
        )
        # current_target should have non-zero wall features
        assert np.any(obs["current_target"] != 0.0)

    def test_current_target_populated_placement(self):
        cg = _make_classified_graph()
        obs = encode_observation(
            classified_graph=cg,
            wall_assignments={},
            room_assignments={},
            current_wall_idx=None,
            current_room_idx=0,
            panel_candidates=[],
            pod_candidates=[],
            phase="placement",
        )
        # current_target should have non-zero room features
        assert np.any(obs["current_target"] != 0.0)
