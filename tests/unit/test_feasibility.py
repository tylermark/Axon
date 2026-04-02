"""Unit tests for src/feasibility/ — Q-017.

Tests the four feasibility submodules:
  - calculator.py  (FS-001): prefab coverage metrics
  - blockers.py    (FS-002): blocker identification and categorization
  - suggestions.py (FS-003): design modification suggestions
  - report.py      (FS-004): full report orchestration
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
from docs.interfaces.drl_output import (
    PanelAssignment,
    PanelizationResult,
    PanelMap,
    PlacementMap,
    ProductPlacement,
    RoomPlacement,
    WallPanelization,
)
from docs.interfaces.feasibility_report import (
    Blocker,
    BlockerCategory,
    CoverageMetrics,
    FeasibilityReport,
    FeasibilitySummary,
    FloorScore,
    SuggestionType,
)
from docs.interfaces.graph_to_serializer import (
    FinalizedGraph,
    Room,
    WallSegment,
    WallType,
)
from src.feasibility.blockers import identify_blockers
from src.feasibility.calculator import calculate_coverage
from src.feasibility.report import generate_feasibility_report
from src.feasibility.suggestions import generate_suggestions
from src.knowledge_graph.loader import KnowledgeGraphStore, load_knowledge_graph

# ---------------------------------------------------------------------------
# Module-scoped KG store (loads real catalog data once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kg_store() -> KnowledgeGraphStore:
    """Load the real KG catalog for tests that need product data."""
    return load_knowledge_graph()


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

_PANEL_SKU = "CAP-PNL-LB-16GA-600D-16OC-96H"


def _make_wall_segment(
    edge_id: int,
    length: float,
    angle: float = 0.0,
    wall_type: WallType = WallType.LOAD_BEARING,
    thickness: float = 6.0,
) -> WallSegment:
    """Build a WallSegment with sensible defaults."""
    start = np.array([0.0, edge_id * 100.0], dtype=np.float64)
    end = start + np.array([length * math.cos(angle), length * math.sin(angle)], dtype=np.float64)
    return WallSegment(
        edge_id=edge_id,
        start_node=edge_id * 2,
        end_node=edge_id * 2 + 1,
        start_coord=start,
        end_coord=end,
        thickness=thickness,
        height=96.0,
        wall_type=wall_type,
        angle=angle,
        length=length,
        confidence=0.95,
    )


def _make_classification(
    edge_id: int,
    wall_type: WallType = WallType.LOAD_BEARING,
    fire_rating: FireRating = FireRating.NONE,
    confidence: float = 0.95,
) -> WallClassification:
    """Build a WallClassification with sensible defaults."""
    return WallClassification(
        edge_id=edge_id,
        wall_type=wall_type,
        fire_rating=fire_rating,
        confidence=confidence,
    )


def _make_wall_panelization(
    edge_id: int,
    wall_length: float,
    is_panelizable: bool = True,
    rejection_reason: str = "",
    panel_sku: str = _PANEL_SKU,
) -> WallPanelization:
    """Build a WallPanelization entry."""
    panels: list[PanelAssignment] = []
    if is_panelizable:
        panels.append(
            PanelAssignment(
                panel_sku=panel_sku,
                cut_length_inches=wall_length,
                position_along_wall=0.0,
                panel_index=0,
            )
        )
    return WallPanelization(
        edge_id=edge_id,
        wall_length_inches=wall_length,
        panels=panels,
        requires_splice=False,
        is_panelizable=is_panelizable,
        rejection_reason=rejection_reason,
        total_material_inches=wall_length if is_panelizable else 0.0,
    )


def _make_room_placement(
    room_id: int,
    label: str = "Bathroom",
    area_sqft: float = 80.0,
    is_eligible: bool = True,
    has_pod: bool = True,
    pod_sku: str = "CAP-POD-BATH-STD",
    rejection_reason: str = "",
) -> RoomPlacement:
    """Build a RoomPlacement entry."""
    placement: ProductPlacement | None = None
    if has_pod and is_eligible:
        placement = ProductPlacement(
            pod_sku=pod_sku,
            position=np.array([100.0, 100.0], dtype=np.float64),
            orientation_deg=0.0,
            clearance_met=True,
        )
    return RoomPlacement(
        room_id=room_id,
        room_label=label,
        room_area_sqft=area_sqft,
        placement=placement,
        is_eligible=is_eligible,
        rejection_reason=rejection_reason,
    )


def _make_room(
    room_id: int,
    label: str = "Bathroom",
    area: float = 11520.0,  # ~80 sqft in PDF user units squared
) -> Room:
    """Build a Room for the FinalizedGraph."""
    return Room(
        room_id=room_id,
        boundary_edges=[0, 1, 2, 3],
        boundary_nodes=[0, 1, 2, 3],
        area=area,
        label=label,
    )


def _make_panelization_result(
    wall_configs: list[dict],
    room_configs: list[dict] | None = None,
    spur_score: float = 0.75,
) -> PanelizationResult:
    """Build a complete PanelizationResult from simplified configs.

    Args:
        wall_configs: List of dicts, each with keys:
            - edge_id (int)
            - length (float, inches)
            - is_panelizable (bool, default True)
            - rejection_reason (str, default "")
            - angle (float, radians, default 0.0)
            - wall_type (WallType, default LOAD_BEARING)
            - fire_rating (FireRating, default NONE)
        room_configs: Optional list of dicts, each with keys:
            - room_id (int)
            - label (str, default "Bathroom")
            - area_sqft (float, default 80.0)
            - is_eligible (bool, default True)
            - has_pod (bool, default True)
            - rejection_reason (str, default "")
        spur_score: SPUR score for the result.

    Returns:
        A fully constructed PanelizationResult.
    """
    if room_configs is None:
        room_configs = []

    wall_segments = []
    classifications = []
    wall_panelizations = []
    nodes_list = []

    for wc in wall_configs:
        eid = wc["edge_id"]
        length = wc["length"]
        angle = wc.get("angle", 0.0)
        wtype = wc.get("wall_type", WallType.LOAD_BEARING)
        fr = wc.get("fire_rating", FireRating.NONE)
        is_pan = wc.get("is_panelizable", True)
        reason = wc.get("rejection_reason", "")

        seg = _make_wall_segment(eid, length, angle=angle, wall_type=wtype)
        wall_segments.append(seg)
        nodes_list.append(seg.start_coord)
        nodes_list.append(seg.end_coord)

        classifications.append(
            _make_classification(eid, wall_type=wtype, fire_rating=fr)
        )
        wall_panelizations.append(
            _make_wall_panelization(eid, length, is_panelizable=is_pan, rejection_reason=reason)
        )

    nodes_arr = np.array(nodes_list, dtype=np.float64) if nodes_list else np.zeros((0, 2))
    num_edges = len(wall_segments)
    edges_arr = np.column_stack([
        np.arange(0, num_edges * 2, 2, dtype=np.int64),
        np.arange(1, num_edges * 2 + 1, 2, dtype=np.int64),
    ]) if num_edges > 0 else np.zeros((0, 2), dtype=np.int64)

    rooms = [_make_room(rc.get("room_id", i), rc.get("label", "Bathroom")) for i, rc in enumerate(room_configs)]

    graph = FinalizedGraph(
        nodes=nodes_arr,
        edges=edges_arr,
        wall_segments=wall_segments,
        openings=[],
        rooms=rooms,
        page_width=612.0,
        page_height=792.0,
    )

    cwg = ClassifiedWallGraph(
        graph=graph,
        classifications=classifications,
    )

    room_placements = []
    for rc in room_configs:
        room_placements.append(
            _make_room_placement(
                room_id=rc.get("room_id", 0),
                label=rc.get("label", "Bathroom"),
                area_sqft=rc.get("area_sqft", 80.0),
                is_eligible=rc.get("is_eligible", True),
                has_pod=rc.get("has_pod", True),
                rejection_reason=rc.get("rejection_reason", ""),
            )
        )

    panelized_count = sum(1 for wp in wall_panelizations if wp.is_panelizable)
    panel_map = PanelMap(
        walls=wall_panelizations,
        panelized_wall_count=panelized_count,
        total_wall_count=len(wall_panelizations),
    )

    placed_count = sum(1 for rp in room_placements if rp.placement is not None)
    eligible_count = sum(1 for rp in room_placements if rp.is_eligible)
    placement_map = PlacementMap(
        rooms=room_placements,
        placed_room_count=placed_count,
        eligible_room_count=eligible_count,
        total_room_count=len(room_placements),
    )

    total_length = sum(wc["length"] for wc in wall_configs)
    panelized_length = sum(
        wc["length"] for wc in wall_configs if wc.get("is_panelizable", True)
    )
    coverage_pct = (panelized_length / total_length * 100.0) if total_length > 0 else 0.0

    return PanelizationResult(
        source_graph=cwg,
        panel_map=panel_map,
        placement_map=placement_map,
        spur_score=spur_score,
        coverage_percentage=coverage_pct,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. calculator.py — calculate_coverage
# ══════════════════════════════════════════════════════════════════════════════


class TestCalculateCoverage:
    """Tests for calculate_coverage (FS-001)."""

    def test_all_walls_panelized_100pct(self) -> None:
        """All walls panelized gives 100% by_wall_length_pct."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 120.0, "is_panelizable": True},
            {"edge_id": 1, "length": 96.0, "is_panelizable": True},
            {"edge_id": 2, "length": 144.0, "is_panelizable": True},
        ])
        metrics = calculate_coverage(result)
        assert metrics.by_wall_length_pct == pytest.approx(100.0)

    def test_some_walls_unpanelized(self) -> None:
        """Partial panelization gives correct percentage."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 100.0, "is_panelizable": True},
            {"edge_id": 1, "length": 100.0, "is_panelizable": False, "rejection_reason": "too long"},
        ])
        metrics = calculate_coverage(result)
        assert metrics.by_wall_length_pct == pytest.approx(50.0)
        assert metrics.panelized_wall_length_inches == pytest.approx(100.0)
        assert metrics.total_wall_length_inches == pytest.approx(200.0)

    def test_no_panels_zero_coverage(self) -> None:
        """No walls panelized gives 0% coverage."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 120.0, "is_panelizable": False, "rejection_reason": "no panel"},
            {"edge_id": 1, "length": 96.0, "is_panelizable": False, "rejection_reason": "no panel"},
        ])
        metrics = calculate_coverage(result)
        assert metrics.by_wall_length_pct == pytest.approx(0.0)
        assert metrics.panelized_wall_length_inches == pytest.approx(0.0)

    def test_area_coverage_uses_wall_height(self) -> None:
        """Area coverage accounts for wall height (default 96 in without KG)."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 120.0, "is_panelizable": True},
        ])
        metrics = calculate_coverage(result)
        # 120 * 96 = 11520 sq-in = 80 sqft
        assert metrics.total_wall_area_sqft == pytest.approx(80.0)
        assert metrics.panelized_wall_area_sqft == pytest.approx(80.0)
        assert metrics.by_area_pct == pytest.approx(100.0)

    def test_area_coverage_partial(self) -> None:
        """Partial panelization area coverage is proportional."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 100.0, "is_panelizable": True},
            {"edge_id": 1, "length": 100.0, "is_panelizable": False, "rejection_reason": "geometry"},
        ])
        metrics = calculate_coverage(result)
        # Both walls same height → area pct equals length pct
        assert metrics.by_area_pct == pytest.approx(50.0)

    def test_cost_coverage_with_kg_store(self, kg_store: KnowledgeGraphStore) -> None:
        """Cost coverage is non-zero when KG store is provided."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 120.0, "is_panelizable": True,
             "wall_type": WallType.LOAD_BEARING, "fire_rating": FireRating.NONE},
            {"edge_id": 1, "length": 120.0, "is_panelizable": False,
             "rejection_reason": "too long",
             "wall_type": WallType.LOAD_BEARING, "fire_rating": FireRating.NONE},
        ])
        metrics = calculate_coverage(result, store=kg_store)
        assert metrics.by_cost_pct == pytest.approx(50.0, abs=5.0)
        assert metrics.by_cost_pct > 0.0

    def test_cost_coverage_zero_without_store(self) -> None:
        """Cost coverage is 0 when no KG store is provided."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 120.0, "is_panelizable": True},
        ])
        metrics = calculate_coverage(result, store=None)
        assert metrics.by_cost_pct == pytest.approx(0.0)

    def test_coverage_metrics_type(self) -> None:
        """Return type is CoverageMetrics."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 100.0, "is_panelizable": True},
        ])
        metrics = calculate_coverage(result)
        assert isinstance(metrics, CoverageMetrics)

    def test_empty_wall_list(self) -> None:
        """Empty wall list results in 0% coverage, no errors."""
        result = _make_panelization_result([])
        metrics = calculate_coverage(result)
        assert metrics.by_wall_length_pct == pytest.approx(0.0)
        assert metrics.total_wall_length_inches == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. blockers.py — identify_blockers
# ══════════════════════════════════════════════════════════════════════════════


class TestIdentifyBlockers:
    """Tests for identify_blockers (FS-002)."""

    def test_wall_exceeding_machine_max_length(self, kg_store: KnowledgeGraphStore) -> None:
        """Wall longer than max fabrication length produces MACHINE_LIMITS blocker."""
        # Max fab length is 420" (HW3500). Use 500".
        result = _make_panelization_result([
            {"edge_id": 0, "length": 500.0, "is_panelizable": False,
             "rejection_reason": "Wall exceeds max length of 420 inches"},
        ])
        blockers = identify_blockers(result, kg_store)
        assert len(blockers) >= 1
        ml_blockers = [b for b in blockers if b.category == BlockerCategory.MACHINE_LIMITS]
        assert len(ml_blockers) >= 1
        assert ml_blockers[0].severity == 1.0

    def test_machine_limits_blocker_from_length_check(self, kg_store: KnowledgeGraphStore) -> None:
        """Even a panelizable wall that exceeds max fab length gets a MACHINE_LIMITS blocker."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 500.0, "is_panelizable": True},
        ])
        blockers = identify_blockers(result, kg_store)
        ml_blockers = [b for b in blockers if b.category == BlockerCategory.MACHINE_LIMITS]
        assert len(ml_blockers) >= 1
        assert ml_blockers[0].affected_edge_ids == [0]
        assert ml_blockers[0].severity == 1.0

    def test_non_orthogonal_wall_geometry_blocker(self, kg_store: KnowledgeGraphStore) -> None:
        """Non-orthogonal wall (e.g. 45 degrees) produces GEOMETRY blocker."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 120.0, "angle": math.radians(45.0),
             "is_panelizable": False, "rejection_reason": "non-orthogonal angle"},
        ])
        blockers = identify_blockers(result, kg_store)
        geo_blockers = [b for b in blockers if b.category == BlockerCategory.GEOMETRY]
        assert len(geo_blockers) >= 1
        assert geo_blockers[0].severity < 1.0

    def test_room_without_pod_clearance_blocker(self, kg_store: KnowledgeGraphStore) -> None:
        """Eligible room without pod and 'clearance' reason produces CLEARANCE blocker."""
        result = _make_panelization_result(
            wall_configs=[{"edge_id": 0, "length": 120.0}],
            room_configs=[
                {"room_id": 1, "label": "Bathroom", "area_sqft": 30.0,
                 "is_eligible": True, "has_pod": False,
                 "rejection_reason": "insufficient clearance for pod"},
            ],
        )
        blockers = identify_blockers(result, kg_store)
        cl_blockers = [b for b in blockers if b.category == BlockerCategory.CLEARANCE]
        assert len(cl_blockers) >= 1
        assert cl_blockers[0].affected_room_ids == [1]

    def test_room_without_pod_product_gap_blocker(self, kg_store: KnowledgeGraphStore) -> None:
        """Eligible room without pod and 'no product' reason produces PRODUCT_GAP blocker."""
        result = _make_panelization_result(
            wall_configs=[{"edge_id": 0, "length": 120.0}],
            room_configs=[
                {"room_id": 2, "label": "Laundry", "area_sqft": 60.0,
                 "is_eligible": True, "has_pod": False,
                 "rejection_reason": "no compatible pod found for this room type"},
            ],
        )
        blockers = identify_blockers(result, kg_store)
        pg_blockers = [b for b in blockers if b.category == BlockerCategory.PRODUCT_GAP]
        assert len(pg_blockers) >= 1

    def test_blocker_severity_hard_vs_soft(self, kg_store: KnowledgeGraphStore) -> None:
        """Hard blockers have severity 1.0; soft blockers have severity < 1.0."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 500.0, "is_panelizable": False,
             "rejection_reason": "Wall exceeds max length"},
            {"edge_id": 1, "length": 120.0, "angle": math.radians(45.0),
             "is_panelizable": False, "rejection_reason": "non-orthogonal angle"},
        ])
        blockers = identify_blockers(result, kg_store)
        ml_blockers = [b for b in blockers if b.category == BlockerCategory.MACHINE_LIMITS]
        geo_blockers = [b for b in blockers if b.category == BlockerCategory.GEOMETRY]
        assert all(b.severity == 1.0 for b in ml_blockers)
        assert all(b.severity < 1.0 for b in geo_blockers)

    def test_no_blockers_all_panelized(self, kg_store: KnowledgeGraphStore) -> None:
        """When everything is panelized and within limits, no blockers."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 120.0, "is_panelizable": True},
            {"edge_id": 1, "length": 96.0, "is_panelizable": True},
        ])
        blockers = identify_blockers(result, kg_store)
        assert blockers == []

    def test_blockers_sorted_by_severity(self, kg_store: KnowledgeGraphStore) -> None:
        """Blockers are sorted by severity descending (hardest first)."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 500.0, "is_panelizable": False,
             "rejection_reason": "Wall exceeds max length"},
            {"edge_id": 1, "length": 120.0, "angle": math.radians(45.0),
             "is_panelizable": False, "rejection_reason": "non-orthogonal angle"},
        ])
        blockers = identify_blockers(result, kg_store)
        assert len(blockers) >= 2
        for i in range(len(blockers) - 1):
            assert blockers[i].severity >= blockers[i + 1].severity

    def test_blocker_ids_are_unique(self, kg_store: KnowledgeGraphStore) -> None:
        """All blocker IDs are unique."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 500.0, "is_panelizable": False,
             "rejection_reason": "too long"},
            {"edge_id": 1, "length": 120.0, "is_panelizable": False,
             "rejection_reason": "no compatible panel"},
            {"edge_id": 2, "length": 96.0, "angle": math.radians(45.0),
             "is_panelizable": False, "rejection_reason": "curved wall"},
        ])
        blockers = identify_blockers(result, kg_store)
        ids = [b.blocker_id for b in blockers]
        assert len(ids) == len(set(ids))

    def test_room_no_reason_defaults_to_clearance(self, kg_store: KnowledgeGraphStore) -> None:
        """Eligible room with no pod and no rejection_reason defaults to CLEARANCE."""
        result = _make_panelization_result(
            wall_configs=[{"edge_id": 0, "length": 120.0}],
            room_configs=[
                {"room_id": 1, "label": "Closet", "area_sqft": 25.0,
                 "is_eligible": True, "has_pod": False, "rejection_reason": ""},
            ],
        )
        blockers = identify_blockers(result, kg_store)
        room_blockers = [b for b in blockers if b.affected_room_ids]
        assert len(room_blockers) >= 1
        assert room_blockers[0].category == BlockerCategory.CLEARANCE


# ══════════════════════════════════════════════════════════════════════════════
# 3. suggestions.py — generate_suggestions
# ══════════════════════════════════════════════════════════════════════════════


class TestGenerateSuggestions:
    """Tests for generate_suggestions (FS-003)."""

    def test_machine_limits_wall_shorten(self, kg_store: KnowledgeGraphStore) -> None:
        """MACHINE_LIMITS blocker on a long wall produces WALL_SHORTEN suggestion."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 500.0, "is_panelizable": False,
             "rejection_reason": "Wall exceeds max length of 420 inches"},
        ])
        blockers = identify_blockers(result, kg_store)
        suggestions = generate_suggestions(blockers, result, kg_store)
        shorten = [s for s in suggestions if s.suggestion_type == SuggestionType.WALL_SHORTEN]
        assert len(shorten) >= 1
        assert shorten[0].resolves_blocker_ids
        assert shorten[0].affected_edge_ids == [0]

    def test_geometry_wall_straighten(self, kg_store: KnowledgeGraphStore) -> None:
        """GEOMETRY blocker produces WALL_STRAIGHTEN suggestion."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 120.0, "angle": math.radians(45.0),
             "is_panelizable": True},
        ])
        # Manually build a geometry blocker
        blockers = [
            Blocker(
                blocker_id="BLK-001",
                category=BlockerCategory.GEOMETRY,
                description="Wall #0 is non-orthogonal (45.0 deg).",
                affected_edge_ids=[0],
                severity=0.7,
            )
        ]
        suggestions = generate_suggestions(blockers, result, kg_store)
        straighten = [s for s in suggestions if s.suggestion_type == SuggestionType.WALL_STRAIGHTEN]
        assert len(straighten) >= 1
        assert "BLK-001" in straighten[0].resolves_blocker_ids

    def test_clearance_room_resize(self, kg_store: KnowledgeGraphStore) -> None:
        """CLEARANCE blocker produces ROOM_RESIZE suggestion."""
        result = _make_panelization_result(
            wall_configs=[{"edge_id": 0, "length": 120.0}],
            room_configs=[
                {"room_id": 1, "label": "bathroom", "area_sqft": 30.0,
                 "is_eligible": True, "has_pod": False,
                 "rejection_reason": "too small"},
            ],
        )
        blockers = [
            Blocker(
                blocker_id="BLK-001",
                category=BlockerCategory.CLEARANCE,
                description="Room #1 (bathroom, 30.0 sqft): too small.",
                affected_room_ids=[1],
                severity=0.8,
            )
        ]
        suggestions = generate_suggestions(blockers, result, kg_store)
        resize = [s for s in suggestions if s.suggestion_type == SuggestionType.ROOM_RESIZE]
        assert len(resize) >= 1
        assert resize[0].affected_room_ids == [1]

    def test_suggestions_sorted_by_coverage_gain(self, kg_store: KnowledgeGraphStore) -> None:
        """Suggestions are sorted by estimated_coverage_gain_pct descending."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 200.0, "is_panelizable": False,
             "rejection_reason": "too long"},
            {"edge_id": 1, "length": 50.0, "is_panelizable": False,
             "rejection_reason": "too long"},
        ])
        blockers = identify_blockers(result, kg_store)
        suggestions = generate_suggestions(blockers, result, kg_store)
        if len(suggestions) >= 2:
            for i in range(len(suggestions) - 1):
                assert suggestions[i].estimated_coverage_gain_pct >= suggestions[i + 1].estimated_coverage_gain_pct

    def test_no_suggestions_when_no_blockers(self, kg_store: KnowledgeGraphStore) -> None:
        """No blockers means no suggestions."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 120.0, "is_panelizable": True},
        ])
        suggestions = generate_suggestions([], result, kg_store)
        assert suggestions == []

    def test_suggestion_ids_are_unique(self, kg_store: KnowledgeGraphStore) -> None:
        """All suggestion IDs are unique."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 500.0, "is_panelizable": False,
             "rejection_reason": "too long"},
            {"edge_id": 1, "length": 120.0, "is_panelizable": False,
             "rejection_reason": "no compatible panel"},
        ])
        blockers = identify_blockers(result, kg_store)
        suggestions = generate_suggestions(blockers, result, kg_store)
        ids = [s.suggestion_id for s in suggestions]
        assert len(ids) == len(set(ids))

    def test_product_gap_wall_reclassify(self, kg_store: KnowledgeGraphStore) -> None:
        """PRODUCT_GAP blocker for a wall produces WALL_RECLASSIFY suggestion."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 120.0, "is_panelizable": False,
             "rejection_reason": "no product found"},
        ])
        blockers = [
            Blocker(
                blocker_id="BLK-001",
                category=BlockerCategory.PRODUCT_GAP,
                description="Wall #0: no product found.",
                affected_edge_ids=[0],
                severity=1.0,
            )
        ]
        suggestions = generate_suggestions(blockers, result, kg_store)
        reclassify = [s for s in suggestions if s.suggestion_type == SuggestionType.WALL_RECLASSIFY]
        assert len(reclassify) >= 1

    def test_suggestion_coverage_gain_positive(self, kg_store: KnowledgeGraphStore) -> None:
        """Wall-related suggestions have positive coverage gain."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 200.0, "is_panelizable": False,
             "rejection_reason": "too long"},
        ])
        blockers = identify_blockers(result, kg_store)
        suggestions = generate_suggestions(blockers, result, kg_store)
        wall_suggestions = [s for s in suggestions if s.affected_edge_ids]
        for s in wall_suggestions:
            assert s.estimated_coverage_gain_pct > 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4. report.py — generate_feasibility_report
# ══════════════════════════════════════════════════════════════════════════════


class TestGenerateFeasibilityReport:
    """Tests for generate_feasibility_report (FS-004)."""

    def test_full_report_mixed_walls(self, kg_store: KnowledgeGraphStore) -> None:
        """Full report with mixed panelized/unpanelized walls."""
        result = _make_panelization_result(
            wall_configs=[
                {"edge_id": 0, "length": 120.0, "is_panelizable": True},
                {"edge_id": 1, "length": 200.0, "is_panelizable": False,
                 "rejection_reason": "too long"},
                {"edge_id": 2, "length": 96.0, "is_panelizable": True},
            ],
            room_configs=[
                {"room_id": 1, "label": "Bathroom", "area_sqft": 80.0,
                 "is_eligible": True, "has_pod": True},
                {"room_id": 2, "label": "Kitchen", "area_sqft": 120.0,
                 "is_eligible": True, "has_pod": False,
                 "rejection_reason": "no compatible pod found"},
            ],
        )
        report = generate_feasibility_report(result, kg_store)
        assert isinstance(report, FeasibilityReport)
        assert report.coverage.by_wall_length_pct > 0.0
        assert report.coverage.by_wall_length_pct < 100.0
        assert len(report.wall_feasibility) == 3
        assert len(report.room_feasibility) == 2
        assert len(report.blockers) >= 1
        assert len(report.floor_scores) == 1

    def test_happy_path_all_panelized(self, kg_store: KnowledgeGraphStore) -> None:
        """All walls panelized and all rooms with pods — high project score."""
        result = _make_panelization_result(
            wall_configs=[
                {"edge_id": 0, "length": 120.0, "is_panelizable": True},
                {"edge_id": 1, "length": 96.0, "is_panelizable": True},
                {"edge_id": 2, "length": 144.0, "is_panelizable": True},
            ],
            room_configs=[
                {"room_id": 1, "label": "Bathroom", "area_sqft": 80.0,
                 "is_eligible": True, "has_pod": True},
            ],
        )
        report = generate_feasibility_report(result, kg_store)
        assert report.coverage.by_wall_length_pct == pytest.approx(100.0)
        assert report.blockers == []
        assert report.suggestions == []
        assert report.project_score > 0.5

    def test_per_wall_feasibility_blocker_refs(self, kg_store: KnowledgeGraphStore) -> None:
        """Per-wall feasibility entries reference correct blocker IDs."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 120.0, "is_panelizable": True},
            {"edge_id": 1, "length": 500.0, "is_panelizable": False,
             "rejection_reason": "exceeds max length"},
        ])
        report = generate_feasibility_report(result, kg_store)
        # Wall 0: panelizable → no blocker IDs
        wall0 = next(w for w in report.wall_feasibility if w.edge_id == 0)
        assert wall0.blocker_ids == []
        assert wall0.is_panelizable is True
        # Wall 1: blocked → at least one blocker ID
        wall1 = next(w for w in report.wall_feasibility if w.edge_id == 1)
        assert len(wall1.blocker_ids) >= 1
        assert wall1.is_panelizable is False

    def test_per_room_feasibility_pod_sku(self, kg_store: KnowledgeGraphStore) -> None:
        """Per-room feasibility includes pod SKU when pod is placed."""
        result = _make_panelization_result(
            wall_configs=[{"edge_id": 0, "length": 120.0}],
            room_configs=[
                {"room_id": 1, "label": "Bathroom", "area_sqft": 80.0,
                 "is_eligible": True, "has_pod": True},
            ],
        )
        report = generate_feasibility_report(result, kg_store)
        room_feas = report.room_feasibility[0]
        assert room_feas.has_pod is True
        assert room_feas.pod_sku == "CAP-POD-BATH-STD"

    def test_floor_score_composite(self, kg_store: KnowledgeGraphStore) -> None:
        """FloorScore composite calculation produces score in [0, 1]."""
        result = _make_panelization_result(
            wall_configs=[
                {"edge_id": 0, "length": 120.0, "is_panelizable": True},
                {"edge_id": 1, "length": 120.0, "is_panelizable": False,
                 "rejection_reason": "too long"},
            ],
            room_configs=[
                {"room_id": 1, "label": "Bathroom", "area_sqft": 80.0,
                 "is_eligible": True, "has_pod": True},
            ],
        )
        report = generate_feasibility_report(result, kg_store)
        fs = report.floor_scores[0]
        assert isinstance(fs, FloorScore)
        assert 0.0 <= fs.feasibility_score <= 1.0
        assert fs.total_wall_count == 2
        assert fs.panelized_wall_count == 1
        assert fs.placed_room_count == 1

    def test_summary_counts_correct(self, kg_store: KnowledgeGraphStore) -> None:
        """FeasibilitySummary counts match expected values."""
        result = _make_panelization_result(
            wall_configs=[
                {"edge_id": 0, "length": 120.0, "is_panelizable": True},
                {"edge_id": 1, "length": 200.0, "is_panelizable": False,
                 "rejection_reason": "exceeds max length"},
                {"edge_id": 2, "length": 96.0, "is_panelizable": True},
            ],
            room_configs=[
                {"room_id": 1, "label": "Bathroom", "area_sqft": 80.0,
                 "is_eligible": True, "has_pod": True},
                {"room_id": 2, "label": "Kitchen", "area_sqft": 120.0,
                 "is_eligible": True, "has_pod": False,
                 "rejection_reason": "no compatible pod found"},
            ],
        )
        report = generate_feasibility_report(result, kg_store)
        s = report.summary
        assert isinstance(s, FeasibilitySummary)
        assert s.total_wall_count == 3
        assert s.panelized_wall_count == 2
        assert s.unpanelized_wall_count == 1
        assert s.total_room_count == 2
        assert s.eligible_room_count == 2
        assert s.placed_room_count == 1
        assert s.total_blocker_count >= 1
        assert s.hard_blocker_count + s.soft_blocker_count == s.total_blocker_count
        assert s.suggestion_count >= 0

    def test_project_score_in_range(self, kg_store: KnowledgeGraphStore) -> None:
        """project_score is always in [0, 1]."""
        # Worst case: nothing panelized, no pods
        result = _make_panelization_result(
            wall_configs=[
                {"edge_id": 0, "length": 500.0, "is_panelizable": False,
                 "rejection_reason": "exceeds max length"},
            ],
            room_configs=[
                {"room_id": 1, "label": "Bathroom", "area_sqft": 30.0,
                 "is_eligible": True, "has_pod": False,
                 "rejection_reason": "clearance"},
            ],
        )
        report = generate_feasibility_report(result, kg_store)
        assert 0.0 <= report.project_score <= 1.0

        # Best case: everything panelized
        result2 = _make_panelization_result(
            wall_configs=[
                {"edge_id": 0, "length": 120.0, "is_panelizable": True},
            ],
            room_configs=[
                {"room_id": 1, "label": "Bathroom", "area_sqft": 80.0,
                 "is_eligible": True, "has_pod": True},
            ],
        )
        report2 = generate_feasibility_report(result2, kg_store)
        assert 0.0 <= report2.project_score <= 1.0

    def test_summary_hard_vs_soft_blockers(self, kg_store: KnowledgeGraphStore) -> None:
        """Hard blockers (severity 1.0) and soft blockers (<1.0) counted separately."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 500.0, "is_panelizable": False,
             "rejection_reason": "exceeds max length"},  # hard (1.0)
            {"edge_id": 1, "length": 120.0, "angle": math.radians(45.0),
             "is_panelizable": False,
             "rejection_reason": "non-orthogonal angle"},  # soft (0.7)
        ])
        report = generate_feasibility_report(result, kg_store)
        s = report.summary
        assert s.hard_blocker_count >= 1
        assert s.soft_blocker_count >= 1

    def test_spur_score_carried_forward(self, kg_store: KnowledgeGraphStore) -> None:
        """SPUR score from PanelizationResult is present in summary."""
        result = _make_panelization_result(
            wall_configs=[{"edge_id": 0, "length": 120.0, "is_panelizable": True}],
            spur_score=0.82,
        )
        report = generate_feasibility_report(result, kg_store)
        assert report.summary.spur_score == pytest.approx(0.82)

    def test_report_source_reference(self, kg_store: KnowledgeGraphStore) -> None:
        """Report retains a reference to the source PanelizationResult."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 120.0, "is_panelizable": True},
        ])
        report = generate_feasibility_report(result, kg_store)
        assert report.source is result

    def test_wall_feasibility_coverage_values(self, kg_store: KnowledgeGraphStore) -> None:
        """Per-wall coverage_pct is 100 for panelized, 0 for rejected."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 120.0, "is_panelizable": True},
            {"edge_id": 1, "length": 96.0, "is_panelizable": False,
             "rejection_reason": "no panel"},
        ])
        report = generate_feasibility_report(result, kg_store)
        wall0 = next(w for w in report.wall_feasibility if w.edge_id == 0)
        wall1 = next(w for w in report.wall_feasibility if w.edge_id == 1)
        assert wall0.coverage_pct == pytest.approx(100.0)
        assert wall1.coverage_pct == pytest.approx(0.0)

    def test_room_not_eligible_no_blocker(self, kg_store: KnowledgeGraphStore) -> None:
        """Ineligible rooms (is_eligible=False) don't generate blockers."""
        result = _make_panelization_result(
            wall_configs=[{"edge_id": 0, "length": 120.0}],
            room_configs=[
                {"room_id": 1, "label": "Hallway", "area_sqft": 50.0,
                 "is_eligible": False, "has_pod": False},
            ],
        )
        report = generate_feasibility_report(result, kg_store)
        # No room blockers because the room is not eligible
        room_blockers = [
            b for b in report.blockers
            if b.affected_room_ids and 1 in b.affected_room_ids
        ]
        assert room_blockers == []

    def test_max_coverage_gain_in_summary(self, kg_store: KnowledgeGraphStore) -> None:
        """max_coverage_gain_pct is the sum of all suggestion gains."""
        result = _make_panelization_result([
            {"edge_id": 0, "length": 200.0, "is_panelizable": False,
             "rejection_reason": "too long"},
            {"edge_id": 1, "length": 100.0, "is_panelizable": False,
             "rejection_reason": "no panel found"},
        ])
        report = generate_feasibility_report(result, kg_store)
        total_gain = sum(s.estimated_coverage_gain_pct for s in report.suggestions)
        assert report.summary.max_coverage_gain_pct == pytest.approx(total_gain)
