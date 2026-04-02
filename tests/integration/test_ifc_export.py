"""Integration test: IFC round-trip — BIM Transplant pipeline.

Q-019: Full pipeline round-trip test verifying the BIM Transplant module
produces valid IFC output (native or JSON fallback) from a realistic
PanelizationResult.

Pipeline under test:
    match_bim_families -> assemble_walls -> attach_openings -> export_ifc

Tests construct a rectangular room (4 walls) with openings, panel/pod
assignments, and a loaded KnowledgeGraphStore, then verify every stage
of the transplant pipeline produces correct output.

Reference: ARCHITECTURE.md (Layer 2), CLAUDE.md (BIM Transplant Agent).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

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
from docs.interfaces.graph_to_serializer import (
    FinalizedGraph,
    Opening,
    OpeningType,
    Room,
    WallSegment,
    WallType,
)
from src.knowledge_graph.loader import KnowledgeGraphStore
from src.knowledge_graph.schema import (
    Connection,
    KnowledgeGraph,
    Panel,
    PanelType,
    Pod,
)
from src.transplant.assembler import WallAssembly, assemble_walls
from src.transplant.ifc_export import export_ifc
from src.transplant.matcher import BIMFamilyMatch, match_bim_families
from src.transplant.openings import OpeningAttachment, attach_openings

# ── Constants ────────────────────────────────────────────────────────────────

_INCHES_TO_MM = 25.4

# Panel SKUs used in test data
_SKU_LB_PANEL = "PNL-LB-362S162-54"
_SKU_PT_PANEL = "PNL-PT-250S125-33"
_SKU_SPLICE = "CONN-SPLICE-362"
_SKU_POD = "POD-BATH-4860"

# ── ifcopenshell availability ────────────────────────────────────────────────

try:
    import ifcopenshell

    _HAS_IFCOPENSHELL = True
except ImportError:
    _HAS_IFCOPENSHELL = False


# ══════════════════════════════════════════════════════════════════════════════
# Test Data Factories
# ══════════════════════════════════════════════════════════════════════════════


def _build_finalized_graph() -> FinalizedGraph:
    """Build a rectangular room with 4 walls, 1 door, and 1 window.

    Layout (PDF user units, scale_factor=1.0 means 1 unit = 1 mm):

        Node 0 (0,0) ----wall 0---- Node 1 (3000,0)
           |                            |
         wall 3                       wall 1
           |                            |
        Node 3 (0,2400) --wall 2-- Node 2 (3000,2400)

    Wall 0 (bottom): 3000mm, load-bearing, has a window at midpoint
    Wall 1 (right):  2400mm, load-bearing
    Wall 2 (top):    3000mm, partition, has a door at 30% along
    Wall 3 (left):   2400mm, partition
    """
    nodes = np.array(
        [
            [0.0, 0.0],
            [3000.0, 0.0],
            [3000.0, 2400.0],
            [0.0, 2400.0],
        ],
        dtype=np.float64,
    )

    edges = np.array(
        [
            [0, 1],  # wall 0 — bottom
            [1, 2],  # wall 1 — right
            [2, 3],  # wall 2 — top
            [3, 0],  # wall 3 — left
        ],
        dtype=np.int64,
    )

    wall_segments = [
        WallSegment(
            edge_id=0,
            start_node=0,
            end_node=1,
            start_coord=nodes[0],
            end_coord=nodes[1],
            thickness=152.4,  # 6 inches in mm
            height=2700.0,
            wall_type=WallType.LOAD_BEARING,
            angle=0.0,
            length=3000.0,
            confidence=0.95,
        ),
        WallSegment(
            edge_id=1,
            start_node=1,
            end_node=2,
            start_coord=nodes[1],
            end_coord=nodes[2],
            thickness=152.4,
            height=2700.0,
            wall_type=WallType.LOAD_BEARING,
            angle=np.pi / 2,
            length=2400.0,
            confidence=0.92,
        ),
        WallSegment(
            edge_id=2,
            start_node=2,
            end_node=3,
            start_coord=nodes[2],
            end_coord=nodes[3],
            thickness=101.6,  # 4 inches in mm
            height=2700.0,
            wall_type=WallType.PARTITION,
            angle=np.pi,
            length=3000.0,
            confidence=0.88,
        ),
        WallSegment(
            edge_id=3,
            start_node=3,
            end_node=0,
            start_coord=nodes[3],
            end_coord=nodes[0],
            thickness=101.6,
            height=2700.0,
            wall_type=WallType.PARTITION,
            angle=3 * np.pi / 2,
            length=2400.0,
            confidence=0.90,
        ),
    ]

    openings = [
        Opening(
            opening_type=OpeningType.WINDOW,
            wall_edge_id=0,
            position_along_wall=0.5,
            width=900.0,
            height=1200.0,
            sill_height=900.0,
            confidence=0.93,
        ),
        Opening(
            opening_type=OpeningType.DOOR,
            wall_edge_id=2,
            position_along_wall=0.3,
            width=900.0,
            height=2100.0,
            sill_height=0.0,
            confidence=0.97,
        ),
    ]

    rooms = [
        Room(
            room_id=0,
            boundary_edges=[0, 1, 2, 3],
            boundary_nodes=[0, 1, 2, 3],
            area=3000.0 * 2400.0,
            label="Bathroom",
            is_exterior=False,
        ),
        Room(
            room_id=99,
            boundary_edges=[0, 1, 2, 3],
            boundary_nodes=[0, 1, 2, 3],
            area=0.0,
            label="Exterior",
            is_exterior=True,
        ),
    ]

    return FinalizedGraph(
        nodes=nodes,
        edges=edges,
        wall_segments=wall_segments,
        openings=openings,
        rooms=rooms,
        page_width=4000.0,
        page_height=3000.0,
        page_index=0,
        source_path="test_floor_plan.pdf",
        assumed_wall_height=2700.0,
        scale_factor=1.0,
        betti_0=1,
        betti_1=1,
    )


def _build_classified_wall_graph(
    graph: FinalizedGraph,
) -> ClassifiedWallGraph:
    """Wrap a FinalizedGraph with wall classifications."""
    classifications = [
        WallClassification(
            edge_id=0,
            wall_type=WallType.LOAD_BEARING,
            fire_rating=FireRating.HOUR_1,
            confidence=0.95,
            is_perimeter=True,
        ),
        WallClassification(
            edge_id=1,
            wall_type=WallType.LOAD_BEARING,
            fire_rating=FireRating.HOUR_1,
            confidence=0.92,
            is_perimeter=True,
        ),
        WallClassification(
            edge_id=2,
            wall_type=WallType.PARTITION,
            fire_rating=FireRating.NONE,
            confidence=0.88,
            is_perimeter=False,
        ),
        WallClassification(
            edge_id=3,
            wall_type=WallType.PARTITION,
            fire_rating=FireRating.NONE,
            confidence=0.90,
            is_perimeter=False,
        ),
    ]

    return ClassifiedWallGraph(
        graph=graph,
        classifications=classifications,
        review_threshold=0.7,
        walls_flagged_for_review=[],
        classification_summary={"load_bearing": 2, "partition": 2},
        perimeter_edge_ids=[0, 1],
    )


def _build_panel_map() -> PanelMap:
    """Build panel assignments for the 4-wall room.

    Wall 0 (3000mm): spliced into two panels (each ~59 inches)
    Wall 1 (2400mm): single panel (~94.5 inches)
    Wall 2 (3000mm): not panelizable (partition, rejected)
    Wall 3 (2400mm): single partition panel
    """
    wall_0_length_inches = 3000.0 / _INCHES_TO_MM  # ~118.1 inches

    wall0 = WallPanelization(
        edge_id=0,
        wall_length_inches=wall_0_length_inches,
        panels=[
            PanelAssignment(
                panel_sku=_SKU_LB_PANEL,
                cut_length_inches=60.0,
                position_along_wall=0.0,
                panel_index=0,
            ),
            PanelAssignment(
                panel_sku=_SKU_LB_PANEL,
                cut_length_inches=58.1,
                position_along_wall=60.0,
                panel_index=1,
            ),
        ],
        requires_splice=True,
        splice_connection_skus=[_SKU_SPLICE],
        total_material_inches=118.1,
        waste_inches=0.0,
        waste_percentage=0.0,
        is_panelizable=True,
    )

    wall1 = WallPanelization(
        edge_id=1,
        wall_length_inches=2400.0 / _INCHES_TO_MM,
        panels=[
            PanelAssignment(
                panel_sku=_SKU_LB_PANEL,
                cut_length_inches=94.5,
                position_along_wall=0.0,
                panel_index=0,
            ),
        ],
        requires_splice=False,
        total_material_inches=94.5,
        waste_inches=0.0,
        waste_percentage=0.0,
        is_panelizable=True,
    )

    # Wall 2: not panelizable
    wall2 = WallPanelization(
        edge_id=2,
        wall_length_inches=3000.0 / _INCHES_TO_MM,
        panels=[],
        requires_splice=False,
        is_panelizable=False,
        rejection_reason="Partition wall below min panel length threshold",
    )

    wall3 = WallPanelization(
        edge_id=3,
        wall_length_inches=2400.0 / _INCHES_TO_MM,
        panels=[
            PanelAssignment(
                panel_sku=_SKU_PT_PANEL,
                cut_length_inches=94.5,
                position_along_wall=0.0,
                panel_index=0,
            ),
        ],
        requires_splice=False,
        total_material_inches=94.5,
        waste_inches=0.0,
        waste_percentage=0.0,
        is_panelizable=True,
    )

    return PanelMap(
        walls=[wall0, wall1, wall2, wall3],
        panelized_wall_count=3,
        total_wall_count=4,
        unique_panel_skus=[_SKU_LB_PANEL, _SKU_PT_PANEL],
        unique_splice_skus=[_SKU_SPLICE],
    )


def _build_placement_map() -> PlacementMap:
    """Build room placements — one room with a pod, exterior room skipped."""
    return PlacementMap(
        rooms=[
            RoomPlacement(
                room_id=0,
                room_label="Bathroom",
                room_area_sqft=77.5,
                placement=ProductPlacement(
                    pod_sku=_SKU_POD,
                    position=np.array([1500.0, 1200.0], dtype=np.float64),
                    orientation_deg=0.0,
                    clearance_met=True,
                    clearance_margins={
                        "north": 6.0,
                        "south": 6.0,
                        "east": 8.0,
                        "west": 8.0,
                    },
                ),
                is_eligible=True,
            ),
        ],
        placed_room_count=1,
        eligible_room_count=1,
        total_room_count=1,
        unique_pod_skus=[_SKU_POD],
    )


def _build_panelization_result() -> PanelizationResult:
    """Build a complete PanelizationResult for the rectangular room."""
    graph = _build_finalized_graph()
    classified = _build_classified_wall_graph(graph)

    return PanelizationResult(
        source_graph=classified,
        panel_map=_build_panel_map(),
        placement_map=_build_placement_map(),
        spur_score=0.82,
        coverage_percentage=75.0,
        waste_percentage=0.0,
        pod_placement_rate=100.0,
        total_panel_count=4,
        total_splice_count=1,
    )


def _build_knowledge_graph_store() -> KnowledgeGraphStore:
    """Build a KnowledgeGraphStore with panels, pods, and connections
    matching the SKUs used in the test PanelizationResult.
    """
    panels = [
        Panel(
            sku=_SKU_LB_PANEL,
            name="Load-Bearing 362S162-54",
            panel_type=PanelType.LOAD_BEARING,
            gauge=54,
            stud_depth_inches=3.625,
            stud_spacing_inches=1.625,
            min_length_inches=24.0,
            max_length_inches=120.0,
            height_inches=106.3,
            fire_rating_hours=1.0,
            load_capacity_plf=1500.0,
            sheathing_type="OSB",
            sheathing_thickness_inches=0.4375,
            insulation_type="mineral_wool",
            insulation_r_value=15.0,
            weight_per_foot_lbs=12.5,
            unit_cost_per_foot=8.50,
            compatible_connections=[_SKU_SPLICE],
            fabricated_by=["MACH-RF-001"],
        ),
        Panel(
            sku=_SKU_PT_PANEL,
            name="Partition 250S125-33",
            panel_type=PanelType.PARTITION,
            gauge=33,
            stud_depth_inches=2.5,
            stud_spacing_inches=1.25,
            min_length_inches=24.0,
            max_length_inches=120.0,
            height_inches=106.3,
            fire_rating_hours=0.0,
            load_capacity_plf=0.0,
            sheathing_type=None,
            sheathing_thickness_inches=None,
            insulation_type=None,
            insulation_r_value=None,
            weight_per_foot_lbs=6.0,
            unit_cost_per_foot=4.25,
            compatible_connections=[_SKU_SPLICE],
            fabricated_by=["MACH-RF-001"],
        ),
    ]

    pods = [
        Pod(
            sku=_SKU_POD,
            name="Bathroom Pod 48x60",
            pod_type="bathroom",
            width_inches=48.0,
            depth_inches=60.0,
            height_inches=96.0,
            min_room_width_inches=54.0,
            min_room_depth_inches=66.0,
            clearance_inches=6.0,
            included_trades=["plumbing", "electrical", "tile"],
            connection_type="drop_in",
            weight_lbs=1800.0,
            unit_cost=12000.0,
            lead_time_days=21,
            compatible_panel_types=[PanelType.LOAD_BEARING, PanelType.PARTITION],
        ),
    ]

    connections = [
        Connection(
            sku=_SKU_SPLICE,
            name="Splice Clip 362",
            connection_type="splice",
            compatible_gauges=[33, 43, 54],
            compatible_stud_depths=[2.5, 3.625],
            load_rating_lbs=500.0,
            fire_rated=False,
            unit_cost=3.75,
            units_per="each",
        ),
    ]

    kg = KnowledgeGraph(
        version="1.0.0-test",
        last_updated="2026-04-02",
        panels=panels,
        pods=pods,
        machines=[],
        connections=connections,
        compliance_rules=[],
    )

    store = KnowledgeGraphStore()
    store.load_from_knowledge_graph(kg)
    return store


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def panelization_result() -> PanelizationResult:
    """Module-scoped PanelizationResult with 4 walls and 1 room."""
    return _build_panelization_result()


@pytest.fixture(scope="module")
def kg_store() -> KnowledgeGraphStore:
    """Module-scoped KnowledgeGraphStore with matching test data."""
    return _build_knowledge_graph_store()


@pytest.fixture(scope="module")
def bim_matches(
    panelization_result: PanelizationResult,
    kg_store: KnowledgeGraphStore,
) -> list[BIMFamilyMatch]:
    """Module-scoped BIM family matches from the matcher stage."""
    return match_bim_families(panelization_result, kg_store)


@pytest.fixture(scope="module")
def wall_assemblies(
    bim_matches: list[BIMFamilyMatch],
    panelization_result: PanelizationResult,
) -> list[WallAssembly]:
    """Module-scoped wall assemblies from the assembler stage."""
    return assemble_walls(bim_matches, panelization_result)


@pytest.fixture(scope="module")
def opening_attachments(
    wall_assemblies: list[WallAssembly],
    panelization_result: PanelizationResult,
) -> list[OpeningAttachment]:
    """Module-scoped opening attachments from the openings stage."""
    return attach_openings(wall_assemblies, panelization_result)


# ══════════════════════════════════════════════════════════════════════════════
# Tests: matcher (match_bim_families)
# ══════════════════════════════════════════════════════════════════════════════


class TestMatchBimFamilies:
    """Verify that match_bim_families produces correct BIMFamilyMatch objects."""

    def test_returns_list_of_bim_family_match(self, bim_matches: list[BIMFamilyMatch]):
        """Every item is a BIMFamilyMatch instance."""
        assert isinstance(bim_matches, list)
        assert all(isinstance(m, BIMFamilyMatch) for m in bim_matches)

    def test_match_count_equals_panel_assignment_count(
        self,
        bim_matches: list[BIMFamilyMatch],
        panelization_result: PanelizationResult,
    ):
        """One match per panel assignment across all panelizable walls."""
        expected = sum(
            len(wp.panels)
            for wp in panelization_result.panel_map.walls
            if wp.is_panelizable
        )
        assert len(bim_matches) == expected

    def test_every_match_has_valid_family_name(self, bim_matches: list[BIMFamilyMatch]):
        """Every match has a non-empty family_name following CFS convention."""
        for m in bim_matches:
            assert m.family_name, f"Empty family_name on edge_id={m.edge_id}"
            assert m.family_name.startswith("CFS_Wall_")

    def test_every_match_has_material_layers(self, bim_matches: list[BIMFamilyMatch]):
        """Every match has at least track + stud material layers."""
        for m in bim_matches:
            assert len(m.material_layers) >= 2, (
                f"Expected >= 2 material layers for edge_id={m.edge_id}, "
                f"got {len(m.material_layers)}"
            )
            layer_names = [layer["name"] for layer in m.material_layers]
            assert "track" in layer_names
            assert "stud" in layer_names

    def test_lb_panel_has_insulation_and_sheathing(self, bim_matches: list[BIMFamilyMatch]):
        """Load-bearing panel matches include insulation and sheathing layers."""
        lb_matches = [m for m in bim_matches if m.panel_sku == _SKU_LB_PANEL]
        assert len(lb_matches) > 0
        for m in lb_matches:
            layer_names = [layer["name"] for layer in m.material_layers]
            assert "insulation" in layer_names, "LB panel should have insulation"
            assert "sheathing" in layer_names, "LB panel should have sheathing"

    def test_pt_panel_has_no_insulation_or_sheathing(
        self, bim_matches: list[BIMFamilyMatch]
    ):
        """Partition panel matches have no insulation or sheathing layers."""
        pt_matches = [m for m in bim_matches if m.panel_sku == _SKU_PT_PANEL]
        assert len(pt_matches) > 0
        for m in pt_matches:
            layer_names = [layer["name"] for layer in m.material_layers]
            assert "insulation" not in layer_names
            assert "sheathing" not in layer_names

    def test_panel_sku_matches_kg(
        self,
        bim_matches: list[BIMFamilyMatch],
        kg_store: KnowledgeGraphStore,
    ):
        """Every match's panel_sku exists in the KG store."""
        for m in bim_matches:
            assert m.panel_sku in kg_store.panels, (
                f"SKU {m.panel_sku} not in KG store"
            )

    def test_fire_rating_propagated(self, bim_matches: list[BIMFamilyMatch]):
        """Load-bearing panels carry fire_rating_hours=1.0 from KG."""
        lb_matches = [m for m in bim_matches if m.panel_sku == _SKU_LB_PANEL]
        for m in lb_matches:
            assert m.fire_rating_hours == 1.0

    def test_non_panelizable_wall_skipped(
        self,
        bim_matches: list[BIMFamilyMatch],
    ):
        """Wall 2 (is_panelizable=False) produces no matches."""
        edge_2_matches = [m for m in bim_matches if m.edge_id == 2]
        assert len(edge_2_matches) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Tests: assembler (assemble_walls)
# ══════════════════════════════════════════════════════════════════════════════


class TestAssembleWalls:
    """Verify that assemble_walls produces correct WallAssembly objects."""

    def test_returns_list_of_wall_assembly(self, wall_assemblies: list[WallAssembly]):
        """Every item is a WallAssembly instance."""
        assert isinstance(wall_assemblies, list)
        assert all(isinstance(a, WallAssembly) for a in wall_assemblies)

    def test_assembly_count_matches_panelized_walls(
        self,
        wall_assemblies: list[WallAssembly],
        panelization_result: PanelizationResult,
    ):
        """One assembly per panelized wall (walls with BIM matches)."""
        panelized_edge_ids = {
            wp.edge_id
            for wp in panelization_result.panel_map.walls
            if wp.is_panelizable and len(wp.panels) > 0
        }
        assert len(wall_assemblies) == len(panelized_edge_ids)

    def test_coordinates_in_mm(self, wall_assemblies: list[WallAssembly]):
        """Wall start/end coordinates are in millimeters (scale_factor=1.0)."""
        # With scale_factor=1.0, PDF user units map directly to mm
        # Wall 0 goes from (0,0) to (3000,0)
        wall_0 = next(a for a in wall_assemblies if a.edge_id == 0)
        assert wall_0.wall_start == (0.0, 0.0)
        assert wall_0.wall_end == (3000.0, 0.0)

    def test_wall_length_in_mm(self, wall_assemblies: list[WallAssembly]):
        """Wall lengths are in millimeters."""
        wall_0 = next(a for a in wall_assemblies if a.edge_id == 0)
        assert wall_0.wall_length_mm == pytest.approx(3000.0, abs=1.0)

        wall_1 = next(a for a in wall_assemblies if a.edge_id == 1)
        assert wall_1.wall_length_mm == pytest.approx(2400.0, abs=1.0)

    def test_height_converted_from_inches(self, wall_assemblies: list[WallAssembly]):
        """Wall height is panel height (inches) converted to mm."""
        wall_0 = next(a for a in wall_assemblies if a.edge_id == 0)
        # Panel height is 106.3 inches -> 106.3 * 25.4 = 2700.02 mm
        expected_height_mm = 106.3 * _INCHES_TO_MM
        assert wall_0.height_mm == pytest.approx(expected_height_mm, abs=0.1)

    def test_spliced_wall_has_seam_positions(self, wall_assemblies: list[WallAssembly]):
        """Wall 0 (spliced into 2 panels) has one seam position."""
        wall_0 = next(a for a in wall_assemblies if a.edge_id == 0)
        assert len(wall_0.seam_positions_mm) == 1
        # Seam at cumulative cut_length of first panel: 60 inches -> 1524 mm
        expected_seam_mm = 60.0 * _INCHES_TO_MM
        assert wall_0.seam_positions_mm[0] == pytest.approx(expected_seam_mm, abs=0.1)

    def test_spliced_wall_has_splice_skus(self, wall_assemblies: list[WallAssembly]):
        """Wall 0 carries splice connection SKUs."""
        wall_0 = next(a for a in wall_assemblies if a.edge_id == 0)
        assert _SKU_SPLICE in wall_0.splice_skus

    def test_single_panel_wall_no_seams(self, wall_assemblies: list[WallAssembly]):
        """Wall 1 (single panel) has no seam positions."""
        wall_1 = next(a for a in wall_assemblies if a.edge_id == 1)
        assert len(wall_1.seam_positions_mm) == 0

    def test_junction_types_detected(self, wall_assemblies: list[WallAssembly]):
        """Junction types are classified from node connectivity."""
        for assembly in wall_assemblies:
            assert assembly.junction_type in ("end", "L", "T", "X")

    def test_non_panelizable_wall_not_assembled(
        self, wall_assemblies: list[WallAssembly]
    ):
        """Wall 2 (not panelizable, no panels) does not get an assembly."""
        edge_ids = {a.edge_id for a in wall_assemblies}
        assert 2 not in edge_ids


# ══════════════════════════════════════════════════════════════════════════════
# Tests: openings (attach_openings)
# ══════════════════════════════════════════════════════════════════════════════


class TestAttachOpenings:
    """Verify that attach_openings produces correct OpeningAttachment objects."""

    def test_returns_list_of_opening_attachment(
        self, opening_attachments: list[OpeningAttachment]
    ):
        """Every item is an OpeningAttachment instance."""
        assert isinstance(opening_attachments, list)
        assert all(isinstance(o, OpeningAttachment) for o in opening_attachments)

    def test_window_attached_to_wall_0(
        self, opening_attachments: list[OpeningAttachment]
    ):
        """The window on wall 0 is attached with correct host edge."""
        window = next(
            (o for o in opening_attachments if o.opening_type == "window"), None
        )
        assert window is not None
        assert window.host_edge_id == 0

    def test_window_position_along_wall(
        self, opening_attachments: list[OpeningAttachment]
    ):
        """Window position is at 50% of wall 0 (3000mm) = 1500mm."""
        window = next(o for o in opening_attachments if o.opening_type == "window")
        assert window.position_along_wall_mm == pytest.approx(1500.0, abs=1.0)

    def test_window_dimensions_in_mm(
        self, opening_attachments: list[OpeningAttachment]
    ):
        """Window width and height are converted to mm."""
        window = next(o for o in opening_attachments if o.opening_type == "window")
        # scale_factor=1.0, so dimensions remain in mm
        assert window.width_mm == pytest.approx(900.0, abs=1.0)
        assert window.height_mm == pytest.approx(1200.0, abs=1.0)
        assert window.sill_height_mm == pytest.approx(900.0, abs=1.0)

    def test_window_void_coords(self, opening_attachments: list[OpeningAttachment]):
        """Window void_coords has (cx, cy, z_bottom, z_top)."""
        window = next(o for o in opening_attachments if o.opening_type == "window")
        cx, cy, z_bottom, z_top = window.void_coords
        # Window at 50% along wall 0: cx=1500, cy=0
        assert cx == pytest.approx(1500.0, abs=1.0)
        assert cy == pytest.approx(0.0, abs=1.0)
        assert z_bottom == pytest.approx(900.0, abs=1.0)
        assert z_top == pytest.approx(2100.0, abs=1.0)

    def test_door_on_non_panelized_wall_not_attached(
        self, opening_attachments: list[OpeningAttachment]
    ):
        """Door on wall 2 (not panelizable, no assembly) is not attached.

        Since wall 2 has is_panelizable=False and no panels, it does not
        get a WallAssembly. The door on that wall will be logged as a
        warning and skipped.
        """
        door = next(
            (o for o in opening_attachments if o.opening_type == "door"), None
        )
        # Door on edge_id=2 which has no assembly -> should not be attached
        assert door is None


# ══════════════════════════════════════════════════════════════════════════════
# Tests: export_ifc — JSON fallback (always available)
# ══════════════════════════════════════════════════════════════════════════════


class TestExportIfcJsonFallback:
    """Verify JSON fallback export produces valid structured output."""

    def test_export_creates_json_file(
        self,
        wall_assemblies: list[WallAssembly],
        opening_attachments: list[OpeningAttachment],
        panelization_result: PanelizationResult,
        tmp_path: Path,
    ):
        """Export produces a non-empty JSON file."""
        output = tmp_path / "output.json"
        actual_path = export_ifc(
            wall_assemblies, opening_attachments, panelization_result, output
        )
        assert actual_path.exists()
        assert actual_path.suffix == ".json"
        assert actual_path.stat().st_size > 0

    def test_json_structure_has_project_hierarchy(
        self,
        wall_assemblies: list[WallAssembly],
        opening_attachments: list[OpeningAttachment],
        panelization_result: PanelizationResult,
        tmp_path: Path,
    ):
        """JSON output has project -> site -> building -> storey hierarchy."""
        output = tmp_path / "hierarchy.json"
        actual_path = export_ifc(
            wall_assemblies, opening_attachments, panelization_result, output
        )
        with open(actual_path) as f:
            data = json.load(f)

        assert data["_format"] == "axon_ifc_fallback_json"
        assert data["_schema"] == "IFC4"
        assert "project" in data

        project = data["project"]
        assert project["name"] == "Axon BIM Transplant Export"
        assert "site" in project
        assert "building" in project["site"]
        assert "storeys" in project["site"]["building"]

        storeys = project["site"]["building"]["storeys"]
        assert len(storeys) == 1
        assert storeys[0]["name"] == "Level 1"

    def test_json_walls_present(
        self,
        wall_assemblies: list[WallAssembly],
        opening_attachments: list[OpeningAttachment],
        panelization_result: PanelizationResult,
        tmp_path: Path,
    ):
        """JSON storey contains walls matching the assembly count."""
        output = tmp_path / "walls.json"
        actual_path = export_ifc(
            wall_assemblies, opening_attachments, panelization_result, output
        )
        with open(actual_path) as f:
            data = json.load(f)

        storey = data["project"]["site"]["building"]["storeys"][0]
        walls = storey["walls"]
        assert len(walls) == len(wall_assemblies)

        for wall in walls:
            assert wall["ifc_class"] == "IfcWallStandardCase"
            assert "geometry" in wall
            assert wall["geometry"]["type"] == "IfcExtrudedAreaSolid"

    def test_json_walls_have_panel_skus_in_property_sets(
        self,
        wall_assemblies: list[WallAssembly],
        opening_attachments: list[OpeningAttachment],
        panelization_result: PanelizationResult,
        tmp_path: Path,
    ):
        """SKU traceability: property sets on walls contain real panel SKUs."""
        output = tmp_path / "skus.json"
        actual_path = export_ifc(
            wall_assemblies, opening_attachments, panelization_result, output
        )
        with open(actual_path) as f:
            data = json.load(f)

        storey = data["project"]["site"]["building"]["storeys"][0]
        for wall in storey["walls"]:
            psets = wall.get("property_sets", {})
            if wall["panels"]:
                assert "Axon_PanelData" in psets
                pset = psets["Axon_PanelData"]
                assert "PanelSKUs" in pset
                # Verify real SKUs from KG appear
                skus = pset["PanelSKUs"]
                assert _SKU_LB_PANEL in skus or _SKU_PT_PANEL in skus

    def test_json_openings_present(
        self,
        wall_assemblies: list[WallAssembly],
        opening_attachments: list[OpeningAttachment],
        panelization_result: PanelizationResult,
        tmp_path: Path,
    ):
        """JSON walls with openings contain IfcOpeningElement entries."""
        output = tmp_path / "openings.json"
        actual_path = export_ifc(
            wall_assemblies, opening_attachments, panelization_result, output
        )
        with open(actual_path) as f:
            data = json.load(f)

        storey = data["project"]["site"]["building"]["storeys"][0]
        all_openings = []
        for wall in storey["walls"]:
            all_openings.extend(wall.get("openings", []))

        assert len(all_openings) == len(opening_attachments)
        for op in all_openings:
            assert op["ifc_class"] == "IfcOpeningElement"
            assert op["relationship"] == "IfcRelVoidsElement"

    def test_json_rooms_present(
        self,
        wall_assemblies: list[WallAssembly],
        opening_attachments: list[OpeningAttachment],
        panelization_result: PanelizationResult,
        tmp_path: Path,
    ):
        """JSON storey contains rooms (IfcSpace) for interior rooms."""
        output = tmp_path / "rooms.json"
        actual_path = export_ifc(
            wall_assemblies, opening_attachments, panelization_result, output
        )
        with open(actual_path) as f:
            data = json.load(f)

        storey = data["project"]["site"]["building"]["storeys"][0]
        spaces = storey["spaces"]
        # Only interior rooms (room_id=0, not the exterior boundary)
        assert len(spaces) >= 1
        bathroom = next(s for s in spaces if s["room_id"] == 0)
        assert bathroom["ifc_class"] == "IfcSpace"
        assert bathroom["label"] == "Bathroom"

    def test_json_pod_placement_present(
        self,
        wall_assemblies: list[WallAssembly],
        opening_attachments: list[OpeningAttachment],
        panelization_result: PanelizationResult,
        tmp_path: Path,
    ):
        """Room with pod has pod_placement entry with correct SKU."""
        output = tmp_path / "pods.json"
        actual_path = export_ifc(
            wall_assemblies, opening_attachments, panelization_result, output
        )
        with open(actual_path) as f:
            data = json.load(f)

        storey = data["project"]["site"]["building"]["storeys"][0]
        bathroom = next(s for s in storey["spaces"] if s["room_id"] == 0)
        assert "pod_placement" in bathroom
        pod = bathroom["pod_placement"]
        assert pod["ifc_class"] == "IfcFurnishingElement"
        assert pod["pod_sku"] == _SKU_POD
        assert pod["clearance_met"] is True
        # Pod position in mm: (1500, 1200) * scale=1.0
        assert pod["position_mm"][0] == pytest.approx(1500.0, abs=1.0)
        assert pod["position_mm"][1] == pytest.approx(1200.0, abs=1.0)

    def test_json_summary_stats(
        self,
        wall_assemblies: list[WallAssembly],
        opening_attachments: list[OpeningAttachment],
        panelization_result: PanelizationResult,
        tmp_path: Path,
    ):
        """JSON summary section contains aggregate statistics."""
        output = tmp_path / "summary.json"
        actual_path = export_ifc(
            wall_assemblies, opening_attachments, panelization_result, output
        )
        with open(actual_path) as f:
            data = json.load(f)

        summary = data["summary"]
        assert summary["wall_count"] == len(wall_assemblies)
        assert summary["opening_count"] == len(opening_attachments)
        assert summary["spur_score"] == pytest.approx(0.82, abs=0.01)
        assert summary["scale_factor"] == 1.0

    def test_ifc_extension_falls_back_to_json_when_no_ifcopenshell(
        self,
        wall_assemblies: list[WallAssembly],
        opening_attachments: list[OpeningAttachment],
        panelization_result: PanelizationResult,
        tmp_path: Path,
    ):
        """Requesting .ifc without ifcopenshell produces a .json file."""
        if _HAS_IFCOPENSHELL:
            pytest.skip("ifcopenshell is installed; fallback not triggered")

        output = tmp_path / "fallback.ifc"
        actual_path = export_ifc(
            wall_assemblies, opening_attachments, panelization_result, output
        )
        assert actual_path.suffix == ".json"
        assert actual_path.exists()


# ══════════════════════════════════════════════════════════════════════════════
# Tests: export_ifc — native IFC (ifcopenshell required)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _HAS_IFCOPENSHELL, reason="ifcopenshell not installed")
class TestExportIfcNative:
    """Verify native IFC4 export produces a valid IFC-SPF file.

    These tests require ifcopenshell to be installed. They are skipped
    automatically when the dependency is unavailable.
    """

    def test_export_creates_ifc_file(
        self,
        wall_assemblies: list[WallAssembly],
        opening_attachments: list[OpeningAttachment],
        panelization_result: PanelizationResult,
        tmp_path: Path,
    ):
        """Export produces a non-empty .ifc file."""
        output = tmp_path / "output.ifc"
        actual_path = export_ifc(
            wall_assemblies, opening_attachments, panelization_result, output
        )
        assert actual_path.exists()
        assert actual_path.suffix == ".ifc"
        assert actual_path.stat().st_size > 0

    def test_ifc_contains_wall_entities(
        self,
        wall_assemblies: list[WallAssembly],
        opening_attachments: list[OpeningAttachment],
        panelization_result: PanelizationResult,
        tmp_path: Path,
    ):
        """IFC file contains IfcWallStandardCase entities matching assemblies."""
        output = tmp_path / "walls.ifc"
        actual_path = export_ifc(
            wall_assemblies, opening_attachments, panelization_result, output
        )
        ifc = ifcopenshell.open(str(actual_path))
        walls = ifc.by_type("IfcWallStandardCase")
        assert len(walls) == len(wall_assemblies)

    def test_ifc_contains_opening_elements(
        self,
        wall_assemblies: list[WallAssembly],
        opening_attachments: list[OpeningAttachment],
        panelization_result: PanelizationResult,
        tmp_path: Path,
    ):
        """IFC file contains IfcOpeningElement entities matching attachments."""
        output = tmp_path / "openings.ifc"
        actual_path = export_ifc(
            wall_assemblies, opening_attachments, panelization_result, output
        )
        ifc = ifcopenshell.open(str(actual_path))
        openings = ifc.by_type("IfcOpeningElement")
        assert len(openings) == len(opening_attachments)

    def test_ifc_contains_spaces(
        self,
        wall_assemblies: list[WallAssembly],
        opening_attachments: list[OpeningAttachment],
        panelization_result: PanelizationResult,
        tmp_path: Path,
    ):
        """IFC file contains IfcSpace for interior rooms."""
        output = tmp_path / "spaces.ifc"
        actual_path = export_ifc(
            wall_assemblies, opening_attachments, panelization_result, output
        )
        ifc = ifcopenshell.open(str(actual_path))
        spaces = ifc.by_type("IfcSpace")
        assert len(spaces) >= 1

    def test_ifc_contains_pod_furnishing(
        self,
        wall_assemblies: list[WallAssembly],
        opening_attachments: list[OpeningAttachment],
        panelization_result: PanelizationResult,
        tmp_path: Path,
    ):
        """IFC file contains IfcFurnishingElement for the placed pod."""
        output = tmp_path / "pods.ifc"
        actual_path = export_ifc(
            wall_assemblies, opening_attachments, panelization_result, output
        )
        ifc = ifcopenshell.open(str(actual_path))
        furnishings = ifc.by_type("IfcFurnishingElement")
        assert len(furnishings) == 1

    def test_ifc_has_project_hierarchy(
        self,
        wall_assemblies: list[WallAssembly],
        opening_attachments: list[OpeningAttachment],
        panelization_result: PanelizationResult,
        tmp_path: Path,
    ):
        """IFC file has IfcProject -> IfcSite -> IfcBuilding -> IfcBuildingStorey."""
        output = tmp_path / "hierarchy.ifc"
        actual_path = export_ifc(
            wall_assemblies, opening_attachments, panelization_result, output
        )
        ifc = ifcopenshell.open(str(actual_path))
        assert len(ifc.by_type("IfcProject")) == 1
        assert len(ifc.by_type("IfcSite")) == 1
        assert len(ifc.by_type("IfcBuilding")) == 1
        assert len(ifc.by_type("IfcBuildingStorey")) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Tests: full pipeline round-trip
# ══════════════════════════════════════════════════════════════════════════════


class TestFullPipelineRoundTrip:
    """End-to-end round-trip: PanelizationResult -> JSON/IFC output.

    Verifies the full chain match_bim_families -> assemble_walls ->
    attach_openings -> export_ifc produces a coherent output.
    """

    def test_round_trip_json(self, tmp_path: Path):
        """Full pipeline round-trip with JSON fallback output."""
        result = _build_panelization_result()
        store = _build_knowledge_graph_store()

        matches = match_bim_families(result, store)
        assemblies = assemble_walls(matches, result)
        openings = attach_openings(assemblies, result)
        output = tmp_path / "round_trip.json"
        actual_path = export_ifc(assemblies, openings, result, output)

        assert actual_path.exists()
        with open(actual_path) as f:
            data = json.load(f)

        storey = data["project"]["site"]["building"]["storeys"][0]

        # Walls
        assert len(storey["walls"]) == 3  # 3 panelized walls
        # Openings (only window on wall 0; door on wall 2 has no assembly)
        total_openings = sum(len(w.get("openings", [])) for w in storey["walls"])
        assert total_openings == 1
        # Rooms with pods
        rooms_with_pods = [
            s for s in storey["spaces"] if "pod_placement" in s
        ]
        assert len(rooms_with_pods) == 1

    @pytest.mark.skipif(not _HAS_IFCOPENSHELL, reason="ifcopenshell not installed")
    def test_round_trip_ifc(self, tmp_path: Path):
        """Full pipeline round-trip with native IFC output."""
        result = _build_panelization_result()
        store = _build_knowledge_graph_store()

        matches = match_bim_families(result, store)
        assemblies = assemble_walls(matches, result)
        openings = attach_openings(assemblies, result)
        output = tmp_path / "round_trip.ifc"
        actual_path = export_ifc(assemblies, openings, result, output)

        assert actual_path.suffix == ".ifc"
        ifc = ifcopenshell.open(str(actual_path))

        assert len(ifc.by_type("IfcWallStandardCase")) == 3
        assert len(ifc.by_type("IfcOpeningElement")) == 1
        assert len(ifc.by_type("IfcSpace")) >= 1
        assert len(ifc.by_type("IfcFurnishingElement")) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Tests: edge cases
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case tests for graceful handling of unusual inputs."""

    def test_non_panelizable_wall_skipped_gracefully(self):
        """Wall with is_panelizable=False is skipped by matcher without error."""
        result = _build_panelization_result()
        store = _build_knowledge_graph_store()

        matches = match_bim_families(result, store)
        # Wall 2 has is_panelizable=False, so no matches for edge_id=2
        assert not any(m.edge_id == 2 for m in matches)

    def test_room_with_no_pod_still_exported(self, tmp_path: Path):
        """Room without a pod gets IfcSpace but no pod product in JSON."""
        result = _build_panelization_result()
        # Override the placement map to have a room with no pod
        result.placement_map = PlacementMap(
            rooms=[
                RoomPlacement(
                    room_id=0,
                    room_label="Bathroom",
                    room_area_sqft=77.5,
                    placement=None,
                    is_eligible=False,
                    rejection_reason="No compatible pod",
                ),
            ],
            placed_room_count=0,
            eligible_room_count=0,
            total_room_count=1,
            unique_pod_skus=[],
        )

        store = _build_knowledge_graph_store()
        matches = match_bim_families(result, store)
        assemblies = assemble_walls(matches, result)
        openings = attach_openings(assemblies, result)

        output = tmp_path / "no_pod.json"
        actual_path = export_ifc(assemblies, openings, result, output)

        with open(actual_path) as f:
            data = json.load(f)

        storey = data["project"]["site"]["building"]["storeys"][0]
        spaces = storey["spaces"]
        assert len(spaces) >= 1
        bathroom = next(s for s in spaces if s["room_id"] == 0)
        assert "pod_placement" not in bathroom

    def test_opening_on_non_panelized_wall_handled(self):
        """Opening attached to a wall with no assembly is skipped gracefully."""
        result = _build_panelization_result()
        store = _build_knowledge_graph_store()

        matches = match_bim_families(result, store)
        assemblies = assemble_walls(matches, result)

        # Wall 2 has no assembly, but the source graph has a door on wall 2
        openings = attach_openings(assemblies, result)

        # Door on wall 2 should be skipped (no assembly), so only window on wall 0
        door_attachments = [o for o in openings if o.opening_type == "door"]
        assert len(door_attachments) == 0

    def test_empty_panel_map_produces_empty_output(self, tmp_path: Path):
        """PanelizationResult with no panelizable walls produces valid output."""
        graph = _build_finalized_graph()
        classified = _build_classified_wall_graph(graph)

        empty_result = PanelizationResult(
            source_graph=classified,
            panel_map=PanelMap(
                walls=[
                    WallPanelization(
                        edge_id=i,
                        wall_length_inches=0.0,
                        panels=[],
                        requires_splice=False,
                        is_panelizable=False,
                        rejection_reason="Test: all walls rejected",
                    )
                    for i in range(4)
                ],
                panelized_wall_count=0,
                total_wall_count=4,
            ),
            placement_map=PlacementMap(
                rooms=[],
                placed_room_count=0,
                eligible_room_count=0,
                total_room_count=0,
            ),
        )

        store = _build_knowledge_graph_store()
        matches = match_bim_families(empty_result, store)
        assert len(matches) == 0

        assemblies = assemble_walls(matches, empty_result)
        assert len(assemblies) == 0

        openings = attach_openings(assemblies, empty_result)
        assert len(openings) == 0

        output = tmp_path / "empty.json"
        actual_path = export_ifc(assemblies, openings, empty_result, output)
        assert actual_path.exists()

        with open(actual_path) as f:
            data = json.load(f)
        assert data["summary"]["wall_count"] == 0

    def test_unknown_panel_sku_raises_value_error(self):
        """Panel SKU not in the KG raises ValueError (contract violation)."""
        result = _build_panelization_result()
        store = _build_knowledge_graph_store()

        # Inject a bogus SKU into wall 0's first panel
        result.panel_map.walls[0].panels[0].panel_sku = "BOGUS-SKU-999"

        with pytest.raises(ValueError, match="not found in Knowledge Graph"):
            match_bim_families(result, store)
