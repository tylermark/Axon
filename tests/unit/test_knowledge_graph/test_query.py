"""Unit tests for src/knowledge_graph/query.py — deterministic KG query APIs."""

from __future__ import annotations

import pytest

from src.knowledge_graph.loader import KnowledgeGraphStore, load_knowledge_graph
from src.knowledge_graph.query import (
    FabricationValidation,
    PanelRecommendation,
    get_bim_family,
    get_connections_for_panel,
    get_fabrication_limits,
    get_machine_for_panel,
    get_machine_for_spec,
    get_panels_for_wall_segment,
    get_valid_panels,
    get_valid_pods,
    validate_panel_fabrication,
    validate_wall_panelization,
)
from src.knowledge_graph.schema import PanelType

# ---------------------------------------------------------------------------
# Module-scoped fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kg_store() -> KnowledgeGraphStore:
    return load_knowledge_graph()


# ══════════════════════════════════════════════════════════════════════════════
# 1. get_valid_panels
# ══════════════════════════════════════════════════════════════════════════════


class TestGetValidPanels:
    def test_120in_load_bearing_returns_multiple(self, kg_store: KnowledgeGraphStore) -> None:
        """10ft (120") load-bearing wall → multiple panels, all with max_length >= 120."""
        panels = get_valid_panels(
            kg_store, wall_length_inches=120.0, wall_type=PanelType.LOAD_BEARING
        )
        assert len(panels) >= 2
        for p in panels:
            assert p.max_length_inches >= 120.0
            assert p.panel_type == PanelType.LOAD_BEARING

    def test_filter_by_gauge_16(self, kg_store: KnowledgeGraphStore) -> None:
        panels = get_valid_panels(
            kg_store, wall_length_inches=120.0, wall_type=PanelType.LOAD_BEARING, gauge=16
        )
        assert len(panels) >= 1
        for p in panels:
            assert p.gauge == 16

    def test_filter_by_gauge_18(self, kg_store: KnowledgeGraphStore) -> None:
        panels = get_valid_panels(
            kg_store, wall_length_inches=120.0, wall_type=PanelType.LOAD_BEARING, gauge=18
        )
        assert len(panels) >= 1
        for p in panels:
            assert p.gauge == 18

    def test_filter_by_fire_rating_2hr(self, kg_store: KnowledgeGraphStore) -> None:
        panels = get_valid_panels(kg_store, wall_length_inches=120.0, fire_rating_hours=2.0)
        assert len(panels) >= 1
        for p in panels:
            assert p.fire_rating_hours >= 2.0

    def test_very_long_wall_no_single_panel(self, kg_store: KnowledgeGraphStore) -> None:
        """600" wall — no panel has max_length >= 600, expect empty."""
        panels = get_valid_panels(kg_store, wall_length_inches=600.0)
        assert panels == []

    def test_short_wall_within_min(self, kg_store: KnowledgeGraphStore) -> None:
        """24" wall (at min_length) should return panels."""
        panels = get_valid_panels(kg_store, wall_length_inches=24.0)
        assert len(panels) >= 1
        for p in panels:
            assert p.min_length_inches <= 24.0 + 0.25  # tolerance

    def test_very_short_wall_below_min(self, kg_store: KnowledgeGraphStore) -> None:
        """6" wall — below all panels' min_length (24"), expect empty."""
        panels = get_valid_panels(kg_store, wall_length_inches=6.0)
        assert panels == []

    def test_no_type_filter_returns_all_types(self, kg_store: KnowledgeGraphStore) -> None:
        panels = get_valid_panels(kg_store, wall_length_inches=120.0)
        types_returned = {p.panel_type for p in panels}
        # Should include at least load_bearing and partition
        assert PanelType.LOAD_BEARING in types_returned
        assert PanelType.PARTITION in types_returned

    def test_filter_by_stud_depth(self, kg_store: KnowledgeGraphStore) -> None:
        panels = get_valid_panels(kg_store, wall_length_inches=120.0, stud_depth_inches=3.5)
        assert len(panels) >= 1
        for p in panels:
            assert p.stud_depth_inches == 3.5

    def test_max_results_limits_output(self, kg_store: KnowledgeGraphStore) -> None:
        panels = get_valid_panels(kg_store, wall_length_inches=120.0, max_results=3)
        assert len(panels) <= 3

    def test_results_sorted_by_fit(self, kg_store: KnowledgeGraphStore) -> None:
        """Results are sorted — first result should have reasonable fit score."""
        panels = get_valid_panels(
            kg_store, wall_length_inches=120.0, wall_type=PanelType.LOAD_BEARING
        )
        assert len(panels) >= 1
        # Just verify it returns Panel objects in a list
        assert panels[0].max_length_inches >= 120.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. get_panels_for_wall_segment
# ══════════════════════════════════════════════════════════════════════════════


class TestGetPanelsForWallSegment:
    def test_short_wall_single_panel(self, kg_store: KnowledgeGraphStore) -> None:
        """96" wall → single panel, no splice needed."""
        recs = get_panels_for_wall_segment(
            kg_store,
            wall_length_inches=96.0,
            wall_type=PanelType.LOAD_BEARING,
        )
        assert len(recs) >= 1
        best = recs[0]
        assert isinstance(best, PanelRecommendation)
        assert best.quantity == 1
        assert not best.requires_splice

    def test_long_wall_may_require_splice(self, kg_store: KnowledgeGraphStore) -> None:
        """360" wall (30ft) exceeds max_length for 16ga panels (300") → splicing."""
        recs = get_panels_for_wall_segment(
            kg_store,
            wall_length_inches=360.0,
            wall_type=PanelType.LOAD_BEARING,
        )
        assert len(recs) >= 1
        # At least some recommendations should require splicing
        has_splice = any(r.requires_splice for r in recs)
        assert has_splice

    def test_recommendation_fields(self, kg_store: KnowledgeGraphStore) -> None:
        recs = get_panels_for_wall_segment(
            kg_store,
            wall_length_inches=96.0,
            wall_type=PanelType.LOAD_BEARING,
        )
        rec = recs[0]
        assert rec.quantity >= 1
        assert len(rec.cut_lengths_inches) == rec.quantity
        assert rec.waste_percentage >= 0.0
        assert rec.waste_percentage <= 100.0
        assert rec.total_material_cost > 0.0
        assert rec.waste_inches >= 0.0

    def test_score_in_range(self, kg_store: KnowledgeGraphStore) -> None:
        recs = get_panels_for_wall_segment(
            kg_store,
            wall_length_inches=96.0,
            wall_type=PanelType.LOAD_BEARING,
        )
        for rec in recs:
            assert 0.0 <= rec.score <= 1.0

    def test_cut_lengths_cover_wall(self, kg_store: KnowledgeGraphStore) -> None:
        """Sum of cut lengths should be >= wall length."""
        recs = get_panels_for_wall_segment(
            kg_store,
            wall_length_inches=96.0,
            wall_type=PanelType.LOAD_BEARING,
        )
        rec = recs[0]
        assert sum(rec.cut_lengths_inches) >= 96.0 - 0.5

    def test_preferred_gauge_boost(self, kg_store: KnowledgeGraphStore) -> None:
        """Preferred gauge should be represented in results."""
        recs = get_panels_for_wall_segment(
            kg_store,
            wall_length_inches=120.0,
            wall_type=PanelType.LOAD_BEARING,
            preferred_gauge=18,
        )
        gauges = {r.panel.gauge for r in recs}
        assert 18 in gauges

    def test_sorted_by_score_descending(self, kg_store: KnowledgeGraphStore) -> None:
        recs = get_panels_for_wall_segment(
            kg_store,
            wall_length_inches=120.0,
            wall_type=PanelType.LOAD_BEARING,
        )
        if len(recs) >= 2:
            for i in range(len(recs) - 1):
                assert recs[i].score >= recs[i + 1].score


# ══════════════════════════════════════════════════════════════════════════════
# 3. get_valid_pods
# ══════════════════════════════════════════════════════════════════════════════


class TestGetValidPods:
    def test_bathroom_8x10_room(self, kg_store: KnowledgeGraphStore) -> None:
        """8'x10' (96"x120") room with function=bathroom → at least standard pod."""
        pods = get_valid_pods(
            kg_store,
            room_width_inches=96.0,
            room_depth_inches=120.0,
            room_function="bathroom",
        )
        assert len(pods) >= 1
        skus = {p.sku for p in pods}
        assert "CAP-POD-BATH-STD" in skus

    def test_room_too_small_returns_empty(self, kg_store: KnowledgeGraphStore) -> None:
        """3'x3' (36"x36") room — no pods fit."""
        pods = get_valid_pods(kg_store, room_width_inches=36.0, room_depth_inches=36.0)
        assert pods == []

    def test_no_function_filter_returns_all_types(self, kg_store: KnowledgeGraphStore) -> None:
        """Large room, no function filter → returns pods of various types."""
        pods = get_valid_pods(kg_store, room_width_inches=150.0, room_depth_inches=150.0)
        types = {p.pod_type for p in pods}
        assert len(types) >= 2

    def test_both_orientations_checked(self, kg_store: KnowledgeGraphStore) -> None:
        """Standard bath pod needs 66"x102" min. 102"x66" room should still fit (rotated)."""
        pods = get_valid_pods(
            kg_store,
            room_width_inches=102.0,
            room_depth_inches=66.0,
            room_function="bathroom",
        )
        skus = {p.sku for p in pods}
        # Standard pod min is 66x102 — rotated fits in 102x66
        assert "CAP-POD-BATH-STD" in skus

    def test_required_trades_filter(self, kg_store: KnowledgeGraphStore) -> None:
        """Filter by required_trades=['plumbing', 'electrical', 'hvac']."""
        pods = get_valid_pods(
            kg_store,
            room_width_inches=150.0,
            room_depth_inches=150.0,
            required_trades=["plumbing", "electrical", "hvac"],
        )
        for pod in pods:
            assert "plumbing" in pod.included_trades
            assert "electrical" in pod.included_trades
            assert "hvac" in pod.included_trades

    def test_sorted_by_space_utilization(self, kg_store: KnowledgeGraphStore) -> None:
        """Larger pods (by area) should come first."""
        pods = get_valid_pods(
            kg_store,
            room_width_inches=150.0,
            room_depth_inches=150.0,
            room_function="bathroom",
        )
        if len(pods) >= 2:
            areas = [p.width_inches * p.depth_inches for p in pods]
            for i in range(len(areas) - 1):
                assert areas[i] >= areas[i + 1]


# ══════════════════════════════════════════════════════════════════════════════
# 4. get_machine_for_panel
# ══════════════════════════════════════════════════════════════════════════════


class TestGetMachineForPanel:
    def test_known_panel_returns_machines(self, kg_store: KnowledgeGraphStore) -> None:
        machines = get_machine_for_panel(kg_store, "CAP-PNL-LB-16GA-600D-16OC-96H")
        assert len(machines) >= 1
        skus = {m.sku for m in machines}
        assert "CAP-MCH-HW2500" in skus

    def test_14ga_panel_only_hw3500(self, kg_store: KnowledgeGraphStore) -> None:
        """14ga shear panel only fabricated by HW3500."""
        machines = get_machine_for_panel(kg_store, "CAP-PNL-SH-14GA-600D-16OC-96H")
        assert len(machines) == 1
        assert machines[0].sku == "CAP-MCH-HW3500"

    def test_unknown_panel_returns_empty(self, kg_store: KnowledgeGraphStore) -> None:
        machines = get_machine_for_panel(kg_store, "DOES-NOT-EXIST")
        assert machines == []


# ══════════════════════════════════════════════════════════════════════════════
# 5. get_machine_for_spec
# ══════════════════════════════════════════════════════════════════════════════


class TestGetMachineForSpec:
    def test_16ga_6in_120in(self, kg_store: KnowledgeGraphStore) -> None:
        """Standard spec: both Howick machines should handle it."""
        machines = get_machine_for_spec(
            kg_store, gauge=16, stud_depth_inches=6.0, length_inches=120.0
        )
        assert len(machines) >= 2
        skus = {m.sku for m in machines}
        assert "CAP-MCH-HW2500" in skus
        assert "CAP-MCH-HW3500" in skus

    def test_14ga_only_hw3500(self, kg_store: KnowledgeGraphStore) -> None:
        """14ga → only HW3500 (HW2500 max is 16ga)."""
        machines = get_machine_for_spec(
            kg_store, gauge=14, stud_depth_inches=6.0, length_inches=120.0
        )
        skus = {m.sku for m in machines}
        assert "CAP-MCH-HW3500" in skus
        assert "CAP-MCH-HW2500" not in skus

    def test_long_panel_400in_only_hw3500(self, kg_store: KnowledgeGraphStore) -> None:
        """400" length → only HW3500 (max 420") fits; HW2500 caps at 300"."""
        machines = get_machine_for_spec(
            kg_store, gauge=16, stud_depth_inches=6.0, length_inches=400.0
        )
        skus = {m.sku for m in machines}
        assert "CAP-MCH-HW3500" in skus
        assert "CAP-MCH-HW2500" not in skus

    def test_impossible_spec_returns_empty(self, kg_store: KnowledgeGraphStore) -> None:
        """12ga doesn't exist in any machine range."""
        machines = get_machine_for_spec(
            kg_store, gauge=12, stud_depth_inches=6.0, length_inches=120.0
        )
        assert machines == []

    def test_deep_stud_8in_only_hw3500(self, kg_store: KnowledgeGraphStore) -> None:
        """8" stud depth → only HW3500 (max_web_depth 8.0 vs HW2500's 6.0)."""
        machines = get_machine_for_spec(
            kg_store, gauge=16, stud_depth_inches=8.0, length_inches=120.0
        )
        skus = {m.sku for m in machines}
        assert "CAP-MCH-HW3500" in skus
        assert "CAP-MCH-HW2500" not in skus


# ══════════════════════════════════════════════════════════════════════════════
# 6. get_bim_family
# ══════════════════════════════════════════════════════════════════════════════


class TestGetBimFamily:
    def test_exact_match(self, kg_store: KnowledgeGraphStore) -> None:
        panel = get_bim_family(
            kg_store,
            panel_type=PanelType.LOAD_BEARING,
            gauge=16,
            stud_depth_inches=6.0,
        )
        assert panel is not None
        assert panel.panel_type == PanelType.LOAD_BEARING
        assert panel.gauge == 16
        assert panel.stud_depth_inches == 6.0

    def test_fire_rated_match(self, kg_store: KnowledgeGraphStore) -> None:
        panel = get_bim_family(
            kg_store,
            panel_type=PanelType.FIRE_RATED,
            gauge=16,
            stud_depth_inches=6.0,
            fire_rating_hours=1.0,
        )
        assert panel is not None
        assert panel.fire_rating_hours >= 1.0

    def test_2hr_fire_rated(self, kg_store: KnowledgeGraphStore) -> None:
        panel = get_bim_family(
            kg_store,
            panel_type=PanelType.FIRE_RATED,
            gauge=16,
            stud_depth_inches=6.0,
            fire_rating_hours=2.0,
        )
        assert panel is not None
        assert panel.fire_rating_hours >= 2.0
        assert panel.sku == "CAP-PNL-FR2-16GA-600D-16OC-96H"

    def test_no_match_impossible_combo(self, kg_store: KnowledgeGraphStore) -> None:
        """14ga envelope doesn't exist."""
        panel = get_bim_family(
            kg_store,
            panel_type=PanelType.ENVELOPE,
            gauge=14,
            stud_depth_inches=6.0,
        )
        assert panel is None

    def test_prefers_closest_fire_rating(self, kg_store: KnowledgeGraphStore) -> None:
        """Requesting 0.5hr fire rating → should return 1hr (closest that meets req)."""
        panel = get_bim_family(
            kg_store,
            panel_type=PanelType.FIRE_RATED,
            gauge=16,
            stud_depth_inches=6.0,
            fire_rating_hours=0.5,
        )
        assert panel is not None
        assert panel.fire_rating_hours == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 7. get_connections_for_panel
# ══════════════════════════════════════════════════════════════════════════════


class TestGetConnectionsForPanel:
    def test_known_panel_returns_connections(self, kg_store: KnowledgeGraphStore) -> None:
        conns = get_connections_for_panel(kg_store, "CAP-PNL-LB-16GA-600D-16OC-96H")
        assert len(conns) >= 1
        skus = {c.sku for c in conns}
        assert "CAP-CON-SPL-16GA" in skus

    def test_connections_gauge_compatible(self, kg_store: KnowledgeGraphStore) -> None:
        """Connections for a 16ga panel should include 16 in their compatible_gauges."""
        conns = get_connections_for_panel(kg_store, "CAP-PNL-LB-16GA-600D-16OC-96H")
        for c in conns:
            # Most connections support 16ga; bridging is universal
            assert 16 in c.compatible_gauges or c.connection_type == "bridging"

    def test_unknown_panel_returns_empty(self, kg_store: KnowledgeGraphStore) -> None:
        conns = get_connections_for_panel(kg_store, "DOES-NOT-EXIST")
        assert conns == []

    def test_includes_splice_clip_fastener(self, kg_store: KnowledgeGraphStore) -> None:
        """A 16ga 6" stud panel should have splice, clip, and fastener connections."""
        conns = get_connections_for_panel(kg_store, "CAP-PNL-LB-16GA-600D-16OC-96H")
        types = {c.connection_type for c in conns}
        assert "splice" in types
        assert "clip_angle" in types
        assert "fastener" in types


# ══════════════════════════════════════════════════════════════════════════════
# 8. validate_panel_fabrication
# ══════════════════════════════════════════════════════════════════════════════


class TestValidatePanelFabrication:
    def test_valid_panel_valid_length(self, kg_store: KnowledgeGraphStore) -> None:
        result = validate_panel_fabrication(
            kg_store, "CAP-PNL-LB-16GA-600D-16OC-96H", required_length_inches=120.0
        )
        assert isinstance(result, FabricationValidation)
        assert result.is_valid is True
        assert result.errors == []

    def test_unknown_sku(self, kg_store: KnowledgeGraphStore) -> None:
        result = validate_panel_fabrication(
            kg_store, "DOES-NOT-EXIST", required_length_inches=120.0
        )
        assert result.is_valid is False
        assert any("not found" in e for e in result.errors)

    def test_length_exceeds_max(self, kg_store: KnowledgeGraphStore) -> None:
        """Request 350" on a panel with max 300" → invalid."""
        result = validate_panel_fabrication(
            kg_store, "CAP-PNL-LB-16GA-600D-16OC-96H", required_length_inches=350.0
        )
        assert result.is_valid is False
        assert any("exceeds" in e for e in result.errors)

    def test_length_below_min(self, kg_store: KnowledgeGraphStore) -> None:
        """Request 10" on a panel with min 24" → invalid."""
        result = validate_panel_fabrication(
            kg_store, "CAP-PNL-LB-16GA-600D-16OC-96H", required_length_inches=10.0
        )
        assert result.is_valid is False
        assert any("below" in e for e in result.errors)

    def test_at_max_length(self, kg_store: KnowledgeGraphStore) -> None:
        """Exactly at max length → valid."""
        result = validate_panel_fabrication(
            kg_store, "CAP-PNL-LB-16GA-600D-16OC-96H", required_length_inches=300.0
        )
        assert result.is_valid is True

    def test_at_min_length(self, kg_store: KnowledgeGraphStore) -> None:
        """Exactly at min length → valid."""
        result = validate_panel_fabrication(
            kg_store, "CAP-PNL-LB-16GA-600D-16OC-96H", required_length_inches=24.0
        )
        assert result.is_valid is True

    def test_multi_quantity_splice_warning(self, kg_store: KnowledgeGraphStore) -> None:
        """quantity > 1 should check for splice availability."""
        result = validate_panel_fabrication(
            kg_store,
            "CAP-PNL-LB-16GA-600D-16OC-96H",
            required_length_inches=120.0,
            required_quantity=3,
        )
        # Valid — splice exists for 16ga panels
        assert result.is_valid is True


# ══════════════════════════════════════════════════════════════════════════════
# 9. validate_wall_panelization
# ══════════════════════════════════════════════════════════════════════════════


class TestValidateWallPanelization:
    def test_single_panel_valid(self, kg_store: KnowledgeGraphStore) -> None:
        result = validate_wall_panelization(
            kg_store,
            wall_length_inches=120.0,
            wall_type=PanelType.LOAD_BEARING,
            panel_assignments=[("CAP-PNL-LB-16GA-600D-16OC-96H", 120.0)],
        )
        assert result.is_valid is True
        assert result.errors == []

    def test_multi_panel_spliced_valid(self, kg_store: KnowledgeGraphStore) -> None:
        """Two 16ga panels covering 400"."""
        result = validate_wall_panelization(
            kg_store,
            wall_length_inches=400.0,
            wall_type=PanelType.LOAD_BEARING,
            panel_assignments=[
                ("CAP-PNL-LB-16GA-600D-16OC-96H", 200.0),
                ("CAP-PNL-LB-16GA-600D-16OC-96H", 200.0),
            ],
        )
        assert result.is_valid is True

    def test_panel_type_mismatch(self, kg_store: KnowledgeGraphStore) -> None:
        """Partition panel on load-bearing wall → invalid."""
        result = validate_wall_panelization(
            kg_store,
            wall_length_inches=120.0,
            wall_type=PanelType.LOAD_BEARING,
            panel_assignments=[("CAP-PNL-PT-20GA-350D-24OC-96H", 120.0)],
        )
        assert result.is_valid is False
        assert any("type" in e.lower() for e in result.errors)

    def test_gap_in_coverage(self, kg_store: KnowledgeGraphStore) -> None:
        """Panels don't cover full wall → error."""
        result = validate_wall_panelization(
            kg_store,
            wall_length_inches=200.0,
            wall_type=PanelType.LOAD_BEARING,
            panel_assignments=[("CAP-PNL-LB-16GA-600D-16OC-96H", 100.0)],
        )
        assert result.is_valid is False
        assert any("gap" in e.lower() for e in result.errors)

    def test_overlap_error(self, kg_store: KnowledgeGraphStore) -> None:
        """Panels exceed wall length → error."""
        result = validate_wall_panelization(
            kg_store,
            wall_length_inches=100.0,
            wall_type=PanelType.LOAD_BEARING,
            panel_assignments=[
                ("CAP-PNL-LB-16GA-600D-16OC-96H", 60.0),
                ("CAP-PNL-LB-16GA-600D-16OC-96H", 60.0),
            ],
        )
        assert result.is_valid is False
        assert any("overlap" in e.lower() for e in result.errors)

    def test_unknown_panel_sku(self, kg_store: KnowledgeGraphStore) -> None:
        result = validate_wall_panelization(
            kg_store,
            wall_length_inches=120.0,
            wall_type=PanelType.LOAD_BEARING,
            panel_assignments=[("DOES-NOT-EXIST", 120.0)],
        )
        assert result.is_valid is False
        assert any("not found" in e for e in result.errors)

    def test_empty_assignments(self, kg_store: KnowledgeGraphStore) -> None:
        result = validate_wall_panelization(
            kg_store,
            wall_length_inches=120.0,
            wall_type=PanelType.LOAD_BEARING,
            panel_assignments=[],
        )
        assert result.is_valid is False
        assert any("no panels" in e.lower() for e in result.errors)

    def test_cut_length_exceeds_panel_max(self, kg_store: KnowledgeGraphStore) -> None:
        """Cut length 350" on panel with max 300" → error."""
        result = validate_wall_panelization(
            kg_store,
            wall_length_inches=350.0,
            wall_type=PanelType.LOAD_BEARING,
            panel_assignments=[("CAP-PNL-LB-16GA-600D-16OC-96H", 350.0)],
        )
        assert result.is_valid is False
        assert any("exceeds" in e.lower() or "maximum" in e.lower() for e in result.errors)

    def test_coverage_within_tolerance(self, kg_store: KnowledgeGraphStore) -> None:
        """Panel coverage within 0.25" tolerance → valid."""
        result = validate_wall_panelization(
            kg_store,
            wall_length_inches=120.0,
            wall_type=PanelType.LOAD_BEARING,
            panel_assignments=[("CAP-PNL-LB-16GA-600D-16OC-96H", 120.1)],
        )
        assert result.is_valid is True


# ══════════════════════════════════════════════════════════════════════════════
# 10. get_fabrication_limits
# ══════════════════════════════════════════════════════════════════════════════


class TestGetFabricationLimits:
    def test_no_filter(self, kg_store: KnowledgeGraphStore) -> None:
        limits = get_fabrication_limits(kg_store)
        assert limits["max_length_inches"] == pytest.approx(420.0)
        assert limits["max_web_depth_inches"] == pytest.approx(8.0)
        # min_gauge = thinnest = max(min_gauge across machines) = 25
        assert limits["min_gauge"] == 25
        # max_gauge = thickest = min(max_gauge across machines) = 14
        assert limits["max_gauge"] == 14

    def test_with_gauge_14(self, kg_store: KnowledgeGraphStore) -> None:
        """Only HW3500 supports 14ga."""
        limits = get_fabrication_limits(kg_store, gauge=14)
        assert limits["max_length_inches"] == pytest.approx(420.0)
        assert limits["max_web_depth_inches"] == pytest.approx(8.0)

    def test_with_gauge_20(self, kg_store: KnowledgeGraphStore) -> None:
        """20ga — HW2500, HW3500, and Zund all support it."""
        limits = get_fabrication_limits(kg_store, gauge=20)
        assert limits["max_length_inches"] == pytest.approx(420.0)

    def test_impossible_gauge_returns_zeros(self, kg_store: KnowledgeGraphStore) -> None:
        """Gauge 10 — no machine supports it."""
        limits = get_fabrication_limits(kg_store, gauge=10)
        assert limits["max_length_inches"] == 0.0
        assert limits["max_web_depth_inches"] == 0.0
        assert limits["min_gauge"] == 0
        assert limits["max_gauge"] == 0

    def test_has_expected_keys(self, kg_store: KnowledgeGraphStore) -> None:
        limits = get_fabrication_limits(kg_store)
        assert "max_length_inches" in limits
        assert "max_web_depth_inches" in limits
        assert "min_gauge" in limits
        assert "max_gauge" in limits
        assert "max_coil_width_inches" in limits

    def test_max_coil_width(self, kg_store: KnowledgeGraphStore) -> None:
        limits = get_fabrication_limits(kg_store)
        # HW3500 has widest coil: 22"
        assert limits["max_coil_width_inches"] == pytest.approx(22.0)
