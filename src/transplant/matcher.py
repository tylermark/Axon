"""BT-001: KG -> BIM family matching.

For each panel assignment in the DRL output, looks up the panel SKU in the
Knowledge Graph to retrieve the full product specification, then builds a
``BIMFamilyMatch`` that maps the 2D panel slot to its 3D BIM family with
material layers.

The family name follows Revit convention:
    ``CFS_Wall_{depth}S{spacing}-{gauge}_{type_code}_{fire_code}``

Example: ``CFS_Wall_362S162-54_LB_1HR``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.knowledge_graph.schema import Panel, PanelType

if TYPE_CHECKING:
    from docs.interfaces.drl_output import PanelizationResult
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)

# ── Panel type -> Revit code mapping ──────────────────────────────────────

_TYPE_CODES: dict[str, str] = {
    PanelType.LOAD_BEARING: "LB",
    PanelType.PARTITION: "PT",
    PanelType.SHEAR: "SW",
    PanelType.FIRE_RATED: "FR",
    PanelType.ENVELOPE: "EN",
}


# ── Dataclass ─────────────────────────────────────────────────────────────


@dataclass
class BIMFamilyMatch:
    """Maps a 2D panel slot to its 3D BIM family via deterministic KG lookup.

    Attributes:
        edge_id: Wall edge this panel belongs to.
        panel_index: Index within the wall's panel list.
        panel_sku: KG product SKU.
        family_name: Revit-style family name.
        panel_type: Panel classification (load_bearing, partition, etc.).
        gauge: Steel gauge number.
        stud_depth_inches: Stud web depth in inches.
        stud_spacing_inches: On-center stud spacing in inches.
        height_inches: Panel height in inches.
        cut_length_inches: Actual cut length for this piece in inches.
        fire_rating_hours: Fire resistance rating in hours.
        sheathing_type: Sheathing material or None.
        sheathing_thickness_inches: Sheathing thickness or None.
        insulation_type: Insulation material or None.
        insulation_r_value: Insulation R-value or None.
        material_layers: Ordered material layers, inner-to-outer.
    """

    edge_id: int
    panel_index: int
    panel_sku: str
    family_name: str
    panel_type: str
    gauge: int
    stud_depth_inches: float
    stud_spacing_inches: float
    height_inches: float
    cut_length_inches: float
    fire_rating_hours: float
    sheathing_type: str | None
    sheathing_thickness_inches: float | None
    insulation_type: str | None
    insulation_r_value: float | None
    material_layers: list[dict] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────


def match_bim_families(
    result: PanelizationResult,
    store: KnowledgeGraphStore,
) -> list[BIMFamilyMatch]:
    """Match every panel assignment to its 3D BIM family via KG lookup.

    Iterates over all walls in ``result.panel_map``, and for each
    ``PanelAssignment`` looks up the full ``Panel`` object from the KG
    store to build a ``BIMFamilyMatch`` with material layers.

    Args:
        result: The DRL agent's panelization output.
        store: The loaded Knowledge Graph store.

    Returns:
        List of ``BIMFamilyMatch`` instances, one per panel assignment
        across all walls, ordered by (edge_id, panel_index).

    Raises:
        ValueError: If a panel SKU from the DRL output is not found in
            the KG store.  This indicates a KG/DRL contract violation.
    """
    matches: list[BIMFamilyMatch] = []

    for wall_pan in result.panel_map.walls:
        if not wall_pan.is_panelizable:
            logger.debug(
                "Skipping non-panelizable wall edge_id=%d: %s",
                wall_pan.edge_id,
                wall_pan.rejection_reason,
            )
            continue

        for assignment in wall_pan.panels:
            panel = store.panels.get(assignment.panel_sku)
            if panel is None:
                raise ValueError(
                    f"Panel SKU '{assignment.panel_sku}' (wall edge_id="
                    f"{wall_pan.edge_id}, panel_index={assignment.panel_index}) "
                    f"not found in Knowledge Graph store. "
                    f"This is a KG/DRL contract violation."
                )

            family_name = _build_family_name(panel)
            layers = _build_material_layers(panel)

            matches.append(
                BIMFamilyMatch(
                    edge_id=wall_pan.edge_id,
                    panel_index=assignment.panel_index,
                    panel_sku=assignment.panel_sku,
                    family_name=family_name,
                    panel_type=panel.panel_type.value,
                    gauge=panel.gauge,
                    stud_depth_inches=panel.stud_depth_inches,
                    stud_spacing_inches=panel.stud_spacing_inches,
                    height_inches=panel.height_inches,
                    cut_length_inches=assignment.cut_length_inches,
                    fire_rating_hours=panel.fire_rating_hours,
                    sheathing_type=panel.sheathing_type,
                    sheathing_thickness_inches=panel.sheathing_thickness_inches,
                    insulation_type=panel.insulation_type,
                    insulation_r_value=panel.insulation_r_value,
                    material_layers=layers,
                )
            )

    logger.info("Matched %d panel assignments to BIM families", len(matches))
    return matches


# ── Helpers ───────────────────────────────────────────────────────────────


def _build_family_name(panel: Panel) -> str:
    """Build a Revit-style CFS wall family name from panel attributes.

    Format: ``CFS_Wall_{depth}S{spacing}-{gauge}_{type}_{fire}``

    Examples:
        - ``CFS_Wall_362S162-54_LB_1HR``
        - ``CFS_Wall_600S200-43_SW_2HR``
        - ``CFS_Wall_250S125-33_PT_0HR``

    The depth/spacing values use the SSMA stud designation convention:
    web depth and flange width in hundredths of an inch.
    """
    # Convert decimal inches to SSMA hundredths (e.g., 3.625 -> 362)
    depth_code = round(panel.stud_depth_inches * 100)
    spacing_code = round(panel.stud_spacing_inches * 100)

    type_code = _TYPE_CODES.get(panel.panel_type, "XX")

    fire_hours = panel.fire_rating_hours
    if fire_hours <= 0:
        fire_code = "0HR"
    elif fire_hours == int(fire_hours):
        fire_code = f"{int(fire_hours)}HR"
    else:
        fire_code = f"{fire_hours:.1f}HR"

    return f"CFS_Wall_{depth_code}S{spacing_code}-{panel.gauge}_{type_code}_{fire_code}"


def _build_material_layers(panel: Panel) -> list[dict]:
    """Build ordered material layers (inner face to outer face).

    Layer order for a typical CFS wall assembly:
        1. Interior finish (drywall) — assumed, not in KG
        2. Stud framing
        3. Insulation (if present)
        4. Sheathing (if present)
        5. Exterior finish — assumed for envelope, not in KG

    Only layers with KG data are included; assumed layers are omitted.
    """
    layers: list[dict] = []

    # Track layer — bottom of wall
    layers.append(
        {
            "name": "track",
            "material": f"CFS_{panel.gauge}ga_track",
            "function": "framing",
            "thickness_inches": panel.stud_depth_inches,
        }
    )

    # Stud framing — the structural core
    layers.append(
        {
            "name": "stud",
            "material": f"CFS_{panel.gauge}ga_stud",
            "function": "structural",
            "thickness_inches": panel.stud_depth_inches,
            "spacing_inches": panel.stud_spacing_inches,
        }
    )

    # Insulation (if present, fills stud cavity)
    if panel.insulation_type is not None:
        layers.append(
            {
                "name": "insulation",
                "material": panel.insulation_type,
                "function": "thermal",
                "r_value": panel.insulation_r_value,
            }
        )

    # Sheathing (if present, exterior face of framing)
    if panel.sheathing_type is not None:
        layers.append(
            {
                "name": "sheathing",
                "material": panel.sheathing_type,
                "function": "shear_bracing",
                "thickness_inches": panel.sheathing_thickness_inches,
            }
        )

    return layers
