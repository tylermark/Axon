"""BT-004: IFC4 serialization (ISO 16739-1:2024).

Creates a valid IFC4-SPF file from wall assemblies, opening attachments, and
room/pod placements.  Uses ``ifcopenshell`` when available; falls back to a
structured JSON representation when the C++ dependency is missing.

IFC structure::

    IfcProject
      -> IfcSite
        -> IfcBuilding
          -> IfcBuildingStorey
            -> IfcWallStandardCase (per wall assembly)
              -> IfcOpeningElement (per void, via IfcRelVoidsElement)
            -> IfcSpace (per room)
              -> IfcProduct (pod placement, if any)

Every ``IfcWallStandardCase`` carries property sets with real product SKUs,
gauge, fire rating, and panel type — per Critical Constraint #4.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from docs.interfaces.drl_output import PanelizationResult
    from src.transplant.assembler import WallAssembly
    from src.transplant.openings import OpeningAttachment

logger = logging.getLogger(__name__)

# ── ifcopenshell availability ─────────────────────────────────────────────

try:
    import ifcopenshell
    import ifcopenshell.api
    import ifcopenshell.guid

    _HAS_IFCOPENSHELL = True
except ImportError:
    _HAS_IFCOPENSHELL = False
    logger.info(
        "ifcopenshell not installed — IFC export will use JSON fallback. "
        "Install ifcopenshell for native IFC4-SPF output."
    )

# Inches to mm
_INCHES_TO_MM: float = 25.4


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def export_ifc(
    assemblies: list[WallAssembly],
    openings: list[OpeningAttachment],
    result: PanelizationResult,
    output_path: str | Path,
) -> Path:
    """Export the BIM model to IFC4 format (or JSON fallback).

    When ``ifcopenshell`` is available, writes a valid IFC4-SPF file
    conforming to ISO 16739-1:2024.  When unavailable, writes a
    structured JSON file containing identical semantic data.

    The output file extension determines the format:
    - ``.ifc`` — IFC4-SPF (requires ifcopenshell)
    - ``.json`` — always uses JSON fallback regardless of ifcopenshell

    If the path ends in ``.ifc`` but ifcopenshell is missing, the
    extension is changed to ``.json`` and a warning is logged.

    Args:
        assemblies: Wall assemblies from :func:`assemble_walls`.
        openings: Opening attachments from :func:`attach_openings`.
        result: The DRL agent's panelization output.
        output_path: Destination file path.

    Returns:
        The actual path written (may differ from ``output_path`` if
        fallback was used).
    """
    output_path = Path(output_path)

    # Decide format
    use_ifc = _HAS_IFCOPENSHELL and output_path.suffix.lower() == ".ifc"

    if output_path.suffix.lower() == ".ifc" and not _HAS_IFCOPENSHELL:
        output_path = output_path.with_suffix(".json")
        logger.warning(
            "ifcopenshell not available — writing JSON fallback to %s",
            output_path,
        )

    if output_path.suffix.lower() == ".json":
        use_ifc = False

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_ifc:
        return _export_ifc_native(assemblies, openings, result, output_path)
    return _export_json_fallback(assemblies, openings, result, output_path)


# ══════════════════════════════════════════════════════════════════════════════
# Native IFC4 export (ifcopenshell)
# ══════════════════════════════════════════════════════════════════════════════


def _export_ifc_native(
    assemblies: list[WallAssembly],
    openings: list[OpeningAttachment],
    result: PanelizationResult,
    output_path: Path,
) -> Path:
    """Write an IFC4-SPF file using ifcopenshell.

    Follows the IFC4 ADD2 TC1 schema (ISO 16739-1:2024).
    """
    ifc = ifcopenshell.api.run("project.create_file", version="IFC4")

    # ── 1. Project hierarchy ──────────────────────────────────────────────
    project = ifcopenshell.api.run(
        "root.create_entity",
        ifc,
        ifc_class="IfcProject",
        name="Axon BIM Transplant Export",
    )

    # Length unit: millimeters
    length_unit = ifcopenshell.api.run("unit.add_si_unit", ifc, unit_type="LENGTHUNIT")
    ifcopenshell.api.run(
        "unit.assign_unit",
        ifc,
        units=[length_unit],
    )

    # Context for 3D geometry
    model_context = ifcopenshell.api.run(
        "context.add_context",
        ifc,
        context_type="Model",
    )
    body_context = ifcopenshell.api.run(
        "context.add_context",
        ifc,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=model_context,
    )

    site = ifcopenshell.api.run(
        "root.create_entity",
        ifc,
        ifc_class="IfcSite",
        name="Default Site",
    )
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)

    building = ifcopenshell.api.run(
        "root.create_entity",
        ifc,
        ifc_class="IfcBuilding",
        name="Default Building",
    )
    ifcopenshell.api.run("aggregate.assign_object", ifc, products=[building], relating_object=site)

    storey = ifcopenshell.api.run(
        "root.create_entity",
        ifc,
        ifc_class="IfcBuildingStorey",
        name="Level 1",
    )
    ifcopenshell.api.run(
        "aggregate.assign_object", ifc, products=[storey], relating_object=building
    )

    # ── 2. Walls ──────────────────────────────────────────────────────────
    assembly_lookup: dict[int, Any] = {}  # edge_id -> IfcWallStandardCase

    for assembly in assemblies:
        wall = _create_ifc_wall(ifc, assembly, body_context)
        ifcopenshell.api.run(
            "spatial.assign_container", ifc, products=[wall], relating_structure=storey
        )
        assembly_lookup[assembly.edge_id] = wall

    # ── 3. Openings ───────────────────────────────────────────────────────
    openings_by_wall: dict[int, list[OpeningAttachment]] = {}
    for op in openings:
        openings_by_wall.setdefault(op.host_edge_id, []).append(op)

    for edge_id, wall_openings in openings_by_wall.items():
        host_wall = assembly_lookup.get(edge_id)
        if host_wall is None:
            continue
        for op_attach in wall_openings:
            _create_ifc_opening(ifc, op_attach, host_wall, body_context)

    # ── 4. Rooms and pod placements ───────────────────────────────────────
    source_graph = result.source_graph.graph
    scale = source_graph.scale_factor

    for room_placement in result.placement_map.rooms:
        room = _find_room(source_graph.rooms, room_placement.room_id)
        if room is None or room.is_exterior:
            continue

        ifc_space = ifcopenshell.api.run(
            "root.create_entity",
            ifc,
            ifc_class="IfcSpace",
            name=room.label or f"Room_{room.room_id}",
        )
        ifcopenshell.api.run(
            "aggregate.assign_object", ifc, products=[ifc_space], relating_object=storey
        )

        # Space boundaries to bounding walls — created directly
        for boundary_edge_id in room.boundary_edges:
            host_wall = assembly_lookup.get(boundary_edge_id)
            if host_wall is not None:
                ifc.create_entity(
                    "IfcRelSpaceBoundary",
                    GlobalId=ifcopenshell.guid.new(),
                    RelatingSpace=ifc_space,
                    RelatedBuildingElement=host_wall,
                    PhysicalOrVirtualBoundary="PHYSICAL",
                    InternalOrExternalBoundary="INTERNAL",
                )

        # Pod placement
        if room_placement.placement is not None:
            _create_ifc_pod(ifc, room_placement, scale, body_context, storey)

    # ── 5. Write ──────────────────────────────────────────────────────────
    ifc.write(str(output_path))
    logger.info("IFC4-SPF written to %s", output_path)
    return output_path


def _create_ifc_wall(
    ifc: Any,
    assembly: WallAssembly,
    body_context: Any,
) -> Any:
    """Create an IfcWallStandardCase with SweptSolid representation.

    The wall is modeled as a rectangular profile (thickness x wall_length)
    extruded to wall height along the Z axis, placed at the wall start
    coordinate and rotated to the wall angle.
    """
    wall = ifcopenshell.api.run(
        "root.create_entity",
        ifc,
        ifc_class="IfcWallStandardCase",
        name=f"Wall_{assembly.edge_id}",
    )

    # Placement at wall start, rotated to wall angle
    x, y = assembly.wall_start
    angle = assembly.wall_angle_rad

    matrix = np.eye(4, dtype=float)
    matrix[0, 0] = math.cos(angle)
    matrix[0, 1] = -math.sin(angle)
    matrix[1, 0] = math.sin(angle)
    matrix[1, 1] = math.cos(angle)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = 0.0

    ifcopenshell.api.run(
        "geometry.edit_object_placement",
        ifc,
        product=wall,
        matrix=matrix,
    )

    # SweptSolid: rectangular profile extruded along Z
    # Profile: wall_length along X (local), thickness along Y (local)
    representation = ifcopenshell.api.run(
        "geometry.add_wall_representation",
        ifc,
        context=body_context,
        length=assembly.wall_length_mm,
        height=assembly.height_mm,
        thickness=assembly.thickness_mm,
    )
    ifcopenshell.api.run(
        "geometry.assign_representation",
        ifc,
        product=wall,
        representation=representation,
    )

    # Property sets with real product data
    if assembly.panels:
        first_panel = assembly.panels[0]
        panel_skus = ", ".join(p.panel_sku for p in assembly.panels)

        pset = ifcopenshell.api.run(
            "pset.add_pset",
            ifc,
            product=wall,
            name="Axon_PanelData",
        )
        ifcopenshell.api.run(
            "pset.edit_pset",
            ifc,
            pset=pset,
            properties={
                "PanelSKUs": panel_skus,
                "PanelType": first_panel.panel_type,
                "Gauge": first_panel.gauge,
                "StudDepthInches": first_panel.stud_depth_inches,
                "StudSpacingInches": first_panel.stud_spacing_inches,
                "FireRatingHours": first_panel.fire_rating_hours,
                "FamilyName": first_panel.family_name,
                "PanelCount": len(assembly.panels),
                "SeamCount": len(assembly.seam_positions_mm),
            },
        )

    return wall


def _create_ifc_opening(
    ifc: Any,
    op_attach: OpeningAttachment,
    host_wall: Any,
    body_context: Any,
) -> None:
    """Create an IfcOpeningElement and link via IfcRelVoidsElement."""
    opening = ifcopenshell.api.run(
        "root.create_entity",
        ifc,
        ifc_class="IfcOpeningElement",
        name=f"{op_attach.opening_type}_{op_attach.host_edge_id}",
    )

    # Place the opening relative to the host wall
    cx, cy, z_bottom, _z_top = op_attach.void_coords
    matrix = np.eye(4, dtype=float)
    matrix[0, 3] = cx
    matrix[1, 3] = cy
    matrix[2, 3] = z_bottom

    ifcopenshell.api.run(
        "geometry.edit_object_placement",
        ifc,
        product=opening,
        matrix=matrix,
    )

    # Void geometry: box width x depth x height
    # Depth = through the wall thickness — we use a generous value
    # Width and height from the opening attachment
    void_depth = 1000.0  # mm, through entire wall assembly
    representation = ifcopenshell.api.run(
        "geometry.add_wall_representation",
        ifc,
        context=body_context,
        length=op_attach.width_mm,
        height=op_attach.height_mm,
        thickness=void_depth,
    )
    ifcopenshell.api.run(
        "geometry.assign_representation",
        ifc,
        product=opening,
        representation=representation,
    )

    # IfcRelVoidsElement — created directly (no high-level void API)
    ifc.create_entity(
        "IfcRelVoidsElement",
        GlobalId=ifcopenshell.guid.new(),
        RelatingBuildingElement=host_wall,
        RelatedOpeningElement=opening,
    )


def _create_ifc_pod(
    ifc: Any,
    room_placement: Any,
    scale: float,
    body_context: Any,
    storey: Any,
) -> None:
    """Create an IfcProduct for a placed pod."""
    placement = room_placement.placement
    pod_name = f"Pod_{placement.pod_sku}_{room_placement.room_id}"

    pod_element = ifcopenshell.api.run(
        "root.create_entity",
        ifc,
        ifc_class="IfcFurnishingElement",
        name=pod_name,
    )

    # Position from PDF user units to mm
    pos_x = float(placement.position[0]) * scale
    pos_y = float(placement.position[1]) * scale
    angle_rad = math.radians(placement.orientation_deg)

    matrix = np.eye(4, dtype=float)
    matrix[0, 0] = math.cos(angle_rad)
    matrix[0, 1] = -math.sin(angle_rad)
    matrix[1, 0] = math.sin(angle_rad)
    matrix[1, 1] = math.cos(angle_rad)
    matrix[0, 3] = pos_x
    matrix[1, 3] = pos_y

    ifcopenshell.api.run(
        "geometry.edit_object_placement",
        ifc,
        product=pod_element,
        matrix=matrix,
    )

    ifcopenshell.api.run(
        "spatial.assign_container", ifc, products=[pod_element], relating_structure=storey
    )

    # Property set with pod data
    pset = ifcopenshell.api.run(
        "pset.add_pset",
        ifc,
        product=pod_element,
        name="Axon_PodData",
    )
    ifcopenshell.api.run(
        "pset.edit_pset",
        ifc,
        pset=pset,
        properties={
            "PodSKU": placement.pod_sku,
            "RoomId": room_placement.room_id,
            "RoomLabel": room_placement.room_label,
            "OrientationDeg": placement.orientation_deg,
            "ClearanceMet": placement.clearance_met,
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# JSON fallback export
# ══════════════════════════════════════════════════════════════════════════════


def _export_json_fallback(
    assemblies: list[WallAssembly],
    openings: list[OpeningAttachment],
    result: PanelizationResult,
    output_path: Path,
) -> Path:
    """Write a structured JSON representation of the IFC model.

    Contains the same semantic data as the native IFC export, enabling
    testing and validation without the ifcopenshell C++ dependency.
    The JSON schema mirrors the IFC entity hierarchy.
    """
    source_graph = result.source_graph.graph
    scale = source_graph.scale_factor
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    model: dict[str, Any] = {
        "_format": "axon_ifc_fallback_json",
        "_schema": "IFC4",
        "_description": "Axon BIM Transplant Export (JSON fallback — ifcopenshell not available)",
        "_timestamp": timestamp,
        "_coordinate_unit": "millimeters",
        "project": {
            "name": "Axon BIM Transplant Export",
            "site": {
                "name": "Default Site",
                "building": {
                    "name": "Default Building",
                    "storeys": [
                        _build_storey_json(assemblies, openings, result, scale),
                    ],
                },
            },
        },
        "summary": {
            "wall_count": len(assemblies),
            "opening_count": len(openings),
            "room_count": len(result.placement_map.rooms),
            "pod_count": result.placement_map.placed_room_count,
            "total_panel_count": sum(len(a.panels) for a in assemblies),
            "total_seam_count": sum(len(a.seam_positions_mm) for a in assemblies),
            "spur_score": result.spur_score,
            "coverage_percentage": result.coverage_percentage,
            "waste_percentage": result.waste_percentage,
            "source_path": source_graph.source_path,
            "scale_factor": scale,
        },
    }

    output_path = output_path.with_suffix(".json") if output_path.suffix != ".json" else output_path
    with open(output_path, "w") as f:
        json.dump(model, f, indent=2, default=_json_default)

    logger.info("JSON fallback written to %s", output_path)
    return output_path


def _build_storey_json(
    assemblies: list[WallAssembly],
    openings: list[OpeningAttachment],
    result: PanelizationResult,
    scale: float,
) -> dict[str, Any]:
    """Build the JSON representation of IfcBuildingStorey and its contents."""
    # Group openings by host wall
    openings_by_wall: dict[int, list[OpeningAttachment]] = {}
    for op in openings:
        openings_by_wall.setdefault(op.host_edge_id, []).append(op)

    # Walls
    walls_json: list[dict[str, Any]] = []
    for assembly in assemblies:
        wall_openings = openings_by_wall.get(assembly.edge_id, [])
        walls_json.append(_wall_to_json(assembly, wall_openings))

    # Rooms and pods
    source_graph = result.source_graph.graph
    rooms_json: list[dict[str, Any]] = []
    for room_placement in result.placement_map.rooms:
        room = _find_room(source_graph.rooms, room_placement.room_id)
        if room is None or room.is_exterior:
            continue
        rooms_json.append(_room_to_json(room_placement, room, scale))

    return {
        "name": "Level 1",
        "ifc_class": "IfcBuildingStorey",
        "walls": walls_json,
        "spaces": rooms_json,
    }


def _wall_to_json(
    assembly: WallAssembly,
    wall_openings: list[OpeningAttachment],
) -> dict[str, Any]:
    """Serialize a WallAssembly to JSON (mirrors IfcWallStandardCase)."""
    panels_json = []
    for p in assembly.panels:
        panels_json.append(
            {
                "panel_sku": p.panel_sku,
                "panel_index": p.panel_index,
                "family_name": p.family_name,
                "panel_type": p.panel_type,
                "gauge": p.gauge,
                "stud_depth_inches": p.stud_depth_inches,
                "stud_spacing_inches": p.stud_spacing_inches,
                "height_inches": p.height_inches,
                "cut_length_inches": p.cut_length_inches,
                "fire_rating_hours": p.fire_rating_hours,
                "sheathing_type": p.sheathing_type,
                "insulation_type": p.insulation_type,
                "material_layers": p.material_layers,
            }
        )

    openings_json = []
    for op in wall_openings:
        openings_json.append(
            {
                "ifc_class": "IfcOpeningElement",
                "opening_type": op.opening_type,
                "position_along_wall_mm": op.position_along_wall_mm,
                "width_mm": op.width_mm,
                "height_mm": op.height_mm,
                "sill_height_mm": op.sill_height_mm,
                "void_coords": list(op.void_coords),
                "relationship": "IfcRelVoidsElement",
            }
        )

    # Property set (same data as the IfcPropertySet in native export)
    first_panel = assembly.panels[0] if assembly.panels else None
    pset: dict[str, Any] = {}
    if first_panel is not None:
        pset = {
            "PanelSKUs": ", ".join(p.panel_sku for p in assembly.panels),
            "PanelType": first_panel.panel_type,
            "Gauge": first_panel.gauge,
            "StudDepthInches": first_panel.stud_depth_inches,
            "StudSpacingInches": first_panel.stud_spacing_inches,
            "FireRatingHours": first_panel.fire_rating_hours,
            "FamilyName": first_panel.family_name,
            "PanelCount": len(assembly.panels),
            "SeamCount": len(assembly.seam_positions_mm),
        }

    return {
        "ifc_class": "IfcWallStandardCase",
        "name": f"Wall_{assembly.edge_id}",
        "edge_id": assembly.edge_id,
        "placement": {
            "start": list(assembly.wall_start),
            "end": list(assembly.wall_end),
            "angle_rad": assembly.wall_angle_rad,
        },
        "geometry": {
            "type": "IfcExtrudedAreaSolid",
            "profile": "IfcRectangleProfileDef",
            "length_mm": assembly.wall_length_mm,
            "thickness_mm": assembly.thickness_mm,
            "height_mm": assembly.height_mm,
        },
        "junction_type": assembly.junction_type,
        "seam_positions_mm": assembly.seam_positions_mm,
        "splice_skus": assembly.splice_skus,
        "panels": panels_json,
        "openings": openings_json,
        "property_sets": {"Axon_PanelData": pset} if pset else {},
    }


def _room_to_json(
    room_placement: Any,
    room: Any,
    scale: float,
) -> dict[str, Any]:
    """Serialize a room/pod placement to JSON (mirrors IfcSpace)."""
    space: dict[str, Any] = {
        "ifc_class": "IfcSpace",
        "name": room.label or f"Room_{room.room_id}",
        "room_id": room.room_id,
        "boundary_edges": room.boundary_edges,
        "area_pdf_units_sq": room.area,
        "label": room.label,
    }

    if room_placement.placement is not None:
        p = room_placement.placement
        space["pod_placement"] = {
            "ifc_class": "IfcFurnishingElement",
            "pod_sku": p.pod_sku,
            "position_mm": [
                round(float(p.position[0]) * scale, 2),
                round(float(p.position[1]) * scale, 2),
            ],
            "orientation_deg": p.orientation_deg,
            "clearance_met": p.clearance_met,
            "clearance_margins": p.clearance_margins,
            "property_sets": {
                "Axon_PodData": {
                    "PodSKU": p.pod_sku,
                    "RoomId": room_placement.room_id,
                    "RoomLabel": room_placement.room_label,
                    "OrientationDeg": p.orientation_deg,
                    "ClearanceMet": p.clearance_met,
                },
            },
        }

    return space


# ── Utility helpers ───────────────────────────────────────────────────────


def _find_room(rooms: list, room_id: int) -> Any | None:
    """Find a Room by room_id in the source graph's room list."""
    for room in rooms:
        if room.room_id == room_id:
            return room
    return None


def _json_default(obj: Any) -> Any:
    """JSON serializer for objects not natively serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
