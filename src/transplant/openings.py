"""BT-003: Opening attachment (IfcRelVoidsElement preparation).

For each opening in the source graph, finds the host WallAssembly and
computes the void geometry needed for ``IfcOpeningElement`` and
``IfcRelVoidsElement`` in the IFC export.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docs.interfaces.drl_output import PanelizationResult
    from src.transplant.assembler import WallAssembly

logger = logging.getLogger(__name__)

# Inches to millimeters
_INCHES_TO_MM: float = 25.4


# ── Dataclass ─────────────────────────────────────────────────────────────


@dataclass
class OpeningAttachment:
    """Void geometry for an opening, ready for IfcRelVoidsElement.

    Attributes:
        opening_type: Classification (door, window, portal, archway).
        host_edge_id: Edge ID of the host wall assembly.
        position_along_wall_mm: Offset from wall start to opening center,
            in millimeters.
        width_mm: Opening width in millimeters.
        height_mm: Opening height in millimeters.
        sill_height_mm: Sill height above floor (windows), in millimeters.
        void_coords: (x, y, z_bottom, z_top) in model coordinates (mm).
            x,y is the center of the void on the wall plane.
    """

    opening_type: str
    host_edge_id: int
    position_along_wall_mm: float
    width_mm: float
    height_mm: float
    sill_height_mm: float
    void_coords: tuple[float, float, float, float]


# ── Public API ────────────────────────────────────────────────────────────


def attach_openings(
    assemblies: list[WallAssembly],
    result: PanelizationResult,
) -> list[OpeningAttachment]:
    """Attach openings to host wall assemblies and compute void geometry.

    For each ``Opening`` in the source graph:
      - Finds the host ``WallAssembly`` by matching ``wall_edge_id``.
      - Converts the normalized position_along_wall [0, 1] to an absolute
        offset in millimeters.
      - Converts width, height, and sill_height from the source graph's
        coordinate system to millimeters.
      - Computes 3D void coordinates (x, y center on the wall plane,
        z_bottom and z_top).

    Args:
        assemblies: Wall assemblies from :func:`assemble_walls`.
        result: The DRL agent's panelization output.

    Returns:
        List of ``OpeningAttachment`` instances, one per opening in the
        source graph that has a host wall assembly.
    """
    source_graph = result.source_graph.graph
    scale = source_graph.scale_factor

    # Build assembly lookup
    assembly_lookup: dict[int, WallAssembly] = {a.edge_id: a for a in assemblies}

    attachments: list[OpeningAttachment] = []

    for opening in source_graph.openings:
        assembly = assembly_lookup.get(opening.wall_edge_id)
        if assembly is None:
            logger.warning(
                "No wall assembly for opening on edge_id=%d (type=%s); "
                "opening will not appear in IFC output",
                opening.wall_edge_id,
                opening.opening_type.value,
            )
            continue

        # Position along wall: normalized [0,1] * wall_length_mm
        position_mm = opening.position_along_wall * assembly.wall_length_mm

        # Opening width in mm (width is in PDF user units)
        width_mm = opening.width * scale

        # Opening height in mm (height is in length units, same as wall height)
        # The Opening.height field uses the same unit system as WallSegment.height.
        # If the source graph's scale_factor converts PDF units -> mm, heights are
        # already in mm in the FinalizedGraph (assumed_wall_height is in mm).
        # However, Opening.height is documented as "length units" — we apply
        # scale only if scale != 1.0, otherwise treat as mm directly.
        height_mm = opening.height * scale if scale != 1.0 else opening.height

        # Sill height (same unit treatment as height)
        sill_mm = opening.sill_height * scale if scale != 1.0 else opening.sill_height

        # Compute void center in model coordinates (mm)
        # Interpolate along the wall axis
        t = opening.position_along_wall
        cx = assembly.wall_start[0] + t * (assembly.wall_end[0] - assembly.wall_start[0])
        cy = assembly.wall_start[1] + t * (assembly.wall_end[1] - assembly.wall_start[1])

        z_bottom = sill_mm
        z_top = sill_mm + height_mm

        attachments.append(
            OpeningAttachment(
                opening_type=opening.opening_type.value,
                host_edge_id=opening.wall_edge_id,
                position_along_wall_mm=round(position_mm, 2),
                width_mm=round(width_mm, 2),
                height_mm=round(height_mm, 2),
                sill_height_mm=round(sill_mm, 2),
                void_coords=(
                    round(cx, 2),
                    round(cy, 2),
                    round(z_bottom, 2),
                    round(z_top, 2),
                ),
            )
        )

    logger.info(
        "Attached %d openings to wall assemblies (%d openings in source graph)",
        len(attachments),
        len(source_graph.openings),
    )
    return attachments
