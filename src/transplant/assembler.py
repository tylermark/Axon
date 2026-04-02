"""BT-002: 3D wall assembly from placed panels.

For each wall in the PanelMap, collects all BIMFamilyMatches and builds a
``WallAssembly`` — the 3D representation of the wall with panel geometry,
seam positions, and splice hardware.

Coordinate conversion: PDF user units -> millimeters using the source
graph's ``scale_factor``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from docs.interfaces.drl_output import PanelizationResult
    from src.transplant.matcher import BIMFamilyMatch

logger = logging.getLogger(__name__)

# Inches to millimeters
_INCHES_TO_MM: float = 25.4


# ── Dataclasses ───────────────────────────────────────────────────────────


@dataclass
class WallAssembly:
    """3D wall assembly built from BIM-matched panels.

    Attributes:
        edge_id: Wall edge identifier from the source graph.
        panels: Ordered list of BIMFamilyMatches along this wall.
        wall_start: Start junction (x, y) in real-world millimeters.
        wall_end: End junction (x, y) in real-world millimeters.
        thickness_mm: Wall thickness in millimeters.
        height_mm: Wall height in millimeters.
        seam_positions_mm: Offsets along the wall axis where panel
            seams (splice joints) occur, in millimeters.
        splice_skus: Connection SKUs for each seam, ordered to match
            ``seam_positions_mm``.
        wall_length_mm: Total wall length in millimeters.
        wall_angle_rad: Angle of the wall relative to positive x-axis.
        junction_type: Junction geometry at the wall's start node
            (T, L, X, or end).
    """

    edge_id: int
    panels: list[BIMFamilyMatch] = field(default_factory=list)
    wall_start: tuple[float, float] = (0.0, 0.0)
    wall_end: tuple[float, float] = (0.0, 0.0)
    thickness_mm: float = 0.0
    height_mm: float = 0.0
    seam_positions_mm: list[float] = field(default_factory=list)
    splice_skus: list[str] = field(default_factory=list)
    wall_length_mm: float = 0.0
    wall_angle_rad: float = 0.0
    junction_type: str = "end"


# ── Public API ────────────────────────────────────────────────────────────


def assemble_walls(
    matches: list[BIMFamilyMatch],
    result: PanelizationResult,
) -> list[WallAssembly]:
    """Build 3D wall assemblies from BIM family matches and DRL output.

    For each wall that has BIMFamilyMatches:
      - Retrieves geometry (start/end coordinates, thickness) from the
        source graph's wall segments.
      - Converts coordinates from PDF user units to millimeters.
      - Computes panel height from the first match's height_inches.
      - Records seam positions and splice SKUs between consecutive panels.
      - Detects junction type from node connectivity.

    Args:
        matches: BIM family matches from :func:`match_bim_families`.
        result: The DRL agent's panelization output (for wall geometry
            and splice SKU data).

    Returns:
        List of ``WallAssembly`` instances, one per panelized wall,
        ordered by ``edge_id``.
    """
    source_graph = result.source_graph.graph
    scale = source_graph.scale_factor

    # Group matches by edge_id
    matches_by_edge: dict[int, list[BIMFamilyMatch]] = defaultdict(list)
    for m in matches:
        matches_by_edge[m.edge_id].append(m)

    # Sort each group by panel_index
    for edge_id in matches_by_edge:
        matches_by_edge[edge_id].sort(key=lambda m: m.panel_index)

    # Build segment lookup: edge_id -> WallSegment
    segment_lookup = {seg.edge_id: seg for seg in source_graph.wall_segments}

    # Build node degree map for junction detection
    node_degree = _compute_node_degrees(source_graph.edges)

    # Build panelization lookup for splice SKU data
    wall_pan_lookup = {wp.edge_id: wp for wp in result.panel_map.walls}

    assemblies: list[WallAssembly] = []

    for edge_id in sorted(matches_by_edge.keys()):
        edge_matches = matches_by_edge[edge_id]
        seg = segment_lookup.get(edge_id)
        if seg is None:
            logger.warning(
                "WallSegment not found for edge_id=%d; skipping assembly",
                edge_id,
            )
            continue

        # Convert coordinates to millimeters
        start_mm = _to_mm(seg.start_coord, scale)
        end_mm = _to_mm(seg.end_coord, scale)
        thickness_mm = seg.thickness * scale
        wall_length_mm = seg.length * scale

        # Panel height: use the BIM match (inches -> mm)
        height_mm = edge_matches[0].height_inches * _INCHES_TO_MM

        # Seam positions: cumulative offsets where panels meet
        seam_positions_mm: list[float] = []
        cumulative_mm = 0.0
        for panel_match in edge_matches[:-1]:
            cumulative_mm += panel_match.cut_length_inches * _INCHES_TO_MM
            seam_positions_mm.append(round(cumulative_mm, 2))

        # Splice SKUs from the DRL wall panelization
        splice_skus: list[str] = []
        wall_pan = wall_pan_lookup.get(edge_id)
        if wall_pan is not None:
            splice_skus = list(wall_pan.splice_connection_skus)

        # Junction type at start node
        junction = _classify_junction(seg.start_node, node_degree)

        assemblies.append(
            WallAssembly(
                edge_id=edge_id,
                panels=edge_matches,
                wall_start=(round(start_mm[0], 2), round(start_mm[1], 2)),
                wall_end=(round(end_mm[0], 2), round(end_mm[1], 2)),
                thickness_mm=round(thickness_mm, 2),
                height_mm=round(height_mm, 2),
                seam_positions_mm=seam_positions_mm,
                splice_skus=splice_skus,
                wall_length_mm=round(wall_length_mm, 2),
                wall_angle_rad=seg.angle,
                junction_type=junction,
            )
        )

    logger.info("Assembled %d wall assemblies", len(assemblies))
    return assemblies


# ── Helpers ───────────────────────────────────────────────────────────────


def _to_mm(coord: np.ndarray, scale_factor: float) -> tuple[float, float]:
    """Convert a 2D coordinate from PDF user units to millimeters.

    Args:
        coord: Shape (2,) array in PDF user units.
        scale_factor: Conversion factor (PDF units -> mm).

    Returns:
        Tuple (x_mm, y_mm).
    """
    return (float(coord[0]) * scale_factor, float(coord[1]) * scale_factor)


def _compute_node_degrees(edges: np.ndarray) -> dict[int, int]:
    """Compute the degree (number of incident edges) for each node.

    Args:
        edges: Shape (E, 2) edge index array.

    Returns:
        Dict mapping node index to degree count.
    """
    degrees: dict[int, int] = defaultdict(int)
    for i in range(edges.shape[0]):
        degrees[int(edges[i, 0])] += 1
        degrees[int(edges[i, 1])] += 1
    return dict(degrees)


def _classify_junction(node_id: int, node_degree: dict[int, int]) -> str:
    """Classify a wall junction by the number of incident edges.

    Args:
        node_id: Node index.
        node_degree: Pre-computed node degree map.

    Returns:
        Junction type string: "end", "L", "T", or "X".
    """
    degree = node_degree.get(node_id, 1)
    if degree <= 1:
        return "end"
    if degree == 2:
        return "L"
    if degree == 3:
        return "T"
    return "X"
