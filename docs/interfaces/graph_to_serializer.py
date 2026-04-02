"""Interface contract: Pipeline → Serializer Agent.

Defines the FinalizedGraph dataclass — the fully validated structural graph
that has passed through all stages (Diffusion → Constraint → Topology → Physics)
and is ready for IFC serialization.

This is the input to Stage 7 (IFC Serialization).

The FinalizedGraph carries:
    - Precise wall junction coordinates (from diffusion + constraint snap)
    - Wall segment edges with thickness (from parallel pair constraint)
    - Opening locations (doors, windows) with type labels
    - Room semantics (IfcSpace assignments)
    - Structural viability certification (from physics validation)

Serialization target:
    IfcWallStandardCase with SweptSolid shape representation
    [MODEL_SPEC.md §Structured Serialization, ARCHITECTURE.md §Stage 7]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class WallType(str, Enum):
    """Semantic wall classification."""

    LOAD_BEARING = "load_bearing"
    """Structural wall carrying vertical loads."""

    PARTITION = "partition"
    """Non-structural interior partition wall."""

    EXTERIOR = "exterior"
    """Exterior envelope wall (may or may not be load-bearing)."""

    SHEAR = "shear"
    """Shear wall resisting lateral forces."""

    CURTAIN = "curtain"
    """Curtain wall (non-structural exterior cladding)."""

    UNKNOWN = "unknown"
    """Classification not determined."""


class OpeningType(str, Enum):
    """Classification of wall openings."""

    DOOR = "door"
    WINDOW = "window"
    PORTAL = "portal"
    """Open passage without a door."""

    ARCHWAY = "archway"
    UNKNOWN = "unknown"


@dataclass
class WallSegment:
    """A single wall segment in the finalized structural graph.

    Maps to one IfcWallStandardCase entity in the IFC output.
    The wall geometry is defined by its start/end junction nodes
    and thickness from the constraint solver's parallel pair analysis.

    Reference: MODEL_SPEC.md §Structured Serialization,
               ARCHITECTURE.md §Stage 7.
    """

    edge_id: int
    """Index of this edge in the FinalizedGraph.edges array."""

    start_node: int
    """Index of the start junction node."""

    end_node: int
    """Index of the end junction node."""

    start_coord: np.ndarray
    """Start junction coordinate, shape (2,) float64, in PDF user units."""

    end_coord: np.ndarray
    """End junction coordinate, shape (2,) float64, in PDF user units."""

    thickness: float
    """Wall thickness in PDF user units, from parallel pair constraint.

    Used for IfcWallStandardCase SweptSolid shape representation:
    the extrusion cross-section width.
    """

    height: float
    """Wall height for 3D extrusion, in length units.

    Default assumed height (e.g., 2.7m / 8.86ft) unless floor-to-floor
    height is detected from the document metadata.
    """

    wall_type: WallType
    """Semantic classification of this wall segment."""

    angle: float
    """Angle of the wall relative to positive x-axis, in radians [0, π)."""

    length: float
    """Euclidean length of the wall segment, in PDF user units."""

    confidence: float
    """Final classification confidence, in [0, 1]."""


@dataclass
class Opening:
    """A wall opening (door, window, portal) attached to a wall segment.

    Maps to IfcOpeningElement with IfcRelVoidsElement relationship
    to the parent IfcWallStandardCase.

    Reference: ARCHITECTURE.md §Stage 7.
    """

    opening_type: OpeningType
    """Classification of this opening."""

    wall_edge_id: int
    """Index of the parent wall segment this opening belongs to."""

    position_along_wall: float
    """Normalized position [0, 1] along the parent wall's length.

    0.0 = at start node, 1.0 = at end node.
    """

    width: float
    """Opening width in PDF user units."""

    height: float
    """Opening height in length units (for 3D representation)."""

    sill_height: float = 0.0
    """Height of the sill above floor level (windows only), in length units."""

    confidence: float = 1.0
    """Detection confidence, in [0, 1]."""


@dataclass
class Room:
    """An enclosed room/space defined by bounding wall segments.

    Maps to IfcSpace with IfcRelSpaceBoundary relationships to
    its bounding walls.

    Reference: ARCHITECTURE.md §Stage 7.
    """

    room_id: int
    """Unique identifier for this room."""

    boundary_edges: list[int]
    """Ordered list of edge indices forming the room boundary polygon.

    Edges listed in counter-clockwise order for interior rooms.
    """

    boundary_nodes: list[int]
    """Ordered list of node indices forming the room boundary polygon."""

    area: float
    """Enclosed area of the room, in PDF user units squared."""

    label: str = ""
    """Semantic label if detected (e.g., 'Kitchen', 'Bedroom', 'Bathroom').

    Derived from cross-modal text parsing in the tokenizer stage.
    Empty string if no label was detected.
    """

    is_exterior: bool = False
    """True if this represents the exterior boundary (unbounded region)."""


@dataclass
class FinalizedGraph:
    """Fully validated structural graph ready for IFC serialization.

    This is the final output of the Axon inference pipeline before
    serialization. It has been:
    1. Generated by the diffusion engine (Stage 3)
    2. Geometrically constrained by the SAT solver (Stage 4)
    3. Topologically validated via persistent homology (Stage 5)
    4. Structurally verified by the PINN/FEA layer (Stage 6)

    The serializer converts this directly to IfcWallStandardCase entities
    with SweptSolid representations, opening attachments, and room semantics.

    Reference: MODEL_SPEC.md §Structured Serialization, ARCHITECTURE.md §Stage 7.
    """

    nodes: np.ndarray
    """Final wall junction coordinates, shape (N, 2) float64.

    In PDF user units (72 units/inch). Post-constraint-snap positions:
    orthogonal angles are exact, parallel pairs are uniform thickness.
    """

    edges: np.ndarray
    """Edge index array, shape (E, 2) int64.

    Each row [i, j] is a directed edge from node i to node j.
    """

    wall_segments: list[WallSegment]
    """Detailed wall segment data, one per edge.

    Contains thickness, type, angle, and other properties needed
    for IFC mapping. Length matches edges.shape[0].
    """

    openings: list[Opening]
    """Detected openings (doors, windows) attached to wall segments."""

    rooms: list[Room]
    """Enclosed rooms/spaces with boundary and semantic data."""

    page_width: float
    """PDF page width in user units."""

    page_height: float
    """PDF page height in user units."""

    page_index: int = 0
    """Zero-based page index within the source PDF."""

    source_path: str = ""
    """File path of the source PDF document."""

    assumed_wall_height: float = 2700.0
    """Default wall height for 3D extrusion, in millimeters.

    2700mm ≈ 8.86ft, standard residential ceiling height.
    """

    structural_viability: str = "unknown"
    """Overall structural viability: 'viable', 'marginal', 'failed', 'unknown'."""

    betti_0: int = 0
    """Final Betti-0 number (connected components) of the validated graph."""

    betti_1: int = 0
    """Final Betti-1 number (enclosed loops/rooms) of the validated graph."""

    coordinate_system: str = "pdf_user_units"
    """Coordinate system identifier. 'pdf_user_units' = 72 units/inch."""

    scale_factor: float = 1.0
    """Scale factor to convert from PDF user units to real-world millimeters.

    Must be determined from scale bar detection or document metadata.
    Default 1.0 means no scale conversion applied.
    """

    metadata: dict[str, object] = field(default_factory=dict)
    """Additional metadata (processing timestamps, model version, etc.)."""
