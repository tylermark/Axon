"""BIM Transplant module — KG-driven 3D model assembly and IFC export.

Converts the 2D DRL-optimized layout (panels placed on walls, pods placed in
rooms) into a full 3D BIM model serialized as IFC4.  Each 2D panel slot is
matched to its exact 3D BIM family via deterministic KG lookup.

Public API
----------
- :func:`match_bim_families` — BT-001: KG -> BIM family matching
- :func:`assemble_walls` — BT-002: 3D wall assembly from placed panels
- :func:`attach_openings` — BT-003: Opening attachment (IfcRelVoidsElement)
- :func:`export_ifc` — BT-004: IFC4 serialization (ISO 16739-1:2024)
"""

from src.transplant.assembler import WallAssembly, assemble_walls
from src.transplant.ifc_export import export_ifc
from src.transplant.matcher import BIMFamilyMatch, match_bim_families
from src.transplant.openings import OpeningAttachment, attach_openings

__all__ = [
    "BIMFamilyMatch",
    "OpeningAttachment",
    "WallAssembly",
    "assemble_walls",
    "attach_openings",
    "export_ifc",
    "match_bim_families",
]
