"""PostScript operator registry for PDF content stream parsing.

Maps PDF drawing operators to typed Python representations with metadata
for parameter counts, categories, and behavioral flags. This is the
foundational lookup table used by the content stream extractor.

Reference: PDF Reference 1.7 §8.5 (Path Construction and Painting),
           MODEL_SPEC.md Table 1, ARCHITECTURE.md §Stage 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class OperatorType(StrEnum):
    """PDF path operator types mapped to their PostScript equivalents.

    These are the geometric primitive types that appear in RawGraph edges.
    Must stay compatible with docs/interfaces/parser_to_tokenizer.py.

    Reference: MODEL_SPEC.md Table 1.
    """

    MOVETO = "moveto"
    LINETO = "lineto"
    CURVETO = "curveto"
    CLOSEPATH = "closepath"


class OperatorCategory(StrEnum):
    """Classification of PDF content stream operators by function."""

    PATH_CONSTRUCTION = "path_construction"
    PATH_PAINTING = "path_painting"
    GRAPHICS_STATE = "graphics_state"
    COLOR = "color"
    CLIPPING = "clipping"


@dataclass(frozen=True, slots=True)
class OperatorInfo:
    """Metadata for a single PDF content stream operator.

    Args:
        name: Human-readable operator name.
        pdf_operator: The raw operator string as it appears in a PDF content stream.
        category: Functional classification of this operator.
        param_count: Number of operands consumed from the stack, or None if variable.
        description: Brief description of what the operator does.
        modifies_path: True if this operator adds to or closes the current path.
        modifies_state: True if this operator changes the graphics state.
    """

    name: str
    pdf_operator: str
    category: OperatorCategory
    param_count: int | None
    description: str
    modifies_path: bool
    modifies_state: bool


# ---------------------------------------------------------------------------
# Operator Registry
# ---------------------------------------------------------------------------

OPERATOR_REGISTRY: dict[str, OperatorInfo] = {
    # -- Path Construction --------------------------------------------------
    "m": OperatorInfo(
        name="moveto",
        pdf_operator="m",
        category=OperatorCategory.PATH_CONSTRUCTION,
        param_count=2,
        description="Begin new subpath at (x, y).",
        modifies_path=True,
        modifies_state=False,
    ),
    "l": OperatorInfo(
        name="lineto",
        pdf_operator="l",
        category=OperatorCategory.PATH_CONSTRUCTION,
        param_count=2,
        description="Append straight line segment to (x, y).",
        modifies_path=True,
        modifies_state=False,
    ),
    "c": OperatorInfo(
        name="curveto",
        pdf_operator="c",
        category=OperatorCategory.PATH_CONSTRUCTION,
        param_count=6,
        description="Append cubic Bézier curve with control points (x1,y1), (x2,y2) to (x3,y3).",
        modifies_path=True,
        modifies_state=False,
    ),
    "v": OperatorInfo(
        name="curveto_v",
        pdf_operator="v",
        category=OperatorCategory.PATH_CONSTRUCTION,
        param_count=4,
        description="Cubic Bézier; first control point equals current point. Params: x2,y2,x3,y3.",
        modifies_path=True,
        modifies_state=False,
    ),
    "y": OperatorInfo(
        name="curveto_y",
        pdf_operator="y",
        category=OperatorCategory.PATH_CONSTRUCTION,
        param_count=4,
        description="Cubic Bézier; last control point equals endpoint. Params: x1,y1,x3,y3.",
        modifies_path=True,
        modifies_state=False,
    ),
    "h": OperatorInfo(
        name="closepath",
        pdf_operator="h",
        category=OperatorCategory.PATH_CONSTRUCTION,
        param_count=0,
        description="Close current subpath with a straight line to the starting point.",
        modifies_path=True,
        modifies_state=False,
    ),
    "re": OperatorInfo(
        name="rectangle",
        pdf_operator="re",
        category=OperatorCategory.PATH_CONSTRUCTION,
        param_count=4,
        description="Append rectangle path: x, y, width, height.",
        modifies_path=True,
        modifies_state=False,
    ),
    # -- Path Painting ------------------------------------------------------
    "S": OperatorInfo(
        name="stroke",
        pdf_operator="S",
        category=OperatorCategory.PATH_PAINTING,
        param_count=0,
        description="Stroke the current path.",
        modifies_path=False,
        modifies_state=False,
    ),
    "s": OperatorInfo(
        name="close_and_stroke",
        pdf_operator="s",
        category=OperatorCategory.PATH_PAINTING,
        param_count=0,
        description="Close and stroke the current path (equivalent to h S).",
        modifies_path=False,
        modifies_state=False,
    ),
    "f": OperatorInfo(
        name="fill_nonzero",
        pdf_operator="f",
        category=OperatorCategory.PATH_PAINTING,
        param_count=0,
        description="Fill the path using the nonzero winding number rule.",
        modifies_path=False,
        modifies_state=False,
    ),
    "F": OperatorInfo(
        name="fill_nonzero_compat",
        pdf_operator="F",
        category=OperatorCategory.PATH_PAINTING,
        param_count=0,
        description="Fill nonzero (PDF 1.0 compatibility; equivalent to f).",
        modifies_path=False,
        modifies_state=False,
    ),
    "f*": OperatorInfo(
        name="fill_evenodd",
        pdf_operator="f*",
        category=OperatorCategory.PATH_PAINTING,
        param_count=0,
        description="Fill the path using the even-odd rule.",
        modifies_path=False,
        modifies_state=False,
    ),
    "B": OperatorInfo(
        name="fill_stroke_nonzero",
        pdf_operator="B",
        category=OperatorCategory.PATH_PAINTING,
        param_count=0,
        description="Fill (nonzero) and stroke the current path.",
        modifies_path=False,
        modifies_state=False,
    ),
    "B*": OperatorInfo(
        name="fill_stroke_evenodd",
        pdf_operator="B*",
        category=OperatorCategory.PATH_PAINTING,
        param_count=0,
        description="Fill (even-odd) and stroke the current path.",
        modifies_path=False,
        modifies_state=False,
    ),
    "b": OperatorInfo(
        name="close_fill_stroke_nonzero",
        pdf_operator="b",
        category=OperatorCategory.PATH_PAINTING,
        param_count=0,
        description="Close, fill (nonzero), and stroke the path (equivalent to h B).",
        modifies_path=False,
        modifies_state=False,
    ),
    "b*": OperatorInfo(
        name="close_fill_stroke_evenodd",
        pdf_operator="b*",
        category=OperatorCategory.PATH_PAINTING,
        param_count=0,
        description="Close, fill (even-odd), and stroke the path (equivalent to h B*).",
        modifies_path=False,
        modifies_state=False,
    ),
    "n": OperatorInfo(
        name="endpath",
        pdf_operator="n",
        category=OperatorCategory.PATH_PAINTING,
        param_count=0,
        description="End path without filling or stroking (used after clipping).",
        modifies_path=False,
        modifies_state=False,
    ),
    # -- Graphics State -----------------------------------------------------
    "q": OperatorInfo(
        name="gsave",
        pdf_operator="q",
        category=OperatorCategory.GRAPHICS_STATE,
        param_count=0,
        description="Push current graphics state onto the stack.",
        modifies_path=False,
        modifies_state=True,
    ),
    "Q": OperatorInfo(
        name="grestore",
        pdf_operator="Q",
        category=OperatorCategory.GRAPHICS_STATE,
        param_count=0,
        description="Pop graphics state from the stack.",
        modifies_path=False,
        modifies_state=True,
    ),
    "cm": OperatorInfo(
        name="concat_matrix",
        pdf_operator="cm",
        category=OperatorCategory.GRAPHICS_STATE,
        param_count=6,
        description="Concatenate matrix [a b c d e f] to the CTM.",
        modifies_path=False,
        modifies_state=True,
    ),
    "w": OperatorInfo(
        name="set_line_width",
        pdf_operator="w",
        category=OperatorCategory.GRAPHICS_STATE,
        param_count=1,
        description="Set line width.",
        modifies_path=False,
        modifies_state=True,
    ),
    "J": OperatorInfo(
        name="set_line_cap",
        pdf_operator="J",
        category=OperatorCategory.GRAPHICS_STATE,
        param_count=1,
        description="Set line cap style (0=butt, 1=round, 2=square).",
        modifies_path=False,
        modifies_state=True,
    ),
    "j": OperatorInfo(
        name="set_line_join",
        pdf_operator="j",
        category=OperatorCategory.GRAPHICS_STATE,
        param_count=1,
        description="Set line join style (0=miter, 1=round, 2=bevel).",
        modifies_path=False,
        modifies_state=True,
    ),
    "M": OperatorInfo(
        name="set_miter_limit",
        pdf_operator="M",
        category=OperatorCategory.GRAPHICS_STATE,
        param_count=1,
        description="Set miter limit for line joins.",
        modifies_path=False,
        modifies_state=True,
    ),
    "d": OperatorInfo(
        name="set_dash",
        pdf_operator="d",
        category=OperatorCategory.GRAPHICS_STATE,
        param_count=2,
        description="Set dash pattern: [dash_array] dash_phase.",
        modifies_path=False,
        modifies_state=True,
    ),
    # -- Color --------------------------------------------------------------
    "CS": OperatorInfo(
        name="set_colorspace_stroke",
        pdf_operator="CS",
        category=OperatorCategory.COLOR,
        param_count=1,
        description="Set color space for stroking operations.",
        modifies_path=False,
        modifies_state=True,
    ),
    "cs": OperatorInfo(
        name="set_colorspace_nonstroke",
        pdf_operator="cs",
        category=OperatorCategory.COLOR,
        param_count=1,
        description="Set color space for non-stroking operations.",
        modifies_path=False,
        modifies_state=True,
    ),
    "SC": OperatorInfo(
        name="set_color_stroke",
        pdf_operator="SC",
        category=OperatorCategory.COLOR,
        param_count=None,
        description="Set color for stroking operations (parameter count depends on color space).",
        modifies_path=False,
        modifies_state=True,
    ),
    "sc": OperatorInfo(
        name="set_color_nonstroke",
        pdf_operator="sc",
        category=OperatorCategory.COLOR,
        param_count=None,
        description="Set color for non-stroking operations (parameter count depends on color space).",
        modifies_path=False,
        modifies_state=True,
    ),
    "SCN": OperatorInfo(
        name="set_color_stroke_extended",
        pdf_operator="SCN",
        category=OperatorCategory.COLOR,
        param_count=None,
        description="Set color for stroking (extended; may include pattern name).",
        modifies_path=False,
        modifies_state=True,
    ),
    "scn": OperatorInfo(
        name="set_color_nonstroke_extended",
        pdf_operator="scn",
        category=OperatorCategory.COLOR,
        param_count=None,
        description="Set color for non-stroking (extended; may include pattern name).",
        modifies_path=False,
        modifies_state=True,
    ),
    "G": OperatorInfo(
        name="set_gray_stroke",
        pdf_operator="G",
        category=OperatorCategory.COLOR,
        param_count=1,
        description="Set gray level for stroking (0=black, 1=white).",
        modifies_path=False,
        modifies_state=True,
    ),
    "g": OperatorInfo(
        name="set_gray_nonstroke",
        pdf_operator="g",
        category=OperatorCategory.COLOR,
        param_count=1,
        description="Set gray level for non-stroking (0=black, 1=white).",
        modifies_path=False,
        modifies_state=True,
    ),
    "RG": OperatorInfo(
        name="set_rgb_stroke",
        pdf_operator="RG",
        category=OperatorCategory.COLOR,
        param_count=3,
        description="Set RGB color for stroking operations.",
        modifies_path=False,
        modifies_state=True,
    ),
    "rg": OperatorInfo(
        name="set_rgb_nonstroke",
        pdf_operator="rg",
        category=OperatorCategory.COLOR,
        param_count=3,
        description="Set RGB color for non-stroking operations.",
        modifies_path=False,
        modifies_state=True,
    ),
    "K": OperatorInfo(
        name="set_cmyk_stroke",
        pdf_operator="K",
        category=OperatorCategory.COLOR,
        param_count=4,
        description="Set CMYK color for stroking operations.",
        modifies_path=False,
        modifies_state=True,
    ),
    "k": OperatorInfo(
        name="set_cmyk_nonstroke",
        pdf_operator="k",
        category=OperatorCategory.COLOR,
        param_count=4,
        description="Set CMYK color for non-stroking operations.",
        modifies_path=False,
        modifies_state=True,
    ),
    # -- Clipping -----------------------------------------------------------
    "W": OperatorInfo(
        name="clip_nonzero",
        pdf_operator="W",
        category=OperatorCategory.CLIPPING,
        param_count=0,
        description="Intersect clipping path using nonzero winding number rule.",
        modifies_path=False,
        modifies_state=True,
    ),
    "W*": OperatorInfo(
        name="clip_evenodd",
        pdf_operator="W*",
        category=OperatorCategory.CLIPPING,
        param_count=0,
        description="Intersect clipping path using even-odd rule.",
        modifies_path=False,
        modifies_state=True,
    ),
}

# Pre-computed category sets for fast lookup.
_PATH_OPERATORS: frozenset[str] = frozenset(
    op
    for op, info in OPERATOR_REGISTRY.items()
    if info.category == OperatorCategory.PATH_CONSTRUCTION
)
_PAINT_OPERATORS: frozenset[str] = frozenset(
    op for op, info in OPERATOR_REGISTRY.items() if info.category == OperatorCategory.PATH_PAINTING
)
_STATE_OPERATORS: frozenset[str] = frozenset(
    op
    for op, info in OPERATOR_REGISTRY.items()
    if info.category in (OperatorCategory.GRAPHICS_STATE, OperatorCategory.COLOR)
)

# Mapping from path-construction operators to their OperatorType output.
PATH_OP_TO_TYPE: dict[str, OperatorType] = {
    "m": OperatorType.MOVETO,
    "l": OperatorType.LINETO,
    "c": OperatorType.CURVETO,
    "v": OperatorType.CURVETO,
    "y": OperatorType.CURVETO,
    "h": OperatorType.CLOSEPATH,
    "re": OperatorType.LINETO,  # rectangle decomposes to line segments
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def is_path_operator(op: str) -> bool:
    """Return True if *op* is a path construction operator (m, l, c, v, y, h, re).

    Args:
        op: PDF operator string.
    """
    return op in _PATH_OPERATORS


def is_paint_operator(op: str) -> bool:
    """Return True if *op* is a path painting operator (S, s, f, B, n, etc.).

    Args:
        op: PDF operator string.
    """
    return op in _PAINT_OPERATORS


def is_state_operator(op: str) -> bool:
    """Return True if *op* modifies graphics state or color.

    Args:
        op: PDF operator string.
    """
    return op in _STATE_OPERATORS


def get_operator_info(op: str) -> OperatorInfo | None:
    """Look up metadata for a PDF operator.

    Args:
        op: PDF operator string (e.g. ``"m"``, ``"cm"``, ``"f*"``).

    Returns:
        The corresponding ``OperatorInfo``, or ``None`` if the operator is
        not in the registry.
    """
    return OPERATOR_REGISTRY.get(op)
