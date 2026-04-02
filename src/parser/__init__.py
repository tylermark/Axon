"""PDF vector parser — Stage 1 of the Axon pipeline.

Extracts geometric primitives from PDF content streams and constructs
the raw spatial graph G₀.
"""

from src.parser.extractor import (
    ExtractedPath,
    GraphicsState,
    GraphicsStateStack,
    PathSegment,
    SubPath,
    extract_paths,
    extract_paths_from_pdf,
)
from src.parser.filters import (
    apply_filters,
    compute_wall_confidence,
)
from src.parser.graph_builder import (
    build_raw_graph,
    deduplicate_vertices,
    sample_bezier,
)
from src.parser.operators import (
    OPERATOR_REGISTRY,
    PATH_OP_TO_TYPE,
    OperatorCategory,
    OperatorInfo,
    OperatorType,
    get_operator_info,
    is_paint_operator,
    is_path_operator,
    is_state_operator,
)

__all__ = [
    "OPERATOR_REGISTRY",
    "PATH_OP_TO_TYPE",
    "ExtractedPath",
    "GraphicsState",
    "GraphicsStateStack",
    "OperatorCategory",
    "OperatorInfo",
    "OperatorType",
    "PathSegment",
    "SubPath",
    "apply_filters",
    "build_raw_graph",
    "compute_wall_confidence",
    "deduplicate_vertices",
    "extract_paths",
    "extract_paths_from_pdf",
    "get_operator_info",
    "is_paint_operator",
    "is_path_operator",
    "is_state_operator",
    "sample_bezier",
]
