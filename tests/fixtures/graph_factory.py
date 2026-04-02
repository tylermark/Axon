"""Factory for creating synthetic RawGraph instances for testing.

Provides lightweight graph construction without PDF parsing,
for use in tokenizer and downstream module tests.
"""

from __future__ import annotations

import numpy as np

from docs.interfaces.parser_to_tokenizer import (
    OperatorType,
    PathMetadata,
    RawGraph,
)


def create_synthetic_raw_graph(
    num_edges: int = 10,
    page_width: float = 612.0,
    page_height: float = 792.0,
    seed: int | None = 42,
) -> RawGraph:
    """Create a synthetic RawGraph for testing.

    Generates a chain graph (N+1 nodes, N edges) with random node positions
    and uniform stroke properties. All edges are LINETO with solid black strokes.

    Args:
        num_edges: Number of edges (tokens) in the graph.
        page_width: Page width in PDF user units.
        page_height: Page height in PDF user units.
        seed: Random seed for reproducibility. None for non-deterministic.

    Returns:
        A valid RawGraph instance.
    """
    rng = np.random.default_rng(seed)

    num_nodes = num_edges + 1
    nodes = np.column_stack(
        [
            rng.uniform(0, page_width, num_nodes),
            rng.uniform(0, page_height, num_nodes),
        ]
    ).astype(np.float64)

    edges = np.column_stack(
        [
            np.arange(num_edges, dtype=np.int64),
            np.arange(1, num_edges + 1, dtype=np.int64),
        ]
    )

    operator_types = [OperatorType.LINETO] * num_edges
    stroke_widths = np.full(num_edges, 1.5, dtype=np.float64)

    stroke_colors = np.zeros((num_edges, 4), dtype=np.float64)
    stroke_colors[:, 3] = 1.0  # opaque black

    dash_patterns: list[tuple[list[float], float]] = [([], 0.0)] * num_edges

    path_metadata = [
        PathMetadata(
            stroke_width=1.5,
            stroke_color=np.array([0, 0, 0, 1], dtype=np.float64),
            fill_color=None,
            dash_pattern=([], 0.0),
            ctm=np.eye(3, dtype=np.float64),
        )
    ]

    edge_to_path = np.zeros(num_edges, dtype=np.int64)
    bezier_controls: list[None] = [None] * num_edges
    confidence_wall = rng.uniform(0.3, 1.0, num_edges).astype(np.float64)

    return RawGraph(
        nodes=nodes,
        edges=edges,
        operator_types=operator_types,
        stroke_widths=stroke_widths,
        stroke_colors=stroke_colors,
        fill_colors=None,
        dash_patterns=dash_patterns,
        path_metadata=path_metadata,
        edge_to_path=edge_to_path,
        bezier_controls=bezier_controls,
        confidence_wall=confidence_wall,
        page_width=page_width,
        page_height=page_height,
    )
