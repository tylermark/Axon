"""Vector token embedding and 2D positional encoding for Stage 2 tokenization.

Converts raw path segments from the Parser's RawGraph into d_model-dimensional
embeddings with learned 2D positional encodings, ready for cross-attention
fusion with raster features.

Tasks: T-001 (VectorTokenEmbedding), T-002 (LearnedPositionalEncoding2D).
Reference: ARCHITECTURE.md §Stage 2, MODEL_SPEC.md §Cross-Modal Feature Alignment.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from docs.interfaces.parser_to_tokenizer import RawGraph
    from src.pipeline.config import TokenizerConfig

# Operator type string values → integer indices for embedding lookup.
# Uses string values for compatibility with both the interface and parser enums.
_OP_STR_TO_IDX: dict[str, int] = {
    "moveto": 0,
    "lineto": 1,
    "curveto": 2,
    "closepath": 3,
}
NUM_OPERATOR_TYPES = len(_OP_STR_TO_IDX)

# Default number of hash buckets for dash pattern embedding.
# Bucket 0 is reserved for solid lines.
DEFAULT_DASH_BUCKETS = 64


def _hash_dash_pattern(
    dash_array: list[float],
    phase: float,
    n_buckets: int = DEFAULT_DASH_BUCKETS,
) -> int:
    """Hash a dash pattern to an integer bucket.

    Solid lines (empty dash_array) always map to bucket 0.

    Args:
        dash_array: Dash segment lengths.
        phase: Dash phase offset.
        n_buckets: Number of hash buckets.

    Returns:
        Integer bucket index in [0, n_buckets).
    """
    if not dash_array:
        return 0
    key = (tuple(round(d, 2) for d in dash_array), round(phase, 2))
    return (hash(key) % (n_buckets - 1)) + 1


# ---------------------------------------------------------------------------
# T-001: Vector Token Embedding
# ---------------------------------------------------------------------------


class VectorTokenEmbedding(nn.Module):
    """Maps raw path segment features into d_model-dimensional token embeddings.

    Combines three sub-embeddings:
    - Operator type: learned categorical embedding (4 types).
    - Continuous features: linear projection of coordinates, stroke width,
      color RGB, and wall confidence (9 scalar inputs).
    - Dash pattern: learned categorical embedding (hash buckets).

    Sub-embeddings are concatenated and projected to d_model, followed by
    layer normalization and dropout.

    Reference: ARCHITECTURE.md §Stage 2.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_operator: int = 32,
        d_dash: int = 32,
        n_dash_buckets: int = DEFAULT_DASH_BUCKETS,
        dropout: float = 0.1,
    ) -> None:
        """Initialize vector token embedding layers.

        Args:
            d_model: Output embedding dimension.
            d_operator: Dimension for operator type embedding.
            d_dash: Dimension for dash pattern embedding.
            n_dash_buckets: Number of hash buckets for dash patterns.
            dropout: Dropout rate after final projection.
        """
        super().__init__()
        self.d_model = d_model
        self.n_dash_buckets = n_dash_buckets

        # Sub-embedding layers
        self.operator_embed = nn.Embedding(NUM_OPERATOR_TYPES, d_operator)
        self.dash_embed = nn.Embedding(n_dash_buckets, d_dash)

        # Continuous features: x1, y1, x2, y2, stroke_width, R, G, B, confidence = 9
        n_continuous = 9
        d_continuous = d_model - d_operator - d_dash
        self.continuous_proj = nn.Linear(n_continuous, d_continuous)

        # Fuse sub-embeddings → d_model
        self.output_proj = nn.Linear(d_operator + d_continuous + d_dash, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, raw_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert raw features to token embeddings.

        Args:
            raw_features: Dict with keys:
                'operator_type': (B, N) int64
                'coordinates': (B, N, 4) float32 [x1, y1, x2, y2] normalized
                'stroke_width': (B, N) float32 normalized
                'dash_hash': (B, N) int64
                'color_rgb': (B, N, 3) float32
                'confidence_wall': (B, N) float32

        Returns:
            Token embeddings (B, N, d_model) float32.
        """
        op_emb = self.operator_embed(raw_features["operator_type"])  # (B, N, d_op)
        dash_emb = self.dash_embed(raw_features["dash_hash"])  # (B, N, d_dash)

        continuous = torch.cat(
            [
                raw_features["coordinates"],  # (B, N, 4)
                raw_features["stroke_width"].unsqueeze(-1),  # (B, N, 1)
                raw_features["color_rgb"],  # (B, N, 3)
                raw_features["confidence_wall"].unsqueeze(-1),  # (B, N, 1)
            ],
            dim=-1,
        )  # (B, N, 9)
        cont_emb = self.continuous_proj(continuous)  # (B, N, d_continuous)

        combined = torch.cat([op_emb, cont_emb, dash_emb], dim=-1)
        output = self.output_proj(combined)
        output = self.layer_norm(output)
        return self.dropout(output)


# ---------------------------------------------------------------------------
# T-002: Learned 2D Positional Encoding
# ---------------------------------------------------------------------------


class LearnedPositionalEncoding2D(nn.Module):
    """Learned 2D positional encoding based on normalized page coordinates.

    Maps the (x, y) midpoint of each token's segment to a d_model-dimensional
    encoding via a 2-layer MLP with GELU activation. Captures spatial layout
    of floor plan elements better than 1D sequential positional encodings.

    Reference: ARCHITECTURE.md §Stage 2.
    """

    def __init__(self, d_model: int = 256, dropout: float = 0.1) -> None:
        """Initialize 2D positional encoding MLP.

        Args:
            d_model: Output encoding dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Generate 2D positional encodings from token coordinates.

        Args:
            coordinates: (B, N, 4) float32 [x1, y1, x2, y2] normalized to [0, 1].

        Returns:
            Positional encodings (B, N, d_model) float32.
        """
        midpoints = torch.stack(
            [
                (coordinates[..., 0] + coordinates[..., 2]) / 2.0,
                (coordinates[..., 1] + coordinates[..., 3]) / 2.0,
            ],
            dim=-1,
        )  # (B, N, 2)
        return self.mlp(midpoints)


# ---------------------------------------------------------------------------
# Combined Module
# ---------------------------------------------------------------------------


class VectorTokenizer(nn.Module):
    """Full vector tokenization: raw features → embedded + positioned tokens.

    Combines VectorTokenEmbedding and LearnedPositionalEncoding2D. The forward
    pass adds token embeddings and positional encodings element-wise.

    Reference: ARCHITECTURE.md §Stage 2.
    """

    def __init__(self, config: TokenizerConfig) -> None:
        """Initialize vector tokenizer from pipeline config.

        Args:
            config: Tokenizer configuration providing d_model and dropout.
        """
        super().__init__()
        self.config = config
        self.embedding = VectorTokenEmbedding(
            d_model=config.d_model,
            dropout=config.dropout,
        )
        self.positional_encoding = LearnedPositionalEncoding2D(
            d_model=config.d_model,
            dropout=config.dropout,
        )

    def forward(self, raw_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute token embeddings with positional encodings.

        Args:
            raw_features: Feature dict from graph_to_token_features / collate_graphs.

        Returns:
            Token representations (B, N, d_model) float32, where each token
            embedding is the sum of its feature embedding and positional encoding.
        """
        token_embeds = self.embedding(raw_features)
        pos_encodings = self.positional_encoding(raw_features["coordinates"])
        return token_embeds + pos_encodings


# ---------------------------------------------------------------------------
# RawGraph → Tensor Conversion
# ---------------------------------------------------------------------------


def graph_to_token_features(
    graph: RawGraph,
    max_tokens: int | None = None,
) -> dict[str, torch.Tensor]:
    """Convert a RawGraph to token feature tensors.

    Each edge in the graph becomes one token. Coordinates are normalized to
    [0, 1] by page dimensions, stroke width by page diagonal. Dash patterns
    are hashed to integer buckets.

    Args:
        graph: Raw spatial graph from the parser.
        max_tokens: If specified, pad or truncate to this sequence length.

    Returns:
        Dict with keys compatible with VectorTokenEmbedding.forward(), plus:
        - 'attention_mask': (N,) bool — True for valid tokens.
        - 'raw_coordinates': (N, 4) float64 — original PDF-unit coordinates.
        Tensors are unbatched (no leading batch dimension).
    """
    num_edges = len(graph.edges)
    seq_len = max_tokens if max_tokens is not None else num_edges
    n_valid = min(num_edges, seq_len)

    # Page dimensions for normalization
    page_w = max(graph.page_width, 1e-6)
    page_h = max(graph.page_height, 1e-6)
    page_diag = math.sqrt(page_w**2 + page_h**2)

    # Edge start/end coordinates
    if n_valid > 0:
        edge_slice = graph.edges[:n_valid]
        starts = graph.nodes[edge_slice[:, 0]]  # (n_valid, 2)
        ends = graph.nodes[edge_slice[:, 1]]  # (n_valid, 2)
        raw_coords = np.concatenate([starts, ends], axis=1)  # (n_valid, 4)
    else:
        raw_coords = np.empty((0, 4), dtype=np.float64)

    # Normalized coordinates [0, 1]
    norm_coords = np.zeros((seq_len, 4), dtype=np.float32)
    if n_valid > 0:
        norm_coords[:n_valid, 0] = raw_coords[:, 0] / page_w
        norm_coords[:n_valid, 1] = raw_coords[:, 1] / page_h
        norm_coords[:n_valid, 2] = raw_coords[:, 2] / page_w
        norm_coords[:n_valid, 3] = raw_coords[:, 3] / page_h

    # Operator type → integer index
    op_indices = np.zeros(seq_len, dtype=np.int64)
    for i in range(n_valid):
        op_indices[i] = _OP_STR_TO_IDX.get(graph.operator_types[i].value, 0)

    # Stroke width normalized by page diagonal
    stroke_width = np.zeros(seq_len, dtype=np.float32)
    if n_valid > 0:
        stroke_width[:n_valid] = graph.stroke_widths[:n_valid].astype(np.float32) / page_diag

    # Dash pattern → hash bucket
    dash_hash = np.zeros(seq_len, dtype=np.int64)
    for i in range(n_valid):
        dash_array, phase = graph.dash_patterns[i]
        dash_hash[i] = _hash_dash_pattern(dash_array, phase)

    # Color RGB (first 3 channels of RGBA stroke_colors)
    color_rgb = np.zeros((seq_len, 3), dtype=np.float32)
    if n_valid > 0:
        color_rgb[:n_valid] = graph.stroke_colors[:n_valid, :3].astype(np.float32)

    # Wall confidence score
    confidence = np.zeros(seq_len, dtype=np.float32)
    if n_valid > 0:
        confidence[:n_valid] = graph.confidence_wall[:n_valid].astype(np.float32)

    # Attention mask: True for valid tokens, False for padding
    mask = np.zeros(seq_len, dtype=bool)
    mask[:n_valid] = True

    # Raw coordinates preserved in PDF user units
    raw_coords_padded = np.zeros((seq_len, 4), dtype=np.float64)
    if n_valid > 0:
        raw_coords_padded[:n_valid] = raw_coords

    return {
        "operator_type": torch.from_numpy(op_indices),
        "coordinates": torch.from_numpy(norm_coords),
        "stroke_width": torch.from_numpy(stroke_width),
        "dash_hash": torch.from_numpy(dash_hash),
        "color_rgb": torch.from_numpy(color_rgb),
        "confidence_wall": torch.from_numpy(confidence),
        "attention_mask": torch.from_numpy(mask),
        "raw_coordinates": torch.from_numpy(raw_coords_padded),
    }


def collate_graphs(
    graphs: list[RawGraph],
    max_tokens: int | None = None,
) -> dict[str, torch.Tensor]:
    """Batch multiple RawGraphs into a single tensor dict.

    All graphs are padded to the same sequence length (the longest graph
    in the batch, clamped to max_tokens if specified).

    Args:
        graphs: List of RawGraph instances.
        max_tokens: If specified, truncate each graph to this length.

    Returns:
        Batched dict with tensors having leading batch dimension (B, ...).

    Raises:
        ValueError: If graphs list is empty.
    """
    if not graphs:
        msg = "Cannot collate an empty list of graphs"
        raise ValueError(msg)

    # Determine padded sequence length
    lengths = [len(g.edges) for g in graphs]
    if max_tokens is not None:
        lengths = [min(n, max_tokens) for n in lengths]
    pad_to = max(lengths)

    features_list = [graph_to_token_features(g, max_tokens=pad_to) for g in graphs]
    return {key: torch.stack([f[key] for f in features_list]) for key in features_list[0]}
