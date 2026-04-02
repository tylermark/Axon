"""Interface contract: Tokenizer Agent → Diffusion Agent.

Defines the EnrichedTokenSequence dataclass — the output of Stage 2
(Cross-Modal Tokenization) and input to Stage 3 (Graph Diffusion Engine).

Each vector token embeds:
    [operator_type, x1, y1, x2, y2, stroke_width, dash_hash, color_rgb, ctm_flat]

enriched via bidirectional cross-attention (TEF) with raster semantic features.

Mathematical basis:
    Attention(Q_vector, K_vision, V_vision) = softmax(Q K^T / sqrt(d_k)) V
    [MODEL_SPEC.md §Cross-Modal Feature Alignment, EQ-05]

Architecture parameters:
    d_model = 256, n_heads = 8  [ARCHITECTURE.md §Stage 2]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TokenEmbedding:
    """Individual enriched vector token.

    Represents a single path segment after cross-modal fusion. Contains
    both the raw geometric parameters and the learned embedding that
    incorporates semantic context from the raster feature map.
    """

    operator_type: str
    """Original PDF operator: 'moveto', 'lineto', 'curveto', 'closepath'."""

    start_coord: np.ndarray
    """Start point (x1, y1), shape (2,) float64, normalized to [0, 1]."""

    end_coord: np.ndarray
    """End point (x2, y2), shape (2,) float64, normalized to [0, 1]."""

    stroke_width: float
    """Normalized stroke width (divided by page diagonal)."""

    dash_hash: int
    """Integer hash of the dash pattern for categorical embedding."""

    color_rgb: np.ndarray
    """Stroke color RGB, shape (3,) float64 in [0, 1]."""


@dataclass
class EnrichedTokenSequence:
    """Semantically enriched token sequence from cross-modal fusion.

    This is the primary input to the Diffusion Engine (Stage 3). It contains
    the fused token embeddings where each vector token has absorbed localized
    semantic context from the raster feature map via bidirectional cross-attention.

    The sequence preserves the geometric structure of the RawGraph while adding
    learned representations that enable the diffusion model to distinguish
    structural walls from decorative elements, dimension lines, and furniture.

    Reference: MODEL_SPEC.md §Cross-Modal Feature Alignment, ARCHITECTURE.md §Stage 2.

    Tensor conventions:
        B = batch size, N = number of tokens, D = d_model = 256
    """

    token_embeddings: torch.Tensor
    """Fused token embedding matrix, shape (B, N, D) float32.

    Each row is a d_model-dimensional embedding combining:
    - Geometric parameters (coordinates, stroke properties)
    - 2D learned positional encoding (normalized page coordinates)
    - Cross-modal semantic features from TEF fusion

    D = 256 per ARCHITECTURE.md §Stage 2.
    """

    attention_mask: torch.Tensor
    """Padding mask for variable-length sequences, shape (B, N) bool.

    True for valid tokens, False for padding positions.
    """

    position_encodings: torch.Tensor
    """2D learned positional encodings, shape (B, N, D) float32.

    Based on normalized page coordinates of each token's midpoint.
    Enables the diffusion model to reason about spatial proximity.
    """

    raw_coordinates: torch.Tensor
    """Original token coordinates before normalization, shape (B, N, 4) float64.

    Each row is [x1, y1, x2, y2] in PDF user units. Preserved for
    geometric loss computation and constraint evaluation.
    """

    stroke_features: torch.Tensor
    """Raw stroke property features, shape (B, N, F_stroke) float32.

    Includes stroke width, dash pattern embedding, and color channels.
    Preserved separately for the constraint agent's use.
    """

    edge_indices: torch.Tensor
    """Graph connectivity from the original RawGraph, shape (2, E_batch) int64.

    COO-format edge index compatible with PyG. Maps token indices to their
    original graph connectivity. Used to initialize the diffusion process.
    """

    confidence_wall: torch.Tensor
    """Per-token wall confidence from Parser, shape (B, N) float32.

    Propagated from RawGraph.confidence_wall. Provides an initial prior
    for the diffusion model.
    """

    raster_features: torch.Tensor | None = None
    """Multi-scale raster feature maps, shape (B, C, H, W) float32.

    Extracted by the vision backbone (HRNet/Swin). None when operating
    in vector-only fallback mode (no raster available).

    Reference: ARCHITECTURE.md §Stage 2 Key Design Decisions.
    """

    vision_to_vector_weights: torch.Tensor | None = None
    """Cross-attention weights from vision→vector pass, shape (B, N_heads, N, H*W) float32.

    Retained for interpretability and debugging. Shows which visual regions
    each vector token attended to. None if not retained.
    """

    page_dimensions: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, 2)
    )
    """Page width and height per batch item, shape (B, 2) float64.

    In PDF user units. Needed for denormalization during serialization.
    """

    batch_indices: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    """Per-token batch assignment, shape (N_total,) int64.

    Used for PyG-style batching where multiple graphs are concatenated
    into a single tensor with batch tracking.
    """

    d_model: int = 256
    """Embedding dimension. Fixed at 256 per architecture spec."""

    n_heads: int = 8
    """Number of attention heads used in cross-modal fusion."""

    is_vector_only: bool = False
    """True if raster features were unavailable and vector-only fallback was used."""
