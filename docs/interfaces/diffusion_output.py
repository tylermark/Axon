"""Interface contract: Diffusion Agent → Constraint / Topology / Physics Agents.

Defines the RefinedStructuralGraph dataclass — the output of Stage 3
(Graph Diffusion Engine). This intermediate representation passes through
Constraint (Stage 4), Topology (Stage 5), and Physics (Stage 6) before
finalization.

Mathematical basis:
    G* = (V, E) where
    V ∈ R^(N×2)      — precise wall junction coordinates
    A ∈ {0,1}^(N×N)  — adjacency matrix (edge connectivity)

    min_θ Σ_t E[||ε - ε_θ(G_t, t, c)||²]
    [MODEL_SPEC.md §Generative Graph Denoising Diffusion Engine, EQ-06]

Architecture:
    12 transformer blocks, d_model=512, n_heads=8
    T=1000 (training), DDIM 50 steps (inference)
    [ARCHITECTURE.md §Stage 3]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch


class JunctionType(str, Enum):
    """Classification of wall junction nodes.

    Determined by the degree and angle configuration at each node
    in the predicted structural graph.
    """

    CORNER = "corner"
    """L-junction: two edges meeting at ~90° or other angle."""

    T_JUNCTION = "t_junction"
    """T-intersection: three edges, one wall terminating into another."""

    CROSS = "cross"
    """Cross-point: four edges forming a + intersection."""

    ENDPOINT = "endpoint"
    """Dangling endpoint: degree-1 node (should be penalized by constraints)."""

    UNCLASSIFIED = "unclassified"
    """Not yet classified. Present in intermediate denoising steps."""


@dataclass
class RefinedStructuralGraph:
    """Predicted structural graph from the diffusion denoising process.

    Represents the output of the reverse diffusion process: a graph where
    nodes are wall junctions (corners, T-intersections, cross-points) and
    edges are wall segments connecting them.

    This is an INTERMEDIATE output — it must pass through Constraint,
    Topology, and Physics validation before becoming a FinalizedGraph.

    Tensor conventions:
        B = batch size, N = max nodes per graph, E = total edges in batch
    """

    node_positions: torch.Tensor
    """Predicted wall junction coordinates, shape (B, N, 2) float32.

    Continuous (x, y) positions in normalized [0, 1] page space.
    These are the denoised coordinates from the reverse diffusion.
    """

    adjacency_logits: torch.Tensor
    """Edge existence logits, shape (B, N, N) float32.

    Pre-sigmoid logits for each potential edge. The discrete adjacency
    matrix is obtained via thresholding: A = (σ(logits) > 0.5).
    Kept as logits for differentiable constraint and topology losses.
    """

    node_mask: torch.Tensor
    """Valid node mask, shape (B, N) bool.

    True for real nodes, False for padding positions.
    Graphs in a batch may have different numbers of junctions.
    """

    edge_index: torch.Tensor
    """Sparse edge index in COO format, shape (2, E_batch) int64.

    Derived from adjacency_logits by thresholding. Compatible with
    PyTorch Geometric message passing.
    """

    edge_logits: torch.Tensor
    """Per-edge existence logits for sparse edges, shape (E_batch,) float32.

    Corresponding logit values for edges in edge_index.
    """

    junction_types: list[list[JunctionType]]
    """Per-node junction classification, shape [B][N].

    Classified based on node degree and edge angles in the predicted graph.
    May be UNCLASSIFIED during intermediate denoising steps.
    """

    node_features: torch.Tensor
    """Per-node feature vectors from the transformer, shape (B, N, D) float32.

    D = 512 (d_model of diffusion transformer). Contains learned
    representations useful for downstream constraint evaluation.
    """

    edge_features: torch.Tensor | None = None
    """Per-edge feature vectors, shape (E_batch, D_edge) float32.

    Optional edge-level features from message passing. Useful for
    the constraint agent's wall thickness estimation.
    """

    denoising_step: int = 0
    """Current denoising timestep (0 = fully denoised, T = pure noise).

    During inference, the constraint agent receives intermediate graphs
    at each step t and returns gradient signals.
    """

    total_steps: int = 1000
    """Total number of diffusion timesteps T."""

    noise_level: float = 0.0
    """Estimated remaining noise level σ_t at current step.

    Useful for the constraint agent to calibrate projection strength:
    softer projection at high noise, harder snap near t=0.
    """

    context_embeddings: torch.Tensor | None = None
    """Cross-modal context from tokenizer, shape (B, N_ctx, D_ctx) float32.

    The conditioning signal c from the EnrichedTokenSequence. Passed
    through for potential use by downstream modules.
    """

    diffusion_loss: float | None = None
    """Variational lower bound loss at current step (training only).

    L_diffusion = E[||ε - ε_θ(G_t, t, c)||²]
    """

    batch_indices: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    """Per-node batch assignment for PyG-style batching, shape (N_total,) int64."""

    page_dimensions: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, 2)
    )
    """Page width and height per batch item, shape (B, 2) float64.

    Needed for denormalization when computing real-world constraint losses.
    """
