"""Differentiable architectural axioms for the NeSy SAT constraint solver.

Implements four geometric axioms enforced at each denoising step during
diffusion (MODEL_SPEC.md Table 2):

    L_ortho     — Orthogonal Integrity (EQ-07)
    L_parallel  — Parallel Pair Constancy (EQ-08)
    L_junction  — Junction Closure (EQ-09)
    L_intersect — Spatial Non-Intersection (EQ-10)

Each axiom is an ``nn.Module`` with a learnable weight and returns an
``AxiomResult`` containing a differentiable scalar loss, per-element
violation mask/scores, and the current weight.

A configurable ``AxiomRegistry`` manages axiom instances and provides
batch evaluation.

Reference: ARCHITECTURE.md §Stage 4, MODEL_SPEC.md §Differentiable
NeSy Constraint Satisfaction.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from docs.interfaces.constraint_signals import AxiomResult

if TYPE_CHECKING:
    from src.pipeline.config import ConstraintConfig


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def find_parallel_pairs(
    angles: torch.Tensor,
    threshold: float,
    max_pairs: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find all edge pairs with near-parallel angles — fully on-device.

    Uses a row-chunked pairwise π-periodic angle-difference test. All work
    stays on the input device; there are no ``.tolist()`` round-trips or
    Python-level loops over edges. Peak memory is bounded to
    ``_TARGET_CHUNK_MEM`` bytes via adaptive row chunking, so the function
    scales to tens of thousands of edges on a 48 GB GPU.

    The old sort + Python-scan implementation was O(P) in pure Python with
    ``.tolist()`` transfers. For axis-aligned floor plans where nearly
    every edge is parallel to nearly every other, P approaches E² and the
    scan degraded to tens of millions of Python iterations per training
    step — a ~20-minute-per-step pathology on Colab.

    Args:
        angles: Edge angles in [0, π), shape ``(E,)``.
        threshold: Max angle difference (radians) to count as parallel.
        max_pairs: Optional hard cap on the number of returned pairs. When
            the full enumeration produces more than ``max_pairs``, the
            output is uniformly subsampled to exactly ``max_pairs`` — an
            unbiased estimator of the downstream mean loss that bounds
            memory on pathological batches where P approaches E².

    Returns:
        ``(ei, ej)`` — original edge indices of parallel pairs (ei < ej),
        on the same device as ``angles``.
    """
    device = angles.device
    num_edges = angles.shape[0]
    if num_edges < 2:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    # NOTE(constrain): the return value is a pair of long index tensors —
    # no gradient ever flows through this function. Wrapping the entire
    # body in ``torch.no_grad()`` is essential, not cosmetic: without it,
    # every per-chunk ``diff`` / ``wrapped`` tensor is retained in the
    # autograd graph (because ``angles`` → ``directions`` → ``positions``
    # is differentiable), turning the bounded 256 MB per-chunk budget
    # into tens of GB of retained activations across a full SFT step.
    # This was the OOM seen at batch=16 after the initial vectorisation.
    with torch.no_grad():
        # Adaptive row chunking. A chunk of K rows allocates ~K * E * 5 bytes
        # peak (one (K, E) float32 diff + one (K, E) bool mask + ancillary
        # (K, E) row-index broadcast). Cap per-chunk memory so the function
        # fits a predictable budget regardless of E.
        _TARGET_CHUNK_MEM = 256 * 1024 * 1024  # 256 MB
        chunk_size = max(1, _TARGET_CHUNK_MEM // max(num_edges * 5, 1))
        chunk_size = min(chunk_size, num_edges)

        ei_parts: list[torch.Tensor] = []
        ej_parts: list[torch.Tensor] = []
        all_cols = torch.arange(num_edges, device=device)

        for s in range(0, num_edges, chunk_size):
            e = min(s + chunk_size, num_edges)

            # Pairwise angle diff for rows [s, e) vs all columns — on-device.
            diff = (angles[s:e].unsqueeze(1) - angles.unsqueeze(0)).abs()  # (K, E)
            # π-periodic distance: edges at ≈0 and ≈π are parallel.
            # NOTE(constrain): use ``<=`` to match the legacy sort + scan loop,
            # which broke only on strict ``diff > threshold`` — the equivalence
            # tests rely on this exact boundary behavior.
            wrapped = torch.minimum(diff, math.pi - diff)
            parallel = wrapped <= threshold

            # Upper-triangular only: enforce global row_index < col_index.
            row_global = torch.arange(s, e, device=device).unsqueeze(1)  # (K, 1)
            parallel = parallel & (all_cols.unsqueeze(0) > row_global)

            rows, cols = torch.where(parallel)
            if rows.numel() > 0:
                ei_parts.append(rows + s)
                ej_parts.append(cols)

        if not ei_parts:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        ei_out = torch.cat(ei_parts)
        ej_out = torch.cat(ej_parts)

        if max_pairs is not None and ei_out.numel() > max_pairs:
            # Uniform random subsample — unbiased estimate of the downstream
            # mean loss, hard memory cap on pathological batches.
            perm = torch.randperm(ei_out.numel(), device=device)[:max_pairs]
            ei_out = ei_out[perm]
            ej_out = ej_out[perm]

        return ei_out, ej_out


def find_nearby_edge_pairs(
    edge_index: torch.Tensor,
    positions: torch.Tensor,
    proximity_threshold: float,
    batch_index: torch.Tensor | None = None,
    max_pairs: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find non-adjacent edge pairs whose midpoints are within proximity.

    Fully vectorised on the input device: computes pairwise midpoint
    distances in row-chunks bounded by ``_TARGET_CHUNK_MEM`` bytes. No
    Python loops over edges, no CPU syncs.

    For batched inputs where ``positions`` and ``edge_index`` span multiple
    samples (see :func:`edges_from_adjacency`), pass ``batch_index`` to
    restrict pair generation to same-batch pairs. This matters both for
    correctness (avoids spurious cross-batch intersection losses) and for
    performance (keeps the candidate set small).

    Args:
        edge_index: ``(2, E)`` COO.
        positions: ``(N, 2)`` node positions (flat).
        proximity_threshold: Distance threshold for candidate pairs.
        batch_index: Optional ``(E,)`` batch membership per edge. When
            provided, cross-batch pairs are excluded.
        max_pairs: Optional hard cap on the number of returned pairs.
            Uniformly subsampled when exceeded — unbiased estimator of
            the downstream mean loss, hard memory cap on pathological
            batches.

    Returns:
        ``(ei, ej)`` — edge indices of nearby non-adjacent pairs (ei < ej).
    """
    device = positions.device
    num_edges = edge_index.shape[1]

    if num_edges < 2:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    # NOTE(constrain): the return value is a pair of long index tensors —
    # no gradient ever flows through this function. Wrapping the entire
    # body in ``torch.no_grad()`` is essential, not cosmetic: without it,
    # every per-chunk ``diff`` / ``dist`` / ``nearby`` tensor is retained
    # in the autograd graph (because ``positions`` is differentiable),
    # turning the bounded 256 MB per-chunk budget into tens of GB of
    # retained activations across a full SFT step. This was the OOM seen
    # at batch=16 after the initial vectorisation.
    with torch.no_grad():
        src, dst = edge_index[0], edge_index[1]
        mid = (positions[src] + positions[dst]) * 0.5  # (E, 2)
        half_len = (positions[dst] - positions[src]).norm(dim=-1) * 0.5  # (E,)

        # Adaptive row-chunking to bound peak memory.
        # A row chunk of size K allocates (K, E, 2) floats for `diff` plus a
        # few (K, E) scratch tensors — ~K * E * 20 bytes peak. Cap per chunk
        # at _TARGET_CHUNK_MEM so the full function stays within a predictable
        # memory budget regardless of E.
        _TARGET_CHUNK_MEM = 256 * 1024 * 1024  # 256 MB
        chunk_size = max(1, _TARGET_CHUNK_MEM // max(num_edges * 20, 1))
        chunk_size = min(chunk_size, num_edges)

        all_col = torch.arange(num_edges, device=device)

        ei_parts: list[torch.Tensor] = []
        ej_parts: list[torch.Tensor] = []

        for s in range(0, num_edges, chunk_size):
            e = min(s + chunk_size, num_edges)

            # Pairwise midpoint distance for rows [s, e) vs all columns.
            diff = mid[s:e].unsqueeze(1) - mid.unsqueeze(0)      # (K, E, 2)
            dist = diff.norm(dim=-1)                              # (K, E)

            # Proximity reach: two segments can come within `proximity_threshold`
            # only if their midpoints are within (half_len_i + half_len_j +
            # proximity_threshold). This is a necessary (not sufficient)
            # condition — the downstream segment-segment test filters further.
            reach = (
                half_len[s:e].unsqueeze(1)
                + half_len.unsqueeze(0)
                + proximity_threshold
            )
            nearby = dist < reach  # (K, E)

            # Keep only upper-triangular pairs (global row index < col index).
            row_global = torch.arange(s, e, device=device).unsqueeze(1)  # (K, 1)
            nearby = nearby & (all_col.unsqueeze(0) > row_global)

            # Same-batch filter (excludes cross-batch pairs).
            if batch_index is not None:
                nearby = nearby & (
                    batch_index[s:e].unsqueeze(1) == batch_index.unsqueeze(0)
                )

            # Exclude adjacent edges (sharing a node).
            src_i = src[s:e].unsqueeze(1)   # (K, 1)
            dst_i = dst[s:e].unsqueeze(1)
            src_j = src.unsqueeze(0)        # (1, E)
            dst_j = dst.unsqueeze(0)
            shares = (
                (src_i == src_j)
                | (src_i == dst_j)
                | (dst_i == src_j)
                | (dst_i == dst_j)
            )
            nearby = nearby & ~shares

            # Gather surviving pair indices within this chunk.
            rows, cols = torch.where(nearby)
            if rows.numel() > 0:
                ei_parts.append(rows + s)
                ej_parts.append(cols)

        if not ei_parts:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        ei_out = torch.cat(ei_parts)
        ej_out = torch.cat(ej_parts)

        if max_pairs is not None and ei_out.numel() > max_pairs:
            # Uniform random subsample — unbiased estimate of the downstream
            # mean loss, hard memory cap on pathological batches.
            perm = torch.randperm(ei_out.numel(), device=device)[:max_pairs]
            ei_out = ei_out[perm]
            ej_out = ej_out[perm]

        return ei_out, ej_out


def compute_edge_directions(
    node_positions: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Compute normalized direction vectors for all edges.

    Args:
        node_positions: Junction coordinates, shape ``(B, N, 2)`` or ``(N, 2)``.
        edge_index: COO edge indices, shape ``(2, E)``.

    Returns:
        Normalized direction vectors, shape ``(E, 2)``.
    """
    if node_positions.dim() == 3:
        # Flatten batch dimension — edge_index references global node ids.
        positions = node_positions.reshape(-1, 2)
    else:
        positions = node_positions

    src, dst = edge_index[0], edge_index[1]
    direction = positions[dst] - positions[src]
    length = direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return direction / length


def compute_edge_angles(
    directions: torch.Tensor,
) -> torch.Tensor:
    """Compute angles in [0, π) for edge direction vectors.

    Args:
        directions: Normalized direction vectors, shape ``(E, 2)``.

    Returns:
        Angles in radians, shape ``(E,)``.
    """
    angles = torch.atan2(directions[:, 1], directions[:, 0])
    # Map to [0, π) — edges are undirected so angle and angle+π are equivalent.
    return angles % math.pi


def edges_from_adjacency(
    adjacency: torch.Tensor,
    node_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Extract edge_index ``(2, E)`` from an adjacency matrix.

    Args:
        adjacency: Adjacency logits or binary, shape ``(B, N, N)`` or ``(N, N)``.
        node_mask: Valid-node mask, shape ``(B, N)`` or ``(N,)``.
            ``None`` means all nodes are valid.
        threshold: Logit/probability threshold for edge existence.

    Returns:
        Edge index in COO format, shape ``(2, E)``.
    """
    if adjacency.dim() == 2:
        adjacency = adjacency.unsqueeze(0)

    adj_binary = (adjacency > threshold).float()

    # Zero out invalid nodes.
    if node_mask is not None:
        mask = node_mask if node_mask.dim() == 3 else node_mask.unsqueeze(-1)
        adj_binary = adj_binary * mask * mask.transpose(-1, -2)

    # Upper-triangular to avoid duplicate undirected edges.
    adj_upper = torch.triu(adj_binary, diagonal=1)

    # Vectorised collection across the batch: a single ``torch.nonzero`` on
    # the full (B, N, N) tensor replaces the Python ``for b in range(B)``
    # loop. The old loop did B ``torch.where`` calls, each forcing a CUDA
    # sync — with B=16 that was 16 round-trips per training step.
    # ``torch.nonzero`` itself pays a single CUDA sync (variable output),
    # so we go from B syncs down to 1.
    nz = torch.nonzero(adj_upper > 0, as_tuple=False)  # (K, 3): (batch, row, col)
    if nz.numel() == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=adjacency.device)

    n = adjacency.shape[-1]
    batch_idx = nz[:, 0]
    src_all = batch_idx * n + nz[:, 1]
    dst_all = batch_idx * n + nz[:, 2]
    return torch.stack([src_all, dst_all], dim=0)


# ---------------------------------------------------------------------------
# Base axiom
# ---------------------------------------------------------------------------


class Axiom(nn.Module, ABC):
    """Base class for differentiable architectural axioms.

    Each axiom evaluates a geometric property of the predicted graph and
    returns an ``AxiomResult`` with a differentiable scalar loss,
    per-element violation mask/scores, and a learned weight for composite
    loss weighting.
    """

    def __init__(self, name: str, initial_weight: float = 1.0) -> None:
        super().__init__()
        self.axiom_name = name
        self.weight = nn.Parameter(torch.tensor(initial_weight))

    @abstractmethod
    def forward(
        self,
        node_positions: torch.Tensor,
        adjacency: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> AxiomResult:
        """Evaluate the axiom on a predicted structural graph.

        Args:
            node_positions: Junction coordinates, ``(B, N, 2)``.
            adjacency: Adjacency logits or binary, ``(B, N, N)``.
            edge_index: COO edge indices, ``(2, E)``.
            node_mask: Valid-node mask, ``(B, N)`` bool.

        Returns:
            ``AxiomResult`` with loss, violation mask/scores, and weight.
        """


# ---------------------------------------------------------------------------
# C-001: Orthogonal Integrity
# ---------------------------------------------------------------------------


class OrthogonalIntegrityAxiom(Axiom):
    """Enforces Manhattan / non-Manhattan alignment of wall edges.

    Wall edges meeting at junctions should form exact 90° or 180° angles.
    The cosine-similarity penalty is zero when edges are perfectly
    perpendicular or parallel, and maximized at 45°.

    .. math::

        \\mathcal{L}_{ortho} = \\sum_{(e_1, e_2) \\in \\text{adj\\_pairs}}
            \\cos^2(\\theta) \\cdot (1 - \\cos^2(\\theta))

    Reference:
        MODEL_SPEC.md Table 2, EQ-07.
    """

    def __init__(
        self,
        tolerance_deg: float = 5.0,
        initial_weight: float = 1.0,
        max_pairs: int | None = None,
    ) -> None:
        super().__init__("orthogonal", initial_weight)
        # Pre-compute the loss value at the tolerance angle boundary.
        # Angles within tolerance_deg of 0° or 90° are not violations.
        c = math.cos(math.radians(tolerance_deg))
        self.violation_threshold = c * c * (1.0 - c * c)
        self.max_pairs = max_pairs

    def forward(
        self,
        node_positions: torch.Tensor,
        adjacency: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> AxiomResult:
        device = node_positions.device
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            return self._empty_result(device)

        directions = compute_edge_directions(node_positions, edge_index)

        # Build adjacent edge pairs: edges sharing a node.
        # For each node, gather its incident edges and form pairs.
        src, dst = edge_index[0], edge_index[1]
        # Map node → incident edge indices.
        all_nodes = torch.cat([src, dst])
        all_edge_ids = torch.arange(num_edges, device=device).repeat(2)

        # Sort by node to group incident edges.
        sorted_order = torch.argsort(all_nodes)
        sorted_nodes = all_nodes[sorted_order]
        sorted_edge_ids = all_edge_ids[sorted_order]

        # Find boundaries between different nodes.
        change_mask = torch.cat(
            [
                torch.ones(1, dtype=torch.bool, device=device),
                sorted_nodes[1:] != sorted_nodes[:-1],
            ]
        )
        group_starts = torch.nonzero(change_mask, as_tuple=False).squeeze(1)
        group_ends = torch.cat(
            [
                group_starts[1:],
                torch.tensor([sorted_nodes.shape[0]], device=device),
            ]
        )

        # Vectorised segmented pair expansion: replaces a Python loop that
        # called ``.item()`` twice per junction (hundreds of CUDA syncs per
        # training step). For each sorted position k, generate pairs
        # ``(k, k+1), (k, k+2), ..., (k, group_end[k] - 1)``.
        num_groups = group_starts.shape[0]
        group_sizes = group_ends - group_starts  # (G,)
        group_id = torch.repeat_interleave(
            torch.arange(num_groups, device=device),
            group_sizes,
        )  # (L,)
        positions = torch.arange(sorted_edge_ids.shape[0], device=device)  # (L,)
        group_end_of_pos = group_ends[group_id]  # (L,)
        forward_count = group_end_of_pos - positions - 1  # (L,) ≥ 0

        # Expand to explicit pair index lists, fully on-device.
        pair_left = torch.repeat_interleave(positions, forward_count)  # (P,)
        if pair_left.numel() == 0:
            return self._empty_result(device)

        # Per-position offset within each block: 0, 1, 2, ..., forward_count[k]-1.
        cum = torch.cumsum(forward_count, dim=0)
        block_start = cum - forward_count  # exclusive cumsum: block start per position
        offset = (
            torch.arange(pair_left.shape[0], device=device) - block_start[pair_left]
        )
        pair_right = pair_left + 1 + offset  # (P,)

        # Uniform random subsample when the full enumeration exceeds the
        # per-axiom cap. Mean-of-losses is unbiased under uniform sampling,
        # so gradient direction stays correct in expectation. This is the
        # hard memory cap for pathological batches.
        if self.max_pairs is not None and pair_left.numel() > self.max_pairs:
            perm = torch.randperm(pair_left.numel(), device=device)[: self.max_pairs]
            pair_left = pair_left[perm]
            pair_right = pair_right[perm]

        pair_i = sorted_edge_ids[pair_left]
        pair_j = sorted_edge_ids[pair_right]

        # Cosine similarity between direction vectors of each pair.
        dot = (directions[pair_i] * directions[pair_j]).sum(dim=-1)
        cos_sq = dot.square()

        # Loss per pair: cos²·(1-cos²) — zero when perfectly parallel (cos²=1)
        # or perpendicular (cos²=0), maximized at 45° (cos²=0.5).
        per_pair_loss = cos_sq * (1.0 - cos_sq)

        loss = per_pair_loss.mean()

        # Violation: pairs whose loss exceeds the tolerance-angle threshold.
        violation_mask = per_pair_loss > self.violation_threshold

        return AxiomResult(
            name=self.axiom_name,
            loss=loss,
            violation_mask=violation_mask,
            violation_scores=per_pair_loss.detach(),
            weight=self.weight,
        )

    def _empty_result(self, device: torch.device) -> AxiomResult:
        return AxiomResult(
            name=self.axiom_name,
            loss=torch.tensor(0.0, device=device, requires_grad=True),
            violation_mask=torch.zeros(0, dtype=torch.bool, device=device),
            violation_scores=torch.zeros(0, device=device),
            weight=self.weight,
        )


# ---------------------------------------------------------------------------
# C-002: Parallel Pair Constancy
# ---------------------------------------------------------------------------


class ParallelPairConstancyAxiom(Axiom):
    """Enforces uniform wall thickness via parallel-pair distance constraint.

    Parallel wall edges (forming wall thickness) should maintain uniform
    distance.  Outlier distances beyond the IQR of observed parallel pairs
    are penalized.

    .. math::

        \\mathcal{L}_{parallel} = \\sum_{(e_1, e_2) \\in \\text{parallel}}
            \\max\\bigl(0,\\; |d(e_1, e_2) - \\mu| - \\text{IQR}/2\\bigr)

    Reference:
        MODEL_SPEC.md Table 2, EQ-08.
    """

    def __init__(
        self,
        angle_threshold_deg: float = 5.0,
        iqr_scale: float = 1.5,
        initial_weight: float = 1.0,
        max_pairs: int | None = None,
    ) -> None:
        super().__init__("parallel_pair", initial_weight)
        self.angle_threshold = math.radians(angle_threshold_deg)
        self.iqr_scale = iqr_scale
        self.max_pairs = max_pairs

    def forward(
        self,
        node_positions: torch.Tensor,
        adjacency: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> AxiomResult:
        device = node_positions.device
        num_edges = edge_index.shape[1]

        if num_edges < 2:
            return self._empty_result(device)

        directions = compute_edge_directions(node_positions, edge_index)
        angles = compute_edge_angles(directions)

        positions = node_positions.reshape(-1, 2) if node_positions.dim() == 3 else node_positions

        # Find parallel pairs on-device, capped at ``max_pairs`` via uniform
        # random subsampling inside the helper. Without the cap, pathological
        # axis-aligned batches produce O(E²) pairs and the downstream
        # differentiable per-pair computation OOMs.
        ei, ej = find_parallel_pairs(
            angles, self.angle_threshold, max_pairs=self.max_pairs,
        )
        if ei.numel() == 0:
            return self._empty_result(device)

        # Perpendicular distance between parallel line segments.
        # Use midpoint-to-line distance for differentiability.
        src_i = edge_index[0, ei]
        src_j, dst_j = edge_index[0, ej], edge_index[1, ej]

        # Midpoint of edge j.
        mid_j = (positions[src_j] + positions[dst_j]) * 0.5

        # Direction of edge i (already normalized).
        d_i = directions[ei]

        # Point-to-line distance: project (mid_j - src_i) onto normal of edge i.
        delta = mid_j - positions[src_i]
        # Normal: rotate direction 90° → (-dy, dx).
        normal = torch.stack([-d_i[:, 1], d_i[:, 0]], dim=-1)
        distances = (delta * normal).sum(dim=-1).abs()

        # Compute IQR of distances (differentiable via soft quantiles).
        sorted_dist, _ = distances.sort()
        n_pairs = sorted_dist.shape[0]
        q1_idx = max(0, int(n_pairs * 0.25) - 1)
        q3_idx = min(n_pairs - 1, int(n_pairs * 0.75))
        q1 = sorted_dist[q1_idx]
        q3 = sorted_dist[q3_idx]
        iqr = (q3 - q1).clamp(min=1e-6)
        median_idx = n_pairs // 2
        mu = sorted_dist[median_idx]

        # Loss: penalize distances outside μ ± (IQR/2 * scale).
        margin = iqr * 0.5 * self.iqr_scale
        per_pair_loss = torch.relu((distances - mu).abs() - margin)
        loss = per_pair_loss.mean()

        violation_mask = per_pair_loss > 0

        return AxiomResult(
            name=self.axiom_name,
            loss=loss,
            violation_mask=violation_mask,
            violation_scores=per_pair_loss.detach(),
            weight=self.weight,
        )

    def _empty_result(self, device: torch.device) -> AxiomResult:
        return AxiomResult(
            name=self.axiom_name,
            loss=torch.tensor(0.0, device=device, requires_grad=True),
            violation_mask=torch.zeros(0, dtype=torch.bool, device=device),
            violation_scores=torch.zeros(0, device=device),
            weight=self.weight,
        )


# ---------------------------------------------------------------------------
# C-003: Junction Closure
# ---------------------------------------------------------------------------


class JunctionClosureAxiom(Axiom):
    """Penalizes dangling edges (nodes with degree < 2).

    Uses the graph Laplacian to encourage closed loops.

    .. math::

        \\mathcal{L}_{junction} = \\|L \\cdot x\\|^2

    where *L = D - A* is the graph Laplacian.  Nodes with low degree
    (especially degree-1 endpoints) are additionally penalized.

    Reference:
        MODEL_SPEC.md Table 2, EQ-09.
    """

    def __init__(
        self,
        min_degree: int = 2,
        initial_weight: float = 0.5,
    ) -> None:
        super().__init__("junction_closure", initial_weight)
        self.min_degree = min_degree

    def forward(
        self,
        node_positions: torch.Tensor,
        adjacency: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> AxiomResult:
        # Build soft adjacency from logits if needed.
        adj = torch.sigmoid(adjacency) if adjacency.requires_grad else adjacency

        # Degree per node.
        degree = adj.sum(dim=-1)

        # Mask invalid nodes.
        if node_mask is not None:
            degree = degree * node_mask.float()

        # Graph Laplacian: L = D - A.
        diag = torch.diag_embed(degree)
        laplacian = diag - adj

        # Laplacian penalty: ||L @ x||².
        lx = torch.bmm(laplacian, node_positions)  # (B, N, 2)
        if node_mask is not None:
            lx = lx * node_mask.unsqueeze(-1).float()
        laplacian_loss = (
            (lx**2).sum() / max(node_mask.sum().item(), 1.0)
            if node_mask is not None
            else (lx**2).mean()
        )

        # Additional dangling-node penalty.
        degree_deficit = torch.relu(self.min_degree - degree)
        if node_mask is not None:
            degree_deficit = degree_deficit * node_mask.float()
            dangling_loss = degree_deficit.sum() / max(node_mask.sum().item(), 1.0)
        else:
            dangling_loss = degree_deficit.mean()

        loss = laplacian_loss + dangling_loss

        # Violation mask: nodes with degree < min_degree.
        with torch.no_grad():
            node_degree_int = adj.sum(dim=-1)
            violation_flat = node_degree_int.reshape(-1) < self.min_degree
            if node_mask is not None:
                violation_flat = violation_flat & node_mask.reshape(-1)
            violation_scores_flat = degree_deficit.reshape(-1)

        return AxiomResult(
            name=self.axiom_name,
            loss=loss,
            violation_mask=violation_flat,
            violation_scores=violation_scores_flat.detach(),
            weight=self.weight,
        )


# ---------------------------------------------------------------------------
# C-004: Spatial Non-Intersection
# ---------------------------------------------------------------------------


class SpatialNonIntersectionAxiom(Axiom):
    """Prevents overlapping / crossing wall segments.

    Non-adjacent edge pairs should not geometrically intersect.  Uses a
    differentiable closest-approach distance between line segments.

    .. math::

        \\mathcal{L}_{intersect} = \\sum_{(e_i, e_j) \\in \\text{non\\_adj}}
            \\text{ReLU}(\\epsilon - d_{\\min}(e_i, e_j))

    Reference:
        MODEL_SPEC.md Table 2, EQ-10.
    """

    def __init__(
        self,
        epsilon: float = 1e-3,
        proximity_threshold: float = 0.15,
        initial_weight: float = 2.0,
        max_pairs: int | None = None,
    ) -> None:
        super().__init__("non_intersection", initial_weight)
        self.epsilon = epsilon
        self.proximity_threshold = proximity_threshold
        self.max_pairs = max_pairs

    def forward(
        self,
        node_positions: torch.Tensor,
        adjacency: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> AxiomResult:
        device = node_positions.device
        num_edges = edge_index.shape[1]

        if num_edges < 2:
            return self._empty_result(device)

        # Recover per-edge batch membership BEFORE flattening positions.
        # edges_from_adjacency returns flat indices with stride N per batch
        # item, so batch_index[e] = edge_index[0, e] // N. Without this
        # filter, cross-batch pairs are considered "nearby" in the shared
        # normalised [0, 1] coordinate frame, producing a spurious O(B²·E²)
        # candidate set.
        batch_index: torch.Tensor | None = None
        if node_positions.dim() == 3:
            n_per_batch = node_positions.shape[1]
            positions = node_positions.reshape(-1, 2)
            batch_index = edge_index[0] // n_per_batch
        else:
            positions = node_positions

        src, dst = edge_index[0], edge_index[1]

        # Find nearby non-adjacent edge pairs on-device — vectorised O(E²)
        # in chunks, with same-batch filtering via `batch_index`. Capped at
        # ``max_pairs`` via uniform random subsampling inside the helper so
        # the downstream differentiable per-pair loop has bounded memory on
        # pathological batches.
        ei, ej = find_nearby_edge_pairs(
            edge_index,
            positions,
            self.proximity_threshold,
            batch_index=batch_index,
            max_pairs=self.max_pairs,
        )
        if ei.numel() == 0:
            return self._empty_result(device)

        # Differentiable segment-segment distance — chunked to bound memory.
        PAIR_CHUNK = 65536
        n_pairs = ei.shape[0]
        loss_parts: list[torch.Tensor] = []
        viol_parts: list[torch.Tensor] = []
        score_parts: list[torch.Tensor] = []
        for ps in range(0, n_pairs, PAIR_CHUNK):
            pe = min(ps + PAIR_CHUNK, n_pairs)
            ei_c, ej_c = ei[ps:pe], ej[ps:pe]
            dists = self._segment_distance(
                positions[src[ei_c]],
                positions[dst[ei_c]],
                positions[src[ej_c]],
                positions[dst[ej_c]],
            )
            chunk_loss = torch.relu(self.epsilon - dists)
            loss_parts.append(chunk_loss.sum())
            viol_parts.append(chunk_loss > 0)
            score_parts.append(chunk_loss.detach())

        per_pair_loss = torch.cat(score_parts)
        loss = sum(loss_parts) / max(n_pairs, 1)
        violation_mask = torch.cat(viol_parts)

        return AxiomResult(
            name=self.axiom_name,
            loss=loss,
            violation_mask=violation_mask,
            violation_scores=per_pair_loss,
            weight=self.weight,
        )

    @staticmethod
    def _segment_distance(
        a1: torch.Tensor,
        a2: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable closest distance between segment pairs.

        Segments are (a1→a2) and (b1→b2), each shape ``(P, 2)``.

        Returns:
            Minimum distances, shape ``(P,)``.
        """
        d1 = a2 - a1  # (P, 2)
        d2 = b2 - b1
        r = a1 - b1

        a = (d1 * d1).sum(dim=-1).clamp(min=1e-12)  # |d1|²
        e = (d2 * d2).sum(dim=-1).clamp(min=1e-12)  # |d2|²
        b = (d1 * d2).sum(dim=-1)
        c = (d1 * r).sum(dim=-1)
        f = (d2 * r).sum(dim=-1)

        denom = (a * e - b * b).clamp(min=1e-12)

        # Parametric closest points on the two infinite lines.
        s = (b * f - c * e) / denom
        t = (a * f - b * c) / denom

        # Clamp to [0, 1] with soft clamping for differentiability.
        s = s.clamp(0.0, 1.0)
        t = t.clamp(0.0, 1.0)

        # Recompute after clamping to handle edge cases.
        # If s was clamped, recompute optimal t, and vice versa.
        t = ((b * s + f) / e).clamp(0.0, 1.0)
        s = ((b * t - c) / a).clamp(0.0, 1.0)

        closest_a = a1 + s.unsqueeze(-1) * d1
        closest_b = b1 + t.unsqueeze(-1) * d2

        return (closest_a - closest_b).norm(dim=-1)

    def _empty_result(self, device: torch.device) -> AxiomResult:
        return AxiomResult(
            name=self.axiom_name,
            loss=torch.tensor(0.0, device=device, requires_grad=True),
            violation_mask=torch.zeros(0, dtype=torch.bool, device=device),
            violation_scores=torch.zeros(0, device=device),
            weight=self.weight,
        )


# ---------------------------------------------------------------------------
# C-008: Configurable Axiom Registry
# ---------------------------------------------------------------------------


class AxiomRegistry:
    """Registry for managing architectural axioms.

    Allows adding/removing axioms without code changes.  Axioms are
    registered by name and can be enabled/disabled via config.
    """

    def __init__(self) -> None:
        self._axioms: dict[str, Axiom] = {}

    def register(self, axiom: Axiom) -> None:
        """Register an axiom instance.

        Args:
            axiom: An ``Axiom`` subclass instance.

        Raises:
            ValueError: If an axiom with the same name is already registered.
        """
        if axiom.axiom_name in self._axioms:
            raise ValueError(f"Axiom '{axiom.axiom_name}' is already registered")
        self._axioms[axiom.axiom_name] = axiom

    def unregister(self, name: str) -> None:
        """Remove a registered axiom by name.

        Args:
            name: The axiom name.

        Raises:
            KeyError: If no axiom with that name is registered.
        """
        if name not in self._axioms:
            raise KeyError(f"Axiom '{name}' is not registered")
        del self._axioms[name]

    def get(self, name: str) -> Axiom | None:
        """Retrieve a registered axiom by name.

        Args:
            name: The axiom name.

        Returns:
            The axiom instance, or ``None`` if not found.
        """
        return self._axioms.get(name)

    def list_axioms(self) -> list[str]:
        """Return names of all registered axioms."""
        return list(self._axioms.keys())

    def evaluate_all(
        self,
        node_positions: torch.Tensor,
        adjacency: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> list[AxiomResult]:
        """Evaluate all registered axioms on the given graph.

        Args:
            node_positions: Junction coordinates, ``(B, N, 2)``.
            adjacency: Adjacency logits or binary, ``(B, N, N)``.
            edge_index: COO edge indices, ``(2, E)``.
            node_mask: Valid-node mask, ``(B, N)`` bool.

        Returns:
            List of ``AxiomResult``, one per registered axiom.
        """
        return [
            axiom(node_positions, adjacency, edge_index, node_mask)
            for axiom in self._axioms.values()
        ]

    def parameters(self) -> list[nn.Parameter]:
        """Collect all learnable parameters from registered axioms."""
        params: list[nn.Parameter] = []
        for axiom in self._axioms.values():
            params.extend(axiom.parameters())
        return params

    @classmethod
    def create_default(cls, config: ConstraintConfig) -> AxiomRegistry:
        """Create a registry pre-loaded with the standard four axioms.

        Args:
            config: Constraint configuration providing tolerances and weights.

        Returns:
            An ``AxiomRegistry`` with orthogonal, parallel_pair,
            junction_closure, and non_intersection axioms.
        """
        registry = cls()
        weights = config.initial_weights
        max_pairs = config.max_pairs_per_axiom

        registry.register(
            OrthogonalIntegrityAxiom(
                tolerance_deg=config.ortho_tolerance_deg,
                initial_weight=weights.get("orthogonal", 1.0),
                max_pairs=max_pairs,
            )
        )
        registry.register(
            ParallelPairConstancyAxiom(
                iqr_scale=config.parallel_iqr_scale,
                initial_weight=weights.get("parallel_pair", 1.0),
                max_pairs=max_pairs,
            )
        )
        registry.register(
            JunctionClosureAxiom(
                min_degree=config.junction_min_degree,
                initial_weight=weights.get("junction_closure", 0.5),
            )
        )
        registry.register(
            SpatialNonIntersectionAxiom(
                initial_weight=weights.get("non_intersection", 2.0),
                max_pairs=max_pairs,
            )
        )

        # Freeze weights if meta-learning is disabled.
        if not config.learn_weights:
            for axiom in registry._axioms.values():
                axiom.weight.requires_grad_(False)

        return registry
