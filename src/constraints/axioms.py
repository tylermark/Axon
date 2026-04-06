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

    # Collect edge indices across all batch items.
    batch_offset = 0
    n = adjacency.shape[-1]
    src_list = []
    dst_list = []
    for b in range(adj_upper.shape[0]):
        rows, cols = torch.where(adj_upper[b] > 0)
        src_list.append(rows + batch_offset)
        dst_list.append(cols + batch_offset)
        batch_offset += n

    if len(src_list) == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=adjacency.device)

    src_all = torch.cat(src_list)
    dst_all = torch.cat(dst_list)
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
    ) -> None:
        super().__init__("orthogonal", initial_weight)
        # Pre-compute the loss value at the tolerance angle boundary.
        # Angles within tolerance_deg of 0° or 90° are not violations.
        c = math.cos(math.radians(tolerance_deg))
        self.violation_threshold = c * c * (1.0 - c * c)

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
                torch.tensor([True], device=device),
                sorted_nodes[1:] != sorted_nodes[:-1],
            ]
        )
        group_starts = torch.where(change_mask)[0]
        group_ends = torch.cat([group_starts[1:], torch.tensor([len(sorted_nodes)], device=device)])

        pair_i_list: list[torch.Tensor] = []
        pair_j_list: list[torch.Tensor] = []

        for g in range(len(group_starts)):
            s, e = group_starts[g].item(), group_ends[g].item()
            if e - s < 2:
                continue
            edge_ids = sorted_edge_ids[s:e]
            # All pairs within this group.
            n_local = len(edge_ids)
            idx_i = torch.arange(n_local, device=device)
            idx_j = torch.arange(n_local, device=device)
            grid_i, grid_j = torch.meshgrid(idx_i, idx_j, indexing="ij")
            upper_mask = grid_i < grid_j
            pair_i_list.append(edge_ids[grid_i[upper_mask]])
            pair_j_list.append(edge_ids[grid_j[upper_mask]])

        if len(pair_i_list) == 0:
            return self._empty_result(device)

        pair_i = torch.cat(pair_i_list)
        pair_j = torch.cat(pair_j_list)

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
    ) -> None:
        super().__init__("parallel_pair", initial_weight)
        self.angle_threshold = math.radians(angle_threshold_deg)
        self.iqr_scale = iqr_scale

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

        # Find parallel pairs: edges with similar angles.
        # Chunked to avoid O(N²) memory — process rows in blocks.
        CHUNK = 4096
        ei_list: list[torch.Tensor] = []
        ej_list: list[torch.Tensor] = []
        for start in range(0, num_edges, CHUNK):
            end = min(start + CHUNK, num_edges)
            # angle_diff: (chunk, num_edges) — only rows [start:end]
            diff = (angles[start:end].unsqueeze(1) - angles.unsqueeze(0)).abs()
            diff = torch.min(diff, math.pi - diff)
            # Upper-triangular: only keep pairs where col > row
            row_idx = torch.arange(start, end, device=device).unsqueeze(1)
            col_idx = torch.arange(num_edges, device=device).unsqueeze(0)
            mask = (diff < self.angle_threshold) & (col_idx > row_idx)
            ri, ci = torch.where(mask)
            ei_list.append(ri + start)
            ej_list.append(ci)

        if not ei_list or sum(t.numel() for t in ei_list) == 0:
            return self._empty_result(device)

        ei = torch.cat(ei_list)
        ej = torch.cat(ej_list)

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
    ) -> None:
        super().__init__("non_intersection", initial_weight)
        self.epsilon = epsilon
        self.proximity_threshold = proximity_threshold

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

        positions = node_positions.reshape(-1, 2) if node_positions.dim() == 3 else node_positions

        src, dst = edge_index[0], edge_index[1]

        # Pre-compute per-edge midpoints and half-lengths for spatial filter.
        mid = (positions[src] + positions[dst]) * 0.5  # (E, 2)
        half_len = (positions[dst] - positions[src]).norm(dim=-1) * 0.5

        # Chunked pair finding to avoid O(N²) memory.
        CHUNK = 4096
        ei_list: list[torch.Tensor] = []
        ej_list: list[torch.Tensor] = []
        for start in range(0, num_edges, CHUNK):
            end = min(start + CHUNK, num_edges)
            chunk_src = src[start:end]  # (C,)
            chunk_dst = dst[start:end]

            # Edge adjacency: edges sharing a node (C, E)
            edge_adj = (
                (chunk_src.unsqueeze(1) == src.unsqueeze(0))
                | (chunk_src.unsqueeze(1) == dst.unsqueeze(0))
                | (chunk_dst.unsqueeze(1) == src.unsqueeze(0))
                | (chunk_dst.unsqueeze(1) == dst.unsqueeze(0))
            )

            # Upper-triangular: col > row
            row_idx = torch.arange(start, end, device=device).unsqueeze(1)
            col_idx = torch.arange(num_edges, device=device).unsqueeze(0)
            mask = ~edge_adj & (col_idx > row_idx)

            # Spatial pre-filter: midpoint proximity
            chunk_mid = mid[start:end]  # (C, 2)
            md = (chunk_mid.unsqueeze(1) - mid.unsqueeze(0)).norm(dim=-1)
            chunk_half = half_len[start:end]
            prox = chunk_half.unsqueeze(1) + half_len.unsqueeze(0) + self.proximity_threshold
            mask = mask & (md < prox)

            ri, ci = torch.where(mask)
            ei_list.append(ri + start)
            ej_list.append(ci)

        if not ei_list or sum(t.numel() for t in ei_list) == 0:
            return self._empty_result(device)

        ei = torch.cat(ei_list)
        ej = torch.cat(ej_list)

        # Differentiable segment-segment closest distance.
        min_dists = self._segment_distance(
            positions[src[ei]],
            positions[dst[ei]],
            positions[src[ej]],
            positions[dst[ej]],
        )

        # Loss: ReLU(epsilon - min_distance).
        per_pair_loss = torch.relu(self.epsilon - min_dists)
        loss = (
            per_pair_loss.mean()
            if per_pair_loss.numel() > 0
            else torch.tensor(0.0, device=device, requires_grad=True)
        )

        violation_mask = per_pair_loss > 0

        return AxiomResult(
            name=self.axiom_name,
            loss=loss,
            violation_mask=violation_mask,
            violation_scores=per_pair_loss.detach(),
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

        registry.register(
            OrthogonalIntegrityAxiom(
                tolerance_deg=config.ortho_tolerance_deg,
                initial_weight=weights.get("orthogonal", 1.0),
            )
        )
        registry.register(
            ParallelPairConstancyAxiom(
                iqr_scale=config.parallel_iqr_scale,
                initial_weight=weights.get("parallel_pair", 1.0),
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
            )
        )

        # Freeze weights if meta-learning is disabled.
        if not config.learn_weights:
            for axiom in registry._axioms.values():
                axiom.weight.requires_grad_(False)

        return registry
