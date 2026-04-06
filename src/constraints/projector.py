"""Hard geometric projection (snapping) for inference.

Applied at the final denoising steps to produce exact geometry:

- Near-orthogonal angles -> exact 90 degrees
- Near-parallel pairs -> exact parallel with uniform thickness
- Dangling endpoints -> connected to nearest junction
- Intersecting segments -> separated

The projector is non-differentiable (inference only) and operates under
``@torch.no_grad``.

Reference:
    ARCHITECTURE.md  Stage 4 Key Design Decisions
    MODEL_SPEC.md  Table 2
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from src.constraints.axioms import (
    compute_edge_angles,
    compute_edge_directions,
    edges_from_adjacency,
    find_nearby_edge_pairs,
    find_parallel_pairs,
)

if TYPE_CHECKING:
    from docs.interfaces.constraint_signals import AxiomResult
    from src.pipeline.config import ConstraintConfig


class GeometricProjector:
    """Hard geometric projection (snapping) for inference.

    Applies deterministic geometric corrections to produce exact wall
    geometry from the soft predictions of the diffusion model.

    This class has no learnable parameters and does not inherit from
    ``nn.Module``.

    Args:
        config: Constraint configuration providing snap tolerances.
    """

    def __init__(self, config: ConstraintConfig) -> None:
        self.config = config

    @torch.no_grad()
    def project(
        self,
        node_positions: torch.Tensor,
        adjacency: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        axiom_results: list[AxiomResult] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply all hard geometric projections.

        Projection order:
        1. Snap orthogonal angles
        2. Snap parallel pairs to uniform thickness
        3. Close dangling junctions
        4. Resolve segment intersections

        Args:
            node_positions: Junction coordinates, ``(B, N, 2)``.
            adjacency: Adjacency logits or binary, ``(B, N, N)``.
            edge_index: COO edge indices, ``(2, E)``.
            node_mask: Valid-node mask, ``(B, N)`` bool.
            axiom_results: Pre-computed axiom results (used for parallel
                pair detection if available).

        Returns:
            ``(projected_positions, projected_adjacency)`` with exact
            geometric corrections applied.
        """
        positions = node_positions.clone()

        # Binarise adjacency.
        adj = (
            (torch.sigmoid(adjacency) > 0.5).float()
            if adjacency.requires_grad
            else (adjacency > 0.5).float()
        )

        if edge_index.shape[1] == 0:
            edge_index = edges_from_adjacency(adj, node_mask)

        # 1. Orthogonal snap.
        positions = self.snap_orthogonal(
            positions,
            edge_index,
            tolerance_deg=self.config.ortho_tolerance_deg,
        )

        # 2. Parallel pair snap.
        positions = self._snap_parallel_from_edges(
            positions,
            edge_index,
            axiom_results,
        )

        # 3. Junction closure.
        positions, adj = self.close_junctions(
            positions,
            adj,
            node_mask,
            min_degree=self.config.junction_min_degree,
        )

        # 4. Resolve intersections.
        # Re-derive edge_index after adjacency may have changed.
        edge_index = edges_from_adjacency(adj, node_mask)
        positions = self.resolve_intersections(positions, edge_index, node_mask)

        return positions, adj

    # ------------------------------------------------------------------
    # Sub-operations
    # ------------------------------------------------------------------

    def snap_orthogonal(
        self,
        node_positions: torch.Tensor,
        edge_index: torch.Tensor,
        tolerance_deg: float = 5.0,
    ) -> torch.Tensor:
        """Snap near-orthogonal edge angles to exact 90 or 0 degrees.

        For each pair of adjacent edges meeting at a shared node, if their
        angle is within ``tolerance_deg`` of 0, 90, or 180 degrees, the
        downstream endpoint is rotated to achieve the exact target angle.

        Args:
            node_positions: ``(B, N, 2)`` or ``(N, 2)``.
            edge_index: ``(2, E)`` COO.
            tolerance_deg: Angular tolerance in degrees.

        Returns:
            Adjusted node positions with the same shape as input.
        """
        positions = node_positions.clone()
        flat = positions.reshape(-1, 2)
        num_edges = edge_index.shape[1]

        if num_edges < 2:
            return positions

        tolerance_rad = math.radians(tolerance_deg)
        src, dst = edge_index[0], edge_index[1]

        # Group edges by shared node (junction).
        all_nodes = torch.cat([src, dst])
        # Track which end of the edge is the shared node.
        # For src entries the shared node is src[e], free node is dst[e].
        # For dst entries the shared node is dst[e], free node is src[e].
        all_free = torch.cat([dst, src])

        sorted_order = torch.argsort(all_nodes)
        sorted_nodes = all_nodes[sorted_order]
        sorted_free = all_free[sorted_order]

        change = torch.cat(
            [
                torch.tensor([True], device=positions.device),
                sorted_nodes[1:] != sorted_nodes[:-1],
            ]
        )
        group_starts = torch.where(change)[0]
        group_ends = torch.cat(
            [
                group_starts[1:],
                torch.tensor([len(sorted_nodes)], device=positions.device),
            ]
        )

        for g in range(len(group_starts)):
            s, e = group_starts[g].item(), group_ends[g].item()
            if e - s < 2:
                continue
            junction_node = sorted_nodes[s].item()
            free_nodes = sorted_free[s:e]
            junction_pos = flat[junction_node]

            for i in range(len(free_nodes)):
                for j in range(i + 1, len(free_nodes)):
                    fi = free_nodes[i].item()
                    fj = free_nodes[j].item()

                    d_i = flat[fi] - junction_pos
                    d_j = flat[fj] - junction_pos

                    len_i = d_i.norm().item()
                    len_j = d_j.norm().item()
                    if len_i < 1e-8 or len_j < 1e-8:
                        continue

                    # Angle between the two edges.
                    cos_val = (d_i @ d_j).item() / (len_i * len_j)
                    cos_val = max(-1.0, min(1.0, cos_val))
                    angle = math.acos(cos_val)

                    # Check proximity to 0, 90, 180.
                    targets = [0.0, math.pi / 2, math.pi]
                    best_target = min(targets, key=lambda t: abs(angle - t))
                    delta = angle - best_target

                    if abs(delta) > tolerance_rad or abs(delta) < 1e-8:
                        continue

                    # Rotate the second edge endpoint by -delta around the junction.
                    cos_r = math.cos(-delta)
                    sin_r = math.sin(-delta)
                    rel = flat[fj] - junction_pos
                    rotated = torch.tensor(
                        [
                            cos_r * rel[0].item() - sin_r * rel[1].item(),
                            sin_r * rel[0].item() + cos_r * rel[1].item(),
                        ],
                        device=flat.device,
                        dtype=flat.dtype,
                    )
                    flat[fj] = junction_pos + rotated

        return flat.reshape(node_positions.shape)

    def snap_parallel_pairs(
        self,
        node_positions: torch.Tensor,
        parallel_pairs: torch.Tensor,
        edge_index: torch.Tensor,
        target_thickness: float | None = None,
    ) -> torch.Tensor:
        """Snap parallel wall pairs to uniform thickness.

        For each identified parallel pair, the two edges are adjusted to
        be exactly parallel with uniform perpendicular distance.

        Args:
            node_positions: ``(B, N, 2)`` or ``(N, 2)``.
            parallel_pairs: ``(N_pairs, 2)`` edge index pairs.
            edge_index: ``(2, E)`` COO.
            target_thickness: If given, force this distance; otherwise
                use the median observed distance.

        Returns:
            Adjusted node positions.
        """
        if parallel_pairs.numel() == 0:
            return node_positions

        positions = node_positions.clone()
        flat = positions.reshape(-1, 2)
        src, dst = edge_index[0], edge_index[1]

        # Compute current distances and median.
        directions = compute_edge_directions(flat.unsqueeze(0), edge_index)
        ei_idx = parallel_pairs[:, 0]
        ej_idx = parallel_pairs[:, 1]

        mid_j = (flat[src[ej_idx]] + flat[dst[ej_idx]]) * 0.5
        d_i = directions[ei_idx]
        normal = torch.stack([-d_i[:, 1], d_i[:, 0]], dim=-1)
        delta = mid_j - flat[src[ei_idx]]
        dists = (delta * normal).sum(dim=-1)

        if target_thickness is None:
            target_thickness_val = dists.abs().median().item()
        else:
            target_thickness_val = target_thickness

        if target_thickness_val < 1e-8:
            return positions

        # Adjust edge j to be exactly target_thickness away from edge i.
        for p in range(parallel_pairs.shape[0]):
            ej = ej_idx[p].item()

            n_vec = normal[p]
            current_dist = dists[p].item()

            # Desired offset from edge i midline.
            sign = 1.0 if current_dist >= 0 else -1.0
            desired_offset = sign * target_thickness_val
            correction = desired_offset - current_dist

            # Shift both endpoints of edge j.
            shift = n_vec * correction
            flat[src[ej]] = flat[src[ej]] + shift
            flat[dst[ej]] = flat[dst[ej]] + shift

        return flat.reshape(node_positions.shape)

    def close_junctions(
        self,
        node_positions: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        min_degree: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Connect dangling endpoints to nearest junction.

        Nodes with degree < ``min_degree`` are connected to the nearest
        valid node that is not already a neighbour.

        Args:
            node_positions: ``(B, N, 2)``.
            adjacency: Binary adjacency, ``(B, N, N)``.
            node_mask: ``(B, N)`` bool.
            min_degree: Minimum acceptable degree.

        Returns:
            ``(positions, modified_adjacency)`` with new edges added.
        """
        positions = node_positions.clone()
        adj = adjacency.clone()

        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
            positions = positions.unsqueeze(0) if positions.dim() == 2 else positions

        batch_size, n, _ = adj.shape

        for b in range(batch_size):
            degree = adj[b].sum(dim=-1)  # (N,)

            valid = torch.ones(n, dtype=torch.bool, device=adj.device)
            if node_mask is not None:
                valid = node_mask[b]

            dangling = (degree < min_degree) & valid

            if not dangling.any():
                continue

            dangling_idx = torch.where(dangling)[0]

            for d in dangling_idx:
                d_val = d.item()
                pos_d = positions[b, d_val]

                # Candidate targets: valid, not self, not already connected.
                candidates = valid.clone()
                candidates[d_val] = False
                candidates = candidates & (adj[b, d_val] < 0.5)

                if not candidates.any():
                    continue

                cand_idx = torch.where(candidates)[0]
                dists = (positions[b, cand_idx] - pos_d).norm(dim=-1)
                nearest = cand_idx[dists.argmin()]

                # Add edge (symmetric).
                adj[b, d_val, nearest] = 1.0
                adj[b, nearest, d_val] = 1.0

        out_adj = adj.squeeze(0) if adjacency.dim() == 2 else adj
        out_pos = positions.squeeze(0) if node_positions.dim() == 2 else positions
        return out_pos, out_adj

    def resolve_intersections(
        self,
        node_positions: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Separate overlapping wall segments by nudging node positions.

        For non-adjacent edges that geometrically intersect, endpoints
        are nudged apart along the edge normals by a small epsilon.

        Args:
            node_positions: ``(B, N, 2)`` or ``(N, 2)``.
            edge_index: ``(2, E)`` COO.
            node_mask: Valid-node mask.

        Returns:
            Adjusted node positions.
        """
        positions = node_positions.clone()
        flat = positions.reshape(-1, 2)
        num_edges = edge_index.shape[1]

        if num_edges < 2:
            return positions

        src, dst = edge_index[0], edge_index[1]

        # Find nearby non-adjacent edge pairs via spatial bucketing — O(E).
        ei, ej = find_nearby_edge_pairs(edge_index, flat, 0.15)
        if ei.numel() == 0:
            return positions

        # Check intersection via parametric line-segment test.
        a1 = flat[src[ei]]
        a2 = flat[dst[ei]]
        b1 = flat[src[ej]]
        b2 = flat[dst[ej]]

        d1 = a2 - a1
        d2 = b2 - b1
        r = b1 - a1

        cross_d = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        # Skip near-parallel pairs.
        valid = cross_d.abs() > 1e-10

        if not valid.any():
            return positions

        t = (r[:, 0] * d2[:, 1] - r[:, 1] * d2[:, 0]) / cross_d.clamp(min=1e-12)
        u = (r[:, 0] * d1[:, 1] - r[:, 1] * d1[:, 0]) / cross_d.clamp(min=1e-12)

        eps = 1e-3
        intersecting = valid & (t > eps) & (t < 1 - eps) & (u > eps) & (u < 1 - eps)

        if not intersecting.any():
            return positions

        # Nudge endpoints of intersecting edge j along its normal.
        nudge_amount = 0.005
        int_idx = torch.where(intersecting)[0]

        for idx in int_idx:
            ej_val = ej[idx].item()
            direction = d2[idx]
            length = direction.norm().item()
            if length < 1e-8:
                continue
            normal = (
                torch.tensor(
                    [-direction[1].item(), direction[0].item()],
                    device=flat.device,
                    dtype=flat.dtype,
                )
                / length
            )
            nudge = normal * nudge_amount
            flat[src[ej_val]] = flat[src[ej_val]] + nudge
            flat[dst[ej_val]] = flat[dst[ej_val]] + nudge

        return flat.reshape(node_positions.shape)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snap_parallel_from_edges(
        self,
        node_positions: torch.Tensor,
        edge_index: torch.Tensor,
        axiom_results: list[AxiomResult] | None,
    ) -> torch.Tensor:
        """Detect parallel pairs and snap them."""
        if edge_index.shape[1] < 2:
            return node_positions

        directions = compute_edge_directions(node_positions, edge_index)
        angles = compute_edge_angles(directions)
        num_edges = edge_index.shape[1]
        device = node_positions.device

        angle_threshold = math.radians(self.config.ortho_tolerance_deg)

        # O(E log E) parallel pair detection via sort + scan.
        ei, ej = find_parallel_pairs(angles, angle_threshold)
        if ei.numel() == 0:
            return node_positions

        parallel_pairs = torch.stack([ei, ej], dim=1)
        return self.snap_parallel_pairs(
            node_positions,
            parallel_pairs,
            edge_index,
        )
