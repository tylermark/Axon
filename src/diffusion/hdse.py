"""Hierarchical Distance Structural Encoding (HDSE).

Encodes multi-level graph structure into attention biases for the diffusion
transformer. Combines three complementary structural signals:

1. **Shortest-path distance** — pairwise hop distance via BFS
2. **Random walk similarity** — k-step landing probabilities (D⁻¹A)^k
3. **Hierarchical level** — spectral-informed soft level assignment

The combined encoding biases transformer attention so nearby nodes attend
strongly and the model implicitly learns wall → room → floor → building
hierarchy.

Reference:
    MODEL_SPEC.md  -- §HDSE, §Generative Graph Denoising Diffusion Engine
    ARCHITECTURE.md -- Stage 3: Graph Diffusion Engine
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from src.pipeline.config import DiffusionConfig


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class HDSEOutput:
    """Output of HDSE computation."""

    attention_bias: torch.Tensor
    """Pairwise attention bias, shape (B, n_heads, N, N).

    Added to transformer attention scores before softmax:
    Attn(Q,K) = (QK^T / sqrt(d) + attention_bias).
    """

    node_encodings: torch.Tensor
    """Per-node structural features, shape (B, N, d_model).

    Can be added to node embeddings for additional structural context.
    """


# ---------------------------------------------------------------------------
# Sub-encoding 1: Shortest-Path Distance
# ---------------------------------------------------------------------------


class ShortestPathEncoding(nn.Module):
    """Encode shortest-path distances between all node pairs.

    For each pair (i, j), computes the shortest-path distance d(i,j) in the
    graph via BFS. Distances beyond ``max_distance`` are clipped. Unreachable
    pairs receive a special embedding index. Distances are embedded via a
    learned embedding table.

    Reference: ARCHITECTURE.md §Stage 3 HDSE
    """

    def __init__(self, max_distance: int = 10, n_out: int = 8) -> None:
        super().__init__()
        self.max_distance = max_distance
        # Embed directly to n_out (n_heads) — avoids (B, N, N, d_model).
        self.embedding = nn.Embedding(max_distance + 2, n_out)

    def forward(
        self,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute shortest-path distance encoding.

        Args:
            adjacency: (B, N, N) adjacency matrix (logits or binary).
            node_mask: (B, N) boolean mask for valid nodes.

        Returns:
            (B, N, N, n_out) distance embeddings.
        """
        bsz, n_nodes, _ = adjacency.shape
        device = adjacency.device

        # Threshold to binary for BFS (handles soft adjacency during denoising).
        adj_binary = (adjacency > 0.5).float()
        # Symmetrize.
        adj_binary = torch.clamp(adj_binary + adj_binary.transpose(-1, -2), max=1.0)

        # BFS on CPU for each graph — integer shortest paths are non-differentiable
        # anyway, so this doesn't break gradient flow.
        adj_np = adj_binary.detach().cpu()
        dist_indices = torch.full(
            (bsz, n_nodes, n_nodes),
            self.max_distance + 1,
            dtype=torch.long,
            device=device,
        )

        for b in range(bsz):
            mask_b = node_mask[b] if node_mask is not None else None
            n_valid = int(mask_b.sum().item()) if mask_b is not None else n_nodes
            adj_b = adj_np[b]

            # Build adjacency list for BFS efficiency.
            neighbors: list[list[int]] = [[] for _ in range(n_valid)]
            for i in range(n_valid):
                for j in range(i + 1, n_valid):
                    if adj_b[i, j] > 0.5:
                        neighbors[i].append(j)
                        neighbors[j].append(i)

            # BFS from each valid node.
            for src in range(n_valid):
                dists = [-1] * n_valid
                dists[src] = 0
                queue: deque[int] = deque([src])
                while queue:
                    u = queue.popleft()
                    if dists[u] >= self.max_distance:
                        continue
                    for v in neighbors[u]:
                        if dists[v] == -1:
                            dists[v] = dists[u] + 1
                            queue.append(v)

                for tgt in range(n_valid):
                    d = dists[tgt]
                    if d == -1:
                        dist_indices[b, src, tgt] = self.max_distance + 1
                    else:
                        dist_indices[b, src, tgt] = min(d, self.max_distance)

        return self.embedding(dist_indices)


# ---------------------------------------------------------------------------
# Sub-encoding 2: Random Walk
# ---------------------------------------------------------------------------


class RandomWalkEncoding(nn.Module):
    """Encode random walk landing probabilities between node pairs.

    Captures structural similarity: nodes in the same neighbourhood have
    similar random walk profiles.  Computes k-step random walk probabilities
    ``RW_k(i,j) = (D⁻¹A)^k [i,j]`` for k = 1 .. ``num_steps`` and projects
    each step to ``n_out`` via a learned weight, summing in-place to avoid
    materializing the full (B, N, N, num_steps) stack.

    Reference: ARCHITECTURE.md §Stage 3 HDSE
    """

    def __init__(self, num_steps: int = 8, n_out: int = 8) -> None:
        super().__init__()
        self.num_steps = num_steps
        # Per-step scalar weights: project each (B,N,N) step → n_out channels
        # and accumulate, avoiding (B, N, N, num_steps) intermediate.
        self.step_weights = nn.Parameter(torch.randn(num_steps, n_out) * 0.02)

    def forward(
        self,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute random walk structural encoding.

        Args:
            adjacency: (B, N, N) adjacency matrix (logits or binary).
            node_mask: (B, N) boolean mask for valid nodes.

        Returns:
            (B, N, N, n_out) random walk bias.
        """
        adj = torch.sigmoid(adjacency)
        adj = (adj + adj.transpose(-1, -2)) / 2.0

        if node_mask is not None:
            pair_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
            adj = adj * pair_mask.float()

        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        trans = adj / degree  # (B, N, N)

        # Accumulate weighted sum: bias = sum_k w_k * P^k
        # Only one (B, N, N, n_out) tensor lives at a time.
        trans_k = trans
        bias = trans_k.unsqueeze(-1) * self.step_weights[0]  # (B, N, N, n_out)
        for k in range(1, self.num_steps):
            trans_k = torch.bmm(trans_k, trans)
            bias = bias + trans_k.unsqueeze(-1) * self.step_weights[k]

        return bias


# ---------------------------------------------------------------------------
# Sub-encoding 3: Hierarchical Level
# ---------------------------------------------------------------------------


class HierarchicalLevelEncoding(nn.Module):
    """Encode hierarchical structural level of each node.

    Uses a differentiable soft-assignment approach: learnable level prototype
    vectors attend to local graph features (degree + clustering coefficient)
    to produce per-node level weights. This avoids hard spectral clustering
    and remains differentiable through the denoising loop.

    Hierarchy levels (conceptual):
        - Level 0: Individual wall segments / leaf nodes (low degree)
        - Level 1: Room-bounding structures (moderate degree, high clustering)
        - Level 2: Floor-level corridor nodes (high degree, low clustering)
        - Level 3: Building envelope (boundary nodes)

    Reference: ARCHITECTURE.md §Stage 3 HDSE
    """

    def __init__(self, num_levels: int = 4, d_model: int = 512) -> None:
        super().__init__()
        self.num_levels = num_levels
        # Learnable level embeddings.
        self.level_embeddings = nn.Embedding(num_levels, d_model)
        # Project graph features to attention scores over levels.
        # Features: [degree, clustering_coeff, degree_centrality, second_order_degree]
        self.level_attention = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, num_levels),
        )

    def forward(
        self,
        adjacency: torch.Tensor,
        node_positions: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute hierarchical level encoding.

        Args:
            adjacency: (B, N, N) adjacency matrix.
            node_positions: (B, N, 2) optional node coordinates (unused).
            node_mask: (B, N) boolean mask for valid nodes.

        Returns:
            (B, N, d_model) per-node level embeddings.
        """
        # Sigmoid normalizes logits to [0,1] for stable graph feature computation.
        adj = torch.sigmoid(adjacency)
        # Symmetrize.
        adj = (adj + adj.transpose(-1, -2)) / 2.0

        if node_mask is not None:
            pair_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
            adj = adj * pair_mask.float()

        # --- Compute lightweight graph features ---

        # Degree: sum of adjacency row.
        degree = adj.sum(dim=-1)  # (B, N)
        max_degree = degree.max(dim=-1, keepdim=True).values.clamp(min=1.0)
        degree_norm = degree / max_degree  # (B, N) normalized

        # Clustering coefficient: proportion of connected neighbour pairs.
        # C(v) = 2 * triangles(v) / (deg(v) * (deg(v) - 1))
        # triangles(v) = diag(A³) / 2 = sum_j (A² * A)[v,j] / 2
        # Compute diag(A³) without materializing full A²:
        # diag(A³)[v] = sum_j sum_k A[v,k]*A[k,j]*A[j,v] = einsum('bvk,bkj,bjv->bv')
        triangles = torch.einsum('bik,bkj,bji->bi', adj, adj, adj) / 2.0  # (B, N)
        deg_factor = (degree * (degree - 1)).clamp(min=1.0)
        clustering = 2.0 * triangles / deg_factor  # (B, N)

        # Second-order degree: approximate via degree² (avoids full A² matrix).
        # True 2-hop degree ≈ degree of degree, bounded by actual reachability.
        second_order_degree = degree ** 2
        max_sod = second_order_degree.max(dim=-1, keepdim=True).values.clamp(min=1.0)
        second_order_norm = second_order_degree / max_sod

        # Stack features: (B, N, 4)
        features = torch.stack([degree_norm, clustering, degree_norm, second_order_norm], dim=-1)

        # Soft level assignment via attention: (B, N, num_levels)
        level_logits = self.level_attention(features)
        if node_mask is not None:
            level_logits = level_logits.masked_fill(~node_mask.unsqueeze(-1), -1e9)
        level_weights = torch.softmax(level_logits, dim=-1)

        # Weighted sum of level embeddings: (B, N, d_model)
        # level_embeddings.weight: (num_levels, d_model)
        embeddings = level_weights @ self.level_embeddings.weight  # (B, N, d_model)

        return embeddings


# ---------------------------------------------------------------------------
# Combined HDSE
# ---------------------------------------------------------------------------


class HDSE(nn.Module):
    """Hierarchical Distance Structural Encoding.

    Combines shortest-path distance, random walk similarity, and hierarchical
    level into a unified attention bias added to transformer attention scores:

        ``Attn(Q,K) = (QK^T / sqrt(d) + HDSE_bias)``

    The three sub-encodings capture complementary structural signals:
    shortest-path gives hard hop distance, random walk gives soft
    neighbourhood similarity, and hierarchical level encodes the
    wall → room → floor → building hierarchy.

    Reference:
        MODEL_SPEC.md §HDSE
        ARCHITECTURE.md §Stage 3
    """

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        d_model = config.d_model
        n_heads = config.n_heads

        self.n_heads = n_heads
        self.d_model = d_model

        # Sub-encodings — output n_heads directly to avoid (B, N, N, d_model).
        self.sp_enc = ShortestPathEncoding(
            max_distance=config.hdse_max_distance,
            n_out=n_heads,
        )
        self.rw_enc = RandomWalkEncoding(
            num_steps=8,
            n_out=n_heads,
        )
        self.hier_enc = HierarchicalLevelEncoding(
            num_levels=4,
            d_model=d_model,
        )

        # Hierarchical is node-level (B, N, d_model) — project to n_heads
        # and broadcast via outer sum (no N² expansion needed).
        self.hier_proj = nn.Linear(d_model, n_heads)

        # Node-level projection for structural features.
        self.node_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        adjacency: torch.Tensor,
        node_positions: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> HDSEOutput:
        """Compute combined HDSE attention bias.

        Args:
            adjacency: (B, N, N) adjacency matrix (logits or binary).
            node_positions: (B, N, 2) optional node coordinates.
            node_mask: (B, N) boolean mask for valid nodes.

        Returns:
            HDSEOutput with pairwise attention bias (B, n_heads, N, N)
            and node-level encodings (B, N, d_model).
        """
        # 1. Shortest-path: (B, N, N, n_heads) — discrete, non-differentiable.
        sp = self.sp_enc(adjacency, node_mask)

        # 2. Random walk: (B, N, N, n_heads) — differentiable through soft adj.
        rw = self.rw_enc(adjacency, node_mask)

        # 3. Hierarchical level: (B, N, d_model).
        hier_node = self.hier_enc(adjacency, node_positions, node_mask)

        # 4. Fuse: SP and RW already output n_heads, just sum.
        bias = sp + rw  # (B, N, N, n_heads)

        # Add hierarchical: project node-level, then broadcast via outer sum.
        hier_bias_i = self.hier_proj(hier_node)  # (B, N, n_heads)
        bias = bias + hier_bias_i.unsqueeze(2) + hier_bias_i.unsqueeze(1)

        bias = bias.permute(0, 3, 1, 2)  # (B, n_heads, N, N)

        # Mask invalid pairs.
        if node_mask is not None:
            pair_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)  # (B, N, N)
            bias = bias.masked_fill(~pair_mask.unsqueeze(1), 0.0)

        # 5. Node-level encodings from hierarchical level.
        node_encodings = self.node_proj(hier_node)  # (B, N, d_model)

        return HDSEOutput(attention_bias=bias, node_encodings=node_encodings)
