"""Tests for ShortestPathEncoding in src/diffusion/hdse.py.

Covers:
  - Correctness: new vectorized implementation matches a reference Python BFS
    across several graph topologies (chain, cycle, disconnected, masked node).
  - Sanity microbench: N=256, B=4, max_distance=4 must complete in <1 s on CPU.
"""

from __future__ import annotations

import time
from collections import deque

import pytest
import torch

from src.diffusion.hdse import ShortestPathEncoding


# ---------------------------------------------------------------------------
# Reference BFS implementation (the old logic, kept here as ground-truth).
# ---------------------------------------------------------------------------


def _bfs_dist_matrix(adj_bin: torch.Tensor, max_distance: int) -> torch.Tensor:
    """Pure-Python BFS shortest-path distances.  Returns (N, N) long tensor.

    adj_bin: (N, N) binary symmetric adjacency (CPU, bool or 0/1 float).
    Unreachable pairs get max_distance+1.
    """
    n = adj_bin.shape[0]
    UNREACHABLE = max_distance + 1
    dist = torch.full((n, n), UNREACHABLE, dtype=torch.long)

    # Build adjacency list.
    adj_np = adj_bin.float().numpy()
    neighbors: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and adj_np[i, j] > 0.5:
                neighbors[i].append(j)

    for src in range(n):
        dists_row = [-1] * n
        dists_row[src] = 0
        queue: deque[int] = deque([src])
        while queue:
            u = queue.popleft()
            if dists_row[u] >= max_distance:
                continue
            for v in neighbors[u]:
                if dists_row[v] == -1:
                    dists_row[v] = dists_row[u] + 1
                    queue.append(v)
        for tgt in range(n):
            d = dists_row[tgt]
            dist[src, tgt] = UNREACHABLE if d == -1 else min(d, max_distance)

    return dist


# ---------------------------------------------------------------------------
# Helper to build the raw dist_indices tensor from ShortestPathEncoding.
# The module returns embeddings; we need to inspect the integer indices.
# We do this by monkey-patching the embedding to be an identity that
# preserves the index value as a float (just for testing).
# ---------------------------------------------------------------------------


def _get_dist_indices(
    enc: ShortestPathEncoding,
    adjacency: torch.Tensor,
    node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the encoder and recover integer dist_indices via inverse embedding.

    Since embedding(i) = weight[i], we can find the index by argmin distance
    to the embedding table rows — but that is fragile for random weights.

    Simpler: temporarily replace the embedding with a known lookup and recover
    the index directly.  We replace weight[i] = i (scalar) and round the output.
    """
    orig_weight = enc.embedding.weight.data.clone()
    n_out = enc.embedding.weight.shape[1]

    # Set embedding weight so embedding(i) produces a constant vector of value i.
    # Shape: (max_distance+2, n_out).
    ids = torch.arange(enc.max_distance + 2, dtype=torch.float32).unsqueeze(1)
    enc.embedding.weight.data = ids.expand(-1, n_out).clone()

    with torch.no_grad():
        out = enc(adjacency, node_mask)  # (B, N, N, n_out)

    # Restore.
    enc.embedding.weight.data = orig_weight

    # out[b,i,j,:] is all the same value = dist_index[b,i,j], so take mean.
    indices = out.mean(dim=-1).round().long()  # (B, N, N)
    return indices


# ---------------------------------------------------------------------------
# Graph factory helpers (return (N, N) binary symmetric tensors).
# ---------------------------------------------------------------------------


def _chain(n: int) -> torch.Tensor:
    """0-1-2-...(n-1) linear chain."""
    a = torch.zeros(n, n, dtype=torch.float32)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def _cycle(n: int) -> torch.Tensor:
    """0-1-2-...-(n-1)-0 cycle."""
    a = _chain(n)
    a[0, n - 1] = a[n - 1, 0] = 1.0
    return a


def _two_components(n: int) -> torch.Tensor:
    """Two disjoint chains of length n//2 each (disconnected pair)."""
    a = torch.zeros(n, n, dtype=torch.float32)
    half = n // 2
    for i in range(half - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    for i in range(half, n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def _single_isolated_node(n: int) -> torch.Tensor:
    """Chain of n-1 nodes, plus node n-1 isolated (no edges)."""
    a = _chain(n - 1)
    # Pad to n×n.
    full = torch.zeros(n, n, dtype=torch.float32)
    full[: n - 1, : n - 1] = a
    return full


# ---------------------------------------------------------------------------
# Correctness tests.
# ---------------------------------------------------------------------------


MAX_DISTANCE = 6


@pytest.fixture
def enc() -> ShortestPathEncoding:
    return ShortestPathEncoding(max_distance=MAX_DISTANCE, n_out=4)


def _assert_matches_bfs(
    enc: ShortestPathEncoding,
    adj: torch.Tensor,
    node_mask: torch.Tensor | None = None,
    label: str = "",
) -> None:
    """Assert new implementation == BFS reference for a single graph."""
    n = adj.shape[0]
    # Batch-of-1.
    adjacency = adj.unsqueeze(0)  # (1, N, N)
    mask = node_mask.unsqueeze(0) if node_mask is not None else None

    got = _get_dist_indices(enc, adjacency, mask)[0]  # (N, N)

    # BFS reference on valid nodes only.
    if node_mask is not None:
        valid_idx = node_mask.nonzero(as_tuple=False).squeeze(-1)
        n_valid = valid_idx.shape[0]
        # Extract sub-adjacency for valid nodes.
        sub_adj = adj[valid_idx][:, valid_idx]
        ref_sub = _bfs_dist_matrix(sub_adj, MAX_DISTANCE)

        # Build expected full matrix — invalid entries = UNREACHABLE.
        UNREACHABLE = MAX_DISTANCE + 1
        expected = torch.full((n, n), UNREACHABLE, dtype=torch.long)
        for ii, i in enumerate(valid_idx.tolist()):
            for jj, j in enumerate(valid_idx.tolist()):
                expected[i, j] = ref_sub[ii, jj]
    else:
        expected = _bfs_dist_matrix(adj, MAX_DISTANCE)

    assert torch.equal(got, expected), (
        f"[{label}] dist_indices mismatch.\n"
        f"Got:\n{got}\nExpected:\n{expected}"
    )


def test_chain(enc: ShortestPathEncoding) -> None:
    """Chain of 8 nodes — distances are linear hop counts."""
    _assert_matches_bfs(enc, _chain(8), label="chain-8")


def test_cycle(enc: ShortestPathEncoding) -> None:
    """Cycle of 8 — distance from 0 to 4 is 4 (half-way), not 8."""
    _assert_matches_bfs(enc, _cycle(8), label="cycle-8")


def test_two_disconnected_components(enc: ShortestPathEncoding) -> None:
    """Two disconnected chains — cross-component distance = UNREACHABLE."""
    _assert_matches_bfs(enc, _two_components(8), label="two-components-8")


def test_masked_isolated_node(enc: ShortestPathEncoding) -> None:
    """Node 7 is both structurally isolated and masked out.

    All pairs involving node 7 must be UNREACHABLE regardless.
    """
    n = 8
    adj = _single_isolated_node(n)
    # Mask: nodes 0..6 valid, node 7 invalid.
    mask = torch.ones(n, dtype=torch.bool)
    mask[7] = False
    _assert_matches_bfs(enc, adj, node_mask=mask, label="isolated-masked")


def test_masked_node_in_connected_graph(enc: ShortestPathEncoding) -> None:
    """Mask out a middle node in a chain; path through that node = UNREACHABLE."""
    n = 8
    adj = _chain(n)
    # Mask node 3 invalid — this severs the chain.
    mask = torch.ones(n, dtype=torch.bool)
    mask[3] = False
    _assert_matches_bfs(enc, adj, node_mask=mask, label="chain-severed-at-3")


def test_batch_consistency(enc: ShortestPathEncoding) -> None:
    """Batching N graphs together gives same result as processing individually."""
    n = 8
    graphs = [_chain(n), _cycle(n), _two_components(n)]
    adjacency = torch.stack(graphs)  # (3, N, N)

    got_batch = _get_dist_indices(enc, adjacency)  # (3, N, N)

    for b, adj in enumerate(graphs):
        got_single = _get_dist_indices(enc, adj.unsqueeze(0))[0]
        assert torch.equal(got_batch[b], got_single), (
            f"Batch item {b} differs from single-graph result."
        )


def test_self_distance_is_zero(enc: ShortestPathEncoding) -> None:
    """dist[i,i] == 0 for every valid node."""
    adjacency = _chain(8).unsqueeze(0)
    got = _get_dist_indices(enc, adjacency)[0]
    assert (got.diagonal() == 0).all(), "Self-distances are not zero."


def test_symmetry(enc: ShortestPathEncoding) -> None:
    """dist[i,j] == dist[j,i] for all pairs (undirected graph)."""
    adjacency = _cycle(8).unsqueeze(0)
    got = _get_dist_indices(enc, adjacency)[0]
    assert torch.equal(got, got.t()), "Distance matrix is not symmetric."


# ---------------------------------------------------------------------------
# Sanity microbench.
# ---------------------------------------------------------------------------


def test_microbench_cpu_under_1s() -> None:
    """N=256, B=4, max_distance=4: new forward pass must complete in <1 s on CPU."""
    enc = ShortestPathEncoding(max_distance=4, n_out=8)
    enc.eval()

    torch.manual_seed(0)
    adjacency = (torch.rand(4, 256, 256) > 0.85).float()
    # Sparsify to ~15% density — realistic floor-plan graph.

    # Warm-up (compile caches etc.).
    with torch.no_grad():
        _ = enc(adjacency)

    # Timed run.
    start = time.perf_counter()
    with torch.no_grad():
        _ = enc(adjacency)
    elapsed = time.perf_counter() - start

    # NOTE(diffuse): 1 s is deliberately loose for CPU. On a 48 GB GPU the same
    # call should take <50 ms. The bound here just guards against regression to
    # the old O(B*N^3) Python loop, which took >60 s for these dimensions.
    assert elapsed < 1.0, (
        f"ShortestPathEncoding forward too slow on CPU: {elapsed:.3f}s >= 1.0s"
    )
