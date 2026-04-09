"""Tests for ``find_nearby_edge_pairs`` — vectorised batch-aware rewrite.

Covers correctness against a brute-force O(E²) reference, batch-awareness
(cross-batch pairs are excluded), edge cases (empty, single edge, shared
nodes), and a microbench that guards against a regression to the Python
dict-bucketing implementation.
"""

from __future__ import annotations

import time

import torch

from src.constraints.axioms import find_nearby_edge_pairs


# ---------------------------------------------------------------------------
# Brute-force reference
# ---------------------------------------------------------------------------


def _brute_force_nearby_pairs(
    edge_index: torch.Tensor,
    positions: torch.Tensor,
    threshold: float,
    batch_index: torch.Tensor | None = None,
) -> set[tuple[int, int]]:
    """Reference implementation — O(E²) Python loop, correctness ground truth."""
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    pos = positions.tolist()
    bi = batch_index.tolist() if batch_index is not None else None
    num_edges = len(src)

    pairs: set[tuple[int, int]] = set()
    for i in range(num_edges):
        for j in range(i + 1, num_edges):
            if bi is not None and bi[i] != bi[j]:
                continue
            # Non-adjacent: must not share a node.
            if {src[i], dst[i]} & {src[j], dst[j]}:
                continue
            mid_i = (
                (pos[src[i]][0] + pos[dst[i]][0]) / 2,
                (pos[src[i]][1] + pos[dst[i]][1]) / 2,
            )
            mid_j = (
                (pos[src[j]][0] + pos[dst[j]][0]) / 2,
                (pos[src[j]][1] + pos[dst[j]][1]) / 2,
            )
            d = (
                (mid_i[0] - mid_j[0]) ** 2 + (mid_i[1] - mid_j[1]) ** 2
            ) ** 0.5
            half_i = (
                (pos[dst[i]][0] - pos[src[i]][0]) ** 2
                + (pos[dst[i]][1] - pos[src[i]][1]) ** 2
            ) ** 0.5 / 2
            half_j = (
                (pos[dst[j]][0] - pos[src[j]][0]) ** 2
                + (pos[dst[j]][1] - pos[src[j]][1]) ** 2
            ) ** 0.5 / 2
            reach = half_i + half_j + threshold
            if d < reach:
                pairs.add((i, j))
    return pairs


def _vectorized_to_set(
    ei: torch.Tensor, ej: torch.Tensor
) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j in zip(ei.tolist(), ej.tolist())}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_edge_index() -> None:
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    positions = torch.zeros(0, 2)
    ei, ej = find_nearby_edge_pairs(edge_index, positions, proximity_threshold=0.1)
    assert ei.numel() == 0 and ej.numel() == 0


def test_single_edge() -> None:
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    ei, ej = find_nearby_edge_pairs(edge_index, positions, proximity_threshold=0.1)
    assert ei.numel() == 0


def test_two_nearby_non_adjacent_edges() -> None:
    """Two parallel segments at distance 0.05 — should pair."""
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.05],
            [1.0, 0.05],
        ]
    )
    edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    ei, ej = find_nearby_edge_pairs(edge_index, positions, proximity_threshold=0.1)
    assert _vectorized_to_set(ei, ej) == {(0, 1)}


def test_two_adjacent_edges_excluded() -> None:
    """Edges sharing a node must not be returned."""
    positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # share node 1
    ei, ej = find_nearby_edge_pairs(edge_index, positions, proximity_threshold=1.0)
    assert ei.numel() == 0


def test_far_apart_edges_excluded() -> None:
    """Edges further apart than threshold must not be returned."""
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ]
    )
    edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    ei, ej = find_nearby_edge_pairs(edge_index, positions, proximity_threshold=0.1)
    assert ei.numel() == 0


def test_output_ordering() -> None:
    """Output pairs must satisfy ei < ej."""
    torch.manual_seed(0)
    positions = torch.rand(20, 2)
    edge_index = torch.tensor(
        [[0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
         [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]],
        dtype=torch.long,
    )
    ei, ej = find_nearby_edge_pairs(edge_index, positions, proximity_threshold=2.0)
    assert (ei < ej).all()


# ---------------------------------------------------------------------------
# Equivalence with brute force on random inputs
# ---------------------------------------------------------------------------


def test_equivalence_small_random() -> None:
    torch.manual_seed(7)
    for trial in range(5):
        n_nodes = 12
        positions = torch.rand(n_nodes, 2)
        # Random non-parallel edge set with ~8 edges, no self-loops, no dups.
        edges = set()
        while len(edges) < 8:
            i, j = torch.randint(0, n_nodes, (2,)).tolist()
            if i != j:
                edges.add((min(i, j), max(i, j)))
        edge_list = list(edges)
        edge_index = torch.tensor(
            [[e[0] for e in edge_list], [e[1] for e in edge_list]],
            dtype=torch.long,
        )

        threshold = 0.15
        ei, ej = find_nearby_edge_pairs(
            edge_index, positions, proximity_threshold=threshold
        )
        got = _vectorized_to_set(ei, ej)
        expected = _brute_force_nearby_pairs(
            edge_index, positions, threshold
        )
        assert got == expected, f"trial={trial} got={got} expected={expected}"


# ---------------------------------------------------------------------------
# Batch-awareness
# ---------------------------------------------------------------------------


def test_batch_index_excludes_cross_batch_pairs() -> None:
    """Two batch items whose edges overlap in xy space must not produce
    cross-batch pairs when batch_index is supplied."""
    # Batch 0: edges [0-1], [2-3] that would be nearby in xy.
    # Batch 1: edges [4-5], [6-7] at the SAME xy coordinates as batch 0.
    positions = torch.tensor(
        [
            # Batch 0
            [0.0, 0.0], [1.0, 0.0],
            [0.0, 0.05], [1.0, 0.05],
            # Batch 1
            [0.0, 0.0], [1.0, 0.0],
            [0.0, 0.05], [1.0, 0.05],
        ]
    )
    edge_index = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]], dtype=torch.long)
    batch_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    ei, ej = find_nearby_edge_pairs(
        edge_index, positions, proximity_threshold=0.1, batch_index=batch_index
    )
    # Expected: only (0,1) [batch 0 pair] and (2,3) [batch 1 pair].
    # NOT (0,2), (0,3), (1,2), (1,3) — those would be cross-batch.
    got = _vectorized_to_set(ei, ej)
    assert got == {(0, 1), (2, 3)}, f"got {got}"


def test_batch_index_none_matches_single_batch() -> None:
    """Passing batch_index=None must behave identically to a single-batch
    call on the same inputs."""
    torch.manual_seed(3)
    positions = torch.rand(10, 2)
    edge_index = torch.tensor(
        [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]], dtype=torch.long
    )

    ei1, ej1 = find_nearby_edge_pairs(
        edge_index, positions, proximity_threshold=0.3
    )
    ei2, ej2 = find_nearby_edge_pairs(
        edge_index,
        positions,
        proximity_threshold=0.3,
        batch_index=torch.zeros(5, dtype=torch.long),
    )
    assert _vectorized_to_set(ei1, ej1) == _vectorized_to_set(ei2, ej2)


# ---------------------------------------------------------------------------
# Microbench — regression guard against the Python-bucket implementation
# ---------------------------------------------------------------------------


def test_microbench_scales_subquadratic_python_time() -> None:
    """At E=2000 on CPU the vectorised implementation must finish in well
    under one second. The old Python-dict bucketing degraded to O(E²)
    Python iterations with ``.item()`` syncs when positions were
    normalised, blowing past minutes at this scale.

    This is a loose regression guard — an order-of-magnitude check, not a
    strict benchmark.
    """
    torch.manual_seed(0)
    n_nodes = 4000
    n_edges = 2000
    positions = torch.rand(n_nodes, 2)
    # Random pairs of distinct node indices.
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index = torch.stack([src, dst])

    t0 = time.perf_counter()
    ei, ej = find_nearby_edge_pairs(
        edge_index, positions, proximity_threshold=0.05
    )
    elapsed = time.perf_counter() - t0
    # Must complete in well under a second on CPU.
    assert elapsed < 1.0, (
        f"find_nearby_edge_pairs took {elapsed*1000:.1f} ms "
        f"on E={edge_index.shape[1]} — regression to Python bucketing?"
    )
    # Sanity: returned pair indices must satisfy ei < ej.
    if ei.numel() > 0:
        assert (ei < ej).all()


# ---------------------------------------------------------------------------
# max_pairs cap — guards the OOM fix for pathological batches
# ---------------------------------------------------------------------------


def test_max_pairs_caps_output() -> None:
    """A dense point cloud produces many nearby pairs; max_pairs must cap."""
    torch.manual_seed(11)
    # Clustered points → many midpoints within the proximity threshold.
    n_nodes = 200
    n_edges = 150
    positions = torch.rand(n_nodes, 2) * 0.1  # tight cluster
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index = torch.stack([src, dst])

    ei_full, ej_full = find_nearby_edge_pairs(
        edge_index, positions, proximity_threshold=0.2
    )
    # Cap to half the full enumeration so the cap is actually exercised.
    assert ei_full.numel() >= 2, "set up a denser test — need > 2 pairs"
    cap = ei_full.numel() // 2

    ei, ej = find_nearby_edge_pairs(
        edge_index, positions, proximity_threshold=0.2, max_pairs=cap
    )
    assert ei.numel() == cap
    assert (ei < ej).all()
    full_set = {(int(a), int(b)) for a, b in zip(ei_full, ej_full)}
    cap_set = {(int(a), int(b)) for a, b in zip(ei, ej)}
    assert cap_set <= full_set


def test_max_pairs_noop_when_under_cap() -> None:
    """max_pairs must be a no-op when the full count is below the cap."""
    torch.manual_seed(12)
    n_nodes = 20
    n_edges = 10
    positions = torch.rand(n_nodes, 2)
    src = torch.arange(n_edges)
    dst = torch.arange(n_edges) + 10
    edge_index = torch.stack([src, dst])

    ei_uncapped, ej_uncapped = find_nearby_edge_pairs(
        edge_index, positions, proximity_threshold=0.5
    )
    ei_capped, ej_capped = find_nearby_edge_pairs(
        edge_index, positions, proximity_threshold=0.5, max_pairs=10_000
    )
    uncapped = {(int(a), int(b)) for a, b in zip(ei_uncapped, ej_uncapped)}
    capped = {(int(a), int(b)) for a, b in zip(ei_capped, ej_capped)}
    assert uncapped == capped
