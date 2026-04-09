"""Tests for the vectorized find_parallel_pairs (Bug B fix).

Verifies that the optimised implementation (tolist() + no O(P) list scan)
produces the same set of parallel pairs as the old loop-with-.item() version
on small synthetic graphs.
"""

from __future__ import annotations

import math

import torch

from src.constraints.axioms import find_parallel_pairs


# ---------------------------------------------------------------------------
# Reference implementation — the old version with per-element .item() calls
# and the O(P) `pair not in ei_list` guard.
# ---------------------------------------------------------------------------


def _find_parallel_pairs_old(
    angles: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verbatim copy of the pre-fix implementation for reference comparison."""
    device = angles.device
    num_edges = angles.shape[0]
    if num_edges < 2:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    sorted_order = torch.argsort(angles)
    sorted_angles = angles[sorted_order].cpu()
    ei_list: list[tuple[int, int]] = []

    for i in range(num_edges):
        j = i + 1
        while j < num_edges:
            diff = abs(sorted_angles[j].item() - sorted_angles[i].item())
            if diff > threshold:
                break
            ei_list.append((sorted_order[i].item(), sorted_order[j].item()))
            j += 1

    if num_edges > 1:
        i = num_edges - 1
        j = 0
        while j < i:
            diff = math.pi - sorted_angles[i].item() + sorted_angles[j].item()
            if diff > threshold:
                break
            ei_list.append((sorted_order[i].item(), sorted_order[j].item()))
            j += 1
        for i in range(num_edges - 2, -1, -1):
            if math.pi - sorted_angles[i].item() > threshold:
                break
            j = 0
            while j < i:
                diff = math.pi - sorted_angles[i].item() + sorted_angles[j].item()
                if diff > threshold:
                    break
                pair = (sorted_order[i].item(), sorted_order[j].item())
                if pair not in ei_list:
                    ei_list.append(pair)
                j += 1

    if not ei_list:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    pairs = torch.tensor(ei_list, dtype=torch.long, device=device)
    ei = torch.min(pairs[:, 0], pairs[:, 1])
    ej = torch.max(pairs[:, 0], pairs[:, 1])
    combined = ei * num_edges + ej
    unique_idx = torch.unique(combined, return_inverse=False, sorted=True)
    ei = unique_idx // num_edges
    ej = unique_idx % num_edges
    return ei, ej


# ---------------------------------------------------------------------------
# Helper: normalise output to a frozenset of (min, max) int pairs.
# ---------------------------------------------------------------------------


def _pairs_to_set(ei: torch.Tensor, ej: torch.Tensor) -> frozenset[tuple[int, int]]:
    if ei.numel() == 0:
        return frozenset()
    return frozenset(
        (int(a), int(b)) if a <= b else (int(b), int(a))
        for a, b in zip(ei.tolist(), ej.tolist())
    )


def _compare(angles: torch.Tensor, threshold: float) -> None:
    ei_new, ej_new = find_parallel_pairs(angles, threshold)
    ei_old, ej_old = _find_parallel_pairs_old(angles, threshold)
    new_set = _pairs_to_set(ei_new, ej_new)
    old_set = _pairs_to_set(ei_old, ej_old)
    assert new_set == old_set, (
        f"Mismatch on angles={angles.tolist()} threshold={threshold}\n"
        f"  new={sorted(new_set)}\n"
        f"  old={sorted(old_set)}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFindParallelPairsEquivalence:
    def test_empty_no_edges(self) -> None:
        angles = torch.empty(0)
        ei, ej = find_parallel_pairs(angles, threshold=0.1)
        assert ei.numel() == 0
        assert ej.numel() == 0

    def test_single_edge_no_pairs(self) -> None:
        angles = torch.tensor([0.5])
        ei, ej = find_parallel_pairs(angles, threshold=0.1)
        assert ei.numel() == 0

    def test_two_parallel_edges(self) -> None:
        """Two edges at almost the same angle → one pair."""
        angles = torch.tensor([0.1, 0.15])
        _compare(angles, threshold=0.1)

    def test_two_non_parallel_edges(self) -> None:
        """Two edges far apart → no pairs."""
        angles = torch.tensor([0.0, 1.5])
        _compare(angles, threshold=0.1)

    def test_axis_aligned_square(self) -> None:
        """4 edges: two horizontal (0) and two vertical (π/2).
        Horizontal pair and vertical pair should both be found."""
        angles = torch.tensor([0.0, math.pi / 2, 0.0, math.pi / 2])
        _compare(angles, threshold=0.05)

    def test_all_parallel(self) -> None:
        """All edges within threshold of each other."""
        angles = torch.tensor([0.0, 0.01, 0.02, 0.03])
        _compare(angles, threshold=0.05)

    def test_wraparound_pairs(self) -> None:
        """Edges near 0 and near π are parallel (π-periodic angles)."""
        # One edge near 0, one near π — should be detected as parallel.
        eps = 0.02
        angles = torch.tensor([eps, math.pi - eps])
        _compare(angles, threshold=0.05)

    def test_wraparound_with_main_pairs(self) -> None:
        """Mix of regular and wraparound parallel pairs."""
        eps = 0.02
        angles = torch.tensor([eps, 0.5, math.pi / 2 + 0.01, math.pi - eps])
        _compare(angles, threshold=0.05)

    def test_larger_random_graph(self) -> None:
        """20 edges at random angles — both implementations must agree."""
        torch.manual_seed(7)
        angles = torch.rand(20) * math.pi
        _compare(angles, threshold=0.1)

    def test_threshold_zero(self) -> None:
        """Threshold=0 → only exactly identical angles pair."""
        angles = torch.tensor([0.0, 0.0, 0.5, 1.0])
        _compare(angles, threshold=0.0)

    def test_output_ei_leq_ej(self) -> None:
        """Output must satisfy ei <= ej (canonical form)."""
        angles = torch.tensor([0.0, 0.01, 0.02, math.pi / 2, math.pi / 2 + 0.01])
        ei, ej = find_parallel_pairs(angles, threshold=0.05)
        if ei.numel() > 0:
            assert (ei <= ej).all(), "ei must be <= ej for all pairs"

    def test_no_self_pairs(self) -> None:
        """An edge must never be paired with itself."""
        angles = torch.tensor([0.0, 0.01, 0.02])
        ei, ej = find_parallel_pairs(angles, threshold=0.05)
        if ei.numel() > 0:
            assert (ei != ej).all(), "Self-pairs found in output"

    def test_max_pairs_caps_output(self) -> None:
        """max_pairs must bound the returned pair count on dense inputs.

        With 50 nearly-identical angles and a wide threshold every pair is
        parallel (1225 pairs). Passing max_pairs=100 must return exactly
        100 pairs, all of which are valid (ei < ej) and drawn from the
        original enumeration.
        """
        torch.manual_seed(13)
        # 50 angles within 0.01 radians — every pair is "parallel"
        angles = 0.5 + torch.rand(50) * 0.01
        ei_full, ej_full = find_parallel_pairs(angles, threshold=0.05)
        # Sanity: dense — expect C(50, 2) = 1225 pairs in the full enumeration
        assert ei_full.numel() == 50 * 49 // 2

        ei, ej = find_parallel_pairs(angles, threshold=0.05, max_pairs=100)
        assert ei.numel() == 100, f"expected exactly 100 pairs, got {ei.numel()}"
        assert (ei < ej).all(), "cap must preserve ei < ej invariant"
        # Every returned pair must be a member of the full enumeration.
        full_set = {(int(a), int(b)) for a, b in zip(ei_full, ej_full)}
        cap_set = {(int(a), int(b)) for a, b in zip(ei, ej)}
        assert cap_set <= full_set

    def test_max_pairs_noop_when_under_cap(self) -> None:
        """max_pairs must be a no-op when the full count is below the cap."""
        angles = torch.tensor([0.0, 0.01, 0.5, 1.5])
        ei_uncapped, ej_uncapped = find_parallel_pairs(angles, threshold=0.05)
        ei_capped, ej_capped = find_parallel_pairs(
            angles, threshold=0.05, max_pairs=1000
        )
        assert _pairs_to_set(ei_uncapped, ej_uncapped) == _pairs_to_set(
            ei_capped, ej_capped
        )


class TestConstraintConfigMaxPairsWiring:
    """Lock the end-to-end wiring from ConstraintConfig to pair-based axioms."""

    def test_registry_propagates_max_pairs(self) -> None:
        from src.constraints.axioms import AxiomRegistry
        from src.pipeline.config import ConstraintConfig

        cfg = ConstraintConfig(max_pairs_per_axiom=12345)
        reg = AxiomRegistry.create_default(cfg)

        for name in ("orthogonal", "parallel_pair", "non_intersection"):
            axiom = reg.get(name)
            assert axiom is not None
            assert getattr(axiom, "max_pairs", None) == 12345, (
                f"{name} axiom did not receive max_pairs from config"
            )
