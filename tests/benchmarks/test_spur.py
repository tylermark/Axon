"""Benchmark: SPUR (Standardized Prefab Utilization Ratio) on synthetic test set.

Q-015: Evaluates the SPUR metric across multiple synthetic floor plan
configurations using the greedy baseline policy.

SPUR is a composite metric defined on PanelizationResult:

    SPUR = w1 * coverage_pct + w2 * (1 - waste_pct) + w3 * pod_placement_rate

where w1=0.5, w2=0.3, w3=0.2.

Target from ARCHITECTURE.md: SPUR > 0.85 for a trained policy.
The greedy baseline uses a relaxed threshold of SPUR > 0.5.

Reference: docs/interfaces/drl_output.py, ARCHITECTURE.md (Evaluation Metrics).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.drl.env import PanelizationEnv
from src.drl.train import (
    generate_classified_graph,
    greedy_policy,
)
from src.knowledge_graph.loader import load_knowledge_graph

# ── SPUR weights from docs/interfaces/drl_output.py ─────────────────────
_W_COVERAGE = 0.5
_W_WASTE = 0.3
_W_POD_PLACEMENT = 0.2

# Relaxed threshold for greedy baseline (untrained).
# Target for trained policy is 0.85 per ARCHITECTURE.md.
_SPUR_THRESHOLD_GREEDY = 0.5
_SPUR_TARGET_TRAINED = 0.85


# ── Helpers ──────────────────────────────────────────────────────────────


def compute_spur_from_results(results: dict) -> float:
    """Compute the composite SPUR score from raw env results.

    Uses the same formula as PanelizationResult.spur_score:
        SPUR = w1 * coverage_pct + w2 * (1 - waste_pct) + w3 * pod_placement_rate

    All component values are normalized to [0, 1].
    """
    # Wall coverage as fraction
    total_walls = results["total_walls"]
    walls_covered = results["walls_covered"]
    coverage_pct = walls_covered / max(total_walls, 1)

    # Average waste percentage from step rewards (fraction in [0, 1])
    step_rewards = results["step_rewards"]
    if step_rewards:
        waste_values = [r.info.get("waste_percentage", 0.0) for r in step_rewards]
        waste_pct = np.mean(waste_values) / 100.0  # Convert from percentage to fraction
    else:
        waste_pct = 0.0
    waste_pct = min(max(waste_pct, 0.0), 1.0)

    # Pod placement rate as fraction
    total_rooms = results["total_rooms"]
    rooms_covered = results["rooms_covered"]
    pod_placement_rate = rooms_covered / max(total_rooms, 1)

    spur = (
        _W_COVERAGE * coverage_pct
        + _W_WASTE * (1.0 - waste_pct)
        + _W_POD_PLACEMENT * pod_placement_rate
    )
    return float(spur)


def run_greedy_episode(
    classified_graph,
    store,
    max_steps: int = 500,
) -> dict:
    """Run one greedy-policy episode and return env results.

    Args:
        classified_graph: A ClassifiedWallGraph to panelize.
        store: Loaded KnowledgeGraphStore.
        max_steps: Safety limit to prevent infinite loops from multi-pod
            placement when SKIP does not advance the room index.

    Returns:
        Results dict from ``PanelizationEnv.get_results()``.
    """
    env = PanelizationEnv(classified_graph=classified_graph, store=store)
    _obs, _info = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = greedy_policy(env)
        _obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
        steps += 1

    return env.get_results()


# ── Test configurations ──────────────────────────────────────────────────

# Each tuple: (label, num_rooms, room_size_range_inches, opening_probability)
_TEST_CONFIGS = [
    ("small_1room", 1, (96.0, 144.0), 0.0),
    ("medium_4rooms", 4, (96.0, 192.0), 0.3),
    ("large_8rooms", 8, (72.0, 240.0), 0.4),
]


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.benchmark
class TestSPUR:
    """SPUR benchmark on synthetic floor plan configurations."""

    @pytest.fixture(scope="class")
    def store(self):
        """Load the Knowledge Graph once for all tests in this class."""
        return load_knowledge_graph()

    @pytest.mark.parametrize(
        "label, num_rooms, room_size_range, opening_prob",
        _TEST_CONFIGS,
        ids=[cfg[0] for cfg in _TEST_CONFIGS],
    )
    def test_spur_greedy_baseline(
        self,
        store,
        label: str,
        num_rooms: int,
        room_size_range: tuple[float, float],
        opening_prob: float,
    ):
        """SPUR score for greedy policy meets relaxed threshold on synthetic plans.

        Generates a synthetic floor plan with the given configuration,
        runs the greedy baseline policy through PanelizationEnv, and
        asserts that the composite SPUR score exceeds the greedy threshold.

        The SPUR target for a trained policy is 0.85 (per ARCHITECTURE.md).
        The greedy baseline uses a relaxed threshold of 0.5.
        """
        rng = np.random.default_rng(42)
        cg = generate_classified_graph(
            rng=rng,
            num_rooms=num_rooms,
            room_size_range_inches=room_size_range,
            opening_probability=opening_prob,
        )

        results = run_greedy_episode(cg, store)
        spur = compute_spur_from_results(results)

        # Greedy baseline: relaxed threshold.
        # Trained policy target: SPUR > 0.85 (ARCHITECTURE.md).
        assert spur > _SPUR_THRESHOLD_GREEDY, (
            f"[{label}] SPUR {spur:.3f} below greedy threshold {_SPUR_THRESHOLD_GREEDY}. "
            f"Coverage: {results['wall_coverage_pct']:.1f}%, "
            f"Rooms: {results['rooms_covered']}/{results['total_rooms']}"
        )

    def test_spur_components_are_valid(self, store):
        """Verify SPUR components are in valid ranges across a set of plans.

        Checks that coverage, waste, and pod placement rate all produce
        sensible values that sum correctly into the composite SPUR score.
        """
        rng = np.random.default_rng(123)

        for num_rooms in [2, 4, 6]:
            cg = generate_classified_graph(
                rng=rng,
                num_rooms=num_rooms,
                room_size_range_inches=(96.0, 192.0),
                opening_probability=0.2,
            )
            results = run_greedy_episode(cg, store)
            spur = compute_spur_from_results(results)

            # SPUR must be in [0, 1]
            assert 0.0 <= spur <= 1.0, (
                f"SPUR {spur:.4f} outside valid range [0, 1] "
                f"for {num_rooms}-room plan"
            )

            # Individual components must be non-negative
            total_walls = results["total_walls"]
            walls_covered = results["walls_covered"]
            rooms_covered = results["rooms_covered"]
            total_rooms = results["total_rooms"]

            assert walls_covered >= 0
            assert walls_covered <= total_walls
            assert rooms_covered >= 0
            assert rooms_covered <= total_rooms

    def test_spur_multi_episode_aggregation(self, store):
        """SPUR is consistent across multiple episodes on varied plans.

        Runs 3 episodes with different random floor plans and verifies
        that the aggregated SPUR scores are all valid and in [0, 1].
        """
        rng = np.random.default_rng(42)
        spur_scores: list[float] = []

        for num_rooms in [2, 3, 4]:
            cg = generate_classified_graph(
                rng=rng,
                num_rooms=num_rooms,
                room_size_range_inches=(96.0, 192.0),
                opening_probability=0.2,
            )
            results = run_greedy_episode(cg, store)
            spur = compute_spur_from_results(results)
            spur_scores.append(spur)

        # All individual SPUR scores must be valid
        for i, score in enumerate(spur_scores):
            assert 0.0 <= score <= 1.0, (
                f"Episode {i}: SPUR {score:.4f} outside valid range"
            )

        # Mean SPUR across episodes should be reasonable
        mean_spur = float(np.mean(spur_scores))
        assert mean_spur > 0.0, (
            "Mean SPUR is 0.0 -- no walls/rooms were covered in any episode"
        )


@pytest.mark.benchmark
class TestSPURTarget:
    """Placeholder assertion for trained-policy SPUR target.

    These tests document the ARCHITECTURE.md target (SPUR > 0.85)
    and will be updated once a trained checkpoint is available.
    """

    @pytest.fixture(scope="class")
    def store(self):
        return load_knowledge_graph()

    @pytest.mark.skip(reason="Requires trained DRL checkpoint; greedy baseline does not meet 0.85")
    def test_spur_trained_policy_target(self, store):
        """Trained policy should achieve SPUR > 0.85.

        ARCHITECTURE.md mandates SPUR > 0.85 for the trained policy.
        This test will be unskipped when a trained model checkpoint
        is available for evaluation.
        """
        # TODO: Load trained model checkpoint and evaluate.
        # eval_result = evaluate_policy(env=..., model=trained_model, num_episodes=20)
        # assert eval_result.mean_spur > _SPUR_TARGET_TRAINED
        pass
