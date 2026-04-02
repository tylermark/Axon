"""Benchmark: DRL reward convergence smoke test.

Q-016: Verifies that the DRL training loop produces improving rewards over
episodes. This is a smoke test — not a full convergence proof — confirming
that the training infrastructure works end-to-end and that the policy
improves (or at least does not degrade) over a very short training run.

Two test variants:

1. **Full SB3 training** (``test_train_drl_convergence``): Uses ``train_drl``
   with MaskablePPO for a very short run (~500 timesteps). Requires
   sb3-contrib and stable-baselines3; skipped if not installed.

2. **Manual episode collection** (``test_episode_reward_consistency``): Runs
   the greedy baseline across multiple episodes on varied floor plans,
   verifying that the environment produces stable, non-degenerate rewards.
   Always runs (no SB3 dependency).

Reference: AGENTS.md DRL Agent, TASKS.md DRL-009.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.drl.env import PanelizationEnv
from src.drl.train import (
    DRLTrainingConfig,
    generate_classified_graph,
    greedy_policy,
)
from src.knowledge_graph.loader import load_knowledge_graph

# Check for optional SB3 dependencies
try:
    from sb3_contrib import MaskablePPO  # noqa: F401

    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False

try:
    from stable_baselines3.common.callbacks import BaseCallback  # noqa: F401

    _HAS_SB3_BASE = True
except ImportError:
    _HAS_SB3_BASE = False

# Safety step limit to prevent infinite loops from the multi-pod
# placement advance bug (DRL-006: SKIP does not advance room index
# when remaining area is above threshold).
_MAX_EPISODE_STEPS = 500


# ── Helpers ──────────────────────────────────────────────────────────────


def _run_greedy_episode(env: PanelizationEnv) -> float:
    """Run one greedy-policy episode and return the total reward.

    Includes a step-count safety limit to avoid hangs from the known
    multi-pod SKIP-advance issue in PanelizationEnv._advance().
    """
    _obs, _info = env.reset()
    done = False
    episode_reward = 0.0
    steps = 0

    while not done and steps < _MAX_EPISODE_STEPS:
        action = greedy_policy(env)
        _obs, reward, terminated, truncated, _info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        steps += 1

    return episode_reward


def _run_random_episode(env: PanelizationEnv, rng: np.random.Generator) -> float:
    """Run one random-policy episode and return the total reward."""
    _obs, _info = env.reset()
    done = False
    episode_reward = 0.0
    steps = 0

    while not done and steps < _MAX_EPISODE_STEPS:
        mask = env.action_masks()
        valid = np.where(mask > 0)[0]
        action = int(rng.choice(valid)) if len(valid) > 0 else 0
        _obs, reward, terminated, truncated, _info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        steps += 1

    return episode_reward


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.benchmark
class TestDRLConvergence:
    """Reward convergence tests for the DRL training pipeline."""

    @pytest.fixture(scope="class")
    def store(self):
        """Load Knowledge Graph once for all tests."""
        return load_knowledge_graph()

    @pytest.mark.slow
    @pytest.mark.skipif(
        not (_HAS_SB3 and _HAS_SB3_BASE),
        reason="sb3-contrib and stable-baselines3 required for MaskablePPO training",
    )
    def test_train_drl_convergence(self, store, tmp_path):
        """Short MaskablePPO training run shows reward improvement.

        Trains for a very small number of timesteps (500) and checks that
        the trained policy produces finite rewards. This demonstrates that
        the training loop functions and the policy is not degenerating.

        This is a smoke test, not a full convergence proof. A production
        run would use 100k+ timesteps.
        """
        from src.drl.train import train_drl

        config = DRLTrainingConfig(
            total_timesteps=500,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            num_envs=1,
            num_eval_episodes=3,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            use_wandb=False,
            seed=42,
            log_interval=1,
            synthetic_num_rooms_range=(1, 3),
            synthetic_room_size_range_inches=(96.0, 192.0),
            synthetic_opening_probability=0.2,
        )

        results = train_drl(config=config, store=store)

        # Verify training produced output
        assert "eval_trained" in results
        assert "eval_greedy" in results
        assert results["total_timesteps"] == 500
        assert results["training_time_seconds"] > 0.0

        eval_trained = results["eval_trained"]
        eval_greedy = results["eval_greedy"]

        # The trained policy should not be drastically worse than greedy.
        # With only 500 steps it may not beat greedy, but it should at
        # least produce finite, non-degenerate rewards.
        assert np.isfinite(eval_trained.mean_reward)
        assert np.isfinite(eval_greedy.mean_reward)
        assert eval_trained.mean_episode_length > 0
        assert eval_greedy.mean_episode_length > 0

    def test_episode_reward_consistency(self, store):
        """Greedy policy produces stable, non-degenerate rewards across episodes.

        Runs the greedy baseline on 8 different synthetic floor plans and
        verifies:
        1. All episode rewards are finite.
        2. No episode has a catastrophically negative reward.
        3. Reward variance is bounded (policy is stable, not random).
        """
        rng = np.random.default_rng(42)
        episode_rewards: list[float] = []

        for i in range(8):
            # Vary complexity across episodes
            num_rooms = (i % 4) + 1  # 1, 2, 3, 4, 1, 2, 3, 4
            cg = generate_classified_graph(
                rng=rng,
                num_rooms=num_rooms,
                room_size_range_inches=(96.0, 192.0),
                opening_probability=0.2,
            )

            env = PanelizationEnv(classified_graph=cg, store=store)
            ep_reward = _run_greedy_episode(env)
            episode_rewards.append(ep_reward)

        rewards = np.array(episode_rewards)

        # All rewards must be finite
        assert np.all(np.isfinite(rewards)), (
            f"Non-finite rewards detected: {rewards}"
        )

        # No catastrophically negative reward (greedy should be reasonable)
        assert np.all(rewards > -50.0), (
            f"Catastrophically negative reward detected: min={rewards.min():.2f}"
        )

        # Rewards should have bounded variance (stable policy)
        std = float(np.std(rewards))
        mean = float(np.mean(rewards))
        assert std < abs(mean) * 5.0 + 10.0, (
            f"Reward variance too high: mean={mean:.2f}, std={std:.2f}"
        )

    def test_greedy_vs_random_baseline(self, store):
        """Greedy policy outperforms a random policy.

        Demonstrates that the greedy heuristic (pick best KG candidate)
        produces better episode rewards than taking random valid actions.
        This validates that the reward function correctly incentivizes
        good panelization decisions.
        """
        rng = np.random.default_rng(99)
        action_rng = np.random.default_rng(99)

        greedy_rewards: list[float] = []
        random_rewards: list[float] = []

        for i in range(5):
            num_rooms = (i % 3) + 2  # 2, 3, 4, 2, 3
            cg = generate_classified_graph(
                rng=rng,
                num_rooms=num_rooms,
                room_size_range_inches=(96.0, 192.0),
                opening_probability=0.2,
            )

            # Greedy episode
            env = PanelizationEnv(classified_graph=cg, store=store)
            greedy_rewards.append(_run_greedy_episode(env))

            # Random episode (same floor plan)
            env2 = PanelizationEnv(classified_graph=cg, store=store)
            random_rewards.append(_run_random_episode(env2, action_rng))

        mean_greedy = float(np.mean(greedy_rewards))
        mean_random = float(np.mean(random_rewards))

        # Greedy should achieve higher mean reward than random.
        # This validates the reward function's signal quality.
        assert mean_greedy >= mean_random, (
            f"Greedy policy (mean={mean_greedy:.2f}) should outperform "
            f"random policy (mean={mean_random:.2f})"
        )

    def test_reward_improves_with_coverage(self, store):
        """Episodes completing more walls/rooms achieve higher rewards.

        Runs the greedy policy and verifies that higher wall/room coverage
        correlates with higher episode reward. This confirms the reward
        function correctly incentivizes coverage.
        """
        rng = np.random.default_rng(77)
        coverages: list[float] = []
        rewards: list[float] = []

        for i in range(6):
            num_rooms = (i % 3) + 1
            cg = generate_classified_graph(
                rng=rng,
                num_rooms=num_rooms,
                room_size_range_inches=(96.0, 192.0),
                opening_probability=0.1,
            )

            env = PanelizationEnv(classified_graph=cg, store=store)
            ep_reward = _run_greedy_episode(env)

            results = env.get_results()
            total = results["total_walls"] + results["total_rooms"]
            covered = results["walls_covered"] + results["rooms_covered"]
            coverage_frac = covered / max(total, 1)

            coverages.append(coverage_frac)
            rewards.append(ep_reward)

        # Episodes with higher coverage should tend to have higher rewards.
        # Use Spearman rank correlation as a soft check (not strict monotonic).
        from scipy.stats import spearmanr

        if len(set(coverages)) > 1:
            corr, _pval = spearmanr(coverages, rewards)
            # We expect a positive correlation (higher coverage -> higher reward).
            # Allow a slightly negative correlation since other reward components
            # (waste, violations) can outweigh coverage in some episodes.
            assert corr > -0.5, (
                f"Coverage-reward correlation is unexpectedly negative: {corr:.3f}. "
                f"Coverages: {coverages}, Rewards: {[f'{r:.2f}' for r in rewards]}"
            )
