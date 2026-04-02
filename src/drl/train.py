"""DRL training pipeline for panelization/placement policy.

DRL-009: Trains a MaskablePPO policy on simulated floor plans using the
PanelizationEnv (DRL-001). Includes:

- Synthetic floor plan generation with configurable complexity
- SB3-contrib MaskablePPO integration with action masking
- Episode metric logging (SPUR, waste %, coverage %, reward)
- Checkpoint saving at configurable intervals
- Optional Weights & Biases integration
- Greedy baseline for evaluation comparison
- CLI entry point for standalone training

Reference: AGENTS.md DRL Agent, TASKS.md DRL-009.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from docs.interfaces.graph_to_serializer import (
    FinalizedGraph,
    Opening,
    OpeningType,
    Room,
    WallSegment,
    WallType,
)
from src.classifier.classifier import classify_wall_graph
from src.drl.env import PanelizationEnv
from src.knowledge_graph.loader import load_knowledge_graph

if TYPE_CHECKING:
    import gymnasium as gym

    from docs.interfaces.classified_wall_graph import ClassifiedWallGraph
    from src.drl.reward import RewardWeights
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)

# ── Optional dependency imports ────────────────────────────────────────────

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False

try:
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv

    _HAS_SB3_BASE = True
except ImportError:
    _HAS_SB3_BASE = False

try:
    import wandb

    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


# ── Configuration ─────────────────────────────────────────────────────────


@dataclass
class DRLTrainingConfig:
    """Hyperparameters and settings for the DRL training pipeline.

    Attributes:
        total_timesteps: Total environment steps for training.
        learning_rate: PPO learning rate.
        n_steps: Number of steps per rollout before update.
        batch_size: Minibatch size for PPO updates.
        n_epochs: Number of PPO optimization epochs per rollout.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_range: PPO clipping range.
        ent_coef: Entropy coefficient for exploration.
        checkpoint_interval: Save model every N timesteps.
        checkpoint_dir: Directory for saving model checkpoints.
        use_wandb: Enable Weights & Biases logging.
        wandb_project: W&B project name.
        num_eval_episodes: Number of episodes for evaluation.
        num_envs: Number of parallel vectorized environments.
        seed: Random seed for reproducibility.
        log_interval: Print training stats every N rollouts.
        synthetic_num_rooms_range: Range of room count for synthetic plans.
        synthetic_room_size_range_inches: Room size range in inches.
        synthetic_opening_probability: Probability of an opening on each wall.
        synthetic_wall_types: Wall types to sample from in synthetic plans.
    """

    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    checkpoint_interval: int = 10_000
    checkpoint_dir: str = "checkpoints/drl"
    use_wandb: bool = False
    wandb_project: str = "axon-drl"
    num_eval_episodes: int = 20
    num_envs: int = 4
    seed: int = 42
    log_interval: int = 1
    synthetic_num_rooms_range: tuple[int, int] = (1, 8)
    synthetic_room_size_range_inches: tuple[float, float] = (72.0, 240.0)
    synthetic_opening_probability: float = 0.3
    synthetic_wall_types: list[WallType] = field(
        default_factory=lambda: [
            WallType.LOAD_BEARING,
            WallType.PARTITION,
            WallType.EXTERIOR,
            WallType.SHEAR,
        ],
    )


# ── Synthetic floor plan generation ───────────────────────────────────────


def generate_synthetic_floor_plan(
    rng: np.random.Generator,
    num_rooms: int = 4,
    room_size_range_inches: tuple[float, float] = (72.0, 240.0),
    opening_probability: float = 0.3,
    wall_types: list[WallType] | None = None,
) -> FinalizedGraph:
    """Generate a random rectangular floor plan with rooms.

    Creates a grid-like layout where rooms are arranged in rows. Each room
    is a rectangle with random dimensions within the given range. Walls
    between adjacent rooms are shared (interior partitions).

    Args:
        rng: NumPy random generator for reproducibility.
        num_rooms: Number of rooms in the floor plan (1-8).
        room_size_range_inches: Min/max room dimension in inches.
        opening_probability: Probability of placing a door/window on each wall.
        wall_types: Wall types to assign randomly. Defaults to a standard mix.

    Returns:
        A FinalizedGraph with walls, rooms, and openings.
    """
    if wall_types is None:
        wall_types = [
            WallType.LOAD_BEARING,
            WallType.PARTITION,
            WallType.EXTERIOR,
        ]

    num_rooms = max(1, min(num_rooms, 8))
    min_size, max_size = room_size_range_inches

    # Arrange rooms in a grid: roughly sqrt(n) x sqrt(n)
    n_cols = max(1, math.ceil(math.sqrt(num_rooms)))
    n_rows = max(1, math.ceil(num_rooms / n_cols))

    # Generate random widths and heights for each column/row
    col_widths = rng.uniform(min_size, max_size, n_cols)
    row_heights = rng.uniform(min_size, max_size, n_rows)

    # Convert inches to PDF user units (72 units/inch)
    pdf_per_inch = 72.0
    col_widths_pdf = col_widths * pdf_per_inch
    row_heights_pdf = row_heights * pdf_per_inch

    # Build grid nodes: (n_rows+1) x (n_cols+1) grid of junction points
    x_positions = np.concatenate([[0.0], np.cumsum(col_widths_pdf)])
    y_positions = np.concatenate([[0.0], np.cumsum(row_heights_pdf)])

    node_grid: dict[tuple[int, int], int] = {}
    nodes_list: list[tuple[float, float]] = []

    for row in range(n_rows + 1):
        for col in range(n_cols + 1):
            node_id = len(nodes_list)
            node_grid[(row, col)] = node_id
            nodes_list.append((float(x_positions[col]), float(y_positions[row])))

    nodes = np.array(nodes_list, dtype=np.float64)

    # Build wall segments from grid edges
    wall_segments: list[WallSegment] = []
    edge_list: list[tuple[int, int]] = []

    # Track which grid cells are actual rooms
    room_cells: list[tuple[int, int]] = []
    for row in range(n_rows):
        for col in range(n_cols):
            room_idx = row * n_cols + col
            if room_idx < num_rooms:
                room_cells.append((row, col))

    # Determine which edges exist (only edges bordering at least one room)
    existing_edges: set[tuple[int, int]] = set()
    for row, col in room_cells:
        # Top edge
        n0, n1 = node_grid[(row, col)], node_grid[(row, col + 1)]
        existing_edges.add((min(n0, n1), max(n0, n1)))
        # Bottom edge
        n0, n1 = node_grid[(row + 1, col)], node_grid[(row + 1, col + 1)]
        existing_edges.add((min(n0, n1), max(n0, n1)))
        # Left edge
        n0, n1 = node_grid[(row, col)], node_grid[(row + 1, col)]
        existing_edges.add((min(n0, n1), max(n0, n1)))
        # Right edge
        n0, n1 = node_grid[(row, col + 1)], node_grid[(row + 1, col + 1)]
        existing_edges.add((min(n0, n1), max(n0, n1)))

    # Build wall segments
    edge_node_to_id: dict[tuple[int, int], int] = {}
    for edge_idx, (n0, n1) in enumerate(sorted(existing_edges)):
        start_coord = nodes[n0]
        end_coord = nodes[n1]
        delta = end_coord - start_coord
        length = float(np.linalg.norm(delta))
        angle = float(np.arctan2(delta[1], delta[0]) % np.pi)

        # Determine if this is a perimeter wall (only one room on one side)
        is_perimeter = _is_perimeter_edge(n0, n1, node_grid, room_cells, n_rows, n_cols)
        wt = rng.choice(wall_types) if not is_perimeter else WallType.EXTERIOR

        # Random wall thickness: 4-8 PDF units (~0.05-0.11 inches)
        thickness = rng.uniform(4.0, 8.0)

        segment = WallSegment(
            edge_id=edge_idx,
            start_node=int(n0),
            end_node=int(n1),
            start_coord=start_coord.copy(),
            end_coord=end_coord.copy(),
            thickness=thickness,
            height=2700.0,
            wall_type=wt,
            angle=angle,
            length=length,
            confidence=1.0,
        )
        wall_segments.append(segment)
        edge_list.append((int(n0), int(n1)))
        edge_node_to_id[(n0, n1)] = edge_idx
        edge_node_to_id[(n1, n0)] = edge_idx

    edges = np.array(edge_list, dtype=np.int64) if edge_list else np.empty((0, 2), dtype=np.int64)

    # Build openings on walls
    openings: list[Opening] = []
    for seg in wall_segments:
        if rng.random() < opening_probability and seg.length > 36.0 * pdf_per_inch / 72.0:
            # Place a door or window
            opening_type = rng.choice([OpeningType.DOOR, OpeningType.WINDOW])
            # Opening width: 30-42 inches for doors, 24-60 inches for windows
            if opening_type == OpeningType.DOOR:
                width_inches = rng.uniform(30.0, 42.0)
                height_mm = 2100.0
                sill = 0.0
            else:
                width_inches = rng.uniform(24.0, 60.0)
                height_mm = rng.uniform(900.0, 1500.0)
                sill = rng.uniform(600.0, 1000.0)

            width_pdf = width_inches * pdf_per_inch
            # Ensure opening fits within wall
            if width_pdf < seg.length * 0.8:
                # Random position along wall, avoiding ends
                position = rng.uniform(0.2, 0.8)
                openings.append(
                    Opening(
                        opening_type=opening_type,
                        wall_edge_id=seg.edge_id,
                        position_along_wall=position,
                        width=width_pdf,
                        height=height_mm,
                        sill_height=sill,
                        confidence=1.0,
                    )
                )

    # Build rooms
    rooms: list[Room] = []
    for idx, (row, col) in enumerate(room_cells):
        # Boundary nodes: the four corners of this grid cell
        boundary_nodes = [
            node_grid[(row, col)],
            node_grid[(row, col + 1)],
            node_grid[(row + 1, col + 1)],
            node_grid[(row + 1, col)],
        ]

        # Boundary edges: the four edges of this grid cell
        boundary_edges = []
        corners = [*boundary_nodes, boundary_nodes[0]]
        for i in range(4):
            n0, n1 = corners[i], corners[i + 1]
            eid = edge_node_to_id.get((n0, n1))
            if eid is not None:
                boundary_edges.append(eid)

        area = col_widths_pdf[col] * row_heights_pdf[row]

        # Random room label
        labels = ["Bedroom", "Bathroom", "Kitchen", "Living", "Office", ""]
        label = rng.choice(labels)

        rooms.append(
            Room(
                room_id=idx,
                boundary_edges=boundary_edges,
                boundary_nodes=boundary_nodes,
                area=area,
                label=label,
                is_exterior=False,
            )
        )

    # Add an exterior room for completeness
    all_boundary_nodes = list(range(len(nodes_list)))
    rooms.append(
        Room(
            room_id=len(room_cells),
            boundary_edges=[],
            boundary_nodes=all_boundary_nodes,
            area=0.0,
            label="",
            is_exterior=True,
        )
    )

    page_width = float(x_positions[-1]) + 100.0
    page_height = float(y_positions[-1]) + 100.0

    return FinalizedGraph(
        nodes=nodes,
        edges=edges,
        wall_segments=wall_segments,
        openings=openings,
        rooms=rooms,
        page_width=page_width,
        page_height=page_height,
        scale_factor=1.0,
    )


def _is_perimeter_edge(
    n0: int,
    n1: int,
    node_grid: dict[tuple[int, int], int],
    room_cells: list[tuple[int, int]],
    n_rows: int,
    n_cols: int,
) -> bool:
    """Check whether an edge lies on the perimeter of the floor plan.

    An edge is on the perimeter if it borders at most one room cell.
    """
    # Build the inverse map from node_id to grid position
    inv_grid: dict[int, tuple[int, int]] = {v: k for k, v in node_grid.items()}
    if n0 not in inv_grid or n1 not in inv_grid:
        return True

    r0, c0 = inv_grid[n0]
    r1, c1 = inv_grid[n1]

    # Determine which cells are adjacent to this edge
    adjacent_cells: list[tuple[int, int]] = []
    if r0 == r1:
        # Horizontal edge at row r0: cells above (r0-1, col) and below (r0, col)
        col_min = min(c0, c1)
        if r0 > 0:
            adjacent_cells.append((r0 - 1, col_min))
        if r0 < n_rows:
            adjacent_cells.append((r0, col_min))
    elif c0 == c1:
        # Vertical edge at col c0: cells left (row, c0-1) and right (row, c0)
        row_min = min(r0, r1)
        if c0 > 0:
            adjacent_cells.append((row_min, c0 - 1))
        if c0 < n_cols:
            adjacent_cells.append((row_min, c0))
    else:
        return True  # Diagonal edge — treat as perimeter

    # Count how many adjacent cells are actual rooms
    room_set = set(room_cells)
    room_count = sum(1 for cell in adjacent_cells if cell in room_set)
    return room_count <= 1


def generate_classified_graph(
    rng: np.random.Generator,
    num_rooms: int = 4,
    room_size_range_inches: tuple[float, float] = (72.0, 240.0),
    opening_probability: float = 0.3,
    wall_types: list[WallType] | None = None,
) -> ClassifiedWallGraph:
    """Generate a synthetic floor plan and classify its walls.

    Combines ``generate_synthetic_floor_plan`` with the wall classifier
    pipeline to produce a ``ClassifiedWallGraph`` ready for the DRL env.

    Args:
        rng: NumPy random generator.
        num_rooms: Number of rooms (1-8).
        room_size_range_inches: Room dimension range in inches.
        opening_probability: Probability of openings on each wall.
        wall_types: Wall types to sample from.

    Returns:
        A ClassifiedWallGraph with walls, rooms, openings, and classifications.
    """
    graph = generate_synthetic_floor_plan(
        rng=rng,
        num_rooms=num_rooms,
        room_size_range_inches=room_size_range_inches,
        opening_probability=opening_probability,
        wall_types=wall_types,
    )
    return classify_wall_graph(graph)


# ── Environment factory ───────────────────────────────────────────────────


def _get_action_masks(env: gym.Env) -> np.ndarray:
    """Retrieve action masks from the underlying PanelizationEnv.

    Used as the mask function for SB3-contrib's ActionMasker wrapper.
    """
    return env.action_masks()


def make_env(
    store: KnowledgeGraphStore,
    config: DRLTrainingConfig,
    seed: int,
    reward_weights: RewardWeights | None = None,
) -> gym.Env:
    """Create a single PanelizationEnv with a random synthetic floor plan.

    The environment generates a new random floor plan on each ``reset()``.
    This is achieved via the ``RandomFloorPlanWrapper`` that calls
    ``generate_classified_graph()`` before delegating to ``PanelizationEnv.reset()``.

    Args:
        store: Loaded Knowledge Graph store.
        config: Training configuration.
        seed: Random seed for this env instance.
        reward_weights: Optional custom reward weights.

    Returns:
        A Gymnasium-compatible environment, optionally wrapped with
        ActionMasker for MaskablePPO.
    """
    rng = np.random.default_rng(seed)

    # Generate an initial floor plan
    num_rooms = rng.integers(
        config.synthetic_num_rooms_range[0],
        config.synthetic_num_rooms_range[1] + 1,
    )
    cg = generate_classified_graph(
        rng=rng,
        num_rooms=int(num_rooms),
        room_size_range_inches=config.synthetic_room_size_range_inches,
        opening_probability=config.synthetic_opening_probability,
        wall_types=config.synthetic_wall_types,
    )

    env = RandomFloorPlanWrapper(
        classified_graph=cg,
        store=store,
        reward_weights=reward_weights,
        config=config,
        rng=rng,
    )

    # Wrap with ActionMasker for MaskablePPO
    if _HAS_SB3:
        env = ActionMasker(env, _get_action_masks)

    return env


class RandomFloorPlanWrapper(PanelizationEnv):
    """PanelizationEnv wrapper that generates a new floor plan on each reset.

    Each episode uses a fresh synthetic floor plan, providing diverse
    training data. The floor plan complexity varies randomly within the
    configured ranges.
    """

    def __init__(
        self,
        classified_graph: ClassifiedWallGraph,
        store: KnowledgeGraphStore,
        reward_weights: RewardWeights | None,
        config: DRLTrainingConfig,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(classified_graph, store, reward_weights)
        self._config = config
        self._rng = rng

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset with a new random floor plan.

        Generates a fresh ClassifiedWallGraph and passes it to the
        underlying PanelizationEnv via the ``options`` mechanism.
        """
        num_rooms = self._rng.integers(
            self._config.synthetic_num_rooms_range[0],
            self._config.synthetic_num_rooms_range[1] + 1,
        )
        new_cg = generate_classified_graph(
            rng=self._rng,
            num_rooms=int(num_rooms),
            room_size_range_inches=self._config.synthetic_room_size_range_inches,
            opening_probability=self._config.synthetic_opening_probability,
            wall_types=self._config.synthetic_wall_types,
        )
        opts = options or {}
        opts["classified_graph"] = new_cg
        return super().reset(seed=seed, options=opts)


def make_vec_env(
    store: KnowledgeGraphStore,
    config: DRLTrainingConfig,
    reward_weights: RewardWeights | None = None,
) -> Any:
    """Create a vectorized environment for parallel training.

    Uses DummyVecEnv (single-process) since each env involves KG queries
    that are fast and the primary bottleneck is policy forward passes.

    Args:
        store: Loaded Knowledge Graph store.
        config: Training configuration.
        reward_weights: Optional custom reward weights.

    Returns:
        A SB3-compatible vectorized environment.

    Raises:
        ImportError: If stable-baselines3 is not installed.
    """
    if not _HAS_SB3_BASE:
        raise ImportError(
            "stable-baselines3 is required for vectorized environments. "
            "Install with: pip install stable-baselines3"
        )

    env_fns = [
        lambda seed=config.seed + i: make_env(store, config, seed, reward_weights)
        for i in range(config.num_envs)
    ]
    return DummyVecEnv(env_fns)


# ── Greedy baseline policy ────────────────────────────────────────────────


def greedy_policy(env: PanelizationEnv) -> int:
    """Greedy baseline: always pick the highest-SPUR candidate.

    For panelization: selects the candidate with the best SPUR score
    (closest to standard panel lengths, lowest waste). Falls back to
    action 1 (first candidate) if valid, otherwise SKIP.

    For placement: selects the first valid non-skip action (normal
    orientation preferred).

    Args:
        env: The PanelizationEnv with current state.

    Returns:
        Discrete action index.
    """
    mask = env.action_masks()
    valid_actions = np.where(mask > 0)[0]

    if len(valid_actions) == 0:
        return 0  # SKIP

    if env.phase == "panelization":
        # Pick the first valid non-skip action (candidates are sorted
        # by KG recommendation score descending, so action 1 is best)
        non_skip = [a for a in valid_actions if a != 0]
        if non_skip:
            return int(non_skip[0])
        return 0

    # Placement: prefer first valid non-skip (normal orientation)
    non_skip = [a for a in valid_actions if a != 0]
    if non_skip:
        # Prefer odd actions (normal orientation) over even (rotated)
        normal = [a for a in non_skip if a % 2 == 1]
        if normal:
            return int(normal[0])
        return int(non_skip[0])
    return 0


# ── Evaluation ────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    """Aggregated evaluation results across multiple episodes.

    Attributes:
        num_episodes: Number of evaluation episodes run.
        mean_reward: Mean total episode reward.
        std_reward: Standard deviation of episode reward.
        mean_spur: Mean SPUR score across episodes.
        std_spur: Standard deviation of SPUR score.
        mean_waste_pct: Mean waste percentage across episodes.
        std_waste_pct: Standard deviation of waste percentage.
        mean_wall_coverage_pct: Mean wall coverage percentage.
        std_wall_coverage_pct: Standard deviation of wall coverage.
        mean_room_coverage_pct: Mean room coverage percentage.
        std_room_coverage_pct: Standard deviation of room coverage.
        mean_violations: Mean violation count per episode.
        mean_episode_length: Mean number of steps per episode.
    """

    num_episodes: int = 0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_spur: float = 0.0
    std_spur: float = 0.0
    mean_waste_pct: float = 0.0
    std_waste_pct: float = 0.0
    mean_wall_coverage_pct: float = 0.0
    std_wall_coverage_pct: float = 0.0
    mean_room_coverage_pct: float = 0.0
    std_room_coverage_pct: float = 0.0
    mean_violations: float = 0.0
    mean_episode_length: float = 0.0


def evaluate_policy(
    env: PanelizationEnv,
    policy_fn: Any | None = None,
    num_episodes: int = 20,
    model: Any | None = None,
) -> EvalResult:
    """Evaluate a policy on the given environment.

    Runs ``num_episodes`` episodes and collects per-episode metrics from
    ``env.get_results()``. Accepts either a callable ``policy_fn(env) -> action``
    or an SB3 ``model`` with a ``.predict()`` method.

    Args:
        env: A PanelizationEnv (unwrapped, for access to ``get_results()``).
        policy_fn: A callable ``(env) -> action``. Used when ``model`` is None.
        num_episodes: Number of evaluation episodes.
        model: An SB3-compatible model with ``.predict(obs, action_masks=...)``.
            If provided, overrides ``policy_fn``.

    Returns:
        An EvalResult with aggregated statistics.
    """
    rewards: list[float] = []
    spurs: list[float] = []
    waste_pcts: list[float] = []
    wall_coverages: list[float] = []
    room_coverages: list[float] = []
    violation_counts: list[int] = []
    episode_lengths: list[int] = []

    # Unwrap to get the base PanelizationEnv for get_results()
    base_env = env
    while hasattr(base_env, "env") and not isinstance(base_env, PanelizationEnv):
        base_env = base_env.env

    for _ in range(num_episodes):
        obs, _info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            if model is not None:
                # SB3 model prediction with action masking
                masks = base_env.action_masks() if isinstance(base_env, PanelizationEnv) else None
                action, _ = model.predict(obs, deterministic=True, action_masks=masks)
                action = int(action)
            elif policy_fn is not None:
                action = policy_fn(base_env)
            else:
                action = 0  # Default to SKIP

            obs, reward, terminated, truncated, _info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        results = base_env.get_results()
        rewards.append(episode_reward)
        wall_coverages.append(results["wall_coverage_pct"])
        room_coverages.append(results["room_coverage_pct"])
        violation_counts.append(results["total_violations"])
        episode_lengths.append(steps)

        # Compute average SPUR and waste from step rewards
        step_rewards = results["step_rewards"]
        if step_rewards:
            ep_spur = np.mean([r.spur for r in step_rewards])
            ep_waste = np.mean([r.info.get("waste_percentage", 0.0) for r in step_rewards])
        else:
            ep_spur = 0.0
            ep_waste = 0.0
        spurs.append(float(ep_spur))
        waste_pcts.append(float(ep_waste))

    return EvalResult(
        num_episodes=num_episodes,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_spur=float(np.mean(spurs)),
        std_spur=float(np.std(spurs)),
        mean_waste_pct=float(np.mean(waste_pcts)),
        std_waste_pct=float(np.std(waste_pcts)),
        mean_wall_coverage_pct=float(np.mean(wall_coverages)),
        std_wall_coverage_pct=float(np.std(wall_coverages)),
        mean_room_coverage_pct=float(np.mean(room_coverages)),
        std_room_coverage_pct=float(np.std(room_coverages)),
        mean_violations=float(np.mean(violation_counts)),
        mean_episode_length=float(np.mean(episode_lengths)),
    )


# ── W&B logging callback ─────────────────────────────────────────────────


if _HAS_SB3_BASE:

    class WandBLoggingCallback(BaseCallback):
        """SB3 callback that logs episode metrics to Weights & Biases.

        Logs SPUR, waste, coverage, violations, and reward per episode
        whenever episode info is available in the training loop.
        """

        def __init__(self, verbose: int = 0) -> None:
            super().__init__(verbose)
            self._episode_count = 0

        def _on_step(self) -> bool:
            # Check for episode completion info in the buffer
            for info in self.locals.get("infos", []):
                if "reward_breakdown" in info:
                    rb = info["reward_breakdown"]
                    self._episode_count += 1
                    if _HAS_WANDB and wandb.run is not None:
                        wandb.log(
                            {
                                "episode/reward": rb.total,
                                "episode/spur": rb.spur,
                                "episode/waste": rb.waste,
                                "episode/violations": len(rb.violations),
                                "episode/coverage": rb.coverage,
                                "episode/count": self._episode_count,
                                "timestep": self.num_timesteps,
                            },
                            step=self.num_timesteps,
                        )
            return True


# ── Training ──────────────────────────────────────────────────────────────


def train_drl(
    config: DRLTrainingConfig | None = None,
    store: KnowledgeGraphStore | None = None,
    reward_weights: RewardWeights | None = None,
) -> dict[str, Any]:
    """Train the DRL policy using MaskablePPO.

    This is the main entry point for DRL training. It:
    1. Loads the Knowledge Graph
    2. Creates vectorized environments with synthetic floor plans
    3. Initializes MaskablePPO with the configured hyperparameters
    4. Trains for ``config.total_timesteps`` steps
    5. Evaluates the trained policy against the greedy baseline
    6. Saves the final model

    Args:
        config: Training configuration. Uses defaults if None.
        store: Pre-loaded KG store. Loads from default data dir if None.
        reward_weights: Custom reward weights for the environment.

    Returns:
        Dictionary with training results:
        - ``model_path``: Path to saved model checkpoint
        - ``eval_trained``: EvalResult for the trained policy
        - ``eval_greedy``: EvalResult for the greedy baseline
        - ``total_timesteps``: Total training timesteps completed
        - ``training_time_seconds``: Wall-clock training time

    Raises:
        ImportError: If sb3-contrib (MaskablePPO) is not installed.
    """
    if not _HAS_SB3:
        raise ImportError(
            "sb3-contrib is required for DRL training. Install with: pip install sb3-contrib"
        )
    if not _HAS_SB3_BASE:
        raise ImportError(
            "stable-baselines3 is required for DRL training. "
            "Install with: pip install stable-baselines3"
        )

    if config is None:
        config = DRLTrainingConfig()

    # Seed for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Load Knowledge Graph
    if store is None:
        logger.info("Loading Knowledge Graph from default data directory...")
        store = load_knowledge_graph()

    # Initialize W&B
    if config.use_wandb:
        if not _HAS_WANDB:
            logger.warning("wandb not installed; disabling W&B logging.")
            config.use_wandb = False
        else:
            wandb.init(
                project=config.wandb_project,
                config={
                    "total_timesteps": config.total_timesteps,
                    "learning_rate": config.learning_rate,
                    "n_steps": config.n_steps,
                    "batch_size": config.batch_size,
                    "n_epochs": config.n_epochs,
                    "gamma": config.gamma,
                    "gae_lambda": config.gae_lambda,
                    "clip_range": config.clip_range,
                    "ent_coef": config.ent_coef,
                    "num_envs": config.num_envs,
                    "seed": config.seed,
                },
            )

    # Create environments
    logger.info(
        "Creating %d vectorized environments with synthetic floor plans...",
        config.num_envs,
    )
    vec_env = make_vec_env(store, config, reward_weights)

    # Build callbacks
    callbacks: list[Any] = []

    # Checkpoint callback
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        CheckpointCallback(
            save_freq=max(config.checkpoint_interval // config.num_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix="drl_panelizer",
        )
    )

    # W&B callback
    if config.use_wandb:
        callbacks.append(WandBLoggingCallback())

    callback = CallbackList(callbacks)

    # Initialize MaskablePPO
    logger.info("Initializing MaskablePPO model...")
    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        verbose=1,
        seed=config.seed,
    )

    # Train
    logger.info(
        "Starting training for %d timesteps...",
        config.total_timesteps,
    )
    start_time = time.time()
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback,
        log_interval=config.log_interval,
    )
    training_time = time.time() - start_time
    logger.info("Training completed in %.1f seconds.", training_time)

    # Save final model
    final_model_path = checkpoint_dir / "drl_panelizer_final"
    model.save(str(final_model_path))
    logger.info("Final model saved to %s", final_model_path)

    # Evaluate trained policy
    logger.info(
        "Evaluating trained policy over %d episodes...",
        config.num_eval_episodes,
    )
    eval_env = make_env(store, config, config.seed + 1000, reward_weights)
    eval_trained = evaluate_policy(
        env=eval_env,
        model=model,
        num_episodes=config.num_eval_episodes,
    )
    logger.info(
        "Trained policy: reward=%.2f +/- %.2f, SPUR=%.3f, wall_cov=%.1f%%, room_cov=%.1f%%",
        eval_trained.mean_reward,
        eval_trained.std_reward,
        eval_trained.mean_spur,
        eval_trained.mean_wall_coverage_pct,
        eval_trained.mean_room_coverage_pct,
    )

    # Evaluate greedy baseline
    logger.info(
        "Evaluating greedy baseline over %d episodes...",
        config.num_eval_episodes,
    )
    greedy_env = make_env(store, config, config.seed + 2000, reward_weights)
    eval_greedy = evaluate_policy(
        env=greedy_env,
        policy_fn=greedy_policy,
        num_episodes=config.num_eval_episodes,
    )
    logger.info(
        "Greedy baseline: reward=%.2f +/- %.2f, SPUR=%.3f, wall_cov=%.1f%%, room_cov=%.1f%%",
        eval_greedy.mean_reward,
        eval_greedy.std_reward,
        eval_greedy.mean_spur,
        eval_greedy.mean_wall_coverage_pct,
        eval_greedy.mean_room_coverage_pct,
    )

    # Log final results to W&B
    if config.use_wandb and _HAS_WANDB and wandb.run is not None:
        wandb.log(
            {
                "eval/trained_reward": eval_trained.mean_reward,
                "eval/trained_spur": eval_trained.mean_spur,
                "eval/trained_wall_coverage": eval_trained.mean_wall_coverage_pct,
                "eval/trained_room_coverage": eval_trained.mean_room_coverage_pct,
                "eval/greedy_reward": eval_greedy.mean_reward,
                "eval/greedy_spur": eval_greedy.mean_spur,
                "eval/greedy_wall_coverage": eval_greedy.mean_wall_coverage_pct,
                "eval/greedy_room_coverage": eval_greedy.mean_room_coverage_pct,
                "training_time_seconds": training_time,
            }
        )
        wandb.finish()

    return {
        "model_path": str(final_model_path),
        "eval_trained": eval_trained,
        "eval_greedy": eval_greedy,
        "total_timesteps": config.total_timesteps,
        "training_time_seconds": training_time,
    }


# ── CLI entry point ───────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone training."""
    parser = argparse.ArgumentParser(
        description="Train the Axon DRL panelization/placement policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Rollout steps before each update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="PPO minibatch size.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="PPO optimization epochs per rollout.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda for advantage estimation.",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clipping range.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10_000,
        help="Save checkpoint every N timesteps.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/drl",
        help="Directory for model checkpoints.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="axon-drl",
        help="W&B project name.",
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=20,
        help="Number of episodes for evaluation.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Print training stats every N rollouts.",
    )
    parser.add_argument(
        "--num-rooms-min",
        type=int,
        default=1,
        help="Minimum number of rooms in synthetic plans.",
    )
    parser.add_argument(
        "--num-rooms-max",
        type=int,
        default=8,
        help="Maximum number of rooms in synthetic plans.",
    )
    parser.add_argument(
        "--opening-probability",
        type=float,
        default=0.3,
        help="Probability of openings on each wall.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; only evaluate greedy baseline.",
    )
    parser.add_argument(
        "--kg-data-dir",
        type=str,
        default=None,
        help="Path to Knowledge Graph data directory.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for DRL training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = _parse_args()

    config = DRLTrainingConfig(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        num_eval_episodes=args.num_eval_episodes,
        num_envs=args.num_envs,
        seed=args.seed,
        log_interval=args.log_interval,
        synthetic_num_rooms_range=(args.num_rooms_min, args.num_rooms_max),
        synthetic_opening_probability=args.opening_probability,
    )

    # Load Knowledge Graph
    store = load_knowledge_graph(args.kg_data_dir)

    if args.eval_only:
        # Evaluate greedy baseline only (no SB3 required)
        logger.info("Running greedy baseline evaluation only...")
        rng = np.random.default_rng(config.seed)
        num_rooms = rng.integers(
            config.synthetic_num_rooms_range[0],
            config.synthetic_num_rooms_range[1] + 1,
        )
        cg = generate_classified_graph(
            rng=rng,
            num_rooms=int(num_rooms),
            room_size_range_inches=config.synthetic_room_size_range_inches,
            opening_probability=config.synthetic_opening_probability,
        )
        env = RandomFloorPlanWrapper(
            classified_graph=cg,
            store=store,
            reward_weights=None,
            config=config,
            rng=rng,
        )
        result = evaluate_policy(
            env=env,
            policy_fn=greedy_policy,
            num_episodes=config.num_eval_episodes,
        )
        logger.info("Greedy baseline results:")
        logger.info("  Reward:        %.2f +/- %.2f", result.mean_reward, result.std_reward)
        logger.info("  SPUR:          %.3f +/- %.3f", result.mean_spur, result.std_spur)
        logger.info("  Waste %%:       %.1f +/- %.1f", result.mean_waste_pct, result.std_waste_pct)
        logger.info(
            "  Wall coverage: %.1f%% +/- %.1f%%",
            result.mean_wall_coverage_pct,
            result.std_wall_coverage_pct,
        )
        logger.info(
            "  Room coverage: %.1f%% +/- %.1f%%",
            result.mean_room_coverage_pct,
            result.std_room_coverage_pct,
        )
        logger.info("  Violations:    %.1f per episode", result.mean_violations)
        logger.info("  Ep length:     %.1f steps", result.mean_episode_length)
        return

    # Full training
    results = train_drl(config=config, store=store)
    logger.info("Training complete. Model saved to: %s", results["model_path"])
    logger.info("Training time: %.1f seconds", results["training_time_seconds"])


if __name__ == "__main__":
    main()
