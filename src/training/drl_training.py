"""TR-005: DRL training pipeline integration.

Wraps the existing DRL training pipeline from ``src.drl.train`` into the
unified training framework with W&B tracking via ``tracking.py`` and
checkpoint management.

Supports:
- Standard synthetic floor plan training (default)
- ResPlan dataset for diverse real-world floor plan layouts
- W&B experiment tracking with configurable project/run naming
- Checkpoint management with automatic pruning
- CLI entry point for standalone or scripted execution

Reference: TASKS.md TR-005, AGENTS.md Training Agent.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.training.tracking import CheckpointManager, ExperimentTracker

if TYPE_CHECKING:
    from docs.interfaces.classified_wall_graph import ClassifiedWallGraph
    from src.drl.reward import RewardWeights
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class DRLTrainingConfig:
    """Configuration for DRL panelization/placement training.

    Wraps ``src.drl.train.DRLTrainingConfig`` with additional tracking
    and checkpoint management settings.  Values are forwarded to the
    underlying SB3-based training loop while the tracking layer is
    managed here.

    Attributes:
        total_timesteps: Total environment steps for training.
        learning_rate: PPO learning rate.
        batch_size: Minibatch size for PPO updates.
        n_envs: Number of parallel vectorized environments.
        checkpoint_dir: Directory for saving model checkpoints.
        device: Compute device (``"auto"``, ``"cuda"``, ``"cpu"``).
        wandb_project: W&B project name.
        wandb_enabled: Whether to enable W&B logging.
        eval_freq: Evaluate the policy every N timesteps.
        num_eval_episodes: Number of episodes per evaluation round.
        resplan_path: Path to ResPlan dataset pickle file.
        use_resplan: If True, load ResPlan for diverse floor plan training.
        run_name: Optional W&B run name override.
        max_checkpoints: Maximum number of retained checkpoints.
        seed: Random seed for reproducibility.
        n_steps: Rollout steps before each PPO update.
        n_epochs: PPO optimization epochs per rollout.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_range: PPO clipping range.
        ent_coef: Entropy coefficient for exploration.
        log_interval: Print training stats every N rollouts.
    """

    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_envs: int = 4
    checkpoint_dir: str = "checkpoints/drl"
    device: str = "auto"
    wandb_project: str = "axon-drl"
    wandb_enabled: bool = True
    eval_freq: int = 10_000
    num_eval_episodes: int = 20
    resplan_path: str = "datasets/ResPlan/ResPlan.pkl"
    use_resplan: bool = True
    run_name: str | None = None
    max_checkpoints: int = 5
    seed: int = 42
    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    log_interval: int = 1


# ── ResPlan loading ──────────────────────────────────────────────────────


def _load_resplan(path: str | Path) -> list[dict[str, Any]]:
    """Load the ResPlan dataset from a pickle file.

    The ResPlan dataset contains dictionaries describing residential floor
    plans (room polygons, wall segments, dimensions).

    Args:
        path: Path to ``ResPlan.pkl``.

    Returns:
        List of floor plan dictionaries.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ResPlan dataset not found at {p}")

    logger.info("Loading ResPlan dataset from %s ...", p)
    with open(p, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        logger.info("Loaded %d floor plans from ResPlan.", len(data))
        return data

    # Handle dict-wrapped datasets
    if isinstance(data, dict) and "plans" in data:
        plans = data["plans"]
        logger.info("Loaded %d floor plans from ResPlan.", len(plans))
        return plans

    logger.warning("Unexpected ResPlan format (%s); returning as single-element list.", type(data))
    return [data]


def _resplan_to_classified_graph(
    plan: dict[str, Any],
    rng: np.random.Generator,
) -> ClassifiedWallGraph | None:
    """Convert a ResPlan floor plan dict to a ClassifiedWallGraph.

    ResPlan entries typically contain room polygons and metadata.  This
    function converts them into the FinalizedGraph → ClassifiedWallGraph
    format expected by PanelizationEnv.

    If conversion fails (e.g. missing keys, degenerate geometry), returns
    None so the caller can skip silently.

    Args:
        plan: A single ResPlan entry dictionary.
        rng: NumPy random generator for stochastic elements.

    Returns:
        ClassifiedWallGraph or None on failure.
    """
    try:
        from docs.interfaces.graph_to_serializer import (
            FinalizedGraph,
            Room,
            WallSegment,
            WallType,
        )
        from src.classifier.classifier import classify_wall_graph

        # ResPlan entries may store rooms as polygons under various keys
        rooms_data = plan.get("rooms") or plan.get("polygons") or plan.get("spaces", [])
        if not rooms_data:
            return None

        # Collect all unique vertices and build wall segments
        node_map: dict[tuple[float, float], int] = {}
        nodes_list: list[tuple[float, float]] = []
        wall_segments: list[WallSegment] = []
        edge_set: set[tuple[int, int]] = set()
        edge_list: list[tuple[int, int]] = []
        rooms: list[Room] = []

        def _get_node(x: float, y: float) -> int:
            key = (round(x, 2), round(y, 2))
            if key not in node_map:
                node_map[key] = len(nodes_list)
                nodes_list.append((x, y))
            return node_map[key]

        # Build rooms and walls from polygon vertices
        for room_idx, room_data in enumerate(rooms_data):
            vertices = room_data.get("vertices") or room_data.get("polygon", [])
            if len(vertices) < 3:
                continue

            boundary_nodes: list[int] = []
            boundary_edges: list[int] = []

            for i in range(len(vertices)):
                x0, y0 = float(vertices[i][0]), float(vertices[i][1])
                x1, y1 = (
                    float(vertices[(i + 1) % len(vertices)][0]),
                    float(vertices[(i + 1) % len(vertices)][1]),
                )

                n0 = _get_node(x0, y0)
                n1 = _get_node(x1, y1)
                boundary_nodes.append(n0)

                edge_key = (min(n0, n1), max(n0, n1))
                if edge_key not in edge_set:
                    edge_idx = len(wall_segments)
                    edge_set.add(edge_key)
                    edge_list.append(edge_key)

                    start = np.array([x0, y0])
                    end = np.array([x1, y1])
                    delta = end - start
                    length = float(np.linalg.norm(delta))
                    angle = float(np.arctan2(delta[1], delta[0]) % np.pi)

                    # Assign wall type heuristically
                    wt = rng.choice([WallType.LOAD_BEARING, WallType.PARTITION, WallType.EXTERIOR])
                    thickness = rng.uniform(4.0, 8.0)

                    wall_segments.append(
                        WallSegment(
                            edge_id=edge_idx,
                            start_node=edge_key[0],
                            end_node=edge_key[1],
                            start_coord=start.copy(),
                            end_coord=end.copy(),
                            thickness=thickness,
                            height=2700.0,
                            wall_type=wt,
                            angle=angle,
                            length=length,
                            confidence=0.9,
                        )
                    )
                    boundary_edges.append(edge_idx)
                else:
                    # Find existing edge index
                    idx = edge_list.index(edge_key)
                    boundary_edges.append(idx)

            label = room_data.get("label", room_data.get("type", ""))
            area = room_data.get("area", 0.0)

            rooms.append(
                Room(
                    room_id=room_idx,
                    boundary_edges=boundary_edges,
                    boundary_nodes=boundary_nodes,
                    area=float(area),
                    label=str(label),
                    is_exterior=False,
                )
            )

        if not wall_segments or not rooms:
            return None

        nodes = np.array(nodes_list, dtype=np.float64)
        edges = (
            np.array(edge_list, dtype=np.int64) if edge_list else np.empty((0, 2), dtype=np.int64)
        )

        # Compute page bounds
        page_width = float(nodes[:, 0].max()) + 100.0
        page_height = float(nodes[:, 1].max()) + 100.0

        graph = FinalizedGraph(
            nodes=nodes,
            edges=edges,
            wall_segments=wall_segments,
            openings=[],
            rooms=rooms,
            page_width=page_width,
            page_height=page_height,
            scale_factor=1.0,
        )

        return classify_wall_graph(graph)

    except Exception:
        logger.debug("Failed to convert ResPlan entry to ClassifiedWallGraph.", exc_info=True)
        return None


# ── DRL Training Pipeline ───────────────────────────────────────────────


class DRLTrainingPipeline:
    """Unified DRL training pipeline with tracking and checkpoint management.

    Wraps ``src.drl.train.train_drl`` with:
    - ``ExperimentTracker`` for W&B + local JSONL logging
    - ``CheckpointManager`` for versioned checkpoint persistence
    - Optional ResPlan dataset loading for diverse environments

    Usage::

        config = DRLTrainingConfig(total_timesteps=500_000)
        pipeline = DRLTrainingPipeline(config)
        pipeline.train()
        results = pipeline.evaluate()

    Args:
        config: DRL training configuration.
        store: Pre-loaded Knowledge Graph store. Loads defaults if None.
        reward_weights: Custom reward weights for the environment.
    """

    def __init__(
        self,
        config: DRLTrainingConfig,
        store: KnowledgeGraphStore | None = None,
        reward_weights: RewardWeights | None = None,
        extra_callbacks: list | None = None,
    ) -> None:
        self.config = config
        self._store = store
        self._reward_weights = reward_weights
        self._extra_callbacks = extra_callbacks or []
        self._model: Any = None
        self._training_results: dict[str, Any] | None = None

        # Tracking
        run_name = config.run_name or f"drl-{config.total_timesteps // 1000}k-{config.seed}"
        self._tracker = ExperimentTracker(
            project=config.wandb_project,
            run_name=run_name,
            config=asdict(config),
            enabled=config.wandb_enabled,
            log_dir=Path(config.checkpoint_dir) / "logs",
        )

        # Checkpoint management
        self._ckpt_mgr = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            max_checkpoints=config.max_checkpoints,
        )

        # ResPlan floor plans (optional)
        self._resplan_graphs: list[ClassifiedWallGraph] = []
        if config.use_resplan:
            self._load_resplan_dataset()

    def _load_resplan_dataset(self) -> None:
        """Attempt to load ResPlan and convert entries to ClassifiedWallGraphs."""
        resplan_path = Path(self.config.resplan_path)
        if not resplan_path.exists():
            logger.info(
                "ResPlan dataset not found at %s — using synthetic plans only.",
                resplan_path,
            )
            return

        try:
            plans = _load_resplan(resplan_path)
            rng = np.random.default_rng(self.config.seed)
            converted = 0
            for plan in plans:
                cg = _resplan_to_classified_graph(plan, rng)
                if cg is not None:
                    self._resplan_graphs.append(cg)
                    converted += 1
            logger.info(
                "Converted %d / %d ResPlan entries to ClassifiedWallGraph.",
                converted,
                len(plans),
            )
        except Exception:
            logger.warning("Failed to load ResPlan dataset.", exc_info=True)

    def _get_store(self) -> KnowledgeGraphStore:
        """Lazily load the Knowledge Graph store."""
        if self._store is None:
            from src.knowledge_graph.loader import load_knowledge_graph

            logger.info("Loading Knowledge Graph from default data directory...")
            self._store = load_knowledge_graph()
        return self._store

    def train(self) -> None:
        """Run the full DRL training pipeline.

        Delegates to ``src.drl.train.train_drl`` with a configuration
        translated from ``DRLTrainingConfig``, then logs final metrics
        and saves a managed checkpoint.
        """
        from src.drl.train import DRLTrainingConfig as BaseDRLConfig
        from src.drl.train import train_drl

        store = self._get_store()

        # Translate to base config
        base_config = BaseDRLConfig(
            total_timesteps=self.config.total_timesteps,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            checkpoint_interval=self.config.eval_freq,
            checkpoint_dir=self.config.checkpoint_dir,
            # W&B is handled by our tracker — disable the base train_drl's W&B
            use_wandb=False,
            wandb_project=self.config.wandb_project,
            num_eval_episodes=self.config.num_eval_episodes,
            num_envs=self.config.n_envs,
            seed=self.config.seed,
            log_interval=self.config.log_interval,
        )

        logger.info("Starting DRL training for %d timesteps...", self.config.total_timesteps)
        start_time = time.time()

        results = train_drl(
            config=base_config,
            store=store,
            reward_weights=self._reward_weights,
            extra_callbacks=self._extra_callbacks,
        )

        training_time = time.time() - start_time
        self._training_results = results

        # Log final metrics via tracker
        eval_trained = results["eval_trained"]
        eval_greedy = results["eval_greedy"]

        final_metrics = {
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
        self._tracker.log(final_metrics, step=self.config.total_timesteps)

        # Save a managed checkpoint
        ckpt_state = {
            "model_path": results["model_path"],
            "total_timesteps": results["total_timesteps"],
            "training_time_seconds": training_time,
            "config": asdict(self.config),
        }
        self._ckpt_mgr.save(
            state=ckpt_state,
            epoch=self.config.total_timesteps,
            metrics={
                "reward": eval_trained.mean_reward,
                "spur": eval_trained.mean_spur,
                "wall_coverage": eval_trained.mean_wall_coverage_pct,
                "room_coverage": eval_trained.mean_room_coverage_pct,
            },
        )

        # Log model artifact
        if results.get("model_path"):
            self._tracker.log_artifact(
                path=results["model_path"],
                name="drl-panelizer",
                type="model",
            )

        logger.info(
            "DRL training complete in %.1fs. Trained reward=%.2f, Greedy reward=%.2f",
            training_time,
            eval_trained.mean_reward,
            eval_greedy.mean_reward,
        )

    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the most recently trained model.

        Returns:
            Dictionary of evaluation metrics.

        Raises:
            RuntimeError: If ``train()`` has not been called yet.
        """
        if self._training_results is None:
            raise RuntimeError("No training results available. Call train() first.")

        eval_trained = self._training_results["eval_trained"]
        eval_greedy = self._training_results["eval_greedy"]

        return {
            "trained_reward": eval_trained.mean_reward,
            "trained_reward_std": eval_trained.std_reward,
            "trained_spur": eval_trained.mean_spur,
            "trained_wall_coverage_pct": eval_trained.mean_wall_coverage_pct,
            "trained_room_coverage_pct": eval_trained.mean_room_coverage_pct,
            "trained_violations": eval_trained.mean_violations,
            "greedy_reward": eval_greedy.mean_reward,
            "greedy_spur": eval_greedy.mean_spur,
            "greedy_wall_coverage_pct": eval_greedy.mean_wall_coverage_pct,
            "greedy_room_coverage_pct": eval_greedy.mean_room_coverage_pct,
            "greedy_violations": eval_greedy.mean_violations,
            "reward_improvement": eval_trained.mean_reward - eval_greedy.mean_reward,
            "spur_improvement": eval_trained.mean_spur - eval_greedy.mean_spur,
        }

    def close(self) -> None:
        """Finalise tracking and clean up resources."""
        self._tracker.finish()


# ── CLI entry point ──────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the DRL training wrapper."""
    parser = argparse.ArgumentParser(
        description="Axon DRL training pipeline (panelization + placement).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="PPO minibatch size.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/drl",
        help="Directory for model checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="axon-drl",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-enabled",
        action="store_true",
        default=False,
        help="Enable W&B logging.",
    )
    parser.add_argument(
        "--no-wandb",
        dest="wandb_enabled",
        action="store_false",
        help="Disable W&B logging.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluate every N timesteps.",
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=20,
        help="Number of episodes per evaluation.",
    )
    parser.add_argument(
        "--resplan-path",
        type=str,
        default="datasets/ResPlan/ResPlan.pkl",
        help="Path to ResPlan dataset pickle.",
    )
    parser.add_argument(
        "--use-resplan",
        action="store_true",
        default=True,
        help="Use ResPlan for diverse training environments.",
    )
    parser.add_argument(
        "--no-resplan",
        dest="use_resplan",
        action="store_false",
        help="Disable ResPlan; use only synthetic plans.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if not set).",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=5,
        help="Maximum number of retained checkpoints.",
    )
    parser.add_argument(
        "--kg-data-dir",
        type=str,
        default=None,
        help="Path to Knowledge Graph data directory.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for the DRL training pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = _parse_args()

    config = DRLTrainingConfig(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_envs=args.n_envs,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_enabled=args.wandb_enabled,
        eval_freq=args.eval_freq,
        num_eval_episodes=args.num_eval_episodes,
        resplan_path=args.resplan_path,
        use_resplan=args.use_resplan,
        seed=args.seed,
        run_name=args.run_name,
        max_checkpoints=args.max_checkpoints,
    )

    # Load KG
    store = None
    if args.kg_data_dir:
        from src.knowledge_graph.loader import load_knowledge_graph

        store = load_knowledge_graph(args.kg_data_dir)

    pipeline = DRLTrainingPipeline(config=config, store=store)
    try:
        pipeline.train()
        results = pipeline.evaluate()

        logger.info("=== DRL Training Results ===")
        logger.info(
            "  Trained reward:       %.2f +/- %.2f",
            results["trained_reward"],
            results["trained_reward_std"],
        )
        logger.info("  Trained SPUR:         %.3f", results["trained_spur"])
        logger.info("  Trained wall cov:     %.1f%%", results["trained_wall_coverage_pct"])
        logger.info("  Trained room cov:     %.1f%%", results["trained_room_coverage_pct"])
        logger.info("  Greedy reward:        %.2f", results["greedy_reward"])
        logger.info("  Reward improvement:   %.2f", results["reward_improvement"])
        logger.info("  SPUR improvement:     %.3f", results["spur_improvement"])
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
