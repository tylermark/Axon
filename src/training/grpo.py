"""TR-004: Group Relative Policy Optimization (GRPO) quality annealing.

Refines the SFT-trained model using reward-based optimization. For each
input floor plan, generates a group of candidate output graphs, scores
them with a composite geometric reward, and updates the model to increase
the probability of higher-reward outputs while penalizing divergence
from the frozen SFT reference model.

Algorithm per iteration:
    1. For each input, sample ``group_size`` output graphs via temperature
       sampling through the diffusion model.
    2. Score each output with a composite reward (default: coord_mse +
       adj_f1; expensive metrics like HIoU/GED/Betti available for eval).
    3. Compute group-relative advantages (normalize rewards within group).
    4. Policy gradient update clipped to ``clip_ratio``.
    5. KL divergence penalty against the frozen SFT reference model.

Reference:
    ARCHITECTURE.md -- Training Pipeline (SSL -> SFT -> GRPO)
    MODEL_SPEC.md   -- Quality Annealing
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as nnf
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from src.diffusion.reverse import GraphDiffusionModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional W&B import
# ---------------------------------------------------------------------------

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    """Configuration for GRPO quality annealing."""

    learning_rate: float = 1e-5
    """Learning rate for policy updates (lower than SFT)."""

    batch_size: int = 4
    """Number of floor plans per iteration."""

    num_iterations: int = 1000
    """Total GRPO training iterations."""

    group_size: int = 8
    """Number of candidate samples generated per input for relative ranking."""

    kl_coeff: float = 0.1
    """KL divergence penalty coefficient against the reference model."""

    reward_weights: dict[str, float] = field(
        default_factory=lambda: {"coord_mse": 0.5, "adj_f1": 0.5}
    )
    """Weights for the composite reward function components.

    Fast (use for training): coord_mse, adj_f1.
    Expensive (eval only): hiou, ged, betti.
    """

    coord_scale: float = 1.0
    """Coordinate scale for MSE normalization in coord_mse reward.

    Set this to the typical magnitude of your coordinate system:
    - For normalized coordinates in [0, 1]: use 1.0 (default)
    - For pixel coordinates in [0, 512]: use 512.0
    - For pixel coordinates in [0, 1024]: use 1024.0

    The MSE is normalized by (coord_scale ** 2) before computing the reward.
    """

    checkpoint_dir: str = "checkpoints/grpo"
    """Directory for saving checkpoints."""

    device: str = "auto"
    """Compute device: 'auto' detects cuda/mps/cpu."""

    wandb_project: str = "axon-grpo"
    """Weights & Biases project name."""

    wandb_enabled: bool = True
    """Enable W&B logging (silently disabled if wandb not installed)."""

    temperature: float = 1.0
    """Sampling temperature for output diversity. Higher = more diverse."""

    clip_ratio: float = 0.2
    """PPO-style clipping ratio for policy gradient updates."""

    max_ged: float = 50.0
    """Maximum GED for reward normalization. GED values above this are clipped.
    Only used when ``ged`` has nonzero weight."""

    coord_scale: float = 1.0
    """Expected coordinate range for coord_mse normalization.

    Must be > 0.  MSE is divided by coord_scale**2 before forming the
    reward, so the reward stays in [0, 1] regardless of coordinate units.
    For data normalised to [0, 1] the default of 1.0 is correct.  Set to
    the bounding-box diagonal (or similar) for unnormalised coordinates."""

    def __post_init__(self) -> None:
        import math

        if (
            isinstance(self.coord_scale, bool)
            or not isinstance(self.coord_scale, (int, float))
            or not math.isfinite(self.coord_scale)
            or self.coord_scale <= 0
        ):
            raise ValueError(
                f"coord_scale must be a positive number, got {self.coord_scale!r}"
            )

    gradient_clip: float = 1.0
    """Max gradient norm for clipping."""

    num_workers: int = 0
    """DataLoader worker processes."""

    seed: int = 42
    """Random seed for reproducibility."""

    log_every_n_iterations: int = 10
    """Log metrics every N iterations."""

    save_every_n_iterations: int = 100
    """Save checkpoint every N iterations."""


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> torch.device:
    """Resolve device string to torch.device, handling 'auto'."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------


def compute_composite_reward(
    pred_nodes: Any,
    pred_edges: Any,
    gt_nodes: Any,
    gt_edges: Any,
    weights: dict[str, float],
    max_ged: float = 50.0,
    wall_thickness: float = 2.0,
    coord_scale: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Compute composite geometric quality reward for a single sample.

    Supports both fast (training-friendly) and expensive (eval-only) metrics:

    Fast metrics (sub-millisecond, safe for GRPO inner loop):
        - ``coord_mse``: ``max(0, 1 - MSE / coord_scale²)``.  Normalised
          by ``coord_scale`` so the reward stays in [0, 1] regardless of
          coordinate units.
        - ``adj_f1``: F1 score between predicted and GT adjacency matrices.

    Expensive metrics (seconds-to-minutes per sample, use for eval only):
        - ``hiou``: Hungarian-matched IoU over wall segments.  O(N_pred * N_gt)
          Shapely polygon intersections — prohibitive for >100 edges.
        - ``ged``: Exact graph edit distance via networkx.  Exponential time.
        - ``betti``: Betti number error (connected components + cycles).

    Args:
        pred_nodes: Predicted node positions, shape (N, 2).
        pred_edges: Predicted edges, shape (E, 2).
        gt_nodes: Ground truth node positions, shape (N_gt, 2).
        gt_edges: Ground truth edges, shape (E_gt, 2).
        weights: Component weights.  Keys with weight 0 are skipped.
        max_ged: Maximum GED for normalization.
        wall_thickness: Default wall thickness for HIoU computation.
        coord_scale: Coordinate scale for MSE normalization (default 1.0).

    Returns:
        Tuple of (total_reward, component_dict) with per-component rewards.
    """
    import numpy as np

    components: dict[str, float] = {}
    total = 0.0

    # --- coord_mse (fast): 1 - normalised MSE between node positions ---
    coord_mse_weight = weights.get("coord_mse", 0.0)
    if coord_mse_weight > 0:
        n = min(len(pred_nodes), len(gt_nodes))
        if n > 0:
            mse = float(np.mean((pred_nodes[:n] - gt_nodes[:n]) ** 2))
            mse_normalized = mse / (coord_scale ** 2)
            coord_reward = max(0.0, 1.0 - mse_normalized)
        else:
            coord_reward = 0.0
        components["coord_mse"] = coord_reward
        total += coord_mse_weight * coord_reward

    # --- adj_f1 (fast): F1 between predicted and GT adjacency ---
    adj_f1_weight = weights.get("adj_f1", 0.0)
    if adj_f1_weight > 0:
        n = min(len(pred_nodes), len(gt_nodes))
        if n > 0:
            # Build dense adjacency for the shared node set
            pred_adj = np.zeros((n, n), dtype=np.float32)
            for s, d in pred_edges:
                if s < n and d < n:
                    pred_adj[int(s), int(d)] = 1.0
                    pred_adj[int(d), int(s)] = 1.0
            gt_adj = np.zeros((n, n), dtype=np.float32)
            for s, d in gt_edges:
                if s < n and d < n:
                    gt_adj[int(s), int(d)] = 1.0
                    gt_adj[int(d), int(s)] = 1.0
            # Upper triangle only
            triu = np.triu_indices(n, k=1)
            p = pred_adj[triu]
            g = gt_adj[triu]
            tp = float(np.sum(p * g))
            fp = float(np.sum(p * (1 - g)))
            fn = float(np.sum((1 - p) * g))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            f1 = 0.0
        components["adj_f1"] = f1
        total += adj_f1_weight * f1

    # --- HIoU (expensive: O(E_pred * E_gt) Shapely calls) ---
    hiou_weight = weights.get("hiou", 0.0)
    if hiou_weight > 0:
        try:
            from tests.benchmarks.test_hiou import compute_hiou

            pred_walls = [
                (pred_nodes[int(s)], pred_nodes[int(d)], wall_thickness) for s, d in pred_edges
            ]
            gt_walls = [(gt_nodes[int(s)], gt_nodes[int(d)], wall_thickness) for s, d in gt_edges]
            hiou = compute_hiou(pred_walls, gt_walls) if pred_walls and gt_walls else 0.0
            components["hiou"] = hiou
            total += hiou_weight * hiou
        except Exception:
            components["hiou"] = 0.0

    # --- GED (expensive: exponential time via networkx) ---
    ged_weight = weights.get("ged", 0.0)
    if ged_weight > 0:
        try:
            from tests.benchmarks.test_ged import compute_ged

            ged = compute_ged(pred_nodes, pred_edges, gt_nodes, gt_edges)
            ged_reward = max(0.0, 1.0 - ged / max_ged)
            components["ged"] = ged
            components["ged_reward"] = ged_reward
            total += ged_weight * ged_reward
        except Exception:
            components["ged"] = max_ged
            components["ged_reward"] = 0.0

    # --- Betti error ---
    betti_weight = weights.get("betti", 0.0)
    if betti_weight > 0:
        try:
            from tests.benchmarks.test_betti import (
                compute_betti_error,
                compute_betti_numbers,
            )

            gt_b0, gt_b1 = compute_betti_numbers(gt_nodes, gt_edges)
            err_b0, err_b1 = compute_betti_error(pred_nodes, pred_edges, gt_b0, gt_b1)
            # Normalize: betti error of 0 -> reward 1, large error -> reward 0
            betti_error = (err_b0 + err_b1) / max(gt_b0 + gt_b1, 1)
            betti_reward = max(0.0, 1.0 - betti_error)
            components["betti_0_error"] = float(err_b0)
            components["betti_1_error"] = float(err_b1)
            components["betti_reward"] = betti_reward
            total += betti_weight * betti_reward
        except Exception:
            components["betti_reward"] = 0.0

    components["total"] = total
    return total, components


# ---------------------------------------------------------------------------
# GRPOTrainer
# ---------------------------------------------------------------------------


class GRPOTrainer:
    """Group Relative Policy Optimization trainer for quality annealing.

    After SFT produces a competent base model, GRPO refines it by generating
    groups of candidate outputs, ranking them by geometric quality rewards,
    and updating the policy to favor higher-reward generations.

    The frozen SFT checkpoint serves as the reference model for KL penalty
    computation, preventing reward hacking / mode collapse.

    Args:
        model: The active diffusion model being optimized.
        reference_model: Frozen copy of the SFT checkpoint (for KL penalty).
        dataset: Training dataset yielding floor plan batches.
        config: GRPO configuration.
    """

    def __init__(
        self,
        model: GraphDiffusionModel,
        reference_model: GraphDiffusionModel,
        dataset: Dataset,  # type: ignore[type-arg]
        config: GRPOConfig,
    ) -> None:
        self.config = config
        self.device = _resolve_device(config.device)

        # Active model (updated during training)
        self.model = model.to(self.device)

        # Reference model (frozen SFT checkpoint, never updated)
        self.reference_model = reference_model.to(self.device)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad_(False)

        # Dataset
        self.dataset = dataset
        self.data_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        # Training state
        self.current_iteration = 0
        self.global_step = 0
        self.metrics_history: list[dict[str, float]] = []

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # W&B initialization
        self._wandb_run = None
        if config.wandb_enabled and _WANDB_AVAILABLE:
            self._wandb_run = wandb.init(
                project=config.wandb_project,
                config={
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "num_iterations": config.num_iterations,
                    "group_size": config.group_size,
                    "kl_coeff": config.kl_coeff,
                    "clip_ratio": config.clip_ratio,
                    "temperature": config.temperature,
                    "reward_weights": config.reward_weights,
                    "coord_scale": config.coord_scale,
                },
                resume="allow",
            )

    # ------------------------------------------------------------------
    # Reward computation for a batch
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        predicted_graph: dict[str, torch.Tensor],
        ground_truth: dict[str, torch.Tensor],
    ) -> float:
        """Compute the composite reward for a single predicted graph.

        Args:
            predicted_graph: Dict with 'node_positions' (N, 2) and
                'adjacency' (N, N) tensors.
            ground_truth: Dict with 'x_0' (N, 2) and 'a_0' (N, N) tensors.

        Returns:
            Scalar reward value.
        """
        pred_nodes = predicted_graph["node_positions"].cpu().numpy()
        pred_adj = predicted_graph["adjacency"].cpu().numpy()
        gt_nodes = ground_truth["x_0"].cpu().numpy()
        gt_adj = ground_truth["a_0"].cpu().numpy()

        pred_edges = _adj_to_edge_list(pred_adj)
        gt_edges = _adj_to_edge_list(gt_adj)

        reward, _ = compute_composite_reward(
            pred_nodes,
            pred_edges,
            gt_nodes,
            gt_edges,
            weights=self.config.reward_weights,
            max_ged=self.config.max_ged,
            coord_scale=self.config.coord_scale,
        )
        return reward

    # ------------------------------------------------------------------
    # Sample generation with temperature
    # ------------------------------------------------------------------

    def _sample_group(
        self,
        model: GraphDiffusionModel,
        num_nodes: int,
        context: torch.Tensor | None,
        node_mask: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        group_size: int,
    ) -> list[dict[str, torch.Tensor]]:
        """Generate a group of candidate graphs from the model.

        Each sample uses stochastic DDIM (eta > 0) with temperature scaling
        to produce diverse candidates.

        Args:
            model: Diffusion model to sample from.
            num_nodes: Number of nodes per graph.
            context: (1, N_ctx, d_ctx) context embeddings.
            node_mask: (1, N) valid node mask.
            context_mask: (1, N_ctx) context mask.
            group_size: Number of samples to generate.

        Returns:
            List of dicts, each with 'node_positions' and 'adjacency'.
        """
        samples = []
        for _ in range(group_size):
            with torch.no_grad():
                result = model.reverse.ddim_sample(
                    num_nodes=num_nodes,
                    batch_size=1,
                    num_steps=model.config.timesteps_inference,
                    context=context,
                    node_mask=node_mask,
                    context_mask=context_mask,
                    eta=self.config.temperature,  # Stochasticity via eta
                    hdse=model.hdse,
                    device=self.device,
                )

            positions = result.node_positions.squeeze(0)  # (N, 2)
            adj = (torch.sigmoid(result.adjacency_logits.squeeze(0)) > 0.5).float()

            samples.append(
                {
                    "node_positions": positions,
                    "adjacency": adj,
                    "adjacency_logits": result.adjacency_logits.squeeze(0),
                }
            )

        return samples

    # ------------------------------------------------------------------
    # KL divergence estimation
    # ------------------------------------------------------------------

    def _compute_kl_divergence(
        self,
        x_0: torch.Tensor,
        a_0: torch.Tensor,
        context: torch.Tensor | None,
        node_mask: torch.Tensor | None,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Estimate KL divergence between current model and reference model.

        Uses the VLB loss difference as a proxy for KL divergence:
        KL ~ L_current - L_reference. This is an approximation since both
        models are evaluated on the same noised samples.

        Args:
            x_0: (B, N, 2) clean node coordinates.
            a_0: (B, N, N) clean adjacency.
            context: Cross-modal context.
            node_mask: Valid node mask.
            context_mask: Context mask.

        Returns:
            Scalar KL divergence estimate.
        """
        # Current model loss
        current_loss = self.model(x_0, a_0, context, node_mask, context_mask)

        # Reference model loss
        with torch.no_grad():
            ref_loss = self.reference_model(x_0, a_0, context, node_mask, context_mask)

        # KL proxy: positive when current model diverges from reference
        kl = nnf.relu(current_loss.total - ref_loss.total)
        return kl

    # ------------------------------------------------------------------
    # Single GRPO iteration
    # ------------------------------------------------------------------

    def train_iteration(self) -> dict[str, float]:
        """Execute a single GRPO training iteration.

        For each floor plan in the batch:
        1. Generate ``group_size`` candidate graphs
        2. Score each candidate with the composite reward
        3. Compute group-relative advantages
        4. Update model via clipped policy gradient + KL penalty

        Returns:
            Dict of per-iteration metrics.
        """
        self.model.train()

        # Get next batch from data loader
        batch = next(self._data_iter)

        x_0 = batch["x_0"].to(self.device)
        a_0 = batch["a_0"].to(self.device)
        node_mask = batch.get("node_mask")
        if node_mask is not None:
            node_mask = node_mask.to(self.device)

        bsz, n_nodes, _ = x_0.shape

        # Context from batch (pre-computed or None)
        context = batch.get("context")
        context_mask = batch.get("context_mask")
        if context is not None:
            context = context.to(self.device)
        if context_mask is not None:
            context_mask = context_mask.to(self.device)

        all_rewards: list[float] = []
        all_advantages: list[float] = []
        policy_losses: list[torch.Tensor] = []

        import numpy as np

        for b in range(bsz):
            # Extract single sample
            x_0_b = x_0[b : b + 1]
            a_0_b = a_0[b : b + 1]
            mask_b = node_mask[b : b + 1] if node_mask is not None else None
            ctx_b = context[b : b + 1] if context is not None else None
            ctx_mask_b = context_mask[b : b + 1] if context_mask is not None else None

            n_valid = int(mask_b.sum()) if mask_b is not None else n_nodes

            # 1. Generate group of candidates
            samples = self._sample_group(
                self.model, n_nodes, ctx_b, mask_b, ctx_mask_b, self.config.group_size
            )

            # Ground truth for reward computation
            gt_nodes = x_0_b.squeeze(0)[:n_valid].cpu().numpy()
            gt_adj = a_0_b.squeeze(0)[:n_valid, :n_valid].cpu().numpy()
            gt_edges = _adj_to_edge_list(gt_adj)

            # 2. Score each candidate
            rewards = []
            for sample in samples:
                pred_nodes = sample["node_positions"][:n_valid].cpu().numpy()
                pred_adj = sample["adjacency"][:n_valid, :n_valid].cpu().numpy()
                pred_edges = _adj_to_edge_list(pred_adj)

                r, _ = compute_composite_reward(
                    pred_nodes,
                    pred_edges,
                    gt_nodes,
                    gt_edges,
                    weights=self.config.reward_weights,
                    max_ged=self.config.max_ged,
                    coord_scale=self.config.coord_scale,
                )
                rewards.append(r)

            all_rewards.extend(rewards)

            # 3. Compute group-relative advantages (normalize within group)
            rewards_arr = np.array(rewards)
            mean_r = rewards_arr.mean()
            std_r = rewards_arr.std() + 1e-8
            advantages = (rewards_arr - mean_r) / std_r
            all_advantages.extend(advantages.tolist())

            # 4. Policy gradient with advantage weighting
            # Use the diffusion VLB loss as a proxy for log-probability.
            # Higher reward samples should have lower loss (model should
            # assign higher probability to them).
            for sample, advantage in zip(samples, advantages, strict=True):
                # Reconstruct the predicted positions as targets
                pred_positions = sample["node_positions"].unsqueeze(0).to(self.device)
                pred_adj_logits = sample["adjacency_logits"].unsqueeze(0).to(self.device)

                # VLB loss on the predicted graph (how likely is this output?)
                loss = self.model(
                    pred_positions,
                    (pred_adj_logits > 0).float(),
                    ctx_b,
                    mask_b,
                    ctx_mask_b,
                )

                # Policy gradient: minimize loss for high-advantage, maximize for low
                # Clipped objective for stability
                advantage_t = torch.tensor(advantage, device=self.device, dtype=torch.float32)
                clipped_adv = torch.clamp(
                    advantage_t,
                    -self.config.clip_ratio,
                    self.config.clip_ratio,
                )

                # Negative advantage * loss: high reward -> decrease loss,
                # low reward -> increase loss (but clipped)
                policy_loss = -clipped_adv * loss.total
                policy_losses.append(policy_loss)

        # --- Aggregate policy loss ---
        if not policy_losses:
            return {"reward/mean": 0.0, "reward/std": 0.0}

        total_policy_loss = torch.stack(policy_losses).mean()

        # --- KL divergence penalty ---
        kl_div = self._compute_kl_divergence(x_0, a_0, context, node_mask, context_mask)
        kl_penalty = self.config.kl_coeff * kl_div

        # --- Total GRPO loss ---
        grpo_loss = total_policy_loss + kl_penalty

        # --- Backward + optimizer step ---
        self.optimizer.zero_grad()
        grpo_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.gradient_clip,
        )

        self.optimizer.step()
        self.global_step += 1

        # --- Metrics ---
        rewards_np = np.array(all_rewards)
        advantages_np = np.array(all_advantages)

        metrics: dict[str, float] = {
            "reward/mean": float(rewards_np.mean()),
            "reward/std": float(rewards_np.std()),
            "reward/min": float(rewards_np.min()),
            "reward/max": float(rewards_np.max()),
            "advantage/mean": float(advantages_np.mean()),
            "advantage/std": float(advantages_np.std()),
            "loss/policy": total_policy_loss.item(),
            "loss/kl": kl_div.item(),
            "loss/kl_penalty": kl_penalty.item(),
            "loss/grpo_total": grpo_loss.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "iteration": float(self.current_iteration),
        }

        # W&B logging
        if self._wandb_run is not None:
            wandb.log(metrics, step=self.global_step)

        return metrics

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full GRPO quality annealing loop.

        Iterates over the dataset for ``num_iterations`` steps, generating
        sample groups, computing rewards, and updating the policy.
        """
        logger.info(
            "Starting GRPO training: %d iterations, lr=%.1e, "
            "group_size=%d, kl_coeff=%.3f, device=%s",
            self.config.num_iterations,
            self.config.learning_rate,
            self.config.group_size,
            self.config.kl_coeff,
            self.device,
        )

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        # Create an infinite data iterator
        self._data_iter = _InfiniteDataIterator(self.data_loader)

        for iteration in range(self.current_iteration, self.config.num_iterations):
            self.current_iteration = iteration
            iter_start = time.monotonic()

            metrics = self.train_iteration()
            metrics["iter_time_s"] = time.monotonic() - iter_start

            self.metrics_history.append(metrics)

            # Logging
            if (iteration + 1) % self.config.log_every_n_iterations == 0:
                logger.info(
                    "Iter %d/%d  reward=%.4f +/- %.4f  policy=%.4f  kl=%.4f  total=%.4f  (%.1fs)",
                    iteration + 1,
                    self.config.num_iterations,
                    metrics.get("reward/mean", 0.0),
                    metrics.get("reward/std", 0.0),
                    metrics.get("loss/policy", 0.0),
                    metrics.get("loss/kl", 0.0),
                    metrics.get("loss/grpo_total", 0.0),
                    metrics.get("iter_time_s", 0.0),
                )

            # Checkpoint
            if (iteration + 1) % self.config.save_every_n_iterations == 0:
                self.save_checkpoint(f"iter_{iteration + 1:06d}.pt")

        # Final checkpoint
        self.save_checkpoint("final.pt")
        logger.info("GRPO training complete after %d iterations.", self.config.num_iterations)

        if self._wandb_run is not None:
            wandb.finish()

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str) -> Path:
        """Save training state to a checkpoint file.

        Args:
            filename: Checkpoint filename (saved under checkpoint_dir).

        Returns:
            Full path to the saved checkpoint.
        """
        path = self.checkpoint_dir / filename
        checkpoint = {
            "iteration": self.current_iteration,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, path)
        logger.info("GRPO checkpoint saved: %s", path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume GRPO training from a checkpoint.

        Note: The reference model is NOT loaded from GRPO checkpoints.
        It must be set separately from the original SFT checkpoint.

        Args:
            path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.current_iteration = checkpoint["iteration"] + 1
        self.global_step = checkpoint["global_step"]

        logger.info(
            "Resumed GRPO from checkpoint: %s (iteration %d, step %d)",
            path,
            self.current_iteration,
            self.global_step,
        )

    @classmethod
    def from_sft_checkpoint(
        cls,
        sft_checkpoint_path: str | Path,
        diffusion_config: Any,
        dataset: Dataset,  # type: ignore[type-arg]
        grpo_config: GRPOConfig,
    ) -> GRPOTrainer:
        """Create a GRPOTrainer from a saved SFT checkpoint.

        Loads the SFT model weights into both the active model and the
        frozen reference model.

        Args:
            sft_checkpoint_path: Path to the SFT checkpoint file.
            diffusion_config: DiffusionConfig for model construction.
            dataset: Training dataset.
            grpo_config: GRPO configuration.

        Returns:
            Initialized GRPOTrainer with SFT weights loaded.
        """
        from src.diffusion.reverse import (
            GraphDiffusionModel,
        )

        device = _resolve_device(grpo_config.device)
        checkpoint = torch.load(sft_checkpoint_path, map_location=device, weights_only=False)

        # Build active model
        model = GraphDiffusionModel(diffusion_config)
        model.load_state_dict(checkpoint["diffusion_state_dict"])

        # Build frozen reference model (deep copy of SFT weights)
        reference_model = GraphDiffusionModel(diffusion_config)
        reference_model.load_state_dict(checkpoint["diffusion_state_dict"])

        return cls(
            model=model,
            reference_model=reference_model,
            dataset=dataset,
            config=grpo_config,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class _InfiniteDataIterator:
    """Wraps a DataLoader for infinite iteration with automatic reset.

    When the underlying DataLoader is exhausted, it restarts from the
    beginning automatically.
    """

    def __init__(self, data_loader: DataLoader) -> None:  # type: ignore[type-arg]
        self.data_loader = data_loader
        self._iterator = iter(data_loader)

    def __next__(self) -> Any:
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.data_loader)
            return next(self._iterator)

    def __iter__(self) -> _InfiniteDataIterator:
        return self


def _adj_to_edge_list(adj: Any) -> Any:
    """Convert a binary adjacency matrix to an (E, 2) edge array.

    Args:
        adj: (N, N) binary adjacency matrix (numpy).

    Returns:
        Edge index array of shape (E, 2), int64.
    """
    import numpy as np

    rows, cols = np.where(adj > 0.5)
    mask = rows < cols
    return np.stack([rows[mask], cols[mask]], axis=1).astype(np.int64)