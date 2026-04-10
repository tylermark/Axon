"""TR-003: Supervised fine-tuning loop for the Layer 1 extraction pipeline.

Trains tokenizer, diffusion, and constraint modules jointly on annotated
floor plan data using the composite loss:

    L_total = L_diffusion + lambda_SAT * L_constraints

Where:
    L_diffusion  = VLB (coordinate MSE + adjacency BCE) from the diffusion engine
    L_constraints = sum of weighted axiom violations from the SAT solver

The SFT loop uses AdamW with cosine annealing LR schedule, gradient
clipping, periodic evaluation (HIoU / GED / Betti), and checkpoint
saving with resume-from-checkpoint support.

Reference:
    ARCHITECTURE.md -- Training Pipeline (SSL -> SFT -> GRPO)
    MODEL_SPEC.md   -- Composite Loss, EQ-06 / EQ-07
"""

from __future__ import annotations

import logging
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.constraints.axioms import edges_from_adjacency


def _host_rss_gb() -> float:
    """Return the current process's resident host RAM in GB (Linux/macOS).

    Uses ``resource.getrusage`` so it works without an extra dep like
    psutil. On Linux ``ru_maxrss`` is returned in kilobytes; on macOS it
    is in bytes. We detect the unit by checking the platform.
    """
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys
    if sys.platform == "darwin":
        return rss_raw / (1024 ** 3)
    return rss_raw / (1024 ** 2)  # Linux: kilobytes → GB

if TYPE_CHECKING:
    from src.diffusion.reverse import DiffusionLoss, GraphDiffusionModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional W&B import (no crash if missing)
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
class SFTConfig:
    """Configuration for the supervised fine-tuning loop."""

    learning_rate: float = 1e-4
    """Peak learning rate for AdamW."""

    batch_size: int = 8
    """Training batch size."""

    num_epochs: int = 50
    """Total training epochs."""

    warmup_steps: int = 500
    """Linear warmup steps before cosine decay."""

    weight_decay: float = 0.01
    """AdamW weight decay coefficient."""

    lambda_sat: float = 0.1
    """Weight for constraint loss in L_total = L_diffusion + lambda_SAT * L_constraints."""

    checkpoint_dir: str = "checkpoints/sft"
    """Directory for saving model checkpoints."""

    device: str = "auto"
    """Compute device: 'auto' detects cuda/mps/cpu."""

    wandb_project: str = "axon-sft"
    """Weights & Biases project name."""

    wandb_enabled: bool = True
    """Enable W&B logging (silently disabled if wandb not installed)."""

    gradient_clip: float = 1.0
    """Max gradient norm for clipping."""

    eval_every_n_epochs: int = 5
    """Run evaluation every N epochs."""

    eval_benchmark_metrics: bool = False
    """Compute HIoU / GED / Betti benchmark metrics during periodic eval.

    Default **False** because ``compute_ged`` uses
    ``networkx.optimize_graph_edit_distance`` on graphs with up to 512
    nodes per sample, which is NP-hard and can allocate GBs of host memory
    per call. Across a full eval pass that exploded past the 176 GB Colab
    host RAM ceiling and OOM-killed the kernel.

    Eval loss (``eval/loss/total``) is always computed regardless — that
    is cheap and sufficient to monitor training progress. Enable this flag
    only for final model-quality evaluation on a small held-out set, not
    for periodic training checks.
    """

    save_every_n_epochs: int = 5
    """Save checkpoint every N epochs."""

    num_workers: int = 0
    """DataLoader worker processes."""

    seed: int = 42
    """Random seed for reproducibility."""

    profile_first_n_steps: int = 5
    """Record per-section wall-clock timings for the first N training steps.

    Probes tokenizer / diffusion / edges_from_adjacency / constraint forward /
    backward / optimizer. Each section is CUDA-synced for accurate GPU timing.
    Set to 0 to disable.
    """


# ---------------------------------------------------------------------------
# Per-step wall-clock timer
# ---------------------------------------------------------------------------


class _StepTimer:
    """Optional CUDA-synced section timer for profiling a single training step.

    When ``enabled`` is False every method is a no-op so there is zero
    overhead on non-profiled steps. When enabled, each ``mark`` call triggers
    a ``torch.cuda.synchronize`` (on CUDA devices) so wall-clock measurements
    reflect actual kernel runtime, not just async kernel launch latency.

    On CUDA, each mark also records:

    - ``alloc_gb``: currently live PyTorch allocations (GB)
    - ``peak_gb``: peak allocation since the previous mark (GB), captured
      via ``reset_peak_memory_stats`` at each boundary

    Timings and memory stats are printed **live** at each mark call — not
    aggregated at step end — so a mid-step CUDA OOM still leaves a trace
    of every section that completed before the crash.
    """

    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.device = device
        self.enabled = enabled
        self.timings: dict[str, float] = {}
        self._t = 0.0
        self._step_idx: int | None = None

    def start(self, step_idx: int | None = None) -> None:
        if not self.enabled:
            return
        self._step_idx = step_idx
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            alloc_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(
                "step %s start: alloc=%.2fGB",
                "?" if step_idx is None else str(step_idx),
                alloc_gb,
            )
        self._t = time.perf_counter()

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        dt = now - self._t
        self.timings[name] = dt
        self._t = now

        short = name.replace("time/", "").replace("_s", "")
        host_gb = _host_rss_gb()
        if self.device.type == "cuda":
            alloc_gb = torch.cuda.memory_allocated() / 1e9
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            logger.info(
                "step %s   %-22s  %7.0f ms   gpu_alloc=%5.2fGB   gpu_peak=%5.2fGB   host_rss=%5.2fGB",
                "?" if self._step_idx is None else str(self._step_idx),
                short,
                dt * 1000,
                alloc_gb,
                peak_gb,
                host_gb,
            )
            torch.cuda.reset_peak_memory_stats()
        else:
            logger.info(
                "step %s   %-22s  %7.0f ms   host_rss=%5.2fGB",
                "?" if self._step_idx is None else str(self._step_idx),
                short,
                dt * 1000,
                host_gb,
            )


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
# Warmup + Cosine scheduler
# ---------------------------------------------------------------------------


class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing LR schedule.

    Wraps a CosineAnnealingLR scheduler with a manual warmup phase.
    During warmup the LR increases linearly from 0 to the base LR.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps (warmup + cosine phase).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - warmup_steps, 1),
        )
        self._step_count = 0

    def step(self) -> None:
        """Advance the scheduler by one step."""
        self._step_count += 1
        if self._step_count <= self.warmup_steps:
            scale = self._step_count / max(self.warmup_steps, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
                pg["lr"] = base_lr * scale
        else:
            self.cosine_scheduler.step()

    def get_last_lr(self) -> list[float]:
        """Return the last computed learning rate for each param group."""
        if self._step_count <= self.warmup_steps:
            scale = self._step_count / max(self.warmup_steps, 1)
            return [lr * scale for lr in self.base_lrs]
        return self.cosine_scheduler.get_last_lr()

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state for checkpointing."""
        return {
            "step_count": self._step_count,
            "cosine_state": self.cosine_scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
        self._step_count = state["step_count"]
        self.cosine_scheduler.load_state_dict(state["cosine_state"])


# ---------------------------------------------------------------------------
# SFTTrainer
# ---------------------------------------------------------------------------


class SFTTrainer:
    """Supervised fine-tuning trainer for the Layer 1 pipeline.

    Trains tokenizer + diffusion + constraint modules jointly using the
    composite loss ``L_total = L_diffusion + lambda_SAT * L_constraints``.

    Supports:
        - Cosine LR schedule with linear warmup
        - Gradient clipping
        - Periodic evaluation with HIoU / GED / Betti metrics
        - Checkpoint save/resume
        - Optional W&B logging

    Args:
        tokenizer_model: Stage 2 tokenizer (produces context embeddings).
        diffusion_model: Stage 3 graph diffusion model (forward + reverse).
        constraint_module: Stage 4 constraint solver.
        dataset: Training dataset yielding batches of floor plan data.
        eval_dataset: Evaluation dataset for periodic metric computation.
        config: SFT configuration.
    """

    def __init__(
        self,
        tokenizer_model: nn.Module,
        diffusion_model: GraphDiffusionModel,
        constraint_module: nn.Module,
        dataset: Dataset,  # type: ignore[type-arg]
        eval_dataset: Dataset | None,  # type: ignore[type-arg]
        config: SFTConfig,
    ) -> None:
        self.config = config
        self.device = _resolve_device(config.device)

        # Models
        self.tokenizer_model = tokenizer_model.to(self.device)
        self.diffusion_model = diffusion_model.to(self.device)
        self.constraint_module = constraint_module.to(self.device)

        # Datasets
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )

        # Optimizer: collect parameters from all three modules.
        all_params = (
            list(self.tokenizer_model.parameters())
            + list(self.diffusion_model.parameters())
            + list(self.constraint_module.parameters())
        )
        self.optimizer = AdamW(
            all_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR scheduler
        steps_per_epoch = len(self.train_loader)
        total_steps = config.num_epochs * steps_per_epoch
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.train_metrics_history: list[dict[str, float]] = []
        self.eval_metrics_history: list[dict[str, float]] = []

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
                    "num_epochs": config.num_epochs,
                    "warmup_steps": config.warmup_steps,
                    "weight_decay": config.weight_decay,
                    "lambda_sat": config.lambda_sat,
                    "gradient_clip": config.gradient_clip,
                },
                resume="allow",
            )

    # ------------------------------------------------------------------
    # Composite loss computation
    # ------------------------------------------------------------------

    def _compute_composite_loss(
        self,
        x_0: torch.Tensor,
        a_0: torch.Tensor,
        context: torch.Tensor | None,
        node_mask: torch.Tensor | None,
        context_mask: torch.Tensor | None = None,
        timer: _StepTimer | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute L_total = L_diffusion + lambda_SAT * L_constraints.

        Args:
            x_0: (B, N, 2) clean node coordinates.
            a_0: (B, N, N) clean adjacency matrix.
            context: (B, N_ctx, d_ctx) cross-modal context from tokenizer.
            node_mask: (B, N) valid node mask.
            context_mask: (B, N_ctx) context token mask.
            timer: Optional ``_StepTimer`` to record per-section wall-clock
                timings. Intended for debugging the first few steps only.

        Returns:
            Tuple of (total_loss, metrics_dict) where metrics_dict contains
            individual loss components for logging.
        """
        # --- Diffusion VLB loss ---
        diff_loss: DiffusionLoss = self.diffusion_model(x_0, a_0, context, node_mask, context_mask)
        if timer is not None:
            timer.mark("time/diffusion_fwd_s")

        # --- Constraint loss ---
        # Run a single denoising step to get the predicted graph for
        # constraint evaluation, or evaluate constraints on the clean
        # graph to provide gradient signal.
        edge_index = edges_from_adjacency(a_0, node_mask)
        if timer is not None:
            timer.mark("time/edges_from_adj_s")

        constraint_output = self.constraint_module(
            node_positions=x_0,
            adjacency=a_0,
            edge_index=edge_index,
            node_mask=node_mask,
        )
        constraint_loss = constraint_output.total_loss
        if timer is not None:
            timer.mark("time/constraint_fwd_s")

        # --- Composite ---
        total_loss = diff_loss.total + self.config.lambda_sat * constraint_loss

        metrics = {
            "loss/total": total_loss.item(),
            "loss/diffusion": diff_loss.total.item(),
            "loss/diffusion_coord": diff_loss.coordinate_loss.item(),
            "loss/diffusion_adj": diff_loss.adjacency_loss.item(),
            "loss/constraint": constraint_loss.item(),
            "loss/constraint_weighted": (self.config.lambda_sat * constraint_loss).item(),
        }

        return total_loss, metrics

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch over the full dataset.

        Returns:
            Dict of average metrics across all batches in the epoch.
        """
        self.tokenizer_model.train()
        self.diffusion_model.train()
        self.constraint_module.train()

        epoch_metrics: dict[str, list[float]] = {}
        epoch_start = time.monotonic()

        for batch in self.train_loader:
            step_metrics = self._train_step(batch)

            for key, value in step_metrics.items():
                epoch_metrics.setdefault(key, []).append(value)

            self.global_step += 1

        # Average metrics over epoch
        avg_metrics: dict[str, float] = {
            key: sum(values) / len(values)
            for key, values in epoch_metrics.items()
            if values
        }
        avg_metrics["epoch"] = float(self.current_epoch)
        avg_metrics["epoch_time_s"] = time.monotonic() - epoch_start
        avg_metrics["lr"] = self.scheduler.get_last_lr()[0]

        self.train_metrics_history.append(avg_metrics)
        return avg_metrics

    def _train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute a single training step on one batch.

        Args:
            batch: Dict containing at minimum 'x_0', 'a_0', and optionally
                'node_mask', 'raw_features' (for tokenizer), 'context_mask'.

        Returns:
            Dict of per-step metric values.
        """
        # Profile only the first N steps so we can locate per-section
        # bottlenecks without permanently syncing on every training step.
        timer = _StepTimer(
            self.device,
            enabled=self.global_step < self.config.profile_first_n_steps,
        )

        # Move batch to device
        x_0 = batch["x_0"].to(self.device)
        a_0 = batch["a_0"].to(self.device)
        node_mask = batch.get("node_mask")
        if node_mask is not None:
            node_mask = node_mask.to(self.device)

        timer.start(step_idx=self.global_step)

        # Tokenizer forward pass for context embeddings
        context = None
        context_mask = None
        if "raw_features" in batch:
            raw_features = {k: v.to(self.device) for k, v in batch["raw_features"].items()}
            tokenizer_output = self.tokenizer_model(raw_features)
            if hasattr(tokenizer_output, "token_embeddings"):
                context = tokenizer_output.token_embeddings
                if hasattr(tokenizer_output, "attention_mask"):
                    context_mask = tokenizer_output.attention_mask
            else:
                # Direct tensor output
                context = tokenizer_output
        timer.mark("time/tokenizer_fwd_s")

        # Compute composite loss (further subdivided inside via the timer).
        self.optimizer.zero_grad()
        total_loss, metrics = self._compute_composite_loss(
            x_0, a_0, context, node_mask, context_mask, timer=timer,
        )

        # Backward pass
        total_loss.backward()
        timer.mark("time/backward_s")

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.tokenizer_model.parameters())
            + list(self.diffusion_model.parameters())
            + list(self.constraint_module.parameters()),
            max_norm=self.config.gradient_clip,
        )
        metrics["grad_norm"] = (
            grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        )

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        timer.mark("time/optimizer_s")

        metrics["lr"] = self.scheduler.get_last_lr()[0]

        # The per-section timing + memory lines are printed live inside
        # ``_StepTimer.mark`` so a mid-step OOM still leaves a trace; only
        # the aggregated metrics are forwarded to W&B here.
        if timer.enabled and timer.timings:
            metrics.update(timer.timings)

        # W&B logging
        if self._wandb_run is not None:
            wandb.log(metrics, step=self.global_step)

        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the eval dataset and compute metrics.

        Computes:
            - Average diffusion and constraint losses
            - HIoU, GED, and Betti error (when ground truth is available)

        Returns:
            Dict of evaluation metrics.
        """
        if self.eval_dataset is None:
            logger.warning("No eval_dataset provided, skipping evaluation.")
            return {}

        self.tokenizer_model.eval()
        self.diffusion_model.eval()
        self.constraint_module.eval()

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        all_metrics: dict[str, list[float]] = {}

        with torch.no_grad():
            for batch in eval_loader:
                x_0 = batch["x_0"].to(self.device)
                a_0 = batch["a_0"].to(self.device)
                node_mask = batch.get("node_mask")
                if node_mask is not None:
                    node_mask = node_mask.to(self.device)

                # Tokenizer forward
                context = None
                context_mask = None
                if "raw_features" in batch:
                    raw_features = {k: v.to(self.device) for k, v in batch["raw_features"].items()}
                    tokenizer_output = self.tokenizer_model(raw_features)
                    if hasattr(tokenizer_output, "token_embeddings"):
                        context = tokenizer_output.token_embeddings
                        if hasattr(tokenizer_output, "attention_mask"):
                            context_mask = tokenizer_output.attention_mask
                    else:
                        context = tokenizer_output

                # Loss computation
                _, metrics = self._compute_composite_loss(
                    x_0, a_0, context, node_mask, context_mask
                )
                for key, value in metrics.items():
                    all_metrics.setdefault(f"eval/{key}", []).append(value)

                # Benchmark metrics on sampled outputs — gated behind
                # ``eval_benchmark_metrics`` because ``compute_ged`` uses an
                # NP-hard networkx GED that blows host memory past the Colab
                # ceiling on graphs with hundreds of nodes.
                if self.config.eval_benchmark_metrics:
                    self._compute_benchmark_metrics(
                        x_0, a_0, node_mask, context, context_mask, batch, all_metrics
                    )

        # Average all eval metrics
        avg_metrics: dict[str, float] = {
            key: sum(values) / len(values)
            for key, values in all_metrics.items()
            if values
        }
        avg_metrics["eval/epoch"] = float(self.current_epoch)

        self.eval_metrics_history.append(avg_metrics)

        # W&B logging
        if self._wandb_run is not None:
            wandb.log(avg_metrics, step=self.global_step)

        return avg_metrics

    def _compute_benchmark_metrics(
        self,
        x_0: torch.Tensor,
        a_0: torch.Tensor,
        node_mask: torch.Tensor | None,
        context: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        batch: dict[str, torch.Tensor],
        all_metrics: dict[str, list[float]],
    ) -> None:
        """Compute HIoU, GED, and Betti benchmark metrics on sampled outputs.

        Generates graphs via DDIM sampling and compares against ground truth.
        Errors are caught gracefully so evaluation continues even if a
        metric computation fails on a particular batch.

        Args:
            x_0: (B, N, 2) ground truth coordinates.
            a_0: (B, N, N) ground truth adjacency.
            node_mask: (B, N) valid node mask.
            context: (B, N_ctx, d_ctx) tokenizer context.
            context_mask: (B, N_ctx) context mask.
            batch: Original batch dict (may contain additional GT fields).
            all_metrics: Accumulator dict to append metric values into.
        """
        try:
            # Sample a graph from the diffusion model
            bsz, n_nodes, _ = x_0.shape
            sampled = self.diffusion_model.sample(
                num_nodes=n_nodes,
                batch_size=bsz,
                context=context,
                node_mask=node_mask,
                context_mask=context_mask,
                device=self.device,
            )

            pred_positions = sampled.node_positions.cpu().numpy()
            pred_adj = (torch.sigmoid(sampled.adjacency_logits) > 0.5).cpu().numpy()
            gt_positions = x_0.cpu().numpy()
            gt_adj = a_0.cpu().numpy()

            for b in range(bsz):
                mask = node_mask[b].cpu().numpy() if node_mask is not None else None
                n_valid = int(mask.sum()) if mask is not None else n_nodes

                pred_nodes_b = pred_positions[b, :n_valid]
                gt_nodes_b = gt_positions[b, :n_valid]

                # Extract edges from adjacency
                pred_edges_b = _adj_to_edge_list(pred_adj[b, :n_valid, :n_valid])
                gt_edges_b = _adj_to_edge_list(gt_adj[b, :n_valid, :n_valid])

                # GED
                try:
                    from tests.benchmarks.test_ged import compute_ged

                    ged = compute_ged(pred_nodes_b, pred_edges_b, gt_nodes_b, gt_edges_b)
                    all_metrics.setdefault("eval/ged", []).append(ged)
                except Exception:
                    pass

                # Betti error
                try:
                    from tests.benchmarks.test_betti import (
                        compute_betti_error,
                        compute_betti_numbers,
                    )

                    gt_b0, gt_b1 = compute_betti_numbers(gt_nodes_b, gt_edges_b)
                    err_b0, err_b1 = compute_betti_error(pred_nodes_b, pred_edges_b, gt_b0, gt_b1)
                    all_metrics.setdefault("eval/betti_0_error", []).append(float(err_b0))
                    all_metrics.setdefault("eval/betti_1_error", []).append(float(err_b1))
                except Exception:
                    pass

                # HIoU (requires wall thickness — use default if not in batch)
                try:
                    from tests.benchmarks.test_hiou import compute_hiou

                    thickness = batch.get("wall_thickness", torch.full((bsz,), 2.0))
                    t_val = float(thickness[b]) if thickness.dim() > 0 else float(thickness)

                    pred_walls = _nodes_edges_to_walls(pred_nodes_b, pred_edges_b, t_val)
                    gt_walls = _nodes_edges_to_walls(gt_nodes_b, gt_edges_b, t_val)

                    if pred_walls and gt_walls:
                        hiou = compute_hiou(pred_walls, gt_walls)
                        all_metrics.setdefault("eval/hiou", []).append(hiou)
                except Exception:
                    pass

        except Exception as e:
            logger.debug("Benchmark metric computation failed: %s", e)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full SFT training loop.

        Iterates over epochs, running training, periodic evaluation,
        and checkpoint saving. Supports resume from a saved checkpoint.
        """
        logger.info(
            "Starting SFT training: %d epochs, lr=%.1e, lambda_SAT=%.3f, device=%s",
            self.config.num_epochs,
            self.config.learning_rate,
            self.config.lambda_sat,
            self.device,
        )

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        logger.info("host_rss before training: %.2f GB", _host_rss_gb())

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.monotonic()

            # Train
            train_metrics = self.train_epoch()
            logger.info(
                "Epoch %d/%d  loss=%.4f  diff=%.4f  constr=%.4f  lr=%.2e  (%.1fs)  host_rss=%.2fGB",
                epoch + 1,
                self.config.num_epochs,
                train_metrics.get("loss/total", 0.0),
                train_metrics.get("loss/diffusion", 0.0),
                train_metrics.get("loss/constraint", 0.0),
                train_metrics.get("lr", 0.0),
                time.monotonic() - epoch_start,
                _host_rss_gb(),
            )

            # Evaluate
            if (epoch + 1) % self.config.eval_every_n_epochs == 0:
                logger.info(
                    "Eval start (epoch %d)  host_rss=%.2fGB",
                    epoch + 1,
                    _host_rss_gb(),
                )
                eval_start = time.monotonic()
                eval_metrics = self.evaluate()
                logger.info(
                    "Eval done  (%.1fs)  host_rss=%.2fGB",
                    time.monotonic() - eval_start,
                    _host_rss_gb(),
                )
                eval_loss = eval_metrics.get("eval/loss/total", float("inf"))
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint("best.pt")
                    logger.info("  New best eval loss: %.4f", eval_loss)

                if eval_metrics:
                    _log_eval_summary(eval_metrics)

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch + 1:04d}.pt")

        # Final checkpoint
        self.save_checkpoint("final.pt")
        logger.info("SFT training complete. Best eval loss: %.4f", self.best_eval_loss)

        if self._wandb_run is not None:
            wandb.finish()

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str) -> Path:
        """Save training state to a checkpoint file.

        Saves model weights, optimizer state, scheduler state, epoch,
        global step, and best eval loss.

        Args:
            filename: Checkpoint filename (saved under checkpoint_dir).

        Returns:
            Full path to the saved checkpoint.
        """
        path = self.checkpoint_dir / filename
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "tokenizer_state_dict": self.tokenizer_model.state_dict(),
            "diffusion_state_dict": self.diffusion_model.state_dict(),
            "constraint_state_dict": self.constraint_module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s", path)
        return path

    @staticmethod
    def find_checkpoint(checkpoint_dir: str | Path) -> Path | None:
        """Return the path to the best available SFT checkpoint.

        Checks in priority order: ``best.pt`` → ``final.pt`` → latest
        ``epoch_NNNN.pt``.  Returns ``None`` when the directory does not
        exist or contains no recognised checkpoint files.

        This is the intended API for downstream stages (e.g. GRPO) that need
        to locate the SFT output without coupling to internal filename
        conventions.

        Args:
            checkpoint_dir: Directory passed to SFTConfig.checkpoint_dir.

        Returns:
            Path to a checkpoint file, or None if none found.
        """
        d = Path(checkpoint_dir)
        if not d.exists():
            return None
        for name in ("best.pt", "final.pt"):
            p = d / name
            if p.exists():
                return p
        epoch_files = sorted(d.glob("epoch_*.pt"))
        return epoch_files[-1] if epoch_files else None

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume training from a checkpoint.

        Restores model weights, optimizer state, scheduler state, and
        training progress counters.

        Args:
            path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.tokenizer_model.load_state_dict(checkpoint["tokenizer_state_dict"])
        self.diffusion_model.load_state_dict(checkpoint["diffusion_state_dict"])
        self.constraint_module.load_state_dict(checkpoint["constraint_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"] + 1  # Resume from next epoch
        self.global_step = checkpoint["global_step"]
        self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))

        logger.info(
            "Resumed from checkpoint: %s (epoch %d, step %d)",
            path,
            self.current_epoch,
            self.global_step,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _adj_to_edge_list(adj: Any) -> Any:
    """Convert a binary adjacency matrix to an (E, 2) edge array.

    Args:
        adj: (N, N) binary adjacency matrix (numpy).

    Returns:
        Edge index array of shape (E, 2), int64.
    """
    import numpy as np

    rows, cols = np.where(adj > 0.5)
    # Upper-triangle only to avoid duplicate undirected edges.
    mask = rows < cols
    return np.stack([rows[mask], cols[mask]], axis=1).astype(np.int64)


def _nodes_edges_to_walls(
    nodes: Any,
    edges: Any,
    thickness: float,
) -> list[tuple[Any, Any, float]]:
    """Convert node positions and edge list to wall tuples for HIoU.

    Args:
        nodes: (N, 2) node positions.
        edges: (E, 2) edge indices.
        thickness: Default wall thickness.

    Returns:
        List of (start_coord, end_coord, thickness) tuples.
    """
    walls = []
    for src, dst in edges:
        walls.append((nodes[int(src)], nodes[int(dst)], thickness))
    return walls


def _log_eval_summary(metrics: dict[str, float]) -> None:
    """Log a concise summary of evaluation metrics."""
    parts = []
    for key in [
        "eval/loss/total",
        "eval/hiou",
        "eval/ged",
        "eval/betti_0_error",
        "eval/betti_1_error",
    ]:
        if key in metrics:
            short = key.split("/")[-1]
            parts.append(f"{short}={metrics[key]:.4f}")
    if parts:
        logger.info("  Eval: %s", "  ".join(parts))
