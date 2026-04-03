"""TR-001: Masked Primitive Modeling (MPM) pre-training loop.

Self-supervised pre-training objective: mask 75--85% of vector tokens
and train the model to reconstruct them.  The reconstruction target
is the continuous coordinate values of masked tokens, measured by
Chamfer Distance, with an optional cross-entropy loss for entity type
prediction.

The MPM loop wraps the tokenizer's ``VectorTokenEmbedding`` for input
projection and a lightweight transformer encoder for denoising.  It
can optionally initialise from the diffusion backbone weights to
share representations learned during pre-training.

Reference:
    ARCHITECTURE.md -- Training Pipeline (SSL Pre-Training)
    MODEL_SPEC.md   -- Masked Primitive Modeling
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as nnf
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MPMConfig:
    """Configuration for Masked Primitive Modeling pre-training.

    Attributes:
        mask_ratio_low: Lower bound of the random mask ratio per batch.
        mask_ratio_high: Upper bound of the random mask ratio per batch.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        learning_rate: Peak learning rate for AdamW.
        weight_decay: AdamW weight decay.
        batch_size: Training batch size per device.
        num_epochs: Total pre-training epochs.
        grad_clip_norm: Maximum gradient norm for clipping.
        warmup_steps: Linear warmup steps for the LR scheduler.
        entity_type_loss_weight: Weight for optional entity-type CE loss.
            Set to 0.0 to disable type prediction head.
        num_entity_types: Number of distinct entity type categories.
        checkpoint_dir: Directory to save checkpoints.
        checkpoint_every: Save a checkpoint every N epochs.
        resume_from: Path to a checkpoint to resume from, or None.
        device: Compute device string (``"cuda"``, ``"cpu"``, ``"mps"``).
            If ``"auto"``, selects the best available device.
        num_workers: DataLoader worker processes.
        seed: Random seed for reproducibility.
        wandb_project: W&B project name, or empty string to disable.
        wandb_run_name: W&B run name, or empty string for auto-generated.
    """

    mask_ratio_low: float = 0.75
    mask_ratio_high: float = 0.85
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 16
    num_epochs: int = 200
    grad_clip_norm: float = 1.0
    warmup_steps: int = 1000
    entity_type_loss_weight: float = 0.1
    num_entity_types: int = 101
    checkpoint_dir: str = "checkpoints/mpm"
    checkpoint_every: int = 10
    resume_from: str | None = None
    device: str = "auto"
    num_workers: int = 4
    seed: int = 42
    wandb_project: str = "axon"
    wandb_run_name: str = ""


# ---------------------------------------------------------------------------
# Device auto-detection
# ---------------------------------------------------------------------------


def _resolve_device(device_str: str) -> torch.device:
    """Resolve a device string to a ``torch.device``.

    ``"auto"`` selects CUDA > MPS > CPU in order of availability.

    Args:
        device_str: One of ``"auto"``, ``"cuda"``, ``"cpu"``, ``"mps"``,
            or a specific ``"cuda:N"``.

    Returns:
        Resolved ``torch.device``.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Chamfer distance (batch, variable-length)
# ---------------------------------------------------------------------------


def chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric Chamfer Distance between predicted and target point sets.

    Operates on the 4D coordinate representation (x1, y1, x2, y2) so each
    token is treated as a point in R^4.

    Args:
        pred: (B, N, D) predicted coordinates.
        target: (B, N, D) ground-truth coordinates.
        mask: (B, N) bool mask -- True for positions that contribute to loss.

    Returns:
        Scalar mean Chamfer Distance over the batch.
    """
    # Pairwise L2 distance: (B, N, N)
    diff = pred.unsqueeze(2) - target.unsqueeze(1)  # (B, N, N, D)
    dists = (diff**2).sum(dim=-1)  # (B, N, N)

    if mask is not None:
        # Invalidate distances to/from padded positions.
        inv_mask = ~mask  # (B, N)
        large = 1e9
        dists = dists + inv_mask.unsqueeze(1).float() * large  # mask target dim
        dists = dists + inv_mask.unsqueeze(2).float() * large  # mask pred dim

    # pred -> nearest target
    forward_min = dists.min(dim=2).values  # (B, N)
    # target -> nearest pred
    backward_min = dists.min(dim=1).values  # (B, N)

    if mask is not None:
        forward_min = forward_min * mask.float()
        backward_min = backward_min * mask.float()
        n_valid = mask.float().sum(dim=1).clamp(min=1.0)  # (B,)
        forward_loss = (forward_min.sum(dim=1) / n_valid).mean()
        backward_loss = (backward_min.sum(dim=1) / n_valid).mean()
    else:
        forward_loss = forward_min.mean()
        backward_loss = backward_min.mean()

    return (forward_loss + backward_loss) / 2.0


# ---------------------------------------------------------------------------
# MPM Model
# ---------------------------------------------------------------------------


class MPMModel(nn.Module):
    """Lightweight model for Masked Primitive Modeling.

    Takes vector token features, applies masking, runs through a transformer
    encoder, and predicts the masked token coordinates and (optionally)
    entity types.

    Architecture:
        Input projection (7 -> d_model) + learned 2D positional encoding
        -> Transformer encoder (n_layers x self-attention blocks)
        -> Coordinate head (d_model -> 4)
        -> Optional type head (d_model -> num_entity_types)

    The input feature dimension of 7 matches the data engine's entity
    representation: [x1, y1, x2, y2, entity_type, feat5, feat6].
    """

    def __init__(self, config: MPMConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        # Input projection: raw 7-dim features -> d_model.
        self.input_proj = nn.Linear(7, d)

        # Learned 2D positional encoding (matches tokenizer design).
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # Learnable [MASK] token embedding.
        self.mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=4 * d,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(d)

        # Output heads.
        self.coord_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 4),
        )

        self.type_head: nn.Module | None = None
        if config.entity_type_loss_weight > 0:
            self.type_head = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, config.num_entity_types),
            )

    def forward(
        self,
        entities: torch.Tensor,
        mask_indices: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through the MPM model.

        Args:
            entities: (B, N, 7) raw entity features.
            mask_indices: (B, N) bool -- True for positions to mask.
            attention_mask: (B, N) bool -- True for valid (non-padding) tokens.

        Returns:
            Tuple of:
                coord_pred: (B, N, 4) predicted coordinates for ALL positions
                    (loss should only be computed on masked positions).
                type_logits: (B, N, num_entity_types) or None if type head
                    is disabled.
        """
        bsz, seq_len, _ = entities.shape

        # Project input features.
        h = self.input_proj(entities)  # (B, N, d)

        # Positional encoding from coordinate midpoints.
        coords = entities[..., :4]  # (B, N, 4) -- x1, y1, x2, y2
        midpoints = torch.stack(
            [(coords[..., 0] + coords[..., 2]) / 2.0, (coords[..., 1] + coords[..., 3]) / 2.0],
            dim=-1,
        )  # (B, N, 2)
        pos_enc = self.pos_mlp(midpoints)  # (B, N, d)
        h = h + pos_enc

        # Apply masking: replace masked positions with the learned mask token.
        mask_expanded = mask_indices.unsqueeze(-1).float()  # (B, N, 1)
        mask_token_expanded = self.mask_token.expand(bsz, seq_len, -1)
        h = h * (1.0 - mask_expanded) + mask_token_expanded * mask_expanded

        # Build causal-free attention mask for the transformer.
        # PyTorch TransformerEncoder expects src_key_padding_mask where True = ignore.
        padding_mask = None
        if attention_mask is not None:
            padding_mask = ~attention_mask  # True = padded = ignore

        # Transformer encoding.
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        h = self.final_norm(h)

        # Predictions.
        coord_pred = self.coord_head(h)  # (B, N, 4)

        type_logits = None
        if self.type_head is not None:
            type_logits = self.type_head(h)  # (B, N, num_entity_types)

        return coord_pred, type_logits


# ---------------------------------------------------------------------------
# MPM Pre-Trainer
# ---------------------------------------------------------------------------


class MPMPreTrainer:
    """Masked Primitive Modeling pre-training loop.

    Self-supervised: masks 75--85% of vector tokens per sample and trains
    the model to reconstruct masked token coordinates (Chamfer Distance)
    with an optional entity-type cross-entropy loss.

    Supports:
        - Resume from checkpoint
        - Gradient clipping
        - Linear warmup + cosine decay LR schedule
        - Optional Weights & Biases logging
        - CPU / CUDA / MPS device auto-detection

    Args:
        model: An ``MPMModel`` instance (or any module with compatible forward).
        dataset: A PyTorch ``Dataset`` producing dicts with ``entities``,
            ``attention_mask``, and ``entity_types`` keys.
        config: MPM configuration dataclass.
    """

    def __init__(
        self,
        model: MPMModel,
        dataset: Dataset,
        config: MPMConfig,
    ) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self.model = model.to(self.device)
        self.dataset = dataset

        # Optimizer.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # DataLoader.
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            persistent_workers=config.num_workers > 0,
        )

        # State.
        self.current_epoch = 0
        self.global_step = 0
        self.loss_history: list[dict[str, float]] = []

        # LR scheduler: linear warmup + cosine decay.
        total_steps = config.num_epochs * max(len(self.dataloader), 1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda(config.warmup_steps, total_steps),
        )

        # Checkpoint directory.
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Resume from checkpoint if specified.
        if config.resume_from is not None:
            self._load_checkpoint(Path(config.resume_from))

        # W&B (conditional import).
        self._wandb_run: Any = None
        if config.wandb_project:
            self._init_wandb()

    # -- LR schedule --------------------------------------------------------

    @staticmethod
    def _lr_lambda(warmup_steps: int, total_steps: int):
        """Return a lambda for linear warmup + cosine decay."""

        def _fn(step: int) -> float:
            if step < warmup_steps:
                return max(step / max(warmup_steps, 1), 1e-7)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.5 * (1.0 + math.cos(math.pi * progress)), 1e-7)

        return _fn

    # -- Masking ------------------------------------------------------------

    def _sample_mask(
        self,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate random mask indices for a batch.

        Mask ratio is sampled uniformly from [mask_ratio_low, mask_ratio_high]
        per batch. Only valid (non-padding) positions are masked.

        Args:
            attention_mask: (B, N) bool -- True for valid tokens.

        Returns:
            (B, N) bool tensor -- True for positions to mask.
        """
        bsz, seq_len = attention_mask.shape
        ratio = (
            torch.empty(1).uniform_(self.config.mask_ratio_low, self.config.mask_ratio_high).item()
        )

        # Per-sample random scores; padded positions get score > 1 so they are
        # never selected for masking.
        scores = torch.rand(bsz, seq_len, device=attention_mask.device)
        scores[~attention_mask] = 2.0

        # For each sample, mask the top `ratio` fraction of valid tokens.
        n_valid = attention_mask.float().sum(dim=1, keepdim=True)  # (B, 1)
        n_mask = (n_valid * ratio).long().clamp(min=1)  # (B, 1)

        # Sort scores and create threshold mask.
        sorted_scores, _ = scores.sort(dim=1)
        # Gather the threshold value at position n_mask for each sample.
        thresholds = sorted_scores.gather(1, n_mask.clamp(max=seq_len - 1))  # (B, 1)
        mask = scores <= thresholds  # (B, N)

        # Ensure padding stays unmasked.
        mask = mask & attention_mask

        return mask

    # -- Training -----------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        """Run a single training epoch.

        Returns:
            Dict with average ``total_loss``, ``coord_loss``, ``type_loss``,
            and ``learning_rate`` for the epoch.
        """
        self.model.train()
        epoch_coord_loss = 0.0
        epoch_type_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        for batch in self.dataloader:
            entities = batch["entities"].to(self.device)  # (B, N, 7)
            attn_mask = batch["attention_mask"].to(self.device)  # (B, N)
            entity_types = batch.get("entity_types")
            if entity_types is not None:
                entity_types = entity_types.to(self.device)  # (B, N)

            # Generate mask.
            mask_indices = self._sample_mask(attn_mask)

            # Forward.
            coord_pred, type_logits = self.model(entities, mask_indices, attn_mask)

            # --- Coordinate reconstruction loss (Chamfer) ---
            # Target: the first 4 columns of entities (normalized coords).
            target_coords = entities[..., :4]  # (B, N, 4)

            # Only compute loss on masked positions.
            coord_loss = chamfer_distance(
                coord_pred,
                target_coords,
                mask=mask_indices,
            )

            # --- Entity type loss (optional cross-entropy) ---
            type_loss = torch.tensor(0.0, device=self.device)
            if (
                type_logits is not None
                and entity_types is not None
                and self.config.entity_type_loss_weight > 0
            ):
                # Flatten masked positions for CE loss.
                masked_logits = type_logits[mask_indices]  # (M, num_types)
                masked_targets = entity_types[mask_indices]  # (M,)
                if masked_logits.numel() > 0:
                    type_loss = nnf.cross_entropy(masked_logits, masked_targets)

            total_loss = coord_loss + self.config.entity_type_loss_weight * type_loss

            # Backward + step.
            self.optimizer.zero_grad()
            total_loss.backward()
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            self.scheduler.step()

            self.global_step += 1
            epoch_coord_loss += coord_loss.item()
            epoch_type_loss += type_loss.item()
            epoch_total_loss += total_loss.item()
            n_batches += 1

            # W&B step logging.
            if self._wandb_run is not None and self.global_step % 50 == 0:
                self._wandb_run.log(
                    {
                        "train/coord_loss": coord_loss.item(),
                        "train/type_loss": type_loss.item(),
                        "train/total_loss": total_loss.item(),
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/mask_ratio": mask_indices.float().sum().item()
                        / attn_mask.float().sum().item(),
                    },
                    step=self.global_step,
                )

        # Epoch averages.
        n = max(n_batches, 1)
        metrics = {
            "total_loss": epoch_total_loss / n,
            "coord_loss": epoch_coord_loss / n,
            "type_loss": epoch_type_loss / n,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        return metrics

    def train(self, num_epochs: int | None = None) -> None:
        """Run the full pre-training loop.

        Args:
            num_epochs: Override the config's num_epochs if specified.
        """
        total_epochs = num_epochs if num_epochs is not None else self.config.num_epochs
        start_epoch = self.current_epoch

        logger.info(
            "Starting MPM pre-training: epochs %d -> %d, device=%s, dataset_size=%d",
            start_epoch,
            start_epoch + total_epochs,
            self.device,
            len(self.dataset),
        )

        # Set seed for reproducibility.
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)

        for epoch in range(start_epoch, start_epoch + total_epochs):
            self.current_epoch = epoch
            t0 = time.time()
            metrics = self.train_epoch()
            elapsed = time.time() - t0

            self.loss_history.append({"epoch": epoch, **metrics})

            logger.info(
                "Epoch %d/%d  total=%.4f  coord=%.4f  type=%.4f  lr=%.2e  (%.1fs)",
                epoch + 1,
                start_epoch + total_epochs,
                metrics["total_loss"],
                metrics["coord_loss"],
                metrics["type_loss"],
                metrics["learning_rate"],
                elapsed,
            )

            # W&B epoch logging.
            if self._wandb_run is not None:
                self._wandb_run.log(
                    {f"epoch/{k}": v for k, v in metrics.items()},
                    step=self.global_step,
                )

            # Checkpoint.
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self._save_checkpoint(epoch)

        # Final checkpoint.
        self._save_checkpoint(self.current_epoch, suffix="_final")
        logger.info(
            "MPM pre-training complete. Final loss: %.4f", self.loss_history[-1]["total_loss"]
        )

        if self._wandb_run is not None:
            self._wandb_run.finish()

    # -- Checkpointing ------------------------------------------------------

    def _save_checkpoint(self, epoch: int, suffix: str = "") -> None:
        """Save model + optimizer + scheduler + state to a checkpoint file.

        Args:
            epoch: Current epoch number.
            suffix: Optional filename suffix (e.g., ``"_final"``).
        """
        path = self.checkpoint_dir / f"mpm_epoch{epoch + 1:04d}{suffix}.pt"
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss_history": self.loss_history,
            "config": {
                "mask_ratio_low": self.config.mask_ratio_low,
                "mask_ratio_high": self.config.mask_ratio_high,
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "n_layers": self.config.n_layers,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "batch_size": self.config.batch_size,
                "num_epochs": self.config.num_epochs,
                "seed": self.config.seed,
            },
        }
        torch.save(state, path)
        logger.info("Saved checkpoint: %s", path)

        # Also save loss history as JSON for easy inspection.
        history_path = self.checkpoint_dir / "loss_history.json"
        with open(history_path, "w") as f:
            json.dump(self.loss_history, f, indent=2)

    def _load_checkpoint(self, path: Path) -> None:
        """Resume training from a checkpoint file.

        Args:
            path: Path to the ``.pt`` checkpoint file.
        """
        if not path.exists():
            logger.warning("Checkpoint not found: %s — starting from scratch.", path)
            return

        logger.info("Resuming from checkpoint: %s", path)
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.current_epoch = state.get("epoch", 0) + 1  # resume from next epoch
        self.global_step = state.get("global_step", 0)
        self.loss_history = state.get("loss_history", [])

        logger.info(
            "Resumed at epoch %d, global_step %d, last loss %.4f",
            self.current_epoch,
            self.global_step,
            self.loss_history[-1]["total_loss"] if self.loss_history else float("nan"),
        )

    # -- W&B ---------------------------------------------------------------

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging (conditional import)."""
        try:
            import wandb

            run_name = self.config.wandb_run_name or None
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config={
                    "mask_ratio_low": self.config.mask_ratio_low,
                    "mask_ratio_high": self.config.mask_ratio_high,
                    "d_model": self.config.d_model,
                    "n_heads": self.config.n_heads,
                    "n_layers": self.config.n_layers,
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "batch_size": self.config.batch_size,
                    "num_epochs": self.config.num_epochs,
                    "seed": self.config.seed,
                },
                reinit=True,
            )
            logger.info("W&B initialized: project=%s", self.config.wandb_project)
        except ImportError:
            logger.info("wandb not installed — logging disabled.")
            self._wandb_run = None
        except Exception:
            logger.warning("Failed to initialize W&B", exc_info=True)
            self._wandb_run = None
