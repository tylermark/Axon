"""Self-contained Colab training callback for the autonomous monitor.

This module is designed to be copy-pasted into any Colab notebook.
It has ZERO imports from the Axon codebase — only stdlib and packages
available by default in Google Colab.

Usage::

    callback = ColabTrainingCallback(
        control_dir="/content/drive/MyDrive/axon/monitor/",
    )

    for epoch in range(num_epochs):
        train_one_epoch(...)
        callback.on_epoch_end(epoch, optimizer=optimizer, model=model)
        if callback.should_stop:
            print("Early stop triggered by monitor.")
            break
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ColabTrainingCallback:
    """Reads monitor decisions from a Google Drive control file each epoch.

    The autonomous training monitor writes a JSON control file to Google
    Drive with a decision (CONTINUE, ADJUST_LR, EARLY_STOP, SNAPSHOT,
    ALERT). This callback reads that file and executes the decision.

    Args:
        control_dir: Path to the directory on mounted Google Drive
            where control files are stored.
        run_id: W&B run ID. If ``None``, attempts to auto-detect from
            the active ``wandb`` run.
        stale_threshold_seconds: Ignore decisions older than this many
            seconds (default: 600 = 10 minutes).
        checkpoint_dir: Where to save checkpoints on EARLY_STOP / SNAPSHOT.
            Defaults to ``{control_dir}/../checkpoints``.
    """

    def __init__(
        self,
        control_dir: str = "/content/drive/MyDrive/axon/monitor/",
        run_id: str | None = None,
        stale_threshold_seconds: int = 600,
        checkpoint_dir: str | None = None,
    ) -> None:
        self._control_dir = Path(control_dir)
        self._stale_threshold = stale_threshold_seconds
        self._checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir
            else self._control_dir.parent / "checkpoints"
        )

        # Auto-detect run ID from wandb if not provided
        if run_id is None:
            run_id = self._detect_run_id()
        self._run_id = run_id

        self.should_stop = False
        self._last_acknowledged_ts: str | None = None

        # Ensure checkpoint dir exists
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ColabTrainingCallback initialized: run_id=%s, control_dir=%s",
            self._run_id,
            self._control_dir,
        )

    def reset(self) -> None:
        """Reset state between training phases."""
        self.should_stop = False
        self._run_id = None

    def on_epoch_end(
        self,
        epoch: int,
        optimizer: Any | None = None,
        model: Any | None = None,
        save_fn: Any | None = None,
        save_dir: str | None = None,
    ) -> None:
        """Check for and execute monitor decisions.

        Call this at the end of each training epoch.

        Args:
            epoch: Current epoch number.
            optimizer: PyTorch optimizer (needed for ADJUST_LR).
            model: PyTorch model (needed for SNAPSHOT / EARLY_STOP).
            save_fn: Optional custom save function ``(path: str) -> None``.
            save_dir: Override checkpoint save directory.
        """
        if not self._run_id:
            self._run_id = self._detect_run_id()
            if not self._run_id:
                return

        control_path = self._control_dir / f"control_{self._run_id}.json"
        if not control_path.exists():
            return

        try:
            data = json.loads(control_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read control file: %s", e)
            return

        # Skip if already acknowledged
        if data.get("acknowledged", False):
            return

        # Skip if stale
        timestamp_str = data.get("timestamp", "")
        if self._is_stale(timestamp_str):
            logger.info("Skipping stale decision from %s", timestamp_str)
            self._acknowledge(control_path, data)
            return

        decision = data.get("decision", {})
        decision_type = decision.get("decision", "CONTINUE")
        reasoning = decision.get("reasoning", "")
        health = data.get("analysis", {}).get("health_score", -1)

        logger.info(
            "[Monitor] epoch %d: %s (health: %.0f) — %s",
            epoch, decision_type, health, reasoning,
        )

        # Execute the decision
        if decision_type == "CONTINUE":
            pass

        elif decision_type == "ADJUST_LR":
            self._adjust_lr(decision, optimizer, epoch)

        elif decision_type == "EARLY_STOP":
            self._save_checkpoint(
                model, optimizer, epoch, "early_stop", save_fn, save_dir
            )
            self.should_stop = True
            logger.info("[Monitor] EARLY_STOP — training will halt after this epoch.")

        elif decision_type == "SNAPSHOT":
            self._save_checkpoint(
                model, optimizer, epoch, "snapshot", save_fn, save_dir
            )

        elif decision_type == "ALERT":
            logger.warning("[Monitor] ALERT: %s", reasoning)

        # Acknowledge
        self._acknowledge(control_path, data)

    def _adjust_lr(
        self, decision: dict, optimizer: Any | None, epoch: int
    ) -> None:
        """Multiply optimizer learning rates by the suggested factor."""
        lr_factor = decision.get("lr_factor")
        if lr_factor is None or optimizer is None:
            logger.warning(
                "[Monitor] ADJUST_LR but no lr_factor or optimizer provided."
            )
            return

        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = param_group["lr"]
            new_lr = old_lr * lr_factor
            param_group["lr"] = new_lr
            logger.info(
                "[Monitor] epoch %d: param_group[%d] LR %.2e → %.2e (factor=%.3f)",
                epoch, i, old_lr, new_lr, lr_factor,
            )

    def _save_checkpoint(
        self,
        model: Any | None,
        optimizer: Any | None,
        epoch: int,
        tag: str,
        save_fn: Any | None = None,
        save_dir: str | None = None,
    ) -> None:
        """Save a checkpoint to the checkpoint directory."""
        ckpt_dir = Path(save_dir) if save_dir else self._checkpoint_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{tag}_epoch{epoch:04d}.pt"

        if save_fn is not None:
            save_fn(str(ckpt_path))
            logger.info("[Monitor] Saved checkpoint via save_fn: %s", ckpt_path)
            return

        if model is None:
            logger.warning("[Monitor] Cannot save checkpoint — no model provided.")
            return

        try:
            import torch

            state = {
                "epoch": epoch,
                "tag": tag,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if hasattr(model, "state_dict"):
                state["model_state_dict"] = model.state_dict()
            if optimizer is not None and hasattr(optimizer, "state_dict"):
                state["optimizer_state_dict"] = optimizer.state_dict()

            torch.save(state, ckpt_path)
            logger.info("[Monitor] Saved checkpoint: %s", ckpt_path)
        except Exception as e:
            logger.error("[Monitor] Failed to save checkpoint: %s", e)

    def _acknowledge(self, path: Path, data: dict) -> None:
        """Write acknowledged=True back to the control file."""
        data["acknowledged"] = True
        data["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            tmp.replace(path)
        except OSError as e:
            logger.warning("Failed to write acknowledgement: %s", e)

    def _is_stale(self, timestamp_str: str) -> bool:
        """Check if a decision timestamp is older than the stale threshold."""
        if not timestamp_str:
            return True
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            return age > self._stale_threshold
        except (ValueError, TypeError):
            return True

    @staticmethod
    def _detect_run_id() -> str | None:
        """Try to detect the active W&B run ID."""
        try:
            import wandb

            if wandb.run is not None:
                return wandb.run.id
        except ImportError:
            pass

        # Fallback: check WANDB_RUN_ID env var
        return os.environ.get("WANDB_RUN_ID")
