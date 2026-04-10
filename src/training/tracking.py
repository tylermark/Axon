"""TR-006: Unified experiment tracking and checkpoint management.

Provides ``ExperimentTracker`` for W&B + local logging and
``CheckpointManager`` for saving/loading training checkpoints with
metadata.  W&B is imported conditionally — when unavailable or disabled,
metrics fall back to local JSON logging so training still works on
machines without network access (e.g. offline Colab runs).

Reference: TASKS.md TR-006, ARCHITECTURE.md §Training Pipeline.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Optional W&B import ──────────────────────────────────────────────────

try:
    import wandb

    _HAS_WANDB = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _HAS_WANDB = False


# ── ExperimentTracker ────────────────────────────────────────────────────


class ExperimentTracker:
    """Unified W&B + local logging for all training phases.

    When W&B is available and ``enabled=True``, metrics are logged to both
    W&B and a local JSONL file.  When W&B is unavailable or disabled, only
    the local JSONL file is written.

    Args:
        project: W&B project name (e.g. ``"axon-drl"``).
        run_name: Human-readable run identifier.
        config: Flat dictionary of hyperparameters to log.
        enabled: If ``False``, W&B is not initialised even when installed.
        log_dir: Directory for local JSONL log files.  Defaults to
            ``"logs/{project}/{run_name}"``.
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        config: dict[str, Any],
        enabled: bool = True,
        log_dir: str | Path | None = None,
    ) -> None:
        self.project = project
        self.run_name = run_name
        self.config = config
        self._wandb_enabled = enabled and _HAS_WANDB
        self._run: Any = None

        # Local JSONL log
        if log_dir is None:
            log_dir = Path("logs") / project / run_name
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / "metrics.jsonl"
        self._log_file = open(self._log_path, "a")  # noqa: SIM115

        # Write config as first entry
        self._write_local({"_type": "config", **config})

        # Initialise W&B
        if self._wandb_enabled:
            try:
                self._run = wandb.init(
                    project=project,
                    name=run_name,
                    config=config,
                    reinit=True,
                )
                logger.info("W&B run initialised: %s/%s", project, run_name)
            except Exception:
                logger.warning("W&B initialisation failed; falling back to local logging.")
                self._wandb_enabled = False
                self._run = None
        else:
            if enabled and not _HAS_WANDB:
                logger.info(
                    "wandb not installed — using local JSONL logging at %s",
                    self._log_path,
                )

    # ── Public API ────────────────────────────────────────────────────

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics at a given global step.

        Args:
            metrics: Dictionary of metric name → value.
            step: Global training step (timestep or epoch).
        """
        entry = {"_step": step, "_timestamp": time.time(), **metrics}
        self._write_local(entry)

        if self._wandb_enabled and self._run is not None:
            wandb.log(metrics, step=step)

    def log_artifact(self, path: str | Path, name: str, type: str) -> None:
        """Log a file artifact (checkpoint, dataset, etc.) to W&B.

        When W&B is disabled the artifact path is recorded locally only.

        Args:
            path: Path to the artifact file or directory.
            name: Artifact name in the W&B registry.
            type: Artifact type (e.g. ``"model"``, ``"dataset"``).
        """
        self._write_local(
            {"_type": "artifact", "path": str(path), "name": name, "artifact_type": type}
        )

        if self._wandb_enabled and self._run is not None:
            try:
                artifact = wandb.Artifact(name=name, type=type)
                p = Path(path)
                if p.is_dir():
                    artifact.add_dir(str(p))
                else:
                    artifact.add_file(str(p))
                self._run.log_artifact(artifact)
            except Exception:
                logger.warning("Failed to log artifact '%s' to W&B.", name, exc_info=True)

    def __del__(self) -> None:
        if hasattr(self, "_log_file") and not self._log_file.closed:
            self._log_file.close()

    def finish(self) -> None:
        """Finalise the tracking run.

        Flushes local logs and finishes the W&B run if active.
        """
        self._log_file.close()
        if self._wandb_enabled and self._run is not None:
            wandb.finish()
            self._run = None
        logger.info("Experiment tracking finished. Local log: %s", self._log_path)

    # ── Internals ─────────────────────────────────────────────────────

    def _write_local(self, entry: dict[str, Any]) -> None:
        """Append a JSON line to the local log file."""
        self._log_file.write(json.dumps(entry, default=str) + "\n")
        self._log_file.flush()


# ── CheckpointManager ────────────────────────────────────────────────────


@dataclass
class CheckpointMeta:
    """Metadata stored alongside each checkpoint."""

    epoch: int
    metrics: dict[str, float]
    timestamp: float
    path: str


class CheckpointManager:
    """Save and load training checkpoints with metadata.

    Keeps the *N* most recent checkpoints on disk and deletes older ones
    automatically.  Supports loading by ``latest`` or ``best`` (for a
    given metric).

    Args:
        checkpoint_dir: Directory where checkpoints are stored.
        max_checkpoints: Maximum number of checkpoint files to retain.
    """

    def __init__(self, checkpoint_dir: str | Path, max_checkpoints: int = 5) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self._manifest_path = self.checkpoint_dir / "manifest.json"
        self._manifest: list[dict[str, Any]] = self._load_manifest()

    # ── Public API ────────────────────────────────────────────────────

    def save(self, state: dict[str, Any], epoch: int, metrics: dict[str, float]) -> Path:
        """Save a checkpoint and return its path.

        The checkpoint is stored as ``checkpoint_epoch_{N}.pt`` containing the
        provided ``state`` dict.  The state should include at minimum::

            {
                "model_state_dict": ...,
                "optimizer_state_dict": ...,
                "scheduler_state_dict": ...,
                "epoch": N,
                "metrics": {...},
                "config": {...},
            }

        After saving, older checkpoints beyond ``max_checkpoints`` are deleted.

        Args:
            state: Dictionary to serialise (must be torch-serialisable).
            epoch: Current epoch / training step.
            metrics: Metrics at time of save (used for ``load_best``).

        Returns:
            Path to the saved checkpoint file.
        """
        import torch

        filename = f"checkpoint_epoch_{epoch}.pt"
        path = self.checkpoint_dir / filename
        torch.save(state, path)

        meta = {
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": time.time(),
            "path": str(path),
        }
        self._manifest.append(meta)
        self._save_manifest()

        # Prune old checkpoints
        self._prune()

        logger.info("Checkpoint saved: %s (epoch %d)", path, epoch)
        return path

    def load_latest(self) -> dict[str, Any] | None:
        """Load the most recent checkpoint.

        Returns:
            The deserialised state dict, or ``None`` if no checkpoints exist.
        """
        if not self._manifest:
            logger.info("No checkpoints found in %s", self.checkpoint_dir)
            return None

        latest = max(self._manifest, key=lambda m: m["epoch"])
        return self._load_checkpoint(latest)

    def load_best(
        self, metric: str = "loss", lower_is_better: bool = True
    ) -> dict[str, Any] | None:
        """Load the checkpoint with the best value for ``metric``.

        Args:
            metric: Name of the metric to optimise.
            lower_is_better: If ``True``, the checkpoint with the lowest
                value of ``metric`` is selected.

        Returns:
            The deserialised state dict, or ``None`` if no checkpoints
            contain the requested metric.
        """
        candidates = [m for m in self._manifest if metric in m.get("metrics", {})]
        if not candidates:
            logger.info(
                "No checkpoints with metric '%s' found in %s",
                metric,
                self.checkpoint_dir,
            )
            return None

        best = (min if lower_is_better else max)(
            candidates,
            key=lambda m: m["metrics"][metric],
        )
        return self._load_checkpoint(best)

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """Return metadata for all retained checkpoints.

        Returns:
            List of dicts with keys ``epoch``, ``metrics``, ``timestamp``, ``path``.
        """
        return list(self._manifest)

    # ── Internals ─────────────────────────────────────────────────────

    def _load_manifest(self) -> list[dict[str, Any]]:
        """Load the checkpoint manifest from disk."""
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt manifest at %s — starting fresh.", self._manifest_path)
        return []

    def _save_manifest(self) -> None:
        """Persist the manifest to disk."""
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2, default=str)

    def _load_checkpoint(self, meta: dict[str, Any]) -> dict[str, Any] | None:
        """Load a checkpoint file referenced by its manifest entry."""
        import torch

        path = Path(meta["path"])
        if not path.exists():
            logger.warning("Checkpoint file missing: %s", path)
            return None

        logger.info("Loading checkpoint: %s (epoch %d)", path, meta["epoch"])
        return torch.load(path, map_location="cpu", weights_only=False)

    def _prune(self) -> None:
        """Delete checkpoints beyond ``max_checkpoints``."""
        if len(self._manifest) <= self.max_checkpoints:
            return

        # Sort by epoch ascending — keep the most recent ones
        sorted_entries = sorted(self._manifest, key=lambda m: m["epoch"])
        to_remove = sorted_entries[: len(sorted_entries) - self.max_checkpoints]

        for entry in to_remove:
            path = Path(entry["path"])
            if path.exists():
                path.unlink()
                logger.debug("Pruned old checkpoint: %s", path)
            self._manifest.remove(entry)

        self._save_manifest()
