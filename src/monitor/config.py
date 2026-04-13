"""Monitor configuration with .env loading and Axon-specific defaults.

Loads secrets from ``.env`` (WANDB_API_KEY, GOOGLE_SERVICE_ACCOUNT_JSON,
MONITOR_DRIVE_FOLDER_ID, ANTHROPIC_API_KEY) and merges with a YAML
config file for watches and thresholds.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class AnalysisThresholds(BaseModel):
    """Thresholds for trend detection algorithms."""

    plateau_window: int = Field(default=10, description="EMA window size in epochs.")
    plateau_min_epochs: int = Field(
        default=15, description="Consecutive epochs below derivative threshold."
    )
    plateau_relative_threshold: float = Field(
        default=0.001, description="Relative derivative threshold for plateau."
    )
    overfit_min_epochs: int = Field(
        default=5, description="Consecutive diverging epochs before flagging."
    )
    overfit_severity_threshold: float = Field(
        default=1.5, description="Val/train divergence ratio for severe overfitting."
    )
    instability_window: int = Field(
        default=10, description="Rolling window for variance calculation."
    )
    instability_cv_threshold: float = Field(
        default=0.3, description="Coefficient of variation threshold."
    )
    spike_std_multiplier: float = Field(
        default=3.0, description="Step-to-step jump threshold in std units."
    )


class WatchConfig(BaseModel):
    """Configuration for a single W&B project to monitor."""

    name: str = Field(description="Human-readable label for this watch.")
    project: str = Field(description="W&B project name.")
    entity: str = Field(
        default="", description="W&B entity (team/user). Empty = default."
    )
    metrics: list[str] = Field(
        default_factory=list, description="Metric keys to track."
    )
    train_loss_key: str = Field(
        default="", description="Metric key for training loss (for overfitting)."
    )
    val_loss_key: str = Field(
        default="", description="Metric key for validation loss (for overfitting)."
    )


# ---------------------------------------------------------------------------
# Default Axon watches
# ---------------------------------------------------------------------------

DEFAULT_AXON_WATCHES: list[dict] = [
    {
        "name": "mpm",
        "project": "axon",
        "metrics": [
            "train/coord_loss",
            "train/type_loss",
            "train/total_loss",
            "train/lr",
        ],
        "train_loss_key": "train/total_loss",
    },
    {
        "name": "sft",
        "project": "axon-sft",
        "metrics": [
            "loss/total",
            "loss/diffusion",
            "loss/constraint",
            "eval/loss/total",
        ],
        "train_loss_key": "loss/total",
        "val_loss_key": "eval/loss/total",
    },
    {
        "name": "grpo",
        "project": "axon-grpo",
        "metrics": ["reward/mean", "kl/mean", "loss/policy", "loss/total"],
        "train_loss_key": "loss/total",
    },
    {
        "name": "drl",
        "project": "axon-drl",
        "metrics": [
            "eval/trained_reward",
            "eval/trained_spur",
            "eval/trained_wall_coverage",
        ],
    },
]


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class MonitorConfig(BaseModel):
    """Top-level configuration for the autonomous training monitor."""

    watches: list[WatchConfig] = Field(
        default_factory=lambda: [WatchConfig(**w) for w in DEFAULT_AXON_WATCHES]
    )
    thresholds: AnalysisThresholds = Field(default_factory=AnalysisThresholds)

    # Google Drive
    drive_folder_id: str = Field(
        default="", description="Google Drive folder ID for control files."
    )
    service_account_path: str = Field(
        default="", description="Path to GCP service account JSON key."
    )

    # Vision layer
    vision_enabled: bool = Field(default=False)
    anthropic_api_key: str = Field(default="")

    # W&B
    wandb_api_key: str = Field(default="")
    wandb_entity: str = Field(
        default="", description="Default W&B entity for all watches."
    )

    # Polling intervals (seconds)
    poll_interval_active: int = Field(
        default=180, description="Poll interval when runs are active and healthy."
    )
    poll_interval_issues: int = Field(
        default=120, description="Poll interval when issues detected."
    )
    poll_interval_idle: int = Field(
        default=1200, description="Poll interval when no active runs."
    )


def load_config(config_path: str | Path | None = None) -> MonitorConfig:
    """Load monitor config from .env + optional YAML file.

    Args:
        config_path: Path to YAML config file. If ``None``, looks for
            ``monitor.yaml`` in the repo root.

    Returns:
        Fully resolved ``MonitorConfig``.
    """
    # Load .env
    if _HAS_DOTENV:
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path)

    # Read YAML if available
    yaml_data: dict = {}
    if config_path is None:
        config_path = Path.cwd() / "monitor.yaml"
    config_path = Path(config_path)

    if config_path.exists() and _HAS_YAML:
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f) or {}

    # Build config, layering YAML over defaults
    config = MonitorConfig(**yaml_data)

    # Override from environment variables (secrets take precedence)
    if api_key := os.environ.get("WANDB_API_KEY", ""):
        config.wandb_api_key = api_key
    if sa_path := os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", ""):
        config.service_account_path = sa_path
    if folder_id := os.environ.get("MONITOR_DRIVE_FOLDER_ID", ""):
        config.drive_folder_id = folder_id
    if anthropic_key := os.environ.get("ANTHROPIC_API_KEY", ""):
        config.anthropic_api_key = anthropic_key
        if not yaml_data.get("vision_enabled") is False:
            config.vision_enabled = True

    # Propagate default entity to watches that don't specify one
    if config.wandb_entity:
        for watch in config.watches:
            if not watch.entity:
                watch.entity = config.wandb_entity

    return config
