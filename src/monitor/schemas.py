"""Shared data models for the autonomous training monitor.

Defines Pydantic schemas for control files, decisions, metric snapshots,
and all inter-component data transfer objects.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DecisionType(str, Enum):
    """Action the monitor recommends for the training run."""

    CONTINUE = "CONTINUE"
    ADJUST_LR = "ADJUST_LR"
    EARLY_STOP = "EARLY_STOP"
    SNAPSHOT = "SNAPSHOT"
    ALERT = "ALERT"


class LRAction(str, Enum):
    """Specific learning-rate adjustment action."""

    REDUCE = "REDUCE"
    INCREASE = "INCREASE"
    WARMUP_RESTART = "WARMUP_RESTART"


class TrendDirection(str, Enum):
    """Direction of a metric's trend."""

    IMPROVING = "improving"
    PLATEAU = "plateau"
    DEGRADING = "degrading"
    UNSTABLE = "unstable"


# ---------------------------------------------------------------------------
# Metric models
# ---------------------------------------------------------------------------


class MetricSnapshot(BaseModel):
    """Point-in-time summary of a single tracked metric."""

    name: str
    current_value: float
    ema_value: float
    trend: TrendDirection
    derivative: float = Field(
        description="Finite-difference derivative of the EMA series."
    )


class MetricHistory(BaseModel):
    """Time-ordered history of a single metric (step, value) pairs."""

    name: str
    steps: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Analysis models
# ---------------------------------------------------------------------------


class TrendAnalysis(BaseModel):
    """Result of statistical trend detection on a training run."""

    plateau_detected: bool = False
    overfitting_detected: bool = False
    overfitting_severity: float = 0.0
    instability_detected: bool = False
    convergence_detected: bool = False
    health_score: float = Field(
        default=100.0, ge=0.0, le=100.0, description="Composite 0-100 health score."
    )
    metrics: list[MetricSnapshot] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Decision models
# ---------------------------------------------------------------------------


class Decision(BaseModel):
    """Actionable decision produced by the DecisionEngine."""

    decision: DecisionType
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    lr_action: LRAction | None = None
    lr_factor: float | None = Field(
        default=None, description="Multiply current LR by this factor."
    )
    vision_insight: str | None = None


# ---------------------------------------------------------------------------
# Control file (Drive communication)
# ---------------------------------------------------------------------------


class ControlFile(BaseModel):
    """JSON schema written to Google Drive for Colab callback consumption."""

    version: str = "1.0"
    timestamp: datetime
    monitor_id: str = Field(description="Unique ID for this monitor instance.")
    wandb_project: str
    wandb_run_id: str
    epoch_observed: int
    decision: Decision
    analysis: TrendAnalysis
    acknowledged: bool = False
    acknowledged_at: datetime | None = None


# ---------------------------------------------------------------------------
# Poller models
# ---------------------------------------------------------------------------


class RunSnapshot(BaseModel):
    """Snapshot of a W&B run's current state and metric histories."""

    run_id: str
    run_name: str = ""
    project: str
    entity: str = ""
    state: str = "running"
    current_step: int = 0
    config: dict = Field(default_factory=dict)
    metric_histories: dict[str, MetricHistory] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Orchestrator result
# ---------------------------------------------------------------------------


class MonitorResult(BaseModel):
    """Output of a single monitor poll-analyze-decide cycle."""

    snapshots: list[RunSnapshot] = Field(default_factory=list)
    analyses: list[TrendAnalysis] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    controls_written: int = 0
    chart_paths: list[str] = Field(default_factory=list)
    suggested_sleep_seconds: int = 270
