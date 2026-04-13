"""Autonomous training monitor for Axon.

Polls W&B for active training runs, analyzes metric trends, makes
decisions (continue, adjust LR, early stop, snapshot, alert), and
writes control files to Google Drive for Colab callback consumption.

Usage (Claude Code /loop)::

    python -m src.monitor once --config monitor.yaml

Usage (standalone daemon)::

    python -m src.monitor watch --config monitor.yaml
"""

from .config import MonitorConfig, load_config
from .orchestrator import MonitorOrchestrator
from .schemas import (
    ControlFile,
    Decision,
    DecisionType,
    LRAction,
    MetricHistory,
    MetricSnapshot,
    MonitorResult,
    RunSnapshot,
    TrendAnalysis,
    TrendDirection,
)

__all__ = [
    "ControlFile",
    "Decision",
    "DecisionType",
    "LRAction",
    "MetricHistory",
    "MetricSnapshot",
    "MonitorConfig",
    "MonitorOrchestrator",
    "MonitorResult",
    "RunSnapshot",
    "TrendAnalysis",
    "TrendDirection",
    "load_config",
]
