"""Metric chart rendering for the vision analysis layer.

Renders multi-panel matplotlib figures from polled metric histories.
Uses the ``Agg`` backend for headless server-side rendering.
"""

from __future__ import annotations

import io
import logging

import numpy as np

from .analyzer import _ema
from .schemas import MetricHistory, RunSnapshot, TrendAnalysis

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class ChartRenderer:
    """Renders training metric charts as PNG images.

    Produces a 2x2 panel figure:
    - Top-left: Loss curves with EMA overlay
    - Top-right: Non-loss metrics (reward, SPUR, coverage, etc.)
    - Bottom-left: Learning rate schedule (if available)
    - Bottom-right: Health score summary + trend indicators
    """

    def __init__(self, ema_span: int = 10) -> None:
        if not _HAS_MPL:
            raise ImportError(
                "matplotlib is required for chart rendering. "
                "Install with: pip install matplotlib"
            )
        self._ema_span = ema_span

    def render(
        self, snapshot: RunSnapshot, analysis: TrendAnalysis
    ) -> bytes:
        """Render a multi-panel chart as PNG bytes.

        Args:
            snapshot: Run state with metric histories.
            analysis: Trend analysis results for annotations.

        Returns:
            PNG image as bytes.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        try:
            fig.suptitle(
                f"Training Monitor — {snapshot.run_name} "
                f"(step {snapshot.current_step})",
                fontsize=14,
                fontweight="bold",
            )

            self._plot_losses(axes[0, 0], snapshot)
            self._plot_metrics(axes[0, 1], snapshot)
            self._plot_lr(axes[1, 0], snapshot)
            self._plot_health(axes[1, 1], analysis)

            fig.tight_layout(rect=[0, 0, 1, 0.95])

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            return buf.read()
        finally:
            plt.close(fig)

    def _plot_losses(self, ax: plt.Axes, snapshot: RunSnapshot) -> None:
        """Plot loss curves with EMA overlay."""
        ax.set_title("Loss Curves")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

        plotted = False
        for key, hist in snapshot.metric_histories.items():
            if "loss" not in key.lower() or len(hist.values) < 2:
                continue

            steps = np.array(hist.steps)
            values = np.array(hist.values, dtype=np.float64)
            ema_vals = _ema(values, self._ema_span)

            label = key.split("/")[-1]
            ax.plot(steps, values, alpha=0.3, linewidth=0.8)
            ax.plot(steps, ema_vals, linewidth=2, label=label)
            plotted = True

        if plotted:
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No loss metrics", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")

    def _plot_metrics(self, ax: plt.Axes, snapshot: RunSnapshot) -> None:
        """Plot non-loss metrics (reward, SPUR, coverage, etc.)."""
        ax.set_title("Metrics")
        ax.set_xlabel("Step")

        plotted = False
        for key, hist in snapshot.metric_histories.items():
            if "loss" in key.lower() or "lr" in key.lower() or len(hist.values) < 2:
                continue

            steps = np.array(hist.steps)
            values = np.array(hist.values, dtype=np.float64)

            label = key.split("/")[-1]
            ax.plot(steps, values, linewidth=1.5, label=label, marker=".", markersize=3)
            plotted = True

        if plotted:
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No non-loss metrics", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")

    def _plot_lr(self, ax: plt.Axes, snapshot: RunSnapshot) -> None:
        """Plot learning rate schedule if available."""
        ax.set_title("Learning Rate")
        ax.set_xlabel("Step")
        ax.set_ylabel("LR")

        lr_hist = None
        for key, hist in snapshot.metric_histories.items():
            if "lr" in key.lower() and len(hist.values) >= 2:
                lr_hist = hist
                break

        if lr_hist:
            steps = np.array(lr_hist.steps)
            values = np.array(lr_hist.values, dtype=np.float64)
            ax.plot(steps, values, linewidth=1.5, color="tab:orange")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No LR data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")

    def _plot_health(self, ax: plt.Axes, analysis: TrendAnalysis) -> None:
        """Plot health score summary with status indicators."""
        ax.set_title("Health Summary")
        ax.axis("off")

        # Health score gauge
        score = analysis.health_score
        if score >= 70:
            color = "#2ecc71"
            status = "HEALTHY"
        elif score >= 40:
            color = "#f39c12"
            status = "WARNING"
        else:
            color = "#e74c3c"
            status = "CRITICAL"

        ax.text(0.5, 0.85, f"{score:.0f}/100", transform=ax.transAxes,
                ha="center", va="center", fontsize=36, fontweight="bold", color=color)
        ax.text(0.5, 0.7, status, transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=color)

        # Status flags
        flags = []
        if analysis.plateau_detected:
            flags.append("PLATEAU detected")
        if analysis.overfitting_detected:
            flags.append(f"OVERFITTING (severity: {analysis.overfitting_severity:.2f})")
        if analysis.instability_detected:
            flags.append("INSTABILITY detected")
        if analysis.convergence_detected:
            flags.append("CONVERGENCE detected")
        if not flags:
            flags.append("Training progressing normally")

        for i, flag in enumerate(flags):
            y = 0.5 - i * 0.12
            marker = "!" if "detected" in flag.lower() else ">"
            ax.text(0.1, y, f"  {marker}  {flag}", transform=ax.transAxes,
                    fontsize=10, va="center", fontfamily="monospace")
