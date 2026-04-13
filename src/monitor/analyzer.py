"""Statistical trend detection on training metric histories.

Detects plateaus, overfitting, instability, and convergence using
exponential moving averages, rolling variance, and finite differences.
"""

from __future__ import annotations

import logging

import numpy as np

from .config import AnalysisThresholds, WatchConfig
from .schemas import (
    MetricHistory,
    MetricSnapshot,
    RunSnapshot,
    TrendAnalysis,
    TrendDirection,
)

logger = logging.getLogger(__name__)


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """Compute exponential moving average."""
    alpha = 2.0 / (span + 1)
    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def _rolling_cv(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling coefficient of variation (std / |mean|)."""
    n = len(values)
    cv = np.zeros(n)
    for i in range(window - 1, n):
        segment = values[i - window + 1 : i + 1]
        mean = np.mean(segment)
        if abs(mean) > 1e-10:
            cv[i] = np.std(segment) / abs(mean)
    return cv


class TrendAnalyzer:
    """Statistical trend detection on metric histories.

    Args:
        thresholds: Detection thresholds and window sizes.
    """

    def __init__(self, thresholds: AnalysisThresholds) -> None:
        self._t = thresholds

    def analyze(
        self, snapshot: RunSnapshot, watch: WatchConfig | None = None
    ) -> TrendAnalysis:
        """Run all trend detectors on a run snapshot.

        Args:
            snapshot: Current run state with metric histories.
            watch: Optional watch config to identify train/val loss keys.

        Returns:
            ``TrendAnalysis`` with per-metric trends and boolean flags.
        """
        metric_snapshots: list[MetricSnapshot] = []
        loss_histories: list[MetricHistory] = []

        for key, hist in snapshot.metric_histories.items():
            if len(hist.values) < 3:
                continue

            values = np.array(hist.values, dtype=np.float64)
            ema_vals = _ema(values, self._t.plateau_window)
            derivative = float(np.diff(ema_vals)[-1]) if len(ema_vals) > 1 else 0.0

            trend = self._classify_trend(values, ema_vals)

            metric_snapshots.append(
                MetricSnapshot(
                    name=key,
                    current_value=float(values[-1]),
                    ema_value=float(ema_vals[-1]),
                    trend=trend,
                    derivative=derivative,
                )
            )

            # Collect loss-like metrics for composite analysis
            if "loss" in key.lower():
                loss_histories.append(hist)

        # Detect plateau on primary loss metric
        plateau = self._detect_plateau(snapshot, watch)

        # Detect overfitting (train vs val divergence)
        overfitting, severity = self._detect_overfitting(snapshot, watch)

        # Detect instability
        instability = self._detect_instability(snapshot)

        # Detect convergence
        convergence = self._detect_convergence(
            metric_snapshots, plateau, overfitting
        )

        # Composite health score
        health = self._compute_health_score(
            metric_snapshots, plateau, overfitting, severity, instability
        )

        return TrendAnalysis(
            plateau_detected=plateau,
            overfitting_detected=overfitting,
            overfitting_severity=severity,
            instability_detected=instability,
            convergence_detected=convergence,
            health_score=health,
            metrics=metric_snapshots,
        )

    def _classify_trend(
        self, values: np.ndarray, ema_vals: np.ndarray
    ) -> TrendDirection:
        """Classify the trend direction of a metric series."""
        if len(ema_vals) < 3:
            return TrendDirection.PLATEAU

        recent_derivs = np.diff(ema_vals[-self._t.plateau_window :])
        mean_deriv = float(np.mean(recent_derivs))
        cv = float(np.std(values[-self._t.instability_window :]))
        mean_val = float(np.mean(np.abs(values[-self._t.instability_window :])))

        if mean_val > 1e-10 and cv / mean_val > self._t.instability_cv_threshold:
            return TrendDirection.UNSTABLE

        rel_threshold = abs(float(ema_vals[-1])) * self._t.plateau_relative_threshold
        if rel_threshold < 1e-10:
            rel_threshold = 1e-10

        if abs(mean_deriv) < rel_threshold:
            return TrendDirection.PLATEAU
        elif mean_deriv < 0:
            return TrendDirection.IMPROVING
        else:
            return TrendDirection.DEGRADING

    def _detect_plateau(
        self, snapshot: RunSnapshot, watch: WatchConfig | None
    ) -> bool:
        """Detect plateau on the primary loss metric."""
        # Find the primary loss metric
        loss_key = ""
        if watch and watch.train_loss_key:
            loss_key = watch.train_loss_key
        else:
            # Heuristic: first metric with "loss" in the name
            for key in snapshot.metric_histories:
                if "loss" in key.lower() and "eval" not in key.lower():
                    loss_key = key
                    break

        if not loss_key or loss_key not in snapshot.metric_histories:
            return False

        hist = snapshot.metric_histories[loss_key]
        if len(hist.values) < self._t.plateau_min_epochs + 1:
            return False

        values = np.array(hist.values, dtype=np.float64)
        ema_vals = _ema(values, self._t.plateau_window)
        derivs = np.diff(ema_vals)

        # Check if the last N derivatives are all below threshold
        recent = derivs[-self._t.plateau_min_epochs :]
        threshold = abs(float(ema_vals[-1])) * self._t.plateau_relative_threshold
        if threshold < 1e-10:
            threshold = 1e-10

        return bool(np.all(np.abs(recent) < threshold))

    def _detect_overfitting(
        self, snapshot: RunSnapshot, watch: WatchConfig | None
    ) -> tuple[bool, float]:
        """Detect train/val loss divergence.

        Returns:
            (overfitting_detected, severity_ratio)
        """
        train_key = ""
        val_key = ""

        if watch:
            train_key = watch.train_loss_key
            val_key = watch.val_loss_key

        # Fallback heuristic
        if not train_key or not val_key:
            for key in snapshot.metric_histories:
                kl = key.lower()
                if "loss" in kl and "eval" not in kl and "val" not in kl:
                    if not train_key:
                        train_key = key
                if "loss" in kl and ("eval" in kl or "val" in kl):
                    if not val_key:
                        val_key = key

        if (
            not train_key
            or not val_key
            or train_key not in snapshot.metric_histories
            or val_key not in snapshot.metric_histories
        ):
            return False, 0.0

        train_hist = snapshot.metric_histories[train_key]
        val_hist = snapshot.metric_histories[val_key]

        # Align lengths
        min_len = min(len(train_hist.values), len(val_hist.values))
        if min_len < self._t.overfit_min_epochs + self._t.plateau_window:
            return False, 0.0

        train_vals = np.array(train_hist.values[:min_len], dtype=np.float64)
        val_vals = np.array(val_hist.values[:min_len], dtype=np.float64)

        train_ema = _ema(train_vals, self._t.plateau_window)
        val_ema = _ema(val_vals, self._t.plateau_window)

        train_derivs = np.diff(train_ema)
        val_derivs = np.diff(val_ema)

        # Overfitting: train decreasing while val increasing
        recent_train = train_derivs[-self._t.overfit_min_epochs :]
        recent_val = val_derivs[-self._t.overfit_min_epochs :]

        train_decreasing = np.all(recent_train < 0)
        val_increasing = np.all(recent_val > 0)

        if train_decreasing and val_increasing:
            # Severity: ratio of val increase rate to train decrease rate
            val_rate = float(np.mean(recent_val))
            train_rate = float(np.mean(np.abs(recent_train)))
            severity = val_rate / max(train_rate, 1e-10)
            return True, severity

        return False, 0.0

    def _detect_instability(self, snapshot: RunSnapshot) -> bool:
        """Detect instability via rolling coefficient of variation."""
        for key, hist in snapshot.metric_histories.items():
            if "loss" not in key.lower():
                continue
            if len(hist.values) < self._t.instability_window:
                continue

            values = np.array(hist.values, dtype=np.float64)
            cv = _rolling_cv(values, self._t.instability_window)

            # Check recent CV
            if cv[-1] > self._t.instability_cv_threshold:
                return True

            # Check for spikes (step-to-step jumps > 3*std)
            diffs = np.abs(np.diff(values))
            if len(diffs) >= self._t.instability_window:
                recent_diffs = diffs[-self._t.instability_window :]
                std = np.std(recent_diffs)
                if std > 1e-10 and np.any(
                    recent_diffs > self._t.spike_std_multiplier * std
                ):
                    return True

        return False

    def _detect_convergence(
        self,
        metric_snapshots: list[MetricSnapshot],
        plateau: bool,
        overfitting: bool,
    ) -> bool:
        """Detect convergence: all metrics plateaued, no overfitting."""
        if overfitting:
            return False
        if not plateau:
            return False

        # All loss metrics should be plateaued or improving
        loss_metrics = [m for m in metric_snapshots if "loss" in m.name.lower()]
        if not loss_metrics:
            return False

        return all(
            m.trend in (TrendDirection.PLATEAU, TrendDirection.IMPROVING)
            for m in loss_metrics
        )

    def _compute_health_score(
        self,
        metric_snapshots: list[MetricSnapshot],
        plateau: bool,
        overfitting: bool,
        overfit_severity: float,
        instability: bool,
    ) -> float:
        """Compute composite 0-100 health score.

        Weights:
        - Loss trend (0.3): improving=high, plateau=medium, degrading=low
        - Stability (0.2): stable=high, unstable=low
        - Overfitting gap (0.3): none=high, detected=low
        - General progress (0.2): based on derivative magnitude
        """
        loss_metrics = [m for m in metric_snapshots if "loss" in m.name.lower()]
        if not loss_metrics:
            return 50.0  # No loss metrics — neutral

        # Loss trend score (0-100)
        trend_scores = {
            TrendDirection.IMPROVING: 100.0,
            TrendDirection.PLATEAU: 50.0,
            TrendDirection.DEGRADING: 10.0,
            TrendDirection.UNSTABLE: 20.0,
        }
        loss_trend_score = float(
            np.mean([trend_scores[m.trend] for m in loss_metrics])
        )

        # Stability score
        stability_score = 20.0 if instability else 100.0

        # Overfitting score
        if overfitting:
            overfit_score = max(0.0, 100.0 - overfit_severity * 50.0)
        else:
            overfit_score = 100.0

        # Progress score (based on whether loss is actively decreasing)
        if any(m.derivative < 0 for m in loss_metrics):
            progress_score = 80.0
        elif plateau:
            progress_score = 40.0
        else:
            progress_score = 60.0

        health = (
            0.3 * loss_trend_score
            + 0.2 * stability_score
            + 0.3 * overfit_score
            + 0.2 * progress_score
        )

        return round(max(0.0, min(100.0, health)), 1)
