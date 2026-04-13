"""W&B API poller for fetching metrics from active training runs.

Uses ``wandb.Api()`` to query active runs across configured projects
and returns ``RunSnapshot`` objects with metric histories for downstream
analysis.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import WatchConfig
from .schemas import MetricHistory, RunSnapshot

logger = logging.getLogger(__name__)

try:
    import wandb

    _HAS_WANDB = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _HAS_WANDB = False


class WandbPoller:
    """Polls W&B API for metrics from active training runs.

    Args:
        watches: List of project watch configurations.
        api_key: W&B API key. Falls back to ``WANDB_API_KEY`` env var.
    """

    def __init__(self, watches: list[WatchConfig], api_key: str = "") -> None:
        if not _HAS_WANDB:
            raise ImportError(
                "wandb is required for polling. Install with: pip install wandb"
            )
        self._watches = watches
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._api = wandb.Api(**kwargs)
        # Cache: run_id -> last seen step, to fetch only incremental data
        self._last_step: dict[str, int] = {}

    def poll(self) -> list[RunSnapshot]:
        """Poll all watched projects for active runs.

        Returns:
            List of ``RunSnapshot`` objects, one per active run found.
        """
        snapshots: list[RunSnapshot] = []

        for watch in self._watches:
            try:
                runs = self._poll_project(watch)
                snapshots.extend(runs)
            except Exception:
                logger.exception("Failed to poll project %s", watch.project)

        return snapshots

    def _poll_project(self, watch: WatchConfig) -> list[RunSnapshot]:
        """Poll a single W&B project for active runs."""
        path = f"{watch.entity}/{watch.project}" if watch.entity else watch.project

        try:
            runs = self._api.runs(path, filters={"state": "running"})
        except Exception:
            logger.warning("Could not fetch runs for %s", path)
            return []

        snapshots: list[RunSnapshot] = []
        for run in runs:
            try:
                snapshot = self._build_snapshot(run, watch)
                if snapshot is not None:
                    snapshots.append(snapshot)
            except Exception:
                logger.exception("Failed to build snapshot for run %s", run.id)

        if not snapshots:
            logger.info("No active runs in %s", path)

        return snapshots

    def _build_snapshot(
        self, run: Any, watch: WatchConfig
    ) -> RunSnapshot | None:
        """Build a RunSnapshot from a W&B run object."""
        run_id: str = run.id
        run_name: str = run.name or run_id

        # Fetch history — only the metrics we care about
        keys = watch.metrics
        if not keys:
            return None

        try:
            history_df = run.history(keys=keys, samples=1000, pandas=True)
        except Exception:
            logger.warning("Could not fetch history for run %s", run_id)
            return None

        if history_df is None or history_df.empty:
            return None

        # Build metric histories
        metric_histories: dict[str, MetricHistory] = {}
        for key in keys:
            if key not in history_df.columns:
                continue
            col = history_df[key].dropna()
            if col.empty:
                continue

            steps = col.index.tolist()
            values = col.tolist()

            # Use _step column if available for accurate step numbers
            if "_step" in history_df.columns:
                step_col = history_df.loc[col.index, "_step"]
                steps = step_col.tolist()

            metric_histories[key] = MetricHistory(
                name=key,
                steps=[int(s) for s in steps],
                values=[float(v) for v in values],
            )

        current_step = 0
        if "_step" in history_df.columns:
            current_step = int(history_df["_step"].dropna().iloc[-1])
        elif len(history_df) > 0:
            current_step = len(history_df) - 1

        # Extract run config (learning rate, batch size, etc.)
        run_config: dict = {}
        try:
            run_config = dict(run.config) if run.config else {}
        except Exception:
            pass

        self._last_step[run_id] = current_step

        return RunSnapshot(
            run_id=run_id,
            run_name=run_name,
            project=watch.project,
            entity=watch.entity,
            state=run.state,
            current_step=current_step,
            config=run_config,
            metric_histories=metric_histories,
        )
