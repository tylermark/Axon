"""Main orchestrator that ties all monitor components together.

Provides ``MonitorOrchestrator.run_once()`` which performs a single
poll → analyze → decide → (chart → vision) → write cycle.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from .analyzer import TrendAnalyzer
from .config import MonitorConfig
from .decision import DecisionEngine
from .schemas import (
    ControlFile,
    DecisionType,
    MonitorResult,
)

logger = logging.getLogger(__name__)


class MonitorOrchestrator:
    """Orchestrates one poll-analyze-decide-write cycle.

    Args:
        config: Fully resolved monitor configuration.
    """

    def __init__(self, config: MonitorConfig) -> None:
        self._config = config
        self._monitor_id = uuid.uuid4().hex[:12]

        # Core components
        from .poller import WandbPoller

        self._poller = WandbPoller(
            watches=config.watches,
            api_key=config.wandb_api_key,
        )
        self._analyzer = TrendAnalyzer(config.thresholds)
        self._engine = DecisionEngine(config.thresholds)

        # Drive channel
        self._drive = None
        if config.drive_folder_id and config.service_account_path:
            try:
                from .drive import DriveChannel

                self._drive = DriveChannel(
                    folder_id=config.drive_folder_id,
                    service_account_path=config.service_account_path,
                )
            except ImportError:
                logger.warning(
                    "Google Drive dependencies not installed. "
                    "Control files will not be written."
                )
            except Exception:
                logger.exception("Failed to initialize Drive channel")

        # Vision layer (optional)
        self._charts = None
        self._vision = None
        if config.vision_enabled and config.anthropic_api_key:
            try:
                from .charts import ChartRenderer
                from .vision import VisionAnalyzer

                self._charts = ChartRenderer()
                self._vision = VisionAnalyzer(api_key=config.anthropic_api_key)
            except ImportError:
                logger.warning(
                    "Vision layer dependencies not installed. "
                    "Charts and vision analysis will be skipped."
                )

        # Map watch names to configs for analyzer
        self._watch_map = {w.project: w for w in config.watches}

    def run_once(self) -> MonitorResult:
        """Execute one poll → analyze → decide → write cycle.

        Returns:
            ``MonitorResult`` summarizing what happened.
        """
        result = MonitorResult()

        # 1. Poll W&B for active runs
        snapshots = self._poller.poll()
        result.snapshots = snapshots

        if not snapshots:
            logger.info("No active runs found.")
            result.suggested_sleep_seconds = self._config.poll_interval_idle
            return result

        has_issues = False
        controls_written = 0

        for snapshot in snapshots:
            # 2. Analyze trends
            watch = self._watch_map.get(snapshot.project)
            analysis = self._analyzer.analyze(snapshot, watch)
            result.analyses.append(analysis)

            # 3. Make decision
            decision = self._engine.decide(analysis, snapshot)

            # 4. Optional vision layer
            if self._charts and self._vision:
                try:
                    chart_bytes = self._charts.render(snapshot, analysis)
                    insight = self._vision.analyze(chart_bytes, analysis)
                    if insight:
                        decision.vision_insight = insight
                except Exception:
                    logger.exception("Vision layer failed for run %s", snapshot.run_id)

            result.decisions.append(decision)

            # 5. Write control file (skip CONTINUE, skip if unacknowledged
            #    unless the new decision is higher severity)
            if decision.decision != DecisionType.CONTINUE and self._drive:
                write_allowed = True
                if not self._drive.is_acknowledged(snapshot.run_id):
                    # Check whether the new decision outranks what's on disk
                    existing = self._drive.read_control(snapshot.run_id)
                    if (
                        existing is not None
                        and decision.decision.severity
                        > existing.decision.decision.severity
                    ):
                        logger.info(
                            "Escalating for run %s — overwriting "
                            "unacknowledged %s with higher-severity %s",
                            snapshot.run_id,
                            existing.decision.decision.value,
                            decision.decision.value,
                        )
                    elif existing is not None:
                        logger.warning(
                            "Skipping write for run %s — previous decision "
                            "%s not yet acknowledged (new: %s, same or lower "
                            "severity)",
                            snapshot.run_id,
                            existing.decision.decision.value,
                            decision.decision.value,
                        )
                        write_allowed = False
                    # existing is None: no file on disk, safe to write

                if write_allowed:
                    control = ControlFile(
                        timestamp=datetime.now(timezone.utc),
                        monitor_id=self._monitor_id,
                        wandb_project=snapshot.project,
                        wandb_run_id=snapshot.run_id,
                        epoch_observed=snapshot.current_step,
                        decision=decision,
                        analysis=analysis,
                    )
                    try:
                        self._drive.write_control(control)
                        controls_written += 1
                    except Exception:
                        logger.exception(
                            "Failed to write control file for run %s",
                            snapshot.run_id,
                        )

            if decision.decision not in (DecisionType.CONTINUE, DecisionType.SNAPSHOT):
                has_issues = True

            # Log the decision
            logger.info(
                "[%s] %s (health: %.0f, confidence: %.2f) — %s",
                snapshot.run_name,
                decision.decision.value,
                analysis.health_score,
                decision.confidence,
                decision.reasoning,
            )

        result.controls_written = controls_written

        # 6. Determine next poll interval
        if has_issues:
            result.suggested_sleep_seconds = self._config.poll_interval_issues
        else:
            result.suggested_sleep_seconds = self._config.poll_interval_active

        return result
