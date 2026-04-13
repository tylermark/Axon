"""Decision engine that maps trend analysis to actionable training decisions.

Uses priority-ordered rules: instability > overfitting > plateau > convergence.
Tracks decision history to avoid repeating LR reductions endlessly.
"""

from __future__ import annotations

import logging

from .schemas import Decision, DecisionType, LRAction, RunSnapshot, TrendAnalysis

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Maps ``TrendAnalysis`` results to ``Decision`` objects.

    Maintains per-run state to track how many times the LR has been
    reduced, preventing infinite adjustment loops.
    """

    def __init__(self) -> None:
        # run_id -> number of LR reductions applied
        self._lr_reductions: dict[str, int] = {}
        # run_id -> last decision type
        self._last_decision: dict[str, DecisionType] = {}

    def decide(self, analysis: TrendAnalysis, run: RunSnapshot) -> Decision:
        """Produce a decision from the trend analysis.

        Priority order:
        1. Instability + low health → ALERT + ADJUST_LR(REDUCE)
        2. Severe overfitting → EARLY_STOP
        3. Mild overfitting → ADJUST_LR(REDUCE)
        4. Plateau → ADJUST_LR(REDUCE) or EARLY_STOP if already reduced 2x
        5. Convergence → SNAPSHOT
        6. Default → CONTINUE
        """
        run_id = run.run_id
        reductions = self._lr_reductions.get(run_id, 0)

        decision = self._evaluate(analysis, run_id, reductions)

        # Track state
        self._last_decision[run_id] = decision.decision
        if decision.decision == DecisionType.ADJUST_LR and decision.lr_action == LRAction.REDUCE:
            self._lr_reductions[run_id] = reductions + 1

        return decision

    def _evaluate(
        self, analysis: TrendAnalysis, run_id: str, reductions: int
    ) -> Decision:
        """Core decision logic."""

        # 1. Instability — urgent
        if analysis.instability_detected and analysis.health_score < 30:
            return Decision(
                decision=DecisionType.ADJUST_LR,
                reasoning=(
                    f"Training instability detected with health score "
                    f"{analysis.health_score:.0f}/100. Reducing learning rate "
                    f"by 50% to stabilize. Consider stopping if this persists."
                ),
                confidence=0.85,
                lr_action=LRAction.REDUCE,
                lr_factor=0.5,
            )

        if analysis.instability_detected:
            return Decision(
                decision=DecisionType.ALERT,
                reasoning=(
                    f"Training instability detected (health {analysis.health_score:.0f}/100) "
                    f"but not yet critical. Monitoring closely."
                ),
                confidence=0.7,
            )

        # 2. Overfitting — severe
        if analysis.overfitting_detected and analysis.overfitting_severity > 1.5:
            return Decision(
                decision=DecisionType.EARLY_STOP,
                reasoning=(
                    f"Severe overfitting detected: validation loss diverging from "
                    f"training loss with severity {analysis.overfitting_severity:.2f}. "
                    f"Stopping training to prevent further degradation."
                ),
                confidence=0.9,
            )

        # 3. Overfitting — mild
        if analysis.overfitting_detected:
            return Decision(
                decision=DecisionType.ADJUST_LR,
                reasoning=(
                    f"Mild overfitting detected (severity {analysis.overfitting_severity:.2f}). "
                    f"Reducing learning rate by 50% to slow down fitting."
                ),
                confidence=0.75,
                lr_action=LRAction.REDUCE,
                lr_factor=0.5,
            )

        # 4. Plateau — reduce LR or stop if already tried
        if analysis.plateau_detected:
            if reductions >= 2:
                return Decision(
                    decision=DecisionType.EARLY_STOP,
                    reasoning=(
                        f"Loss plateau detected after {reductions} learning rate "
                        f"reductions. Model has likely converged. Stopping training."
                    ),
                    confidence=0.8,
                )
            return Decision(
                decision=DecisionType.ADJUST_LR,
                reasoning=(
                    f"Loss plateau detected. Reducing learning rate by 10x "
                    f"(reduction #{reductions + 1}) to escape plateau."
                ),
                confidence=0.7,
                lr_action=LRAction.REDUCE,
                lr_factor=0.1,
            )

        # 5. Convergence — snapshot
        if analysis.convergence_detected:
            return Decision(
                decision=DecisionType.SNAPSHOT,
                reasoning=(
                    f"Training converged with health score "
                    f"{analysis.health_score:.0f}/100. Saving checkpoint."
                ),
                confidence=0.85,
            )

        # 6. Default — continue
        return Decision(
            decision=DecisionType.CONTINUE,
            reasoning=(
                f"Training progressing normally. Health score: "
                f"{analysis.health_score:.0f}/100."
            ),
            confidence=0.9,
        )

    def reset(self, run_id: str) -> None:
        """Reset tracked state for a run (e.g., after a restart)."""
        self._lr_reductions.pop(run_id, None)
        self._last_decision.pop(run_id, None)
