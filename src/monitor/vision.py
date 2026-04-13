"""Vision analysis layer — sends rendered charts to Claude API.

Uses the Anthropic Python SDK to send training metric charts to Claude
for qualitative assessment of training curves.
"""

from __future__ import annotations

import base64
import logging

from .schemas import TrendAnalysis

logger = logging.getLogger(__name__)

try:
    import anthropic

    _HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    _HAS_ANTHROPIC = False

_ANALYSIS_PROMPT = """\
You are analyzing training metric charts for a machine learning model.

The numerical analysis has already detected the following:
- Health score: {health_score}/100
- Plateau detected: {plateau}
- Overfitting detected: {overfitting} (severity: {overfit_severity:.2f})
- Instability detected: {instability}
- Convergence detected: {convergence}

Per-metric trends:
{metric_summary}

Now look at the chart image and provide your independent qualitative assessment.
Focus on:
1. Is the loss curve trajectory healthy? Any inflection points or regime changes?
2. Does the validation loss suggest early stopping is warranted?
3. Are there any visual patterns the numerical analysis might have missed?
4. Overall training health assessment in one sentence.

Be concise — 3-5 sentences maximum. Focus on actionable observations."""


class VisionAnalyzer:
    """Sends training charts to Claude API for qualitative analysis.

    Args:
        api_key: Anthropic API key.
        model: Model to use. Defaults to claude-sonnet-4-6 for cost efficiency.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        if not _HAS_ANTHROPIC:
            raise ImportError(
                "anthropic SDK required for vision analysis. "
                "Install with: pip install anthropic"
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def analyze(
        self,
        chart_bytes: bytes,
        analysis: TrendAnalysis,
    ) -> str:
        """Send a chart image to Claude API for qualitative assessment.

        Args:
            chart_bytes: PNG image bytes of the training chart.
            analysis: Numerical analysis to include as context.

        Returns:
            Qualitative assessment string.
        """
        metric_lines = []
        for m in analysis.metrics:
            metric_lines.append(
                f"  - {m.name}: value={m.current_value:.6f}, "
                f"ema={m.ema_value:.6f}, trend={m.trend.value}, "
                f"derivative={m.derivative:.6f}"
            )
        metric_summary = "\n".join(metric_lines) if metric_lines else "  (no metrics)"

        prompt = _ANALYSIS_PROMPT.format(
            health_score=analysis.health_score,
            plateau=analysis.plateau_detected,
            overfitting=analysis.overfitting_detected,
            overfit_severity=analysis.overfitting_severity,
            instability=analysis.instability_detected,
            convergence=analysis.convergence_detected,
            metric_summary=metric_summary,
        )

        image_b64 = base64.standard_b64encode(chart_bytes).decode("ascii")

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )
            return response.content[0].text
        except Exception:
            logger.exception("Vision analysis failed")
            return ""
