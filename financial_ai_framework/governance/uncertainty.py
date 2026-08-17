"""Uncertainty gate: predictive entropy and bootstrap stability of ranking power.

Two questions, one gate:

1. **How much of the book does the model refuse to call?** Binary predictive
   entropy ``-(p ln p + (1-p) ln(1-p))`` peaks at ``ln 2 = 0.693`` when ``p = 0.5``.
   A large mass of high-entropy borrowers means most of the portfolio is being
   scored as a coin flip, which is a manual-review workload, not a decision.
2. **Is the headline ROC-AUC stable?** A bootstrap over the evaluation set gives a
   confidence interval, so the scorecard can say how much of the reported
   discrimination is sampling luck.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from ..config import Settings
from .gates import GateResult

_EPSILON = 1e-12
_MAX_BINARY_ENTROPY = float(np.log(2.0))


def binary_entropy(y_prob: np.ndarray) -> np.ndarray:
    """Per-borrower predictive entropy in nats, max ln(2) at p = 0.5."""
    p = np.clip(np.asarray(y_prob, dtype=float).ravel(), _EPSILON, 1.0 - _EPSILON)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


class UncertaintyAnalyzer:
    """Governance gate on how much of the portfolio the model cannot separate."""

    def __init__(self, settings: Settings, tracker=None):
        self.settings = settings
        self.tracker = tracker
        self.thresholds = settings.governance.uncertainty

    def run(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "model",
    ) -> GateResult:
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_prob = np.asarray(y_prob, dtype=float).ravel()

        entropy = binary_entropy(y_prob)
        high_mask = entropy > self.thresholds.high_entropy_threshold
        high_ratio = float(high_mask.mean())

        y_pred = (y_prob >= self.settings.models.decision_threshold).astype(int)
        correct = y_pred == y_true
        # Distance from the decision boundary as a confidence proxy.
        margin = np.abs(y_prob - self.settings.models.decision_threshold)
        margin_correct = float(margin[correct].mean()) if correct.any() else float("nan")
        margin_wrong = float(margin[~correct].mean()) if (~correct).any() else float("nan")

        auc_ci = self._bootstrap_auc(y_true, y_prob)
        status = self.thresholds.high_uncertainty_ratio.classify(high_ratio)

        findings: list[str] = []
        if status != "pass":
            findings.append(
                f"{high_ratio:.1%} of borrowers score above the entropy threshold "
                f"{self.thresholds.high_entropy_threshold:.2f} nats "
                f"({self.thresholds.high_uncertainty_ratio.describe()}) - that share of "
                f"the book is effectively a coin flip at the decision boundary"
            )
        if np.isfinite(margin_correct) and np.isfinite(margin_wrong) and margin_correct <= margin_wrong:
            findings.append(
                f"Confidence does not discriminate: mean margin on correct calls "
                f"{margin_correct:.4f} is not above the margin on errors {margin_wrong:.4f}"
            )
        if auc_ci["width"] > 0.05:
            findings.append(
                f"ROC-AUC 95% bootstrap interval is wide "
                f"([{auc_ci['lower']:.4f}, {auc_ci['upper']:.4f}], width {auc_ci['width']:.4f}); "
                f"the headline figure is not tightly determined by this evaluation set"
            )

        metrics: dict[str, Any] = {
            "high_uncertainty_ratio": high_ratio,
            "high_uncertainty_count": int(high_mask.sum()),
            "entropy_mean": float(entropy.mean()),
            "entropy_max_possible": _MAX_BINARY_ENTROPY,
            "entropy_p90": float(np.percentile(entropy, 90)),
            "mean_margin_correct": margin_correct,
            "mean_margin_incorrect": margin_wrong,
            "roc_auc_bootstrap_mean": auc_ci["mean"],
            "roc_auc_ci_lower": auc_ci["lower"],
            "roc_auc_ci_upper": auc_ci["upper"],
            "roc_auc_ci_width": auc_ci["width"],
        }

        if self.tracker is not None:
            for key in ("high_uncertainty_ratio", "entropy_mean", "roc_auc_ci_width"):
                self.tracker.log_metric(f"{model_name}_uncertainty_{key}", metrics[key])

        return GateResult(
            name="uncertainty",
            status=status,
            headline_metric="high_uncertainty_ratio",
            headline_value=high_ratio,
            threshold=self.thresholds.high_uncertainty_ratio.describe(),
            metrics=metrics,
            findings=findings,
            details={
                "model": model_name,
                "high_entropy_threshold_nats": self.thresholds.high_entropy_threshold,
                "bootstrap_samples": self.thresholds.bootstrap_samples,
                "bootstrap_sample_size": min(
                    self.thresholds.bootstrap_sample_size, int(len(y_true))
                ),
            },
        )

    def _bootstrap_auc(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
        """Percentile bootstrap interval for ROC-AUC, seeded for reproducibility."""
        rng = np.random.default_rng(self.settings.seed)
        size = min(self.thresholds.bootstrap_sample_size, len(y_true))
        scores: list[float] = []

        for _ in range(self.thresholds.bootstrap_samples):
            idx = rng.integers(0, len(y_true), size=size)
            sample_true = y_true[idx]
            if len(np.unique(sample_true)) < 2:
                continue
            scores.append(float(roc_auc_score(sample_true, y_prob[idx])))

        if not scores:  # pragma: no cover - only if a class is absent everywhere
            return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"), "width": 0.0}

        lower = float(np.percentile(scores, 2.5))
        upper = float(np.percentile(scores, 97.5))
        return {
            "mean": float(np.mean(scores)),
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
        }
