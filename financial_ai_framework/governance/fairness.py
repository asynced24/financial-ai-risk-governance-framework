"""Fairness gate: demographic parity and equalized odds on protected attributes.

The attributes audited here (``sex``, ``age_band``) are deliberately withheld from
the feature matrix - see ``data/processor.py``. Measuring a gap anyway is the
point: correlated repayment-behaviour features can reintroduce disparate impact
that excluding the attribute does nothing to prevent.

Definitions used
----------------
demographic parity gap
    ``max - min`` of the predicted default rate across groups. Distribution of
    the adverse decision, ignoring whether it was correct.
equalized odds gap
    The larger of the ``max - min`` true-positive-rate spread and the
    ``max - min`` false-positive-rate spread. Error rates conditioned on the
    borrower's actual outcome, which is the parity that matters when the adverse
    decision is "declined credit".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..config import Settings
from .gates import GateResult, worst_status


def _rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """True-positive and false-positive rate, NaN when the stratum is empty."""
    positives = y_true == 1
    negatives = y_true == 0
    tpr = float(y_pred[positives].mean()) if positives.any() else float("nan")
    fpr = float(y_pred[negatives].mean()) if negatives.any() else float("nan")
    return tpr, fpr


def _spread(values: list[float]) -> float:
    """max - min over the finite values, or 0.0 when fewer than two exist."""
    finite = [v for v in values if np.isfinite(v)]
    if len(finite) < 2:
        return 0.0
    return float(max(finite) - min(finite))


class FairnessAnalyzer:
    """Governance gate on demographic parity and equalized odds."""

    def __init__(self, settings: Settings, tracker=None):
        self.settings = settings
        self.tracker = tracker
        self.thresholds = settings.governance.fairness

    def run(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        audit: pd.DataFrame,
        model_name: str = "model",
    ) -> GateResult:
        """Measure parity gaps for every configured protected attribute."""
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_prob = np.asarray(y_prob, dtype=float).ravel()
        y_pred = (y_prob >= self.settings.models.decision_threshold).astype(int)

        attributes = self.thresholds.protected_attributes
        min_group = self.thresholds.min_group_size

        per_attribute: dict[str, Any] = {}
        # (label, observed value, threshold spec, gate status)
        scored: list[tuple[str, float, str, str]] = []
        findings: list[str] = []
        skipped: list[str] = []

        for attribute in attributes:
            if attribute not in audit.columns:
                skipped.append(f"{attribute} (column absent)")
                continue

            values = audit[attribute].astype(str).to_numpy()
            groups: dict[str, dict[str, Any]] = {}

            for group in sorted(pd.unique(values)):
                mask = values == group
                size = int(mask.sum())
                if size < min_group:
                    continue

                yt, yp, prob = y_true[mask], y_pred[mask], y_prob[mask]
                tpr, fpr = _rates(yt, yp)
                try:
                    group_auc = float(roc_auc_score(yt, prob)) if len(np.unique(yt)) > 1 else float("nan")
                except ValueError:  # pragma: no cover - degenerate stratum
                    group_auc = float("nan")

                groups[group] = {
                    "n": size,
                    "predicted_default_rate": float(yp.mean()),
                    "observed_default_rate": float(yt.mean()),
                    "mean_predicted_pd": float(prob.mean()),
                    "true_positive_rate": tpr,
                    "false_positive_rate": fpr,
                    "roc_auc": group_auc,
                }

            if len(groups) < 2:
                skipped.append(f"{attribute} (fewer than 2 groups of >= {min_group} rows)")
                continue

            dp_gap = _spread([g["predicted_default_rate"] for g in groups.values()])
            tpr_gap = _spread([g["true_positive_rate"] for g in groups.values()])
            fpr_gap = _spread([g["false_positive_rate"] for g in groups.values()])
            eo_gap = max(tpr_gap, fpr_gap)

            dp_status = self.thresholds.demographic_parity.classify(dp_gap)
            eo_status = self.thresholds.equalized_odds.classify(eo_gap)

            per_attribute[attribute] = {
                "groups": groups,
                "n_groups": len(groups),
                "demographic_parity_gap": dp_gap,
                "true_positive_rate_gap": tpr_gap,
                "false_positive_rate_gap": fpr_gap,
                "equalized_odds_gap": eo_gap,
                "demographic_parity_status": dp_status,
                "equalized_odds_status": eo_status,
            }

            scored.append(
                (
                    f"{attribute} demographic parity gap",
                    dp_gap,
                    self.thresholds.demographic_parity.describe(),
                    dp_status,
                )
            )
            scored.append(
                (
                    f"{attribute} equalized odds gap",
                    eo_gap,
                    self.thresholds.equalized_odds.describe(),
                    eo_status,
                )
            )

            if dp_status != "pass":
                findings.append(
                    f"{attribute}: demographic parity gap {dp_gap:.4f} ({dp_status}, "
                    f"{self.thresholds.demographic_parity.describe()}) across "
                    f"{len(groups)} groups"
                )
            if eo_status != "pass":
                driver = "TPR" if tpr_gap >= fpr_gap else "FPR"
                findings.append(
                    f"{attribute}: equalized odds gap {eo_gap:.4f} ({eo_status}, "
                    f"{self.thresholds.equalized_odds.describe()}), driven by the "
                    f"{driver} spread (TPR {tpr_gap:.4f} / FPR {fpr_gap:.4f})"
                )

            if self.tracker is not None:
                self.tracker.log_metric(f"{model_name}_fairness_{attribute}_dp_gap", dp_gap)
                self.tracker.log_metric(f"{model_name}_fairness_{attribute}_eo_gap", eo_gap)

        if not scored:
            return GateResult(
                name="fairness",
                status="warn",
                headline_metric="worst_parity_gap",
                headline_value=float("nan"),
                threshold=self.thresholds.demographic_parity.describe(),
                metrics={"attributes_analysed": 0},
                findings=[
                    "No protected attribute could be audited: " + "; ".join(skipped)
                ],
                details={"model": model_name, "skipped": skipped},
            )

        status = worst_status(s for _, _, _, s in scored)
        # Headline = the largest gap among those at the worst observed status.
        driving = max((s for s in scored if s[3] == status), key=lambda s: s[1])

        return GateResult(
            name="fairness",
            status=status,
            headline_metric=driving[0].replace(" ", "_"),
            headline_value=driving[1],
            threshold=driving[2],
            metrics={
                "attributes_analysed": len(per_attribute),
                "worst_gap": driving[1],
                "worst_gap_metric": driving[0],
                **{
                    f"{attr}_{key}": data[key]
                    for attr, data in per_attribute.items()
                    for key in ("demographic_parity_gap", "equalized_odds_gap")
                },
            },
            findings=findings,
            details={
                "model": model_name,
                "decision_threshold": self.settings.models.decision_threshold,
                "min_group_size": min_group,
                "withheld_from_features": True,
                "skipped": skipped,
                "attributes": per_attribute,
            },
        )
