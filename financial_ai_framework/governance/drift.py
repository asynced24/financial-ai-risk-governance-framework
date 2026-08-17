"""Drift gate: Population Stability Index between the train and test splits.

PSI is the standard measure in credit risk for "has this feature's distribution
moved". Reference (train) quantile bins are held fixed and the current (test)
population is scored into them:

    PSI = sum over bins of  (current_share - reference_share) * ln(current_share / reference_share)

Conventional reading: below 0.10 stable, 0.10-0.25 moderate shift, above 0.25
significant shift. Those cut-offs live in ``config.yaml``, not here.

Drift is a property of the data, not of any one model, so the suite computes this
gate once per run and attaches the same verdict to every model's scorecard.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..config import Settings
from .gates import GateResult, worst_status

_EPSILON = 1e-6


def population_stability_index(
    reference: Sequence[float],
    current: Sequence[float],
    bins: int = 10,
) -> float:
    """PSI of ``current`` against ``reference`` using reference quantile bins.

    Constant reference features return 0.0 - there is no distribution to shift.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        # Degenerate / near-constant feature: fall back to a share comparison of
        # the single distinct reference value.
        value = ref[0]
        ref_share = float(np.mean(ref == value))
        cur_share = float(np.mean(cur == value))
        if min(ref_share, cur_share) <= 0 or abs(ref_share - cur_share) < _EPSILON:
            return 0.0
        return float((cur_share - ref_share) * np.log(cur_share / ref_share))

    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_share = np.maximum(ref_counts / len(ref), _EPSILON)
    cur_share = np.maximum(cur_counts / len(cur), _EPSILON)

    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


class DriftDetector:
    """Governance gate on train-vs-test feature drift."""

    def __init__(self, settings: Settings, tracker=None):
        self.settings = settings
        self.tracker = tracker
        self.thresholds = settings.governance.drift

    def run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: Sequence[str] | None = None,
    ) -> GateResult:
        """Score every feature deterministically - no sampling, no shortcuts."""
        features = list(feature_names) if feature_names is not None else list(X_train.columns)
        bins = self.thresholds.psi_bins

        per_feature: dict[str, dict[str, float]] = {}
        for name in features:
            if name not in X_train.columns or name not in X_test.columns:
                continue

            ref = X_train[name].to_numpy(dtype=float)
            cur = X_test[name].to_numpy(dtype=float)

            psi = population_stability_index(ref, cur, bins=bins)
            try:
                ks_stat, ks_p = ks_2samp(ref, cur)
            except ValueError:  # pragma: no cover - degenerate feature
                ks_stat, ks_p = 0.0, 1.0

            per_feature[name] = {
                "psi": psi,
                "psi_status": self.thresholds.feature_psi.classify(psi),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_p),
                "reference_mean": float(np.mean(ref)),
                "current_mean": float(np.mean(cur)),
            }

        if not per_feature:
            return GateResult(
                name="drift",
                status="warn",
                headline_metric="max_feature_psi",
                headline_value=float("nan"),
                threshold=self.thresholds.feature_psi.describe(),
                metrics={"features_tested": 0},
                findings=["No overlapping numeric features to test for drift"],
                details={},
            )

        psis = {name: data["psi"] for name, data in per_feature.items()}
        drifted = [name for name, data in per_feature.items() if data["psi_status"] != "pass"]
        drifted_ratio = len(drifted) / len(per_feature)

        max_name = max(psis, key=psis.get)
        max_psi = psis[max_name]

        feature_status = self.thresholds.feature_psi.classify(max_psi)
        ratio_status = self.thresholds.drifted_feature_ratio.classify(drifted_ratio)
        status = worst_status([feature_status, ratio_status])

        findings: list[str] = []
        for name in sorted(drifted, key=lambda n: psis[n], reverse=True):
            findings.append(
                f"{name}: PSI {psis[name]:.4f} ({per_feature[name]['psi_status']}, "
                f"{self.thresholds.feature_psi.describe()})"
            )
        if ratio_status != "pass":
            findings.append(
                f"{len(drifted)}/{len(per_feature)} features drifted "
                f"({drifted_ratio:.1%}), breaching the {ratio_status} limit "
                f"({self.thresholds.drifted_feature_ratio.describe()})"
            )

        metrics: dict[str, Any] = {
            "features_tested": len(per_feature),
            "max_feature_psi": max_psi,
            "max_psi_feature": max_name,
            "mean_feature_psi": float(np.mean(list(psis.values()))),
            "median_feature_psi": float(np.median(list(psis.values()))),
            "drifted_features": len(drifted),
            "drifted_feature_ratio": drifted_ratio,
        }

        if self.tracker is not None:
            for key in ("max_feature_psi", "mean_feature_psi", "drifted_feature_ratio"):
                self.tracker.log_metric(f"drift_{key}", metrics[key])

        return GateResult(
            name="drift",
            status=status,
            headline_metric="max_feature_psi",
            headline_value=max_psi,
            threshold=self.thresholds.feature_psi.describe(),
            metrics=metrics,
            findings=findings,
            details={
                "scope": "data_level_shared_across_models",
                "psi_bins": bins,
                "feature_psi_threshold": self.thresholds.feature_psi.describe(),
                "drifted_ratio_threshold": self.thresholds.drifted_feature_ratio.describe(),
                "ratio_status": ratio_status,
                "features": per_feature,
            },
        )
