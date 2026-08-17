"""Calibration gate for probability-of-default models.

A PD model that ranks well but is miscalibrated is unusable for provisioning: the
number itself has to mean what it says. This module is the single source of truth
for Expected Calibration Error in the framework - the benchmark suite calls
:func:`expected_calibration_error` rather than computing its own.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss

from ..config import Settings
from .gates import GateResult, worst_status


def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> dict[str, list[float]]:
    """Equal-width reliability bins over the predicted probability of default.

    Returns the per-bin mean predicted probability, observed default rate, and
    share of the population, keeping only non-empty bins.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize puts values exactly at 0.0 in bin 0; clamp the top edge in too.
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    mean_predicted: list[float] = []
    observed_rate: list[float] = []
    weight: list[float] = []
    counts: list[float] = []

    total = len(y_prob)
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_predicted.append(float(y_prob[mask].mean()))
        observed_rate.append(float(y_true[mask].mean()))
        weight.append(count / total)
        counts.append(count)

    return {
        "mean_predicted": mean_predicted,
        "observed_rate": observed_rate,
        "weight": weight,
        "count": counts,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Population-weighted mean gap between predicted and observed default rate."""
    bins = calibration_bins(y_true, y_prob, n_bins=n_bins)
    if not bins["weight"]:
        return 0.0
    return float(
        sum(
            w * abs(obs - pred)
            for w, obs, pred in zip(bins["weight"], bins["observed_rate"], bins["mean_predicted"], strict=True)
        )
    )


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Worst single-bin gap between predicted and observed default rate."""
    bins = calibration_bins(y_true, y_prob, n_bins=n_bins)
    if not bins["weight"]:
        return 0.0
    return float(
        max(abs(obs - pred) for obs, pred in zip(bins["observed_rate"], bins["mean_predicted"], strict=True))
    )


class CalibrationAnalyzer:
    """Governance gate on ECE and Brier score."""

    def __init__(self, settings: Settings, tracker=None):
        self.settings = settings
        self.tracker = tracker
        self.thresholds = settings.governance.calibration

    def run(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "model",
    ) -> GateResult:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_prob = np.asarray(y_prob, dtype=float).ravel()
        n_bins = self.thresholds.n_bins

        ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
        mce = maximum_calibration_error(y_true, y_prob, n_bins=n_bins)
        brier = float(brier_score_loss(y_true, y_prob))
        bins = calibration_bins(y_true, y_prob, n_bins=n_bins)

        base_rate = float(y_true.mean())
        # A constant predictor at the base rate scores base*(1-base); anything
        # above that is worse than predicting the average for everyone.
        reference_brier = base_rate * (1.0 - base_rate)

        mean_predicted = float(y_prob.mean())
        bias = mean_predicted - base_rate

        ece_status = self.thresholds.ece.classify(ece)
        brier_status = self.thresholds.brier.classify(brier)
        status = worst_status([ece_status, brier_status])

        findings: list[str] = []
        if ece_status != "pass":
            findings.append(
                f"ECE {ece:.4f} breaches the {ece_status} limit "
                f"({self.thresholds.ece.describe()}) over {n_bins} reliability bins"
            )
        if brier_status != "pass":
            findings.append(
                f"Brier score {brier:.4f} breaches the {brier_status} limit "
                f"({self.thresholds.brier.describe()}); a constant base-rate "
                f"predictor scores {reference_brier:.4f}"
            )
        if abs(bias) > 0.02:
            findings.append(
                f"Mean predicted PD {mean_predicted:.4f} is off the observed base rate "
                f"{base_rate:.4f} by {bias:+.4f}"
            )

        metrics: dict[str, Any] = {
            "ece": ece,
            "mce": mce,
            "brier_score": brier,
            "reference_brier_base_rate": reference_brier,
            "mean_predicted_pd": mean_predicted,
            "observed_default_rate": base_rate,
            "calibration_bias": bias,
            "n_bins_populated": len(bins["weight"]),
        }

        if self.tracker is not None:
            for key in ("ece", "mce", "brier_score", "calibration_bias"):
                self.tracker.log_metric(f"{model_name}_calibration_{key}", metrics[key])

        return GateResult(
            name="calibration",
            status=status,
            headline_metric="ece",
            headline_value=ece,
            threshold=self.thresholds.ece.describe(),
            metrics=metrics,
            findings=findings,
            details={
                "model": model_name,
                "ece_status": ece_status,
                "brier_status": brier_status,
                "brier_threshold": self.thresholds.brier.describe(),
                "reliability_bins": bins,
            },
        )
