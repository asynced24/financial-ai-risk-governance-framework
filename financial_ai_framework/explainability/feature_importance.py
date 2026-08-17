"""Native feature importance, aggregated across the benchmarked models.

Cheap counterpart to SHAP: each fitted model already exposes either tree split
gains or linear coefficients. Ranking features by the *agreement* across model
families is a useful sanity check - a driver that only one model family cares
about is usually an artefact rather than a signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from ..config import Settings


def native_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    """Extract per-feature importance from a fitted estimator or pipeline.

    Tree models report split importance directly; linear models report absolute
    coefficient magnitude. Returns an empty dict for models that expose neither.
    """
    estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return {}

    if len(values) != len(feature_names):
        return {}

    total = values.sum()
    if total > 0:
        values = values / total  # comparable scale across families

    return {name: float(value) for name, value in zip(feature_names, values, strict=True)}


class FeatureImportanceAnalyzer:
    """Aggregates native importances across every benchmarked model."""

    def __init__(self, settings: Settings, tracker=None):
        self.settings = settings
        self.tracker = tracker

    def analyse(
        self,
        models: dict[str, Any],
        feature_names: list[str],
    ) -> dict[str, Any]:
        per_model: dict[str, dict[str, float]] = {}
        for name, model in models.items():
            importance = native_importance(model, feature_names)
            if importance:
                per_model[name] = importance

        if not per_model:
            return {"status": "no_importances_available"}

        stats: dict[str, dict[str, float]] = {}
        for feature in feature_names:
            values = [importance.get(feature, 0.0) for importance in per_model.values()]
            stats[feature] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "models_counted": len(values),
            }

        ranked = sorted(stats.items(), key=lambda kv: kv[1]["mean"], reverse=True)
        top_n = self.settings.explainability.top_features

        return {
            "status": "success",
            "per_model": per_model,
            "aggregated": dict(ranked),
            "top_features": {name: data["mean"] for name, data in ranked[:top_n]},
            "consensus_note": (
                "Importances are normalised to sum to 1.0 per model so tree split "
                "gains and linear coefficient magnitudes are comparable in rank."
            ),
        }
