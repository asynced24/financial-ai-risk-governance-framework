"""SHAP explanations for the selected probability-of-default model.

Produces two artefacts a model-risk reviewer actually asks for:

* **Global attribution** - mean absolute SHAP value per feature, i.e. how much
  each feature moves the predicted PD on average across the evaluation sample.
* **Local explanations** - a full per-feature contribution breakdown for a handful
  of individual borrowers, written to ``reports/`` as JSON so they can be pasted
  into an adverse-action review.

Explainer selection is by model family: exact tree SHAP for the gradient-boosted
models, exact linear SHAP for the scaled logistic-regression pipeline. Neither
path uses sampling approximation, so the numbers are reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from ..config import Settings


def _split_pipeline(model: Any) -> tuple[Pipeline | None, Any]:
    """Return (preprocessing pipeline or None, final estimator)."""
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
        preprocessing = Pipeline(model.steps[:-1]) if len(model.steps) > 1 else None
        return preprocessing, estimator
    return None, model


def _normalise_shap(values: Any, n_features: int) -> np.ndarray:
    """Coerce any SHAP output shape into a 2-D (n_samples, n_features) array.

    Binary classifiers variously return one array, a stacked
    ``(samples, features, classes)`` array, or a per-class list; the positive
    class is the one that matters for a PD model.
    """
    if isinstance(values, list):
        values = values[-1]

    array = np.asarray(getattr(values, "values", values), dtype=float)

    if array.ndim == 3:
        # (samples, features, classes) -> positive class
        array = array[:, :, -1]
    elif array.ndim == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2 or array.shape[1] != n_features:
        raise ValueError(
            f"Unexpected SHAP value shape {array.shape} for {n_features} features"
        )
    return array


class ShapAnalyzer:
    """Computes and persists global and local SHAP attributions."""

    def __init__(self, settings: Settings, tracker=None):
        self.settings = settings
        self.tracker = tracker
        self.config = settings.explainability

    def analyse(
        self,
        model: Any,
        model_name: str,
        X_background: pd.DataFrame,
        X_sample: pd.DataFrame,
        y_prob: np.ndarray,
        reports_dir: Path,
    ) -> dict[str, Any]:
        """Run SHAP on ``model`` and write artefacts into ``reports_dir``."""
        if not self.config.enable_shap:
            return {"status": "disabled"}

        feature_names = list(X_sample.columns)
        n_sample = min(self.config.shap_sample_size, len(X_sample))
        n_background = min(self.config.shap_background_size, len(X_background))

        X_eval = X_sample.iloc[:n_sample]
        X_ref = X_background.iloc[:n_background]

        print(f"[shap] explaining {model_name} on {n_sample} rows ({n_background} background rows)")

        try:
            shap_values, base_value, explainer_kind = self._compute(model, X_ref, X_eval)
        except Exception as exc:  # pragma: no cover - depends on shap internals
            print(f"[shap] explanation failed: {exc}")
            return {"status": "failed", "error": str(exc), "model": model_name}

        mean_abs = np.abs(shap_values).mean(axis=0)
        ranked = sorted(zip(feature_names, mean_abs, strict=True), key=lambda kv: kv[1], reverse=True)
        global_importance = {name: float(value) for name, value in ranked}

        reports_dir.mkdir(parents=True, exist_ok=True)
        artefacts: list[str] = []

        global_path = reports_dir / f"shap_global_importance_{model_name}.json"
        global_path.write_text(
            json.dumps(
                {
                    "model": model_name,
                    "explainer": explainer_kind,
                    "sample_size": int(n_sample),
                    "metric": "mean_absolute_shap_value",
                    "units": "probability_of_default",
                    "global_importance": global_importance,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        artefacts.append(str(global_path))

        local_path = self._write_local_explanations(
            shap_values=shap_values,
            base_value=base_value,
            X_eval=X_eval,
            y_prob=np.asarray(y_prob, dtype=float)[:n_sample],
            feature_names=feature_names,
            model_name=model_name,
            explainer_kind=explainer_kind,
            reports_dir=reports_dir,
        )
        if local_path:
            artefacts.append(str(local_path))

        plot_path = self._write_importance_plot(ranked, model_name, reports_dir)
        if plot_path:
            artefacts.append(str(plot_path))

        if self.tracker is not None:
            for rank, (name, value) in enumerate(ranked[: self.config.top_features], start=1):
                self.tracker.log_metric(f"{model_name}_shap_rank{rank:02d}_{name}", float(value))
            for path in artefacts:
                self.tracker.log_artifact(path)

        top = ranked[: self.config.top_features]
        print(
            "[shap] top drivers: "
            + ", ".join(f"{name} ({value:.4f})" for name, value in top[:5])
        )

        return {
            "status": "success",
            "model": model_name,
            "explainer": explainer_kind,
            "sample_size": int(n_sample),
            "base_value": float(base_value),
            "global_importance": global_importance,
            "top_features": dict(top),
            "artifacts": artefacts,
        }

    # ---------------------------------------------------------------- internals

    def _compute(
        self,
        model: Any,
        X_ref: pd.DataFrame,
        X_eval: pd.DataFrame,
    ) -> tuple[np.ndarray, float, str]:
        """Pick the exact explainer for the model family and evaluate it."""
        preprocessing, estimator = _split_pipeline(model)
        n_features = X_eval.shape[1]

        if hasattr(estimator, "coef_"):
            ref = preprocessing.transform(X_ref) if preprocessing is not None else X_ref.to_numpy()
            evaluated = (
                preprocessing.transform(X_eval) if preprocessing is not None else X_eval.to_numpy()
            )
            explainer = shap.LinearExplainer(estimator, np.asarray(ref, dtype=float))
            values = explainer(np.asarray(evaluated, dtype=float))
            base = float(np.mean(np.asarray(values.base_values, dtype=float)))
            return _normalise_shap(values, n_features), base, "LinearExplainer(log-odds)"

        explainer = shap.TreeExplainer(estimator)
        values = explainer(X_eval)
        base_values = np.asarray(getattr(values, "base_values", 0.0), dtype=float)
        base = float(base_values.mean()) if base_values.size else 0.0
        return _normalise_shap(values, n_features), base, "TreeExplainer(exact)"

    def _write_local_explanations(
        self,
        shap_values: np.ndarray,
        base_value: float,
        X_eval: pd.DataFrame,
        y_prob: np.ndarray,
        feature_names: list[str],
        model_name: str,
        explainer_kind: str,
        reports_dir: Path,
    ) -> Path | None:
        """Write per-borrower contribution breakdowns for a few sample rows."""
        n_local = min(self.config.local_explanations, len(X_eval))
        if n_local <= 0:
            return None

        # Pick a spread: the highest-risk, the lowest-risk, and the most uncertain
        # borrower in the sample, so the artefacts are not three near-identical rows.
        order = np.argsort(y_prob)
        candidates = [int(order[-1]), int(order[0])]
        candidates.append(int(np.argmin(np.abs(y_prob - 0.5))))
        selected: list[int] = []
        for idx in candidates:
            if idx not in selected:
                selected.append(idx)
        for idx in range(len(X_eval)):
            if len(selected) >= n_local:
                break
            if idx not in selected:
                selected.append(idx)
        selected = selected[:n_local]

        explanations: list[dict[str, Any]] = []
        for rank, idx in enumerate(selected, start=1):
            contributions = {
                name: float(shap_values[idx, col])
                for col, name in enumerate(feature_names)
            }
            top = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[
                : self.config.top_features
            ]
            explanations.append(
                {
                    "explanation_id": rank,
                    "row_index_in_sample": idx,
                    "predicted_pd": float(y_prob[idx]),
                    "base_value": float(base_value),
                    "sum_of_contributions": float(shap_values[idx].sum()),
                    "top_contributions": [
                        {
                            "feature": name,
                            "shap_value": value,
                            "feature_value": float(X_eval.iloc[idx][name]),
                            "direction": "increases_pd" if value > 0 else "decreases_pd",
                        }
                        for name, value in top
                    ],
                    "all_contributions": contributions,
                }
            )

        path = reports_dir / f"shap_local_explanations_{model_name}.json"
        path.write_text(
            json.dumps(
                {
                    "model": model_name,
                    "explainer": explainer_kind,
                    "note": (
                        "Local SHAP contributions for individual borrowers. Values are "
                        "additive on the explainer's output scale and sum with the base "
                        "value to that borrower's score."
                    ),
                    "explanations": explanations,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[shap] wrote {len(explanations)} local explanations -> {path}")
        return path

    def _write_importance_plot(
        self,
        ranked: list[tuple[str, float]],
        model_name: str,
        reports_dir: Path,
    ) -> Path | None:
        """Horizontal bar chart of the top global attributions."""
        try:  # pragma: no cover - plotting is a convenience, never load-bearing
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            top = ranked[: self.config.top_features][::-1]
            names = [name for name, _ in top]
            values = [value for _, value in top]

            fig, ax = plt.subplots(figsize=(8, max(3.0, 0.32 * len(top))))
            ax.barh(names, values, color="#31708e")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"Global feature attribution - {model_name}")
            fig.tight_layout()

            path = reports_dir / f"shap_global_importance_{model_name}.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            return path
        except Exception as exc:  # pragma: no cover
            print(f"[shap] importance plot skipped: {exc}")
            return None
