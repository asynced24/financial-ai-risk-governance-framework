"""Model benchmark suite with an attached governance gate pipeline.

Protocol
--------
For every enabled model:

1. Stratified k-fold cross-validation on the **train** split (ranking stability).
2. A single fit on the full train split, scored once on a held-out **test** split
   that no fold ever saw.
3. Six metrics: ROC-AUC, PR-AUC, KS, Brier, log-loss and ECE. ECE is imported
   from ``governance.calibration`` rather than reimplemented here.
4. Five governance gates - fairness, drift, calibration, uncertainty and Bayesian
   segment stability - each returning pass / warn / fail against ``config.yaml``.

Class weights are deliberately left at their natural values. Rebalancing would
lift the headline discrimination metrics while destroying the calibration the
Brier and ECE gates exist to measure, and a probability of default that no longer
means "probability of default" is not usable for provisioning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ..bayes.segment_shrinkage import SegmentShrinkageModel, SegmentStabilityCheck
from ..config import STATUS_ORDER, Settings
from ..data.processor import CreditDataset
from ..governance.calibration import CalibrationAnalyzer, expected_calibration_error
from ..governance.drift import DriftDetector
from ..governance.fairness import FairnessAnalyzer
from ..governance.gates import GateResult, aggregate_status
from ..governance.uncertainty import UncertaintyAnalyzer


@dataclass
class BenchmarkResult:
    """Everything recorded about one model in one benchmark run."""

    model_name: str
    roc_auc: float
    pr_auc: float
    ks_statistic: float
    brier_score: float
    log_loss: float
    ece: float
    cv_metric: str
    cv_mean: float
    cv_std: float
    cv_scores: list[float]
    train_seconds: float
    predict_seconds: float
    n_train: int
    n_test: int
    n_features: int
    params: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, GateResult] = field(default_factory=dict)

    @property
    def gate_status(self) -> str:
        """Worst verdict across this model's governance gates."""
        return aggregate_status(self.gates) if self.gates else "pass"

    @property
    def failed_gates(self) -> list[str]:
        return [name for name, gate in self.gates.items() if gate.status == "fail"]

    @property
    def warned_gates(self) -> list[str]:
        return [name for name, gate in self.gates.items() if gate.status == "warn"]

    def metrics_dict(self) -> dict[str, float]:
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "ks_statistic": self.ks_statistic,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "ece": self.ece,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "metrics": self.metrics_dict(),
            "cross_validation": {
                "metric": self.cv_metric,
                "mean": self.cv_mean,
                "std": self.cv_std,
                "fold_scores": self.cv_scores,
            },
            "timings_seconds": {
                "train": self.train_seconds,
                "predict": self.predict_seconds,
            },
            "shape": {
                "n_train": self.n_train,
                "n_test": self.n_test,
                "n_features": self.n_features,
            },
            "params": self.params,
            "gate_status": self.gate_status,
            "gates": {name: gate.to_dict() for name, gate in self.gates.items()},
        }


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between defaulter and non-defaulter scores."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()

    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.0
    return float(ks_2samp(positives, negatives).statistic)


class ModelBenchmarkSuite:
    """Trains, scores and governance-gates every enabled model."""

    def __init__(self, settings: Settings, tracker=None):
        self.settings = settings
        self.tracker = tracker

        self.fairness = FairnessAnalyzer(settings, tracker)
        self.drift = DriftDetector(settings, tracker)
        self.calibration = CalibrationAnalyzer(settings, tracker)
        self.uncertainty = UncertaintyAnalyzer(settings, tracker)
        self.segment_stability = SegmentStabilityCheck(settings)

        self.models: dict[str, Any] = {}
        self.predictions: dict[str, np.ndarray] = {}
        self.results: list[BenchmarkResult] = []
        self.shrinkage: SegmentShrinkageModel | None = None
        self.drift_gate: GateResult | None = None

    # -------------------------------------------------------------- model zoo

    def build_models(self) -> dict[str, Any]:
        """Instantiate the enabled models with seeds fixed for reproducibility."""
        cfg = self.settings.models
        seed = self.settings.seed

        factories = {
            "logistic_regression": lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=cfg.logistic_max_iter,
                            random_state=seed,
                            n_jobs=None,
                        ),
                    ),
                ]
            ),
            "xgboost": lambda: XGBClassifier(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                subsample=cfg.subsample,
                colsample_bytree=cfg.colsample_bytree,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=seed,
                n_jobs=2,
            ),
            "lightgbm": lambda: LGBMClassifier(
                n_estimators=cfg.n_estimators,
                num_leaves=cfg.num_leaves,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                subsample=cfg.subsample,
                subsample_freq=1,
                colsample_bytree=cfg.colsample_bytree,
                random_state=seed,
                deterministic=True,
                force_row_wise=True,
                verbose=-1,
                n_jobs=2,
            ),
        }

        return {name: factories[name]() for name in cfg.enabled}

    # ------------------------------------------------------------------- run

    def run(self, dataset: CreditDataset) -> list[BenchmarkResult]:
        """Benchmark every enabled model and gate it. Returns one result per model."""
        self.shrinkage = SegmentShrinkageModel(self.settings).fit(
            dataset.audit_train["segment_id"], dataset.y_train
        )

        print("\n[gates] scoring train-vs-test feature drift (data level, shared across models)")
        self.drift_gate = self.drift.run(
            dataset.X_train, dataset.X_test, dataset.feature_names
        )

        results: list[BenchmarkResult] = []
        for name, model in self.build_models().items():
            results.append(self._benchmark_one(name, model, dataset))

        order = {"roc_auc": "roc_auc", "pr_auc": "pr_auc"}[self.settings.models.selection_metric]
        results.sort(key=lambda r: getattr(r, order), reverse=True)

        self.results = results
        return results

    def _benchmark_one(
        self,
        name: str,
        model: Any,
        dataset: CreditDataset,
    ) -> BenchmarkResult:
        print(f"\n[bench] {name}")
        cfg = self.settings.models

        cv = StratifiedKFold(
            n_splits=cfg.cv_folds, shuffle=True, random_state=self.settings.seed
        )
        cv_scoring = "roc_auc" if cfg.selection_metric == "roc_auc" else "average_precision"
        cv_scores = cross_val_score(
            model,
            dataset.X_train,
            dataset.y_train,
            cv=cv,
            scoring=cv_scoring,
            n_jobs=1,
        )
        print(
            f"[bench] {cfg.cv_folds}-fold CV {cv_scoring}: "
            f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}"
        )

        start = time.perf_counter()
        model.fit(dataset.X_train, dataset.y_train)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        y_prob = model.predict_proba(dataset.X_test)[:, 1]
        predict_seconds = time.perf_counter() - start

        y_test = dataset.y_test.to_numpy()
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "pr_auc": float(average_precision_score(y_test, y_prob)),
            "ks_statistic": ks_statistic(y_test, y_prob),
            "brier_score": float(np.mean((y_prob - y_test) ** 2)),
            "log_loss": float(log_loss(y_test, y_prob, labels=[0, 1])),
            "ece": expected_calibration_error(
                y_test, y_prob, n_bins=self.settings.governance.calibration.n_bins
            ),
        }
        print(
            f"[bench] test ROC-AUC {metrics['roc_auc']:.4f} | PR-AUC {metrics['pr_auc']:.4f} | "
            f"KS {metrics['ks_statistic']:.4f} | Brier {metrics['brier_score']:.4f} | "
            f"log-loss {metrics['log_loss']:.4f} | ECE {metrics['ece']:.4f}"
        )

        gates = self._run_gates(name, dataset, y_prob)

        self.models[name] = model
        self.predictions[name] = y_prob

        params = self._extract_params(model)
        result = BenchmarkResult(
            model_name=name,
            cv_metric=cv_scoring,
            cv_mean=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
            cv_scores=[float(s) for s in cv_scores],
            train_seconds=train_seconds,
            predict_seconds=predict_seconds,
            n_train=int(len(dataset.X_train)),
            n_test=int(len(dataset.X_test)),
            n_features=dataset.n_features,
            params=params,
            gates=gates,
            **metrics,
        )

        self._log_to_tracker(result)
        return result

    def _run_gates(
        self,
        name: str,
        dataset: CreditDataset,
        y_prob: np.ndarray,
    ) -> dict[str, GateResult]:
        """Run all five governance gates for one model's test-set predictions."""
        y_test = dataset.y_test.to_numpy()
        assert self.shrinkage is not None and self.drift_gate is not None

        gates: dict[str, GateResult] = {
            "fairness": self.fairness.run(y_test, y_prob, dataset.audit_test, name),
            "drift": self.drift_gate,
            "calibration": self.calibration.run(y_test, y_prob, name),
            "uncertainty": self.uncertainty.run(y_test, y_prob, name),
            "segment_stability": self.segment_stability.run(
                self.shrinkage, dataset.audit_test["segment_id"], y_prob, name
            ),
        }

        verdicts = " | ".join(
            f"{gate_name}:{gate.symbol}" for gate_name, gate in gates.items()
        )
        print(f"[gates] {verdicts}")
        return gates

    # ------------------------------------------------------------- reporting

    @staticmethod
    def _extract_params(model: Any) -> dict[str, Any]:
        """Serialisable subset of the estimator's hyper-parameters."""
        estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
        params = estimator.get_params()
        return {
            key: value
            for key, value in sorted(params.items())
            if isinstance(value, (int, float, str, bool, type(None)))
        }

    def _log_to_tracker(self, result: BenchmarkResult) -> None:
        if self.tracker is None:
            return

        name = result.model_name
        for key, value in result.metrics_dict().items():
            self.tracker.log_metric(f"{name}_{key}", value)
        self.tracker.log_metric(f"{name}_train_seconds", result.train_seconds)
        for key, value in result.params.items():
            self.tracker.log_param(f"{name}_{key}", value)
        for gate_name, gate in result.gates.items():
            self.tracker.log_param(f"{name}_gate_{gate_name}", gate.status)
            self.tracker.log_metric(f"{name}_gate_{gate_name}_value", gate.headline_value)

    @property
    def best_result(self) -> BenchmarkResult | None:
        """Top model on the configured selection metric (results are pre-sorted)."""
        return self.results[0] if self.results else None

    @property
    def best_model(self) -> Any | None:
        best = self.best_result
        return self.models.get(best.model_name) if best else None

    def leaderboard(self) -> pd.DataFrame:
        """Ranked comparison table across models."""
        rows = []
        for result in self.results:
            row = {"model": result.model_name, **result.metrics_dict()}
            row["gate_status"] = result.gate_status
            row.update({f"gate_{n}": g.status for n, g in result.gates.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def worst_gate_status(self) -> str:
        """Worst gate verdict across every benchmarked model."""
        statuses = [result.gate_status for result in self.results] or ["pass"]
        return max(statuses, key=lambda s: STATUS_ORDER[s])
