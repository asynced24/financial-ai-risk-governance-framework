"""Experiment tracking and reproducibility utilities.

``ExperimentTracker`` wraps MLflow with a local-JSON fallback so a run is always
recorded somewhere: with MLflow enabled the params, metrics and artefacts land in
the tracking store, and without it they land in
``reports/experiment_log_<run>.json``. Every module logs through this one object,
so nothing has to care which backend is live.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from ..config import HAS_MLFLOW, SEED, Settings

if HAS_MLFLOW:  # pragma: no cover - depends on the installed environment
    import mlflow


def ensure_reproducibility(seed: int = SEED) -> None:
    """Pin every seed the framework depends on."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[repro] seeds pinned to {seed}")


class ExperimentTracker:
    """MLflow experiment tracking with a local JSON fallback."""

    def __init__(self, settings: Settings, artifacts_dir: Path | None = None):
        self.settings = settings
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else settings.reports_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.run_name: str | None = None
        self.run_active = False
        # Always present: it doubles as the fallback sink and the in-memory record.
        self.local_log: dict[str, Any] = {"parameters": {}, "metrics": {}, "artifacts": []}

        self.use_mlflow = bool(settings.tracking.enable_mlflow and HAS_MLFLOW)
        if settings.tracking.enable_mlflow and not HAS_MLFLOW:
            print("[track] mlflow not installed - falling back to local JSON logging")

        if self.use_mlflow:
            self._setup_mlflow()
        else:
            print("[track] local JSON experiment logging enabled")

    def _setup_mlflow(self) -> None:  # pragma: no cover - requires a tracking store
        try:
            mlflow.set_experiment(self.settings.tracking.experiment_name)
            print(f"[track] mlflow tracking enabled: {self.settings.tracking.experiment_name}")
        except Exception as exc:
            print(f"[track] mlflow setup failed ({exc}) - falling back to local JSON logging")
            self.use_mlflow = False

    @property
    def _mlflow_live(self) -> bool:
        return self.use_mlflow and self.run_active

    def start_run(self, run_name: str | None = None) -> None:
        """Open a run. Safe to call when MLflow is unavailable."""
        self.run_name = run_name
        self.local_log["run_name"] = run_name
        if self.use_mlflow:  # pragma: no cover - requires a tracking store
            mlflow.start_run(run_name=run_name)
        self.run_active = True

    def log_param(self, key: str, value: Any) -> None:
        if self._mlflow_live:  # pragma: no cover
            mlflow.log_param(key, value)
        self.local_log["parameters"][key] = value

    def log_params(self, params: dict[str, Any]) -> None:
        for key, value in params.items():
            self.log_param(key, value)

    def log_metric(self, key: str, value: float) -> None:
        if value is None or not np.isfinite(value):
            return
        if self._mlflow_live:  # pragma: no cover
            mlflow.log_metric(key, float(value))
        self.local_log["metrics"][key] = float(value)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.log_metric(key, value)

    def log_artifact(self, path: str | Path) -> None:
        path = str(path)
        if self._mlflow_live:  # pragma: no cover
            mlflow.log_artifact(path)
        if path not in self.local_log["artifacts"]:
            self.local_log["artifacts"].append(path)

    def end_run(self) -> Path:
        """Close the run and always persist the local log. Returns its path."""
        if self._mlflow_live:  # pragma: no cover
            mlflow.end_run()
        self.run_active = False

        suffix = f"_{self.run_name}" if self.run_name else ""
        log_path = self.artifacts_dir / f"experiment_log{suffix}.json"
        log_path.write_text(
            json.dumps(self.local_log, indent=2, default=str), encoding="utf-8"
        )
        print(f"[track] run log written -> {log_path}")
        return log_path
