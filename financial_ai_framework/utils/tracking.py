"""Experiment tracking and reproducibility utilities."""

import json
import random
import os
from pathlib import Path
from typing import Any
from ..config.settings import ExperimentConfig, SEED, HAS_MLFLOW

if HAS_MLFLOW:
    import mlflow
    import mlflow.sklearn


def ensure_reproducibility():
    """Ensure complete reproducibility across all components."""
    # Python random seed
    random.seed(SEED)
    import numpy as np
    np.random.seed(SEED)
    
    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['SKLEARN_SEED'] = str(SEED)
    
    print(f"🔒 Reproducibility ensured with seed: {SEED}")


class ExperimentTracker:
    """MLflow experiment tracking with fallback logging."""
    
    def __init__(self, config: ExperimentConfig, artifacts_dir: Path):
        self.config = config
        self.artifacts_dir = artifacts_dir
        self.run_active = False
        
        if config.enable_mlflow and HAS_MLFLOW:
            self._setup_mlflow()
        else:
            self._setup_local_logging()
    
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        try:
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            print(f"📊 MLflow tracking enabled: {self.config.mlflow_experiment_name}")
        except Exception as e:
            print(f"⚠️ MLflow setup failed: {e}. Falling back to local logging.")
            self._setup_local_logging()
    
    def _setup_local_logging(self):
        """Setup local logging as fallback."""
        self.local_log = {"parameters": {}, "metrics": {}, "artifacts": []}
        print("📝 Local experiment logging enabled")
    
    def start_run(self, run_name: str = None):
        """Start experiment run."""
        if self.config.enable_mlflow and HAS_MLFLOW:
            mlflow.start_run(run_name=run_name)
            self.run_active = True
    
    def log_param(self, key: str, value: Any):
        """Log parameter."""
        if self.config.enable_mlflow and HAS_MLFLOW and self.run_active:
            mlflow.log_param(key, value)
        else:
            self.local_log["parameters"][key] = value
    
    def log_metric(self, key: str, value: float):
        """Log metric."""
        if self.config.enable_mlflow and HAS_MLFLOW and self.run_active:
            mlflow.log_metric(key, value)
        else:
            self.local_log["metrics"][key] = value
    
    def log_artifact(self, path: str):
        """Log artifact."""
        if self.config.enable_mlflow and HAS_MLFLOW and self.run_active:
            mlflow.log_artifact(path)
        else:
            self.local_log["artifacts"].append(path)
    
    def end_run(self):
        """End experiment run."""
        if self.config.enable_mlflow and HAS_MLFLOW and self.run_active:
            mlflow.end_run()
            self.run_active = False
        else:
            # Save local log
            with open(self.artifacts_dir / "experiment_log.json", "w") as f:
                json.dump(self.local_log, f, indent=2, default=str)