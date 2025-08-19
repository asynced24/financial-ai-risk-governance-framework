"""Configuration and constants for Financial AI Framework."""

import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
from pathlib import Path

# Reproducibility settings
SEED = 42
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# S&P Rating Mapping (22 classes)
SP_RATING_MAPPING = {
    'AAA': 0, 'AA+': 1, 'AA': 2, 'AA-': 3, 'A+': 4, 'A': 5, 'A-': 6,
    'BBB+': 7, 'BBB': 8, 'BBB-': 9, 'BB+': 10, 'BB': 11, 'BB-': 12,
    'B+': 13, 'B': 14, 'B-': 15, 'CCC+': 16, 'CCC': 17, 'CCC-': 18,
    'CC': 19, 'C': 20, 'D': 21
}

RATING_INVERSE = {v: k for k, v in SP_RATING_MAPPING.items()}
INVESTMENT_GRADE_THRESHOLD = 9  # BBB- and above

# Risk thresholds for governance
RISK_THRESHOLDS = {
    'accuracy_decline': 0.05,
    'fairness_violation': 0.1,
    'drift_pvalue': 0.05,
    'uncertainty_high': 0.7,
    'calibration_error': 0.1,
    'performance_variance': 0.05
}

# Advanced libraries flags
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

@dataclass
class ExperimentConfig:
    """Comprehensive experiment configuration."""
    # Data configuration
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = SEED
    
    # Model training
    cv_folds: int = 5
    models_to_train: List[str] = None
    enable_ensemble: bool = True
    
    # Risk governance
    enable_uncertainty: bool = True
    enable_fairness: bool = True
    enable_drift_detection: bool = True
    enable_calibration: bool = True
    
    # Explainability
    enable_shap: bool = True
    enable_feature_importance: bool = True
    shap_sample_size: int = 100
    
    # Reproducibility & tracking
    enable_mlflow: bool = False
    mlflow_experiment_name: str = "credit_rating_governance"
    track_artifacts: bool = True
    
    # Output configuration
    create_executive_report: bool = True
    create_detailed_plots: bool = True
    save_models: bool = True
    
    def __post_init__(self):
        if self.models_to_train is None:
            self.models_to_train = ['logreg', 'rf', 'xgb', 'lgbm'] if HAS_XGBOOST and HAS_LIGHTGBM else ['logreg', 'rf']

@dataclass
class BenchmarkResults:
    """Structured benchmark results."""
    model_name: str
    test_accuracy: float
    test_f1_macro: float
    investment_grade_acc: float
    within_1_notches: float
    cv_mean: float
    cv_std: float
    training_time: float
    prediction_time: float
    model_size_mb: float

@dataclass
class GovernanceReport:
    """Comprehensive governance assessment results."""
    # Performance metrics
    model_performance: Dict[str, BenchmarkResults]
    best_model_name: str
    
    # Risk assessments
    uncertainty_analysis: Dict[str, Any]
    fairness_metrics: Dict[str, Any]
    drift_detection: Dict[str, Any]
    calibration_analysis: Dict[str, Any]
    
    # Explainability
    feature_importance: Dict[str, float]
    shap_analysis: Dict[str, Any]
    
    # Governance summary
    risk_alerts: List[str]
    compliance_score: float
    governance_recommendation: str
    
    # Metadata
    run_id: str
    timestamp: str
    data_hash: str