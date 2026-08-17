"""Financial AI Risk & Governance Framework.

A benchmark-plus-governance pipeline for credit-default (probability of default)
models: train several model families under one protocol, put each of them through
five governance gates whose thresholds are declared in ``config.yaml``, explain
the winner with SHAP, and emit a governance scorecard and per-model model cards.

Public surface::

    from financial_ai_framework import load_settings, load_credit_data, \\
        CreditDataProcessor, ModelBenchmarkSuite, GovernanceReporter
"""

from .bayes.segment_shrinkage import (
    SegmentPosterior,
    SegmentShrinkageModel,
    SegmentStabilityCheck,
    ShrinkagePrior,
)
from .config import (
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    SEED,
    GateThreshold,
    Settings,
    load_settings,
)
from .data.loader import (
    CITATION,
    SOURCE_URL,
    add_segment_labels,
    build_stratified_sample,
    fetch_uci_credit,
    load_credit_data,
    refresh_offline_sample,
)
from .data.processor import CreditDataProcessor, CreditDataset
from .explainability.feature_importance import FeatureImportanceAnalyzer
from .explainability.shap import ShapAnalyzer
from .governance.calibration import CalibrationAnalyzer, expected_calibration_error
from .governance.drift import DriftDetector, population_stability_index
from .governance.fairness import FairnessAnalyzer
from .governance.gates import GateResult, aggregate_status, worst_status
from .governance.reporter import GovernanceReporter, ReportBundle
from .governance.uncertainty import UncertaintyAnalyzer, binary_entropy
from .models.benchmark import BenchmarkResult, ModelBenchmarkSuite, ks_statistic
from .utils.tracking import ExperimentTracker, ensure_reproducibility

__version__ = "2.0.0"

__all__ = [
    # configuration
    "Settings",
    "GateThreshold",
    "load_settings",
    "DEFAULT_CONFIG_PATH",
    "REPO_ROOT",
    "SEED",
    # data
    "load_credit_data",
    "fetch_uci_credit",
    "build_stratified_sample",
    "refresh_offline_sample",
    "add_segment_labels",
    "CreditDataProcessor",
    "CreditDataset",
    "CITATION",
    "SOURCE_URL",
    # benchmarking
    "ModelBenchmarkSuite",
    "BenchmarkResult",
    "ks_statistic",
    # governance
    "GateResult",
    "aggregate_status",
    "worst_status",
    "FairnessAnalyzer",
    "DriftDetector",
    "population_stability_index",
    "CalibrationAnalyzer",
    "expected_calibration_error",
    "UncertaintyAnalyzer",
    "binary_entropy",
    "GovernanceReporter",
    "ReportBundle",
    # bayesian segment shrinkage
    "SegmentShrinkageModel",
    "SegmentStabilityCheck",
    "SegmentPosterior",
    "ShrinkagePrior",
    # explainability
    "ShapAnalyzer",
    "FeatureImportanceAnalyzer",
    # tracking
    "ExperimentTracker",
    "ensure_reproducibility",
    "__version__",
]
