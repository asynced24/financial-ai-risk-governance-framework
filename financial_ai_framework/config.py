"""Typed configuration for the Bayesian Credit Risk Gate.

Every governance threshold in the framework is declared in ``config.yaml`` at the
repository root and validated here by pydantic v2 at startup. No gate module
hard-codes a limit: each one reads its thresholds from a :class:`Settings`
instance that was already proven well-formed before any model was trained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

#: Fallback seed used before a ``Settings`` instance exists.
SEED = 42

#: Gate verdicts, ordered from best to worst.
GateStatus = Literal["pass", "warn", "fail"]
STATUS_ORDER: dict[str, int] = {"pass": 0, "warn": 1, "fail": 2}

try:  # pragma: no cover - exercised by whether mlflow is installed
    import mlflow  # noqa: F401

    HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    HAS_MLFLOW = False


class _Base(BaseModel):
    """Shared strictness: unknown keys in config.yaml are an error, not a shrug."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class GateThreshold(_Base):
    """A warn/fail limit pair for a metric where a higher value is worse."""

    warn: float
    fail: float

    @model_validator(mode="after")
    def _warn_before_fail(self) -> GateThreshold:
        if self.warn > self.fail:
            raise ValueError(
                f"warn threshold ({self.warn}) must not exceed fail threshold ({self.fail})"
            )
        return self

    def classify(self, value: float) -> GateStatus:
        """Return the gate verdict for an observed ``value``."""
        if value is None:
            return "warn"
        if value >= self.fail:
            return "fail"
        if value >= self.warn:
            return "warn"
        return "pass"

    def describe(self) -> str:
        return f"warn >= {self.warn:g}, fail >= {self.fail:g}"


class ProjectSettings(_Base):
    name: str = "bayesian-credit-risk-gate"
    version: str = "2.0.0"
    random_seed: int = Field(SEED, ge=0)


class DataSettings(_Base):
    uci_dataset_id: int = Field(350, gt=0)
    target_column: str = "default_payment_next_month"
    raw_cache_dir: Path = Path("data/raw")
    sample_path: Path = Path("data/sample/uci_credit_sample.csv")
    sample_rows: int = Field(5000, ge=500)
    test_size: float = Field(0.25, gt=0.05, lt=0.5)


class SegmentSettings(_Base):
    education_column: str = "education"
    age_column: str = "age"
    age_bins: list[int] = Field(default_factory=lambda: [20, 30, 40, 50, 60, 80])
    min_segment_size: int = Field(30, ge=1)
    credible_mass: float = Field(0.95, gt=0.5, lt=1.0)
    prior_variance_floor: float = Field(1e-6, gt=0.0)

    @model_validator(mode="after")
    def _bins_ascending(self) -> SegmentSettings:
        if len(self.age_bins) < 3:
            raise ValueError("age_bins needs at least 3 edges to define 2 bands")
        if any(b <= a for a, b in zip(self.age_bins, self.age_bins[1:], strict=False)):
            raise ValueError(f"age_bins must be strictly increasing, got {self.age_bins}")
        return self


class ModelSettings(_Base):
    enabled: list[Literal["logistic_regression", "xgboost", "lightgbm"]] = Field(
        default_factory=lambda: ["logistic_regression", "xgboost", "lightgbm"]
    )
    cv_folds: int = Field(5, ge=2, le=10)
    decision_threshold: float = Field(0.5, gt=0.0, lt=1.0)
    selection_metric: Literal["roc_auc", "pr_auc"] = "roc_auc"
    logistic_max_iter: int = Field(2000, ge=100)
    n_estimators: int = Field(300, ge=10)
    learning_rate: float = Field(0.05, gt=0.0, le=1.0)
    max_depth: int = Field(4, ge=1)
    num_leaves: int = Field(31, ge=2)
    subsample: float = Field(0.9, gt=0.0, le=1.0)
    colsample_bytree: float = Field(0.9, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _at_least_one_model(self) -> ModelSettings:
        if not self.enabled:
            raise ValueError("models.enabled must list at least one model")
        if len(set(self.enabled)) != len(self.enabled):
            raise ValueError(f"models.enabled contains duplicates: {self.enabled}")
        return self


class FairnessSettings(_Base):
    protected_attributes: list[str] = Field(default_factory=lambda: ["sex", "age_band"])
    min_group_size: int = Field(50, ge=2)
    demographic_parity: GateThreshold
    equalized_odds: GateThreshold


class DriftSettings(_Base):
    psi_bins: int = Field(10, ge=2, le=50)
    feature_psi: GateThreshold
    drifted_feature_ratio: GateThreshold


class CalibrationSettings(_Base):
    n_bins: int = Field(15, ge=3, le=50)
    ece: GateThreshold
    brier: GateThreshold


class UncertaintySettings(_Base):
    high_entropy_threshold: float = Field(0.60, gt=0.0, lt=0.6931472)
    high_uncertainty_ratio: GateThreshold
    bootstrap_samples: int = Field(200, ge=10)
    bootstrap_sample_size: int = Field(500, ge=50)


class SegmentStabilitySettings(_Base):
    outside_interval_ratio: GateThreshold


class GovernanceSettings(_Base):
    fairness: FairnessSettings
    drift: DriftSettings
    calibration: CalibrationSettings
    uncertainty: UncertaintySettings
    segment_stability: SegmentStabilitySettings


class ExplainabilitySettings(_Base):
    enable_shap: bool = True
    shap_sample_size: int = Field(300, ge=10)
    shap_background_size: int = Field(100, ge=10)
    local_explanations: int = Field(3, ge=0)
    top_features: int = Field(15, ge=1)


class TrackingSettings(_Base):
    enable_mlflow: bool = False
    experiment_name: str = "credit_default_governance"
    reports_dir: Path = Path("reports")


class Settings(_Base):
    """Fully validated runtime configuration for one framework invocation."""

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    segments: SegmentSettings = Field(default_factory=SegmentSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    governance: GovernanceSettings
    explainability: ExplainabilitySettings = Field(default_factory=ExplainabilitySettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)

    @property
    def seed(self) -> int:
        return self.project.random_seed

    def resolve(self, relative: Path) -> Path:
        """Resolve a config-declared relative path against the repository root."""
        path = Path(relative)
        return path if path.is_absolute() else (REPO_ROOT / path)

    @property
    def reports_dir(self) -> Path:
        return self.resolve(self.tracking.reports_dir)

    @property
    def raw_cache_dir(self) -> Path:
        return self.resolve(self.data.raw_cache_dir)

    @property
    def sample_path(self) -> Path:
        return self.resolve(self.data.sample_path)

    def age_band_labels(self) -> list[str]:
        """Human-readable labels for the configured age bands."""
        edges = self.segments.age_bins
        return [f"{lo}-{hi}" for lo, hi in zip(edges, edges[1:], strict=False)]


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load and validate ``config.yaml``.

    Raises
    ------
    FileNotFoundError
        If the config file is missing.
    pydantic.ValidationError
        If any value is missing, mistyped, out of range, or unrecognised.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    return Settings.model_validate(raw)
