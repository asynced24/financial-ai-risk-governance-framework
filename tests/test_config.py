"""The shipped config.yaml must validate, and bad config must fail loudly."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from financial_ai_framework import DEFAULT_CONFIG_PATH, GateThreshold, Settings, load_settings


def test_shipped_config_validates(settings: Settings):
    assert settings.project.random_seed == 42
    assert settings.models.enabled == ["logistic_regression", "xgboost", "lightgbm"]
    assert settings.data.uci_dataset_id == 350
    assert settings.governance.fairness.protected_attributes == ["sex", "age_band"]


def test_every_gate_threshold_is_ordered(settings: Settings):
    """warn must never sit above fail for any gate in the shipped config."""
    governance = settings.governance
    thresholds = [
        governance.fairness.demographic_parity,
        governance.fairness.equalized_odds,
        governance.drift.feature_psi,
        governance.drift.drifted_feature_ratio,
        governance.calibration.ece,
        governance.calibration.brier,
        governance.uncertainty.high_uncertainty_ratio,
        governance.segment_stability.outside_interval_ratio,
    ]
    for threshold in thresholds:
        assert threshold.warn <= threshold.fail


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, "pass"), (0.049, "pass"), (0.05, "warn"), (0.099, "warn"), (0.10, "fail"), (1.0, "fail")],
)
def test_gate_threshold_classify_boundaries(value, expected):
    threshold = GateThreshold(warn=0.05, fail=0.10)
    assert threshold.classify(value) == expected


def test_gate_threshold_rejects_inverted_limits():
    with pytest.raises(ValidationError, match="must not exceed"):
        GateThreshold(warn=0.20, fail=0.10)


def _base_config() -> dict:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_unknown_config_key_is_rejected():
    """extra='forbid' turns a typo in config.yaml into a startup error."""
    raw = _base_config()
    raw["governance"]["fairness"]["demogrpahic_parity"] = {"warn": 0.1, "fail": 0.2}
    with pytest.raises(ValidationError):
        Settings.model_validate(raw)


def test_non_increasing_age_bins_rejected():
    raw = _base_config()
    raw["segments"]["age_bins"] = [20, 40, 30, 60]
    with pytest.raises(ValidationError, match="strictly increasing"):
        Settings.model_validate(raw)


def test_empty_model_list_rejected():
    raw = _base_config()
    raw["models"]["enabled"] = []
    with pytest.raises(ValidationError, match="at least one model"):
        Settings.model_validate(raw)


def test_duplicate_model_rejected():
    raw = _base_config()
    raw["models"]["enabled"] = ["xgboost", "xgboost"]
    with pytest.raises(ValidationError, match="duplicates"):
        Settings.model_validate(raw)


def test_out_of_range_test_size_rejected():
    raw = _base_config()
    raw["data"]["test_size"] = 0.9
    with pytest.raises(ValidationError):
        Settings.model_validate(raw)


def test_age_band_labels_match_bins(settings: Settings):
    labels = settings.age_band_labels()
    assert labels == ["20-30", "30-40", "40-50", "50-60", "60-80"]
    assert len(labels) == len(settings.segments.age_bins) - 1


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "does_not_exist.yaml")


def test_paths_resolve_against_repo_root(settings: Settings):
    assert settings.sample_path.is_absolute()
    assert settings.sample_path.exists()
    assert settings.reports_dir.name == "reports"
