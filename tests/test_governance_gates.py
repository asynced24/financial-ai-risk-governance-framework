"""Each governance gate must fire on a known-bad input and stay quiet on a good one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from financial_ai_framework import (
    CalibrationAnalyzer,
    DriftDetector,
    FairnessAnalyzer,
    Settings,
    UncertaintyAnalyzer,
    binary_entropy,
    expected_calibration_error,
    population_stability_index,
)
from financial_ai_framework.governance.calibration import (
    calibration_bins,
    maximum_calibration_error,
)
from financial_ai_framework.governance.gates import aggregate_status, worst_status

# ------------------------------------------------------------------ calibration


def _calibrated(n: int = 8000, seed: int = 0):
    """Probabilities that are true by construction: y ~ Bernoulli(p).

    Drawn from Beta(1.2, 4.3) so the implied book looks like the real one - a ~22%
    base rate with most borrowers scored low. That matters for the Brier gate,
    whose limits are set against a 22% base rate: a *perfectly* calibrated model on
    a 50/50 book scores Brier ~0.17 and would legitimately trip them.
    """
    rng = np.random.default_rng(seed)
    prob = rng.beta(1.2, 4.3, size=n)
    return rng.binomial(1, prob), prob


def test_ece_near_zero_for_calibrated_probabilities():
    y_true, y_prob = _calibrated()
    assert expected_calibration_error(y_true, y_prob, n_bins=15) < 0.03


def test_ece_detects_a_systematic_shift():
    y_true, y_prob = _calibrated()
    inflated = np.clip(y_prob + 0.25, 0, 1)
    assert expected_calibration_error(y_true, inflated, n_bins=15) > 0.15


def test_ece_of_a_constant_predictor_equals_its_bias():
    y_true = np.array([1] * 300 + [0] * 700)
    y_prob = np.full(1000, 0.5)
    # One populated bin: |observed 0.30 - predicted 0.50| = 0.20
    assert expected_calibration_error(y_true, y_prob, n_bins=10) == pytest.approx(0.20, abs=1e-9)


def test_mce_is_at_least_ece():
    y_true, y_prob = _calibrated()
    ece = expected_calibration_error(y_true, y_prob, n_bins=15)
    mce = maximum_calibration_error(y_true, y_prob, n_bins=15)
    assert mce >= ece


def test_calibration_bin_weights_sum_to_one():
    y_true, y_prob = _calibrated()
    bins = calibration_bins(y_true, y_prob, n_bins=15)
    assert sum(bins["weight"]) == pytest.approx(1.0)
    assert sum(bins["count"]) == len(y_true)


def test_ece_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="Shape mismatch"):
        expected_calibration_error(np.zeros(10), np.zeros(5))


def test_calibration_gate_passes_on_calibrated_and_fails_on_skewed(settings: Settings):
    analyzer = CalibrationAnalyzer(settings)
    y_true, y_prob = _calibrated()

    good = analyzer.run(y_true, y_prob, "calibrated")
    assert good.status == "pass"
    assert good.findings == []
    assert good.metrics["ece"] < settings.governance.calibration.ece.warn
    assert good.metrics["brier_score"] < settings.governance.calibration.brier.warn

    bad = analyzer.run(y_true, np.clip(y_prob + 0.3, 0, 1), "skewed")
    assert bad.status == "fail"
    assert bad.metrics["calibration_bias"] > 0.1
    assert any("ECE" in finding for finding in bad.findings)


def test_calibration_gate_reports_the_base_rate_reference(settings: Settings):
    y_true, y_prob = _calibrated()
    gate = CalibrationAnalyzer(settings).run(y_true, y_prob, "m")
    base = gate.metrics["observed_default_rate"]
    assert gate.metrics["reference_brier_base_rate"] == pytest.approx(base * (1 - base))


# ------------------------------------------------------------------------ drift


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(1)
    values = rng.normal(size=5000)
    assert population_stability_index(values, values.copy(), bins=10) == pytest.approx(0.0, abs=1e-6)


def test_psi_flags_a_large_location_shift():
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, 5000)
    shifted = rng.normal(2, 1, 5000)
    assert population_stability_index(reference, shifted, bins=10) > 0.25


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 5000)
    small = population_stability_index(reference, rng.normal(0.2, 1, 5000), bins=10)
    large = population_stability_index(reference, rng.normal(1.5, 1, 5000), bins=10)
    assert small < large


def test_psi_handles_a_constant_feature():
    constant = np.zeros(500)
    assert population_stability_index(constant, np.zeros(500), bins=10) == 0.0
    assert population_stability_index(constant, np.ones(500), bins=10) == 0.0


def test_drift_gate_passes_on_a_random_split(settings: Settings):
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(size=(4000, 6)), columns=[f"f{i}" for i in range(6)])
    gate = DriftDetector(settings).run(frame.iloc[:3000], frame.iloc[3000:])

    assert gate.status == "pass"
    assert gate.metrics["features_tested"] == 6
    assert gate.metrics["drifted_features"] == 0


def test_drift_gate_fails_when_features_are_shifted(settings: Settings):
    rng = np.random.default_rng(5)
    columns = [f"f{i}" for i in range(6)]
    train = pd.DataFrame(rng.normal(0, 1, size=(3000, 6)), columns=columns)
    test = pd.DataFrame(rng.normal(2.5, 1, size=(1000, 6)), columns=columns)

    gate = DriftDetector(settings).run(train, test)
    assert gate.status == "fail"
    assert gate.metrics["drifted_features"] == 6
    assert gate.metrics["max_feature_psi"] > 0.25
    assert gate.metrics["max_psi_feature"] in columns


def test_drift_gate_is_deterministic(settings: Settings):
    """No feature subsampling: the same input must give the same verdict."""
    rng = np.random.default_rng(6)
    frame = pd.DataFrame(rng.normal(size=(2000, 8)), columns=[f"f{i}" for i in range(8)])
    detector = DriftDetector(settings)

    first = detector.run(frame.iloc[:1500], frame.iloc[1500:])
    second = detector.run(frame.iloc[:1500], frame.iloc[1500:])
    assert first.headline_value == second.headline_value
    assert first.metrics == second.metrics


# --------------------------------------------------------------------- fairness


def _fairness_inputs(gap: float, per_group: int = 1000, seed: int = 7):
    """Two groups that are exact statistical twins, except group B is shifted.

    Group B reuses group A's outcomes and scores verbatim, so at ``gap = 0`` every
    parity measure is exactly zero. Any gap the analyzer reports therefore comes
    from ``gap`` alone rather than from sampling noise between two draws.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.05, 0.60, size=per_group)
    outcomes = rng.binomial(1, base)

    group = np.array(["a"] * per_group + ["b"] * per_group)
    y_true = np.concatenate([outcomes, outcomes])
    prob = np.concatenate([base, np.clip(base + gap, 0.0, 1.0)])
    audit = pd.DataFrame({"sex": group, "age_band": group})
    return y_true, prob, audit


def test_fairness_gate_passes_when_groups_are_treated_alike(settings: Settings):
    y_true, y_prob, audit = _fairness_inputs(gap=0.0)
    gate = FairnessAnalyzer(settings).run(y_true, y_prob, audit, "even")

    assert gate.status == "pass"
    assert gate.metrics["attributes_analysed"] == 2
    assert gate.findings == []
    assert gate.metrics["sex_demographic_parity_gap"] == pytest.approx(0.0)
    assert gate.metrics["sex_equalized_odds_gap"] == pytest.approx(0.0)


def test_fairness_gate_fails_on_a_large_parity_gap(settings: Settings):
    y_true, y_prob, audit = _fairness_inputs(gap=0.35)
    gate = FairnessAnalyzer(settings).run(y_true, y_prob, audit, "skewed")

    assert gate.status == "fail"
    assert gate.metrics["worst_gap"] >= settings.governance.fairness.demographic_parity.fail
    assert any("demographic parity" in finding for finding in gate.findings)


def test_fairness_gate_records_per_group_rates(settings: Settings):
    y_true, y_prob, audit = _fairness_inputs(gap=0.35)
    gate = FairnessAnalyzer(settings).run(y_true, y_prob, audit, "skewed")

    groups = gate.details["attributes"]["sex"]["groups"]
    assert set(groups) == {"a", "b"}
    assert groups["b"]["predicted_default_rate"] > groups["a"]["predicted_default_rate"]
    assert groups["a"]["n"] + groups["b"]["n"] == 2000


def test_fairness_gate_ignores_groups_below_the_size_floor(settings: Settings):
    n = 400
    rng = np.random.default_rng(8)
    group = np.array(["big"] * (n - 5) + ["tiny"] * 5)
    y_prob = np.where(group == "tiny", 0.99, 0.1)
    y_true = rng.binomial(1, 0.2, size=n)
    audit = pd.DataFrame({"sex": group, "age_band": group})

    gate = FairnessAnalyzer(settings).run(y_true, y_prob, audit, "one_real_group")
    # Only one group clears min_group_size, so nothing is auditable.
    assert gate.status == "warn"
    assert gate.metrics["attributes_analysed"] == 0
    assert any("fewer than 2 groups" in note for note in gate.details["skipped"])


def test_fairness_gate_warns_when_the_attribute_is_absent(settings: Settings):
    y_true, y_prob, _ = _fairness_inputs(gap=0.0)
    gate = FairnessAnalyzer(settings).run(y_true, y_prob, pd.DataFrame({"other": ["x"] * 2000}), "m")

    assert gate.status == "warn"
    assert any("column absent" in note for note in gate.details["skipped"])


# ------------------------------------------------------------------ uncertainty


def test_binary_entropy_peaks_at_one_half():
    assert binary_entropy(np.array([0.5]))[0] == pytest.approx(np.log(2))
    assert binary_entropy(np.array([0.01]))[0] < 0.1
    assert binary_entropy(np.array([0.99]))[0] < 0.1
    # Symmetric around 0.5.
    assert binary_entropy(np.array([0.3]))[0] == pytest.approx(binary_entropy(np.array([0.7]))[0])


def test_uncertainty_gate_passes_on_confident_predictions(settings: Settings):
    rng = np.random.default_rng(9)
    y_true = rng.binomial(1, 0.25, size=3000)
    # Confident and correct: probabilities pinned near the true label.
    y_prob = np.where(y_true == 1, 0.95, 0.05)

    gate = UncertaintyAnalyzer(settings).run(y_true, y_prob, "confident")
    assert gate.status == "pass"
    assert gate.metrics["high_uncertainty_ratio"] == 0.0


def test_uncertainty_gate_fails_when_everything_is_a_coin_flip(settings: Settings):
    rng = np.random.default_rng(10)
    y_true = rng.binomial(1, 0.5, size=2000)
    y_prob = np.full(2000, 0.5)

    gate = UncertaintyAnalyzer(settings).run(y_true, y_prob, "coinflip")
    assert gate.status == "fail"
    assert gate.metrics["high_uncertainty_ratio"] == pytest.approx(1.0)
    assert any("coin flip" in finding for finding in gate.findings)


def test_uncertainty_gate_bootstrap_interval_brackets_the_point_estimate(settings: Settings):
    rng = np.random.default_rng(11)
    y_true = rng.binomial(1, 0.3, size=3000)
    y_prob = np.clip(0.3 + 0.4 * (y_true - 0.5) + rng.normal(0, 0.1, 3000), 0.01, 0.99)

    gate = UncertaintyAnalyzer(settings).run(y_true, y_prob, "m")
    assert gate.metrics["roc_auc_ci_lower"] < gate.metrics["roc_auc_bootstrap_mean"]
    assert gate.metrics["roc_auc_bootstrap_mean"] < gate.metrics["roc_auc_ci_upper"]
    assert gate.metrics["roc_auc_ci_width"] > 0


def test_uncertainty_gate_is_reproducible(settings: Settings):
    """The bootstrap is seeded, so two runs must agree exactly."""
    rng = np.random.default_rng(12)
    y_true = rng.binomial(1, 0.3, size=1500)
    y_prob = rng.uniform(0.05, 0.95, size=1500)
    analyzer = UncertaintyAnalyzer(settings)

    first = analyzer.run(y_true, y_prob, "m")
    second = analyzer.run(y_true, y_prob, "m")
    assert first.metrics == second.metrics


# ---------------------------------------------------------------- gate plumbing


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([], "pass"),
        (["pass", "pass"], "pass"),
        (["pass", "warn"], "warn"),
        (["warn", "fail", "pass"], "fail"),
        (["fail"], "fail"),
    ],
)
def test_worst_status(statuses, expected):
    assert worst_status(statuses) == expected


def test_aggregate_status_uses_the_worst_gate(settings: Settings):
    y_true, y_prob, audit = _fairness_inputs(gap=0.35)
    gates = {
        "fairness": FairnessAnalyzer(settings).run(y_true, y_prob, audit, "m"),
        "calibration": CalibrationAnalyzer(settings).run(*_calibrated(), "m"),
    }
    assert gates["calibration"].status == "pass"
    assert aggregate_status(gates) == "fail"


def test_gate_result_round_trips_to_dict(settings: Settings):
    gate = CalibrationAnalyzer(settings).run(*_calibrated(), "m")
    payload = gate.to_dict()
    assert payload["name"] == "calibration"
    assert payload["status"] in {"pass", "warn", "fail"}
    assert "ece" in payload["metrics"]
    assert gate.passed is (gate.status == "pass")
