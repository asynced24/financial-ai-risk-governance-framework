"""The benchmark suite must train every enabled model and gate all of them."""

from __future__ import annotations

import numpy as np
import pytest

from financial_ai_framework import Settings, expected_calibration_error, ks_statistic
from financial_ai_framework.models.benchmark import BenchmarkResult

EXPECTED_GATES = {"fairness", "drift", "calibration", "uncertainty", "segment_stability"}


def test_one_result_per_enabled_model(benchmark_run, fast_settings: Settings):
    _, results = benchmark_run
    assert len(results) == len(fast_settings.models.enabled)
    assert {r.model_name for r in results} == set(fast_settings.models.enabled)
    assert all(isinstance(r, BenchmarkResult) for r in results)


def test_every_model_is_fitted_and_has_predictions(
    benchmark_run, dataset, fast_settings: Settings
):
    suite, _ = benchmark_run
    assert set(suite.models) == set(fast_settings.models.enabled)
    assert set(suite.predictions) == set(fast_settings.models.enabled)

    for probabilities in suite.predictions.values():
        assert probabilities.shape == (len(dataset.X_test),)
        assert probabilities.min() >= 0.0
        assert probabilities.max() <= 1.0


def test_all_five_gates_run_for_every_model(benchmark_run):
    _, results = benchmark_run
    for result in results:
        assert set(result.gates) == EXPECTED_GATES
        for name, gate in result.gates.items():
            assert gate.status in {"pass", "warn", "fail"}
            assert gate.threshold
            assert gate.name == name or name == "drift"


def test_metrics_are_in_range_and_beat_chance(benchmark_run):
    _, results = benchmark_run
    for result in results:
        assert 0.5 < result.roc_auc <= 1.0, f"{result.model_name} does not beat chance"
        assert 0.0 < result.pr_auc <= 1.0
        assert 0.0 < result.ks_statistic <= 1.0
        assert 0.0 < result.brier_score < 0.25
        assert result.log_loss > 0.0
        assert 0.0 <= result.ece <= 1.0


def test_reported_ece_matches_the_calibration_module(
    benchmark_run, dataset, fast_settings: Settings
):
    """The suite must not compute its own ECE - it calls into calibration.py."""
    suite, results = benchmark_run
    y_test = dataset.y_test.to_numpy()

    for result in results:
        recomputed = expected_calibration_error(
            y_test,
            suite.predictions[result.model_name],
            n_bins=fast_settings.governance.calibration.n_bins,
        )
        assert result.ece == pytest.approx(recomputed, rel=1e-12)
        # And the calibration gate must report the same number.
        assert result.gates["calibration"].metrics["ece"] == pytest.approx(recomputed, rel=1e-12)


def test_cross_validation_was_actually_run(benchmark_run, fast_settings: Settings):
    _, results = benchmark_run
    for result in results:
        assert len(result.cv_scores) == fast_settings.models.cv_folds
        assert result.cv_metric == "roc_auc"
        assert 0.5 < result.cv_mean <= 1.0
        assert result.cv_std >= 0.0
        assert result.cv_mean == pytest.approx(float(np.mean(result.cv_scores)))


def test_results_are_sorted_by_the_selection_metric(benchmark_run, fast_settings: Settings):
    _, results = benchmark_run
    key = fast_settings.models.selection_metric
    values = [getattr(r, key) for r in results]
    assert values == sorted(values, reverse=True)


def test_best_model_is_the_top_of_the_leaderboard(benchmark_run):
    suite, results = benchmark_run
    assert suite.best_result is results[0]
    assert suite.best_model is suite.models[results[0].model_name]


def test_leaderboard_frame_has_a_row_per_model(benchmark_run, fast_settings: Settings):
    suite, results = benchmark_run
    frame = suite.leaderboard()
    assert len(frame) == len(results)
    assert set(frame["model"]) == set(fast_settings.models.enabled)
    for gate in EXPECTED_GATES:
        assert f"gate_{gate}" in frame.columns


def test_drift_gate_is_shared_across_models(benchmark_run):
    """Drift is a property of the data, so every model must see one verdict."""
    _, results = benchmark_run
    statuses = {r.gates["drift"].headline_value for r in results}
    assert len(statuses) == 1


def test_shrinkage_model_is_fitted_on_the_training_split(benchmark_run, dataset):
    suite, _ = benchmark_run
    assert suite.shrinkage is not None
    assert suite.shrinkage.prior is not None

    total = sum(p.n for p in suite.shrinkage.posteriors.values())
    assert total == len(dataset.X_train)


def test_timings_and_shapes_are_recorded(benchmark_run, dataset):
    _, results = benchmark_run
    for result in results:
        assert result.train_seconds > 0
        assert result.predict_seconds >= 0
        assert result.n_train == len(dataset.X_train)
        assert result.n_test == len(dataset.X_test)
        assert result.n_features == dataset.n_features


def test_result_serialises_for_the_reporter(benchmark_run):
    _, results = benchmark_run
    payload = results[0].to_dict()
    assert payload["model_name"] == results[0].model_name
    assert set(payload["gates"]) == EXPECTED_GATES
    assert payload["gate_status"] in {"pass", "warn", "fail"}
    assert payload["shape"]["n_features"] > 0
    assert payload["params"]


def test_gate_status_helpers_agree(benchmark_run):
    _, results = benchmark_run
    for result in results:
        failed = result.failed_gates
        warned = result.warned_gates
        if failed:
            assert result.gate_status == "fail"
        elif warned:
            assert result.gate_status == "warn"
        else:
            assert result.gate_status == "pass"


def test_worst_gate_status_across_the_suite(benchmark_run):
    suite, results = benchmark_run
    statuses = {r.gate_status for r in results}
    worst = suite.worst_gate_status()
    assert worst in statuses
    if "fail" in statuses:
        assert worst == "fail"


# ------------------------------------------------------------------ KS helper


def test_ks_statistic_is_one_for_perfect_separation():
    y_true = np.array([0] * 100 + [1] * 100)
    y_score = np.array([0.1] * 100 + [0.9] * 100)
    assert ks_statistic(y_true, y_score) == pytest.approx(1.0)


def test_ks_statistic_is_small_for_no_separation():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.5, size=4000)
    y_score = rng.uniform(size=4000)
    assert ks_statistic(y_true, y_score) < 0.1


def test_ks_statistic_handles_a_single_class():
    assert ks_statistic(np.zeros(50), np.random.default_rng(0).uniform(size=50)) == 0.0
