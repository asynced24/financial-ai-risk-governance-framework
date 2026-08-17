"""The reporter must render every model, every gate, and the dataset lineage."""

from __future__ import annotations

import json

import pytest

from financial_ai_framework import GovernanceReporter, Settings
from financial_ai_framework.governance.reporter import RECOMMENDATION


@pytest.fixture(scope="module")
def rendered(benchmark_run, credit_frame, fast_settings: Settings, tmp_path_factory):
    """Render a full report bundle into a temporary directory."""
    suite, results = benchmark_run
    _, metadata = credit_frame
    out = tmp_path_factory.mktemp("reports")

    reporter = GovernanceReporter(fast_settings, tracker=None, reports_dir=out)
    bundle = reporter.generate(
        results=results,
        data_metadata=metadata,
        shrinkage_summary=suite.shrinkage.summary(),
        shap_result={"status": "disabled"},
        importance_result={"top_features": {"pay_status_1": 0.4}},
        run_id="test_run",
    )
    return bundle, results, out


def test_writes_scorecard_and_one_card_per_model(rendered):
    bundle, results, _ = rendered
    assert bundle.scorecard_markdown.exists()
    assert bundle.scorecard_json.exists()
    assert len(bundle.model_cards) == len(results)
    for path in bundle.all_paths():
        assert path.stat().st_size > 0


def test_model_card_filenames_match_the_models(rendered):
    bundle, results, _ = rendered
    names = {path.stem.replace("model_card_", "") for path in bundle.model_cards}
    assert names == {result.model_name for result in results}


def test_scorecard_json_is_valid_and_complete(rendered):
    bundle, results, _ = rendered
    payload = json.loads(bundle.scorecard_json.read_text(encoding="utf-8"))

    assert payload["run_id"] == "test_run"
    assert payload["overall_status"] in {"pass", "warn", "fail"}
    assert payload["recommendation"] == RECOMMENDATION[payload["overall_status"]]
    assert payload["selected_model"] == results[0].model_name
    assert len(payload["models"]) == len(results)

    for model in payload["models"]:
        assert set(model["gates"]) == set(results[0].gates)
        assert "roc_auc" in model["metrics"]


def test_scorecard_json_records_the_thresholds_actually_used(rendered, fast_settings: Settings):
    bundle, _, _ = rendered
    payload = json.loads(bundle.scorecard_json.read_text(encoding="utf-8"))
    thresholds = payload["thresholds"]

    assert (
        thresholds["fairness"]["demographic_parity"]["fail"]
        == fast_settings.governance.fairness.demographic_parity.fail
    )
    assert thresholds["drift"]["feature_psi"]["warn"] == (
        fast_settings.governance.drift.feature_psi.warn
    )


def test_overall_status_is_the_worst_model_status(rendered):
    bundle, results, _ = rendered
    payload = json.loads(bundle.scorecard_json.read_text(encoding="utf-8"))
    statuses = {result.gate_status for result in results}

    if "fail" in statuses:
        assert payload["overall_status"] == "fail"
    elif "warn" in statuses:
        assert payload["overall_status"] == "warn"
    else:
        assert payload["overall_status"] == "pass"


def test_scorecard_markdown_covers_models_gates_and_lineage(rendered, credit_frame):
    bundle, results, _ = rendered
    _, metadata = credit_frame
    text = bundle.scorecard_markdown.read_text(encoding="utf-8")

    assert text.startswith("# Governance Scorecard")
    for result in results:
        assert result.model_name in text
    for gate in results[0].gates:
        assert gate in text

    assert metadata["data_hash"] in text
    assert "CC BY 4.0" in text
    assert "Yeh" in text
    assert "Bayesian segment shrinkage" in text


def test_scorecard_reports_the_shrinkage_prior(rendered):
    bundle, _, _ = rendered
    text = bundle.scorecard_markdown.read_text(encoding="utf-8")
    assert "Population prior" in text
    assert "equivalent borrowers" in text


def test_model_card_has_the_required_sections(rendered):
    bundle, _, _ = rendered
    text = bundle.model_cards[0].read_text(encoding="utf-8")

    for heading in (
        "# Model Card",
        "## Intended use",
        "## Data lineage",
        "## Training setup",
        "## Held-out metrics",
        "## Governance gates",
        "## Limitations",
        "## Reproduction",
    ):
        assert heading in text


def test_model_card_states_the_withheld_prohibited_bases(rendered):
    bundle, _, _ = rendered
    text = bundle.model_cards[0].read_text(encoding="utf-8")
    assert "prohibited bases" in text
    assert "`sex`" in text
    assert "`marriage`" in text


def _card_for(bundle, model_name: str):
    return next(path for path in bundle.model_cards if path.stem == f"model_card_{model_name}")


def test_model_card_records_metrics_and_citation(rendered):
    bundle, results, _ = rendered
    best = results[0]
    text = _card_for(bundle, best.model_name).read_text(encoding="utf-8")

    assert f"{best.roc_auc:.4f}" in text
    assert f"{best.ece:.4f}" in text
    assert f"{best.brier_score:.4f}" in text
    assert "Yeh" in text
    assert best.gate_status.upper() in text


def test_model_card_lists_open_findings_as_limitations(rendered):
    bundle, results, _ = rendered
    for result in results:
        text = _card_for(bundle, result.model_name).read_text(encoding="utf-8")
        if result.failed_gates or result.warned_gates:
            assert "Open `" in text


def test_generate_rejects_an_empty_result_list(fast_settings: Settings, tmp_path):
    reporter = GovernanceReporter(fast_settings, tracker=None, reports_dir=tmp_path)
    with pytest.raises(ValueError, match="at least one BenchmarkResult"):
        reporter.generate(results=[], data_metadata={})


def test_print_summary_renders_the_leaderboard_and_gate_matrix(
    rendered, fast_settings: Settings, tmp_path, capsys
):
    _, results, _ = rendered
    reporter = GovernanceReporter(fast_settings, tracker=None, reports_dir=tmp_path)
    reporter.print_summary(results)

    captured = capsys.readouterr().out
    assert "GOVERNANCE SUMMARY" in captured
    for result in results:
        assert result.model_name in captured
    for gate in results[0].gates:
        assert gate in captured
    assert "Selected model" in captured
