"""Governance reporting: scorecard plus an auto-generated model card per model.

Consumes the ``BenchmarkResult`` list and every gate output, and writes:

* ``reports/governance_scorecard.md``   - the human-readable review document
* ``reports/governance_scorecard.json`` - the same content, machine-readable
* ``reports/model_card_<model>.md``     - one model card per benchmarked model

Nothing here recomputes a metric or re-derives a verdict. The reporter renders
what the gates decided, which keeps the document and the pipeline honest with
each other.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import STATUS_ORDER, Settings

RECOMMENDATION = {
    "pass": (
        "APPROVED FOR PILOT - every governance gate passed at the configured limits."
    ),
    "warn": (
        "CONDITIONAL - no gate failed, but the warnings below must be remediated or "
        "formally accepted with compensating controls before production use."
    ),
    "fail": (
        "NOT APPROVED - at least one governance gate failed. The model must not be "
        "deployed for credit decisioning until the failure is resolved."
    ),
}

GATE_PURPOSE = {
    "fairness": (
        "Disparate impact on protected attributes - sex and marriage are withheld "
        "from the feature set, age_band is a binned view of a feature the model uses"
    ),
    "drift": "Train-vs-test population stability of every model input",
    "calibration": "Whether the predicted probability of default means what it says",
    "uncertainty": "Share of the book the model cannot separate, and AUC stability",
    "segment_stability": (
        "Agreement between the model's segment-level PD and the empirical-Bayes "
        "shrunken default experience for that segment"
    ),
}


@dataclass
class ReportBundle:
    """Paths written by one reporting pass."""

    scorecard_markdown: Path
    scorecard_json: Path
    model_cards: list[Path] = field(default_factory=list)

    def all_paths(self) -> list[Path]:
        return [self.scorecard_markdown, self.scorecard_json, *self.model_cards]


def _fmt(value: Any, places: int = 4) -> str:
    """Format a metric for a markdown table, tolerating None and NaN."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.{places}f}"
    return str(value)


def _rows(data_metadata: dict[str, Any]) -> str:
    """Row count with thousands separators when it is an integer."""
    rows = data_metadata.get("rows")
    return f"{rows:,}" if isinstance(rows, int) else str(rows if rows is not None else "n/a")


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


class GovernanceReporter:
    """Renders the governance scorecard and the per-model model cards."""

    def __init__(self, settings: Settings, tracker=None, reports_dir: Path | None = None):
        self.settings = settings
        self.tracker = tracker
        self.reports_dir = Path(reports_dir) if reports_dir else settings.reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ public

    def generate(
        self,
        results: list[Any],
        data_metadata: dict[str, Any],
        shrinkage_summary: dict[str, Any] | None = None,
        shap_result: dict[str, Any] | None = None,
        importance_result: dict[str, Any] | None = None,
        run_id: str = "run",
    ) -> ReportBundle:
        """Write the scorecard and one model card per benchmarked model."""
        if not results:
            raise ValueError("GovernanceReporter.generate() needs at least one BenchmarkResult")

        generated_at = datetime.now(UTC).isoformat(timespec="seconds")
        overall = max(
            (result.gate_status for result in results), key=lambda s: STATUS_ORDER[s]
        )

        payload = self._build_payload(
            results=results,
            data_metadata=data_metadata,
            shrinkage_summary=shrinkage_summary,
            shap_result=shap_result,
            importance_result=importance_result,
            run_id=run_id,
            generated_at=generated_at,
            overall=overall,
        )

        json_path = self.reports_dir / "governance_scorecard.json"
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        md_path = self.reports_dir / "governance_scorecard.md"
        md_path.write_text(
            self._render_scorecard(
                results=results,
                data_metadata=data_metadata,
                shrinkage_summary=shrinkage_summary,
                shap_result=shap_result,
                run_id=run_id,
                generated_at=generated_at,
                overall=overall,
            ),
            encoding="utf-8",
        )

        cards: list[Path] = []
        for result in results:
            card = self.reports_dir / f"model_card_{result.model_name}.md"
            card.write_text(
                self._render_model_card(
                    result=result,
                    data_metadata=data_metadata,
                    shap_result=shap_result,
                    run_id=run_id,
                    generated_at=generated_at,
                ),
                encoding="utf-8",
            )
            cards.append(card)

        bundle = ReportBundle(scorecard_markdown=md_path, scorecard_json=json_path, model_cards=cards)

        if self.tracker is not None:
            for path in bundle.all_paths():
                self.tracker.log_artifact(path)

        print(f"[report] scorecard -> {md_path}")
        print(f"[report] scorecard json -> {json_path}")
        print(f"[report] {len(cards)} model card(s) -> {self.reports_dir}")
        return bundle

    def print_summary(self, results: list[Any], overall: str | None = None) -> None:
        """Console summary of the leaderboard and gate verdicts."""
        overall = overall or max(
            (r.gate_status for r in results), key=lambda s: STATUS_ORDER[s]
        )
        width = 96

        print("\n" + "=" * width)
        print("GOVERNANCE SUMMARY")
        print("=" * width)

        header = f"{'model':<22}{'ROC-AUC':>9}{'PR-AUC':>9}{'KS':>8}{'Brier':>9}{'ECE':>8}{'gates':>9}"
        print(header)
        print("-" * width)
        for result in results:
            print(
                f"{result.model_name:<22}{result.roc_auc:>9.4f}{result.pr_auc:>9.4f}"
                f"{result.ks_statistic:>8.4f}{result.brier_score:>9.4f}{result.ece:>8.4f}"
                f"{result.gate_status.upper():>9}"
            )

        print("-" * width)
        gate_names = list(results[0].gates.keys())
        print(f"{'gate':<22}" + "".join(f"{r.model_name[:12]:>14}" for r in results))
        for gate_name in gate_names:
            row = "".join(f"{r.gates[gate_name].symbol:>14}" for r in results)
            print(f"{gate_name:<22}{row}")

        print("=" * width)
        best = results[0]
        print(f"Selected model: {best.model_name} (by {self.settings.models.selection_metric})")
        print(f"Overall verdict: {overall.upper()}")
        print(RECOMMENDATION[overall])

        findings = [
            f"  - {gate_name}: {finding}"
            for gate_name, gate in best.gates.items()
            if gate.status != "pass"
            for finding in gate.findings[:2]
        ]
        if findings:
            print(f"\nOpen findings for the selected model (`{best.model_name}`):")
            for line in findings:
                print(line)
            print("\nFull detail for every model: reports/governance_scorecard.md")
        print("=" * width)

    # ----------------------------------------------------------------- payload

    def _build_payload(
        self,
        results: list[Any],
        data_metadata: dict[str, Any],
        shrinkage_summary: dict[str, Any] | None,
        shap_result: dict[str, Any] | None,
        importance_result: dict[str, Any] | None,
        run_id: str,
        generated_at: str,
        overall: str,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "generated_at": generated_at,
            "framework_version": self.settings.project.version,
            "overall_status": overall,
            "recommendation": RECOMMENDATION[overall],
            "selected_model": results[0].model_name,
            "selection_metric": self.settings.models.selection_metric,
            "dataset": data_metadata,
            "thresholds": self.settings.governance.model_dump(mode="json"),
            "segment_shrinkage": shrinkage_summary or {},
            "explainability": shap_result or {},
            "feature_importance_consensus": (importance_result or {}).get("top_features", {}),
            "models": [result.to_dict() for result in results],
        }

    # ---------------------------------------------------------------- markdown

    def _render_scorecard(
        self,
        results: list[Any],
        data_metadata: dict[str, Any],
        shrinkage_summary: dict[str, Any] | None,
        shap_result: dict[str, Any] | None,
        run_id: str,
        generated_at: str,
        overall: str,
    ) -> str:
        lines: list[str] = [
            "# Governance Scorecard",
            "",
            f"- **Run ID:** `{run_id}`",
            f"- **Generated:** {generated_at}",
            f"- **Framework version:** {self.settings.project.version}",
            f"- **Overall verdict:** **{overall.upper()}**",
            f"- **Selected model:** `{results[0].model_name}` "
            f"(by {self.settings.models.selection_metric})",
            "",
            f"> {RECOMMENDATION[overall]}",
            "",
            "## Dataset",
            "",
            f"- Source: {data_metadata.get('source', 'unknown')} "
            f"(`{data_metadata.get('path', 'n/a')}`)",
            f"- Rows: {_rows(data_metadata)}",
            f"- Observed default rate: {_fmt(data_metadata.get('default_rate'))}",
            f"- Demographic segments: {data_metadata.get('n_segments', 'n/a')}",
            f"- Content hash: `{data_metadata.get('data_hash', 'n/a')}`",
            f"- Licence: {data_metadata.get('licence', 'n/a')}",
            f"- Citation: {data_metadata.get('citation', 'n/a')}",
            "",
            "## Model leaderboard",
            "",
        ]

        lines += _table(
            ["Model", "ROC-AUC", "PR-AUC", "KS", "Brier", "Log-loss", "ECE", "CV mean", "CV std", "Gates"],
            [
                [
                    f"`{r.model_name}`",
                    _fmt(r.roc_auc),
                    _fmt(r.pr_auc),
                    _fmt(r.ks_statistic),
                    _fmt(r.brier_score),
                    _fmt(r.log_loss),
                    _fmt(r.ece),
                    _fmt(r.cv_mean),
                    _fmt(r.cv_std),
                    r.gate_status.upper(),
                ]
                for r in results
            ],
        )

        lines += ["", "## Governance gate matrix", ""]
        gate_names = list(results[0].gates.keys())
        lines += _table(
            ["Gate", "What it checks", *[f"`{r.model_name}`" for r in results]],
            [
                [
                    gate_name,
                    GATE_PURPOSE.get(gate_name, ""),
                    *[r.gates[gate_name].symbol for r in results],
                ]
                for gate_name in gate_names
            ],
        )

        lines += ["", "## Gate detail", ""]
        for result in results:
            lines += [f"### `{result.model_name}` - {result.gate_status.upper()}", ""]
            lines += _table(
                ["Gate", "Verdict", "Headline metric", "Value", "Threshold"],
                [
                    [
                        gate_name,
                        gate.symbol,
                        gate.headline_metric,
                        _fmt(gate.headline_value),
                        gate.threshold,
                    ]
                    for gate_name, gate in result.gates.items()
                ],
            )
            findings = [
                (gate_name, finding)
                for gate_name, gate in result.gates.items()
                if gate.status != "pass"
                for finding in gate.findings
            ]
            if findings:
                lines += ["", "**Findings**", ""]
                lines += [f"- `{gate_name}`: {finding}" for gate_name, finding in findings]
            else:
                lines += ["", "No findings: every gate passed at the configured limits."]
            lines.append("")

        if shrinkage_summary:
            lines += self._render_shrinkage_section(shrinkage_summary, results)

        if shap_result and shap_result.get("status") == "success":
            lines += ["## Explainability", ""]
            lines += [
                f"SHAP attribution for the selected model `{shap_result['model']}` "
                f"using {shap_result['explainer']} over {shap_result['sample_size']} "
                f"evaluation rows.",
                "",
            ]
            lines += _table(
                ["Rank", "Feature", "Mean absolute SHAP value"],
                [
                    [str(rank), f"`{name}`", _fmt(value, 5)]
                    for rank, (name, value) in enumerate(
                        list(shap_result["top_features"].items()), start=1
                    )
                ],
            )
            lines += ["", "Artefacts:", ""]
            lines += [f"- `{Path(p).name}`" for p in shap_result.get("artifacts", [])]
            lines.append("")

        lines += ["## Configured thresholds", "", "```yaml"]
        lines += json.dumps(
            self.settings.governance.model_dump(mode="json"), indent=2
        ).splitlines()
        lines += ["```", ""]

        return "\n".join(lines)

    def _render_shrinkage_section(
        self,
        shrinkage_summary: dict[str, Any],
        results: list[Any],
    ) -> list[str]:
        prior = shrinkage_summary.get("prior", {})
        lines = [
            "## Bayesian segment shrinkage",
            "",
            "Segment-level default rates are shrunk toward the population rate by an "
            "empirical-Bayes Beta-Binomial model, so thin demographic cells are not "
            "judged on a handful of borrowers. The blend weight is not configured - it "
            "is `n / (n + a + b)` from the posterior.",
            "",
            _bullet("Population prior", f"Beta({_fmt(prior.get('alpha'), 3)}, {_fmt(prior.get('beta'), 3)})"),
            _bullet("Prior mean (population default rate)", _fmt(prior.get("prior_mean"))),
            _bullet(
                "Prior strength",
                f"{_fmt(prior.get('concentration_equivalent_borrowers'), 1)} equivalent borrowers",
            ),
            _bullet("Segments", str(shrinkage_summary.get("n_segments", "n/a"))),
            _bullet(
                "Sparse segments (below the prior-fitting floor)",
                str(shrinkage_summary.get("n_sparse_segments", "n/a")),
            ),
            _bullet(
                "Median weight on a segment's own data",
                _fmt(shrinkage_summary.get("median_shrinkage_weight")),
            ),
            _bullet(
                "Largest shrinkage applied",
                f"{_fmt(shrinkage_summary.get('max_abs_shrinkage'))} rate points",
            ),
            "",
        ]

        # Segments the selected model prices outside their credible interval.
        gate = results[0].gates.get("segment_stability")
        if gate is not None:
            segments = [
                s
                for s in gate.details.get("segments", [])
                if s.get("gated") and not s.get("inside_interval")
            ]
            if segments:
                lines += [
                    f"### Segments flagged for `{results[0].model_name}`",
                    "",
                ]
                lines += _table(
                    [
                        "Segment",
                        "Reference n",
                        "Raw rate",
                        "Shrunken mean",
                        "Credible interval",
                        "Mean predicted PD",
                        "Direction",
                    ],
                    [
                        [
                            f"`{s['segment_id']}`",
                            str(s["n_reference"]),
                            _fmt(s["empirical_rate"], 3),
                            _fmt(s["posterior_mean"], 3),
                            f"[{_fmt(s['ci_lower'], 3)}, {_fmt(s['ci_upper'], 3)}]",
                            _fmt(s["mean_predicted_pd"], 3),
                            s["direction"],
                        ]
                        for s in segments
                    ],
                )
                lines.append("")
            else:
                lines += [
                    f"Every gated segment's mean predicted PD for "
                    f"`{results[0].model_name}` fell inside its credible interval.",
                    "",
                ]
        return lines

    def _render_model_card(
        self,
        result: Any,
        data_metadata: dict[str, Any],
        shap_result: dict[str, Any] | None,
        run_id: str,
        generated_at: str,
    ) -> str:
        gates = result.gates
        fairness = gates.get("fairness")

        lines: list[str] = [
            f"# Model Card - `{result.model_name}`",
            "",
            f"- **Run ID:** `{run_id}`",
            f"- **Generated:** {generated_at}",
            f"- **Framework version:** {self.settings.project.version}",
            f"- **Governance verdict:** **{result.gate_status.upper()}**",
            "",
            "## Intended use",
            "",
            "Estimating the probability that a revolving credit-card customer defaults "
            "on their next monthly payment, from six months of billing and repayment "
            "history plus limited demographics.",
            "",
            "**In scope:** portfolio-level risk ranking, provisioning inputs, "
            "early-warning triage, and governance research.",
            "",
            "**Out of scope:** this model is trained on a single public research "
            "dataset of Taiwanese credit-card customers from 2005. It is not fit for "
            "live credit decisioning on any other population, and it is not a "
            "regulatory-approved scorecard. Any adverse action would additionally "
            "require reason codes derived from the local explanations, not the model "
            "score alone.",
            "",
            "## Data lineage",
            "",
            _bullet("Source", f"{data_metadata.get('source')} (`{data_metadata.get('path')}`)"),
            _bullet("Rows loaded", f"{data_metadata.get('rows')}"),
            _bullet("Observed default rate", _fmt(data_metadata.get("default_rate"))),
            _bullet("Content hash", f"`{data_metadata.get('data_hash')}`"),
            _bullet("Licence", str(data_metadata.get("licence"))),
            _bullet("Citation", str(data_metadata.get("citation"))),
            "",
            "### Withheld inputs",
            "",
            "`sex` and `marriage` are prohibited bases for a credit decision and are "
            "excluded from the feature matrix. They are retained only as audit columns, "
            "so a gap the fairness gate reports on `sex` is a gap on an attribute the "
            "model never observed.",
            "",
            "`age` is not withheld. It is used as a predictor, on the standard "
            "fair-lending view that age is a legitimate underwriting input rather than "
            "a prohibited basis, and the fairness gate audits `age_band` on top of that "
            "to check for disparate impact. A gap there is a question about whether that "
            "use is justified, not evidence of a withheld attribute leaking back in.",
            "",
            "## Training setup",
            "",
            _bullet("Train / test rows", f"{result.n_train:,} / {result.n_test:,}"),
            _bullet("Features", str(result.n_features)),
            _bullet(
                "Cross-validation",
                f"{len(result.cv_scores)}-fold stratified, {result.cv_metric} "
                f"{_fmt(result.cv_mean)} +/- {_fmt(result.cv_std)}",
            ),
            _bullet("Random seed", str(self.settings.seed)),
            _bullet("Fit time", f"{result.train_seconds:.2f}s"),
            _bullet(
                "Class weighting",
                "none - rebalancing would break the calibration the ECE and Brier gates measure",
            ),
            "",
            "## Held-out metrics",
            "",
        ]

        lines += _table(
            ["Metric", "Value", "Reading"],
            [
                ["ROC-AUC", _fmt(result.roc_auc), "Rank-ordering power over the whole book"],
                ["PR-AUC", _fmt(result.pr_auc), "Precision-recall area, sensitive to the minority class"],
                ["KS statistic", _fmt(result.ks_statistic), "Max separation between defaulter and non-defaulter score distributions"],
                ["Brier score", _fmt(result.brier_score), "Mean squared probability error (lower is better)"],
                ["Log-loss", _fmt(result.log_loss), "Penalises confident mistakes"],
                ["ECE", _fmt(result.ece), "Population-weighted predicted-vs-observed default-rate gap"],
            ],
        )

        lines += ["", "## Governance gates", ""]
        lines += _table(
            ["Gate", "Verdict", "Headline metric", "Value", "Threshold"],
            [
                [name, gate.symbol, gate.headline_metric, _fmt(gate.headline_value), gate.threshold]
                for name, gate in gates.items()
            ],
        )

        if fairness is not None and fairness.details.get("attributes"):
            lines += ["", "## Fairness detail", ""]
            for attribute, data in fairness.details["attributes"].items():
                lines += [
                    f"### `{attribute}`",
                    "",
                    _bullet("Demographic parity gap", f"{_fmt(data['demographic_parity_gap'])} ({data['demographic_parity_status']})"),
                    _bullet("Equalized odds gap", f"{_fmt(data['equalized_odds_gap'])} ({data['equalized_odds_status']})"),
                    "",
                ]
                lines += _table(
                    ["Group", "n", "Predicted default rate", "Observed default rate", "TPR", "FPR", "ROC-AUC"],
                    [
                        [
                            f"`{group}`",
                            f"{stats['n']:,}",
                            _fmt(stats["predicted_default_rate"]),
                            _fmt(stats["observed_default_rate"]),
                            _fmt(stats["true_positive_rate"]),
                            _fmt(stats["false_positive_rate"]),
                            _fmt(stats["roc_auc"]),
                        ]
                        for group, stats in data["groups"].items()
                    ],
                )
                lines.append("")

        if shap_result and shap_result.get("status") == "success" and shap_result.get("model") == result.model_name:
            lines += ["## Explainability", ""]
            lines += [
                f"{shap_result['explainer']} over {shap_result['sample_size']} evaluation rows. "
                f"Local per-borrower explanations are written alongside this card.",
                "",
            ]
            lines += _table(
                ["Rank", "Feature", "Mean absolute SHAP value"],
                [
                    [str(rank), f"`{name}`", _fmt(value, 5)]
                    for rank, (name, value) in enumerate(
                        list(shap_result["top_features"].items())[:10], start=1
                    )
                ],
            )
            lines.append("")

        lines += ["## Limitations", ""]
        limitations = [
            "Single-jurisdiction 2005 research data; no macroeconomic cycle is represented, "
            "so performance under stress is unobservable here.",
            "The target is default on the *next* payment only - it is not a through-the-cycle "
            "or lifetime PD, and must not be read as one.",
            "`education` and `marriage` category codes 0, 5 and 6 are undocumented upstream "
            "and are folded into the authors' 'other' bucket.",
            "Drift is measured train-vs-test on a random split, which detects sampling "
            "instability but cannot detect true temporal drift; production monitoring needs "
            "a time-ordered reference window.",
        ]
        for gate_name, gate in gates.items():
            if gate.status != "pass":
                for finding in gate.findings[:2]:
                    limitations.append(f"Open `{gate_name}` finding: {finding}")
        lines += [f"- {item}" for item in limitations]

        lines += [
            "",
            "## Reproduction",
            "",
            "```bash",
            "pip install -r requirements.txt",
            "python main.py --sample",
            "```",
            "",
            f"Seeded with `project.random_seed = {self.settings.seed}`; all thresholds in "
            "`config.yaml`.",
            "",
        ]
        return "\n".join(lines)


def _bullet(label: str, value: str) -> str:
    return f"- **{label}:** {value}"
