#!/usr/bin/env python3
"""Financial AI Risk & Governance Framework - pipeline entry point.

Runs the full pipeline end to end: load data, benchmark every enabled model,
fit the empirical-Bayes segment shrinkage model, put each model through the five
governance gates, explain the selected model with SHAP, and write the governance
scorecard and model cards.

Usage
-----
    python main.py --sample          # offline: committed 5,000-row UCI sample
    python main.py                   # live: fetch the full UCI dataset (30,000 rows)
    python main.py --refresh-sample  # re-fetch UCI and rewrite the committed sample
    python main.py --sample --no-shap
    python main.py --help

Author: Aryan Singh
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from financial_ai_framework import (
    CreditDataProcessor,
    ExperimentTracker,
    FeatureImportanceAnalyzer,
    GovernanceReporter,
    ModelBenchmarkSuite,
    ShapAnalyzer,
    ensure_reproducibility,
    load_credit_data,
    load_settings,
    refresh_offline_sample,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Benchmark credit-default models and run them through the governance "
            "gate pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --sample            run offline on the committed sample (~60s)\n"
            "  python main.py                     fetch the full UCI dataset and run\n"
            "  python main.py --refresh-sample    regenerate the committed offline sample\n"
            "  python main.py --sample --no-shap  skip the explainability stage\n"
        ),
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="run against the committed offline sample instead of fetching from UCI",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="path to a config.yaml (defaults to the repository root config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="directory for reports and run logs (defaults to tracking.reports_dir)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["logistic_regression", "xgboost", "lightgbm"],
        default=None,
        help="subset of models to benchmark (default: models.enabled from config)",
    )
    parser.add_argument(
        "--no-shap", action="store_true", help="skip SHAP explainability"
    )
    parser.add_argument(
        "--mlflow", action="store_true", help="log this run to MLflow instead of local JSON"
    )
    parser.add_argument(
        "--refresh-sample",
        action="store_true",
        help="re-fetch the live UCI dataset, rewrite the committed sample, then exit",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="ignore the cached raw dataset and re-fetch from UCI",
    )
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)

    # Command-line overrides, re-validated by pydantic on assignment.
    if args.models:
        settings.models.enabled = args.models
    if args.no_shap:
        settings.explainability.enable_shap = False
    if args.mlflow:
        settings.tracking.enable_mlflow = True
    if args.output:
        settings.tracking.reports_dir = args.output

    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_sample:
        refresh_offline_sample(settings)
        return 0

    run_id = f"{'sample' if args.sample else 'full'}_{datetime.now():%Y%m%d_%H%M%S}"
    header = f" {settings.project.name} v{settings.project.version} "
    print("=" * 96)
    print(header.center(96, "="))
    print("=" * 96)
    print(f"run id      : {run_id}")
    print(f"mode        : {'offline sample' if args.sample else 'live UCI fetch'}")
    print(f"models      : {', '.join(settings.models.enabled)}")
    print(f"reports dir : {reports_dir}")

    ensure_reproducibility(settings.seed)
    tracker = ExperimentTracker(settings, reports_dir)
    tracker.start_run(run_id)

    try:
        print("\n--- 1/6 data ---")
        frame, data_metadata = load_credit_data(
            settings, use_sample=args.sample, refresh=args.refresh_cache
        )
        tracker.log_params(
            {
                "run_mode": "sample" if args.sample else "full",
                "data_source": data_metadata["source"],
                "data_hash": data_metadata["data_hash"],
                "data_rows": data_metadata["rows"],
                "random_seed": settings.seed,
            }
        )
        tracker.log_metric("data_default_rate", data_metadata["default_rate"])

        print("\n--- 2/6 features and split ---")
        dataset = CreditDataProcessor(settings).prepare(frame)
        tracker.log_params(
            {
                "n_train": len(dataset.X_train),
                "n_test": len(dataset.X_test),
                "n_features": dataset.n_features,
            }
        )

        print("\n--- 3/6 benchmark and governance gates ---")
        suite = ModelBenchmarkSuite(settings, tracker)
        results = suite.run(dataset)

        print("\n--- 4/6 bayesian segment shrinkage ---")
        shrinkage_summary: dict[str, Any] = (
            suite.shrinkage.summary() if suite.shrinkage else {}
        )
        if suite.shrinkage is not None:
            most_shrunken = suite.shrinkage.most_shrunken(3)
            print(
                f"[bayes] {shrinkage_summary['n_sparse_segments']} of "
                f"{shrinkage_summary['n_segments']} segments sit below the "
                f"{settings.segments.min_segment_size}-borrower prior-fitting floor"
            )
            for row in most_shrunken.itertuples(index=False):
                print(
                    f"[bayes] {row.segment_id}: raw {row.empirical_rate:.3f} "
                    f"(n={row.n}) -> posterior {row.posterior_mean:.3f} "
                    f"[{row.ci_lower:.3f}, {row.ci_upper:.3f}] "
                    f"(own-data weight {row.shrinkage_weight_on_own_data:.2f})"
                )
            tracker.log_metrics(
                {
                    "shrinkage_prior_alpha": shrinkage_summary["prior"]["alpha"],
                    "shrinkage_prior_beta": shrinkage_summary["prior"]["beta"],
                    "shrinkage_population_rate": shrinkage_summary["prior"]["population_rate"],
                    "shrinkage_median_own_data_weight": shrinkage_summary[
                        "median_shrinkage_weight"
                    ],
                }
            )

        print("\n--- 5/6 explainability ---")
        best = results[0]
        importance = FeatureImportanceAnalyzer(settings, tracker).analyse(
            suite.models, dataset.feature_names
        )
        if importance.get("status") == "success":
            top = list(importance["top_features"].items())[:5]
            print(
                "[importance] cross-model consensus: "
                + ", ".join(f"{name} ({value:.4f})" for name, value in top)
            )

        shap_result: dict[str, Any] = {"status": "disabled"}
        if settings.explainability.enable_shap:
            shap_result = ShapAnalyzer(settings, tracker).analyse(
                model=suite.models[best.model_name],
                model_name=best.model_name,
                X_background=dataset.X_train,
                X_sample=dataset.X_test,
                y_prob=suite.predictions[best.model_name],
                reports_dir=reports_dir,
            )
        else:
            print("[shap] skipped (--no-shap)")

        print("\n--- 6/6 reporting ---")
        reporter = GovernanceReporter(settings, tracker, reports_dir)
        bundle = reporter.generate(
            results=results,
            data_metadata=data_metadata,
            shrinkage_summary=shrinkage_summary,
            shap_result=shap_result,
            importance_result=importance,
            run_id=run_id,
        )

        overall = suite.worst_gate_status()
        tracker.log_param("overall_gate_status", overall)
        reporter.print_summary(results, overall)

        print("\nArtefacts:")
        for path in bundle.all_paths():
            print(f"  {path}")
        for path in shap_result.get("artifacts", []):
            print(f"  {path}")

        # A failed governance gate is a real signal, not a crash: exit 0 so CI's
        # smoke job verifies the pipeline, and read the verdict from the scorecard.
        return 0

    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        tracker.end_run()


def main(argv: list[str] | None = None) -> int:
    return run_pipeline(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
