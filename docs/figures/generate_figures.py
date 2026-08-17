#!/usr/bin/env python3
"""Generate the figures used in docs/whitepaper.tex.

Every figure is drawn from a real run of this repository. Nothing is mocked.

Two of the figures re-execute pipeline code directly:

* the architecture diagram is drawn from a hand-maintained description of the
  pipeline stages (it is a schematic, so there is nothing to compute);
* the segment-shrinkage figure calls ``load_credit_data`` ->
  ``CreditDataProcessor`` -> ``SegmentShrinkageModel.fit`` on the committed
  offline sample, exactly as ``ModelBenchmarkSuite`` does, and plots the
  posteriors it returns.

The other two read the artefacts that ``python main.py --sample`` writes into
``reports/``:

* the benchmark bar chart reads ``reports/governance_scorecard.json``;
* the SHAP figure reads ``reports/shap_global_importance_<model>.json`` for the
  model the selection metric picked on that run.

So the order is: run ``python main.py --sample`` first, then this script.

    python main.py --sample
    python docs/figures/generate_figures.py

Each figure is written twice, as PDF (vector, embedded in the paper) and as PNG
(so it renders inline when browsing the repository on GitHub).

Author: Aryan Singh
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REPORTS_DIR = REPO_ROOT / "reports"

sys.path.insert(0, str(REPO_ROOT))

# Muted palette, consistent across all four figures.
INK = "#1b1b1b"
POSTERIOR = "#31708e"
RAW = "#c1622d"
ACCENT = "#7a7a7a"
FAIL = "#a8322d"
PANEL_FILL = "#eef2f5"
PANEL_EDGE = "#5f7d8c"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#4a4a4a",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#d8d8d8",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    """Write one figure as both PDF and PNG."""
    for suffix in ("pdf", "png"):
        path = HERE / f"{stem}.{suffix}"
        fig.savefig(path)
        print(f"  wrote {path.relative_to(REPO_ROOT)}")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 1: architecture
# --------------------------------------------------------------------------

def figure_architecture() -> None:
    """Box-and-arrow schematic of the pipeline stages.

    Boxes are sized from their own content rather than hand-placed, so text
    cannot spill past a border when a label is edited. Labels are deliberately
    terse: the figure carries the shape of the pipeline, and the prose carries
    what each stage does.
    """
    # Point sizes are chosen against the width this figure is included at in
    # whitepaper.tex, so that the smallest label still sets around 6.5pt.
    TITLE_H, LINE_H = 0.44, 0.29
    PAD_TOP, PAD_BOT, GAP = 0.16, 0.20, 0.44
    BODY_PT, TITLE_PT = 9.5, 10.5

    fig, ax = plt.subplots(figsize=(7.2, 6.48))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9.0)
    ax.axis("off")
    ax.grid(False)

    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def measure(text, pt, weight):
        """Rendered width of ``text`` in data units."""
        probe = ax.text(0, 0, text, fontsize=pt, fontweight=weight)
        extent = probe.get_window_extent(renderer)
        probe.remove()
        (x0, _), (x1, _) = inv.transform([(0, 0), (extent.width, 0)])
        return x1 - x0

    def fit(text, budget, start_pt, weight="normal", floor=5.5):
        """Largest point size at or below ``start_pt`` that fits ``budget``."""
        pt = start_pt
        while pt > floor and measure(text, pt, weight) > budget:
            pt -= 0.25
        return pt

    def box(
        x, w, top, title, lines=(), pairs=(),
        fill="#ffffff", edge=ACCENT, lw=0.9, name_dx=0.30,
        title_size=TITLE_PT,
    ):
        """Draw a content-sized box; return its bottom edge.

        Font sizes are measured against the box width and stepped down until
        they fit, so a label can be reworded without silently overflowing.
        """
        n_lines = len(lines) + len(pairs)
        height = PAD_TOP + TITLE_H + LINE_H * n_lines + PAD_BOT
        bottom = top - height

        ax.add_patch(
            FancyBboxPatch(
                (x, bottom), w, height,
                boxstyle="round,pad=0.02,rounding_size=0.09",
                linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2,
            )
        )
        ax.text(
            x + w / 2, top - PAD_TOP - TITLE_H / 2, title,
            ha="center", va="center",
            fontsize=fit(title, w - 0.24, title_size, "bold"),
            fontweight="bold", color=INK, zorder=3,
        )

        cursor = top - PAD_TOP - TITLE_H
        row = 0

        for line in lines:
            ax.text(
                x + w / 2, cursor - LINE_H * (row + 0.5), line,
                ha="center", va="center",
                fontsize=fit(line, w - 0.28, BODY_PT), color="#3d3d3d",
                zorder=3,
            )
            row += 1

        if pairs:
            # One point size for the whole block, and a description column
            # placed off the widest gate name so the two columns stay aligned.
            pt = min(
                min(fit(n, (w - 0.6) * 0.42, BODY_PT, "bold") for n, _ in pairs),
                min(fit(d, (w - 0.6) * 0.55, BODY_PT) for _, d in pairs),
            )
            desc_dx = name_dx + max(
                measure(n, pt, "bold") for n, _ in pairs
            ) + 0.34

            for name, desc in pairs:
                y = cursor - LINE_H * (row + 0.5)
                ax.text(
                    x + name_dx, y, name, ha="left", va="center", fontsize=pt,
                    color=PANEL_EDGE, fontweight="bold", zorder=3,
                )
                ax.text(
                    x + desc_dx, y, desc, ha="left", va="center", fontsize=pt,
                    color="#3d3d3d", zorder=3,
                )
                row += 1

        return bottom

    def arrow(start, end, colour=INK, lw=1.0):
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=10, linewidth=lw,
                color=colour, shrinkA=1, shrinkB=1, zorder=4,
            )
        )

    LEFT, FULL_W, HALF_W, THIRD_W = 0.55, 8.90, 4.15, 2.83

    # Row 1: configuration, read by every stage below it.
    y = box(
        LEFT, FULL_W, 8.90, "config.yaml",
        lines=("thresholds and seeds, validated at startup",),
        fill="#f4f0e6", edge="#a08a5b",
    )

    # Row 2: data.
    top = y - GAP
    box(
        LEFT, HALF_W, top, "data/loader.py",
        lines=("UCI 350 fetch + cache",),
    )
    y = box(
        5.30, HALF_W, top, "data/processor.py",
        lines=("49 predictors, audit columns",),
    )
    arrow((LEFT + HALF_W, top - 0.50), (5.30, top - 0.50))

    # Row 3: the two things the gates consume. Both read the processed split.
    top = y - GAP
    bayes_w = 3.70
    box(
        LEFT, bayes_w, top, "bayes/segment_shrinkage.py",
        lines=("Beta-Binomial empirical Bayes",
               "posterior + 95% CI per segment"),
        fill="#eef3ec", edge="#6a8a5f", title_size=9.6,
    )
    bench_x, bench_w = 4.45, 5.00
    y = box(
        bench_x, bench_w, top, "models/benchmark.py",
        lines=("3 models, 5-fold stratified CV",
               "ROC-AUC / PR-AUC / KS / Brier / ECE"),
    )
    arrow((7.40, top + GAP), (bench_x + bench_w / 2, top))
    arrow((5.60, top + GAP), (LEFT + bayes_w / 2, top))

    # Row 4: the gate panel.
    top = y - GAP
    y = box(
        LEFT, FULL_W, top, "governance/  -  five gates, one verdict per model",
        pairs=(
            ("fairness", "parity + equalized odds"),
            ("drift", "PSI, train vs test"),
            ("calibration", "ECE, MCE, Brier"),
            ("uncertainty", "entropy, bootstrap CI"),
            ("segment_stability", "PD vs credible interval"),
        ),
        fill=PANEL_FILL, edge=PANEL_EDGE, lw=1.1,
    )
    arrow((bench_x + bench_w / 2, top + GAP), (6.60, top))
    arrow((LEFT + bayes_w / 2, top + GAP), (2.60, top), colour="#5f7d5a")

    # Row 5: artefacts and tracking.
    top = y - GAP
    box(
        LEFT, THIRD_W, top, "explainability/shap.py",
        lines=("exact Tree / Linear SHAP",), title_size=9.6,
    )
    box(
        3.59, THIRD_W, top, "governance/reporter.py",
        lines=("scorecard + model cards",), title_size=9.6,
    )
    box(
        6.62, THIRD_W, top, "utils/tracking.py",
        lines=("MLflow or JSON run log",), title_size=9.6,
        fill="#f2f2f4", edge="#8a8a94",
    )
    arrow((3.20, top + GAP), (LEFT + THIRD_W / 2, top))
    arrow((6.20, top + GAP), (3.59 + THIRD_W / 2, top))
    arrow((8.10, top + GAP), (6.62 + THIRD_W / 2, top), colour=ACCENT, lw=0.85)

    save(fig, "fig_architecture")


# --------------------------------------------------------------------------
# Figure 2: empirical-Bayes segment shrinkage
# --------------------------------------------------------------------------

SHORT_EDUCATION = {
    "graduate_school": "grad",
    "university": "univ",
    "high_school": "high sch",
    "other_unknown": "other",
}


def short_label(segment_id: str) -> str:
    if "|" in segment_id:
        education, age = (part.strip() for part in segment_id.split("|", 1))
        return f"{SHORT_EDUCATION.get(education, education)} | {age}"
    return segment_id


def figure_segment_shrinkage() -> dict:
    """Fit the real shrinkage model and plot posteriors against raw rates."""
    from financial_ai_framework import (
        CreditDataProcessor,
        ensure_reproducibility,
        load_credit_data,
        load_settings,
    )
    from financial_ai_framework.bayes.segment_shrinkage import SegmentShrinkageModel

    settings = load_settings()
    ensure_reproducibility(settings.seed)

    frame, _ = load_credit_data(settings, use_sample=True)
    dataset = CreditDataProcessor(settings).prepare(frame)

    # Same call ModelBenchmarkSuite makes: prior and posteriors off the training split.
    model = SegmentShrinkageModel(settings).fit(
        dataset.audit_train["segment_id"], dataset.y_train
    )
    prior = model.prior
    assert prior is not None

    table = model.to_frame().sort_values("n").reset_index(drop=True)

    x = np.arange(len(table))
    posterior = table["posterior_mean"].to_numpy()
    raw = table["empirical_rate"].to_numpy()
    lower = posterior - table["ci_lower"].to_numpy()
    upper = table["ci_upper"].to_numpy() - posterior

    fig, ax = plt.subplots(figsize=(6.3, 4.1))

    # Shade the segments below the prior-fitting floor. Sorted by n, so leftmost.
    n_sparse = int((~table["used_for_prior"]).sum())
    if n_sparse:
        ax.axvspan(-0.6, n_sparse - 0.4, color="#f6ece4", zorder=0)
        ax.text(
            (n_sparse - 1) / 2,
            0.545,
            f"$n < {settings.segments.min_segment_size}$\n"
            "below prior-fitting floor",
            ha="center", va="top", fontsize=7, color="#8a5a3a", linespacing=1.4,
        )

    # Movement from the raw rate to the posterior mean.
    for xi, r, p in zip(x, raw, posterior, strict=True):
        ax.plot([xi, xi], [r, p], color=ACCENT, linewidth=0.7, alpha=0.55,
                zorder=1)

    ax.axhline(
        prior.population_rate, color=FAIL, linewidth=1.0,
        linestyle=(0, (5, 3)), zorder=2,
        label=f"population rate {prior.population_rate:.4f} "
              rf"$= \alpha/(\alpha+\beta)$",
    )

    ax.scatter(
        x, raw, s=26, marker="x", color=RAW, linewidths=1.2, zorder=3,
        label="raw empirical rate $k_i/n_i$",
    )
    ax.errorbar(
        x, posterior, yerr=[lower, upper], fmt="o", markersize=4.2,
        color=POSTERIOR, ecolor=POSTERIOR, elinewidth=1.1, capsize=2.6,
        capthick=1.0, zorder=4,
        label=r"posterior mean $E[\theta_i\,|\,k_i,n_i]$, 95% credible interval",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([short_label(s) for s in table["segment_id"]],
                       rotation=90)
    ax.set_ylabel("segment default rate")

    # Segment sizes go on a top axis: they are short enough to set horizontally
    # there, which keeps them legible without deepening the rotated labels below.
    top = ax.secondary_xaxis("top")
    top.set_xticks(x)
    top.set_xticklabels(table["n"], fontsize=6.6)
    top.set_xlabel("segment size $n$", fontsize=8, labelpad=2)
    top.tick_params(length=2, pad=1.5)
    ax.set_ylim(-0.03, 0.56)
    ax.set_xlim(-0.6, len(table) - 0.4)
    ax.set_title(
        f"Empirical-Bayes shrinkage over {len(table)} education "
        rf"$\times$ age-band segments"
        "\n"
        rf"prior Beta({prior.alpha:.2f}, {prior.beta:.2f}), "
        f"worth {prior.concentration:.0f} equivalent borrowers",
        fontsize=9.2, pad=22,
    )
    ax.legend(loc="upper right", fontsize=7.6)
    ax.set_axisbelow(True)

    save(fig, "fig_segment_shrinkage")

    thinnest = table.iloc[0]
    print(
        f"  [check] thinnest segment {thinnest['segment_id']!r}: "
        f"n={thinnest['n']} raw={thinnest['empirical_rate']:.3f} "
        f"-> posterior {thinnest['posterior_mean']:.3f} "
        f"[{thinnest['ci_lower']:.3f}, {thinnest['ci_upper']:.3f}] "
        f"own-data weight {thinnest['shrinkage_weight_on_own_data']:.2f}"
    )
    return {
        "alpha": prior.alpha,
        "beta": prior.beta,
        "concentration": prior.concentration,
        "population_rate": prior.population_rate,
        "n_segments": len(table),
        "n_sparse": n_sparse,
    }


# --------------------------------------------------------------------------
# Figure 3: benchmark ROC-AUC
# --------------------------------------------------------------------------

def load_scorecard() -> dict:
    path = REPORTS_DIR / "governance_scorecard.json"
    if not path.exists():
        raise SystemExit(
            f"{path.relative_to(REPO_ROOT)} not found. "
            "Run `python main.py --sample` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def figure_benchmark(scorecard: dict) -> None:
    """ROC-AUC per model, with the cross-validation spread alongside it."""
    models = sorted(
        scorecard["models"], key=lambda m: m["metrics"]["roc_auc"], reverse=True
    )
    names = [m["model_name"] for m in models]
    test_auc = [m["metrics"]["roc_auc"] for m in models]
    cv_mean = [m["metrics"]["cv_mean"] for m in models]
    cv_std = [m["metrics"]["cv_std"] for m in models]

    fig, ax = plt.subplots(figsize=(4.6, 2.95))
    x = np.arange(len(names))

    bars = ax.bar(x, test_auc, width=0.46, color=POSTERIOR, zorder=2,
                  label="test ROC-AUC")
    for bar, value in zip(bars, test_auc, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.4f}",
            ha="center", va="bottom", fontsize=8, color=INK,
        )

    ax.errorbar(
        x + 0.30, cv_mean, yerr=cv_std, fmt="D", markersize=4,
        color=RAW, ecolor=RAW, elinewidth=1.1, capsize=3.2, capthick=1.0,
        zorder=3, label=r"5-fold CV mean $\pm$ 1 s.d.",
    )

    # 0.50 reference line. Left unlabelled in the plot: at the width this figure
    # is included at there is no clear space for a label. The caption names it.
    ax.axhline(0.5, color=ACCENT, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)

    spread = max(test_auc) - min(test_auc)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names])
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.48, 0.82)
    ax.set_xlim(-0.5, len(names) - 0.35)
    ax.set_title(
        "Benchmark discrimination, committed offline sample\n"
        f"test spread across models {spread:.4f}, "
        f"against a fold-to-fold s.d. of {min(cv_std):.4f}-{max(cv_std):.4f}",
        fontsize=9.2, pad=24,
    )
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.6,
        borderaxespad=0.4,
    )
    ax.set_axisbelow(True)

    save(fig, "fig_benchmark_roc_auc")
    print(
        "  [check] "
        + " | ".join(
            f"{n} {a:.4f} (CV {m:.4f}+/-{s:.4f})"
            for n, a, m, s in zip(names, test_auc, cv_mean, cv_std, strict=True)
        )
    )


# --------------------------------------------------------------------------
# Figure 4: SHAP global attribution
# --------------------------------------------------------------------------

def figure_shap(scorecard: dict, top_n: int = 15) -> None:
    """Top global SHAP attributions for the model the selection metric picked."""
    selected = scorecard["selected_model"]
    path = REPORTS_DIR / f"shap_global_importance_{selected}.json"
    if not path.exists():
        raise SystemExit(
            f"{path.relative_to(REPO_ROOT)} not found. "
            "Run `python main.py --sample` (without --no-shap) first."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    ranked = list(payload["global_importance"].items())[:top_n][::-1]
    names = [name for name, _ in ranked]
    values = [value for _, value in ranked]

    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    ax.barh(names, values, height=0.66, color=POSTERIOR, zorder=2)
    for name, value in zip(names, values, strict=True):
        ax.text(value + max(values) * 0.012, name, f"{value:.3f}",
                va="center", ha="left", fontsize=7.2, color=INK)

    ax.set_xlabel("mean $|$SHAP value$|$  (log-odds)")
    ax.set_xlim(0, max(values) * 1.16)
    ax.set_title(
        f"Global feature attribution, {selected} "
        f"({payload['explainer']})\n"
        f"top {len(ranked)} of {len(payload['global_importance'])} predictors, "
        f"{payload['sample_size']} evaluation rows",
        fontsize=9.2,
    )
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    save(fig, "fig_shap_importance")
    print(
        f"  [check] {selected} top drivers: "
        + ", ".join(f"{n} ({v:.4f})" for n, v in ranked[::-1][:4])
    )


# --------------------------------------------------------------------------

def main() -> int:
    set_style()

    print("figure 1/4  architecture diagram")
    figure_architecture()

    print("figure 2/4  empirical-Bayes segment shrinkage (re-fits the model)")
    figure_segment_shrinkage()

    scorecard = load_scorecard()
    print(
        f"loaded reports/governance_scorecard.json  "
        f"(run {scorecard['run_id']}, data hash {scorecard['dataset']['data_hash']}, "
        f"selected {scorecard['selected_model']}, "
        f"overall {scorecard['overall_status'].upper()})"
    )

    print("figure 3/4  benchmark ROC-AUC")
    figure_benchmark(scorecard)

    print("figure 4/4  SHAP global attribution")
    figure_shap(scorecard)

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
