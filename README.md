# Financial AI Risk & Governance Framework

[![CI](https://github.com/asynced24/financial-ai-risk-governance-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/asynced24/financial-ai-risk-governance-framework/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

Benchmarks credit-default models against each other and then puts every one of them
through five governance gates - fairness, drift, calibration, uncertainty, and a
Bayesian segment-stability check - emitting a governance scorecard and a model card
per model.

A technical writeup covering the methodology, the shrinkage math, and the
governance findings, with generated figures, is at
[`docs/whitepaper.pdf`](docs/whitepaper.pdf).

## Why

A credit-risk model that ranks well is not the same as a credit-risk model you can
deploy. The number has to mean what it says, it has to hold up across demographic
segments, and someone has to be able to explain any individual decision. In
practice those questions usually get asked in a review meeting weeks after
training, and get answered from a deck rather than from a check that runs.

This project treats them as gates in the pipeline instead. Every threshold lives in
`config.yaml`, is validated by pydantic at startup, and produces a pass/warn/fail
verdict on every run. Three things follow from that:

- **A model can win on AUC and still be blocked.** Selection and approval are
  separate steps here, as they should be.
- **Thin demographic segments are handled properly.** A cell with 17 borrowers and
  one default has a raw default rate of 5.9%, which is noise. The empirical-Bayes
  model shrinks it toward the population rate by an amount that falls out of the
  posterior, not out of a tuned constant.
- **Fairness is measured on an attribute the model never saw.** `sex` and
  `marriage` are withheld from the feature matrix entirely as prohibited bases and
  kept only for auditing. The logistic regression model's predictions still show an
  equalized-odds gap of **0.1617** on `sex`, past the 0.10 fail limit - real
  evidence that fairness is not fixed by removing protected columns. `age`, unlike
  sex or marital status, is used directly as a feature - 8th to 12th of 49 by SHAP
  importance depending on the model - and is separately audited via `age_band`; a
  gap there reflects age being used as intended, not a proxy leak.

The sample run ends in an overall **FAIL** verdict. That is the intended result, not
a broken build: a governance harness tuned so that everything passes is not
measuring anything. The findings are real and reproducible, and they are discussed
below.

## Architecture

```
                      config.yaml  (pydantic v2 Settings, validated at startup)
                           |
                           v  thresholds read by every stage
   data/loader.py ---> data/processor.py ---> models/benchmark.py
   UCI 350 fetch          feature eng.          LogisticRegression
   + local cache          + train/test          XGBClassifier
   + offline sample       + audit cols          LGBMClassifier
                                               5-fold stratified CV
                                                     |
                                    ROC-AUC / PR-AUC / KS / Brier
                                    log-loss / ECE   |
                                                     v
                                      +--------------------------------+
                                      |    governance gate pipeline    |
                                      |  fairness  (DP + equalized odds)|
                                      |  drift     (PSI train vs test)  |
                                      |  calibration (ECE, Brier)       |
                                      |  uncertainty (entropy, AUC CI)  |
                                      |  segment_stability  <-----------+---- bayes/
                                      +--------------------------------+      segment_
                                                     |                        shrinkage.py
                                     pass / warn / fail per gate               (Beta-Binomial
                                                     |                          empirical Bayes)
                              +----------------------+---------------+
                              v                                      v
                  explainability/shap.py                  governance/reporter.py
                  global attribution                      governance_scorecard.md
                  + 3 local explanations                  governance_scorecard.json
                              |                           model_card_<model>.md
                              +-------------+-------------+
                                            v
                                    utils/tracking.py
                              MLflow, or local JSON run log
```

## Dataset and licence

UCI Machine Learning Repository dataset **350 - "Default of Credit Card Clients"**:
30,000 Taiwanese credit-card customers, 23 features covering six months of billing
and repayment history plus limited demographics, and a binary target for default on
the next monthly payment. Base default rate 22.1%.

> Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the
> predictive accuracy of probability of default of credit card clients.
> *Expert Systems with Applications*, 36(2), 2473-2480.
> <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>

Distributed by UCI under **CC BY 4.0**.

The upstream columns are named `X1`...`X23`/`Y`; `data/loader.py` renames them
positionally to the canonical names from the source paper and folds the undocumented
`education`/`marriage` codes (0, 5, 6) into the authors' "other" category.

**Committed offline sample.** `data/sample/uci_credit_sample.csv` is a 5,000-row
class-stratified draw (448 KB, base rate 22.12% against the full set's 22.12%),
redistributed under CC BY 4.0 with the attribution above. It exists so the whole
pipeline runs with no network call - which is what makes the CI smoke job immune to
a UCI outage, and what lets anyone reproduce the numbers below in one command.
Regenerate it with `python main.py --refresh-sample`.

## Quickstart

Requires Python 3.12 or newer - the pinned numpy, scipy, xgboost and shap releases
all declare `Requires-Python >= 3.12`.

```bash
git clone https://github.com/asynced24/financial-ai-risk-governance-framework.git
cd financial-ai-risk-governance-framework

pip install -r requirements.txt
python main.py --sample
```

Runs end to end in roughly 15-20 seconds on a laptop, most of which is importing
`shap` and `mlflow`. No network, no credentials, no configuration. Reports land in
`reports/`.

```bash
python main.py                        # full 30,000-row dataset, fetched from UCI and cached
python main.py --sample --no-shap     # skip explainability
python main.py --models xgboost       # benchmark a subset
python main.py --mlflow               # log to MLflow instead of the local JSON run log
python main.py --refresh-sample       # re-fetch UCI and rewrite the committed sample
python main.py --help
```

## Sample output

Real output from `python main.py --sample`, seed 42, sample hash `b1ba275532b83033`,
3,750 train / 1,250 test rows, 49 predictors (28 engineered from billing history).
The table below is the CI run on Linux / Python 3.13 - download the
`governance-reports` artifact from any green build to check it.

| Model | ROC-AUC | PR-AUC | KS | Brier | Log-loss | ECE | CV mean | CV std | Gates |
|---|---|---|---|---|---|---|---|---|---|
| `xgboost` | 0.7672 | 0.5420 | 0.4251 | 0.1372 | 0.4452 | 0.0320 | 0.7632 | 0.0221 | FAIL |
| `lightgbm` | 0.7634 | 0.5355 | 0.4220 | 0.1374 | 0.4469 | 0.0411 | 0.7671 | 0.0179 | FAIL |
| `logistic_regression` | 0.7607 | 0.5042 | 0.4155 | 0.1403 | 0.4459 | 0.0333 | 0.7511 | 0.0224 | FAIL |

Gate matrix:

| Gate | `xgboost` | `lightgbm` | `logistic_regression` |
|---|---|---|---|
| fairness | FAIL | FAIL | FAIL |
| drift | PASS | PASS | PASS |
| calibration | WARN | WARN | WARN |
| uncertainty | PASS | PASS | PASS |
| segment_stability | FAIL | FAIL | WARN |

The top two models are separated by 0.0038 of ROC-AUC against a fold-to-fold CV std
of 0.018-0.022. Treating that as a ranking would be overreading it; the useful
signal in this run is the gate matrix, not the leaderboard.

The worked examples below all use **LightGBM**, whose numbers are bit-identical on
Windows and Linux (see the reproducibility note). Supporting figures for it: max
feature PSI 0.0156 across all 49 features (no drift, as expected on a random split);
17.8% of the test set above the 0.60-nat entropy threshold; ROC-AUC 95% bootstrap
interval [0.7022, 0.8097]. Top SHAP drivers: `pay_status_1` (0.359),
`max_delinquency` (0.152), `limit_bal` (0.135), `delinquent_months` (0.122) - most
recent repayment status dominates, which is what the source paper found too. SHAP
runs on whichever model the selection metric picks, so the committed CI artifact
holds XGBoost's attributions rather than these.

### Reproducibility note

Across Windows and Linux, on the same seed and the same sample hash:

- `logistic_regression` and `lightgbm` reproduce **bit-identically** - every metric,
  every gate verdict, every credible interval.
- The Bayesian layer reproduces bit-identically everywhere; it is pure numpy/scipy.
- `xgboost` does **not**. Its histogram tree builder reduces floating-point sums in a
  thread- and platform-dependent order, so the same seed gives ROC-AUC 0.7672 on
  Linux and 0.7625 on Windows (ECE 0.0320 vs 0.0438).

That 0.0047 spread is larger than the 0.0038 gap between the top two models, so
**which model "wins" depends on the machine you run it on**: XGBoost on Linux,
LightGBM on Windows. Every gate verdict in the matrix above is identical on both,
which is the reassuring part - the approval decision is stable even where the
leaderboard is not. If you need a stable winner rather than a stable verdict, pin
the platform or set `models.enabled` to a single model in `config.yaml`.

### A flagged example: fair on parity, unfair on error rates

Both audited attributes pass demographic parity and fail equalized odds, and the
way they fail is the point. Taking `age_band` on LightGBM first: predicted default
rate is nearly flat across age bands - a demographic parity gap of just **0.0192**,
comfortably passing. But recall on borrowers who actually defaulted is not flat at
all:

| Age band | n | Predicted default rate | Observed default rate | TPR | FPR |
|---|---|---|---|---|---|
| 20-30 | 396 | 0.1162 | 0.2222 | 0.3409 | 0.0519 |
| 30-40 | 479 | 0.1211 | 0.2088 | **0.4100** | 0.0449 |
| 40-50 | 257 | 0.1128 | 0.2529 | 0.3538 | 0.0312 |
| 50-60 | 108 | 0.1019 | 0.1852 | **0.2500** | 0.0682 |

Equalized odds gap **0.1600**, against a fail limit of 0.10, driven entirely by the
TPR spread. A defaulting 50-59-year-old is caught 25% of the time; a defaulting
30-39-year-old, 41%. A parity-only fairness check would have signed this model off.

`age` is a model feature here - 12th of 49 by SHAP importance on LightGBM, 8th on
XGBoost - so this gap is at least partly the model using an input it was handed.
That makes it a question about whether the use is justified, not proof of a hidden
proxy.

The withheld-attribute case is `sex`, and the sharper result there is on
**logistic regression** rather than LightGBM:

| `sex` | n | Predicted default rate | Observed default rate | TPR | FPR |
|---|---|---|---|---|---|
| male (1) | 495 | 0.1333 | 0.2283 | **0.4071** | 0.0524 |
| female (2) | 755 | 0.0874 | 0.2159 | **0.2454** | 0.0439 |

Demographic parity gap 0.0459, passing. Equalized odds gap **0.1617**, failing,
again on the TPR spread alone - the FPR spread is 0.0084. `sex` is not in the
feature matrix and neither is `marriage`, so the model produced that gap without
ever reading the column; the effect arrives through correlated repayment behaviour.
That is the case excluding a protected column does not cover, and the reason the
gate audits withheld attributes instead of treating exclusion as sufficient.
Logistic regression is also one of the two models that reproduce bit-identically
across platforms, so this figure is not machine-dependent.

The 60-80 age band is excluded from the gate: 10 test rows is below the configured
`min_group_size` of 50, and reporting a TPR on that is worse than reporting nothing.

### What the Bayesian shrinkage does

Fitted prior across 19 segments (`education` x age-band): **Beta(88.87, 309.62)**,
population default rate 0.2230, prior worth 398 equivalent borrowers. Method of
moments, with the binomial sampling variance subtracted from the observed spread of
segment rates first - skip that step and the prior comes out too diffuse and
under-shrinks.

7 of the 19 segments sit below the 30-borrower floor for prior fitting. They still
get posteriors, which is the entire point:

| Segment | n | Raw rate | Posterior mean | 95% credible interval | Weight on own data |
|---|---|---|---|---|---|
| `other_unknown \| 50-60` | 6 | 0.000 | 0.220 | [0.181, 0.261] | 0.01 |
| `other_unknown \| 30-40` | 18 | 0.056 | 0.216 | [0.178, 0.256] | 0.04 |
| `other_unknown \| 20-30` | 17 | 0.059 | 0.216 | [0.178, 0.257] | 0.04 |

A raw 0% default rate on six borrowers becomes a posterior mean of 0.220 with an
honest interval. The 0.01 weight is not configured anywhere - it is
`n / (n + a + b)` = `6 / (6 + 398)`.

The stability gate then compares each model's mean predicted PD per segment against
these intervals. For LightGBM, 4 of 11 gated segments fall outside: it prices
`graduate_school | 30-40` at a mean PD of 0.163 against a shrunken posterior of 0.213
with interval [0.187, 0.239], built from 560 reference borrowers. The model
systematically under-prices default risk for better-educated mid-career borrowers
relative to their actual default experience. That is a provisioning problem that no
aggregate metric on the leaderboard surfaces.

## Tech stack

| Area | Choice |
|---|---|
| Models | scikit-learn `LogisticRegression`, `xgboost`, `lightgbm` (all required) |
| Bayesian layer | numpy + scipy only - no extra modelling dependency |
| Config | pydantic v2 `Settings` loaded from `config.yaml`, `extra="forbid"` |
| Explainability | `shap` exact TreeExplainer / LinearExplainer |
| Tracking | MLflow, with a local JSON run log as fallback |
| Tests | pytest |
| Lint | ruff |
| CI | GitHub Actions: lint, tests on 3.12/3.13, end-to-end smoke |

Class weights are left at their natural values throughout. Rebalancing would lift
ROC-AUC and destroy the calibration that the ECE and Brier gates exist to measure,
and a probability of default that no longer means "probability of default" is not
usable for provisioning.

## Project layout

```
.
|-- main.py                          CLI entry point
|-- config.yaml                      every governance threshold, validated at startup
|-- pyproject.toml                   ruff + pytest configuration
|-- requirements.txt                 runtime deps, exact pins
|-- requirements-dev.txt             pytest, ruff
|-- data/
|   |-- raw/                         UCI fetch cache (gitignored)
|   `-- sample/uci_credit_sample.csv 5,000-row stratified offline sample (committed)
|-- reports/                         generated scorecards, model cards, SHAP (gitignored)
|-- financial_ai_framework/
|   |-- config.py                    pydantic v2 Settings + GateThreshold
|   |-- bayes/
|   |   `-- segment_shrinkage.py     SegmentShrinkageModel, SegmentStabilityCheck
|   |-- data/
|   |   |-- loader.py                UCI fetch, cache, stratified sample, segments
|   |   `-- processor.py             feature engineering, prohibited-basis exclusion
|   |-- models/
|   |   `-- benchmark.py             ModelBenchmarkSuite, BenchmarkResult
|   |-- governance/
|   |   |-- gates.py                 GateResult, status aggregation
|   |   |-- fairness.py              demographic parity, equalized odds
|   |   |-- drift.py                 Population Stability Index
|   |   |-- calibration.py           ECE, MCE, Brier (single source of truth for ECE)
|   |   |-- uncertainty.py           predictive entropy, bootstrap AUC interval
|   |   `-- reporter.py              GovernanceReporter: scorecard + model cards
|   |-- explainability/
|   |   |-- shap.py                  global attribution + local explanations
|   |   `-- feature_importance.py    cross-model native importance consensus
|   `-- utils/
|       `-- tracking.py              ExperimentTracker (MLflow / local JSON)
|-- tests/                           pytest suite, one module per component
`-- .github/workflows/ci.yml         lint, test matrix, end-to-end smoke
```

## Testing and CI

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
ruff check .
```

Three CI jobs run on every push and pull request:

1. **lint** - `ruff check .`
2. **test** - pytest on Python 3.12 and 3.13
3. **smoke** - `python main.py --sample` end to end, then a verification step that
   asserts both scorecards and all three model cards were written, that three models
   were benchmarked with all five gates each, that every ROC-AUC beat chance, and
   that the shrinkage prior was actually fitted. Reports upload as a build artifact.

The tests are written so that each gate has to fire on a known-bad synthetic input
and stay quiet on a known-good one, since a gate that does not distinguish the two
would still report a verdict. The Bayesian module is tested against the posterior
algebra directly:
that the blend weight equals `n / (n + a + b)`, that the posterior mean is the
implied weighted average of the raw and population rates, that shrinkage is monotone
in segment size, and that thin segments get wider intervals.

One detail worth flagging for anyone extending the suite: the calibration gate's
Brier limits are set against this dataset's 22% base rate. A *perfectly* calibrated
model on a 50/50 book scores a Brier of about 0.17 and would legitimately trip them,
so synthetic fixtures need a realistic base rate or they will fail for the wrong
reason.

## Author

Aryan Singh - [github.com/asynced24](https://github.com/asynced24)

Code is MIT licensed, see [LICENSE](LICENSE). The committed data sample carries its
own CC BY 4.0 attribution, recorded in [NOTICE](NOTICE).
