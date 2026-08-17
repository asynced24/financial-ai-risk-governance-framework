"""Tests the empirical-Bayes shrinkage against the posterior algebra."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from financial_ai_framework import SegmentShrinkageModel, SegmentStabilityCheck, Settings


def _synthetic(rates: dict[str, tuple[int, int]]) -> tuple[pd.Series, pd.Series]:
    """Build borrower-level rows from {segment: (n, k)} counts."""
    segments: list[str] = []
    outcomes: list[int] = []
    for segment, (n, k) in rates.items():
        segments += [segment] * n
        outcomes += [1] * k + [0] * (n - k)
    return pd.Series(segments), pd.Series(outcomes)


@pytest.fixture
def fitted(settings: Settings) -> SegmentShrinkageModel:
    """A wide segment, a mid segment, and a thin segment with an extreme raw rate."""
    segments, outcomes = _synthetic(
        {
            "wide": (4000, 800),  # 0.200
            "mid": (400, 100),  # 0.250
            "thin": (10, 0),  # 0.000 - only because it is tiny
            "thin_high": (8, 8),  # 1.000 - same
            "other": (600, 120),  # 0.200
        }
    )
    return SegmentShrinkageModel(settings).fit(segments, outcomes)


def test_prior_recovers_the_population_rate(fitted: SegmentShrinkageModel):
    prior = fitted.prior
    assert prior is not None
    pooled = (800 + 100 + 120) / (4000 + 400 + 600)
    assert prior.population_rate == pytest.approx(pooled, abs=0.01)
    assert prior.alpha > 0 and prior.beta > 0
    assert not prior.degenerate


def test_prior_subtracts_binomial_sampling_noise(fitted: SegmentShrinkageModel):
    """The estimated prior variance must be below the raw spread of the rates."""
    prior = fitted.prior
    assert prior.sampling_variance > 0
    assert prior.prior_variance <= prior.observed_variance
    assert prior.prior_variance == pytest.approx(
        max(prior.observed_variance - prior.sampling_variance, 1e-6), rel=1e-6
    )


def test_thin_segments_shrink_toward_the_population_rate(fitted: SegmentShrinkageModel):
    prior_mean = fitted.prior.prior_mean
    thin = fitted.posteriors["thin"]
    wide = fitted.posteriors["wide"]

    # The thin segment's raw rate is 0.0 but its posterior sits near the population.
    assert thin.empirical_rate == 0.0
    assert abs(thin.posterior_mean - prior_mean) < abs(thin.empirical_rate - prior_mean)
    # The wide segment keeps its own signal.
    assert abs(wide.posterior_mean - wide.empirical_rate) < 0.01


def test_shrinkage_is_monotone_in_segment_size(fitted: SegmentShrinkageModel):
    weights = {name: p.shrinkage_weight for name, p in fitted.posteriors.items()}
    assert weights["thin_high"] < weights["thin"] < weights["mid"] < weights["wide"]


def test_shrinkage_weight_matches_the_posterior_formula(fitted: SegmentShrinkageModel):
    """w = n / (n + a + b), with no hand-set blending constant anywhere."""
    concentration = fitted.prior.concentration
    for posterior in fitted.posteriors.values():
        expected = posterior.n / (posterior.n + concentration)
        assert posterior.shrinkage_weight == pytest.approx(expected, rel=1e-9)


def test_posterior_mean_is_the_implied_weighted_blend(fitted: SegmentShrinkageModel):
    prior_mean = fitted.prior.prior_mean
    for posterior in fitted.posteriors.values():
        weight = posterior.shrinkage_weight
        blended = weight * posterior.empirical_rate + (1 - weight) * prior_mean
        assert posterior.posterior_mean == pytest.approx(blended, rel=1e-9)


def test_posterior_mean_lies_between_raw_rate_and_population(fitted: SegmentShrinkageModel):
    prior_mean = fitted.prior.prior_mean
    for posterior in fitted.posteriors.values():
        low, high = sorted((posterior.empirical_rate, prior_mean))
        assert low - 1e-9 <= posterior.posterior_mean <= high + 1e-9


def test_credible_intervals_bracket_the_posterior_mean(fitted: SegmentShrinkageModel):
    for posterior in fitted.posteriors.values():
        assert 0.0 <= posterior.ci_lower < posterior.posterior_mean < posterior.ci_upper <= 1.0
        assert posterior.contains(posterior.posterior_mean)


def test_thin_segments_get_wider_intervals(fitted: SegmentShrinkageModel):
    thin = fitted.posteriors["thin"]
    wide = fitted.posteriors["wide"]
    assert (thin.ci_upper - thin.ci_lower) > (wide.ci_upper - wide.ci_lower)


def test_sparse_segments_are_excluded_from_prior_fitting(fitted: SegmentShrinkageModel):
    assert fitted.posteriors["thin"].used_for_prior is False
    assert fitted.posteriors["wide"].used_for_prior is True
    assert fitted.summary()["n_sparse_segments"] == 2


def test_summary_and_frame_are_consistent(fitted: SegmentShrinkageModel):
    summary = fitted.summary()
    frame = fitted.to_frame()

    assert summary["n_segments"] == len(frame) == 5
    assert set(frame["segment_id"]) == set(fitted.posteriors)
    assert (frame["interval_width"] > 0).all()
    # to_frame is sorted widest interval first.
    assert frame["interval_width"].is_monotonic_decreasing


def test_most_shrunken_surfaces_the_extreme_thin_segment(fitted: SegmentShrinkageModel):
    top = fitted.most_shrunken(2)
    assert set(top["segment_id"]) == {"thin", "thin_high"}


def test_uniform_segments_fall_back_to_a_weak_prior(settings: Settings):
    """When every segment has the identical rate the spread is pure noise."""
    segments, outcomes = _synthetic({f"s{i}": (200, 40) for i in range(6)})
    model = SegmentShrinkageModel(settings).fit(segments, outcomes)

    assert model.prior.observed_variance == pytest.approx(0.0, abs=1e-12)
    assert model.prior.prior_variance == pytest.approx(settings.segments.prior_variance_floor)
    for posterior in model.posteriors.values():
        assert posterior.posterior_mean == pytest.approx(0.2, abs=0.01)


def test_fit_on_empty_input_raises(settings: Settings):
    with pytest.raises(ValueError, match="no segments"):
        SegmentShrinkageModel(settings).fit(pd.Series([], dtype=str), pd.Series([], dtype=int))


def test_unfitted_model_refuses_to_report(settings: Settings):
    with pytest.raises(RuntimeError, match="fit"):
        SegmentShrinkageModel(settings).summary()


# --------------------------------------------------------------- stability gate


def test_stability_gate_passes_when_pd_matches_the_posterior(
    settings: Settings, fitted: SegmentShrinkageModel
):
    segments = pd.Series(["wide"] * 200 + ["mid"] * 200)
    predicted = np.concatenate(
        [
            np.full(200, fitted.posteriors["wide"].posterior_mean),
            np.full(200, fitted.posteriors["mid"].posterior_mean),
        ]
    )

    gate = SegmentStabilityCheck(settings).run(fitted, segments, predicted, "aligned")
    assert gate.status == "pass"
    assert gate.metrics["segments_outside"] == 0
    assert gate.metrics["segments_gated"] == 2
    assert gate.findings == []


def test_stability_gate_fails_on_wildly_wrong_pd(
    settings: Settings, fitted: SegmentShrinkageModel
):
    segments = pd.Series(["wide"] * 200 + ["mid"] * 200)
    predicted = np.full(400, 0.95)

    gate = SegmentStabilityCheck(settings).run(fitted, segments, predicted, "broken")
    assert gate.status == "fail"
    assert gate.metrics["outside_interval_ratio"] == pytest.approx(1.0)
    assert len(gate.findings) == 2
    assert all(row["direction"] == "above" for row in gate.details["segments"])


def test_stability_gate_reports_direction_and_gap(
    settings: Settings, fitted: SegmentShrinkageModel
):
    segments = pd.Series(["wide"] * 200)
    predicted = np.full(200, 0.01)

    gate = SegmentStabilityCheck(settings).run(fitted, segments, predicted, "under")
    row = gate.details["segments"][0]
    assert row["direction"] == "below"
    assert row["gap"] < 0
    assert "below" in gate.findings[0]


def test_stability_gate_only_gates_segments_with_enough_rows(
    settings: Settings, fitted: SegmentShrinkageModel
):
    """A handful of evaluation rows is reported but must not drive the verdict."""
    few = settings.segments.min_segment_size - 1
    segments = pd.Series(["wide"] * few)
    predicted = np.full(few, 0.95)

    gate = SegmentStabilityCheck(settings).run(fitted, segments, predicted, "tiny")
    assert gate.metrics["segments_gated"] == 0
    assert gate.metrics["segments_evaluated"] == 1
    # Nothing auditable warns rather than passes: a 0.0 outside-ratio here means
    # "not measured", not "clean".
    assert gate.status == "warn"
    assert "not enough" in gate.findings[0].lower() or "information only" in gate.findings[0]
