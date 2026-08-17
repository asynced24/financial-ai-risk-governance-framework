"""Empirical-Bayes Beta-Binomial shrinkage of segment-level default rates.

Why this exists
---------------
Segment-level default rates are the unit most credit-risk governance questions get
asked in: "does the model price 20-somethings with only a high-school education
the way the book actually defaults?" The naive answer - the raw default rate in
that cell - is useless for thin cells, where one extra default swings the rate by
several points.

The fix is an empirical-Bayes hierarchical model. Segment default counts are
treated as Binomial draws from segment-specific rates, which are themselves draws
from one population-level Beta distribution:

    theta_i  ~  Beta(a, b)                     (population prior, shared)
    k_i      ~  Binomial(n_i, theta_i)         (that segment's observed defaults)

Because the Beta is conjugate to the Binomial, each segment's posterior is closed
form:

    theta_i | k_i, n_i  ~  Beta(a + k_i, b + n_i - k_i)

and the posterior mean is an *automatic* precision-weighted blend of the
segment's own rate and the population rate:

    E[theta_i | data]  =  w_i * (k_i / n_i)  +  (1 - w_i) * a / (a + b)
    w_i                =  n_i / (n_i + a + b)

There is no hand-tuned blending constant anywhere in this module. Thin segments
shrink hard toward the population rate and wide segments barely move, purely as a
consequence of ``w_i`` falling out of the posterior algebra.

Fitting the prior
-----------------
``(a, b)`` are estimated by method of moments across segments. The wrinkle worth
being careful about: the observed spread of the empirical rates ``p_i = k_i / n_i``
overstates the true spread of ``theta_i``, because each ``p_i`` carries its own
binomial sampling noise. Using it directly yields a prior that is too diffuse and
therefore under-shrinks. So the sampling component is subtracted first:

    Var(p_i)  =  Var(theta_i)  +  E[ theta(1 - theta) / n_i ]

giving ``Var(theta) = Var(p_i) - mean( mu(1 - mu) / n_i )``, floored at a small
positive constant. The Beta concentration then follows from the standard identity
``Var = mu(1 - mu) / (a + b + 1)``.

This is textbook hierarchical-model statistics (the beta-binomial moment
estimator), applied here to demographic credit segments. Pure numpy/scipy - no
additional modelling dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Settings
from ..governance.gates import GateResult


@dataclass
class ShrinkagePrior:
    """The fitted population-level Beta prior and the moments behind it."""

    alpha: float
    beta: float
    population_rate: float
    prior_variance: float
    observed_variance: float
    sampling_variance: float
    n_segments_used: int
    degenerate: bool = False

    @property
    def concentration(self) -> float:
        """``a + b`` - the prior's equivalent sample size, in borrowers."""
        return self.alpha + self.beta

    @property
    def prior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "prior_mean": self.prior_mean,
            "concentration_equivalent_borrowers": self.concentration,
            "population_rate": self.population_rate,
            "observed_variance_of_segment_rates": self.observed_variance,
            "binomial_sampling_variance": self.sampling_variance,
            "estimated_prior_variance": self.prior_variance,
            "n_segments_used_for_prior": self.n_segments_used,
            "degenerate_prior_fallback": self.degenerate,
        }


@dataclass
class SegmentPosterior:
    """One segment's shrunken posterior over its default rate."""

    segment_id: str
    n: int
    k: int
    empirical_rate: float
    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float
    ci_lower: float
    ci_upper: float
    shrinkage_weight: float
    used_for_prior: bool

    @property
    def shrinkage_applied(self) -> float:
        """How far the posterior mean moved off the raw rate, in rate points."""
        return self.posterior_mean - self.empirical_rate

    def contains(self, value: float) -> bool:
        return bool(self.ci_lower <= value <= self.ci_upper)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "n": self.n,
            "defaults": self.k,
            "empirical_rate": self.empirical_rate,
            "posterior_mean": self.posterior_mean,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "shrinkage_weight_on_own_data": self.shrinkage_weight,
            "shrinkage_applied": self.shrinkage_applied,
            "used_for_prior": self.used_for_prior,
        }


class SegmentShrinkageModel:
    """Fits the population Beta prior and every segment's posterior.

    Usage::

        model = SegmentShrinkageModel(settings).fit(df["segment_id"], df["default"])
        model.posteriors["university | 30-40"].ci_upper
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.credible_mass = settings.segments.credible_mass
        self.min_segment_size = settings.segments.min_segment_size
        self.variance_floor = settings.segments.prior_variance_floor

        self.prior: ShrinkagePrior | None = None
        self.posteriors: dict[str, SegmentPosterior] = {}
        self.counts: pd.DataFrame | None = None

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _aggregate(segment_ids: pd.Series, outcomes: pd.Series) -> pd.DataFrame:
        """Collapse borrower rows into per-segment (n, k) counts."""
        frame = pd.DataFrame(
            {
                "segment_id": pd.Series(segment_ids).astype(str).to_numpy(),
                "outcome": pd.Series(outcomes).astype(int).to_numpy(),
            }
        )
        counts = (
            frame.groupby("segment_id")["outcome"]
            .agg(n="size", k="sum")
            .reset_index()
            .sort_values("segment_id")
            .reset_index(drop=True)
        )
        counts["empirical_rate"] = counts["k"] / counts["n"]
        return counts

    def _fit_prior(self, counts: pd.DataFrame) -> ShrinkagePrior:
        """Method-of-moments Beta prior across segments, net of sampling noise."""
        eligible = counts[counts["n"] >= self.min_segment_size]
        if len(eligible) < 2:
            # Not enough well-populated segments to estimate a spread; fall back
            # to the pooled rate with a deliberately weak prior.
            eligible = counts

        n = eligible["n"].to_numpy(dtype=float)
        k = eligible["k"].to_numpy(dtype=float)
        rates = k / n

        # Pooled rate is the lowest-variance estimator of the population mean.
        mu = float(k.sum() / n.sum()) if n.sum() > 0 else 0.5
        mu = float(np.clip(mu, 1e-6, 1 - 1e-6))

        observed_var = float(np.var(rates, ddof=1)) if len(rates) > 1 else 0.0
        sampling_var = float(np.mean(mu * (1.0 - mu) / n))
        prior_var = max(observed_var - sampling_var, self.variance_floor)

        max_var = mu * (1.0 - mu)
        degenerate = False
        if prior_var >= max_var:
            # A Beta cannot be that dispersed; fall back to a near-flat prior.
            concentration = 1.0
            degenerate = True
        else:
            concentration = max_var / prior_var - 1.0
            if not np.isfinite(concentration) or concentration <= 0:
                concentration = 1.0
                degenerate = True

        return ShrinkagePrior(
            alpha=mu * concentration,
            beta=(1.0 - mu) * concentration,
            population_rate=mu,
            prior_variance=prior_var,
            observed_variance=observed_var,
            sampling_variance=sampling_var,
            n_segments_used=int(len(eligible)),
            degenerate=degenerate,
        )

    # --------------------------------------------------------------------- API

    def fit(self, segment_ids: pd.Series, outcomes: pd.Series) -> SegmentShrinkageModel:
        """Fit the population prior and every segment posterior.

        Parameters
        ----------
        segment_ids:
            Segment label per borrower (education x age-band).
        outcomes:
            Binary default indicator per borrower.
        """
        counts = self._aggregate(segment_ids, outcomes)
        if counts.empty:
            raise ValueError("Cannot fit shrinkage model: no segments in the input")

        prior = self._fit_prior(counts)
        tail = (1.0 - self.credible_mass) / 2.0

        posteriors: dict[str, SegmentPosterior] = {}
        for row in counts.itertuples(index=False):
            post_a = prior.alpha + float(row.k)
            post_b = prior.beta + float(row.n) - float(row.k)
            lower, upper = stats.beta.ppf([tail, 1.0 - tail], post_a, post_b)

            posteriors[row.segment_id] = SegmentPosterior(
                segment_id=row.segment_id,
                n=int(row.n),
                k=int(row.k),
                empirical_rate=float(row.empirical_rate),
                posterior_alpha=float(post_a),
                posterior_beta=float(post_b),
                posterior_mean=float(post_a / (post_a + post_b)),
                ci_lower=float(lower),
                ci_upper=float(upper),
                shrinkage_weight=float(row.n / (row.n + prior.concentration)),
                used_for_prior=bool(row.n >= self.min_segment_size),
            )

        self.prior = prior
        self.counts = counts
        self.posteriors = posteriors

        print(
            f"[bayes] fitted Beta({prior.alpha:.2f}, {prior.beta:.2f}) prior across "
            f"{len(posteriors)} segments | population rate {prior.population_rate:.4f} | "
            f"prior worth {prior.concentration:.0f} borrowers"
        )
        return self

    def to_frame(self) -> pd.DataFrame:
        """Per-segment posteriors as a frame, widest interval first."""
        self._require_fit()
        frame = pd.DataFrame([p.to_dict() for p in self.posteriors.values()])
        frame["interval_width"] = frame["ci_upper"] - frame["ci_lower"]
        return frame.sort_values("interval_width", ascending=False).reset_index(drop=True)

    def most_shrunken(self, top: int = 5) -> pd.DataFrame:
        """Segments whose raw rate moved furthest under shrinkage."""
        frame = self.to_frame()
        frame["abs_shrinkage"] = frame["shrinkage_applied"].abs()
        return frame.nlargest(top, "abs_shrinkage").reset_index(drop=True)

    def summary(self) -> dict[str, Any]:
        self._require_fit()
        assert self.prior is not None
        frame = self.to_frame()
        return {
            "prior": self.prior.to_dict(),
            "n_segments": int(len(self.posteriors)),
            "n_sparse_segments": int(
                sum(1 for p in self.posteriors.values() if not p.used_for_prior)
            ),
            "credible_mass": self.credible_mass,
            "min_segment_size": self.min_segment_size,
            "median_interval_width": float(frame["interval_width"].median()),
            "median_shrinkage_weight": float(frame["shrinkage_weight_on_own_data"].median()),
            "max_abs_shrinkage": float(frame["shrinkage_applied"].abs().max()),
        }

    def _require_fit(self) -> None:
        if self.prior is None or not self.posteriors:
            raise RuntimeError("SegmentShrinkageModel.fit() must be called first")


class SegmentStabilityCheck:
    """Governance gate: does the model agree with the shrunken segment posteriors?

    For each segment, the model's mean predicted probability of default across the
    evaluation rows in that segment is compared with the segment's shrunken
    credible interval. Falling outside means the model prices that demographic
    cell differently from the (noise-corrected) historical default experience -
    which is a finding regardless of the model's headline AUC.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.thresholds = settings.governance.segment_stability

    def run(
        self,
        shrinkage: SegmentShrinkageModel,
        segment_ids: pd.Series,
        predicted_pd: np.ndarray,
        model_name: str = "model",
    ) -> GateResult:
        """Compare mean predicted PD per segment against the credible intervals."""
        shrinkage._require_fit()

        frame = pd.DataFrame(
            {
                "segment_id": pd.Series(segment_ids).astype(str).to_numpy(),
                "predicted_pd": np.asarray(predicted_pd, dtype=float),
            }
        )
        grouped = (
            frame.groupby("segment_id")["predicted_pd"]
            .agg(mean_predicted_pd="mean", n_eval="size")
            .reset_index()
        )

        rows: list[dict[str, Any]] = []
        for row in grouped.itertuples(index=False):
            posterior = shrinkage.posteriors.get(row.segment_id)
            if posterior is None:
                continue

            mean_pd = float(row.mean_predicted_pd)
            inside = posterior.contains(mean_pd)
            if mean_pd < posterior.ci_lower:
                direction = "below"
            elif mean_pd > posterior.ci_upper:
                direction = "above"
            else:
                direction = "inside"

            rows.append(
                {
                    "segment_id": row.segment_id,
                    "n_eval": int(row.n_eval),
                    "n_reference": posterior.n,
                    "reference_defaults": posterior.k,
                    "empirical_rate": posterior.empirical_rate,
                    "posterior_mean": posterior.posterior_mean,
                    "ci_lower": posterior.ci_lower,
                    "ci_upper": posterior.ci_upper,
                    "mean_predicted_pd": mean_pd,
                    "inside_interval": inside,
                    "direction": direction,
                    "gap": 0.0
                    if inside
                    else float(
                        mean_pd - posterior.ci_upper
                        if direction == "above"
                        else mean_pd - posterior.ci_lower
                    ),
                    "gated": bool(row.n_eval >= self.settings.segments.min_segment_size),
                }
            )

        detail = pd.DataFrame(rows)
        gated = detail[detail["gated"]] if not detail.empty else detail

        n_gated = int(len(gated))
        n_outside = int((~gated["inside_interval"]).sum()) if n_gated else 0
        outside_ratio = (n_outside / n_gated) if n_gated else 0.0

        status = self.thresholds.outside_interval_ratio.classify(outside_ratio)

        findings: list[str] = []
        if n_gated == 0:
            findings.append(
                "No segment had enough evaluation rows to gate; check reported for information only."
            )
        if n_outside:
            breaches = gated[~gated["inside_interval"]].copy()
            breaches = breaches.reindex(
                breaches["gap"].abs().sort_values(ascending=False).index
            )
        else:
            breaches = detail.iloc[0:0]

        for row in breaches.itertuples(index=False):
            findings.append(
                f"{row.segment_id}: mean predicted PD {row.mean_predicted_pd:.3f} sits "
                f"{row.direction} the {self.settings.segments.credible_mass:.0%} credible interval "
                f"[{row.ci_lower:.3f}, {row.ci_upper:.3f}] "
                f"(shrunken posterior mean {row.posterior_mean:.3f}, "
                f"{row.n_reference} reference borrowers)"
            )

        return GateResult(
            name="segment_stability",
            status=status,
            headline_metric="segments_outside_credible_interval_ratio",
            headline_value=outside_ratio,
            threshold=self.thresholds.outside_interval_ratio.describe(),
            metrics={
                "segments_gated": n_gated,
                "segments_outside": n_outside,
                "outside_interval_ratio": outside_ratio,
                "segments_evaluated": int(len(detail)),
            },
            findings=findings,
            details={
                "model": model_name,
                "credible_mass": self.settings.segments.credible_mass,
                "min_eval_rows_to_gate": self.settings.segments.min_segment_size,
                "prior": shrinkage.prior.to_dict() if shrinkage.prior else {},
                "segments": detail.to_dict(orient="records"),
            },
        )
