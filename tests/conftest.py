"""Shared fixtures, all running against the committed offline sample."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from financial_ai_framework import (
    CreditDataProcessor,
    ModelBenchmarkSuite,
    Settings,
    load_credit_data,
    load_settings,
)

#: Rows drawn from the committed sample for the model-training fixtures.
FAST_ROWS = 2500


@pytest.fixture(scope="session")
def settings() -> Settings:
    """The real repository configuration, exactly as shipped."""
    return load_settings()


@pytest.fixture(scope="session")
def fast_settings(settings: Settings) -> Settings:
    """Same configuration with the expensive knobs turned down."""
    fast = settings.model_copy(deep=True)
    fast.models.n_estimators = 40
    fast.models.cv_folds = 2
    fast.governance.uncertainty.bootstrap_samples = 25
    fast.explainability.shap_sample_size = 40
    fast.explainability.shap_background_size = 25
    return fast


@pytest.fixture(scope="session")
def credit_frame(settings: Settings):
    """The committed offline sample, with segment labels attached."""
    frame, metadata = load_credit_data(settings, use_sample=True)
    return frame, metadata


@pytest.fixture(scope="session")
def fast_frame(credit_frame, fast_settings: Settings) -> pd.DataFrame:
    """A class-stratified subset of the sample, for the training fixtures."""
    frame, _ = credit_frame
    target = fast_settings.data.target_column
    rng = np.random.default_rng(fast_settings.seed)

    parts = []
    for _label, group in frame.groupby(target):
        take = max(2, round(FAST_ROWS * len(group) / len(frame)))
        idx = rng.choice(len(group), size=min(take, len(group)), replace=False)
        parts.append(group.iloc[np.sort(idx)])

    return pd.concat(parts).sort_index().reset_index(drop=True)


@pytest.fixture(scope="session")
def dataset(fast_frame: pd.DataFrame, fast_settings: Settings):
    """A prepared train/test split over the subset."""
    return CreditDataProcessor(fast_settings).prepare(fast_frame)


@pytest.fixture(scope="session")
def benchmark_run(dataset, fast_settings: Settings):
    """One full benchmark pass. Returns (suite, results)."""
    suite = ModelBenchmarkSuite(fast_settings, tracker=None)
    results = suite.run(dataset)
    return suite, results
