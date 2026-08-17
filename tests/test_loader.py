"""The committed offline sample must be loadable, labelled and correctly shaped."""

from __future__ import annotations

import pandas as pd
import pytest

from financial_ai_framework import Settings, build_stratified_sample
from financial_ai_framework.data.loader import (
    EDUCATION_LABELS,
    UCI_COLUMN_ORDER,
    add_age_band,
    data_hash,
    normalise_categories,
)
from financial_ai_framework.data.processor import (
    AUDIT_COLUMNS,
    PROHIBITED_FEATURES,
    CreditDataProcessor,
)


def test_committed_sample_shape(settings: Settings, credit_frame):
    frame, metadata = credit_frame
    assert metadata["source"] == "committed_offline_sample"
    assert metadata["rows"] == settings.data.sample_rows == len(frame)
    assert metadata["target_column"] == settings.data.target_column


def test_committed_sample_has_every_canonical_column(credit_frame):
    frame, _ = credit_frame
    missing = [column for column in UCI_COLUMN_ORDER if column not in frame.columns]
    assert missing == []


def test_sample_preserves_base_rate(credit_frame):
    """The stratified draw must track the full dataset's ~22.1% default rate."""
    _, metadata = credit_frame
    assert metadata["default_rate"] == pytest.approx(0.2212, abs=0.005)


def test_metadata_carries_licence_and_citation(credit_frame):
    _, metadata = credit_frame
    assert metadata["licence"] == "CC BY 4.0"
    assert "Yeh" in metadata["citation"]
    assert "archive.ics.uci.edu" in metadata["source_url"]
    assert len(metadata["data_hash"]) == 16


def test_metadata_path_is_repo_relative(credit_frame):
    _, metadata = credit_frame
    assert metadata["path"] == "data/sample/uci_credit_sample.csv"


def test_segment_labels_attached(settings: Settings, credit_frame):
    frame, metadata = credit_frame
    for column in ("age_band", "education_label", "sex_label", "segment_id"):
        assert column in frame.columns
        assert frame[column].notna().all()

    assert set(frame["age_band"]).issubset(set(settings.age_band_labels()))
    assert set(frame["education_label"]).issubset(set(EDUCATION_LABELS.values()))
    assert metadata["n_segments"] == frame["segment_id"].nunique()


def test_segment_id_is_education_and_age_band(credit_frame):
    frame, _ = credit_frame
    row = frame.iloc[0]
    assert row["segment_id"] == f"{row['education_label']} | {row['age_band']}"


def test_data_hash_is_deterministic_and_content_sensitive(credit_frame):
    frame, _ = credit_frame
    assert data_hash(frame) == data_hash(frame.copy())

    mutated = frame.copy()
    mutated.loc[0, "limit_bal"] = mutated.loc[0, "limit_bal"] + 1
    assert data_hash(mutated) != data_hash(frame)


def test_undocumented_category_codes_are_folded():
    raw = pd.DataFrame(
        {
            "education": [0, 1, 2, 3, 4, 5, 6],
            "marriage": [0, 1, 2, 3, 1, 2, 3],
            "sex": [1, 2, 1, 2, 1, 2, 1],
        }
    )
    folded = normalise_categories(raw)
    assert sorted(folded["education"].unique()) == [1, 2, 3, 4]
    assert sorted(folded["marriage"].unique()) == [1, 2, 3]


def test_age_band_clamps_out_of_range_ages(settings: Settings):
    frame = pd.DataFrame({"age": [1, 20, 29, 30, 79, 80, 200]})
    banded = add_age_band(frame, settings)
    labels = settings.age_band_labels()

    assert banded["age_band"].notna().all()
    assert set(banded["age_band"]).issubset(set(labels))
    assert banded.loc[0, "age_band"] == labels[0]  # below the lowest edge
    assert banded.loc[6, "age_band"] == labels[-1]  # above the highest edge
    assert banded.loc[2, "age_band"] == "20-30"
    assert banded.loc[3, "age_band"] == "30-40"


def test_stratified_sample_preserves_class_balance(credit_frame, settings: Settings):
    frame, _ = credit_frame
    target = settings.data.target_column
    drawn = build_stratified_sample(frame, 500, target, settings.seed)

    assert len(drawn) == 500
    assert drawn[target].mean() == pytest.approx(frame[target].mean(), abs=0.01)


def test_prohibited_bases_excluded_from_features(dataset):
    """sex and marriage are audited but must never reach the model."""
    for column in PROHIBITED_FEATURES:
        assert column not in dataset.X_train.columns
        assert column not in dataset.feature_names


def test_audit_columns_align_with_the_split(dataset):
    assert len(dataset.audit_train) == len(dataset.X_train)
    assert len(dataset.audit_test) == len(dataset.X_test)
    for column in AUDIT_COLUMNS:
        assert column in dataset.audit_test.columns


def test_split_is_stratified_and_leakage_free(dataset):
    summary = dataset.summary()
    assert summary["train_default_rate"] == pytest.approx(summary["test_default_rate"], abs=0.02)
    assert summary["n_train"] > summary["n_test"]
    assert set(dataset.X_train.columns) == set(dataset.X_test.columns)


def test_engineered_features_are_finite(fast_frame, fast_settings: Settings):
    processor = CreditDataProcessor(fast_settings)
    engineered = processor.engineer_features(fast_frame)

    assert processor.engineered_features
    values = engineered[processor.engineered_features]
    assert values.notna().all().all()
    assert (values.abs() < 1e9).all().all()


def test_target_is_not_a_feature(dataset, fast_settings: Settings):
    assert fast_settings.data.target_column not in dataset.feature_names
