"""Loading, caching and sampling of the UCI credit-default dataset.

Source
------
UCI Machine Learning Repository, dataset 350 - "Default of Credit Card Clients".
Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for
the predictive accuracy of probability of default of credit card clients.*
Expert Systems with Applications, 36(2), 2473-2480.
Distributed by UCI under the Creative Commons Attribution 4.0 International
(CC BY 4.0) licence, which permits redistribution of the committed sample with
attribution.

The live dataset ships with opaque column names (``X1`` ... ``X23``, ``Y``), so
this module renames them positionally to the canonical names documented by the
dataset authors and folds the documented "unknown" category codes together.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import REPO_ROOT, Settings

#: Positional rename of the raw UCI columns to canonical names.
UCI_COLUMN_ORDER: List[str] = [
    "limit_bal",
    "sex",
    "education",
    "marriage",
    "age",
    "pay_status_1",
    "pay_status_2",
    "pay_status_3",
    "pay_status_4",
    "pay_status_5",
    "pay_status_6",
    "bill_amt_1",
    "bill_amt_2",
    "bill_amt_3",
    "bill_amt_4",
    "bill_amt_5",
    "bill_amt_6",
    "pay_amt_1",
    "pay_amt_2",
    "pay_amt_3",
    "pay_amt_4",
    "pay_amt_5",
    "pay_amt_6",
]

#: Codes 0, 5 and 6 are undocumented in the source paper; the authors' category 4
#: is "others", so they are folded into it rather than silently kept as levels.
EDUCATION_LABELS: Dict[int, str] = {
    1: "graduate_school",
    2: "university",
    3: "high_school",
    4: "other_unknown",
}
SEX_LABELS: Dict[int, str] = {1: "male", 2: "female"}
MARRIAGE_LABELS: Dict[int, str] = {1: "married", 2: "single", 3: "other_unknown"}

CITATION = (
    "Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques "
    "for the predictive accuracy of probability of default of credit card clients. "
    "Expert Systems with Applications, 36(2), 2473-2480. "
    "UCI Machine Learning Repository, dataset 350 (CC BY 4.0)."
)
SOURCE_URL = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"


def _display_path(path: Path) -> str:
    """Repo-relative path with forward slashes, so reports are portable."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def data_hash(df: pd.DataFrame) -> str:
    """Stable content hash of a frame, for run-to-run lineage checks."""
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()
    return digest[:16]


def _canonicalise(features: pd.DataFrame, targets: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Rename raw UCI columns positionally and attach the target."""
    if features.shape[1] != len(UCI_COLUMN_ORDER):
        raise ValueError(
            f"Expected {len(UCI_COLUMN_ORDER)} UCI feature columns, got {features.shape[1]}. "
            "The upstream dataset layout has changed; update UCI_COLUMN_ORDER."
        )

    df = features.copy()
    df.columns = list(UCI_COLUMN_ORDER)

    target = targets.iloc[:, 0] if isinstance(targets, pd.DataFrame) else targets
    df[target_column] = pd.to_numeric(target, errors="coerce").astype("int64")

    return df


def normalise_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Fold the documented 'unknown' category codes into their 'other' bucket."""
    df = df.copy()
    if "education" in df.columns:
        df["education"] = df["education"].where(df["education"].isin([1, 2, 3]), 4).astype("int64")
    if "marriage" in df.columns:
        df["marriage"] = df["marriage"].where(df["marriage"].isin([1, 2]), 3).astype("int64")
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype("int64")
    return df


def fetch_uci_credit(settings: Settings, refresh: bool = False) -> pd.DataFrame:
    """Fetch dataset 350 from UCI, caching the canonical CSV under ``data/raw``.

    A cached copy is reused unless ``refresh`` is set, so repeat runs are offline
    after the first fetch.
    """
    cache_dir = settings.raw_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"uci_credit_{settings.data.uci_dataset_id}.csv"

    if cache_path.exists() and not refresh:
        print(f"[data] using cached raw dataset: {cache_path}")
        return normalise_categories(pd.read_csv(cache_path))

    print(f"[data] fetching UCI dataset {settings.data.uci_dataset_id} (network required)...")
    from ucimlrepo import fetch_ucirepo  # imported lazily so --sample never needs it

    repo = fetch_ucirepo(id=settings.data.uci_dataset_id)
    df = _canonicalise(repo.data.features, repo.data.targets, settings.data.target_column)
    df = normalise_categories(df)

    df.to_csv(cache_path, index=False)
    print(f"[data] cached raw dataset -> {cache_path} ({len(df):,} rows)")
    return df


def build_stratified_sample(
    df: pd.DataFrame,
    n_rows: int,
    target_column: str,
    seed: int,
) -> pd.DataFrame:
    """Draw a class-stratified sample that preserves the default base rate."""
    if n_rows >= len(df):
        return df.copy()

    from sklearn.model_selection import train_test_split

    sample, _ = train_test_split(
        df,
        train_size=n_rows,
        random_state=seed,
        stratify=df[target_column],
    )
    return sample.sort_index().reset_index(drop=True)


def write_sample(df: pd.DataFrame, settings: Settings) -> Path:
    """Persist the offline sample used by ``python main.py --sample``."""
    path = settings.sample_path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[data] wrote offline sample -> {path} ({len(df):,} rows)")
    return path


def add_age_band(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Attach the configured age band as a string-labelled column.

    Ages outside the configured range are clamped into the nearest band rather
    than dropped, so every borrower is assignable to exactly one segment.
    """
    df = df.copy()
    edges = settings.segments.age_bins
    labels = settings.age_band_labels()

    age = pd.to_numeric(df[settings.segments.age_column], errors="coerce").astype("float64")
    clamped = age.clip(lower=float(edges[0]), upper=float(edges[-1]) - 1e-9)

    band = pd.cut(clamped, bins=edges, labels=labels, right=False)
    df["age_band"] = band.astype("object").fillna(labels[-1]).astype(str)
    return df


def add_segment_labels(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Attach human-readable education / sex labels and the segment identifier.

    Segments are ``education x age-band``, the demographic cut used by the
    empirical-Bayes shrinkage model.
    """
    df = add_age_band(df, settings)
    edu_col = settings.segments.education_column

    df["education_label"] = (
        df[edu_col].map(EDUCATION_LABELS).fillna(EDUCATION_LABELS[4]).astype(str)
    )
    df["sex_label"] = df["sex"].map(SEX_LABELS).fillna("unknown").astype(str)
    df["marriage_label"] = df["marriage"].map(MARRIAGE_LABELS).fillna("other_unknown").astype(str)
    df["segment_id"] = df["education_label"] + " | " + df["age_band"]
    return df


def load_credit_data(
    settings: Settings,
    use_sample: bool = False,
    refresh: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load the dataset and return it with lineage metadata.

    Parameters
    ----------
    use_sample:
        When true, read the committed 5,000-row stratified sample so the whole
        pipeline runs offline. Otherwise fetch (or reuse the cache of) the full
        30,000-row UCI dataset.
    """
    target = settings.data.target_column

    if use_sample:
        path = settings.sample_path
        if not path.exists():
            raise FileNotFoundError(
                f"Offline sample not found at {path}. Run `python main.py --refresh-sample` "
                "once with network access to regenerate it."
            )
        df = normalise_categories(pd.read_csv(path))
        source = "committed_offline_sample"
        print(f"[data] loaded offline sample: {path} ({len(df):,} rows)")
    else:
        df = fetch_uci_credit(settings, refresh=refresh)
        path = settings.raw_cache_dir / f"uci_credit_{settings.data.uci_dataset_id}.csv"
        source = "uci_live_fetch"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' missing from loaded data: {list(df.columns)}")

    missing = [c for c in UCI_COLUMN_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Loaded data is missing expected columns: {missing}")

    df = add_segment_labels(df, settings)

    metadata: Dict[str, Any] = {
        "source": source,
        "path": _display_path(path),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "target_column": target,
        "default_rate": float(df[target].mean()),
        "n_segments": int(df["segment_id"].nunique()),
        "data_hash": data_hash(df),
        "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "citation": CITATION,
        "source_url": SOURCE_URL,
        "licence": "CC BY 4.0",
    }

    print(
        f"[data] {metadata['rows']:,} rows | default rate {metadata['default_rate']:.3f} "
        f"| {metadata['n_segments']} segments | hash {metadata['data_hash']}"
    )
    return df, metadata


def refresh_offline_sample(settings: Settings) -> Path:
    """Re-fetch the live dataset and rewrite the committed offline sample."""
    full = fetch_uci_credit(settings, refresh=True)
    sample = build_stratified_sample(
        full,
        n_rows=settings.data.sample_rows,
        target_column=settings.data.target_column,
        seed=settings.seed,
    )
    full_rate = float(full[settings.data.target_column].mean())
    sample_rate = float(sample[settings.data.target_column].mean())
    print(
        f"[data] base rate preserved: full {full_rate:.4f} vs sample {sample_rate:.4f} "
        f"(delta {abs(full_rate - sample_rate):.4f})"
    )
    if not np.isclose(full_rate, sample_rate, atol=0.01):
        raise ValueError(
            f"Stratified sample base rate {sample_rate:.4f} drifted from full {full_rate:.4f}"
        )
    return write_sample(sample, settings)
