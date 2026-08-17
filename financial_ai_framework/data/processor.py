"""Feature engineering and splitting for the credit-default task.

Two design choices here are governance decisions rather than modelling ones, so
they are stated explicitly:

1. ``sex`` and ``marriage`` are **excluded from the feature matrix** as
   prohibited bases for a credit decision, but retained alongside it as audit
   columns. The fairness gate therefore measures disparate impact on attributes
   the model never saw - which is the case that actually matters, because
   correlated features can reintroduce the gap.
2. Every engineered feature is a ratio or count derivable from the borrower's own
   billing history. Nothing is fitted across the train/test boundary, so the
   split stays leakage-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import Settings

#: Prohibited bases for a credit decision - audited, never used as predictors.
PROHIBITED_FEATURES: Tuple[str, ...] = ("sex", "marriage")

#: Non-predictive columns carried alongside the matrix for governance checks.
AUDIT_COLUMNS: Tuple[str, ...] = (
    "sex",
    "sex_label",
    "age_band",
    "education_label",
    "marriage_label",
    "segment_id",
)

_EPS = 1.0


@dataclass
class CreditDataset:
    """A leakage-free split plus the audit columns the governance gates need."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    audit_train: pd.DataFrame
    audit_test: pd.DataFrame
    feature_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def summary(self) -> Dict[str, Any]:
        return {
            "n_train": int(len(self.X_train)),
            "n_test": int(len(self.X_test)),
            "n_features": self.n_features,
            "train_default_rate": float(self.y_train.mean()),
            "test_default_rate": float(self.y_test.mean()),
        }


class CreditDataProcessor:
    """Turns the canonical UCI frame into model-ready features."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.engineered_features: List[str] = []

    # ------------------------------------------------------------------ features

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add repayment-behaviour features derived from the billing history."""
        df = df.copy()
        limit = df["limit_bal"].replace(0, np.nan)

        bill_cols = [f"bill_amt_{i}" for i in range(1, 7)]
        pay_cols = [f"pay_amt_{i}" for i in range(1, 7)]
        status_cols = [f"pay_status_{i}" for i in range(1, 7)]

        created: List[str] = []

        df["log_limit_bal"] = np.log1p(df["limit_bal"].clip(lower=0))
        created.append("log_limit_bal")

        # Credit utilisation per statement month.
        for i, col in enumerate(bill_cols, start=1):
            name = f"utilization_{i}"
            df[name] = (df[col] / limit).clip(-1.0, 3.0)
            created.append(name)

        util_cols = [f"utilization_{i}" for i in range(1, 7)]
        df["utilization_mean"] = df[util_cols].mean(axis=1)
        df["utilization_max"] = df[util_cols].max(axis=1)
        df["utilization_std"] = df[util_cols].std(axis=1)
        created += ["utilization_mean", "utilization_max", "utilization_std"]

        # Share of the previous statement actually repaid.
        for i in range(1, 6):
            name = f"payment_ratio_{i}"
            prior_bill = df[f"bill_amt_{i + 1}"].clip(lower=0)
            df[name] = (df[f"pay_amt_{i}"] / (prior_bill + _EPS)).clip(0.0, 2.0)
            created.append(name)

        ratio_cols = [f"payment_ratio_{i}" for i in range(1, 6)]
        df["payment_ratio_mean"] = df[ratio_cols].mean(axis=1)
        df["payment_ratio_min"] = df[ratio_cols].min(axis=1)
        created += ["payment_ratio_mean", "payment_ratio_min"]

        # Delinquency profile across the six observed months.
        status = df[status_cols]
        df["delinquent_months"] = (status > 0).sum(axis=1)
        df["max_delinquency"] = status.max(axis=1)
        df["mean_delinquency"] = status.mean(axis=1)
        df["ever_delinquent"] = (status > 0).any(axis=1).astype(int)
        df["revolving_months"] = (status == 0).sum(axis=1)
        created += [
            "delinquent_months",
            "max_delinquency",
            "mean_delinquency",
            "ever_delinquent",
            "revolving_months",
        ]

        # Balance and repayment trajectory.
        df["bill_trend"] = ((df["bill_amt_1"] - df["bill_amt_6"]) / (limit + _EPS)).clip(-3.0, 3.0)
        df["bill_amt_mean"] = df[bill_cols].mean(axis=1)
        df["pay_amt_mean"] = df[pay_cols].mean(axis=1)
        df["pay_amt_total"] = df[pay_cols].sum(axis=1)
        df["zero_payment_months"] = (df[pay_cols] <= 0).sum(axis=1)
        df["payment_coverage"] = (
            df["pay_amt_total"] / (df[bill_cols].clip(lower=0).sum(axis=1) + _EPS)
        ).clip(0.0, 2.0)
        created += [
            "bill_trend",
            "bill_amt_mean",
            "pay_amt_mean",
            "pay_amt_total",
            "zero_payment_months",
            "payment_coverage",
        ]

        self.engineered_features = created
        df[created] = df[created].replace([np.inf, -np.inf], np.nan)
        df[created] = df[created].fillna(0.0)

        print(f"[features] engineered {len(created)} behavioural features")
        return df

    # ------------------------------------------------------------------- matrix

    def feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Numeric predictors, with prohibited bases and audit columns removed."""
        target = self.settings.data.target_column
        excluded = set(PROHIBITED_FEATURES) | set(AUDIT_COLUMNS) | {target}

        numeric = df.select_dtypes(include=[np.number]).columns
        return [c for c in numeric if c not in excluded]

    def build_matrix(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
        """Split the frame into predictors, target and audit columns."""
        target = self.settings.data.target_column
        features = self.feature_columns(df)

        X = df[features].astype("float64")
        y = df[target].astype("int64")
        audit = df[[c for c in AUDIT_COLUMNS if c in df.columns]].copy()

        dropped = [c for c in PROHIBITED_FEATURES if c in df.columns]
        print(
            f"[features] {len(features)} predictors | "
            f"withheld as prohibited bases: {', '.join(dropped) or 'none'}"
        )
        return X, y, audit, features

    # -------------------------------------------------------------------- split

    def prepare(self, df: pd.DataFrame) -> CreditDataset:
        """Engineer features and produce a stratified train/test split."""
        engineered = self.engineer_features(df)
        X, y, audit, features = self.build_matrix(engineered)

        (
            X_train,
            X_test,
            y_train,
            y_test,
            audit_train,
            audit_test,
        ) = train_test_split(
            X,
            y,
            audit,
            test_size=self.settings.data.test_size,
            random_state=self.settings.seed,
            stratify=y,
        )

        dataset = CreditDataset(
            X_train=X_train.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
            audit_train=audit_train.reset_index(drop=True),
            audit_test=audit_test.reset_index(drop=True),
            feature_names=features,
            metadata={
                "engineered_features": list(self.engineered_features),
                "withheld_features": [c for c in PROHIBITED_FEATURES if c in df.columns],
                "test_size": self.settings.data.test_size,
                "random_state": self.settings.seed,
            },
        )

        summary = dataset.summary()
        print(
            f"[split] train {summary['n_train']:,} | test {summary['n_test']:,} | "
            f"features {summary['n_features']} | "
            f"default rate train {summary['train_default_rate']:.3f} / "
            f"test {summary['test_default_rate']:.3f}"
        )
        return dataset
