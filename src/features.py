"""
Feature engineering.

Turns the enriched daily table into the numeric design matrix `X` used by the
model.  The exact same transformation is applied to historical rows (training)
and to synthetic future rows (forecasting), guaranteeing the columns line up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Categorical columns that get one-hot encoded.
CATEGORICAL_COLS = ["Region", "Season", "Climate_Zone"]

# Columns that must never enter the feature matrix (identifiers / targets /
# leakage).  `Base_Demand` is dropped so the model can't trivially copy it.
DROP_COLS = [
    "Date", "Target_Demand", "Base_Demand", "City_Proxy",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar + cyclical time features from the Date column."""
    df = df.copy()
    d = pd.to_datetime(df["Date"])
    df["Year"] = d.dt.year
    df["Month"] = d.dt.month
    df["Day"] = d.dt.day
    df["DayOfWeek"] = d.dt.dayofweek
    df["DayOfYear"] = d.dt.dayofyear
    df["WeekOfYear"] = d.dt.isocalendar().week.astype(int)
    df["Quarter"] = d.dt.quarter
    df["IsWeekend"] = (d.dt.dayofweek >= 5).astype(int)

    # Cyclical encodings so the model understands that Dec is next to Jan.
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DayOfYear_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365)
    df["DayOfYear_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365)

    # Linear trend index (days since the start of the series).
    df["TrendIndex"] = (d - d.min()).dt.days
    return df


def make_model_matrix(
    df: pd.DataFrame, feature_columns: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the numeric feature matrix.

    Parameters
    ----------
    df : enriched dataframe (must contain the raw feature + categorical cols).
    feature_columns : if given, the output is re-indexed to exactly these
        columns (missing ones filled with 0).  Use this at forecast time so the
        future matrix matches the training matrix.

    Returns
    -------
    (X, columns)
    """
    df = add_temporal_features(df)

    present_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=present_cats, drop_first=False)

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Keep only numeric columns and coerce booleans to ints.
    X = X.select_dtypes(include=[np.number, "bool"]).astype(float)

    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0.0)
    return X, list(X.columns)
