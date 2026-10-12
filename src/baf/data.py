"""Cleaning the raw BAF data and loading the preprocessed splits written by 02."""
from typing import NamedTuple

import numpy as np
import pandas as pd

from baf.config import PREPROCESSED_DIR, TARGET

# Negative values in these columns are missing-value codes
MISSING_CODED_COLS = [
    "intended_balcon_amount",
    "prev_address_months_count",
    "bank_months_count",
    "current_address_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w"
]

CATEGORICAL_COLS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]


def flag_missing_codes(df):
    # Each negative code becomes NaN, with a <col>_missing_flag indicator added for the column
    df = df.copy()
    for col in MISSING_CODED_COLS:
        df[f"{col}_missing_flag"] = (df[col] < 0).astype(int)
        df.loc[df[col] < 0, col] = np.nan
    return df


def cast_categoricals(df):
    df = df.copy()
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype("category")
    return df


class Split(NamedTuple):
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series


def load_split(model):
    # Preprocessed features and targets for one model family ("lr" or "xgb"). Fields can be read by name
    # (split.X_val) or unpacked in order (X_train, X_val, y_train, y_val = split)
    data_dir = PREPROCESSED_DIR / model
    return Split(
        X_train=pd.read_parquet(data_dir / "X_train.parquet"),
        X_val=pd.read_parquet(data_dir / "X_val.parquet"),
        y_train=pd.read_parquet(data_dir / "y_train.parquet")[TARGET],
        y_val=pd.read_parquet(data_dir / "y_val.parquet")[TARGET],
    )
