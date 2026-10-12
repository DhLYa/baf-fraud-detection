"""Feature engineering for the XGBoost and logistic regression feature sets.

Kept free of heavy imports: restore_categories is loaded inside served MLflow models.
"""
import numpy as np
import pandas as pd

# Decided in 01_eda: device_fraud_count is constant 0, and these two flags cover few rows at about the base rate
COLS_TO_DROP = [
    "device_fraud_count",
    "session_length_in_minutes_missing_flag",
    "device_distinct_emails_8w_missing_flag",
]

# Each rare level is merged into the level with the closest training fraud rate
RARE_LEVEL_MERGES = {
    "payment_type": {"AE": "AA"},                # AE 0.38% -> AA 0.51%
    "employment_status": {"CG": "CA"},           # CG 1.20% -> CA 1.18%
    "housing_status": {"BF": "BE", "BG": "BE"},  # BF 0.39%, BG 0.41% -> BE 0.34%
}

# Fraud jumps from ~2% at 1,500 to 12-17% above it, which a single slope badly underfits
CREDIT_LIMIT_BINS = [-np.inf, 200, 500, 1000, 1500, np.inf]
CREDIT_LIMIT_LABELS = ["<=200", "201-500", "501-1000", "1001-1500", ">1500"]

# Features one-hot encoded for logistic regression
LR_ONEHOT_COLS = [
    "payment_type", "employment_status", "housing_status", "device_os",
    "proposed_credit_limit", "device_distinct_emails_8w",
]


def encode_source(X):
    X = X.copy()
    X["source"] = (X["source"] == "TELEAPP").astype(int)
    return X


def prepare_lr_features(X, income_top_decile_cutoff, device_emails_mode):
    # income_top_decile_cutoff and device_emails_mode must come from the training months only
    X = encode_source(X)
    for col, merges in RARE_LEVEL_MERGES.items():
        X[col] = X[col].astype(str).replace(merges)
    X["device_os"] = X["device_os"].astype(str)

    X["income_top_decile"] = (X["income"] >= income_top_decile_cutoff).astype(int)
    X["proposed_credit_limit"] = pd.cut(X["proposed_credit_limit"], bins=CREDIT_LIMIT_BINS,
                                        labels=CREDIT_LIMIT_LABELS).astype(str)

    # Levels 0 and 2 are both higher-risk than 1, so a single slope would let them cancel out
    X["device_distinct_emails_8w"] = (X["device_distinct_emails_8w"].fillna(device_emails_mode)
                                      .astype(int).astype(str))
    return X


def restore_categories(X, dtypes):
    # Served requests arrive as JSON, so categorical columns come in as strings
    return X.astype(dtypes)
