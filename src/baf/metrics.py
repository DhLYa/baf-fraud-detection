"""Evaluation at a fixed FPR threshold, plus predictive equality across age groups."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score, roc_curve

from baf.config import AGE_BUCKET_CAP, AGE_BUCKET_LABELS, AGE_GROUP_CUTOFF, TARGET_FPR


def threshold_at_fpr(y_true, scores, target_fpr=TARGET_FPR):
    # Lowest threshold whose FPR stays at or below the target
    fpr, _, thresholds = roc_curve(y_true, scores)
    return thresholds[np.searchsorted(fpr, target_fpr, side="right") - 1]


def false_positive_rate(y_true, y_pred):
    return y_pred[y_true == 0].mean()


def age_buckets(age):
    return pd.Series(np.minimum(age, AGE_BUCKET_CAP)).map(AGE_BUCKET_LABELS)


def fpr_by_age_bucket(y_true, y_pred, age):
    # Only non-fraud rows can be false positives, so each bucket's FPR is the share of its non-fraud rows flagged
    non_fraud = y_true == 0
    buckets = age_buckets(age[non_fraud])
    flagged = pd.Series(y_pred[non_fraud])

    table = flagged.groupby(buckets).agg(non_fraud="size", false_positives="sum")
    table = table.reindex(list(AGE_BUCKET_LABELS.values())).rename_axis("age_bucket")
    table["fpr"] = table["false_positives"] / table["non_fraud"]
    return table


def evaluate_model(y_true, scores, age):
    threshold = threshold_at_fpr(y_true, scores)
    y_pred = (scores >= threshold).astype(int)
    recall = recall_score(y_true, y_pred)

    older = age >= AGE_GROUP_CUTOFF
    fpr_younger = false_positive_rate(y_true[~older], y_pred[~older])
    fpr_older = false_positive_rate(y_true[older], y_pred[older])
    bucket_fpr = fpr_by_age_bucket(y_true, y_pred, age)["fpr"]

    return {
        "ROC AUC": roc_auc_score(y_true, scores),
        "PR AUC": average_precision_score(y_true, scores),
        "Threshold": threshold,
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall,
        "TPR": recall,
        "FPR": false_positive_rate(y_true, y_pred),
        f"FPR age < {AGE_GROUP_CUTOFF}": fpr_younger,
        f"FPR age >= {AGE_GROUP_CUTOFF}": fpr_older,
        "Predictive equality": min(fpr_younger, fpr_older) / max(fpr_younger, fpr_older),
        "Predictive equality (all ages)": bucket_fpr.min() / bucket_fpr.max(),
    }


def wilson_interval(successes, n, z=1.96):
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half_width, centre + half_width
