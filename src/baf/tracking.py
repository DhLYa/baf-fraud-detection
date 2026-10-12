"""MLflow logging for the model runs in 03 and 04."""
from contextlib import nullcontext

import cloudpickle
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import baf
from baf.config import AGE_BUCKET_LABELS, AGE_GROUP_CUTOFF, TARGET_FPR, TRAIN_MONTHS, VAL_MONTHS
from baf.features import restore_categories
from baf.metrics import evaluate_model, fpr_by_age_bucket
from baf.plotting import plot_roc_pr

TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "BAF account creation fraud detection"

# MLflow metric names can't contain characters like < or >=
METRIC_NAMES = {
    "ROC AUC": "roc_auc",
    "PR AUC": "pr_auc",
    "Precision": "precision",
    "Recall": "recall",
    "TPR": "tpr",
    "FPR": "fpr",
    f"FPR age < {AGE_GROUP_CUTOFF}": f"fpr_age_under_{AGE_GROUP_CUTOFF}",
    f"FPR age >= {AGE_GROUP_CUTOFF}": f"fpr_age_{AGE_GROUP_CUTOFF}_plus",
    "Predictive equality": "predictive_equality",
    "Predictive equality (all ages)": "predictive_equality_all_ages",
}
AGE_BUCKET_METRIC_NAMES = {label: "fpr_age_" + label.replace("-", "_").replace("+", "_plus")
                           for label in AGE_BUCKET_LABELS.values()}
CLASS_NAMES = ["Non-Fraud", "Fraud"]

def to_mlflow_metrics(metrics):
    # MLflow-safe names for the metrics evaluate_model returns; any others (e.g. Threshold) are left out
    return {METRIC_NAMES[name]: value for name, value in metrics.items() if name in METRIC_NAMES}


def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def start_run(log_to_mlflow, **kwargs):
    # A do-nothing context when logging is off, so the training code reads the same either way
    return mlflow.start_run(**kwargs) if log_to_mlflow else nullcontext()


def xgb_serving_model(model, X_train):
    # Served requests arrive as JSON, so XGBoost's categorical columns come in as strings.
    # Logging the model behind this cast restores the training categories before predicting.
    dtypes = {col: X_train[col].dtype for col in X_train.select_dtypes("category").columns}
    return make_pipeline(FunctionTransformer(restore_categories, kw_args={"dtypes": dtypes}), model)


def log_model_run(name, model, X_val_model, y_true, scores, age, *, log_to_mlflow,
                  serving_model=None, run_name=None, params=None, extra_params=None):
    if serving_model is None:
        serving_model = model
    if run_name is None:
        run_name = f"{name} baseline"
    if params is None:
        params = model.get_params()

    row = pd.Series(evaluate_model(y_true, scores, age), name=name)
    y_pred = (scores >= row["Threshold"]).astype(int)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    age_fpr = fpr_by_age_bucket(y_true, y_pred, age)

    run_params = {
        "train_months": TRAIN_MONTHS,
        "val_months": VAL_MONTHS,
        "n_features": X_val_model.shape[1],
        "target_fpr": TARGET_FPR,
        "decision_threshold": row["Threshold"],
    }
    if extra_params:
        run_params.update(extra_params)

    metrics = to_mlflow_metrics(row)
    for bucket, fpr in age_fpr["fpr"].items():
        metrics[AGE_BUCKET_METRIC_NAMES[bucket]] = fpr

    if log_to_mlflow:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_params(run_params)
            mlflow.log_metrics(metrics)

            fig = plot_roc_pr(y_true, {name: scores}, row.to_frame().T, [name],
                              f"{name} on validation month 7 (dot = threshold at {TARGET_FPR:.0%} FPR)")
            mlflow.log_figure(fig, "roc_pr_curves.png")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(5, 4.5))
            ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=CLASS_NAMES,
                                                    cmap="Blues", colorbar=False, values_format=",", ax=ax)
            ax.set_title(f"{name} at {TARGET_FPR:.0%} FPR threshold", fontsize=10, loc="left")
            fig.tight_layout()
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            mlflow.log_text(report, "classification_report.txt")
            mlflow.log_text(age_fpr.to_csv(), "fpr_by_age_bucket.csv")

            input_example = X_val_model.head(5)
            signature = infer_signature(input_example, serving_model.predict_proba(input_example))
            # Pickle baf functions (restore_categories) into the model file itself, so serving doesn't need baf installed
            cloudpickle.register_pickle_by_value(baf)
            try:
                mlflow.sklearn.log_model(serving_model, name="model", signature=signature, input_example=input_example,
                                         pyfunc_predict_fn="predict_proba")
            finally:
                cloudpickle.unregister_pickle_by_value(baf)

    print(f"{name} (threshold {row['Threshold']:.4f})")
    print(report)
