"""Shared chart styling, model colours and the ROC / precision-recall comparison."""
import matplotlib.pyplot as plot
from sklearn.metrics import precision_recall_curve, roc_curve

MODEL_COLORS = {
    "Logistic Regression": "#2a78d6",
    "XGBoost": "#1baf7a",
    "Logistic Regression (tuned)": "#7b4fd6",
    "XGBoost (tuned)": "#eb6834",
    "Logistic Regression (tuned, no age)": "#c79a1e",
    "XGBoost (tuned, no age)": "#0f8a8a",
}
REFERENCE_COLOR = "#8a8984"


def style_axis(ax, title, ylabel):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel(ylabel, fontsize=8, color="#52514e")
    ax.tick_params(labelsize=8, colors="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_roc_pr(y_true, val_scores, results, model_names, title):
    # val_scores maps model name -> validation scores; results has one evaluate_model row per model
    fig, (ax_roc, ax_pr) = plot.subplots(1, 2, figsize=(13, 5.5))

    for name in model_names:
        scores = val_scores[name]
        color = MODEL_COLORS[name]
        row = results.loc[name]
        # Dot marking where the decision threshold puts this model on each curve
        threshold_marker = dict(marker="o", linestyle="none", color=color, markersize=9,
                                markeredgecolor="white", markeredgewidth=2)

        fpr, tpr, _ = roc_curve(y_true, scores)
        ax_roc.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (ROC AUC {row['ROC AUC']:.3f})")
        ax_roc.plot(row["FPR"], row["TPR"], **threshold_marker)

        # Skip the final (recall 0, precision 1) point sklearn appends, which draws a vertical spike at the edge
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax_pr.plot(recall[:-1], precision[:-1], color=color, linewidth=2,
                   label=f"{name} (PR AUC {row['PR AUC']:.3f})")
        ax_pr.plot(row["Recall"], row["Precision"], **threshold_marker)

    ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=REFERENCE_COLOR, label="Chance")
    style_axis(ax_roc, "ROC curve", "True positive rate")
    ax_roc.set_xlabel("False positive rate", fontsize=8, color="#52514e")
    ax_roc.legend(loc="lower right", frameon=False, fontsize=9)

    base_rate = y_true.mean()
    ax_pr.axhline(base_rate, linestyle="--", linewidth=1, color=REFERENCE_COLOR,
                  label=f"Base rate ({base_rate:.2%})")
    style_axis(ax_pr, "Precision-recall curve", "Precision")
    ax_pr.set_xlabel("Recall", fontsize=8, color="#52514e")
    ax_pr.set_ylim(0, 1)
    ax_pr.legend(loc="upper right", frameon=False, fontsize=9)

    fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
