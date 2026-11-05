import os
import tempfile
from pathlib import Path
import matplotlib
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve


def plot_confusion_matrix(cm_raw: list | np.ndarray, model_name: str, save_path: str | Path):
    """Plot confusion matrix heatmap with counts and percentages."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(cm_raw)
    group_counts = [f"{v:0.0f}" for v in cm.flatten()]
    group_percentages = [f"{v:.1%}" for v in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Pure (0)", "Adulterated (1)"],
        yticklabels=["Pure (0)", "Adulterated (1)"],
    )
    plt.title(f"Confusion Matrix: {model_name.upper()}", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(roc_data: dict, save_path: str | Path):
    """Plot overlaid ROC curves for all models."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5.5))
    for model_name, data in roc_data.items():
        y_true = data["y_true"]
        y_score = data["y_score"]
        auc = data.get("auc", 0.0)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        label = f"{model_name.upper()} (AUC = {auc:.4f})" if auc is not None else model_name.upper()
        plt.plot(fpr, tpr, lw=2, label=label)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance (AUC = 0.50)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10)
    plt.title("ROC Curves Comparison (Held-out Test Set)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr_curves(pr_data: dict, save_path: str | Path):
    """Plot overlaid Precision-Recall curves for all models."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5.5))
    for model_name, data in pr_data.items():
        y_true = data["y_true"]
        y_score = data["y_score"]
        pr_auc = data.get("pr_auc", 0.0)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        label = f"{model_name.upper()} (PR-AUC = {pr_auc:.4f})" if pr_auc is not None else model_name.upper()
        plt.plot(rec, prec, lw=2, label=label)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall", fontsize=10)
    plt.ylabel("Precision", fontsize=10)
    plt.title("Precision-Recall Curves (Held-out Test Set)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison_bar(results_dict: dict, save_path: str | Path):
    """Plot multi-metric bar chart comparing all models on held-out test data."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    models = list(results_dict.keys())
    metrics_names = ["accuracy", "precision", "recall", "f1"]
    display_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

    x = np.arange(len(models))
    width = 0.18

    plt.figure(figsize=(9, 5))
    for i, (m_key, m_disp) in enumerate(zip(metrics_names, display_names)):
        vals = [results_dict[m][m_key] for m in models]
        plt.bar(x + i * width, vals, width, label=m_disp)

    plt.xlabel("Model", fontsize=11, fontweight="bold")
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
    plt.title("Model Performance Comparison on Held-out Test Set", fontsize=12, fontweight="bold")
    plt.xticks(x + width * 1.5, [m.upper() for m in models], fontsize=10)
    plt.ylim([0.0, 1.08])
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
