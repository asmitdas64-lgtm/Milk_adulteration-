import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict:
    """Compute binary classification metrics for pure (0) vs adulterated (1).

    Metrics computed:
    - accuracy: (TP + TN) / total
    - precision: TP / (TP + FP) (positive = 1)
    - recall: TP / (TP + FN) (sensitivity for adulterated class)
    - f1: harmonic mean of precision and recall
    - specificity: TN / (TN + FP) (true negative rate for pure class)
    - balanced_accuracy: (recall + specificity) / 2
    - roc_auc: Area Under ROC Curve (requires y_score)
    - pr_auc: Area Under Precision-Recall Curve (average precision, requires y_score)
    - confusion_matrix: {'tn': int, 'fp': int, 'fn': int, 'tp': int}
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Confusion matrix
    # sklearn confusion_matrix returns [[tn, fp], [fn, tp]] for labels=[0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    roc_auc = None
    pr_auc = None
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            roc_auc = None
        try:
            pr_auc = float(average_precision_score(y_true, y_score))
        except Exception:
            pr_auc = None

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "raw": [[tn, fp], [fn, tp]],
        },
    }
