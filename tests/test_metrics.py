import numpy as np
import pytest
from foodadult.evaluation.metrics import compute_metrics


def test_compute_metrics_hand_calculated():
    """Verify metrics match exact hand-calculated values for a known confusion matrix."""
    # 5 samples:
    # y_true: [1, 1, 0, 0, 1] -> 3 positives, 2 negatives
    # y_pred: [1, 0, 0, 0, 1]
    # TP: 2, FN: 1, TN: 2, FP: 0
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    y_score = np.array([0.9, 0.4, 0.1, 0.2, 0.95])

    res = compute_metrics(y_true, y_pred, y_score)

    assert pytest.approx(res["accuracy"], abs=1e-5) == 4 / 5  # 0.80
    assert pytest.approx(res["precision"], abs=1e-5) == 2 / 2  # 1.00
    assert pytest.approx(res["recall"], abs=1e-5) == 2 / 3  # ~0.66667
    # F1 = 2 * (1.0 * 2/3) / (1.0 + 2/3) = 4/5 = 0.80
    assert pytest.approx(res["f1"], abs=1e-5) == 0.80
    assert pytest.approx(res["specificity"], abs=1e-5) == 2 / 2  # 1.00

    cm = res["confusion_matrix"]
    # CM: [[TN, FP], [FN, TP]] -> [[2, 0], [1, 2]]
    assert cm["tn"] == 2
    assert cm["fp"] == 0
    assert cm["fn"] == 1
    assert cm["tp"] == 2


def test_compute_metrics_perfect_ranking_auc():
    """Verify ROC-AUC is 1.0 when scores perfectly separate positive and negative classes."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    res = compute_metrics(y_true, y_pred, y_score)
    assert pytest.approx(res["roc_auc"], abs=1e-5) == 1.0
    assert pytest.approx(res["pr_auc"], abs=1e-5) == 1.0
    assert pytest.approx(res["f1"], abs=1e-5) == 1.0
    assert pytest.approx(res["accuracy"], abs=1e-5) == 1.0


def test_compute_metrics_without_y_score():
    """Verify compute_metrics works when probability scores are not available."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])

    res = compute_metrics(y_true, y_pred, y_score=None)
    assert res["roc_auc"] is None
    assert res["pr_auc"] is None
    assert pytest.approx(res["recall"], abs=1e-5) == 0.5
