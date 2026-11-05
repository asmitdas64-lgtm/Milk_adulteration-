import json
from pathlib import Path
import pytest
from foodadult.evaluation.report import generate_results_markdown


@pytest.fixture
def report_env(tmp_path):
    reports_dir = tmp_path / "reports"
    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    feat_dir = reports_dir / "features"
    feat_dir.mkdir(parents=True)

    # Mock test_metrics.json
    test_metrics = {
        "svm": {
            "accuracy": 0.995283,
            "precision": 0.993827,
            "recall": 1.0,
            "f1": 0.996904,
            "specificity": 0.980392,
            "roc_auc": 1.0,
            "confusion_matrix": {"tn": 50, "fp": 1, "fn": 0, "tp": 161},
        },
        "rf": {
            "accuracy": 0.995283,
            "precision": 0.993827,
            "recall": 1.0,
            "f1": 0.996904,
            "specificity": 0.980392,
            "roc_auc": 0.999878,
            "confusion_matrix": {"tn": 50, "fp": 1, "fn": 0, "tp": 161},
        },
    }
    with open(metrics_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f)

    # Mock tuned_cv.json
    tuned_cv = {
        "selected_best_model": "svm",
        "models": {
            "svm": {"f1_mean": 0.9992, "accuracy_mean": 0.9988, "recall_mean": 1.0},
        },
    }
    with open(metrics_dir / "tuned_cv.json", "w") as f:
        json.dump(tuned_cv, f)

    return reports_dir


def test_generate_results_markdown_matches_metrics(report_env):
    """Verify generated RESULTS.md dynamically embeds exact numbers from metrics JSON."""
    out_file = generate_results_markdown(report_env)
    assert out_file.exists()

    content = out_file.read_text(encoding="utf-8")
    assert "99.53%" in content or "0.9953" in content or "0.9969" in content
    assert "SVM" in content
    assert "161" in content
    assert "TP" in content or "True Positives" in content

