import numpy as np
import pandas as pd
import pytest
from foodadult.models.train import evaluate_baseline_models


@pytest.fixture
def mock_dataset(tmp_path):
    np.random.seed(42)
    n = 100
    # Two easily separable clusters
    x1 = np.concatenate([np.random.normal(-2.0, 0.5, 50), np.random.normal(2.0, 0.5, 50)])
    x2 = np.concatenate([np.random.normal(-2.0, 0.5, 50), np.random.normal(2.0, 0.5, 50)])
    y = np.array([0] * 50 + [1] * 50)

    train_df = pd.DataFrame({"x1": x1, "x2": x2, "label": y})
    train_csv = tmp_path / "train.csv"
    train_df.to_csv(train_csv, index=False)

    reports_dir = tmp_path / "reports"
    cfg = {
        "random_state": 42,
        "class_weight": None,
        "paths": {
            "processed_dir": str(tmp_path),
            "reports_dir": str(reports_dir),
        },
        "data": {
            "target": "label",
            "features": ["x1", "x2"],
        },
        "preprocess": {
            "imputer": "median",
            "outliers": "clip",
            "iqr_factor": 1.5,
            "scaler": "standard",
        },
        "cv": {
            "n_splits": 3,
            "shuffle": True,
        }
    }
    return cfg, reports_dir


def test_evaluate_baseline_models(mock_dataset):
    cfg, reports_dir = mock_dataset
    results = evaluate_baseline_models(cfg)

    # 1. Check all 6 models evaluated
    assert len(results) == 6
    assert "dummy" in results

    # 2. Check metrics calculated for each model
    for model_key, metrics in results.items():
        assert "f1_mean" in metrics
        assert "accuracy_mean" in metrics
        assert "recall_mean" in metrics
        assert "precision_mean" in metrics
        assert "roc_auc_mean" in metrics

    # 3. Core rule: all 5 real models must beat dummy on F1
    dummy_f1 = results["dummy"]["f1_mean"]
    for m in ["rf", "svm", "knn", "dt", "gb"]:
        assert results[m]["f1_mean"] > dummy_f1, f"Model {m} failed to beat dummy F1!"

    # 4. Check artifact files created
    metrics_dir = reports_dir / "metrics"
    assert (metrics_dir / "baseline_cv.json").is_file()
    assert (metrics_dir / "baseline_cv.csv").is_file()

