import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from foodadult.evaluation.evaluate import evaluate_models


@pytest.fixture
def eval_env(tmp_path):
    """Create a minimal synthetic train and test set and config for evaluation."""
    np.random.seed(42)
    # Train set (60 samples)
    x1_tr = np.concatenate([np.random.normal(-2.0, 0.5, 30), np.random.normal(2.0, 0.5, 30)])
    x2_tr = np.concatenate([np.random.normal(-2.0, 0.5, 30), np.random.normal(2.0, 0.5, 30)])
    y_tr = np.array([0] * 30 + [1] * 30)
    train_df = pd.DataFrame({"x1": x1_tr, "x2": x2_tr, "label": y_tr})

    # Test set (20 samples)
    x1_te = np.concatenate([np.random.normal(-2.0, 0.5, 10), np.random.normal(2.0, 0.5, 10)])
    x2_te = np.concatenate([np.random.normal(-2.0, 0.5, 10), np.random.normal(2.0, 0.5, 10)])
    y_te = np.array([0] * 10 + [1] * 10)
    test_df = pd.DataFrame({"x1": x1_te, "x2": x2_te, "label": y_te})

    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()
    train_df.to_csv(proc_dir / "train.csv", index=False)
    test_df.to_csv(proc_dir / "test.csv", index=False)

    reports_dir = tmp_path / "reports"
    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True)
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Synthetic best params for dt and knn
    best_params = {
        "dt": {"clf__max_depth": 3, "clf__min_samples_split": 2},
        "knn": {"clf__n_neighbors": 3, "clf__metric": "euclidean"},
    }
    with open(metrics_dir / "best_params.json", "w") as f:
        json.dump(best_params, f)

    cfg = {
        "random_state": 42,
        "class_weight": None,
        "paths": {
            "processed_dir": str(proc_dir),
            "reports_dir": str(reports_dir),
            "models_dir": str(models_dir),
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
        "models": {
            "dt": {"grid": {"max_depth": [3]}},
            "knn": {"grid": {"n_neighbors": [3]}},
        },
    }
    return cfg, tmp_path


def test_evaluate_models_runs_and_saves_artifacts(eval_env):
    """Verify evaluation fits on train, predicts on test once, and saves metrics, plots, and best model."""
    cfg, base_dir = eval_env
    results = evaluate_models(cfg, models_to_eval=["dt", "knn"])

    assert "dt" in results
    assert "knn" in results
    for m in ["dt", "knn"]:
        assert "f1" in results[m]
        assert "accuracy" in results[m]
        assert "roc_auc" in results[m]
        assert "confusion_matrix" in results[m]
        # Performance on easily separable fixture should be high
        assert results[m]["accuracy"] >= 0.85

    # Check that output files were written
    metrics_dir = Path(cfg["paths"]["reports_dir"]) / "metrics"
    assert (metrics_dir / "test_metrics.json").exists()
    assert (metrics_dir / "test_metrics.csv").exists()
    assert (metrics_dir / "test_metrics.md").exists()

    # Check model persistence
    models_dir = Path(cfg["paths"]["models_dir"])
    assert (models_dir / "best_model.joblib").exists()
    assert (models_dir / "best_model_meta.json").exists()

    with open(models_dir / "best_model_meta.json", "r") as f:
        meta = json.load(f)
    assert "model_name" in meta
    assert "test_metrics" in meta
    assert "features" in meta

