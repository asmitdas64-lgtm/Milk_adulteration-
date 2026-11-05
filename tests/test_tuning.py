from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from foodadult.models.train import tune_model, tune_all_models


@pytest.fixture
def tiny_dataset(tmp_path):
    np.random.seed(42)
    n = 60
    x1 = np.concatenate([np.random.normal(-2.0, 0.5, 30), np.random.normal(2.0, 0.5, 30)])
    x2 = np.concatenate([np.random.normal(-2.0, 0.5, 30), np.random.normal(2.0, 0.5, 30)])
    y = np.array([0] * 30 + [1] * 30)

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
            "scoring": "f1",
        },
        "models": {
            "dt": {
                "grid": {
                    "max_depth": [3, 5],
                    "min_samples_split": [2, 5],
                }
            },
            "knn": {
                "grid": {
                    "n_neighbors": [1, 3],
                    "metric": ["euclidean"],
                }
            }
        }
    }
    return cfg, reports_dir


def test_tune_single_model_dt(tiny_dataset):
    cfg, _ = tiny_dataset
    res = tune_model("dt", cfg)

    assert "best_params" in res
    assert "best_score" in res
    assert "cv_results_df" in res
    assert res["best_score"] > 0.8
    # Best parameters should belong to the search grid
    assert res["best_params"]["clf__max_depth"] in [3, 5]
    assert res["best_params"]["clf__min_samples_split"] in [2, 5]


def test_tune_all_models_creates_artifacts(tiny_dataset):
    cfg, reports_dir = tiny_dataset
    all_res = tune_all_models(cfg, model_keys=["dt", "knn"])

    assert "dt" in all_res
    assert "knn" in all_res

    metrics_dir = reports_dir / "metrics"
    assert (metrics_dir / "best_params.json").is_file()
    assert (metrics_dir / "tuned_cv.json").is_file()
    assert (metrics_dir / "tuning_dt.csv").is_file()
    assert (metrics_dir / "tuning_knn.csv").is_file()
