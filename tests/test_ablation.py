from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from foodadult.evaluation.ablation import (
    run_ablation_outliers,
    run_ablation_features,
    run_ablation_class_weight,
    run_mcnemar_test,
    run_all_ablations,
)


@pytest.fixture
def ablation_env(tmp_path):
    """Create minimal synthetic dataset for testing ablations."""
    np.random.seed(42)
    n = 60
    x1 = np.concatenate([np.random.normal(-2.0, 0.5, 30), np.random.normal(2.0, 0.5, 30)])
    x2 = np.concatenate([np.random.normal(-2.0, 0.5, 30), np.random.normal(2.0, 0.5, 30)])
    # Add a couple outliers
    x1[0] = 50.0
    x2[1] = -40.0
    y = np.array([0] * 30 + [1] * 30)

    train_df = pd.DataFrame({"x1": x1, "x2": x2, "label": y})
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()
    train_df.to_csv(proc_dir / "train.csv", index=False)
    train_df.to_csv(proc_dir / "test.csv", index=False)

    reports_dir = tmp_path / "reports"
    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    models_dir = tmp_path / "models"
    models_dir.mkdir()

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
        "cv": {
            "n_splits": 3,
            "shuffle": True,
            "scoring": "f1",
        },
        "models": {
            "dt": {
                "grid": {"max_depth": [3]},
            },
        },
    }
    return cfg, tmp_path


def test_mcnemar_test_identical_predictions():
    """McNemar p-value should be 1.0 when both models make identical predictions."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred_a = np.array([0, 1, 0, 1, 1])
    y_pred_b = np.array([0, 1, 0, 1, 1])

    res = run_mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert res["n01"] == 0
    assert res["n10"] == 0
    assert pytest.approx(res["p_value"], abs=1e-5) == 1.0


def test_mcnemar_test_different_predictions():
    """McNemar detects discordant predictions."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred_a = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # perfect
    y_pred_b = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # all wrong

    res = run_mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert res["n01"] == 8  # A correct, B wrong
    assert res["n10"] == 0  # A wrong, B correct
    assert res["p_value"] < 0.05


def test_run_all_ablations_produces_table_and_markdown(ablation_env):
    """Run all ablations and ensure summary table and markdown are produced."""
    cfg, tmp_path = ablation_env
    df = run_all_ablations(cfg, model_name="dt")

    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 5
    assert "ablation_type" in df.columns
    assert "setting" in df.columns
    assert "cv_f1_mean" in df.columns

    reports_dir = Path(cfg["paths"]["reports_dir"])
    assert (reports_dir / "metrics" / "ablations.csv").exists()
    assert (reports_dir / "metrics" / "ablations_summary.md").exists()
