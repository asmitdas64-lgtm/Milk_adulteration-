from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from foodadult.evaluation.interpret import (
    compute_test_permutation_importance,
    compare_feature_rankings,
    plot_decision_tree_splits,
    plot_permutation_importance,
    run_interpretability_analysis,
)


@pytest.fixture
def interpret_env(tmp_path):
    """Create a minimal synthetic setup for interpretability testing."""
    np.random.seed(42)
    # Feature x1 has strong signal; x2 is mostly noise
    x1_tr = np.concatenate([np.random.normal(-3.0, 0.5, 30), np.random.normal(3.0, 0.5, 30)])
    x2_tr = np.random.normal(0, 1.0, 60)
    y_tr = np.array([0] * 30 + [1] * 30)
    train_df = pd.DataFrame({"x1": x1_tr, "x2": x2_tr, "label": y_tr})

    x1_te = np.concatenate([np.random.normal(-3.0, 0.5, 10), np.random.normal(3.0, 0.5, 10)])
    x2_te = np.random.normal(0, 1.0, 20)
    y_te = np.array([0] * 10 + [1] * 10)
    test_df = pd.DataFrame({"x1": x1_te, "x2": x2_te, "label": y_te})

    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()
    train_df.to_csv(proc_dir / "train.csv", index=False)
    test_df.to_csv(proc_dir / "test.csv", index=False)

    reports_dir = tmp_path / "reports"
    feat_dir = reports_dir / "features"
    feat_dir.mkdir(parents=True)
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True)
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create dummy Phase 4 ranking
    phase4_ranking = pd.DataFrame({
        "feature": ["x1", "x2"],
        "rf_importance": [0.85, 0.15],
        "perm_importance_mean": [0.35, 0.02],
    })
    phase4_ranking.to_csv(feat_dir / "feature_ranking.csv", index=False)

    # Train a minimal pipeline and save
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=3, random_state=42)),
    ])
    pipe.fit(train_df[["x1", "x2"]], y_tr)

    import joblib
    joblib.dump(pipe, models_dir / "best_model.joblib")

    cfg = {
        "random_state": 42,
        "paths": {
            "processed_dir": str(proc_dir),
            "reports_dir": str(reports_dir),
            "models_dir": str(models_dir),
        },
        "data": {
            "target": "label",
            "features": ["x1", "x2"],
        },
    }
    return cfg, pipe, test_df[["x1", "x2"]], y_te, tmp_path


def test_permutation_importance_has_all_features_and_finite(interpret_env):
    cfg, pipe, X_test, y_test, _ = interpret_env
    imp_df = compute_test_permutation_importance(pipe, X_test, y_test, scoring="f1", random_state=42)

    assert set(imp_df["feature"]) == {"x1", "x2"}
    assert np.all(np.isfinite(imp_df["importance_mean"]))
    assert np.all(np.isfinite(imp_df["importance_std"]))
    # x1 had the strong signal, so its importance should be higher than x2
    assert imp_df.loc[imp_df["feature"] == "x1", "importance_mean"].values[0] > \
           imp_df.loc[imp_df["feature"] == "x2", "importance_mean"].values[0]


def test_compare_feature_rankings(interpret_env):
    cfg, _, _, _, tmp_path = interpret_env
    feat_dir = Path(cfg["paths"]["reports_dir"]) / "features"
    test_imp_df = pd.DataFrame({
        "feature": ["x1", "x2"],
        "importance_mean": [0.45, 0.01],
        "importance_std": [0.02, 0.01],
    })

    comp_df = compare_feature_rankings(test_imp_df, feat_dir / "feature_ranking.csv")
    assert "feature" in comp_df.columns
    assert "test_perm_importance" in comp_df.columns
    assert "train_rf_importance" in comp_df.columns
    assert len(comp_df) == 2


def test_run_interpretability_pipeline(interpret_env):
    cfg, _, _, _, tmp_path = interpret_env
    results = run_interpretability_analysis(cfg)

    reports_dir = Path(cfg["paths"]["reports_dir"])
    assert (reports_dir / "features" / "final_importance.csv").exists()
    assert (reports_dir / "features" / "interpretability_summary.md").exists()
    assert (reports_dir / "figures" / "interpret_perm_importance.png").exists()
    assert (reports_dir / "figures" / "tree_interpretability.png").exists()
    assert "ranking" in results
