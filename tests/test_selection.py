import numpy as np
import pandas as pd
import pytest
from foodadult.features.selection import (
    find_correlated_pairs,
    compute_rf_importance,
    compute_permutation_importance,
    run_rfecv_selection,
    run_feature_selection_pipeline,
)


@pytest.fixture
def mock_train_data():
    np.random.seed(42)
    n = 100
    # Create informative feature x1, noisy feature x2, and redundant feature x3 (= 2*x1 + tiny noise)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = 2.0 * x1 + np.random.normal(0, 0.01, n)
    # y depends heavily on x1
    prob = 1.0 / (1.0 + np.exp(-x1))
    y = (prob > 0.5).astype(int)

    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    return X, y


def test_find_correlated_pairs_detects_redundancy(mock_train_data):
    X, _ = mock_train_data
    pairs = find_correlated_pairs(X, threshold=0.90)
    assert len(pairs) == 1
    p = pairs[0]
    assert set([p["feature_a"], p["feature_b"]]) == {"x1", "x3"}
    assert p["correlation"] > 0.95


def test_compute_rf_importance(mock_train_data):
    X, y = mock_train_data
    ranking = compute_rf_importance(X, y, random_state=42)
    assert len(ranking) == 3
    assert set(ranking.keys()) == {"x1", "x2", "x3"}
    assert all(val >= 0.0 for val in ranking.values())


def test_run_rfecv_selection(mock_train_data):
    X, y = mock_train_data
    selected_feats = run_rfecv_selection(X, y, random_state=42)
    assert isinstance(selected_feats, list)
    assert len(selected_feats) >= 1
    assert set(selected_feats).issubset({"x1", "x2", "x3"})


def test_feature_selection_pipeline_writes_artifacts(tmp_path, mock_train_data):
    X, y = mock_train_data
    train_df = X.copy()
    train_df["label"] = y

    train_csv = tmp_path / "train.csv"
    train_df.to_csv(train_csv, index=False)
    reports_dir = tmp_path / "reports"

    cfg = {
        "paths": {
            "processed_dir": str(tmp_path),
            "reports_dir": str(reports_dir),
        },
        "data": {
            "target": "label",
            "features": ["x1", "x2", "x3"],
        },
        "features": {
            "corr_threshold": 0.90,
            "use_selected_subset": False,
        },
        "random_state": 42,
    }

    result = run_feature_selection_pipeline(cfg)
    assert "ranking_df" in result
    assert (reports_dir / "features" / "feature_ranking.csv").is_file()
    assert (reports_dir / "features" / "selected_features.json").is_file()
    assert (reports_dir / "figures" / "feat_importance.png").is_file()
