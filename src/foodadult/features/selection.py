"""Feature selection and ranking methods (Pearson, RF importance, Permutation, RFECV)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold


def find_correlated_pairs(X: pd.DataFrame, threshold: float = 0.90) -> List[Dict[str, Any]]:
    """Identify highly correlated feature pairs above threshold."""
    corr_matrix = X.corr().abs()
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = float(corr_matrix.iloc[i, j])
            if val >= threshold:
                pairs.append({
                    "feature_a": cols[i],
                    "feature_b": cols[j],
                    "correlation": val,
                })
    return pairs


def compute_rf_importance(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Dict[str, float]:
    """Compute Random Forest MDI (Gini impurity) feature importances."""
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    return {col: float(imp) for col, imp in zip(X.columns, importances)}


def compute_permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_repeats: int = 10,
) -> Dict[str, float]:
    """Compute Permutation feature importance on training data using F1 score."""
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, scoring="f1", n_repeats=n_repeats, random_state=random_state)
    return {col: float(val) for col, val in zip(X.columns, perm.importances_mean)}


def run_rfecv_selection(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
) -> List[str]:
    """Run Recursive Feature Elimination with Cross-Validation (RFECV)."""
    estimator = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring="f1",
        min_features_to_select=1,
        n_jobs=-1,
    )
    rfecv.fit(X, y)
    selected = [col for col, support in zip(X.columns, rfecv.support_) if support]
    return selected


def plot_feature_importance(ranking_df: pd.DataFrame, save_path: Path):
    """Plot comparative horizontal bar chart of feature importances."""
    plt.figure(figsize=(9, 5))
    plot_df = ranking_df.sort_values(by="rf_importance", ascending=True)

    y_pos = np.arange(len(plot_df))
    plt.barh(y_pos, plot_df["rf_importance"], color="#1f77b4", alpha=0.85, edgecolor="black", label="RF Impurity Importance")

    plt.yticks(y_pos, plot_df["feature"], fontsize=10, fontweight="bold")
    plt.xlabel("Relative Importance Score", fontsize=10)
    plt.title("Feature Importance Hierarchy (Random Forest)", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for i, v in enumerate(plot_df["rf_importance"]):
        plt.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_feature_selection_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute complete Phase 4 feature selection pipeline on training set."""
    processed_dir = Path(config["paths"]["processed_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    target_col = config["data"].get("target", "label")
    features = config["data"].get("features", [])
    threshold = float(config.get("features", {}).get("corr_threshold", 0.90))
    random_state = int(config.get("random_state", 42))

    train_path = processed_dir / "train.csv"
    train_df = pd.read_csv(train_path)

    X = train_df[features].copy()
    y = train_df[target_col].copy()

    # 1. Pearson correlations
    corr_pairs = find_correlated_pairs(X, threshold=threshold)

    # 2. Random Forest Importance
    rf_imp = compute_rf_importance(X, y, random_state=random_state)

    # 3. Permutation Importance
    perm_imp = compute_permutation_importance(X, y, random_state=random_state)

    # 4. RFECV Selection
    selected_features = run_rfecv_selection(X, y, random_state=random_state)

    # 5. Compile feature_ranking.csv
    ranking_data = []
    for feat in features:
        ranking_data.append({
            "feature": feat,
            "rf_importance": rf_imp.get(feat, 0.0),
            "perm_importance": perm_imp.get(feat, 0.0),
            "rfecv_selected": feat in selected_features,
        })

    ranking_df = pd.DataFrame(ranking_data).sort_values(by="rf_importance", ascending=False).reset_index(drop=True)

    # Save ranking CSV
    feat_reports_dir = reports_dir / "features"
    feat_reports_dir.mkdir(parents=True, exist_ok=True)
    ranking_csv_path = feat_reports_dir / "feature_ranking.csv"
    ranking_df.to_csv(ranking_csv_path, index=False)

    # Save selected_features.json
    selected_json_path = feat_reports_dir / "selected_features.json"
    with open(selected_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "all_features": features,
            "selected_features": selected_features,
            "correlated_pairs_above_threshold": corr_pairs,
        }, f, indent=2)

    # Save bar chart
    fig_path = reports_dir / "figures" / "feat_importance.png"
    plot_feature_importance(ranking_df, fig_path)

    return {
        "ranking_df": ranking_df,
        "selected_features": selected_features,
        "correlated_pairs": corr_pairs,
    }
