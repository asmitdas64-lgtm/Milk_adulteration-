"""Model training and cross-validation workflows."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from foodadult.models.registry import get_available_models, get_model


def evaluate_baseline_models(
    config: Dict[str, Any],
    model_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate all baseline models using Stratified K-Fold Cross-Validation on training data.

    Args:
        config: Configuration dictionary.
        model_keys: Optional list of specific models to run. If None, runs all registered models.

    Returns:
        Dictionary mapping each model key to its mean and std CV metrics.
    """
    processed_dir = Path(config["paths"]["processed_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    target_col = config["data"].get("target", "label")
    features = config["data"].get("features", [])
    random_state = int(config.get("random_state", 42))
    n_splits = int(config.get("cv", {}).get("n_splits", 5))

    train_path = processed_dir / "train.csv"
    train_df = pd.read_csv(train_path)

    X = train_df[features].copy()
    y = train_df[target_col].copy()

    if model_keys is None:
        model_keys = get_available_models()

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = {}
    summary_rows = []

    for key in model_keys:
        pipeline = get_model(key, config)
        cv_res = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        metrics = {
            "accuracy_mean": float(cv_res["test_accuracy"].mean()),
            "accuracy_std": float(cv_res["test_accuracy"].std()),
            "precision_mean": float(cv_res["test_precision"].mean()),
            "precision_std": float(cv_res["test_precision"].std()),
            "recall_mean": float(cv_res["test_recall"].mean()),
            "recall_std": float(cv_res["test_recall"].std()),
            "f1_mean": float(cv_res["test_f1"].mean()),
            "f1_std": float(cv_res["test_f1"].std()),
            "roc_auc_mean": float(cv_res["test_roc_auc"].mean()),
            "roc_auc_std": float(cv_res["test_roc_auc"].std()),
        }
        results[key] = metrics

        summary_rows.append({
            "model": key,
            "f1_mean": metrics["f1_mean"],
            "f1_std": metrics["f1_std"],
            "accuracy_mean": metrics["accuracy_mean"],
            "accuracy_std": metrics["accuracy_std"],
            "recall_mean": metrics["recall_mean"],
            "recall_std": metrics["recall_std"],
            "precision_mean": metrics["precision_mean"],
            "precision_std": metrics["precision_std"],
            "roc_auc_mean": metrics["roc_auc_mean"],
            "roc_auc_std": metrics["roc_auc_std"],
        })

    # Sort by f1_mean descending
    summary_df = pd.DataFrame(summary_rows).sort_values(by="f1_mean", ascending=False).reset_index(drop=True)

    # Save artifacts
    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "baseline_cv.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_df.to_csv(metrics_dir / "baseline_cv.csv", index=False)

    return results
