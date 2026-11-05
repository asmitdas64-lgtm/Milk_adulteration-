"""Model training, cross-validation, and hyperparameter tuning workflows."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate

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

    summary_df = pd.DataFrame(summary_rows).sort_values(by="f1_mean", ascending=False).reset_index(drop=True)

    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "baseline_cv.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_df.to_csv(metrics_dir / "baseline_cv.csv", index=False)

    return results


def tune_model(model_key: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Tune a single model's hyperparameters using GridSearchCV or RandomizedSearchCV.

    Args:
        model_key: Model key in config['models'].
        config: Configuration dictionary.

    Returns:
        Dictionary with best_params, best_score, best_estimator, and cv_results_df.
    """
    processed_dir = Path(config["paths"]["processed_dir"])
    target_col = config["data"].get("target", "label")
    features = config["data"].get("features", [])
    random_state = int(config.get("random_state", 42))
    n_splits = int(config.get("cv", {}).get("n_splits", 5))

    train_path = processed_dir / "train.csv"
    train_df = pd.read_csv(train_path)

    X = train_df[features].copy()
    y = train_df[target_col].copy()

    pipeline = get_model(model_key, config)
    model_cfg = config.get("models", {}).get(model_key, {})

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    if "grid" in model_cfg:
        param_grid = {f"clf__{k}": v for k, v in model_cfg["grid"].items()}
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit="f1",
            cv=cv,
            n_jobs=-1,
        )
    elif "random" in model_cfg:
        param_dist = {f"clf__{k}": v for k, v in model_cfg["random"].items()}
        n_iter = int(model_cfg.get("n_iter", 20))
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            refit="f1",
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Model '{model_key}' has neither 'grid' nor 'random' config defined.")

    search.fit(X, y)
    cv_results_df = pd.DataFrame(search.cv_results_)

    best_idx = search.best_index_
    best_metrics = {
        "f1_mean": float(cv_results_df.loc[best_idx, "mean_test_f1"]),
        "f1_std": float(cv_results_df.loc[best_idx, "std_test_f1"]),
        "accuracy_mean": float(cv_results_df.loc[best_idx, "mean_test_accuracy"]),
        "accuracy_std": float(cv_results_df.loc[best_idx, "std_test_accuracy"]),
        "recall_mean": float(cv_results_df.loc[best_idx, "mean_test_recall"]),
        "recall_std": float(cv_results_df.loc[best_idx, "std_test_recall"]),
        "precision_mean": float(cv_results_df.loc[best_idx, "mean_test_precision"]),
        "precision_std": float(cv_results_df.loc[best_idx, "std_test_precision"]),
        "roc_auc_mean": float(cv_results_df.loc[best_idx, "mean_test_roc_auc"]),
        "roc_auc_std": float(cv_results_df.loc[best_idx, "std_test_roc_auc"]),
    }

    return {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "best_metrics": best_metrics,
        "best_estimator": search.best_estimator_,
        "cv_results_df": cv_results_df,
    }


def tune_all_models(
    config: Dict[str, Any],
    model_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Tune all specified models and record hyperparameter results and best model."""
    reports_dir = Path(config["paths"]["reports_dir"])
    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if model_keys is None:
        model_keys = [k for k in config.get("models", {}).keys()]

    all_tuning_results = {}
    best_params_map = {}
    tuned_cv_summary = {}
    summary_rows = []

    for key in model_keys:
        res = tune_model(key, config)
        all_tuning_results[key] = res
        best_params_map[key] = res["best_params"]
        tuned_cv_summary[key] = res["best_metrics"]

        # Save individual model's tuning grid csv
        res["cv_results_df"].to_csv(metrics_dir / f"tuning_{key}.csv", index=False)

        metrics = res["best_metrics"]
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

    # Save summary tables and maps
    summary_df = pd.DataFrame(summary_rows).sort_values(by="f1_mean", ascending=False).reset_index(drop=True)
    summary_df.to_csv(metrics_dir / "tuned_cv.csv", index=False)

    with open(metrics_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_map, f, indent=2)

    # Decision D-10: Select best model by CV F1 -> tiebreak recall -> interpretability
    best_model_name = summary_df.iloc[0]["model"]
    top_models = summary_df[summary_df["f1_mean"] >= summary_df.iloc[0]["f1_mean"] - 0.01]
    if len(top_models) > 1:
        # Tie-break by recall
        best_model_name = top_models.sort_values(by="recall_mean", ascending=False).iloc[0]["model"]

    with open(metrics_dir / "tuned_cv.json", "w", encoding="utf-8") as f:
        json.dump({
            "models": tuned_cv_summary,
            "selected_best_model": best_model_name,
            "selection_rule": "Highest CV F1 with Recall tie-breaker (Decision D-10)",
        }, f, indent=2)

    return all_tuning_results
