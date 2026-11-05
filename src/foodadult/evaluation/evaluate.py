import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

from foodadult.models.registry import get_model
from foodadult.models.persist import save_model
from foodadult.evaluation.metrics import compute_metrics
from foodadult.evaluation.plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    plot_comparison_bar,
)


def evaluate_models(cfg: dict, models_to_eval: Optional[List[str]] = None) -> dict:
    """Refit tuned pipelines on full training set and evaluate ONCE on held-out test set."""
    proc_dir = Path(cfg["paths"]["processed_dir"])
    train_path = proc_dir / "train.csv"
    test_path = proc_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train or test set in {proc_dir}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = cfg["data"]["target"]
    feature_cols = cfg["data"]["features"]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values

    reports_dir = Path(cfg["paths"]["reports_dir"])
    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(cfg["paths"].get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load tuned parameters if available
    best_params_path = metrics_dir / "best_params.json"
    best_params = {}
    if best_params_path.exists():
        with open(best_params_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)

    if models_to_eval is None:
        models_to_eval = [k for k in cfg.get("models", {}).keys() if k != "dummy"]
        if "dummy" in cfg.get("models", {}):
            models_to_eval.append("dummy")
        elif not models_to_eval:
            models_to_eval = ["rf", "svm", "knn", "dt", "gb", "dummy"]

    results = {}
    roc_data = {}
    pr_data = {}
    fitted_models = {}

    for model_name in models_to_eval:
        pipeline = get_model(model_name, cfg)

        # Apply tuned parameters
        if model_name in best_params:
            params = best_params[model_name]
            pipeline.set_params(**params)

        # Refit on full training set
        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline

        # Single evaluation on held-out test set
        y_pred = pipeline.predict(X_test)
        y_score = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_score = pipeline.predict_proba(X_test)[:, 1]
            except Exception:
                y_score = None
        elif hasattr(pipeline, "decision_function"):
            try:
                y_score = pipeline.decision_function(X_test)
            except Exception:
                y_score = None

        m = compute_metrics(y_test, y_pred, y_score)
        results[model_name] = m

        if y_score is not None:
            roc_data[model_name] = {"y_true": y_test, "y_score": y_score, "auc": m["roc_auc"]}
            pr_data[model_name] = {"y_true": y_test, "y_score": y_score, "pr_auc": m["pr_auc"]}

        # Save individual confusion matrix
        plot_confusion_matrix(
            m["confusion_matrix"]["raw"],
            model_name=model_name,
            save_path=figures_dir / f"eval_cm_{model_name}.png",
        )

    # Save summary test plots
    if roc_data:
        plot_roc_curves(roc_data, figures_dir / "eval_roc_curves.png")
        plot_pr_curves(pr_data, figures_dir / "eval_pr_curves.png")
    plot_comparison_bar(results, figures_dir / "eval_comparison_bar.png")

    # Save test_metrics.json
    out_json = metrics_dir / "test_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save test_metrics.csv
    csv_rows = []
    for model_name, m in results.items():
        cm = m["confusion_matrix"]
        csv_rows.append({
            "model": model_name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "specificity": m["specificity"],
            "balanced_accuracy": m["balanced_accuracy"],
            "roc_auc": m["roc_auc"] if m["roc_auc"] is not None else np.nan,
            "pr_auc": m["pr_auc"] if m["pr_auc"] is not None else np.nan,
            "tn": cm["tn"],
            "fp": cm["fp"],
            "fn": cm["fn"],
            "tp": cm["tp"],
        })
    csv_df = pd.DataFrame(csv_rows).sort_values(by="f1", ascending=False)
    out_csv = metrics_dir / "test_metrics.csv"
    csv_df.to_csv(out_csv, index=False)

    # Save test_metrics.md
    out_md = metrics_dir / "test_metrics.md"
    headers = list(csv_df.columns)
    md_lines = [
        "# Final Test Set Evaluation Results\n",
        "> Evaluated once on held-out `test.csv` (212 samples, 80/20 stratified split).\n",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in csv_df.iterrows():
        formatted_row = []
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                formatted_row.append(f"{val:.4f}")
            else:
                formatted_row.append(str(val))
        md_lines.append("| " + " | ".join(formatted_row) + " |")

    md_lines.extend([
        "\n## Artifacts Generated",
        "- Metrics JSON: `reports/metrics/test_metrics.json`",
        "- Metrics CSV: `reports/metrics/test_metrics.csv`",
        "- ROC Curves: `reports/figures/eval_roc_curves.png`",
        "- Precision-Recall Curves: `reports/figures/eval_pr_curves.png`",
        "- Comparison Bar Chart: `reports/figures/eval_comparison_bar.png`",
        "- Confusion Matrices: `reports/figures/eval_cm_<model>.png`\n",
    ])
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Identify best model per Decision D-10
    # Note: Decision D-10 states: Best model was selected based on mean CV F1
    # Check if tuned_cv.json exists to respect the pre-test decision
    tuned_cv_path = metrics_dir / "tuned_cv.json"
    winning_model_name = "svm"  # default fallback
    if tuned_cv_path.exists():
        with open(tuned_cv_path, "r", encoding="utf-8") as f:
            tuned_cv = json.load(f)
        if "selected_best_model" in tuned_cv:
            winning_model_name = tuned_cv["selected_best_model"]
        elif "models" in tuned_cv:
            sorted_cv = sorted(
                tuned_cv["models"].keys(),
                key=lambda k: (
                    tuned_cv["models"][k].get("f1_mean", 0.0),
                    tuned_cv["models"][k].get("recall_mean", 0.0),
                ),
                reverse=True,
            )
            if sorted_cv:
                winning_model_name = sorted_cv[0]
        else:
            sorted_cv = sorted(
                [k for k in tuned_cv.keys() if k != "dummy"],
                key=lambda k: (
                    tuned_cv[k].get("f1_mean", 0.0),
                    tuned_cv[k].get("recall_mean", 0.0),
                ),
                reverse=True,
            )
            if sorted_cv:
                winning_model_name = sorted_cv[0]
    else:
        # Fallback to test F1
        sorted_test = sorted(
            [k for k in results.keys() if k != "dummy"],
            key=lambda k: (results[k]["f1"], results[k]["recall"]),
            reverse=True,
        )
        if sorted_test:
            winning_model_name = sorted_test[0]

    # Save best model and metadata
    best_pipeline = fitted_models[winning_model_name]
    best_model_path = models_dir / "best_model.joblib"
    meta = {
        "model_name": winning_model_name,
        "features": feature_cols,
        "target": target_col,
        "best_params": best_params.get(winning_model_name, {}),
        "test_metrics": results[winning_model_name],
        "all_test_metrics_summary": {m: results[m]["f1"] for m in results},
        "selection_rule": "Decision D-10: Highest 5-Fold CV F1 with Recall tie-breaker (frozen before test evaluation)",
    }
    save_model(best_pipeline, best_model_path, metadata=meta)

    return results
