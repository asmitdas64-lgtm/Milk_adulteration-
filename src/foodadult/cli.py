"""Command-line interface for the Food Adulteration ML Pipeline."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

from foodadult.config import load_config
from foodadult.data.load import load_raw, standardize_columns, make_binary_label
from foodadult.data.split import perform_stratified_split
from foodadult.data.validate import validate_dataframe
from foodadult.eda import run_eda
from foodadult.features.selection import run_feature_selection_pipeline
from foodadult.logging_utils import create_run_directory, setup_logger
from foodadult.models.train import evaluate_baseline_models, tune_all_models
from foodadult.evaluation.evaluate import evaluate_models
from foodadult.evaluation.interpret import run_interpretability_analysis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Food Adulteration ML Pipeline CLI — MIT Bengaluru CSE Minor Project",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: validate
    cmd_validate = subparsers.add_parser("validate", help="Load & validate raw dataset")
    cmd_validate.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")

    # Command: eda
    cmd_eda = subparsers.add_parser("eda", help="Run exploratory data analysis")
    cmd_eda.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")

    # Command: split
    cmd_split = subparsers.add_parser("split", help="Split data into train & test sets")
    cmd_split.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")

    # Command: select-features
    cmd_features = subparsers.add_parser("select-features", help="Compute feature importance & selection")
    cmd_features.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")

    # Command: train
    cmd_train = subparsers.add_parser("train", help="Train baseline or tuned ML models")
    cmd_train.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    cmd_train.add_argument("--models", nargs="+", default=None, help="Specific models to train/tune")
    cmd_train.add_argument("--tune", action="store_true", help="Execute hyperparameter tuning search")

    # Command: evaluate
    cmd_eval = subparsers.add_parser("evaluate", help="Evaluate models on test set once")
    cmd_eval.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")

    # Command: interpret
    cmd_interp = subparsers.add_parser("interpret", help="Run interpretability & feature importance analysis")
    cmd_interp.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")

    # Command: predict
    cmd_pred = subparsers.add_parser("predict", help="Predict adulteration on new samples CSV")
    cmd_pred.add_argument("--model", required=True, help="Path to saved .joblib model")
    cmd_pred.add_argument("--input", required=True, help="Path to input samples CSV")
    cmd_pred.add_argument("--output", required=True, help="Path to save predictions CSV")

    # Command: report
    cmd_report = subparsers.add_parser("report", help="Generate final markdown report summary")
    cmd_report.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")

    # Command: all
    cmd_all = subparsers.add_parser("all", help="Execute complete pipeline end-to-end")
    cmd_all.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")

    return parser.parse_args()


def run_validate(config_path: str):
    cfg = load_config(config_path)
    reports_dir = Path(cfg["paths"]["reports_dir"])
    run_dir, logger = create_run_directory(reports_dir, cfg)

    logger.info("Starting raw dataset loading and validation...")
    raw_path = Path(cfg["paths"]["raw"])
    logger.info(f"Reading raw data from: {raw_path}")
    raw_df = load_raw(raw_path)
    logger.info(f"Loaded {len(raw_df)} rows and {len(raw_df.columns)} columns.")

    # Standardize columns
    rename_map = cfg["data"].get("column_rename", {})
    clean_df = standardize_columns(raw_df, rename_map)

    # Map target label
    raw_target_col = cfg["data"].get("raw_target", "Grade").lower()
    label_col = cfg["data"].get("target", "label")
    label_map = cfg["data"].get("label_mapping", {"low": 1, "medium": 1, "high": 0})

    if raw_target_col in clean_df.columns:
        logger.info(f"Mapping raw target column '{raw_target_col}' into binary '{label_col}'...")
        clean_df[label_col] = make_binary_label(clean_df[raw_target_col], label_map)
        clean_df = clean_df.drop(columns=[raw_target_col])
    elif label_col not in clean_df.columns:
        raise ValueError(f"Neither '{raw_target_col}' nor '{label_col}' found in data.")

    # Validate DataFrame
    logger.info("Executing validation checks against schema and physical bounds...")
    report = validate_dataframe(clean_df, cfg)
    logger.info(f"Validation successful: rows={report['row_count']}, duplicates={report['duplicate_count']}")

    if report["out_of_range"]:
        logger.warning(f"Out of physical bounds warnings: {report['out_of_range']}")

    # Save to data/interim/clean.csv
    interim_path = Path(cfg["paths"]["interim"])
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(interim_path, index=False)
    logger.info(f"Clean interim data successfully saved to: {interim_path}")

    # Log class balance
    class_counts = clean_df[label_col].value_counts().to_dict()
    logger.info(f"Target binary class balance: {class_counts} (1=Adulterated/Substandard, 0=Pure)")
    print(f"\n[SUCCESS] Clean interim dataset saved to: {interim_path}")
    print(f"Total samples: {len(clean_df)} | Class distribution: {class_counts}")


def execute_eda(config_path: str):
    cfg = load_config(config_path)
    reports_dir = Path(cfg["paths"]["reports_dir"])
    run_dir, logger = create_run_directory(reports_dir, cfg)

    logger.info("Starting Exploratory Data Analysis (EDA)...")
    result = run_eda(cfg)
    summary = result["summary"]
    
    logger.info(f"EDA Complete. Rows analyzed: {summary['row_count']}")
    logger.info(f"Class distribution: {summary['class_balance']['counts']}")
    logger.info(f"Artifacts saved to {reports_dir}/eda and {reports_dir}/figures")

    print(f"\n[SUCCESS] Exploratory Data Analysis completed!")
    print(f"- Data Card: {reports_dir}/eda/data_card.md")
    print(f"- Summary JSON: {reports_dir}/eda/summary.json")
    print(f"- Figures: {reports_dir}/figures/ (class balance, distributions, correlation matrix)")


def execute_split(config_path: str):
    cfg = load_config(config_path)
    reports_dir = Path(cfg["paths"]["reports_dir"])
    run_dir, logger = create_run_directory(reports_dir, cfg)

    logger.info("Starting Stratified Train/Test Partitioning...")
    train_df, test_df, split_info = perform_stratified_split(cfg)

    logger.info(f"Partitioning complete. Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    logger.info(f"Train class balance: {split_info['train_class_balance']}")
    logger.info(f"Test class balance: {split_info['test_class_balance']}")
    logger.info(f"Saved artifacts to {cfg['paths']['processed_dir']}/train.csv, test.csv, split.json")

    print(f"\n[SUCCESS] Stratified Train/Test split complete!")
    print(f"- Train set: {len(train_df)} samples ({split_info['train_class_balance']})")
    print(f"- Test set:  {len(test_df)} samples ({split_info['test_class_balance']}) [LOCKED]")
    print(f"- Saved to:  {cfg['paths']['processed_dir']}/")


def execute_select_features(config_path: str):
    cfg = load_config(config_path)
    reports_dir = Path(cfg["paths"]["reports_dir"])
    run_dir, logger = create_run_directory(reports_dir, cfg)

    logger.info("Starting Feature Selection and Ranking Pipeline...")
    result = run_feature_selection_pipeline(cfg)
    ranking_df = result["ranking_df"]
    selected = result["selected_features"]
    corr_pairs = result["correlated_pairs"]

    logger.info(f"Feature ranking completed. Top feature: {ranking_df.iloc[0]['feature']}")
    logger.info(f"RFECV selected {len(selected)} features: {selected}")
    if corr_pairs:
        logger.info(f"Highly correlated feature pairs (|r| >= threshold): {corr_pairs}")
    else:
        logger.info("No pairwise collinearity exceeding threshold found.")

    print(f"\n[SUCCESS] Feature Selection & Ranking complete!")
    print(f"- Ranked Features Table:\n{ranking_df.to_string(index=False)}")
    print(f"\n- RFECV Selected Features: {selected}")
    print(f"- Artifacts saved to: {reports_dir}/features/ and {reports_dir}/figures/feat_importance.png")


def execute_train_or_tune(config_path: str, models: Optional[list], tune: bool):
    cfg = load_config(config_path)
    reports_dir = Path(cfg["paths"]["reports_dir"])
    run_dir, logger = create_run_directory(reports_dir, cfg)

    if tune:
        logger.info("Starting Hyperparameter Optimization Search (Phase 6)...")
        tune_all_models(cfg, model_keys=models)
        
        metrics_csv = reports_dir / "metrics" / "tuned_cv.csv"
        tuned_df = pd.read_csv(metrics_csv)
        
        with open(reports_dir / "metrics" / "tuned_cv.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            best_model_name = meta["selected_best_model"]

        print(f"\n[SUCCESS] Hyperparameter Optimization Complete!")
        print("=" * 80)
        print(tuned_df[["model", "f1_mean", "accuracy_mean", "recall_mean", "precision_mean", "roc_auc_mean"]].to_string(index=False))
        print("=" * 80)
        print(f"\n[WINNING MODEL] (Selected by Decision D-10): [{best_model_name.upper()}]")
        print(f"- Selection rule: Highest CV F1 with Recall tie-breaker")
        print(f"- Best parameters saved to: {reports_dir}/metrics/best_params.json")
        print(f"- Tuning curves saved to: {reports_dir}/metrics/tuning_<model>.csv")
    else:
        logger.info("Starting Baseline Models 5-Fold Stratified Cross-Validation on Train Set...")
        results = evaluate_baseline_models(cfg, model_keys=models)

        metrics_csv = reports_dir / "metrics" / "baseline_cv.csv"
        summary_df = pd.read_csv(metrics_csv)

        dummy_f1 = results["dummy"]["f1_mean"]
        all_beat_dummy = True
        for m in [k for k in results.keys() if k != "dummy"]:
            m_f1 = results[m]["f1_mean"]
            if m_f1 <= dummy_f1:
                all_beat_dummy = False

        print(f"\n[SUCCESS] Baseline Models Evaluation Complete!")
        print("=" * 70)
        print(summary_df[["model", "f1_mean", "accuracy_mean", "recall_mean", "precision_mean", "roc_auc_mean"]].to_string(index=False))
        print("=" * 70)
        print(f"All real models beat dummy baseline: {'YES' if all_beat_dummy else 'NO'}")
        print(f"Artifacts saved to: {reports_dir}/metrics/baseline_cv.csv and baseline_cv.json")


def execute_evaluate(config_path: str):
    cfg = load_config(config_path)
    reports_dir = Path(cfg["paths"]["reports_dir"])
    run_dir, logger = create_run_directory(reports_dir, cfg)

    logger.info("Starting One-shot Evaluation on Held-out Test Set (Phase 7)...")
    results = evaluate_models(cfg)

    # Load summary CSV for clean printing
    metrics_csv = reports_dir / "metrics" / "test_metrics.csv"
    summary_df = pd.read_csv(metrics_csv)

    models_dir = Path(cfg["paths"].get("models_dir", "models"))
    meta_path = models_dir / "best_model_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    logger.info(f"Evaluation complete. Winning model: {meta['model_name'].upper()}")
    logger.info(f"Test metrics saved to {metrics_csv}")

    print("\n[SUCCESS] Final Test Set Evaluation Complete (One-Shot)!")
    print("=" * 80)
    cols_to_print = ["model", "f1", "accuracy", "recall", "precision", "specificity", "roc_auc"]
    print(summary_df[cols_to_print].to_string(index=False))
    print("=" * 80)
    print(f"\n[WINNING MODEL] (Frozen before test eval per Decision D-10): [{meta['model_name'].upper()}]")
    print(f"- Test F1 Score:    {meta['test_metrics']['f1']:.4f}")
    print(f"- Test Accuracy:    {meta['test_metrics']['accuracy']:.4%}")
    print(f"- Test Recall:      {meta['test_metrics']['recall']:.4f}")
    print(f"- Test ROC-AUC:     {meta['test_metrics']['roc_auc']:.4f}")
    print(f"- Best Model saved: {models_dir / 'best_model.joblib'}")
    print(f"- Metadata saved:   {meta_path}")
    print(f"- Visualizations:   {reports_dir}/figures/ (eval_roc_curves.png, eval_pr_curves.png, etc.)")


def execute_interpret(config_path: str):
    cfg = load_config(config_path)
    reports_dir = Path(cfg["paths"]["reports_dir"])
    run_dir, logger = create_run_directory(reports_dir, cfg)

    logger.info("Starting Phase 8 Interpretability & Feature Hierarchy Analysis...")
    res = run_interpretability_analysis(cfg)
    ranking_df = res["ranking"]

    logger.info("Interpretability analysis complete.")
    logger.info(f"Top diagnostic features: {res['top_features']}")

    print("\n[SUCCESS] Interpretability & Feature Ranking Analysis Complete (Phase 8)!")
    print("=" * 80)
    cols_to_show = ["test_rank", "feature", "test_perm_importance", "train_rf_importance"]
    present_cols = [c for c in cols_to_show if c in ranking_df.columns]
    print(ranking_df[present_cols].to_string(index=False))
    print("=" * 80)
    print(f"\n[TOP DIAGNOSTIC PARAMETERS]: {', '.join([f.upper() for f in res['top_features']])}")
    print(f"- Feature Importance Table: {reports_dir}/features/final_importance.csv")
    print(f"- Interpretability Summary: {reports_dir}/features/interpretability_summary.md")
    print(f"- Visualizations:           {reports_dir}/figures/interpret_perm_importance.png")
    print(f"                            {reports_dir}/figures/tree_interpretability.png")


def main():
    args = parse_args()
    if args.command is None:
        print("Use --help to inspect available pipeline subcommands.")
        sys.exit(0)

    if args.command == "validate":
        run_validate(args.config)
    elif args.command == "eda":
        execute_eda(args.config)
    elif args.command == "split":
        execute_split(args.config)
    elif args.command == "select-features":
        execute_select_features(args.config)
    elif args.command == "train":
        execute_train_or_tune(args.config, args.models, args.tune)
    elif args.command == "evaluate":
        execute_evaluate(args.config)
    elif args.command == "interpret":
        execute_interpret(args.config)
    else:
        print(f"Command '{args.command}' is not yet implemented or scheduled in upcoming phase.")


if __name__ == "__main__":
    main()
