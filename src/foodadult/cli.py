"""Command-line interface for the Food Adulteration ML Pipeline."""

import argparse
import sys
from pathlib import Path
import pandas as pd

from foodadult.config import load_config
from foodadult.data.load import load_raw, standardize_columns, make_binary_label
from foodadult.data.split import perform_stratified_split
from foodadult.data.validate import validate_dataframe
from foodadult.eda import run_eda
from foodadult.logging_utils import create_run_directory, setup_logger


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
    cmd_train = subparsers.add_parser("train", help="Train baseline and tuned ML models")
    cmd_train.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    cmd_train.add_argument("--models", nargs="+", default=None, help="Specific models to train")

    # Command: evaluate
    cmd_eval = subparsers.add_parser("evaluate", help="Evaluate models on test set once")
    cmd_eval.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")

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
    else:
        print(f"Command '{args.command}' is not yet implemented or scheduled in upcoming phase.")


if __name__ == "__main__":
    main()
