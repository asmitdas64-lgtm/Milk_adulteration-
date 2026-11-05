"""Command-line interface for the Food Adulteration ML Pipeline."""

import argparse
import sys
from pathlib import Path
from foodadult.config import load_config
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


def main():
    args = parse_args()
    if args.command is None:
        print("Use --help to inspect available pipeline subcommands.")
        sys.exit(0)

    print(f"Executing command: {args.command}")


if __name__ == "__main__":
    main()
