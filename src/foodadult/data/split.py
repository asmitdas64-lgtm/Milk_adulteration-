"""Dataset partitioning and stratified splitting utilities."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def perform_stratified_split(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Partition interim data into stratified train and test sets.

    Args:
        config: Configuration dictionary with paths, data target, and split settings.

    Returns:
        Tuple containing (train_df, test_df, split_info).
    """
    interim_path = Path(config["paths"]["interim"])
    processed_dir = Path(config["paths"]["processed_dir"])
    target_col = config["data"].get("target", "label")
    test_size = float(config.get("split", {}).get("test_size", 0.2))
    random_state = int(config.get("random_state", 42))

    df = pd.read_csv(interim_path)

    # Perform stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(df, df[target_col]))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Save artifacts
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    split_info_path = processed_dir / "split.json"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    split_info = {
        "random_state": random_state,
        "test_size": test_size,
        "total_samples": len(df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_class_balance": {str(k): int(v) for k, v in train_df[target_col].value_counts().items()},
        "test_class_balance": {str(k): int(v) for k, v in test_df[target_col].value_counts().items()},
        "train_indices": [int(i) for i in train_idx],
        "test_indices": [int(i) for i in test_idx],
    }

    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    return train_df, test_df, split_info
