import json
from pathlib import Path
import pandas as pd
import pytest
from foodadult.data.split import perform_stratified_split


@pytest.fixture
def dummy_df():
    # 100 samples: 75 positive (1), 25 negative (0)
    data = {
        "feat1": list(range(100)),
        "label": [1] * 75 + [0] * 25
    }
    return pd.DataFrame(data)


def test_stratified_split_proportions_and_sizes(tmp_path, dummy_df):
    clean_csv = tmp_path / "clean.csv"
    dummy_df.to_csv(clean_csv, index=False)
    out_dir = tmp_path / "processed"

    cfg = {
        "paths": {
            "interim": str(clean_csv),
            "processed_dir": str(out_dir),
        },
        "data": {"target": "label"},
        "split": {"test_size": 0.2, "stratify": True},
        "random_state": 42,
    }

    train_df, test_df, split_info = perform_stratified_split(cfg)

    # 1. Sizes check: 80 train, 20 test
    assert len(train_df) == 80
    assert len(test_df) == 20

    # 2. Stratification check: ratio of label=1 should be exactly 75% in both
    assert train_df["label"].mean() == pytest.approx(0.75, abs=0.02)
    assert test_df["label"].mean() == pytest.approx(0.75, abs=0.02)

    # 3. Disjoint check: train and test indices have no overlap
    train_idx = set(split_info["train_indices"])
    test_idx = set(split_info["test_indices"])
    assert len(train_idx.intersection(test_idx)) == 0

    # 4. Files written check
    assert (out_dir / "train.csv").is_file()
    assert (out_dir / "test.csv").is_file()
    assert (out_dir / "split.json").is_file()
