import json
from pathlib import Path
import pandas as pd
import pytest
from foodadult.eda import generate_summary_stats, run_eda


@pytest.fixture
def sample_clean_df():
    return pd.DataFrame({
        "ph": [6.6, 6.7, 8.5, 9.5, 6.6],
        "temprature": [35.0, 36.0, 70.0, 34.0, 37.0],
        "taste": [1, 0, 1, 1, 0],
        "odor": [0, 1, 1, 1, 0],
        "fat": [1, 0, 1, 0, 0],
        "turbidity": [0, 1, 1, 1, 0],
        "colour": [254.0, 253.0, 246.0, 255.0, 255.0],
        "label": [0, 0, 1, 1, 1],
    })


def test_generate_summary_stats(sample_clean_df):
    stats = generate_summary_stats(sample_clean_df, target_col="label")
    assert stats["row_count"] == 5
    assert stats["column_count"] == 8
    assert "class_balance" in stats
    assert stats["class_balance"]["counts"] == {"0": 2, "1": 3}
    assert sum(stats["class_balance"]["proportions"].values()) == pytest.approx(1.0)
    assert stats["missing_total"] == 0


def test_run_eda_creates_artifacts(tmp_path, sample_clean_df):
    # Save a temporary clean CSV
    interim_csv = tmp_path / "clean.csv"
    sample_clean_df.to_csv(interim_csv, index=False)

    reports_dir = tmp_path / "reports"
    cfg = {
        "paths": {
            "interim": str(interim_csv),
            "reports_dir": str(reports_dir),
        },
        "data": {
            "target": "label",
            "features": ["ph", "temprature", "taste", "odor", "fat", "turbidity", "colour"],
        }
    }

    result = run_eda(cfg)
    assert result["summary"]["row_count"] == 5

    # Check that summary.json and data_card.md were created
    eda_dir = reports_dir / "eda"
    assert (eda_dir / "summary.json").is_file()
    assert (eda_dir / "data_card.md").is_file()

    # Check figures directory
    fig_dir = reports_dir / "figures"
    assert (fig_dir / "eda_class_balance.png").is_file()
    assert (fig_dir / "eda_correlation_matrix.png").is_file()

