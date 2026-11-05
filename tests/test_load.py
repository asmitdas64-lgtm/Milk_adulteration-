import pandas as pd
import pytest
from foodadult.data.load import load_raw, standardize_columns, make_binary_label


def test_standardize_columns():
    df = pd.DataFrame({
        "pH": [6.6],
        "Temprature": [35],
        "Taste": [1],
        "Odor": [0],
        "Fat ": [1],
        "Turbidity": [0],
        "Colour": [254],
        "Grade": ["high"]
    })
    rename_map = {
        "pH": "ph",
        "Temprature": "temprature",
        "Taste": "taste",
        "Odor": "odor",
        "Fat ": "fat",
        "Turbidity": "turbidity",
        "Colour": "colour",
        "Grade": "grade",
    }
    cleaned = standardize_columns(df, rename_map)
    expected_cols = ["ph", "temprature", "taste", "odor", "fat", "turbidity", "colour", "grade"]
    assert list(cleaned.columns) == expected_cols


def test_make_binary_label():
    series = pd.Series(["high", "medium", "low", "high"])
    mapping = {"high": 0, "medium": 1, "low": 1}
    labels = make_binary_label(series, mapping)
    assert list(labels) == [0, 1, 1, 0]


def test_make_binary_label_unknown_raises():
    series = pd.Series(["high", "unknown_grade"])
    mapping = {"high": 0, "low": 1}
    with pytest.raises(ValueError, match="Unmapped label values found"):
        make_binary_label(series, mapping)

