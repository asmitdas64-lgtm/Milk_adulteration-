import pandas as pd
import pytest
from foodadult.data.validate import validate_dataframe, SchemaError


@pytest.fixture
def valid_config():
    return {
        "data": {
            "target": "label",
            "features": ["ph", "temprature", "taste", "odor", "fat", "turbidity", "colour"],
            "ranges": {
                "ph": [0.0, 14.0],
                "temprature": [0.0, 100.0],
                "taste": [0, 1],
                "odor": [0, 1],
                "fat": [0, 1],
                "turbidity": [0, 1],
                "colour": [0.0, 300.0],
            }
        }
    }


def test_validate_dataframe_passes_on_valid_data(valid_config):
    df = pd.DataFrame({
        "ph": [6.6, 6.7],
        "temprature": [35.0, 40.0],
        "taste": [1, 0],
        "odor": [0, 1],
        "fat": [1, 0],
        "turbidity": [0, 1],
        "colour": [254.0, 250.0],
        "label": [0, 1]
    })
    report = validate_dataframe(df, valid_config)
    assert report["is_valid"] is True
    assert report["row_count"] == 2
    assert report["duplicate_count"] == 0


def test_validate_dataframe_fails_on_missing_column(valid_config):
    df = pd.DataFrame({
        "ph": [6.6],
        "temprature": [35.0],
        "label": [0]
    })
    with pytest.raises(SchemaError, match="Missing required columns"):
        validate_dataframe(df, valid_config)


def test_validate_dataframe_fails_on_invalid_label_values(valid_config):
    df = pd.DataFrame({
        "ph": [6.6],
        "temprature": [35.0],
        "taste": [1],
        "odor": [0],
        "fat": [1],
        "turbidity": [0],
        "colour": [254.0],
        "label": [99]  # Invalid binary label
    })
    with pytest.raises(SchemaError, match="Target label values must be a subset of {0, 1}"):
        validate_dataframe(df, valid_config)


def test_validate_dataframe_flags_out_of_range_values(valid_config):
    df = pd.DataFrame({
        "ph": [99.0],  # Out of range!
        "temprature": [35.0],
        "taste": [1],
        "odor": [0],
        "fat": [1],
        "turbidity": [0],
        "colour": [254.0],
        "label": [1]
    })
    report = validate_dataframe(df, valid_config)
    assert report["is_valid"] is True
    assert "ph" in report["out_of_range"]
    assert report["out_of_range"]["ph"]["count"] == 1

