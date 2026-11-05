from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from foodadult.predict import predict_csv, validate_input_data
from foodadult.models.persist import save_model


@pytest.fixture
def predict_fixture(tmp_path):
    """Create a minimal model and input CSV for prediction testing."""
    # Features: ph, temprature, taste, odor, fat, turbidity, colour
    features = ["ph", "temprature", "taste", "odor", "fat", "turbidity", "colour"]
    X_train = pd.DataFrame({
        "ph": [6.6, 6.7, 3.0, 8.5],
        "temprature": [35.0, 37.0, 45.0, 50.0],
        "taste": [1, 1, 0, 0],
        "odor": [0, 0, 1, 1],
        "fat": [1, 1, 0, 0],
        "turbidity": [0, 0, 1, 1],
        "colour": [254.0, 255.0, 240.0, 245.0],
    })
    y_train = np.array([0, 0, 1, 1])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=3, random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    model_path = tmp_path / "test_model.joblib"
    meta = {
        "model_name": "dt",
        "features": features,
        "target": "label",
    }
    save_model(pipe, model_path, metadata=meta)

    # Input sample 1 (pure like) and sample 2 (adulterated like)
    input_df = pd.DataFrame({
        "ph": [6.6, 3.5],
        "temprature": [36.0, 48.0],
        "taste": [1, 0],
        "odor": [0, 1],
        "fat": [1, 0],
        "turbidity": [0, 1],
        "colour": [255.0, 242.0],
    })
    in_csv = tmp_path / "input.csv"
    input_df.to_csv(in_csv, index=False)

    out_csv = tmp_path / "output.csv"
    return model_path, in_csv, out_csv, features


def test_predict_csv_valid_output(predict_fixture):
    """Test batch prediction produces valid predictions and probabilities."""
    model_path, in_csv, out_csv, _ = predict_fixture
    result_df = predict_csv(model_path, in_csv, out_csv)

    assert out_csv.exists()
    assert len(result_df) == 2
    assert "prediction" in result_df.columns
    assert "predicted_label" in result_df.columns
    assert "probability_adulterated" in result_df.columns
    assert set(result_df["prediction"].unique()).issubset({0, 1})
    assert all(0.0 <= p <= 1.0 for p in result_df["probability_adulterated"])

    # Sample 0 should be Pure (0), Sample 1 should be Adulterated (1)
    assert result_df["prediction"].iloc[0] == 0
    assert result_df["predicted_label"].iloc[0] == "Pure"
    assert result_df["prediction"].iloc[1] == 1
    assert result_df["predicted_label"].iloc[1] == "Adulterated"


def test_predict_csv_missing_columns_raises_error(predict_fixture):
    """Missing required feature should raise ValueError."""
    model_path, _, out_csv, _ = predict_fixture
    bad_df = pd.DataFrame({
        "ph": [6.6],
        "temprature": [35.0],
        # Missing other required features
    })
    bad_csv = out_csv.parent / "bad.csv"
    bad_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="Missing required feature columns"):
        predict_csv(model_path, bad_csv, out_csv)


def test_validate_input_data_out_of_range(predict_fixture):
    """Physically impossible pH or temperature should raise ValueError."""
    _, _, _, features = predict_fixture
    bad_ph_df = pd.DataFrame({
        "ph": [25.0],  # pH > 14 is impossible
        "temprature": [35.0],
        "taste": [1],
        "odor": [0],
        "fat": [1],
        "turbidity": [0],
        "colour": [255.0],
    })

    with pytest.raises(ValueError, match="out of physically plausible range"):
        validate_input_data(bad_ph_df, features)

