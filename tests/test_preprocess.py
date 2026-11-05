import numpy as np
import pandas as pd
import pytest
from foodadult.features.preprocess import build_preprocessor


@pytest.fixture
def mock_cfg():
    return {
        "data": {
            "features": ["ph", "temprature", "colour"],
        },
        "preprocess": {
            "imputer": "median",
            "outliers": "clip",
            "iqr_factor": 1.5,
            "scaler": "standard",
        }
    }


def test_build_preprocessor_runs_and_scales(mock_cfg):
    X = pd.DataFrame({
        "ph": [6.5, 6.7, np.nan, 7.0, 6.6],
        "temprature": [35.0, 40.0, 38.0, 42.0, 36.0],
        "colour": [250.0, 255.0, 252.0, 254.0, 248.0],
    })

    pipe = build_preprocessor(mock_cfg)
    transformed = pipe.fit_transform(X)

    # Output should have same number of rows and columns, with no NaNs
    assert transformed.shape == (5, 3)
    assert not np.isnan(transformed).any()

    # Transformed data should have mean approx 0 and std approx 1
    assert np.mean(transformed, axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-5)
    assert np.std(transformed, axis=0) == pytest.approx([1.0, 1.0, 1.0], abs=1e-5)

