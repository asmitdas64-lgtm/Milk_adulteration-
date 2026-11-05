import numpy as np
import pandas as pd
import pytest
from foodadult.features.preprocess import build_preprocessor


def test_no_data_leakage_between_train_and_test():
    # Training set
    X_train = pd.DataFrame({
        "ph": [6.5, 6.6, 6.7, 6.8, 6.9],
        "temprature": [35.0, 36.0, 37.0, 38.0, 39.0],
        "colour": [250.0, 251.0, 252.0, 253.0, 254.0],
    })

    # Test set contains a massive out-of-distribution anomaly (e.g. pH = 100.0, temp = 500.0)
    X_test = pd.DataFrame({
        "ph": [100.0],
        "temprature": [500.0],
        "colour": [999.0],
    })

    cfg = {
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

    preprocessor = build_preprocessor(cfg)
    # Fit ONLY on train
    preprocessor.fit(X_train)

    # Extract the scaler step inside the pipeline
    num_pipeline = preprocessor.named_transformers_["num"]
    scaler = num_pipeline.named_steps["scaler"]
    imputer = num_pipeline.named_steps["imputer"]

    # Verify that imputer learned statistics match train statistics alone
    expected_medians = X_train.median().values
    assert np.allclose(imputer.statistics_, expected_medians)

    # Verify that scaler mean and variance match train data alone
    # (after clipping, which doesn't alter linear X_train)
    assert np.allclose(scaler.mean_, X_train.mean().values)

    # Now transform test: verify test extreme values do not change scaler.mean_
    preprocessor.transform(X_test)
    assert np.allclose(scaler.mean_, X_train.mean().values)
