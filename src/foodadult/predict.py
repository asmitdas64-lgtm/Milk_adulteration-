import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

from foodadult.models.persist import load_model

DEFAULT_FEATURES = ["ph", "temprature", "taste", "odor", "fat", "turbidity", "colour"]

PHYSICAL_RANGES = {
    "ph": (0.0, 14.0),
    "temprature": (0.0, 100.0),
    "taste": (0, 1),
    "odor": (0, 1),
    "fat": (0, 1),
    "turbidity": (0, 1),
    "colour": (200.0, 260.0),
}


def validate_input_data(df: pd.DataFrame, expected_features: Optional[List[str]] = None) -> pd.DataFrame:
    """Validate input DataFrame schema, dtypes, and physically plausible ranges."""
    if expected_features is None:
        expected_features = DEFAULT_FEATURES

    # Standardize column headers: strip whitespace and lowercase
    clean_cols = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=clean_cols)

    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}. Expected features: {expected_features}")

    # Check numeric conversion and ranges
    for col in expected_features:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except Exception as e:
            raise ValueError(f"Feature '{col}' contains non-numeric values: {e}")

        if col in PHYSICAL_RANGES:
            min_val, max_val = PHYSICAL_RANGES[col]
            out_of_bounds = (df[col] < min_val) | (df[col] > max_val)
            if out_of_bounds.any():
                invalid_vals = df.loc[out_of_bounds, col].tolist()
                raise ValueError(
                    f"Feature '{col}' contains values {invalid_vals} out of physically plausible range [{min_val}, {max_val}]."
                )

    return df


def predict_dataframe(model, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Generate predictions and adulteration probabilities for a validated DataFrame."""
    X = df[feature_cols]
    preds = model.predict(X).astype(int)

    probs = None
    if hasattr(model, "predict_proba"):
        try:
            prob_matrix = model.predict_proba(X)
            # Probability of adulterated (class 1)
            probs = prob_matrix[:, 1]
            confidences = np.max(prob_matrix, axis=1)
        except Exception:
            probs = None
            confidences = np.ones(len(preds), dtype=float)
    else:
        confidences = np.ones(len(preds), dtype=float)

    result_df = df.copy()
    result_df["prediction"] = preds
    result_df["predicted_label"] = np.where(preds == 1, "Adulterated", "Pure")

    if probs is not None:
        result_df["probability_adulterated"] = np.round(probs, 4)
        result_df["confidence"] = np.round(confidences, 4)
    else:
        result_df["probability_adulterated"] = np.where(preds == 1, 1.0, 0.0)
        result_df["confidence"] = 1.0

    return result_df


def predict_csv(
    model_path: str | Path,
    in_csv: str | Path,
    out_csv: str | Path,
) -> pd.DataFrame:
    """Batch prediction from input CSV to output CSV using a saved model pipeline."""
    model_path = Path(model_path)
    in_csv = Path(in_csv)
    out_csv = Path(out_csv)

    if not in_csv.exists():
        raise FileNotFoundError(f"Input file not found: {in_csv}")

    model, meta = load_model(model_path)
    feature_cols = meta.get("features", DEFAULT_FEATURES)

    raw_df = pd.read_csv(in_csv)
    validated_df = validate_input_data(raw_df, expected_features=feature_cols)

    predictions_df = predict_dataframe(model, validated_df, feature_cols=feature_cols)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(out_csv, index=False)
    return predictions_df
