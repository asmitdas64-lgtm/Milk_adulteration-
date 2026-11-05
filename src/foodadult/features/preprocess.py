"""Preprocessing pipelines and ColumnTransformer builders."""

from typing import Any, Dict
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from foodadult.features.transformers import IQRClipper


def build_preprocessor(config: Dict[str, Any]) -> ColumnTransformer:
    """Build Scikit-Learn ColumnTransformer based on configuration.

    Chains median imputation, IQR outlier clipping, and standard scaling.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured ColumnTransformer instance.
    """
    data_cfg = config.get("data", {})
    features = data_cfg.get("features", [])
    prep_cfg = config.get("preprocess", {})

    imputer_strategy = prep_cfg.get("imputer", "median")
    outlier_method = prep_cfg.get("outliers", "clip")
    iqr_factor = float(prep_cfg.get("iqr_factor", 1.5))

    steps = [
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
    ]

    if outlier_method == "clip":
        steps.append(("iqr_clipper", IQRClipper(factor=iqr_factor)))

    steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, features),
        ],
        remainder="drop",
    )
    return preprocessor
