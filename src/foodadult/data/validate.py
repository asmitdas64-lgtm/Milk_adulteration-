"""Data validation utilities and schema enforcement."""

from typing import Any, Dict
import pandas as pd


class SchemaError(ValueError):
    """Raised when DataFrame schema violates data specifications."""
    pass


def validate_dataframe(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate DataFrame columns, data types, physical bounds, and target labels.

    Args:
        df: DataFrame to validate.
        config: Configuration dictionary containing data specifications.

    Returns:
        Validation report dictionary.

    Raises:
        SchemaError: If critical schema checks fail.
    """
    data_cfg = config.get("data", {})
    target_col = data_cfg.get("target", "label")
    feature_cols = data_cfg.get("features", [])
    ranges = data_cfg.get("ranges", {})

    # 1. Check required columns present
    required_cols = set(feature_cols + [target_col])
    existing_cols = set(df.columns)
    missing_cols = required_cols - existing_cols
    if missing_cols:
        raise SchemaError(f"Missing required columns in DataFrame: {sorted(list(missing_cols))}")

    # 2. Check target column values
    target_vals = set(df[target_col].dropna().unique())
    if not target_vals.issubset({0, 1}):
        raise SchemaError(
            f"Target label values must be a subset of {{0, 1}}, found: {target_vals}"
        )

    # 3. Check feature types are numeric
    non_numeric = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    if non_numeric:
        raise SchemaError(f"Columns must have numeric dtypes, but found non-numeric: {non_numeric}")

    # 4. Check out of physical range values
    out_of_range = {}
    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                out_of_range[col] = {
                    "count": invalid_count,
                    "min_found": float(df[col].min()),
                    "max_found": float(df[col].max()),
                    "allowed": [min_val, max_val],
                }

    # 5. Check duplicate rows
    duplicate_count = int(df.duplicated().sum())

    # 6. Check missingness
    missing_counts = df[list(required_cols)].isna().sum().to_dict()

    return {
        "is_valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_count": duplicate_count,
        "missing_counts": missing_counts,
        "out_of_range": out_of_range,
    }
