"""Data loading and standardisation utilities."""

from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd


def load_raw(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load raw dataset from CSV without modifying source file.

    Args:
        file_path: Path to the raw CSV file.

    Returns:
        DataFrame containing the raw data.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    # Read fresh copy
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame, rename_map: Dict[str, str]) -> pd.DataFrame:
    """Standardise column names using a configuration mapping.

    Args:
        df: Input DataFrame.
        rename_map: Dictionary mapping original column names to standardized snake_case.

    Returns:
        DataFrame with standardized column names.
    """
    df_clean = df.copy()
    # Strip any accidental leading/trailing whitespace in column names
    df_clean.columns = [c.strip() if isinstance(c, str) else c for c in df_clean.columns]
    
    # Strip whitespace in mapping keys as well
    cleaned_map = {k.strip(): v for k, v in rename_map.items()}
    df_clean = df_clean.rename(columns=cleaned_map)
    return df_clean


def make_binary_label(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
    """Map categorical or multi-class target labels to binary integers {0, 1}.

    Args:
        series: Series of raw target values.
        mapping: Dictionary mapping raw label string to binary integer (0 or 1).

    Returns:
        Series of binary integers (0 or 1).

    Raises:
        ValueError: If any value in series is not defined in the mapping.
    """
    unmapped = set(series.dropna().unique()) - set(mapping.keys())
    if unmapped:
        raise ValueError(f"Unmapped label values found in target: {unmapped}")

    mapped = series.map(mapping)
    return mapped.astype(int)
