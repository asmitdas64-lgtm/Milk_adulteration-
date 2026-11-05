import json
from pathlib import Path
from typing import Any, Tuple
import joblib


def save_model(model: Any, filepath: str | Path, metadata: dict | None = None) -> Path:
    """Save a fitted model to disk via joblib, and optionally save metadata JSON alongside it."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)

    if metadata is not None:
        meta_path = filepath.parent / f"{filepath.stem}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return filepath


def load_model(filepath: str | Path) -> Tuple[Any, dict]:
    """Load a trained model and its associated metadata JSON."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    model = joblib.load(filepath)
    meta_path = filepath.parent / f"{filepath.stem}_meta.json"
    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, metadata

