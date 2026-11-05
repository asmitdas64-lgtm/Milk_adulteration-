"""Logging configuration and run management."""

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yaml


def setup_logger(
    name: str = "foodadult",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up and return a console and optional file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def create_run_directory(base_reports_dir: Path, config: Dict[str, Any]) -> Tuple[Path, logging.Logger]:
    """Create a timestamped run directory and snapshot the configuration."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_reports_dir / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot configuration
    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    logger = setup_logger(name=f"run_{timestamp}", log_file=run_dir / "run.log")
    return run_dir, logger

