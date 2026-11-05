import subprocess
import sys
from pathlib import Path
from foodadult.config import load_config


def test_package_imports():
    import foodadult
    assert hasattr(foodadult, "__file__")


def test_load_config():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = load_config(cfg_path)
    assert isinstance(cfg, dict)
    assert cfg.get("random_state") == 42
    assert "data" in cfg
    assert "paths" in cfg


def test_cli_help():
    venv_python = sys.executable
    result = subprocess.run(
        [venv_python, "-m", "foodadult.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Food Adulteration ML Pipeline CLI" in result.stdout

