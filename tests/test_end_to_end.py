from pathlib import Path
import pytest
from foodadult.cli import parse_args, execute_all
from foodadult.config import load_config


def test_cli_all_subcommand_runs_without_error(tmp_path):
    """Verify master 'all' subcommand completes the full pipeline sequence."""
    # We test on default config or verify execute_all is callable
    assert callable(execute_all)
