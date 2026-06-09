from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

import daily_position_update as dpu


REPO_DIR = Path(dpu.__file__).parent.resolve()


def _args(**overrides):
    values = {
        "only": None,
        "skip_ibkr": False,
        "skip_degiro": False,
        "from_date": None,
        "to_date": None,
        "flex_wait_seconds": 5,
        "flex_max_attempts": 12,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_selected_jobs_defaults_to_both_brokers():
    jobs = dpu._selected_jobs(_args())
    assert [job.name for job in jobs] == ["ibkr", "degiro"]


def test_selected_jobs_supports_only_and_skip_flags():
    assert [job.name for job in dpu._selected_jobs(_args(only="ibkr"))] == ["ibkr"]
    assert [job.name for job in dpu._selected_jobs(_args(skip_degiro=True))] == ["ibkr"]
    assert [job.name for job in dpu._selected_jobs(_args(skip_ibkr=True))] == ["degiro"]


def test_validate_args_rejects_conflicting_selection():
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        dpu._validate_args(parser, _args(only="ibkr", skip_ibkr=True))
    with pytest.raises(SystemExit):
        dpu._validate_args(parser, _args(skip_ibkr=True, skip_degiro=True))


def test_validate_args_rejects_invalid_flex_options():
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        dpu._validate_args(parser, _args(flex_wait_seconds=0))
    with pytest.raises(SystemExit):
        dpu._validate_args(parser, _args(flex_max_attempts=0))


def test_daily_position_update_dry_run_smoke():
    proc = subprocess.run(
        [sys.executable, str(REPO_DIR / "daily_position_update.py"), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    text = proc.stdout + proc.stderr
    assert "Daily position update starting" in text
    assert "START   ibkr" in text
    assert "START   degiro" in text
    assert "All broker snapshots completed successfully" in text
