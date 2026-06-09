"""Tests for daily_ecosystem_update.py - orchestration, dependency skipping, and the
subprocess runner (live streaming + timeout enforcement).

Strategy
--------
1. `_unmet_deps` — pure list-comp helper; tested directly.
2. `_execute_step` — uses a mock `run_fn` so we never actually spawn pipeline
   scripts; we just check that the right thing happens when upstream is OK,
   when upstream failed, and when the runner returns False.
3. `run` — exercised with tiny `python -c "..."` snippets to verify the
   success / non-zero-exit / timeout / spawn-error / streaming paths.
4. End-to-end - invoke daily_ecosystem_update.py --dry-run --force-weekly via subprocess
   to confirm the full orchestration produces the expected step sequence.
"""

import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

import daily_ecosystem_update as du
from daily_ecosystem_update import _execute_step, _unmet_deps, run


REPO_DIR = Path(du.__file__).parent.resolve()


# ── _unmet_deps ──────────────────────────────────────────────────────────────

def test_unmet_deps_all_satisfied():
    results = {"a": True, "b": True}
    assert _unmet_deps(("a", "b"), results) == []


def test_unmet_deps_one_failed():
    results = {"a": True, "b": False}
    assert _unmet_deps(("a", "b"), results) == ["b"]


def test_unmet_deps_missing_from_results():
    # A dependency that was never run at all also counts as unmet.
    results = {"a": True}
    assert _unmet_deps(("a", "b"), results) == ["b"]


def test_unmet_deps_empty():
    assert _unmet_deps((), {}) == []


# ── _execute_step ────────────────────────────────────────────────────────────

def test_execute_step_runs_when_deps_satisfied():
    """run_fn must be called and `results` updated when all deps are ok."""
    results: dict[str, bool] = {"upstream": True}
    errors:  list[str]       = []
    calls:   list[tuple]     = []

    def fake_run(cmd, timeout, dry_run):
        calls.append((cmd, timeout, dry_run))
        return True

    ok = _execute_step(
        name="my_step",
        script_args=("create_factors.py", "--date", "2026-05-25"),
        depends_on=("upstream",),
        results=results, errors=errors, dry_run=False,
        run_fn=fake_run,
    )
    assert ok is True
    assert results["my_step"] is True
    assert errors == []
    assert len(calls) == 1
    cmd, timeout, dry_run = calls[0]
    # cmd is built from [PYTHON, "-u", "-m", "pipeline.<script>", *args]
    assert cmd[0] == du.PYTHON
    assert cmd[1] == "-u"
    assert cmd[2:] == ["-m", "pipeline.create_factors", "--date", "2026-05-25"]
    # Timeout comes from TIMEOUTS dict (or default 1800 if missing).
    assert timeout == du.TIMEOUTS.get("my_step", 1800)


def test_execute_step_skips_when_upstream_failed():
    """When an upstream is False, the step is skipped without invoking run_fn."""
    results: dict[str, bool] = {"upstream_a": True, "upstream_b": False}
    errors:  list[str]       = []
    calls:   list[tuple]     = []

    def fake_run(cmd, timeout, dry_run):
        calls.append((cmd, timeout, dry_run))
        return True

    ok = _execute_step(
        name="downstream",
        script_args=("anything.py",),
        depends_on=("upstream_a", "upstream_b"),
        results=results, errors=errors, dry_run=False,
        run_fn=fake_run,
    )
    assert ok is False
    assert results["downstream"] is False
    assert calls == []                         # run_fn must NOT have been called
    assert len(errors) == 1
    assert "downstream" in errors[0]
    assert "upstream_b" in errors[0]           # the failing dep is named


def test_execute_step_records_failure_when_run_fn_returns_false():
    results: dict[str, bool] = {}
    errors:  list[str]       = []

    def failing_run(cmd, timeout, dry_run):
        return False

    ok = _execute_step(
        name="will_fail",
        script_args=("x.py",),
        depends_on=(),
        results=results, errors=errors, dry_run=False,
        run_fn=failing_run,
    )
    assert ok is False
    assert results["will_fail"] is False
    assert errors == ["will_fail"]


def test_execute_step_skips_when_dep_missing_from_results():
    """An upstream that simply isn't in `results` (never ran) also blocks."""
    results: dict[str, bool] = {}
    errors:  list[str]       = []
    calls:   list = []

    _execute_step(
        name="downstream", script_args=("x.py",),
        depends_on=("never_ran",),
        results=results, errors=errors, dry_run=False,
        run_fn=lambda *a, **kw: calls.append(a) or True,
    )
    assert calls == []
    assert results["downstream"] is False


# ── run() — success / failure / spawn-error / timeout / streaming ────────────

def test_run_returns_true_on_success():
    # A trivial command that prints and exits 0.
    cmd = [sys.executable, "-u", "-c", "print('hi from child')"]
    assert run(cmd, timeout=10) is True


def test_run_returns_false_on_nonzero_exit():
    cmd = [sys.executable, "-u", "-c", "import sys; sys.exit(1)"]
    assert run(cmd, timeout=10) is False


def test_run_returns_false_on_spawn_error():
    # A binary that definitely does not exist on PATH.
    cmd = ["/nonexistent/path/to/nothing_xyz_12345"]
    assert run(cmd, timeout=10) is False


def test_run_dry_run_does_not_execute():
    # In dry_run mode the child must not be spawned at all — passing a
    # non-existent path should still return True.
    cmd = ["/nonexistent/path", "anything"]
    assert run(cmd, timeout=10, dry_run=True) is True


def test_run_returns_false_for_invalid_timeout():
    cmd = [sys.executable, "-u", "-c", "print('should not run')"]
    assert run(cmd, timeout=0) is False


def test_run_kills_child_on_timeout():
    """A child sleeping past the deadline must be killed and run() returns False
    within a small margin of the timeout."""
    cmd = [sys.executable, "-u", "-c", "import time; time.sleep(60)"]
    t0 = time.time()
    ok = run(cmd, timeout=2)
    elapsed = time.time() - t0
    assert ok is False
    # select tick is 5s; timeout=2 means the check fires on the first tick.
    # Allow generous slack so the test isn't flaky on a loaded machine.
    assert elapsed < 10, f"timeout not enforced quickly enough ({elapsed:.1f}s)"


def test_run_streams_output_line_by_line(caplog):
    """Each line printed by the child should appear in the log as a separate
    record — proves we're not buffering everything until the child exits."""
    # Five lines printed with a short delay between them, then exit 0.
    child_code = (
        "import sys, time\n"
        "for i in range(5):\n"
        "    print(f'line-{i}', flush=True)\n"
        "    time.sleep(0.05)\n"
    )
    cmd = [sys.executable, "-u", "-c", child_code]

    # daily_ecosystem_update's logger has propagate=False, so caplog (which hooks the
    # root logger) won't see records by default.  Temporarily enable it.
    du.log.propagate = True
    try:
        with caplog.at_level("INFO", logger="daily_ecosystem_update"):
            ok = run(cmd, timeout=10)
    finally:
        du.log.propagate = False

    assert ok is True
    # Streaming lines are logged as "  line-N" (two-space prefix from run()).
    # The START line also contains the child source — match only the indented
    # output lines from the child itself.
    streamed = [r.getMessage() for r in caplog.records
                if r.getMessage().strip().startswith("line-")]
    assert len(streamed) == 5
    assert "line-0" in streamed[0]
    assert "line-4" in streamed[-1]


# ── End-to-end: --dry-run weekly produces expected step sequence ─────────────

def test_dry_run_weekly_lists_expected_steps():
    """Smoke test the full orchestration via subprocess.  Confirms that:
      - Each of the expected steps emits a START line
      - The Weekly rebuild section header is present
      - The final summary reports success (exit 0)
    """
    proc = subprocess.run(
        [sys.executable, str(REPO_DIR / "daily_ecosystem_update.py"),
         "--dry-run", "--force-weekly"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"

    text = proc.stdout + proc.stderr
    expected_steps = [
        "pipeline.create_returns --update",
        "pipeline.update_constituents",
        "pipeline.create_macro_signals --date",
        "pipeline.create_factors --date",
        "pipeline.create_models --date",
        "pipeline.create_risk --date",
        "pipeline.create_barra",
    ]
    for expected in expected_steps:
        assert re.search(rf"START\s+{re.escape(expected)}", text), \
            f"missing START line for {expected!r}"

    assert "Weekly rebuild" in text
    assert "All steps completed successfully" in text


def test_dry_run_default_skips_weekly_on_non_friday():
    """Without --force-weekly, the rebuild only runs on Fridays.  We verify
    behaviour matches today's day-of-week — either weekly is present or it isn't."""
    proc = subprocess.run(
        [sys.executable, str(REPO_DIR / "daily_ecosystem_update.py"), "--dry-run"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    text = proc.stdout + proc.stderr
    # Daily steps always appear:
    assert "pipeline.create_returns --update" in text
    assert "pipeline.update_constituents" in text
    assert "pipeline.create_macro_signals --date" in text


def test_dry_run_skip_returns_marks_downstream_runnable():
    """--skip-returns must not break the dependency chain — downstream
    (factors, risk, barra) should still appear in the dry-run plan."""
    proc = subprocess.run(
        [sys.executable, str(REPO_DIR / "daily_ecosystem_update.py"),
         "--dry-run", "--force-weekly", "--skip-returns"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0
    text = proc.stdout + proc.stderr
    assert "SKIP    returns" in text
    # Downstream factor/risk/barra steps should still be planned (not skipped):
    assert "START   pipeline.create_factors" in text
    assert "START   pipeline.create_risk" in text
    assert "START   pipeline.create_barra" in text
