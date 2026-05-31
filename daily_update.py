#!/usr/bin/env python3
"""
daily_update.py — Automated pipeline update.

Schedule:
  Daily   — update prices (create_returns.py --update)
  Daily   — FINRA short volume ratio (create_svr.py)
  Daily   — EDGAR filing index: process companies that actually filed (10-K + 10-Q)
  Weekly  — on Fridays: rebuild factors, models, risk, Barra, portfolio
             using today's date as the snapshot so everything is aligned

Behaviour:
  - Child output streams live (line-by-line) via Popen + select.
  - Each step has its own timeout; on timeout the child is killed.
  - Steps declare upstream dependencies — if an upstream fails (or times out),
    downstream steps are skipped rather than running on stale data.
  - --skip-X marks a step as "ok by user choice" so downstream still runs
    against the existing DB.

Usage:
    python daily_update.py                   # normal run (Friday triggers full rebuild)
    python daily_update.py --force-weekly    # force full weekly rebuild today
    python daily_update.py --dry-run         # print steps without running
"""

import argparse
import select
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

from utils import get_logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Use whichever Python is running this script — no hardcoded interpreter path.
PYTHON   = sys.executable
REPO_DIR = Path(__file__).parent.resolve()

# Per-step soft timeouts (seconds).  A step is killed if it neither prints
# output nor exits within this window.  Values are "definitely stuck" rather
# than "expected runtime."
TIMEOUTS: dict[str, int] = {
    "returns":   3600,   # 1h — yfinance bulk pull
    "svr":       1800,   # 30m — FINRA short volume incremental
    "filings":   3600,   # 1h — EDGAR index + per-company fetches
    "factors":   2700,   # 45m
    "models":     900,   # 15m
    "risk":      2700,   # 45m
    "barra":     2700,   # 45m
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = get_logger("daily_update")


# ---------------------------------------------------------------------------
# Runner — streams child output and enforces a hard timeout.
# ---------------------------------------------------------------------------

def run(cmd: list[str], timeout: int, dry_run: bool = False) -> bool:
    """Run a subprocess, streaming its stdout/stderr line-by-line to the log.

    Returns True on clean exit (rc=0), False on non-zero exit, timeout, or
    spawn error.  Stderr is merged into stdout so the log shows one
    chronological stream.
    """
    label = " ".join(cmd[2:])    # skip [PYTHON, "-u"] prefix for readability
    log.info("START   %s   (timeout=%ds)", label, timeout)
    if dry_run:
        log.info("  [dry-run] %s", " ".join(cmd))
        return True

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, cwd=REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,                 # line-buffered on the parent's read side
        )
    except Exception as exc:
        log.error("FAILED  %s — could not spawn: %s", label, exc)
        return False

    deadline = t0 + timeout
    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                log.error("TIMEOUT %s after %.0fs (limit %ds)",
                          label, time.time() - t0, timeout)
                return False

            # Wake at most every 5 s to recheck the deadline even if the child
            # is silent — prevents a stuck step from blocking the whole job.
            ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 5))
            if ready:
                line = proc.stdout.readline()
                if not line:
                    break                # EOF — child closed its stdout
                log.info("  %s", line.rstrip())
            elif proc.poll() is not None:
                break                    # child exited and no buffered output
    except Exception as exc:
        log.error("FAILED  %s — output read error: %s", label, exc)
        try:
            proc.kill()
        except Exception:
            pass
        return False

    # Drain any final bytes left in the pipe after the child closed it.
    rest = proc.stdout.read()
    if rest:
        for ln in rest.rstrip().splitlines():
            log.info("  %s", ln)

    rc      = proc.wait()
    elapsed = time.time() - t0
    if rc != 0:
        log.error("FAILED  %s (exit %d, %.0fs)", label, rc, elapsed)
        return False
    log.info("OK      %s (%.0fs)", label, elapsed)
    return True


# ---------------------------------------------------------------------------
# Pipeline orchestration — dependency-aware fail-fast.
# ---------------------------------------------------------------------------

def _unmet_deps(depends_on: tuple[str, ...], results: dict[str, bool]) -> list[str]:
    """Names of upstream steps that didn't succeed (or aren't in results yet)."""
    return [d for d in depends_on if not results.get(d, False)]


def _execute_step(
    name: str,
    script_args: tuple[str, ...],
    depends_on: tuple[str, ...],
    results: dict[str, bool],
    errors: list[str],
    dry_run: bool,
    run_fn=run,
) -> bool:
    """Run one pipeline step, honouring its upstream dependencies.

    If any upstream isn't satisfied, the step is skipped (results[name]=False,
    a SKIP line is logged, and the missing deps are appended to `errors`).
    Otherwise the step is run via `run_fn` and the result recorded.
    """
    unmet = _unmet_deps(depends_on, results)
    if unmet:
        log.warning("SKIP    %s — upstream not satisfied: %s",
                    name, ", ".join(unmet))
        results[name] = False
        errors.append(f"{name} (skipped — upstream {unmet} not satisfied)")
        return False

    cmd     = [PYTHON, "-u", *script_args]
    timeout = TIMEOUTS.get(name, 1800)
    ok      = run_fn(cmd, timeout=timeout, dry_run=dry_run)
    results[name] = ok
    if not ok:
        errors.append(name)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated pipeline update")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Print steps without executing")
    parser.add_argument("--force-weekly",   action="store_true",
                        help="Run the full weekly rebuild today regardless of day")
    parser.add_argument("--skip-returns",   action="store_true",
                        help="Skip price update (downstream runs against existing DB)")
    parser.add_argument("--skip-svr",       action="store_true",
                        help="Skip FINRA SVR update")
    parser.add_argument("--skip-filings",   action="store_true",
                        help="Skip EDGAR filings (downstream runs against existing DB)")

    args = parser.parse_args()

    today       = date.today()
    is_friday   = today.weekday() == 4
    run_weekly  = is_friday or args.force_weekly
    snap_date   = today.isoformat()

    log.info("=" * 60)
    log.info("Pipeline update starting — %s", datetime.now().isoformat())
    log.info("Python: %s", PYTHON)
    log.info("Weekly rebuild: %s", f"YES ({snap_date})" if run_weekly else "no (not Friday)")
    log.info("=" * 60)

    # results[name] ∈ {True, False}.  Steps marked True (either ran successfully
    # or user explicitly skipped) allow downstream steps to proceed.  False
    # means failed/timed out — downstream is skipped.
    results: dict[str, bool] = {}
    errors:  list[str]       = []

    def step(name: str, *script_args: str, depends_on: tuple[str, ...] = ()) -> None:
        _execute_step(name, script_args, depends_on, results, errors, args.dry_run)

    # ── Daily steps ────────────────────────────────────────────────────────
    if args.skip_returns:
        log.info("SKIP    returns (--skip-returns; downstream uses existing DB)")
        results["returns"] = True
    else:
        step("returns", "create_returns.py", "--update")

    # SVR writes to returns.db too — must run after returns to avoid lock conflict.
    if args.skip_svr:
        log.info("SKIP    svr (--skip-svr)")
        results["svr"] = True
    else:
        step("svr", "create_svr.py", depends_on=("returns",))

    if args.skip_filings:
        log.info("SKIP    filings (--skip-filings; downstream uses existing DB)")
        results["filings"] = True
    else:
        step("filings", "update_constituents.py")

    # ── Weekly rebuild ─────────────────────────────────────────────────────
    if run_weekly:
        log.info("--- Weekly rebuild: snapshot %s ---", snap_date)

        # Discover latest IWB N-PORT-P accession for this snapshot date and
        # refresh universe_snapshots. Required for Barra's PIT R1000 filter.
        step("universe", "create_universe.py", "--ensure-snapshot", snap_date,
             depends_on=("filings",))
        step("factors", "create_factors.py", "--date", snap_date,
             depends_on=("returns", "filings"))
        step("models",  "create_models.py",  "--date", snap_date,
             depends_on=("factors",))
        step("risk",    "create_risk.py",    "--date", snap_date,
             depends_on=("returns",))
        step("barra",   "create_barra.py",
             depends_on=("factors", "returns", "universe"))

        # Portfolio optimiser is on-demand only — run manually when needed.
        # Excluded from automated weekly to avoid long MOSEK MIP runtimes
        # and to allow reviewing risk model output before re-optimising.

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("=" * 60)
    if errors:
        log.error("Completed with %d error(s):", len(errors))
        for e in errors:
            log.error("  FAILED/SKIPPED: %s", e)
        sys.exit(1)
    log.info("All steps completed successfully.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
