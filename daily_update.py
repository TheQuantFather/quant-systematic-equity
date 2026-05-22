#!/usr/bin/env python3
"""
daily_update.py — Automated pipeline update.

Schedule:
  Daily   — update prices (create_returns.py --update)
  Daily   — EDGAR filing index: process companies that actually filed (10-K + 10-Q)
  Weekly  — on Fridays: rebuild factors, models, risk, Barra, portfolio
             using today's date as the snapshot so everything is aligned

Usage:
    python daily_update.py                   # normal run (Friday triggers full rebuild)
    python daily_update.py --force-weekly    # force full weekly rebuild today
    python daily_update.py --dry-run         # print steps without running
"""

import argparse
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

from utils import get_logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PYTHON   = sys.executable
REPO_DIR = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = get_logger("daily_update")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(cmd: list[str], dry_run: bool = False) -> bool:
    """Run a subprocess command, stream output to log, return True on success."""
    label = " ".join(cmd[1:])   # drop python path for readability
    log.info(f"START  {label}")
    if dry_run:
        log.info(f"  [dry-run] {' '.join(cmd)}")
        return True
    try:
        result = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            log.info(f"  {line}")
        if result.returncode != 0:
            log.error(f"FAILED {label} (exit {result.returncode})")
            for line in result.stderr.splitlines():
                log.error(f"  {line}")
            return False
        log.info(f"OK     {label}")
        return True
    except Exception as exc:
        log.error(f"FAILED {label} — {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Automated pipeline update")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print steps without executing")
    parser.add_argument("--force-weekly", action="store_true",
                        help="Run the full weekly rebuild today regardless of day")
    parser.add_argument("--skip-returns", action="store_true",
                        help="Skip price update")
    parser.add_argument("--skip-filings", action="store_true",
                        help="Skip EDGAR filing steps")
    parser.add_argument("--skip-portfolio", action="store_true",
                        help="Skip portfolio optimiser")
    args = parser.parse_args()

    today       = date.today()
    is_friday   = today.weekday() == 4   # Monday=0, Friday=4
    run_weekly  = is_friday or args.force_weekly
    snap_date   = today.isoformat()

    log.info("=" * 60)
    log.info(f"Pipeline update starting — {datetime.now().isoformat()}")
    log.info(f"Weekly rebuild: {'YES (' + snap_date + ')' if run_weekly else 'no (not Friday)'}")
    log.info("=" * 60)

    errors: list[str] = []

    def step(cmd: list[str]) -> None:
        if not run(cmd, dry_run=args.dry_run):
            errors.append(" ".join(cmd[1:]))

    # ------------------------------------------------------------------
    # Daily: update prices
    # ------------------------------------------------------------------
    if not args.skip_returns:
        step([PYTHON, "create_returns.py", "--update"])

    # ------------------------------------------------------------------
    # Daily: EDGAR filing index (10-K + 10-Q, last 8 days)
    # ------------------------------------------------------------------
    if not args.skip_filings:
        step([PYTHON, "update_constituents.py"])

    # ------------------------------------------------------------------
    # Weekly (Friday): rebuild full model stack with today as snapshot
    # ------------------------------------------------------------------
    if run_weekly:
        log.info(f"--- Weekly rebuild: snapshot {snap_date} ---")
        step([PYTHON, "create_factors.py",  "--date", snap_date])
        step([PYTHON, "create_models.py",   "--date", snap_date])
        step([PYTHON, "create_risk.py",     "--date", snap_date])
        step([PYTHON, "create_barra.py"])

    # ------------------------------------------------------------------
    # Weekly: portfolio optimiser (only meaningful after model rebuild)
    # ------------------------------------------------------------------
    if run_weekly and not args.skip_portfolio:
        step([PYTHON, "optimize_portfolio.py"])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("=" * 60)
    if errors:
        log.error(f"Completed with {len(errors)} error(s):")
        for e in errors:
            log.error(f"  FAILED: {e}")
        sys.exit(1)
    else:
        log.info("All steps completed successfully.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
