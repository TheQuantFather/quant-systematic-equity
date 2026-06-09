#!/usr/bin/env python3
"""
daily_position_update.py - Automated broker position/cash snapshot update.

Schedule:
  Daily - IBKR Flex statement snapshot for the US quant portfolio
  Daily - DEGIRO account-update snapshot for the opportunistic buying portfolio

Behaviour:
  - Runs each broker independently and logs one START/OK/FAILED line per job.
  - A failed broker does not stop the next broker from running.
  - Exits non-zero if any broker sync fails.
  - Uses IBKR Flex by default, so no TWS/Gateway login is needed for the daily
    reporting snapshot.

Usage:
    python daily_position_update.py
    python daily_position_update.py --dry-run
    python daily_position_update.py --skip-degiro
    python daily_position_update.py --only ibkr
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from brokers.broker_sync import sync_snapshot
from brokers.schema import PORTFOLIO_ANALYTICS_DB
from utils import get_logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).parent.resolve()


@dataclass(frozen=True)
class SnapshotJob:
    name: str
    broker: str
    portfolio: str
    source: str


DEFAULT_JOBS = [
    SnapshotJob(name="ibkr", broker="ibkr", portfolio="ibkr_us_quant", source="flex"),
    SnapshotJob(name="degiro", broker="degiro", portfolio="degiro_us_opportunistic", source="api"),
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = get_logger("daily_position_update")


# ---------------------------------------------------------------------------
# Broker sync orchestration
# ---------------------------------------------------------------------------

def _date_arg(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def _selected_jobs(args: argparse.Namespace) -> list[SnapshotJob]:
    jobs = DEFAULT_JOBS
    if args.only is not None:
        jobs = [job for job in jobs if job.name == args.only]
    if args.skip_ibkr:
        jobs = [job for job in jobs if job.name != "ibkr"]
    if args.skip_degiro:
        jobs = [job for job in jobs if job.name != "degiro"]
    return jobs


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.from_date and args.to_date and args.from_date > args.to_date:
        parser.error("--from-date must be on or before --to-date")
    if args.flex_wait_seconds < 1:
        parser.error("--flex-wait-seconds must be at least 1")
    if args.flex_max_attempts < 1:
        parser.error("--flex-max-attempts must be at least 1")
    if args.skip_ibkr and args.skip_degiro:
        parser.error("At least one broker must be enabled")
    if args.only == "ibkr" and args.skip_ibkr:
        parser.error("--only ibkr conflicts with --skip-ibkr")
    if args.only == "degiro" and args.skip_degiro:
        parser.error("--only degiro conflicts with --skip-degiro")


def _build_sync_args(job: SnapshotJob, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        db=args.db,
        broker=job.broker,
        portfolio=job.portfolio,
        snapshot=True,
        source=job.source,
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        account=None,
        allow_live_account=False,
        flex_query_id=args.flex_query_id,
        from_date=args.from_date,
        to_date=args.to_date,
        flex_wait_seconds=args.flex_wait_seconds,
        flex_max_attempts=args.flex_max_attempts,
        allow_inferred_flex_nav=False,
    )


def _run_job(job: SnapshotJob, args: argparse.Namespace) -> bool:
    label = f"{job.name} ({job.portfolio}, source={job.source})"
    log.info("START   %s", label)
    if args.dry_run:
        log.info("  [dry-run] broker=%s portfolio=%s source=%s", job.broker, job.portfolio, job.source)
        log.info("OK      %s", label)
        return True

    t0 = time.time()
    try:
        sync_snapshot(_build_sync_args(job, args))
    except Exception:
        log.exception("FAILED  %s (%.0fs)", label, time.time() - t0)
        return False
    log.info("OK      %s (%.0fs)", label, time.time() - t0)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated broker position/cash snapshot update")
    parser.add_argument("--db", type=Path, default=PORTFOLIO_ANALYTICS_DB)
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without executing")
    parser.add_argument("--only", choices=["ibkr", "degiro"], default=None, help="Run only one broker")
    parser.add_argument("--skip-ibkr", action="store_true", help="Skip IBKR Flex snapshot")
    parser.add_argument("--skip-degiro", action="store_true", help="Skip DEGIRO snapshot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=91)
    parser.add_argument("--flex-query-id", default=None)
    parser.add_argument("--from-date", type=_date_arg, default=None)
    parser.add_argument("--to-date", type=_date_arg, default=None)
    parser.add_argument("--flex-wait-seconds", type=int, default=5)
    parser.add_argument("--flex-max-attempts", type=int, default=12)
    args = parser.parse_args()
    _validate_args(parser, args)

    os.chdir(REPO_DIR)
    jobs = _selected_jobs(args)
    log.info("=" * 60)
    log.info("Daily position update starting - %s", datetime.now().isoformat())
    log.info("Python: %s", sys.executable)
    log.info("Repo: %s", REPO_DIR)
    log.info("Database: %s", args.db)
    log.info("Jobs: %s", ", ".join(job.name for job in jobs) or "none")
    log.info("=" * 60)

    errors: list[str] = []
    for job in jobs:
        if not _run_job(job, args):
            errors.append(job.name)

    log.info("=" * 60)
    if errors:
        log.error("Completed with %d error(s):", len(errors))
        for name in errors:
            log.error("  FAILED: %s", name)
        log.info("=" * 60)
        sys.exit(1)
    log.info("All broker snapshots completed successfully.")
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.error("Interrupted by user")
        sys.exit(130)
    except SystemExit:
        raise
    except Exception:
        log.exception("Unhandled daily_position_update failure")
        sys.exit(1)
