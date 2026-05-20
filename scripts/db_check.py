#!/usr/bin/env python3
"""Health check across all pipeline databases."""
import sqlite3
from datetime import date

def q(db, sql, *args):
    try:
        conn = sqlite3.connect(db)
        row = conn.execute(sql, args).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        return f"ERR: {e}"

today = str(date.today())
print("=== Pipeline DB Health Check ===")
print(f"Today: {today}\n")

n_companies   = q("data/universe.db", "SELECT COUNT(*) FROM companies")
n_with_cik    = q("data/universe.db", "SELECT COUNT(*) FROM companies WHERE cik IS NOT NULL")
n_with_simfin = q("data/universe.db", "SELECT COUNT(*) FROM companies WHERE simfin_id IS NOT NULL")
print(f"universe.db     {n_companies} companies  ({n_with_cik} with CIK, {n_with_simfin} with SimFin ID)")

n_rows    = q("data/constituents.db", "SELECT COUNT(*) FROM constituents")
latest_pub = q("data/constituents.db", "SELECT MAX(publish_date) FROM constituents")
n_tickers  = q("data/constituents.db", "SELECT COUNT(DISTINCT security_id) FROM constituents")
n_fy2025   = q("data/constituents.db",
    "SELECT COUNT(DISTINCT security_id) FROM constituents "
    "WHERE fiscal_year=2025 AND fiscal_period IN ('Q4','FY') AND statement_type='Income Statement'")
print(f"constituents.db {n_rows:,} rows  {n_tickers} securities  latest_publish={latest_pub}  FY2025_income={n_fy2025}")

n_prices   = q("data/returns.db", "SELECT COUNT(*) FROM returns")
latest_px  = q("data/returns.db", "SELECT MAX(date) FROM returns")
n_svr      = q("data/returns.db", "SELECT COUNT(*) FROM svr_daily")
latest_svr = q("data/returns.db", "SELECT MAX(date) FROM svr_daily")
print(f"returns.db      {n_prices:,} price rows  latest={latest_px}  svr={n_svr:,} rows  latest_svr={latest_svr}")

n_factors   = q("data/factors.db", "SELECT COUNT(*) FROM factors")
latest_fac  = q("data/factors.db", "SELECT MAX(data_date) FROM factors")
n_fac_dates = q("data/factors.db", "SELECT COUNT(DISTINCT data_date) FROM factors")
print(f"factors.db      {n_factors:,} rows  {n_fac_dates} snapshots  latest={latest_fac}")

n_models    = q("data/models.db", "SELECT COUNT(*) FROM models")
latest_mod  = q("data/models.db", "SELECT MAX(data_date) FROM models")
n_mod_dates = q("data/models.db", "SELECT COUNT(DISTINCT data_date) FROM models")
print(f"models.db       {n_models:,} rows  {n_mod_dates} snapshots  latest={latest_mod}")

n_risk      = q("data/risk.db", "SELECT COUNT(*) FROM covariance_matrix")
latest_risk = q("data/risk.db", "SELECT MAX(data_date) FROM covariance_matrix")
print(f"risk.db         {n_risk} covariance snapshots  latest={latest_risk}")

latest_barra = q("data/risk.db", "SELECT MAX(snapshot_date) FROM factor_covariance")
n_barra_fr   = q("data/risk.db", "SELECT COUNT(DISTINCT trade_date) FROM factor_returns")
latest_bfr   = q("data/risk.db", "SELECT MAX(trade_date) FROM factor_returns")
print(f"risk.db (barra) factor_returns={n_barra_fr} days  latest_fr={latest_bfr}  latest_cov={latest_barra}\n")

dates = {"factors": latest_fac, "models": latest_mod, "risk": latest_risk, "barra": latest_barra}
unique = set(dates.values())
if len(unique) == 1:
    print(f"Sync: OK — all snapshots on {list(unique)[0]}")
else:
    print("Sync: MISMATCH")
    max_date = max(d for d in unique if d)
    for k, v in dates.items():
        lag = "  <-- behind" if v != max_date else ""
        print(f"  {k:8s} {v}{lag}")
