#!/usr/bin/env python3
"""LTM financial summary for a ticker from constituents.db."""
import sys
import sqlite3
from collections import defaultdict

import pandas as pd

TICKER = sys.argv[1].upper() if len(sys.argv) > 1 else ""
if not TICKER:
    print("Usage: validate_ticker.py <TICKER>")
    sys.exit(1)

conn_u = sqlite3.connect("data/universe.db")
row = conn_u.execute("SELECT isin, simfin_id FROM companies WHERE ticker=?", (TICKER,)).fetchone()
conn_u.close()
if not row:
    print(f"Ticker {TICKER} not found in universe.db")
    sys.exit(1)
isin, simfin_id = row
print(f"{TICKER}  isin={isin}  simfin_id={simfin_id}")

ref = pd.read_csv("data/constituents_reference.csv")
id_to_kind = dict(zip(ref.constituent_id, ref.data_kind))

conn = sqlite3.connect("data/constituents.db")
sids = [isin] + ([str(simfin_id)] if simfin_id else [])
placeholders = ",".join("?" * len(sids))
rows = conn.execute(
    f"SELECT security_id, constituent_id, constituent_value, fiscal_year, fiscal_period, publish_date "
    f"FROM constituents "
    f"WHERE security_id IN ({placeholders}) "
    f"  AND fiscal_period IN ('Q1','Q2','Q3','Q4') "
    f"  AND publish_date IS NOT NULL "
    f"ORDER BY fiscal_year DESC, fiscal_period DESC, publish_date DESC",
    sids,
).fetchall()
conn.close()

seen: dict = {}
quarters: dict = defaultdict(dict)
for sid, cid, val, fy, fp, pub in rows:
    k = (cid, fy, fp)
    if k not in seen or pub > seen[k]:
        seen[k] = pub
        quarters[(fy, fp)][cid] = val

period_order = {"Q4": 4, "Q3": 3, "Q2": 2, "Q1": 1}
sorted_periods = sorted(quarters.keys(), key=lambda x: (x[0], period_order.get(x[1], 0)), reverse=True)
recent_4 = sorted_periods[:4]
latest_q = sorted_periods[0] if sorted_periods else None

print(f"Periods in DB: {[(fy, fp) for fy, fp in sorted_periods[:8]]}")
print(f"LTM from:      {recent_4}\n")

def ltm(cid: str):
    if id_to_kind.get(cid) == "Stock":
        return quarters.get(latest_q, {}).get(cid)
    total, missing = 0, 0
    for period in recent_4:
        v = quarters.get(period, {}).get(cid)
        if v is None:
            missing += 1
        else:
            total += v
    return None if missing > 1 else total

def fmt(v, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    if pct:
        return f"{v * 100:.1f}%"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.1f}M"
    return f"${v:,.0f}"

rev   = ltm("9801FC7E")
gp    = ltm("7A1B2BB6")
ebit  = ltm("80C2558A")
ni    = ltm("CDD1D338")
da    = ltm("E7754E82")
ocf   = ltm("6835500D")
capex = ltm("CA8A4027")
fcf   = (ocf + capex) if (ocf is not None and capex is not None) else None
ebitda = (ebit + da) if (ebit is not None and da is not None) else None
gm    = (gp / rev) if (gp and rev) else None
nm    = (ni / rev) if (ni is not None and rev) else None
assets  = ltm("3BD29B6F")
equity  = ltm("06EF64B2")
lt_debt = ltm("D7815EBF")
st_debt = ltm("2D2B0CAC")
cash    = ltm("79E5D14B")
net_debt = ((lt_debt or 0) + (st_debt or 0) - (cash or 0)) if (lt_debt is not None or st_debt is not None) else None

print("  Income Statement (LTM)")
print(f"    Revenue:          {fmt(rev)}")
print(f"    Gross Profit:     {fmt(gp)}  ({fmt(gm, pct=True)} margin)")
print(f"    Operating Income: {fmt(ebit)}")
print(f"    EBITDA:           {fmt(ebitda)}")
print(f"    Net Income:       {fmt(ni)}  ({fmt(nm, pct=True)} margin)")
print()
print("  Cash Flow (LTM)")
print(f"    Operating CF:     {fmt(ocf)}")
print(f"    CapEx:            {fmt(capex)}")
print(f"    FCF:              {fmt(fcf)}")
print()
print("  Balance Sheet (latest quarter)")
print(f"    Total Assets:     {fmt(assets)}")
print(f"    Total Equity:     {fmt(equity)}")
print(f"    Net Debt:         {fmt(net_debt)}")
