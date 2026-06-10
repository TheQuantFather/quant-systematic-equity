#!/usr/bin/env python3
"""Quantitative diagnostics for a single factor model — backend for /model-review.

Emits the mechanical parts of a holistic model review (composition, coverage,
outliers, redundancy, sector bias, predictive power) for any model defined in
models_reference.csv. The /model-review command feeds this output to Claude, which
adds the judgement parts: calculation audit (reading the compute_* code), literature
cross-check, and the ranked findings write-up.

Generic: nothing about the model is hardcoded — factors, weights, directions, and
sector overrides are read from the reference CSVs, so it works for any base or
composite model (VAL001, PROF001, ALP001, …).

Usage:
    python scripts/model_review.py Value
    python scripts/model_review.py VAL001
"""
import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for config/utils
from config import FACTORS_DB, MODELS_DB, UNIVERSE_DB, RETURNS_DB, FACTORS_REF, MODELS_REF
from utils import classify_sector, factor_applies_to_company

pd.set_option('display.width', 140)
pd.set_option('display.max_columns', 30)

MIN_COVERAGE = 0.5  # mirror create_models coverage floor


def resolve_model(arg: str, mref: pd.DataFrame) -> tuple[str, str, bool]:
    """Return (model_id, model_name, is_composite) from a name or id argument."""
    a = arg.strip().lower()
    hit = mref[(mref['ModelID'].str.lower() == a) | (mref['Model'].str.lower() == a)]
    if hit.empty:
        names = mref[['Model', 'ModelID']].drop_duplicates()
        sys.exit(f"Model '{arg}' not found. Available:\n{names.to_string(index=False)}")
    row = hit.iloc[0]
    return row['ModelID'], row['Model'], bool(int(row['IsComposite']))


def latest_snapshot() -> str:
    with sqlite3.connect(FACTORS_DB) as c:
        return c.execute("SELECT MAX(data_date) FROM snapshot_dates").fetchone()[0]


def company_sector_types() -> dict:
    with sqlite3.connect(UNIVERSE_DB) as c:
        rows = c.execute("SELECT isin, simfin_sector, simfin_industry FROM companies").fetchall()
    return {i: classify_sector(s, ind) for i, s, ind in rows}


def gics_sectors() -> dict:
    with sqlite3.connect(UNIVERSE_DB) as c:
        rows = c.execute("SELECT isin, simfin_sector FROM companies").fetchall()
    return dict(rows)


def tickers() -> dict:
    with sqlite3.connect(UNIVERSE_DB) as c:
        return dict(c.execute("SELECT isin, ticker FROM companies").fetchall())


def forward_returns(snaps: list[str]) -> pd.DataFrame:
    """Compounded total return from each snapshot to the next, per security."""
    with sqlite3.connect(RETURNS_DB) as c:
        ret = pd.read_sql_query(
            "SELECT isin, date, total_return FROM returns WHERE date >= ?",
            c, params=(snaps[0],))
    ret['date'] = ret['date'].astype(str)
    frames = []
    for d, dn in zip(snaps[:-1], snaps[1:]):
        win = ret[(ret['date'] > d) & (ret['date'] <= dn)]
        fr = win.groupby('isin')['total_return'].apply(lambda x: np.prod(1 + x.values) - 1)
        frames.append(pd.DataFrame({'data_date': d, 'security_id': fr.index, 'fwd_ret': fr.values}))
    return pd.concat(frames, ignore_index=True)


def ic_stats(panel: pd.DataFrame, valcol: str, fwd: pd.DataFrame) -> tuple:
    """(n_dates, mean_IC, t_stat, hit_rate) of cross-sectional rank IC vs fwd return."""
    g = panel.merge(fwd, on=['data_date', 'security_id']).dropna(subset=[valcol, 'fwd_ret'])
    ics = {d: gg[valcol].corr(gg['fwd_ret'], method='spearman')
           for d, gg in g.groupby('data_date') if len(gg) >= 30}
    s = pd.Series(ics).dropna()
    if s.empty:
        return (0, np.nan, np.nan, np.nan)
    t = s.mean() / s.std() * np.sqrt(len(s)) if s.std() else np.nan
    return (len(s), round(s.mean(), 4), round(t, 2), round((s > 0).mean(), 2))


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/model_review.py <model name or id>")
    arg = ' '.join(sys.argv[1:])

    mref = pd.read_csv(MODELS_REF)
    fref = pd.read_csv(FACTORS_REF)
    model_id, model_name, is_comp = resolve_model(arg, mref)
    rows = mref[mref['ModelID'] == model_id].copy()
    latest = latest_snapshot()
    ctype = company_sector_types()
    gics = gics_sectors()
    tkr = tickers()

    print(f"================ MODEL REVIEW: {model_name} ({model_id}) ================")
    print(f"composite={is_comp}   latest snapshot={latest}\n")

    # ---- 1. Composition -----------------------------------------------------
    print("## 1. Composition")
    if is_comp:
        comp = rows[['Factors', 'Weights', 'Description']].rename(
            columns={'Factors': 'base_model_id'})
        comp['base_model'] = comp['base_model_id'].map(
            dict(zip(mref['ModelID'], mref['Model'])))
        comp['weight_norm'] = (comp['Weights'] / comp['Weights'].sum()).round(3)
        print(comp[['base_model_id', 'base_model', 'Weights', 'weight_norm', 'Description']]
              .to_string(index=False))
        constituents = list(comp['base_model_id'])
        direction = {m: 1 for m in constituents}
    else:
        comp = rows.merge(fref, left_on='Factors', right_on='factor_id', how='left')
        comp['sector_override'] = comp['sector_type_x'].fillna('') if 'sector_type_x' in comp \
            else comp.get('sector_type', '')
        show = comp[['factor_id', 'factor_name', 'Weights', 'direction',
                     'sector_type_y' if 'sector_type_y' in comp else 'sector_type',
                     'sector_override', 'formula']].copy()
        show.columns = ['factor_id', 'name', 'weight', 'dir', 'factor_sector', 'model_override', 'formula']
        print(show.to_string(index=False))
        constituents = list(comp['factor_id'])
        direction = dict(zip(comp['factor_id'], comp['direction']))
    weights = dict(zip(rows['Factors'], rows['Weights']))
    overrides = {}
    if not is_comp and 'sector_type' in rows:
        for _, r in rows.iterrows():
            raw = str(r.get('sector_type') or '').strip()
            overrides[r['Factors']] = frozenset(raw.split('|')) if raw and raw != 'nan' else None
    fsec = dict(zip(fref['factor_id'], fref['sector_type'].fillna('all')))

    # ---- Pull latest-snapshot constituent values ---------------------------
    if is_comp:
        with sqlite3.connect(MODELS_DB) as c:
            vals = pd.read_sql_query(
                "SELECT security_id, model_id AS cid, model_value AS val, model_value_z AS z "
                "FROM models WHERE data_date=? AND model_id IN (%s)" %
                ','.join('?' * len(constituents)), c, params=[latest, *constituents])
    else:
        with sqlite3.connect(FACTORS_DB) as c:
            vals = pd.read_sql_query(
                "SELECT security_id, factor_id AS cid, factor_value AS val, factor_value_z AS z "
                "FROM factors WHERE data_date=? AND factor_id IN (%s)" %
                ','.join('?' * len(constituents)), c, params=[latest, *constituents])
    name_of = (dict(zip(fref['factor_id'], fref['factor_name'])) if not is_comp
               else dict(zip(mref['ModelID'], mref['Model'])))
    vals['cname'] = vals['cid'].map(name_of)
    vals['stype'] = vals['security_id'].map(ctype).fillna('?')

    # ---- 2. Coverage --------------------------------------------------------
    print("\n## 2. Coverage by constituent × company sector_type (latest)")
    cov = vals.pivot_table(index='cname', columns='stype', values='security_id',
                           aggfunc='count', fill_value=0)
    print(cov.to_string())

    # model-level coverage ratio per security (applicable-weight aware, base models only)
    if not is_comp:
        present = vals.dropna(subset=['z']).groupby('security_id')['cid'].apply(set).to_dict()
        ratios = []
        for sid in present:                      # only names in this snapshot
            ct = ctype.get(sid, 'general')
            applic = 0.0
            valid = 0.0
            for fid, w in weights.items():
                ov = overrides.get(fid)
                applies = (ct in ov) if ov else factor_applies_to_company(fsec.get(fid, 'all'), ct)
                if not applies:
                    continue
                applic += w
                if fid in present.get(sid, set()):
                    valid += w
            if applic > 0:
                ratios.append(valid / applic)
        r = pd.Series(ratios)
        print(f"\ncoverage ratio across {len(r)} names: "
              f"min={r.min():.2f} median={r.median():.2f} | "
              f"below {MIN_COVERAGE:.0%} floor (shrunk toward neutral): {(r < MIN_COVERAGE).sum()}")

    # ---- 3. Outliers --------------------------------------------------------
    print("\n## 3. Raw-value outliers per constituent (top/bottom 3, latest)")
    for cid in constituents:
        sub = vals[vals['cid'] == cid].dropna(subset=['val'])
        if sub.empty:
            continue
        sub = sub.assign(ticker=sub['security_id'].map(tkr))
        hi = sub.nlargest(3, 'val')[['ticker', 'val']].to_dict('records')
        lo = sub.nsmallest(3, 'val')[['ticker', 'val']].to_dict('records')
        nm = name_of.get(cid, cid)
        fmt = lambda d: ', '.join(f"{x['ticker']}={x['val']:.3g}" for x in d)
        print(f"  {nm:28} HIGH[{fmt(hi)}]  LOW[{fmt(lo)}]")

    # ---- 4. Redundancy ------------------------------------------------------
    print("\n## 4. Constituent z-score correlations (latest, |ρ|>0.7 flagged)")
    piv = vals.pivot_table(index='security_id', columns='cname', values='z')
    corr = piv.corr().round(2)
    print(corr.to_string())
    pairs = [(corr.index[i], corr.columns[j], corr.iloc[i, j])
             for i in range(len(corr)) for j in range(i + 1, len(corr))
             if abs(corr.iloc[i, j]) > 0.7]
    if pairs:
        print("  REDUNDANT:", '; '.join(f"{a}~{b} ({c})" for a, b, c in pairs))

    # ---- 5. Sector bias -----------------------------------------------------
    print("\n## 5. Model z by GICS sector (latest) — mean should not be systematically extreme")
    with sqlite3.connect(MODELS_DB) as c:
        mv = pd.read_sql_query(
            "SELECT security_id, model_value_z z FROM models WHERE data_date=? AND model_id=?",
            c, params=(latest, model_id))
    mv['sec'] = mv['security_id'].map(gics)
    mv['stype'] = mv['security_id'].map(ctype)
    print(mv.groupby('sec')['z'].agg(['count', 'mean', 'std']).round(3).sort_values('mean').to_string())
    print("\n   by sector_type (dispersion σ should be comparable across groups):")
    print(mv.groupby('stype')['z'].agg(['count', 'mean', 'std']).round(3).to_string())
    within = mv.dropna(subset=['z', 'sec'])
    if within['z'].var() > 0:
        expl = 1 - within.groupby('sec')['z'].transform(lambda x: x - x.mean()).var() / within['z'].var()
        print(f"\n   fraction of model-z variance explained by sector: {expl:.1%}")

    # ---- 6. Predictive power ------------------------------------------------
    print("\n## 6. Predictive power — rank IC vs next-snapshot forward return (all snapshots)")
    with sqlite3.connect(FACTORS_DB) as c:
        snaps = [r[0] for r in c.execute("SELECT data_date FROM snapshot_dates ORDER BY data_date")]
    fwd = forward_returns(snaps)
    out = []
    # composite model IC
    with sqlite3.connect(MODELS_DB) as c:
        mpanel = pd.read_sql_query(
            "SELECT data_date, security_id, model_value_z z FROM models WHERE model_id=?",
            c, params=(model_id,))
    out.append((f"** {model_name} (composite z)", *ic_stats(mpanel, 'z', fwd)))
    # per-constituent IC (direction-adjusted)
    if is_comp:
        with sqlite3.connect(MODELS_DB) as c:
            cp = pd.read_sql_query(
                "SELECT data_date, security_id, model_id AS cid, model_value_z z FROM models "
                "WHERE model_id IN (%s)" % ','.join('?' * len(constituents)), c, params=constituents)
    else:
        with sqlite3.connect(FACTORS_DB) as c:
            cp = pd.read_sql_query(
                "SELECT data_date, security_id, factor_id AS cid, factor_value_z z FROM factors "
                "WHERE factor_id IN (%s)" % ','.join('?' * len(constituents)), c, params=constituents)
    for cid in constituents:
        sub = cp[cp['cid'] == cid].copy()
        sub['sig'] = sub['z'] * direction.get(cid, 1)
        out.append((name_of.get(cid, cid), *ic_stats(sub, 'sig', fwd)))
    ic_df = pd.DataFrame(out, columns=['constituent', 'n_dates', 'mean_IC', 't_stat', 'hit'])
    print(ic_df.to_string(index=False))
    print("\n(IC is direction-adjusted: positive = the factor as used predicts higher forward return.)")


if __name__ == '__main__':
    main()
