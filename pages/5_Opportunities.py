"""
5_Opportunities.py — Opportunistic-buying opportunity scanner.

Answers "where in the market is there an opportunity?" for an opportunistic
buying book, by ranking industries (and sectors) on four stacked pillars:

  1. Correction   — how far price has fallen (drawdown from 52-wk high, 12M return)
  2. Cheapness    — valuation vs the group's OWN 3–5yr history (dislocation, not
                    just structurally-cheap sectors)
  3. Stabilization— recent 1M return vs the trailing 12M monthly pace (favour
                    "corrected but turning" over "still a falling knife")
  4. Quality gate — profitability / balance-sheet quality + fundamental trend,
                    used to flag value traps rather than exclude them.

Composite opportunity score = equal-weight mean of correction rank, own-history
cheapness percentile, and stabilization rank (all in [0,1]). Quality colours the
map and drives the value-trap flag; it does not gate the ranking.

Aggregation is at `simfin_industry` level (the de-facto industry field in this
codebase — `gics_industry` is not populated). Thin industries below a name-count
floor are suppressed from the ranking; a Sector toggle rolls up to GICS sector
for a robust top-down read.

Pure read/aggregate layer over models.db + factors.db + returns.db + universe.db
— no pipeline dependencies.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import get_db
from config import UNIVERSE_DB, RETURNS_DB, FACTORS_DB, MODELS_DB

st.set_page_config(page_title="Opportunities", layout="wide")
try:
    from utils import inject_css
    inject_css()
except Exception:
    pass
st.title("Opportunities")

# Trading-day windows for trailing metrics
WIN_1M, WIN_6M, WIN_12M, WIN_52W, WIN_200D = 21, 126, 252, 252, 200

# Models used per pillar (direction already applied in models.db, so higher = better)
VAL_ID, PROF_ID, DEF_ID, GRO_ID, ALP_ID, SHI_ID = (
    "VAL001", "PROF001", "DEF001", "GRO001", "ALP001", "SHI001",
)


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_universe() -> pd.DataFrame:
    """Current investable set = securities present in the latest models snapshot,
    joined to sector/industry. simfin_industry is the industry field."""
    with get_db(MODELS_DB) as conn:
        latest = conn.execute("SELECT MAX(data_date) FROM models").fetchone()[0]
        ids = pd.read_sql(
            "SELECT DISTINCT security_id FROM models WHERE data_date = ?",
            conn, params=(latest,),
        )["security_id"].astype(str).tolist()
    with get_db(UNIVERSE_DB) as conn:
        comp = pd.read_sql(
            "SELECT isin AS security_id, ticker, company_name, "
            "       gics_sector, simfin_industry FROM companies",
            conn,
        )
    comp["security_id"] = comp["security_id"].astype(str)
    comp = comp[comp["security_id"].isin(ids)].copy()
    comp["sector"] = comp["gics_sector"].fillna("").replace("", "Unclassified")
    comp["industry"] = comp["simfin_industry"].fillna("").replace("", "Unclassified")
    comp["ticker"] = comp["ticker"].fillna("")
    return comp, latest


@st.cache_data(show_spinner=False)
def load_market_caps(latest_date: str) -> pd.Series:
    """Market cap per security = exp(Log Market Cap factor) at the latest snapshot."""
    with get_db(FACTORS_DB) as conn:
        df = pd.read_sql(
            "SELECT security_id, factor_value FROM factors "
            "WHERE factor_id = 'LMC11234' AND data_date = ?",
            conn, params=(latest_date,),
        )
    df["security_id"] = df["security_id"].astype(str)
    df["mktcap"] = np.exp(pd.to_numeric(df["factor_value"], errors="coerce"))
    return df.set_index("security_id")["mktcap"]


@st.cache_data(show_spinner=False)
def load_price_metrics(isins: tuple[str, ...]) -> pd.DataFrame:
    """Per-security trailing return / drawdown / stabilization metrics from the
    total-return index (compounded total_return)."""
    with get_db(RETURNS_DB) as conn:
        df = pd.read_sql(
            "SELECT date, isin, total_return, close FROM returns "
            f"WHERE isin IN ({','.join('?' * len(isins))}) ORDER BY isin, date",
            conn, params=list(isins),
        )
    if df.empty:
        return pd.DataFrame()
    df["total_return"] = pd.to_numeric(df["total_return"], errors="coerce").fillna(0.0)

    out = []
    for isin, g in df.groupby("isin", sort=False):
        tr = (1.0 + g["total_return"]).cumprod().to_numpy()
        n = len(tr)
        if n < WIN_1M + 2:
            continue

        def trail(w: int) -> float:
            return tr[-1] / tr[-w - 1] - 1.0 if n > w else np.nan

        ret_1m, ret_6m, ret_12m = trail(WIN_1M), trail(WIN_6M), trail(WIN_12M)
        hi = np.nanmax(tr[-WIN_52W:]) if n >= 20 else np.nan
        dd_from_high = tr[-1] / hi - 1.0 if hi and hi > 0 else np.nan
        ma200 = np.nanmean(tr[-WIN_200D:]) if n >= 20 else np.nan
        below_200d = bool(tr[-1] < ma200) if not np.isnan(ma200) else False
        # Stabilization: recent month vs the year's average monthly pace
        pace = (1.0 + ret_12m) ** (1 / 12) - 1.0 if pd.notna(ret_12m) else np.nan
        stab = ret_1m - pace if pd.notna(ret_1m) and pd.notna(pace) else np.nan

        out.append(dict(
            security_id=str(isin), ret_1m=ret_1m, ret_6m=ret_6m, ret_12m=ret_12m,
            dd_from_high=dd_from_high, below_200d=below_200d, stabilization=stab,
        ))
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False)
def load_model_history() -> pd.DataFrame:
    """Full model z-score time series for the pillars we need."""
    ids = (VAL_ID, PROF_ID, DEF_ID, GRO_ID, ALP_ID, SHI_ID)
    with get_db(MODELS_DB) as conn:
        df = pd.read_sql(
            "SELECT data_date, model_id, security_id, model_value_z FROM models "
            f"WHERE model_id IN ({','.join('?' * len(ids))})",
            conn, params=list(ids),
        )
    df["security_id"] = df["security_id"].astype(str)
    df["model_value_z"] = pd.to_numeric(df["model_value_z"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _pctile_rank(s: pd.Series) -> pd.Series:
    """Cross-group percentile rank in [0,1] (higher value → higher rank)."""
    return s.rank(pct=True)


@st.cache_data(show_spinner=False)
def build_opportunity_table(level: str, min_names: int, trend_window: int) -> pd.DataFrame:
    comp, latest = load_universe()
    caps = load_market_caps(latest)
    px_m = load_price_metrics(tuple(comp["security_id"].tolist()))
    mh = load_model_history()

    grp_col = "sector" if level == "Sector" else "industry"
    comp = comp.merge(caps.rename("mktcap"), left_on="security_id", right_index=True, how="left")
    comp = comp.merge(px_m, on="security_id", how="left")

    # --- Model medians per (date, group) → industry/sector time series ------
    md = mh.merge(comp[["security_id", grp_col]], on="security_id", how="inner")
    grp_ts = (
        md.groupby(["data_date", grp_col, "model_id"])["model_value_z"]
          .median().reset_index()
    )
    dates = sorted(grp_ts["data_date"].unique())
    latest_snap = dates[-1]

    def latest_median(model_id: str) -> pd.Series:
        sl = grp_ts[(grp_ts["model_id"] == model_id) & (grp_ts["data_date"] == latest_snap)]
        return sl.set_index(grp_col)["model_value_z"]

    # Cheapness vs OWN history: percentile of latest VAL median within its own series
    val_ts = grp_ts[grp_ts["model_id"] == VAL_ID].pivot(
        index="data_date", columns=grp_col, values="model_value_z"
    ).sort_index()
    own_pctile, own_depth = {}, {}
    for g in val_ts.columns:
        ser = val_ts[g].dropna()
        own_depth[g] = len(ser)
        if len(ser) >= 4:
            own_pctile[g] = (ser <= ser.iloc[-1]).mean()   # 1.0 = cheapest it's been
    cheap_own = pd.Series(own_pctile, name="cheap_own_pctile")

    # Fundamental trend: change in PROF median over the trailing `trend_window` snapshots
    prof_ts = grp_ts[grp_ts["model_id"] == PROF_ID].pivot(
        index="data_date", columns=grp_col, values="model_value_z"
    ).sort_index()
    if len(prof_ts) > trend_window:
        fund_trend = prof_ts.iloc[-1] - prof_ts.iloc[-1 - trend_window]
    else:
        fund_trend = prof_ts.iloc[-1] - prof_ts.iloc[0]
    fund_trend.name = "fund_trend"

    # --- Price / cap aggregation per group ----------------------------------
    def cap_wt(g: pd.DataFrame, col: str) -> float:
        w = g["mktcap"].where(g[col].notna())
        v = g[col]
        m = v.notna() & w.notna()
        return float(np.average(v[m], weights=w[m])) if m.any() and w[m].sum() > 0 else np.nan

    rows = []
    for g, gd in comp.groupby(grp_col):
        if g == "Unclassified":
            continue
        rows.append(dict(
            group=g, n_names=int(gd["security_id"].nunique()),
            mktcap_bn=float(gd["mktcap"].sum(skipna=True)) / 1e9,
            ret_12m_med=float(gd["ret_12m"].median(skipna=True)),
            ret_12m_capw=cap_wt(gd, "ret_12m"),
            ret_6m_med=float(gd["ret_6m"].median(skipna=True)),
            ret_1m_med=float(gd["ret_1m"].median(skipna=True)),
            dd_med=float(gd["dd_from_high"].median(skipna=True)),
            stab_med=float(gd["stabilization"].median(skipna=True)),
            pct_below_200d=float(gd["below_200d"].mean(skipna=True)),
            pct_neg_12m=float((gd["ret_12m"] < 0).mean()),
        ))
    agg = pd.DataFrame(rows).set_index("group")

    # Attach model medians
    agg["val_med"] = latest_median(VAL_ID)
    agg["prof_med"] = latest_median(PROF_ID)
    agg["def_med"] = latest_median(DEF_ID)
    agg["gro_med"] = latest_median(GRO_ID)
    agg["alp_med"] = latest_median(ALP_ID)
    agg["shi_med"] = latest_median(SHI_ID)
    agg["cheap_own_pctile"] = cheap_own
    agg["own_depth"] = pd.Series(own_depth)
    agg["fund_trend"] = fund_trend

    agg = agg.reset_index()
    agg["thin"] = agg["n_names"] < min_names

    # --- Component ranks (computed on the qualifying, non-thin set) ----------
    ok = agg[~agg["thin"]].copy()
    # Correction: deeper drawdown → higher rank
    ok["correction_rank"] = _pctile_rank(-ok["dd_med"])
    # Stabilization: recent month beating the year's pace → higher rank
    ok["stabilization_rank"] = _pctile_rank(ok["stab_med"])
    # Cheapness: own-history percentile is already [0,1] (1 = cheapest it's been)
    ok["cheapness_component"] = ok["cheap_own_pctile"]
    # Quality: cross-group rank of (prof + def) median → colour + trap flag
    ok["quality_rank"] = _pctile_rank(ok["prof_med"].fillna(0) + ok["def_med"].fillna(0))

    comps = ["correction_rank", "cheapness_component", "stabilization_rank"]
    ok["opportunity_score"] = ok[comps].mean(axis=1, skipna=True)

    # Value-trap flag: weak quality AND deteriorating fundamentals
    ok["trap"] = (ok["quality_rank"] < 0.30) & (ok["fund_trend"] < 0)
    # Divergence (positive setup): fundamentals up but price down
    ok["divergence"] = (ok["gro_med"] > 0) & (ok["ret_12m_med"] < 0)

    merged = agg.merge(
        ok[["group", "correction_rank", "stabilization_rank", "cheapness_component",
            "quality_rank", "opportunity_score", "trap", "divergence"]],
        on="group", how="left",
    )
    merged.attrs["latest_snap"] = latest_snap
    merged.attrs["price_asof"] = latest
    merged.attrs["val_ts"] = val_ts
    merged.attrs["n_snaps"] = len(dates)
    return merged


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.caption(
    "Where the opportunistic-buying opportunities are: industries that have "
    "**corrected**, are **cheap vs their own history**, and are **stabilising** — "
    "colour-coded by quality so overreactions separate from value traps."
)

with st.sidebar:
    st.subheader("Controls")
    level = st.radio(
        "Aggregation", ["Industry", "Sector"], index=0,
        help="Industry = simfin_industry (finer, some groups thin). Sector = GICS (11, robust top-down).",
    )
    min_names = st.slider(
        "Min names per group", 1, 20, 8,
        help="Groups below this are suppressed from the ranking (too few names to trust the median).",
    )
    trend_window = st.slider(
        "Fundamental-trend lookback (snapshots)", 1, 8, 4,
        help="Snapshots back over which the profitability trend is measured for the trap flag.",
    )

df = build_opportunity_table(level, min_names, trend_window)
qual = df[~df["thin"]].copy()
latest_snap = df.attrs["latest_snap"]
price_asof = df.attrs["price_asof"]
n_snaps = df.attrs["n_snaps"]

st.caption(
    f"Model snapshot **{latest_snap}** · price data through **{price_asof}** · "
    f"own-history percentile over **{n_snaps}** snapshots · "
    f"{len(qual)} of {len(df)} {level.lower()}s meet the ≥{min_names}-name floor."
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
opp = qual.sort_values("opportunity_score", ascending=False)
n_opp = int(((opp["opportunity_score"] >= 0.6) & (~opp["trap"])).sum())
n_trap = int(opp["trap"].sum())
deepest = opp.loc[opp["dd_med"].idxmin()] if opp["dd_med"].notna().any() else None
cheapest = opp.loc[opp["cheapness_component"].idxmax()] if opp["cheapness_component"].notna().any() else None

k = st.columns(4)
k[0].metric("Opportunities (score ≥0.6, no trap)", n_opp)
k[1].metric("Value-trap flags", n_trap)
if deepest is not None:
    k[2].metric("Deepest drawdown", deepest["group"], f"{deepest['dd_med']:.0%} from high")
if cheapest is not None:
    k[3].metric("Cheapest vs own history", cheapest["group"], f"{cheapest['cheapness_component']:.0%} pctile")

# ---------------------------------------------------------------------------
# Hero: opportunity map
# ---------------------------------------------------------------------------
st.subheader("Opportunity map")
st.caption(
    "**Corrected + cheap = top-left.** x: drawdown from 52-wk high (further left = deeper "
    "correction) · y: valuation percentile vs own history (higher = cheaper than it usually is) · "
    "bubble size: market cap · colour: quality rank (green = quality intact, red = weak). "
    "Green bubbles in the top-left are the sweet spot; red bubbles there are likely traps."
)
plot = qual.dropna(subset=["dd_med", "cheapness_component"]).copy()
if not plot.empty:
    plot["Quality"] = plot["quality_rank"].fillna(0.5)
    fig = px.scatter(
        plot, x="dd_med", y="cheapness_component",
        size="mktcap_bn", color="Quality",
        color_continuous_scale="RdYlGn", range_color=(0, 1),
        size_max=55, hover_name="group",
        labels={"dd_med": "Drawdown from 52-wk high", "cheapness_component": "Cheapness (own-history pctile)"},
        custom_data=["group", "opportunity_score", "ret_12m_med", "n_names"],
    )
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>Drawdown: %{x:.0%}<br>"
                      "Cheapness pctile: %{y:.0%}<br>12M return: %{customdata[2]:.1%}<br>"
                      "Opportunity: %{customdata[1]:.2f}<br>Names: %{customdata[3]}<extra></extra>"
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
    fig.update_layout(height=560, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                      coloraxis_colorbar_title="Quality")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Ranked opportunity table
# ---------------------------------------------------------------------------
st.subheader("Ranked opportunities")
st.caption(
    "Opportunity score = equal-weight mean of correction, cheapness (own-history) and "
    "stabilisation. ⚠️ = value-trap risk (weak quality + deteriorating fundamentals). "
    "🔀 = positive divergence (fundamentals up, price down)."
)
tbl = opp.copy()
tbl["Flags"] = (
    tbl["trap"].map({True: "⚠️", False: ""}) + tbl["divergence"].map({True: " 🔀", False: ""})
).str.strip()
# Streamlit's %% format does NOT auto-multiply fractions → scale display cols to ×100.
for col in ["correction_rank", "cheapness_component", "stabilization_rank",
            "quality_rank", "ret_12m_med", "dd_med", "ret_1m_med"]:
    tbl[col] = tbl[col] * 100
show = tbl[[
    "group", "opportunity_score", "correction_rank", "cheapness_component",
    "stabilization_rank", "quality_rank", "ret_12m_med", "dd_med", "ret_1m_med",
    "alp_med", "n_names", "Flags",
]].rename(columns={
    "group": level, "opportunity_score": "Opportunity", "correction_rank": "Correction",
    "cheapness_component": "Cheapness", "stabilization_rank": "Stabilising",
    "quality_rank": "Quality", "ret_12m_med": "12M ret", "dd_med": "Drawdown",
    "ret_1m_med": "1M ret", "alp_med": "Alpha (z)", "n_names": "Names",
})
st.dataframe(
    show, use_container_width=True, hide_index=True,
    column_config={
        "Opportunity": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
        "Correction": st.column_config.NumberColumn(format="%.0f%%", help="Cross-group rank"),
        "Cheapness": st.column_config.NumberColumn(format="%.0f%%", help="Percentile vs own history"),
        "Stabilising": st.column_config.NumberColumn(format="%.0f%%", help="Cross-group rank"),
        "Quality": st.column_config.NumberColumn(format="%.0f%%", help="Cross-group rank"),
        "12M ret": st.column_config.NumberColumn(format="%.1f%%"),
        "Drawdown": st.column_config.NumberColumn(format="%.1f%%"),
        "1M ret": st.column_config.NumberColumn(format="%.1f%%"),
        "Alpha (z)": st.column_config.NumberColumn(format="%.2f"),
    },
)

# ---------------------------------------------------------------------------
# Drill-down
# ---------------------------------------------------------------------------
st.subheader("Drill-down")
pick = st.selectbox(f"Inspect a {level.lower()}", opp["group"].tolist())
if pick:
    row = df[df["group"] == pick].iloc[0]
    c = st.columns(5)
    c[0].metric("Opportunity", f"{row['opportunity_score']:.2f}")
    c[1].metric("12M return", f"{row['ret_12m_med']:.0%}")
    c[2].metric("Drawdown", f"{row['dd_med']:.0%}")
    c[3].metric("Cheapness pctile", f"{row['cheapness_component']:.0%}")
    c[4].metric("% below 200d MA", f"{row['pct_below_200d']:.0%}")

    if row["trap"]:
        st.warning("⚠️ Value-trap risk: weak quality and deteriorating fundamentals — the "
                   "correction may be justified.")
    if row["divergence"]:
        st.info("🔀 Positive divergence: fundamentals (growth) are up while price is down.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Valuation vs own history**")
        val_ts = df.attrs["val_ts"]
        if pick in val_ts.columns:
            ser = val_ts[pick].dropna()
            vfig = go.Figure()
            vfig.add_trace(go.Scatter(x=ser.index, y=ser.values, mode="lines",
                                      name="Value model (z)", line=dict(color="#4C78A8")))
            vfig.add_trace(go.Scatter(x=[ser.index[-1]], y=[ser.iloc[-1]], mode="markers",
                                      marker=dict(color="#E45756", size=11), name="Now"))
            vfig.update_layout(height=300, showlegend=False, margin=dict(t=10, b=10),
                               yaxis_title="Value model (z) — higher = cheaper")
            st.plotly_chart(vfig, use_container_width=True)
            st.caption(f"Now at the **{row['cheapness_component']:.0%}** percentile of its own "
                       f"{int(row['own_depth'])}-snapshot history.")
    with right:
        st.markdown("**Member names**")
        comp, _ = load_universe()
        grp_col = "sector" if level == "Sector" else "industry"
        members = comp[comp[grp_col] == pick][["security_id", "ticker", "company_name"]]
        px_m = load_price_metrics(tuple(comp["security_id"].tolist()))
        mh = load_model_history()
        latest_snap2 = mh["data_date"].max()
        vsl = mh[(mh["model_id"] == VAL_ID) & (mh["data_date"] == latest_snap2)][
            ["security_id", "model_value_z"]].rename(columns={"model_value_z": "Value (z)"})
        asl = mh[(mh["model_id"] == ALP_ID) & (mh["data_date"] == latest_snap2)][
            ["security_id", "model_value_z"]].rename(columns={"model_value_z": "Alpha (z)"})
        mem = (members.merge(px_m[["security_id", "ret_12m", "dd_from_high"]], on="security_id", how="left")
                      .merge(vsl, on="security_id", how="left")
                      .merge(asl, on="security_id", how="left"))
        mem = mem.rename(columns={"ticker": "Ticker", "company_name": "Company",
                                  "ret_12m": "12M ret", "dd_from_high": "Drawdown"})
        mem["12M ret"] = mem["12M ret"] * 100
        mem["Drawdown"] = mem["Drawdown"] * 100
        mem = mem.drop(columns=["security_id"]).sort_values("Alpha (z)", ascending=False)
        st.dataframe(
            mem, use_container_width=True, hide_index=True, height=300,
            column_config={
                "12M ret": st.column_config.NumberColumn(format="%.0f%%"),
                "Drawdown": st.column_config.NumberColumn(format="%.0f%%"),
                "Value (z)": st.column_config.NumberColumn(format="%.2f"),
                "Alpha (z)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
