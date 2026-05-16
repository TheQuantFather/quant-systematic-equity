"""
9_Risk_Explorer.py — Deep-dive risk analysis.

Supports two risk models selectable from the sidebar:
  • Ledoit-Wolf (risk.db)     — sample covariance with shrinkage
  • Barra Factor Model (barra.db) — Σ = X F X' + Δ, 29-factor decomposition

Shared tabs: Market Overview, Correlation Explorer, Stock Deep Dive, Risk vs Factors
Barra-only tab: Factor Analysis (factor vols, covariance, systematic/idio split)
"""

import io
import json
import sqlite3
import zlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import get_db, inject_css
from config import (
    RISK_DB as _RISK_DB_PATH, BARRA_DB as _BARRA_DB_PATH,
    UNIVERSE_DB as _UNIV_DB_PATH, MODELS_DB as _MODELS_DB_PATH,
    FACTORS_REF, MODELS_REF,
    BARRA_GROUPS,
    BARRA_STYLE_IDS, BARRA_FUNDAMENTAL_IDS,
)

st.set_page_config(page_title="Risk Explorer", layout="wide")
inject_css()
st.title("Risk Explorer")

# String paths (this file used str() on all paths previously)
RISK_DB   = str(_RISK_DB_PATH)
BARRA_DB  = str(_BARRA_DB_PATH)
UNIV_DB   = str(_UNIV_DB_PATH)
MODELS_DB = str(_MODELS_DB_PATH)

# Factor group slices derived from config (avoids duplicating the numbers here)
_SECTOR_COLS  = BARRA_GROUPS["Sector"]
_STYLE_COLS   = BARRA_GROUPS["Style"]
_BETA_COL     = BARRA_GROUPS["Beta"]
_FUND_COLS    = BARRA_GROUPS["Fundamental"]

_GROUP_LABELS = {
    "Sector":      _SECTOR_COLS,
    "Style":       _STYLE_COLS,
    "Beta":        _BETA_COL,
    "Fundamental": _FUND_COLS,
}

_STYLE_IDS = set(BARRA_STYLE_IDS)
_FUND_IDS  = set(BARRA_FUNDAMENTAL_IDS)

def _factor_group(fid: str) -> str:
    if fid.startswith("sec_"):   return "Sector"
    if fid == "beta_60d":        return "Beta"
    if fid in _STYLE_IDS:        return "Style"
    return "Fundamental"


@st.cache_data
def load_factor_returns() -> pd.DataFrame:
    with get_db(_BARRA_DB_PATH) as conn:
        df = pd.read_sql(
            "SELECT trade_date, factor_id, factor_return FROM factor_returns ORDER BY trade_date",
            conn,
        )
    df["trade_date"]    = pd.to_datetime(df["trade_date"])
    df["factor_return"] = pd.to_numeric(df["factor_return"], errors="coerce")
    df["group"]         = df["factor_id"].map(_factor_group)
    df["factor_name"]   = df["factor_id"].map(_pretty_factor)
    return df


@st.cache_data
def _load_factor_name_map() -> dict[str, str]:
    if not FACTORS_REF.exists():
        return {}
    ref = pd.read_csv(FACTORS_REF)
    return dict(zip(ref["factor_id"], ref["factor_name"]))


@st.cache_data
def _load_model_name_map() -> dict[str, str]:
    if not MODELS_REF.exists():
        return {}
    ref = pd.read_csv(MODELS_REF)
    return dict(zip(ref["ModelID"], ref["Model"]))


def _pretty_factor(name: str, _cache: dict = {}) -> str:
    """Convert a Barra factor_id to a human-readable display name."""
    if not _cache:
        _cache.update(_load_factor_name_map())
    if name.startswith("sec_"):
        return name.replace("sec_", "").replace("_", " ").title()
    if name == "beta_60d":
        return "Beta (60d)"
    return _cache.get(name, name)


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_universe() -> pd.DataFrame:
    with get_db(UNIV_DB) as conn:
        df = pd.read_sql_query(
            "SELECT isin, ticker, company_name, gics_sector FROM companies", conn
        )
    return df


@st.cache_data
def load_model_scores(data_date: str) -> pd.DataFrame:
    with get_db(MODELS_DB) as conn:
        avail = pd.read_sql_query("SELECT DISTINCT data_date FROM models ORDER BY data_date", conn)
        if avail.empty:
            return pd.DataFrame()
        avail_dt = pd.to_datetime(avail["data_date"])
        target   = pd.to_datetime(data_date)
        closest  = avail["data_date"].iloc[(avail_dt - target).abs().argsort().iloc[0]]
        df = pd.read_sql_query(
            "SELECT security_id, model_id, model_value_z FROM models "
            "WHERE data_date = ? AND is_composite = 0",
            conn, params=(closest,),
        )
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="security_id", columns="model_id", values="model_value_z").reset_index()


def to_corr(mat: np.ndarray) -> np.ndarray:
    d_inv = 1.0 / np.sqrt(np.diag(mat).clip(1e-12))
    corr = mat * np.outer(d_inv, d_inv)
    np.fill_diagonal(corr, 1.0)
    return corr.clip(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Ledoit-Wolf loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def lw_snapshot_meta() -> pd.DataFrame:
    with get_db(RISK_DB) as conn:
        df = pd.read_sql_query(
            "SELECT data_date, n_stocks, shrinkage_coeff, lookback_days "
            "FROM covariance_matrix ORDER BY data_date",
            conn,
        )
    return df


@st.cache_data(ttl=300)
def lw_load_matrix(data_date: str) -> tuple[np.ndarray, list[str]]:
    with get_db(RISK_DB) as conn:
        row = conn.execute(
            "SELECT matrix_blob, isin_list FROM covariance_matrix WHERE data_date = ?",
            (data_date,),
        ).fetchone()
    mat = np.load(io.BytesIO(zlib.decompress(row[0])))
    return mat, json.loads(row[1])


# ---------------------------------------------------------------------------
# Barra loaders
# ---------------------------------------------------------------------------

def barra_available() -> bool:
    return _BARRA_DB_PATH.exists()


@st.cache_data(ttl=300)
def barra_snapshot_dates() -> list[str]:
    with get_db(BARRA_DB) as conn:
        rows = conn.execute(
            "SELECT snapshot_date FROM factor_covariance ORDER BY snapshot_date"
        ).fetchall()
    return [r[0] for r in rows]


@st.cache_data(ttl=300)
def barra_load_snapshot(snap_date: str) -> tuple:
    """
    Returns (F: K×K, factor_names: list[str],
             X_df: DataFrame[isin × K],
             delta: Series[isin → annualised idio var])
    """
    with get_db(BARRA_DB) as conn:
        row = conn.execute(
            "SELECT factor_names, cov_blob FROM factor_covariance WHERE snapshot_date=?",
            (snap_date,),
        ).fetchone()
        fnames = json.loads(row[0])
        K = len(fnames)
        F = np.frombuffer(zlib.decompress(row[1]), dtype=np.float32).reshape(K, K).astype(np.float64)

        x_rows = conn.execute(
            "SELECT security_id, factor_id, exposure FROM factor_exposures WHERE snapshot_date=?",
            (snap_date,),
        ).fetchall()

        d_rows = conn.execute(
            "SELECT security_id, idio_var FROM idiosyncratic_vars WHERE snapshot_date=?",
            (snap_date,),
        ).fetchall()

    # Build X DataFrame (N × K)
    x_data: dict = {}
    for sec_id, fac_id, exp in x_rows:
        x_data.setdefault(sec_id, {})[fac_id] = float(exp)
    X_df = pd.DataFrame.from_dict(x_data, orient="index").reindex(columns=fnames, fill_value=0.0).fillna(0.0)

    delta = pd.Series({r[0]: float(r[1]) for r in d_rows}, name="idio_var")
    delta = delta.reindex(X_df.index, fill_value=0.04)

    return F, fnames, X_df, delta


def barra_to_dense(X: np.ndarray, F: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Compute full N×N Σ_barra = X F X' + diag(δ) for visualisation."""
    XF = X @ F
    return XF @ X.T + np.diag(delta)


# ---------------------------------------------------------------------------
# Sidebar — model + date selector
# ---------------------------------------------------------------------------

univ = load_universe()

with st.sidebar:
    st.header("Settings")
    model_choice = st.radio(
        "Risk model",
        ["Barra Factor Model", "Ledoit-Wolf"],
        index=0 if barra_available() else 1,
        disabled=not barra_available(),
        help="Barra: Σ = XFX'+Δ (29 factors). Ledoit-Wolf: shrunk sample covariance.",
    )
    use_barra = model_choice == "Barra Factor Model"

    st.divider()

    if use_barra:
        b_dates = barra_snapshot_dates()
        sel_date = st.selectbox("Snapshot", b_dates, index=len(b_dates) - 1)
        F, factor_names, X_df, delta_s = barra_load_snapshot(sel_date)
        # Materialise full Σ for shared visualisation code
        X_arr     = X_df.values
        delta_arr = delta_s.values
        isins     = X_df.index.tolist()
        with st.spinner("Building Barra covariance…"):
            mat = barra_to_dense(X_arr, F, delta_arr)
        sys_var   = (X_arr @ F * X_arr).sum(axis=1)   # N — per-stock systematic variance
        total_var = sys_var + delta_arr
        st.caption(f"Stocks:  **{len(isins)}**")
        st.caption(f"Factors: **{len(factor_names)}**  (K)")
        st.caption(f"Model:   **Barra**")
    else:
        lw_meta  = lw_snapshot_meta()
        lw_dates = lw_meta["data_date"].tolist()
        sel_date = st.selectbox("Snapshot", lw_dates, index=len(lw_dates) - 1)
        row_meta = lw_meta[lw_meta["data_date"] == sel_date].iloc[0]
        with st.spinner("Loading covariance…"):
            mat, isins = lw_load_matrix(sel_date)
        st.caption(f"Stocks:    **{row_meta['n_stocks']}**")
        st.caption(f"Shrinkage: **{row_meta['shrinkage_coeff']:.4f}**")
        st.caption(f"Lookback:  **{int(row_meta['lookback_days'])} days**")

# Common derived quantities
vols        = np.sqrt(np.diag(mat))
corr        = to_corr(mat)
isin_to_idx = {isin: i for i, isin in enumerate(isins)}
isin_df     = (
    pd.DataFrame({"isin": isins, "vol": vols})
    .merge(univ, on="isin", how="left")
)
isin_df["vol_pct"] = isin_df["vol"] * 100

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

if use_barra:
    tab_overview, tab_corr, tab_stock, tab_factor, tab_barra = st.tabs([
        "Market Overview", "Correlation Explorer", "Stock Deep Dive",
        "Risk vs Factors", "Factor Analysis",
    ])
else:
    tab_overview, tab_corr, tab_stock, tab_factor = st.tabs([
        "Market Overview", "Correlation Explorer", "Stock Deep Dive", "Risk vs Factors",
    ])
    tab_barra = None


# ===========================================================================
# Tab 1 — Market Overview
# ===========================================================================
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    avg_offdiag = corr[np.triu_indices(len(isins), k=1)].mean()
    c1.metric("Stocks",          len(isins))
    c2.metric("Median vol",      f"{np.median(vols):.1%}")
    c3.metric("90th pct vol",    f"{np.percentile(vols, 90):.1%}")
    c4.metric("Avg correlation", f"{avg_offdiag:.3f}")
    if use_barra:
        sys_pct_med = float(np.median(sys_var / total_var.clip(1e-12)))
        c5.metric("Median systematic %", f"{sys_pct_med:.1%}")
    else:
        c5.metric("Shrinkage", f"{row_meta['shrinkage_coeff']:.4f}")

    col_left, col_right = st.columns(2)

    with col_left:
        fig = px.histogram(
            isin_df, x="vol_pct", color="gics_sector", nbins=60, opacity=0.8,
            labels={"vol_pct": "Annualised Volatility (%)", "gics_sector": "Sector"},
            title="Volatility Distribution by Sector",
        )
        fig.update_layout(barmode="overlay", legend=dict(font=dict(size=10)), height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        sector_df = isin_df.dropna(subset=["gics_sector"])
        order = sector_df.groupby("gics_sector")["vol_pct"].median().sort_values().index.tolist()
        fig2 = px.box(
            sector_df, x="gics_sector", y="vol_pct",
            category_orders={"gics_sector": order},
            labels={"vol_pct": "Annualised Volatility (%)", "gics_sector": ""},
            title="Volatility by Sector",
        )
        fig2.update_layout(xaxis_tickangle=-40, height=380)
        st.plotly_chart(fig2, use_container_width=True)

    if use_barra:
        # Systematic vs idiosyncratic breakdown
        st.subheader("Systematic vs Idiosyncratic Risk")
        sys_pct = sys_var / total_var.clip(1e-12) * 100
        idio_pct = delta_arr / total_var.clip(1e-12) * 100
        breakdown_df = pd.DataFrame({
            "isin": isins,
            "Systematic %": sys_pct,
            "Idiosyncratic %": idio_pct,
        }).merge(univ[["isin", "gics_sector"]], on="isin", how="left")

        fig_sys = px.histogram(
            breakdown_df, x="Systematic %", color="gics_sector",
            nbins=40, opacity=0.8,
            labels={"Systematic %": "Systematic variance share (%)"},
            title="Distribution of Systematic Risk Share (Barra)",
        )
        fig_sys.update_layout(barmode="overlay", height=320, legend=dict(font=dict(size=10)))
        st.plotly_chart(fig_sys, use_container_width=True)

        # Sector summary
        sec_summary = (
            breakdown_df.dropna(subset=["gics_sector"])
            .groupby("gics_sector")[["Systematic %", "Idiosyncratic %"]]
            .median()
            .round(1)
            .reset_index()
            .sort_values("Systematic %", ascending=False)
            .rename(columns={"gics_sector": "Sector",
                              "Systematic %": "Median systematic %",
                              "Idiosyncratic %": "Median idio %"})
        )
        st.dataframe(sec_summary, use_container_width=False, hide_index=True, width=500)

    else:
        # LW evolution charts
        st.subheader("Risk model evolution")
        ev_cols = st.columns(2)
        lw_meta_all = lw_snapshot_meta()
        with ev_cols[0]:
            fig_sh = px.line(lw_meta_all, x="data_date", y="shrinkage_coeff", markers=True,
                             labels={"data_date": "", "shrinkage_coeff": "Shrinkage"},
                             title="Ledoit-Wolf Shrinkage Over Time")
            fig_sh.update_layout(height=280)
            st.plotly_chart(fig_sh, use_container_width=True)
        with ev_cols[1]:
            med_vols = []
            for d in lw_dates:
                m, _ = lw_load_matrix(d)
                med_vols.append({"date": d, "vol": np.sqrt(np.diag(m)).mean() * 100})
            fig_ev = px.line(pd.DataFrame(med_vols), x="date", y="vol", markers=True,
                             labels={"date": "", "vol": "Mean vol (%)"},
                             title="Universe Mean Volatility Over Time")
            fig_ev.update_layout(height=280)
            st.plotly_chart(fig_ev, use_container_width=True)


# ===========================================================================
# Tab 2 — Correlation Explorer
# ===========================================================================
with tab_corr:
    subtab_heat, subtab_sector, subtab_pairs = st.tabs(["Heatmap", "Sector Matrix", "Top Pairs"])

    with subtab_heat:
        filter_mode = st.radio("Select stocks by", ["Top N by volatility", "Sector"], horizontal=True)
        if filter_mode == "Top N by volatility":
            top_n = st.slider("Number of stocks", 20, min(120, len(isins)), 50, step=10)
            idx = np.argsort(vols)[::-1][:top_n]
        else:
            sectors_list = sorted(isin_df["gics_sector"].dropna().unique())
            sel_sec = st.selectbox("Sector", sectors_list)
            idx = np.array([isin_to_idx[x] for x in isin_df[isin_df["gics_sector"] == sel_sec]["isin"]
                            if x in isin_to_idx])

        if len(idx) == 0:
            st.info("No stocks match the selection.")
        else:
            sub_corr = corr[np.ix_(idx, idx)]
            tickers  = isin_df.set_index("isin")["ticker"]
            labels   = [tickers.get(isins[i], isins[i]) for i in idx]
            fig_hm = go.Figure(go.Heatmap(
                z=sub_corr, x=labels, y=labels,
                colorscale="RdBu_r", zmin=-1, zmax=1,
                colorbar=dict(title="ρ"),
                hovertemplate="%{y} / %{x}<br>ρ = %{z:.3f}<extra></extra>",
            ))
            fig_hm.update_layout(
                height=640, title=f"Pairwise Correlation ({len(idx)} stocks) — {sel_date}",
                xaxis=dict(tickfont=dict(size=8)),
                yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

    with subtab_sector:
        sectors_all  = sorted(isin_df["gics_sector"].dropna().unique())
        sector_means = pd.DataFrame(index=sectors_all, columns=sectors_all, dtype=float)
        for si in sectors_all:
            idx_i = np.array([isin_to_idx[x] for x in isin_df[isin_df["gics_sector"] == si]["isin"]
                              if x in isin_to_idx])
            for sj in sectors_all:
                idx_j = np.array([isin_to_idx[x] for x in isin_df[isin_df["gics_sector"] == sj]["isin"]
                                  if x in isin_to_idx])
                if len(idx_i) and len(idx_j):
                    sub = corr[np.ix_(idx_i, idx_j)]
                    if si == sj:
                        mask = ~np.eye(len(idx_i), dtype=bool)
                        sector_means.loc[si, sj] = sub[mask].mean() if mask.any() else np.nan
                    else:
                        sector_means.loc[si, sj] = sub.mean()

        sector_means = sector_means.astype(float)
        short = [s.replace(" & ", "/").replace("Consumer ", "Cons. ").replace("Information ", "Info. ")
                  .replace("Communication Services", "Comm. Svcs").replace("Real Estate", "Real Est.")
                  for s in sectors_all]
        fig_sec = go.Figure(go.Heatmap(
            z=sector_means.values, x=short, y=short,
            colorscale="RdBu_r", zmin=-0.2, zmax=0.8,
            colorbar=dict(title="Avg ρ"),
            hovertemplate="%{y} / %{x}<br>Avg ρ = %{z:.3f}<extra></extra>",
        ))
        fig_sec.update_layout(
            height=520, xaxis_tickangle=-40,
            yaxis=dict(autorange="reversed"),
            title=f"Average Pairwise Correlation by Sector — {sel_date}",
        )
        st.plotly_chart(fig_sec, use_container_width=True)

        intra = {s: float(sector_means.loc[s, s]) for s in sectors_all}
        intra_df = pd.DataFrame(
            [{"Sector": s, "Avg intra-sector ρ": v, "Stocks": int((isin_df["gics_sector"] == s).sum())}
             for s, v in sorted(intra.items(), key=lambda x: -x[1]) if not np.isnan(v)]
        )
        intra_df["Avg intra-sector ρ"] = intra_df["Avg intra-sector ρ"].map("{:.3f}".format)
        st.dataframe(intra_df, use_container_width=False, hide_index=True, width=460)

    with subtab_pairs:
        k_pairs = st.slider("Pairs to show", 10, 100, 25)
        mode    = st.radio("Sort by", ["Highest correlation", "Lowest correlation"], horizontal=True)
        triu_idx  = np.triu_indices(len(isins), k=1)
        triu_vals = corr[triu_idx]
        order     = np.argsort(triu_vals)[::-1] if mode == "Highest correlation" else np.argsort(triu_vals)
        top_k     = order[:k_pairs]
        tick_map  = isin_df.set_index("isin")[["ticker", "company_name", "gics_sector"]].to_dict("index")
        pairs_rows = []
        for flat in top_k:
            i, j = triu_idx[0][flat], triu_idx[1][flat]
            a, b = isins[i], isins[j]
            pairs_rows.append({
                "Ticker A":  tick_map.get(a, {}).get("ticker", a),
                "Company A": tick_map.get(a, {}).get("company_name", ""),
                "Sector A":  tick_map.get(a, {}).get("gics_sector", ""),
                "Ticker B":  tick_map.get(b, {}).get("ticker", b),
                "Company B": tick_map.get(b, {}).get("company_name", ""),
                "Sector B":  tick_map.get(b, {}).get("gics_sector", ""),
                "ρ":         round(float(corr[i, j]), 4),
                "Vol A":     f"{vols[i]:.1%}",
                "Vol B":     f"{vols[j]:.1%}",
            })
        st.dataframe(pd.DataFrame(pairs_rows), use_container_width=True, hide_index=True)


# ===========================================================================
# Tab 3 — Stock Deep Dive
# ===========================================================================
with tab_stock:
    isin_opts = (
        isin_df[["isin", "ticker", "company_name"]]
        .dropna(subset=["ticker"])
        .assign(label=lambda d: d["ticker"] + " — " + d["company_name"])
        .sort_values("ticker")
    )
    sel_label  = st.selectbox("Select stock", isin_opts["label"].tolist())
    sel_isin   = isin_opts.loc[isin_opts["label"] == sel_label, "isin"].iloc[0]
    sel_ticker = sel_label.split(" — ")[0]

    if sel_isin not in isin_to_idx:
        st.warning("This stock is not in the selected snapshot's risk model.")
    else:
        i = isin_to_idx[sel_isin]
        w_eq   = np.ones(len(isins)) / len(isins)
        cov_im = mat[i, :] @ w_eq
        var_m  = w_eq @ mat @ w_eq
        beta   = cov_im / var_m

        if use_barra:
            sys_v  = float(sys_var[i])
            idio_v = float(delta_arr[i])
            tot_v  = sys_v + idio_v
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Annualised vol",    f"{vols[i]:.1%}")
            c2.metric("Beta (equal-wt)",   f"{beta:.2f}")
            c3.metric("Systematic risk",   f"{sys_v / tot_v:.1%}")
            c4.metric("Idiosyncratic risk",f"{idio_v / tot_v:.1%}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Annualised vol",  f"{vols[i]:.1%}")
            c2.metric("Beta (equal-wt)", f"{beta:.2f}")
            c3.metric("Sector",          isin_df.loc[isin_df["isin"] == sel_isin, "gics_sector"].fillna("—").iloc[0])

        # Factor exposures (Barra only)
        if use_barra and sel_isin in X_df.index:
            st.subheader("Factor Exposures")
            exp_s = X_df.loc[sel_isin]
            pretty = [_pretty_factor(n) for n in factor_names]
            group_colors = (
                ["Sector"] * 11 + ["Style"] * 5 + ["Beta"] + ["Fundamental"] * 12
            )
            exp_df = pd.DataFrame({
                "factor":    pretty,
                "exposure":  exp_s.values,
                "group":     group_colors,
            })

            fig_exp = px.bar(
                exp_df, x="exposure", y="factor", color="group",
                orientation="h",
                color_discrete_map={"Sector": "#4C78A8", "Style": "#F58518",
                                    "Beta": "#E45756", "Fundamental": "#72B7B2"},
                labels={"exposure": "Factor exposure (z-score / dummy)", "factor": ""},
                title=f"{sel_ticker} — Factor Exposures",
            )
            fig_exp.update_layout(height=700, yaxis=dict(tickfont=dict(size=9)))
            st.plotly_chart(fig_exp, use_container_width=True)

            # Risk decomposition by group
            st.subheader("Risk Decomposition by Factor Group")
            grp_rows = []
            for g_name, g_slice in _GROUP_LABELS.items():
                if isinstance(g_slice, int):
                    xi = exp_s.values[[g_slice]]
                    Fi = F[np.ix_([g_slice], [g_slice])]
                else:
                    xi = exp_s.values[g_slice]
                    Fi = F[g_slice, g_slice]
                g_var = float(xi @ Fi @ xi)
                grp_rows.append({"Group": g_name,
                                  "Variance": g_var,
                                  "Ann. Vol (%)": float(np.sqrt(max(g_var, 0))) * 100})
            grp_rows.append({"Group": "Idiosyncratic",
                              "Variance": float(delta_arr[i]),
                              "Ann. Vol (%)": float(np.sqrt(delta_arr[i])) * 100})
            total_approx = sum(r["Variance"] for r in grp_rows)
            for r in grp_rows:
                r["Share (%)"] = round(r["Variance"] / max(total_approx, 1e-12) * 100, 1)
                r["Ann. Vol (%)"] = round(r["Ann. Vol (%)"], 2)
                r["Variance"] = f"{r['Variance']:.5f}"

            st.dataframe(pd.DataFrame(grp_rows), use_container_width=False,
                         hide_index=True, width=500)

        # Correlation time-series for LW
        if not use_barra:
            vol_ts = []
            for d in lw_dates:
                m_d, isins_d = lw_load_matrix(d)
                if sel_isin in isins_d:
                    idx_d = isins_d.index(sel_isin)
                    vol_ts.append({"date": d, "vol": np.sqrt(m_d[idx_d, idx_d]) * 100})
            if vol_ts:
                fig_vt = px.line(pd.DataFrame(vol_ts), x="date", y="vol", markers=True,
                                 labels={"date": "", "vol": "Annualised vol (%)"},
                                 title=f"{sel_ticker} — Volatility Over Time (Ledoit-Wolf)")
                fig_vt.update_layout(height=300)
                st.plotly_chart(fig_vt, use_container_width=True)

        # Top / bottom correlates
        st.subheader(f"Most correlated peers — {sel_ticker}")
        n_peers  = st.slider("Peers", 10, 50, 20)
        row_corr = corr[i].copy()
        row_corr[i] = -999
        top_idx = np.argsort(row_corr)[::-1][:n_peers]
        bot_idx = np.argsort(row_corr)[:n_peers]

        tick_map = isin_df.set_index("isin")[["ticker", "company_name", "gics_sector"]].to_dict("index")

        def peer_table(idx_list):
            return pd.DataFrame([{
                "Ticker":  tick_map.get(isins[j], {}).get("ticker", isins[j]),
                "Company": tick_map.get(isins[j], {}).get("company_name", ""),
                "Sector":  tick_map.get(isins[j], {}).get("gics_sector", ""),
                "ρ":       round(float(corr[i, j]), 4),
                "Vol":     f"{vols[j]:.1%}",
            } for j in idx_list])

        col_hi, col_lo = st.columns(2)
        with col_hi:
            st.caption("Highest correlated")
            st.dataframe(peer_table(top_idx), use_container_width=True, hide_index=True)
        with col_lo:
            st.caption("Lowest correlated (diversifiers)")
            st.dataframe(peer_table(bot_idx), use_container_width=True, hide_index=True)


# ===========================================================================
# Tab 4 — Risk vs Factors (model scores)
# ===========================================================================
with tab_factor:
    scores = load_model_scores(sel_date)
    if scores.empty:
        st.info("No model scores found for this snapshot date.")
    else:
        risk_df = isin_df[["isin", "ticker", "company_name", "gics_sector", "vol_pct"]].copy()
        risk_df = risk_df.merge(scores.rename(columns={"security_id": "isin"}), on="isin", how="inner")
        raw_model_cols = [c for c in risk_df.columns if c not in
                          {"isin", "ticker", "company_name", "gics_sector", "vol_pct"}]

        if not raw_model_cols:
            st.info("No overlapping ISINs between risk model and factor scores.")
        else:
            # Rename model ID columns to readable names for display
            _model_name_map = _load_model_name_map()
            model_rename  = {c: _model_name_map.get(c, c) for c in raw_model_cols}
            risk_df       = risk_df.rename(columns=model_rename)
            model_cols    = [model_rename[c] for c in raw_model_cols]

            col_left, col_right = st.columns([1, 3])
            with col_left:
                sel_model = st.selectbox("Factor model", model_cols)
                color_by  = st.checkbox("Colour by sector", value=True)
            with col_right:
                fig_sc = px.scatter(
                    risk_df.dropna(subset=[sel_model, "vol_pct"]),
                    x=sel_model, y="vol_pct",
                    color="gics_sector" if color_by else None,
                    hover_data={"ticker": True, "company_name": True, "gics_sector": True,
                                "vol_pct": ":.1f", sel_model: ":.2f"},
                    labels={sel_model: f"{sel_model} z-score", "vol_pct": "Annualised vol (%)"},
                    title=f"Volatility vs {sel_model} — {sel_date}",
                    opacity=0.65,
                )
                x_vals = risk_df[sel_model].dropna()
                y_vals = risk_df.loc[x_vals.index, "vol_pct"]
                m_coef = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                fig_sc.add_trace(go.Scatter(
                    x=x_line, y=np.polyval(m_coef, x_line), mode="lines",
                    line=dict(color="black", width=1.5, dash="dash"),
                    name="OLS trend", showlegend=True,
                ))
                fig_sc.update_layout(height=480)
                st.plotly_chart(fig_sc, use_container_width=True)

            st.subheader(f"Average volatility by {sel_model} quintile")
            plot_df = risk_df.dropna(subset=[sel_model, "vol_pct"]).copy()
            plot_df["quintile"] = pd.qcut(plot_df[sel_model], 5,
                                          labels=["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"])
            q_summary = plot_df.groupby("quintile", observed=True)["vol_pct"].agg(
                mean="mean", median="median", count="count"
            ).reset_index()
            fig_q = px.bar(q_summary, x="quintile", y="mean", text_auto=".1f",
                           labels={"quintile": f"{sel_model} quintile", "mean": "Mean vol (%)"},
                           title=f"Mean vol by {sel_model} quintile")
            fig_q.update_layout(height=320)
            st.plotly_chart(fig_q, use_container_width=True)


# ===========================================================================
# Tab 5 — Barra Factor Analysis (Barra model only)
# ===========================================================================
if tab_barra is not None:
    with tab_barra:
        K = len(factor_names)
        pretty_names = [_pretty_factor(n) for n in factor_names]
        group_labels = ["Sector"] * 11 + ["Style"] * 5 + ["Beta"] + ["Fundamental"] * 12

        # ── Factor volatilities ──────────────────────────────────────────────
        st.subheader("Factor Annual Volatilities")
        fvols = np.sqrt(np.diag(F)) * 100
        fvol_df = pd.DataFrame({
            "factor":  pretty_names,
            "vol_pct": fvols,
            "group":   group_labels,
        }).sort_values("vol_pct", ascending=True)

        fig_fv = px.bar(
            fvol_df, x="vol_pct", y="factor", color="group",
            orientation="h",
            color_discrete_map={"Sector": "#4C78A8", "Style": "#F58518",
                                 "Beta": "#E45756", "Fundamental": "#72B7B2"},
            labels={"vol_pct": "Annualised factor vol (%)", "factor": ""},
            title=f"Factor Annual Volatilities — {sel_date}",
        )
        fig_fv.update_layout(height=700, yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig_fv, use_container_width=True)

        # ── Factor covariance heatmap ────────────────────────────────────────
        st.subheader("Factor Correlation Matrix")
        fvol_vec = np.sqrt(np.diag(F)).clip(1e-12)
        F_corr = F / np.outer(fvol_vec, fvol_vec)
        np.fill_diagonal(F_corr, 1.0)
        F_corr = F_corr.clip(-1.0, 1.0)

        fig_fc = go.Figure(go.Heatmap(
            z=F_corr, x=pretty_names, y=pretty_names,
            colorscale="RdBu_r", zmin=-1, zmax=1,
            colorbar=dict(title="ρ"),
            hovertemplate="%{y} / %{x}<br>ρ = %{z:.3f}<extra></extra>",
        ))
        fig_fc.update_layout(
            height=700,
            xaxis=dict(tickfont=dict(size=8), tickangle=-45),
            yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
            title=f"Factor Correlation Matrix (K={K}) — {sel_date}",
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # ── Systematic risk share by sector ────────────────────────────────
        st.subheader("Systematic Risk Share by Sector")
        sys_share_df = pd.DataFrame({
            "isin":        isins,
            "sys_pct":     sys_var / total_var.clip(1e-12) * 100,
            "idio_pct":    delta_arr / total_var.clip(1e-12) * 100,
            "total_vol":   vols * 100,
        }).merge(univ[["isin", "gics_sector", "ticker"]], on="isin", how="left")

        sec_risk = (
            sys_share_df.dropna(subset=["gics_sector"])
            .groupby("gics_sector")
            .agg(
                median_sys_pct=("sys_pct",    "median"),
                median_idio_pct=("idio_pct",  "median"),
                median_vol=("total_vol",       "median"),
                n_stocks=("isin",              "count"),
            )
            .reset_index()
            .sort_values("median_sys_pct", ascending=False)
        )
        fig_sec_sys = px.bar(
            sec_risk, x="gics_sector", y=["median_sys_pct", "median_idio_pct"],
            barmode="stack",
            labels={"value": "Risk share (%)", "gics_sector": "",
                    "variable": "Component"},
            color_discrete_map={"median_sys_pct": "#4C78A8", "median_idio_pct": "#F28E2B"},
            title="Median Systematic vs Idiosyncratic Risk Share by Sector",
        )
        fig_sec_sys.for_each_trace(lambda t: t.update(
            name="Systematic" if "sys" in t.name else "Idiosyncratic"
        ))
        fig_sec_sys.update_layout(height=360, xaxis_tickangle=-35)
        st.plotly_chart(fig_sec_sys, use_container_width=True)

        # ── Factor group contribution to market portfolio ────────────────────
        st.subheader("Factor Group Contribution to Equal-Weight Portfolio Risk")
        x_mean = X_arr.mean(axis=0)   # (K,) mean exposure of equal-weight portfolio
        grp_rows = []
        for g_name, g_slice in _GROUP_LABELS.items():
            if isinstance(g_slice, int):
                xi = x_mean[[g_slice]]
                Fi = F[np.ix_([g_slice], [g_slice])]
            else:
                xi = x_mean[g_slice]
                Fi = F[g_slice, g_slice]
            g_var = float(xi @ Fi @ xi)
            grp_rows.append({"Group": g_name, "Variance": g_var})
        idio_ew = float(delta_arr.mean()) / len(isins)
        grp_rows.append({"Group": "Idiosyncratic", "Variance": idio_ew})
        total_g = sum(r["Variance"] for r in grp_rows)
        for r in grp_rows:
            r["Share (%)"] = round(r["Variance"] / max(total_g, 1e-12) * 100, 1)
            r["Ann. Vol (%)"] = round(np.sqrt(max(r["Variance"] / 252, 0)) * 100 * np.sqrt(252), 2)

        grp_df = pd.DataFrame(grp_rows).sort_values("Share (%)", ascending=False)
        col_pie, col_tbl = st.columns([1, 1])
        with col_pie:
            fig_pie = px.pie(
                grp_df, values="Share (%)", names="Group",
                title="Equal-Weight Portfolio Risk Decomposition",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pie.update_layout(height=360)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_tbl:
            st.dataframe(
                grp_df[["Group", "Share (%)", "Ann. Vol (%)"]],
                use_container_width=True, hide_index=True,
            )

        # ── Factor returns ────────────────────────────────────────────────────
        st.divider()
        st.subheader("Factor Returns")

        fr = load_factor_returns()
        if fr.empty:
            st.info("No factor return data in barra.db.")
        else:
            GROUP_COLORS = {
                "Sector":      "#4C78A8",
                "Style":       "#F58518",
                "Beta":        "#E45756",
                "Fundamental": "#72B7B2",
            }

            # Sector returns absorb the daily market component (no intercept in WLS).
            # Demean sector returns cross-sectionally each day so they show
            # relative sector performance (vs equal-weight sector average).
            sec_mean = (
                fr[fr["group"] == "Sector"]
                .groupby("trade_date")["factor_return"]
                .mean()
                .rename("sec_mean")
            )
            fr = fr.join(sec_mean, on="trade_date")
            fr["factor_return_adj"] = np.where(
                fr["group"] == "Sector",
                fr["factor_return"] - fr["sec_mean"],
                fr["factor_return"],
            )
            fr = fr.drop(columns="sec_mean")

            fr_grp = st.segmented_control(
                "Show groups", ["All", "Sector", "Style", "Beta", "Fundamental"],
                default="All", key="fr_grp",
            )
            fr_filt = fr if fr_grp == "All" else fr[fr["group"] == fr_grp]

            wide = fr_filt.pivot_table(
                index="trade_date", columns="factor_id", values="factor_return_adj"
            ).fillna(0)

            id_to_group = fr[["factor_id", "group"]].drop_duplicates().set_index("factor_id")["group"]

            chart_mode = st.segmented_control(
                "Chart view", ["Rolling 1Y (ann.)", "Cumulative"],
                default="Rolling 1Y (ann.)", key="fr_chart_mode",
            )

            fig_fr = go.Figure()
            if chart_mode == "Cumulative":
                plot_data = ((1 + wide).cumprod() - 1) * 100
                y_title   = "Cumulative return (%)"
            else:
                # Rolling 252-day annualised return: (prod of daily 1+r)^(252/252) - 1
                plot_data = (wide.add(1).rolling(252).apply(np.prod, raw=True) - 1) * 100
                plot_data = plot_data.dropna(how="all")
                y_title   = "Rolling 1Y annualised return (%)"

            for fid in plot_data.columns:
                grp   = id_to_group.get(fid, "Fundamental")
                color = GROUP_COLORS.get(grp, "#94A3B8")
                fig_fr.add_trace(go.Scatter(
                    x=plot_data.index, y=plot_data[fid],
                    name=_pretty_factor(fid),
                    line=dict(color=color, width=1.5),
                    hovertemplate=f"{_pretty_factor(fid)}: %{{y:.1f}}%<extra></extra>",
                ))
            fig_fr.add_hline(y=0, line_dash="dot", line_color="#64748B", line_width=1)
            fig_fr.update_layout(
                height=420, yaxis_title=y_title,
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
                margin=dict(l=0, r=0, t=10, b=10),
            )
            st.plotly_chart(fig_fr, use_container_width=True)
            st.caption(
                "Sector returns are demeaned daily vs the equal-weight sector average "
                "to show relative sector spread. Style, Beta, and Fundamental are raw WLS returns."
            )

            # Summary stats table
            st.subheader("Factor Return Summary")
            rows   = []
            for fid, grp_df2 in fr.groupby("factor_id"):
                s = grp_df2.set_index("trade_date")["factor_return_adj"].sort_index()
                ann_ret   = s.mean() * 252 * 100
                ann_vol   = s.std() * np.sqrt(252) * 100
                sharpe    = (ann_ret / ann_vol) if ann_vol > 0 else np.nan
                t = s.index[-1]
                ret_1m    = (s.loc[t - pd.Timedelta(days=21):].add(1).prod() - 1) * 100
                ret_3m    = (s.loc[t - pd.Timedelta(days=63):].add(1).prod() - 1) * 100
                ret_6m    = (s.loc[t - pd.Timedelta(days=126):].add(1).prod() - 1) * 100
                rows.append({
                    "Factor":          _pretty_factor(fid),
                    "Group":           _factor_group(fid),
                    "Ann. Return (%)": round(ann_ret, 2),
                    "Ann. Vol (%)":    round(ann_vol, 2),
                    "Sharpe":          round(sharpe, 2) if pd.notna(sharpe) else None,
                    "1M (%)":          round(ret_1m, 2),
                    "3M (%)":          round(ret_3m, 2),
                    "6M (%)":          round(ret_6m, 2),
                })
            summary_df = (
                pd.DataFrame(rows)
                .sort_values(["Group", "Ann. Return (%)"], ascending=[True, False])
                .reset_index(drop=True)
            )
            st.dataframe(
                summary_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Ann. Return (%)": st.column_config.NumberColumn(format="%.2f"),
                    "Ann. Vol (%)":    st.column_config.NumberColumn(format="%.2f"),
                    "Sharpe":          st.column_config.NumberColumn(format="%.2f"),
                    "1M (%)":          st.column_config.NumberColumn(format="%.2f"),
                    "3M (%)":          st.column_config.NumberColumn(format="%.2f"),
                    "6M (%)":          st.column_config.NumberColumn(format="%.2f"),
                },
            )
