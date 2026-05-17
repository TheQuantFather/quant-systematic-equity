"""
10_Data_Quality.py — Pipeline data validation and quality checks.
"""

import sys
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    UNIVERSE_DB, RETURNS_DB, FACTORS_DB, MODELS_DB,
    CONSTITUENTS_DB, RISK_DB, BARRA_DB,
    FACTORS_REF, MODELS_REF,
)
from utils import get_db, inject_css

st.set_page_config(page_title="Data Quality", layout="wide")
inject_css()
st.title("Data Quality & Pipeline Health")
st.caption("Validation checks across all pipeline databases. All queries run live against the local DBs.")

# ---------------------------------------------------------------------------
# Cached data-loading helpers (page-local, no st.cache_data on lambdas)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _db_meta() -> pd.DataFrame:
    """File size and modification time for each database."""
    dbs = {
        "universe":     UNIVERSE_DB,
        "constituents": CONSTITUENTS_DB,
        "returns":      RETURNS_DB,
        "factors":      FACTORS_DB,
        "models":       MODELS_DB,
        "risk":         RISK_DB,
        "barra":        BARRA_DB,
    }
    rows = []
    for name, path in dbs.items():
        p = Path(path)
        if p.exists():
            stat = p.stat()
            rows.append({
                "DB": name,
                "Size (MB)": round(stat.st_size / 1_048_576, 1),
                "Last Modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
        else:
            rows.append({"DB": name, "Size (MB)": None, "Last Modified": "missing"})
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def _snapshot_coverage() -> pd.DataFrame:
    """Security counts per snapshot date across factors, models, barra."""
    with get_db(FACTORS_DB) as conn:
        f = pd.read_sql(
            "SELECT data_date, COUNT(DISTINCT security_id) AS factors "
            "FROM factors GROUP BY data_date ORDER BY data_date",
            conn,
        )
    with get_db(MODELS_DB) as conn:
        m = pd.read_sql(
            "SELECT data_date, COUNT(DISTINCT security_id) AS models "
            "FROM models GROUP BY data_date ORDER BY data_date",
            conn,
        )
    with get_db(BARRA_DB) as conn:
        b = pd.read_sql(
            "SELECT snapshot_date AS data_date, COUNT(DISTINCT security_id) AS barra "
            "FROM factor_exposures GROUP BY snapshot_date ORDER BY snapshot_date",
            conn,
        )
    cov = f.merge(m, on="data_date", how="outer").merge(b, on="data_date", how="outer")
    cov = cov.sort_values("data_date").fillna(0).astype({"factors": int, "models": int, "barra": int})
    return cov


@st.cache_data(ttl=300)
def _factor_fill(snapshot_date: str) -> pd.DataFrame:
    ref = pd.read_csv(FACTORS_REF)[["factor_id", "factor_name", "category", "direction"]]
    with get_db(FACTORS_DB) as conn:
        raw = pd.read_sql(
            "SELECT factor_id, "
            "  COUNT(*) AS total, "
            "  COUNT(factor_value) AS filled, "
            "  AVG(factor_value_z) AS avg_z, "
            "  MIN(factor_value_z) AS min_z, "
            "  MAX(factor_value_z) AS max_z "
            "FROM factors WHERE data_date = ? GROUP BY factor_id",
            conn,
            params=(snapshot_date,),
        )
    df = raw.merge(ref, on="factor_id", how="left")
    df["filled"] = pd.to_numeric(df["filled"], errors="coerce")
    df["total"]  = pd.to_numeric(df["total"],  errors="coerce")
    df["fill_pct"] = (df["filled"] / df["total"] * 100).round(1)
    return df.sort_values("category")


@st.cache_data(ttl=300)
def _factor_zscore_dist(snapshot_date: str) -> pd.DataFrame:
    ref = pd.read_csv(FACTORS_REF)[["factor_id", "factor_name", "category"]]
    with get_db(FACTORS_DB) as conn:
        df = pd.read_sql(
            "SELECT factor_id, factor_value_z "
            "FROM factors WHERE data_date = ? AND factor_value_z IS NOT NULL",
            conn,
            params=(snapshot_date,),
        )
    return df.merge(ref, on="factor_id", how="left")


@st.cache_data(ttl=300)
def _model_null_rates() -> pd.DataFrame:
    """Null rate per model per date."""
    ref = pd.read_csv(MODELS_REF)[["ModelID", "Model"]].drop_duplicates()
    with get_db(MODELS_DB) as conn:
        df = pd.read_sql(
            "SELECT data_date, model_id, "
            "  COUNT(*) AS total, "
            "  SUM(CASE WHEN model_value_z IS NULL THEN 1 ELSE 0 END) AS nulls "
            "FROM models GROUP BY data_date, model_id ORDER BY data_date",
            conn,
        )
    df["null_pct"] = (df["nulls"] / df["total"] * 100).round(2)
    df = df.merge(ref.rename(columns={"ModelID": "model_id"}), on="model_id", how="left")
    return df


@st.cache_data(ttl=300)
def _model_score_dist(snapshot_date: str) -> pd.DataFrame:
    ref = pd.read_csv(MODELS_REF)[["ModelID", "Model"]].drop_duplicates()
    with get_db(MODELS_DB) as conn:
        df = pd.read_sql(
            "SELECT model_id, security_id, model_value_z "
            "FROM models WHERE data_date = ? AND model_value_z IS NOT NULL",
            conn,
            params=(snapshot_date,),
        )
    return df.merge(ref.rename(columns={"ModelID": "model_id"}), on="model_id", how="left")


@st.cache_data(ttl=300)
def _model_coverage_by_date() -> pd.DataFrame:
    ref = pd.read_csv(MODELS_REF)[["ModelID", "Model"]].drop_duplicates()
    with get_db(MODELS_DB) as conn:
        df = pd.read_sql(
            "SELECT data_date, model_id, COUNT(DISTINCT security_id) AS n_scored "
            "FROM models WHERE model_value_z IS NOT NULL "
            "GROUP BY data_date, model_id ORDER BY data_date",
            conn,
        )
    return df.merge(ref.rename(columns={"ModelID": "model_id"}), on="model_id", how="left")


@st.cache_data(ttl=300)
def _barra_snapshot_stats() -> pd.DataFrame:
    """Per-snapshot: n_stocks, avg/max idio_var from barra.db factor snapshots."""
    with get_db(BARRA_DB) as conn:
        exp = pd.read_sql(
            "SELECT snapshot_date, COUNT(DISTINCT security_id) AS n_stocks "
            "FROM factor_exposures GROUP BY snapshot_date ORDER BY snapshot_date",
            conn,
        )
        idio = pd.read_sql(
            "SELECT snapshot_date, AVG(idio_var) AS avg_idio, MAX(idio_var) AS max_idio "
            "FROM idiosyncratic_vars GROUP BY snapshot_date ORDER BY snapshot_date",
            conn,
        )
        fr = pd.read_sql(
            "SELECT MIN(trade_date) AS min_date, MAX(trade_date) AS max_date, COUNT(*) AS n_rows "
            "FROM factor_returns",
            conn,
        ).iloc[0]
    df = exp.merge(idio, on="snapshot_date", how="left")
    df["avg_idio"] = df["avg_idio"].round(5)
    df["max_idio"] = df["max_idio"].round(5)
    return df, fr


@st.cache_data(ttl=300)
def _returns_coverage() -> pd.DataFrame:
    """Last price date and days of history per ISIN."""
    with get_db(RETURNS_DB) as conn:
        df = pd.read_sql(
            "SELECT isin, MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS n_days "
            "FROM returns GROUP BY isin",
            conn,
        )
    df["first_date"] = pd.to_datetime(df["first_date"])
    df["last_date"]  = pd.to_datetime(df["last_date"])
    today = pd.Timestamp.today().normalize()
    df["days_stale"] = (today - df["last_date"]).dt.days
    return df


@st.cache_data(ttl=300)
def _constituent_coverage() -> pd.DataFrame:
    """Per-company: distinct fiscal years of data, first/last year, latest publish date."""
    with get_db(CONSTITUENTS_DB) as conn:
        df = pd.read_sql(
            "SELECT security_id, "
            "  COUNT(DISTINCT fiscal_year) AS n_years, "
            "  MIN(fiscal_year) AS first_fy, "
            "  MAX(fiscal_year) AS last_fy, "
            "  MAX(publish_date) AS latest_publish "
            "FROM constituents GROUP BY security_id",
            conn,
        )
    return df


@st.cache_data(ttl=300)
def _recent_filings(n_days: int = 90) -> pd.DataFrame:
    """Filings published in the last N days (by publish_date)."""
    cutoff = (pd.Timestamp.today() - pd.Timedelta(days=n_days)).strftime("%Y-%m-%d")
    with get_db(CONSTITUENTS_DB) as conn:
        df = pd.read_sql(
            "SELECT security_id, fiscal_year, fiscal_period, statement_type, publish_date "
            "FROM constituents WHERE publish_date >= ? "
            "GROUP BY security_id, fiscal_year, fiscal_period "
            "ORDER BY publish_date DESC LIMIT 500",
            conn,
            params=(cutoff,),
        )
    return df


@st.cache_data(ttl=300)
def _universe_summary() -> dict:
    with get_db(UNIVERSE_DB) as conn:
        total = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
        with_ticker = conn.execute("SELECT COUNT(*) FROM companies WHERE ticker IS NOT NULL").fetchone()[0]
        with_isin   = conn.execute("SELECT COUNT(*) FROM companies WHERE isin IS NOT NULL").fetchone()[0]
        snap_dates  = conn.execute(
            "SELECT COUNT(DISTINCT snapshot_date) FROM universe_snapshots WHERE index_name='russell_1000'"
        ).fetchone()[0]
        latest_snap = conn.execute(
            "SELECT MAX(snapshot_date) FROM universe_snapshots WHERE index_name='russell_1000'"
        ).fetchone()[0]
    return {
        "total": total,
        "with_ticker": with_ticker,
        "with_isin": with_isin,
        "snap_dates": snap_dates,
        "latest_snap": latest_snap,
    }


@st.cache_data(ttl=300)
def _risk_summary() -> pd.DataFrame:
    if not RISK_DB.exists():
        return pd.DataFrame()
    with get_db(RISK_DB) as conn:
        return pd.read_sql(
            "SELECT data_date, n_stocks, shrinkage_coeff, computation_date "
            "FROM covariance_matrix ORDER BY data_date",
            conn,
        )


# ---------------------------------------------------------------------------
# TOP-LEVEL KPIs
# ---------------------------------------------------------------------------

with st.spinner("Loading pipeline status…"):
    db_meta       = _db_meta()
    snap_cov      = _snapshot_coverage()
    univ_summary  = _universe_summary()

latest_factor_date = snap_cov["data_date"].max() if not snap_cov.empty else "N/A"
latest_factor_n    = int(snap_cov.loc[snap_cov["data_date"] == latest_factor_date, "factors"].iloc[0]) if not snap_cov.empty else 0
n_snap_dates       = len(snap_cov)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Universe companies",  f"{univ_summary['total']:,}")
c2.metric("Snapshot dates",      f"{n_snap_dates}")
c3.metric("Latest snapshot",     latest_factor_date)
c4.metric("Companies (latest)",  f"{latest_factor_n:,}")
c5.metric("Russell 1000 snaps",  f"{univ_summary['snap_dates']}")

st.divider()

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "🗄️ Pipeline Health",
    "📅 Snapshot Coverage",
    "🔢 Factor Quality",
    "📊 Model Quality",
    "⚡ Barra Quality",
    "📈 Return Coverage",
    "📋 Constituents",
])
tab_health, tab_snap, tab_factor, tab_model, tab_barra, tab_ret, tab_const = tabs


# ============================================================
# TAB 1: Pipeline Health
# ============================================================
with tab_health:
    st.subheader("Database file health")

    # DB size / modification table
    styled = db_meta.copy()
    existing = styled[styled["Last Modified"] != "missing"]
    missing  = styled[styled["Last Modified"] == "missing"]

    col_tbl, col_sync = st.columns([2, 1])

    with col_tbl:
        st.dataframe(styled, use_container_width=True, hide_index=True)

    with col_sync:
        st.markdown("**Sync status**")
        if not snap_cov.empty:
            # Use the latest date where barra has coverage — barra can't be built
            # beyond returns.db's last price date, so the very latest factor snapshot
            # may legitimately have barra=0.
            synced = snap_cov[snap_cov["barra"] > 0]
            latest_synced = synced.iloc[-1] if not synced.empty else snap_cov.iloc[-1]
            latest_all    = snap_cov.iloc[-1]

            f_n = latest_synced["factors"]
            m_n = latest_synced["models"]
            b_n = latest_synced["barra"]

            def _check(label, val, ref):
                diff = abs(val - ref)
                if val == 0:
                    st.error(f"❌ {label}: 0 (missing)")
                elif diff <= 2:
                    st.success(f"✅ {label}: {val:,}")
                else:
                    st.warning(f"⚠️ {label}: {val:,} (Δ{diff} vs factors)")

            st.caption(f"At latest Barra-synced snapshot **{latest_synced['data_date']}**")
            _check("Factors", f_n, f_n)
            _check("Models",  m_n, f_n)
            _check("Barra",   b_n, f_n)

            # Note if there are newer factor/model dates without barra
            extra = snap_cov[
                (snap_cov["data_date"] > latest_synced["data_date"]) &
                (snap_cov["factors"] > 0)
            ]
            if not extra.empty:
                dates_str = ", ".join(extra["data_date"].tolist())
                st.caption(
                    f"ℹ️ {dates_str}: factors/models exist but no Barra "
                    f"(price history ends {latest_synced['data_date']})"
                )
        else:
            st.info("No snapshot data found.")

    st.divider()
    st.subheader("Row counts per table")

    row_counts = []
    db_map = {
        "universe":     (UNIVERSE_DB,      ["companies", "universe_snapshots", "isin_patch", "ticker_alias", "index_registry", "nport_accessions"]),
        "constituents": (CONSTITUENTS_DB,  ["constituents"]),
        "returns":      (RETURNS_DB,       ["returns", "svr_daily"]),
        "factors":      (FACTORS_DB,       ["factors"]),
        "models":       (MODELS_DB,        ["models"]),
        "risk":         (RISK_DB,          ["covariance_matrix"]),
        "barra":        (BARRA_DB,         ["factor_returns", "factor_covariance", "idiosyncratic_vars", "factor_exposures"]),
    }

    for db_name, (db_path, tables) in db_map.items():
        if not Path(db_path).exists():
            row_counts.append({"Database": db_name, "Table": "(missing)", "Rows": None})
            continue
        with get_db(db_path) as conn:
            existing_tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        for tbl in tables:
            if tbl not in existing_tables:
                row_counts.append({"Database": db_name, "Table": tbl, "Rows": None})
                continue
            with get_db(db_path) as conn:
                n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            row_counts.append({"Database": db_name, "Table": tbl, "Rows": n})

    rc_df = pd.DataFrame(row_counts)
    rc_df["Rows"] = rc_df["Rows"].apply(lambda x: f"{x:,}" if pd.notna(x) else "—")

    fig_rc = px.bar(
        rc_df[rc_df["Rows"] != "—"].assign(n=lambda d: d["Rows"].str.replace(",", "").astype(float)),
        x="n", y="Table", orientation="h", color="Database",
        labels={"n": "Rows", "Table": ""},
        text="Rows",
    )
    fig_rc.update_traces(textposition="outside")
    fig_rc.update_layout(height=420, margin=dict(l=0, r=60, t=20, b=20), showlegend=True)
    st.plotly_chart(fig_rc, use_container_width=True)

    with st.expander("Raw row counts"):
        st.dataframe(rc_df, hide_index=True, use_container_width=True)


# ============================================================
# TAB 2: Snapshot Coverage
# ============================================================
with tab_snap:
    st.subheader("Snapshot coverage")

    # Barra is expected to be 0 for dates beyond the last price date —
    # mark those as "Expected" rather than broken.
    last_barra_date = snap_cov.loc[snap_cov["barra"] > 0, "data_date"].max() if (snap_cov["barra"] > 0).any() else ""

    def _status(row) -> str:
        if row["factors"] == 0:
            return "❌ Missing"
        if row["models"] != row["factors"]:
            return "⚠️ Models gap"
        if row["barra"] == 0:
            return "ℹ️ No Barra (expected)" if row["data_date"] > last_barra_date else "⚠️ No Barra"
        if abs(row["barra"] - row["factors"]) > 2:
            return "⚠️ Barra gap"
        return "✅ Synced"

    cov_display = snap_cov.copy()
    cov_display["Status"] = cov_display.apply(_status, axis=1)

    # ── Top KPIs ──────────────────────────────────────────────
    synced_count = (cov_display["Status"] == "✅ Synced").sum()
    issues_count = cov_display["Status"].str.startswith("⚠️").sum()
    latest_synced_n = int(snap_cov.loc[snap_cov["data_date"] == last_barra_date, "factors"].iloc[0]) if last_barra_date else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total snapshot dates", len(cov_display))
    k2.metric("Fully synced dates",   synced_count)
    k3.metric("Dates with issues",    issues_count)
    k4.metric("Latest synced companies", f"{latest_synced_n:,}")

    st.divider()

    # ── Universe growth chart (single clear line) ─────────────
    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(
        x=cov_display["data_date"],
        y=cov_display["factors"],
        mode="lines+markers",
        name="Companies with factor data",
        line=dict(color="#4C8BF5", width=2),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>%{y} companies<extra></extra>",
    ))
    fig_growth.update_layout(
        height=300,
        xaxis_title=None,
        yaxis_title="Companies with factor data",
        margin=dict(l=0, r=0, t=10, b=40),
        showlegend=False,
        xaxis=dict(tickangle=-40),
        hovermode="x unified",
    )
    st.plotly_chart(fig_growth, use_container_width=True)

    # ── Coverage table ────────────────────────────────────────
    tbl = cov_display[["data_date", "factors", "models", "barra", "Status"]].copy()
    tbl.columns = ["Date", "Factors", "Models", "Barra", "Status"]
    tbl = tbl.sort_values("Date", ascending=False).reset_index(drop=True)
    st.dataframe(tbl, hide_index=True, use_container_width=True, height=550)


# ============================================================
# TAB 3: Factor Quality
# ============================================================
with tab_factor:
    st.subheader("Factor fill rates & z-score distributions")

    factor_dates = snap_cov["data_date"].tolist()
    sel_date_f = st.selectbox("Snapshot date", factor_dates, index=len(factor_dates)-1, key="fq_date")

    with st.spinner("Loading factor data…"):
        fill_df  = _factor_fill(sel_date_f)
        zscore_df = _factor_zscore_dist(sel_date_f)

    # Fill rate bar chart
    st.markdown(f"**Fill rate by factor** — {sel_date_f}")
    fig_fill = px.bar(
        fill_df.sort_values("fill_pct", ascending=True),
        x="fill_pct", y="factor_name", orientation="h",
        color="category",
        labels={"fill_pct": "Fill rate (%)", "factor_name": ""},
        text="filled",
        hover_data={"total": True, "filled": True, "fill_pct": True},
    )
    fig_fill.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig_fill.update_layout(
        height=max(400, len(fill_df) * 20),
        margin=dict(l=0, r=60, t=20, b=20),
        xaxis_range=[0, 110],
    )
    st.plotly_chart(fig_fill, use_container_width=True)

    # Z-score summary table
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Fill rate summary**")
        summary_tbl = fill_df[["factor_name", "category", "filled", "total", "fill_pct", "avg_z", "min_z", "max_z"]].copy()
        summary_tbl.columns = ["Factor", "Category", "Filled", "Total", "Fill %", "Avg Z", "Min Z", "Max Z"]
        st.dataframe(summary_tbl, hide_index=True, use_container_width=True)

    with col_r:
        st.markdown("**Factors with extreme z-scores (|z| > 4)**")
        extremes = zscore_df[zscore_df["factor_value_z"].abs() > 4]
        if extremes.empty:
            st.success("✅ No extreme z-scores detected.")
        else:
            ext_summary = (
                extremes.groupby(["factor_name", "category"])
                .agg(n_extreme=("factor_value_z", "count"),
                     max_abs_z=("factor_value_z", lambda x: x.abs().max()))
                .reset_index()
                .sort_values("max_abs_z", ascending=False)
            )
            st.dataframe(ext_summary, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown(f"**Z-score distributions by factor — {sel_date_f}**")

    categories = sorted(zscore_df["category"].dropna().unique())
    sel_cat = st.selectbox("Filter by category", ["All"] + categories, key="fq_cat")

    plot_df = zscore_df if sel_cat == "All" else zscore_df[zscore_df["category"] == sel_cat]

    fig_z = px.box(
        plot_df,
        x="factor_value_z",
        y="factor_name",
        orientation="h",
        labels={"factor_value_z": "Z-score", "factor_name": ""},
        color="category",
        points=False,
    )
    fig_z.update_layout(
        height=max(400, plot_df["factor_name"].nunique() * 25),
        margin=dict(l=0, r=0, t=20, b=20),
        showlegend=True,
        xaxis=dict(zeroline=True, zerolinecolor="grey", zerolinewidth=1),
    )
    fig_z.add_vline(x=3,  line_dash="dash", line_color="orange", annotation_text="+3σ")
    fig_z.add_vline(x=-3, line_dash="dash", line_color="orange", annotation_text="−3σ")
    st.plotly_chart(fig_z, use_container_width=True)


# ============================================================
# TAB 4: Model Quality
# ============================================================
with tab_model:
    st.subheader("Model score quality")

    with st.spinner("Loading model data…"):
        null_rates = _model_null_rates()
        model_cov  = _model_coverage_by_date()

    model_dates = null_rates["data_date"].unique()
    sel_date_m  = st.selectbox(
        "Snapshot date for distribution", sorted(model_dates)[::-1],
        index=0, key="mq_date"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Null rate per model per date**")
        pivot_null = null_rates.pivot_table(
            index="data_date", columns="Model", values="null_pct", aggfunc="first"
        ).fillna(0)

        fig_heat = px.imshow(
            pivot_null.T,
            color_continuous_scale=["#34A853", "#FBBC04", "#EA4335"],
            zmin=0, zmax=5,
            labels={"color": "Null %"},
            aspect="auto",
        )
        fig_heat.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=20, b=60),
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        st.markdown("**Scored securities per model over time**")
        fig_cov_line = px.line(
            model_cov, x="data_date", y="n_scored", color="Model",
            labels={"data_date": "Date", "n_scored": "Companies scored"},
            markers=True,
        )
        fig_cov_line.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=20, b=40),
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cov_line, use_container_width=True)

    st.divider()

    with st.spinner("Loading score distributions…"):
        score_df = _model_score_dist(sel_date_m)

    st.markdown(f"**Score distributions at {sel_date_m}**")
    fig_violin = px.violin(
        score_df, x="Model", y="model_value_z",
        box=True, points=False,
        labels={"model_value_z": "Z-score", "Model": ""},
        color="Model",
    )
    fig_violin.update_layout(
        height=420,
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=80),
        xaxis_tickangle=-30,
    )
    fig_violin.add_hline(y=0, line_dash="dash", line_color="grey")
    st.plotly_chart(fig_violin, use_container_width=True)

    # Summary stats table
    summary_m = (
        score_df.groupby("Model")["model_value_z"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(3)
        .reset_index()
    )
    summary_m.columns = ["Model", "N", "Mean Z", "Std Z", "Min Z", "Max Z"]
    st.dataframe(summary_m, hide_index=True, use_container_width=True)

    # Flag off-centre models
    off_centre = summary_m[summary_m["Mean Z"].abs() > 0.1]
    if not off_centre.empty:
        st.warning(f"⚠️ Models with |mean z| > 0.1: {', '.join(off_centre['Model'].tolist())} — z-score centering may be off.")
    else:
        st.success("✅ All model z-scores are well-centred (|mean| ≤ 0.1).")


# ============================================================
# TAB 5: Barra Quality
# ============================================================
with tab_barra:
    st.subheader("Barra factor risk model quality")

    with st.spinner("Loading Barra data…"):
        barra_stats, fr_info = _barra_snapshot_stats()
        risk_df = _risk_summary()

    # Factor returns summary
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Factor return rows",  f"{fr_info['n_rows']:,}")
    col_b.metric("Factor returns start", fr_info["min_date"])
    col_c.metric("Factor returns end",   fr_info["max_date"])

    st.divider()

    # Filter to factor-snapshot dates only (those with large n_stocks)
    # The factor-snapshot dates have significantly more stocks than weekly snapshots
    q75 = barra_stats["n_stocks"].quantile(0.75)
    factor_snaps = barra_stats[barra_stats["n_stocks"] >= q75 * 0.9].copy()
    weekly_snaps = barra_stats[barra_stats["n_stocks"] < q75 * 0.9].copy()

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(f"**Factor-snapshot dates ({len(factor_snaps)} dates)**")
        st.caption("These dates align with the factors.db/models.db snapshot dates.")

        fig_fsnap = go.Figure()
        fig_fsnap.add_trace(go.Bar(
            x=factor_snaps["snapshot_date"],
            y=factor_snaps["n_stocks"],
            name="Securities",
            marker_color="#4C8BF5",
        ))
        fig_fsnap.add_trace(go.Scatter(
            x=factor_snaps["snapshot_date"],
            y=factor_snaps["avg_idio"],
            name="Avg idio var",
            yaxis="y2",
            mode="lines+markers",
            marker_color="#EA4335",
            line_width=2,
        ))
        fig_fsnap.update_layout(
            height=320,
            margin=dict(l=0, r=60, t=20, b=60),
            xaxis_tickangle=-45,
            yaxis=dict(title="# securities"),
            yaxis2=dict(title="Avg idio var (annualised)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_fsnap, use_container_width=True)

        # Table of factor snapshots
        tbl = factor_snaps[["snapshot_date", "n_stocks", "avg_idio", "max_idio"]].copy()
        tbl.columns = ["Date", "Securities", "Avg Idio Var", "Max Idio Var"]
        st.dataframe(tbl, hide_index=True, use_container_width=True)

    with col_r:
        st.markdown("**Idiosyncratic variance over time (all weekly snapshots)**")
        st.caption("High avg idio_var → factor model explains less. Expected during volatility spikes.")

        recent_weekly = weekly_snaps.tail(100)
        fig_idio = go.Figure()
        fig_idio.add_trace(go.Scatter(
            x=recent_weekly["snapshot_date"],
            y=recent_weekly["avg_idio"],
            mode="lines",
            name="Avg idio var",
            fill="tozeroy",
            line_color="#4C8BF5",
        ))
        fig_idio.add_hline(
            y=weekly_snaps["avg_idio"].mean(),
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Mean: {weekly_snaps['avg_idio'].mean():.4f}",
        )
        fig_idio.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=20, b=40),
            yaxis_title="Avg idiosyncratic variance",
            xaxis_title="Snapshot date",
        )
        st.plotly_chart(fig_idio, use_container_width=True)

        st.caption(
            "ℹ️ **Healthy range for factor-snapshot VRA B²: 0.8–1.3.** "
            "B² > 1.3 → factor model underestimates variance (high vol regime). "
            "B² < 0.8 → overestimates (calm market). "
            "Very high idio_var (like April/May 2025 tariff shock) produces B² > 2 — expected behaviour."
        )

    st.divider()

    # Ledoit-Wolf risk snapshots
    st.markdown("**Ledoit-Wolf covariance matrices (risk.db)**")
    if risk_df.empty:
        st.info("risk.db not found or empty.")
    else:
        fig_risk = px.bar(
            risk_df, x="data_date", y="n_stocks",
            labels={"data_date": "Date", "n_stocks": "# stocks"},
            text="n_stocks",
            color="shrinkage_coeff",
            color_continuous_scale="Blues",
        )
        fig_risk.update_traces(textposition="outside")
        fig_risk.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=40))
        st.plotly_chart(fig_risk, use_container_width=True)
        st.dataframe(risk_df, hide_index=True, use_container_width=True)

    # Total weekly snapshot count
    st.divider()
    st.metric("Total Barra weekly snapshots", f"{len(barra_stats):,}")
    with st.expander("All Barra snapshot dates"):
        st.dataframe(
            barra_stats.sort_values("snapshot_date", ascending=False),
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# TAB 6: Return Coverage
# ============================================================
with tab_ret:
    st.subheader("Price / return data coverage")

    with st.spinner("Loading returns metadata…"):
        ret_cov = _returns_coverage()

    total_isins = len(ret_cov)
    stale_threshold = st.slider("Stale threshold (days since last price)", 5, 60, 10, key="stale_thresh")

    stale   = ret_cov[ret_cov["days_stale"] > stale_threshold]
    current = ret_cov[ret_cov["days_stale"] <= stale_threshold]
    short_hist = ret_cov[ret_cov["n_days"] < 252]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ISINs with prices",           f"{total_isins:,}")
    c2.metric("Current (≤ threshold days)",  f"{len(current):,}")
    c3.metric("Stale (> threshold days)",    f"{len(stale):,}")
    c4.metric("Short history (< 1yr)",       f"{len(short_hist):,}")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Days since last price (staleness)**")
        # Almost all ISINs share the same last date so a date histogram is a single bar.
        # Show the staleness distribution instead — how stale is each company.
        buckets = pd.cut(
            ret_cov["days_stale"],
            bins=[-1, 2, 5, 10, 30, 60, float("inf")],
            labels=["0–2 d", "3–5 d", "6–10 d", "11–30 d", "31–60 d", ">60 d"],
        )
        stale_counts = buckets.value_counts().sort_index().reset_index()
        stale_counts.columns = ["Bucket", "Count"]
        stale_counts["color"] = ["#34A853","#34A853","#FBBC04","#EA4335","#EA4335","#EA4335"]
        fig_last = px.bar(
            stale_counts, x="Bucket", y="Count",
            labels={"Bucket": "Days since last price", "Count": "# ISINs"},
            color="Bucket",
            color_discrete_sequence=stale_counts["color"].tolist(),
        )
        fig_last.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=40), showlegend=False
        )
        st.plotly_chart(fig_last, use_container_width=True)

    with col_r:
        st.markdown("**History length (years of daily prices)**")
        ret_cov_plot = ret_cov.copy()
        ret_cov_plot["years"] = (ret_cov_plot["n_days"] / 252).round(1)
        fig_hist = px.histogram(
            ret_cov_plot, x="years", nbins=30,
            labels={"years": "Years of price history", "count": "# ISINs"},
            color_discrete_sequence=["#34A853"],
        )
        fig_hist.add_vline(x=3, line_dash="dash", line_color="orange", annotation_text="3 yr")
        fig_hist.add_vline(x=5, line_dash="dash", line_color="grey",   annotation_text="5 yr")
        fig_hist.add_vline(x=6, line_dash="dash", line_color="grey",   annotation_text="6 yr")
        fig_hist.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=40))
        st.plotly_chart(fig_hist, use_container_width=True)

    # Stale companies table
    if not stale.empty:
        st.markdown(f"**Stale companies ({len(stale)} ISINs with last price > {stale_threshold} days ago)**")
        stale_disp = stale.sort_values("last_date")[["isin", "first_date", "last_date", "n_days", "days_stale"]].copy()
        stale_disp["last_date"]  = stale_disp["last_date"].dt.strftime("%Y-%m-%d")
        stale_disp["first_date"] = stale_disp["first_date"].dt.strftime("%Y-%m-%d")

        # Join ticker from universe
        with get_db(UNIVERSE_DB) as conn:
            tickers = pd.read_sql("SELECT isin, ticker, company_name FROM companies", conn)
        stale_disp = stale_disp.merge(tickers, on="isin", how="left")
        stale_disp = stale_disp[["ticker", "company_name", "isin", "first_date", "last_date", "n_days", "days_stale"]]
        st.dataframe(stale_disp.head(100), hide_index=True, use_container_width=True)
    else:
        st.success(f"✅ All ISINs have prices within the last {stale_threshold} days.")

    st.divider()
    st.markdown("**Companies in universe but missing from returns.db**")

    with get_db(UNIVERSE_DB) as conn:
        universe_isins = set(r[0] for r in conn.execute("SELECT isin FROM companies WHERE isin IS NOT NULL").fetchall())
    returns_isins = set(ret_cov["isin"].tolist())
    missing_from_returns = universe_isins - returns_isins

    if missing_from_returns:
        with get_db(UNIVERSE_DB) as conn:
            missing_df = pd.read_sql(
                f"SELECT isin, ticker, company_name, gics_sector FROM companies "
                f"WHERE isin IN ({','.join('?' * len(missing_from_returns))})",
                conn,
                params=list(missing_from_returns),
            )
        st.warning(f"⚠️ {len(missing_from_returns)} companies in universe.db have no price data:")
        st.dataframe(missing_df, hide_index=True, use_container_width=True)
    else:
        st.success("✅ All universe companies have price data in returns.db.")


# ============================================================
# TAB 7: Constituents
# ============================================================
with tab_const:
    st.subheader("Constituent (financial statement) data quality")

    with st.spinner("Loading constituent coverage…"):
        const_cov = _constituent_coverage()

    c1, c2, c3 = st.columns(3)
    c1.metric("Companies with constituent data", f"{len(const_cov):,}")
    c2.metric("Avg fiscal years per company",    f"{const_cov['n_years'].mean():.1f}")
    c3.metric("Median fiscal years per company", f"{int(const_cov['n_years'].median())}")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Fiscal years of data per company**")
        fig_per = px.histogram(
            const_cov, x="n_years", nbins=const_cov["n_years"].max(),
            labels={"n_years": "Distinct fiscal years", "count": "# companies"},
            color_discrete_sequence=["#4C8BF5"],
        )
        fig_per.add_vline(x=3, line_dash="dash", line_color="orange", annotation_text="3 yrs")
        fig_per.add_vline(x=5, line_dash="dash", line_color="grey",   annotation_text="5 yrs")
        fig_per.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=40))
        st.plotly_chart(fig_per, use_container_width=True)

    with col_r:
        st.markdown("**Distribution of latest publish date**")
        const_cov_dates = const_cov.copy()
        const_cov_dates["latest_publish"] = pd.to_datetime(const_cov_dates["latest_publish"], errors="coerce")
        fig_pub = px.histogram(
            const_cov_dates.dropna(subset=["latest_publish"]),
            x="latest_publish", nbins=40,
            labels={"latest_publish": "Latest filing date", "count": "# companies"},
            color_discrete_sequence=["#34A853"],
        )
        fig_pub.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=40))
        st.plotly_chart(fig_pub, use_container_width=True)

    # Thin data flag
    st.markdown("**Companies with sparse data (< 3 fiscal years)**")
    sparse = const_cov[const_cov["n_years"] < 3].sort_values("n_years")
    if sparse.empty:
        st.success("✅ All companies have at least 3 fiscal years of data.")
    else:
        with get_db(UNIVERSE_DB) as conn:
            tickers = pd.read_sql("SELECT isin, ticker, company_name FROM companies", conn)
        sparse = sparse.merge(tickers.rename(columns={"isin": "security_id"}), on="security_id", how="left")
        sparse_disp = sparse[["ticker", "company_name", "security_id", "n_years", "first_fy", "last_fy", "latest_publish"]]
        st.warning(f"⚠️ {len(sparse)} companies have fewer than 3 fiscal years of data:")
        st.dataframe(sparse_disp.head(50), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("**Companies in universe with NO constituent data**")

    # constituents.db uses security_id = ISIN (EDGAR-sourced) OR SimFin numeric ID (SimFin-sourced).
    # Must check both keys; comparing only ISINs gives a false ~600-company gap.
    const_sids = set(const_cov["security_id"].astype(str).tolist())
    with get_db(UNIVERSE_DB) as conn:
        companies_all = pd.read_sql(
            "SELECT isin, ticker, company_name, gics_sector, simfin_id, cik FROM companies", conn
        )
    companies_all["simfin_str"] = companies_all["simfin_id"].dropna().apply(lambda x: str(int(x)))
    companies_all["has_const"] = (
        companies_all["isin"].isin(const_sids) |
        companies_all["simfin_str"].isin(const_sids)
    )
    no_const_df = companies_all[~companies_all["has_const"]].copy()

    if not no_const_df.empty:
        nc_disp = no_const_df[["ticker", "company_name", "gics_sector", "simfin_id", "cik"]].copy()
        nc_disp["simfin_id"] = nc_disp["simfin_id"].apply(lambda x: str(int(x)) if pd.notna(x) else "—")
        nc_disp["cik"] = nc_disp["cik"].fillna("—")
        st.warning(f"⚠️ {len(no_const_df)} universe companies have no constituent data (neither ISIN nor SimFin ID found):")
        st.dataframe(nc_disp, hide_index=True, use_container_width=True)
    else:
        st.success("✅ All universe companies have constituent data.")

    st.divider()
    st.subheader("Recent filings (last 90 days)")
    days_window = st.slider("Window (days)", 30, 180, 90, step=30, key="const_window")
    with st.spinner("Fetching recent filings…"):
        recent = _recent_filings(n_days=days_window)

    if recent.empty:
        st.info(f"No filings with publish_date in the last {days_window} days.")
    else:
        with get_db(UNIVERSE_DB) as conn:
            tickers = pd.read_sql("SELECT isin, ticker, company_name FROM companies", conn)
        recent = recent.merge(tickers.rename(columns={"isin": "security_id"}), on="security_id", how="left")
        st.metric("Recent filings", f"{len(recent):,}")
        st.dataframe(
            recent[["publish_date", "ticker", "company_name", "fiscal_year", "fiscal_period", "statement_type"]]
            .sort_values("publish_date", ascending=False)
            .head(200),
            hide_index=True,
            use_container_width=True,
        )
