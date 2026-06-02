"""
11_Macro.py — Macro regime dashboard.

Tells the investment story across five dimensions: equities, rates/curve,
credit conditions, inflation trajectory, and labor market health.
Every element is designed to be actionable, not decorative.
"""

from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import MACRO_DB, RETURNS_DB
from macro_db import load_signals_reference
from utils import get_db, inject_css

st.set_page_config(page_title="Macro", layout="wide")
inject_css()

# ── Regime thresholds ─────────────────────────────────────────────────────────
# Domain-knowledge constants keyed by signal_id. Each entry: list of
# (lo_inclusive, hi_exclusive, label, colour) evaluated top-to-bottom.
# Adjust thresholds here; nothing else in the file needs to change.
REGIME: dict[str, dict] = {
    "VIX": {
        "label": "VIX",
        "caption": "Market fear gauge. < 15 = complacency; > 30 = elevated fear; > 40 = crisis.",
        "levels": [
            ( 0,  15, "Low",      "🟢"),
            (15,  25, "Normal",   "🟡"),
            (25,  35, "Elevated", "🟠"),
            (35, 999, "Fear",     "🔴"),
        ],
    },
    "US2Y10Y_SPREAD": {
        "label": "Yield Curve",
        "caption": "Inversion (< 0) reliably precedes recessions by 12–18 months.",
        "levels": [
            (-999,  0.0, "Inverted",   "🔴"),
            (  0.0, 0.5, "Flat",       "🟡"),
            (  0.5, 999, "Steepening", "🟢"),
        ],
    },
    "HY_OAS": {
        "label": "HY Credit",
        "caption": "Tight spreads = risk appetite. Widening = flight to quality.",
        "levels": [
            (0.0, 3.0, "Tight",  "🟢"),
            (3.0, 5.5, "Normal", "🟡"),
            (5.5, 999, "Wide",   "🔴"),
        ],
    },
    "CPI_YoY": {
        "label": "Inflation",
        "caption": "Fed's 2% target. > 2.5% = Fed pressure; > 4% = active tightening.",
        "levels": [
            (0.0, 2.5, "On Target",    "🟢"),
            (2.5, 4.0, "Above Target", "🟡"),
            (4.0, 999, "Running Hot",  "🔴"),
        ],
    },
    "UNEMPLOYMENT": {
        "label": "Labor",
        "caption": "Sub-4.5% = full employment. Rising rate lags recessions by 2–4 quarters.",
        "levels": [
            (0.0, 4.5, "Tight",         "🟢"),
            (4.5, 5.5, "Softening",     "🟡"),
            (5.5, 999, "Deteriorating", "🔴"),
        ],
    },
}

FED_TARGET_PCT = 2.0  # Fed 2% inflation target

# Index groups used in the equities section — driven by index_name values in
# benchmark_returns. Adding a new index there will surface it automatically
# once it appears in the right group below.
EQUITY_GROUPS: dict[str, list[str]] = {
    "Broad Market":    ["sp500", "russell_1000", "russell_2000"],
    "Style":           ["russell_1000_growth", "russell_1000_value",
                        "sp500_growth", "sp500_value"],
    "Factor":          ["msci_usa_quality", "msci_usa_momentum",
                        "msci_usa_min_vol", "msci_usa_value", "msci_usa_size"],
    "Global":          ["msci_usa", "europe_equity", "japan_equity", "em_equity"],
}

# Category display order and section titles
CATEGORY_ORDER  = ["equities", "rates", "credit", "inflation", "labor"]
CATEGORY_TITLES = {
    "equities":  "Equities & Volatility",
    "rates":     "Rates & Yield Curve",
    "credit":    "Credit Conditions",
    "inflation": "Inflation",
    "labor":     "Labor Market",
}

# Chart palette — consistent across sections
PALETTE  = ["#4C9BE8", "#E88D4C", "#2ECC71", "#E74C3C", "#9B59B6", "#F39C12",
            "#1ABC9C", "#E67E22", "#3498DB", "#E91E63"]
DARK_BG  = "#0E1117"
CHART_LAYOUT = dict(
    plot_bgcolor=DARK_BG,
    paper_bgcolor=DARK_BG,
    font_color="white",
    hovermode="x unified",
    margin=dict(l=0, r=80, t=24, b=0),
)

# Human-readable display names for index_name values in benchmark_returns
INDEX_LABELS: dict[str, str] = {
    "sp500":               "S&P 500",
    "russell_1000":        "Russell 1000",
    "russell_2000":        "Russell 2000",
    "russell_1000_growth": "R1000 Growth",
    "russell_1000_value":  "R1000 Value",
    "sp500_growth":        "S&P 500 Growth",
    "sp500_value":         "S&P 500 Value",
    "msci_usa":            "MSCI USA",
    "msci_usa_quality":    "Quality",
    "msci_usa_momentum":   "Momentum",
    "msci_usa_min_vol":    "Min Vol",
    "msci_usa_value":      "Value",
    "msci_usa_size":       "Size",
    "europe_equity":       "Europe",
    "japan_equity":        "Japan",
    "em_equity":           "Emerging Mkts",
    "gold":                "Gold",
    "global_reits":        "Global REITs",
    "treasury_long":       "20yr Treasury",
    "treasury_mid":        "7-10yr Treasury",
    "treasury_short":      "1-3yr Treasury",
    "corp_bonds":          "Corp Bonds",
    "em_bonds":            "EM Bonds",
    "ai_tech":             "AI/Tech",
}


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _load_macro() -> tuple[pd.DataFrame, dict]:
    """Return (wide_df, signals_ref) from macro.db. Cached 1 hour."""
    ref = load_signals_reference()
    with get_db(MACRO_DB) as conn:
        df = pd.read_sql(
            "SELECT published_date, signal_id, value FROM daily_signals ORDER BY published_date",
            conn,
            parse_dates=["published_date"],
        )
    wide = df.pivot(index="published_date", columns="signal_id", values="value")
    return wide, ref


@st.cache_data(ttl=3600)
def _load_indices() -> pd.DataFrame:
    """Return benchmark_returns as a wide DataFrame (date × index_name → close).
    Only indices with data in EQUITY_GROUPS are loaded."""
    all_indices = {idx for group in EQUITY_GROUPS.values() for idx in group}
    placeholders = ",".join("?" * len(all_indices))
    with get_db(RETURNS_DB) as conn:
        df = pd.read_sql(
            f"SELECT date, index_name, close FROM benchmark_returns "
            f"WHERE index_name IN ({placeholders}) ORDER BY date",
            conn,
            params=list(all_indices),
            parse_dates=["date"],
        )
    return df.pivot(index="date", columns="index_name", values="close")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_transform(series: pd.Series, transform: str | None) -> pd.Series:
    if transform == "diff_mom":
        return series.diff().dropna()
    return series.dropna()


def _by_category(ref: dict) -> dict[str, list[tuple]]:
    """Group signals_ref by category, each list sorted by display_order."""
    groups: dict[str, list] = defaultdict(list)
    for sid, meta in ref.items():
        cat = meta.get("category")
        if cat:
            groups[cat].append((meta["display_order"], sid, meta))
    for cat in groups:
        groups[cat].sort()
    return dict(groups)


def _get_regime(signal_id: str, value: float) -> tuple[str, str]:
    config = REGIME.get(signal_id)
    if config is None or pd.isna(value):
        return "—", "⬜"
    for lo, hi, label, emoji in config["levels"]:
        if lo <= value < hi:
            return label, emoji
    return "—", "⬜"


def _latest(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return float(s.iloc[-1]) if len(s) else None


def _delta(df: pd.DataFrame, col: str, periods: int) -> float | None:
    """Absolute change over last `periods` available observations."""
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return float(s.iloc[-1] - s.iloc[-1 - periods]) if len(s) > periods else None


def _rebase(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rebase each column to 100 at its first non-NaN value."""
    available = [c for c in cols if c in df.columns]
    out = df[available].copy()
    for c in available:
        first = out[c].first_valid_index()
        if first is not None:
            out[c] = out[c] / out[c].loc[first] * 100
    return out


def _ytd_return(df: pd.DataFrame, col: str) -> float | None:
    """Total return from last trading day of prior year to latest available."""
    if col not in df.columns:
        return None
    s = df[col].dropna()
    if s.empty:
        return None
    year_start = s[s.index < f"{s.index[-1].year}-01-01"]
    if year_start.empty:
        return None
    return float((s.iloc[-1] / year_start.iloc[-1] - 1) * 100)


# ── Chart builders ────────────────────────────────────────────────────────────

def _chart_vix(df: pd.DataFrame, col: str, signal_name: str) -> go.Figure:
    """VIX time series with regime bands as background shading."""
    s = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
    fig = go.Figure()

    if not s.empty:
        # Regime band fills
        x_range = [s.index[0], s.index[-1]]
        bands = [(0, 15, "rgba(46,204,113,0.07)"),
                 (15, 25, "rgba(241,196,15,0.07)"),
                 (25, 35, "rgba(230,126,34,0.07)"),
                 (35, 80, "rgba(231,76,60,0.07)")]
        for lo, hi, fill in bands:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=fill, line_width=0)

        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=signal_name,
            line=dict(color="#E88D4C", width=1.5),
        ))

        # Horizontal regime labels
        for level, label in [(15, "15"), (25, "25"), (35, "35")]:
            fig.add_hline(y=level, line_dash="dot", line_color="white",
                          line_width=0.6, opacity=0.4,
                          annotation_text=label, annotation_position="right",
                          annotation_font_size=9)

    fig.update_layout(
        height=240, yaxis_title="VIX",
        showlegend=False, **CHART_LAYOUT,
    )
    return fig


def _chart_indexed(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    height_per_row: int = 280,
) -> go.Figure:
    """One subplot per equity group, each rebased to 100 at the lookback start."""
    group_names = [g for g in groups if any(i in df.columns for i in groups[g])]
    n = len(group_names)
    if n == 0:
        return go.Figure()

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[g for g in group_names],
    )
    for row, group in enumerate(group_names, start=1):
        indices = [i for i in groups[group] if i in df.columns]
        rebased = _rebase(df, indices)
        for i, idx in enumerate(indices):
            if idx not in rebased.columns:
                continue
            s = rebased[idx].dropna()
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values,
                name=INDEX_LABELS.get(idx, idx),
                line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
                legendgroup=group,
                legendgrouptitle_text=group if i == 0 else None,
            ), row=row, col=1)
        fig.add_hline(y=100, line_dash="dot", line_color="white",
                      line_width=0.5, opacity=0.3, row=row, col=1)
        fig.update_yaxes(title_text="Rebased", row=row, col=1)

    fig.update_layout(
        height=height_per_row * n,
        legend=dict(groupclick="toggleitem"),
        **{**CHART_LAYOUT, "margin": dict(l=0, r=80, t=32, b=0)},
    )
    return fig


def _chart_rates(df: pd.DataFrame, signals: list[tuple]) -> go.Figure:
    yield_signals  = [(o, sid, m) for o, sid, m in signals if sid != "US2Y10Y_SPREAD"]
    spread_signals = [(o, sid, m) for o, sid, m in signals if sid == "US2Y10Y_SPREAD"]

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.6, 0.4],
        shared_xaxes=True, vertical_spacing=0.05,
    )
    for i, (_, sid, meta) in enumerate(yield_signals):
        if sid not in df.columns:
            continue
        s = df[sid].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=meta["signal_name"],
            line=dict(color=PALETTE[i], width=1.5),
        ), row=1, col=1)

    for _, sid, meta in spread_signals:
        if sid not in df.columns:
            continue
        s = df[sid].dropna()
        bar_colors = ["#2ECC71" if v >= 0 else "#E74C3C" for v in s.values]
        fig.add_trace(go.Bar(
            x=s.index, y=s.values,
            name=meta["signal_name"],
            marker_color=bar_colors,
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="white",
                      line_width=1, row=2, col=1)

    fig.update_yaxes(title_text="Yield (%)", row=1, col=1)
    fig.update_yaxes(title_text="Spread (%)", row=2, col=1)
    fig.update_layout(
        height=440,
        legend=dict(orientation="h", y=1.06),
        **CHART_LAYOUT,
    )
    return fig


def _chart_credit(df: pd.DataFrame, signals: list[tuple]) -> go.Figure:
    fig = go.Figure()
    for i, (_, sid, meta) in enumerate(signals):
        if sid not in df.columns:
            continue
        s = df[sid].dropna()
        color = PALETTE[i]
        avg = s.mean()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=meta["signal_name"],
            line=dict(color=color, width=1.5),
        ))
        fig.add_hline(
            y=avg, line_dash="dash", line_color=color, line_width=0.8, opacity=0.5,
            annotation_text=f"avg {avg:.2f}%",
            annotation_position="right", annotation_font_size=10,
        )
    fig.update_layout(
        height=300, yaxis_title="OAS (%)",
        legend=dict(orientation="h", y=1.08),
        **CHART_LAYOUT,
    )
    return fig


def _chart_inflation(df: pd.DataFrame, signals: list[tuple]) -> go.Figure:
    fig = go.Figure()
    for i, (_, sid, meta) in enumerate(signals):
        if sid not in df.columns:
            continue
        s = _apply_transform(df[sid], meta.get("display_transform"))
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=meta["signal_name"],
            line=dict(color=PALETTE[i + 3], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(243,156,18,0.08)",
        ))
    fig.add_hline(
        y=FED_TARGET_PCT, line_dash="dash", line_color="#2ECC71", line_width=1,
        annotation_text=f"Fed target {FED_TARGET_PCT}%",
        annotation_position="right", annotation_font_size=10,
    )
    fig.update_layout(
        height=280, yaxis_title="YoY %", showlegend=False, **CHART_LAYOUT,
    )
    return fig


def _chart_labor(df: pd.DataFrame, signals: list[tuple]) -> go.Figure:
    n = len(signals)
    subtitles = [
        m["signal_name"] if m.get("display_transform") is None
        else f"{m['signal_name']} MoM (000s)"
        for _, _, m in signals
    ]
    fig = make_subplots(rows=1, cols=n, subplot_titles=subtitles,
                        horizontal_spacing=0.12)
    for i, (_, sid, meta) in enumerate(signals, start=1):
        if sid not in df.columns:
            continue
        s = _apply_transform(df[sid], meta.get("display_transform"))
        if meta.get("display_transform") == "diff_mom":
            bar_colors = ["#2ECC71" if v >= 0 else "#E74C3C" for v in s.values]
            fig.add_trace(go.Bar(
                x=s.index, y=s.values,
                name=meta["signal_name"],
                marker_color=bar_colors,
            ), row=1, col=i)
        else:
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values,
                name=meta["signal_name"],
                line=dict(color=PALETTE[4 + i], width=1.5),
            ), row=1, col=i)

    fig.update_layout(
        height=280, showlegend=False,
        margin=dict(l=0, r=0, t=32, b=0),
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        font_color="white", hovermode="x unified",
    )
    return fig


# ── Page ──────────────────────────────────────────────────────────────────────

st.title("Macro Regime")

wide, ref       = _load_macro()
equity_wide     = _load_indices()
by_cat          = _by_category(ref)

# Lookback selector
lb_options = {"1Y": 1, "3Y": 3, "5Y": 5, "Full": None}
lb_label   = st.radio("Lookback", list(lb_options.keys()),
                      horizontal=True, index=1, label_visibility="collapsed")
lb_years   = lb_options[lb_label]

if lb_years is not None:
    cutoff       = pd.Timestamp("today") - pd.DateOffset(years=lb_years)
    display_wide = wide[wide.index >= cutoff]
    display_eq   = equity_wide[equity_wide.index >= cutoff]
else:
    display_wide = wide
    display_eq   = equity_wide

# ── Regime KPI bar ────────────────────────────────────────────────────────────
st.divider()
regime_sids = [sid for sid in REGIME if sid in ref or sid == "VIX"]
kpi_cols    = st.columns(len(regime_sids))

for col, sid in zip(kpi_cols, regime_sids):
    config           = REGIME[sid]
    val              = _latest(wide, sid)
    delta            = _delta(wide, sid, 20)
    label, emoji     = _get_regime(sid, val or 0.0)

    with col:
        st.caption(f"**{config['label']}**  {emoji} {label}")
        if val is not None:
            unit = (ref.get(sid) or {}).get("unit") or "%"
            fmt  = f"{val:.1f}" if sid == "VIX" else f"{val:.2f}%"
            st.metric(
                label=ref[sid]["signal_name"] if sid in ref else sid,
                value=fmt,
                delta=f"{delta:+.2f}pp" if delta is not None else None,
            )
            st.caption(config["caption"])
        else:
            st.write("No data")

st.divider()

# ── Equities & Volatility ─────────────────────────────────────────────────────
st.subheader(CATEGORY_TITLES["equities"])
st.caption(
    "Broad market trend sets the risk backdrop. Style and factor rotation reveal "
    "where capital is flowing. VIX confirms whether price action reflects genuine "
    "conviction or fragile calm."
)

# VIX sub-section
vix_signals = by_cat.get("equities", [])
if vix_signals:
    for _, sid, meta in vix_signals:
        if sid in wide.columns:
            st.plotly_chart(
                _chart_vix(display_wide, sid, meta["signal_name"]),
                use_container_width=True,
            )

# Indexed performance per group
if not display_eq.empty:
    # Filter EQUITY_GROUPS to only groups that have data in the current lookback
    active_groups = {
        g: idxs for g, idxs in EQUITY_GROUPS.items()
        if any(i in display_eq.columns for i in idxs)
    }
    if active_groups:
        st.plotly_chart(
            _chart_indexed(display_eq, active_groups),
            use_container_width=True,
        )

# Latest values table: broad market only (S&P 500, R1000, R2000)
broad = EQUITY_GROUPS.get("Broad Market", [])
rows = []
for idx in broad:
    if idx not in equity_wide.columns:
        continue
    s   = equity_wide[idx].dropna()
    val = float(s.iloc[-1]) if not s.empty else None
    ytd = _ytd_return(equity_wide, idx)
    d1  = _delta(equity_wide, idx, 1)
    if val is None:
        continue
    rows.append({
        "Index": INDEX_LABELS.get(idx, idx),
        "Last Close": f"{val:,.2f}",
        "1D Chg":     f"{d1:+.2f}" if d1 is not None else "—",
        "YTD":        f"{ytd:+.1f}%" if ytd is not None else "—",
        "As of":      s.index[-1].strftime("%Y-%m-%d"),
    })
if rows:
    st.dataframe(
        pd.DataFrame(rows).set_index("Index"),
        use_container_width=True,
        height=min(36 * len(rows) + 38, 180),
    )

st.divider()

# ── Macro signal sections ─────────────────────────────────────────────────────
SECTION_CAPTIONS = {
    "rates":     "Watch for curve steepness — a re-inversion signals renewed recession risk; "
                 "steepening post-inversion marks early recovery.",
    "credit":    "Spread compression is a green light for risk assets. Widening > 500bps HY "
                 "historically correlates with equity drawdowns > 20%.",
    "inflation": "Trajectory matters more than level. CPI falling toward 2% removes the Fed's "
                 "hawkish constraint; re-acceleration is the bear case.",
    "labor":     "A rising unemployment rate that lasts 3+ months has never failed to coincide "
                 "with a recession. NFP below +100k signals cooling.",
}

for cat in [c for c in CATEGORY_ORDER if c != "equities"]:
    signals = by_cat.get(cat)
    if not signals:
        continue

    st.subheader(CATEGORY_TITLES[cat])
    st.caption(SECTION_CAPTIONS.get(cat, ""))

    if cat == "rates":
        st.plotly_chart(_chart_rates(display_wide, signals),
                        use_container_width=True)
    elif cat == "credit":
        st.plotly_chart(_chart_credit(display_wide, signals),
                        use_container_width=True)
    elif cat == "inflation":
        st.plotly_chart(_chart_inflation(display_wide, signals),
                        use_container_width=True)
    elif cat == "labor":
        st.plotly_chart(_chart_labor(display_wide, signals),
                        use_container_width=True)

    # Latest values table for this section
    table_rows = []
    for _, sid, meta in signals:
        if sid not in wide.columns:
            continue
        s = wide[sid].dropna()
        if s.empty:
            continue
        val  = float(s.iloc[-1])
        prev = float(s.iloc[-2]) if len(s) > 1 else None
        table_rows.append({
            "Signal": meta["signal_name"],
            "Latest": f"{val:,.3f}",
            "Prev":   f"{prev:,.3f}" if prev is not None else "—",
            "Δ":      f"{val - prev:+.3f}" if prev is not None else "—",
            "As of":  s.index[-1].strftime("%Y-%m-%d"),
        })
    if table_rows:
        st.dataframe(
            pd.DataFrame(table_rows).set_index("Signal"),
            use_container_width=True,
            height=min(36 * len(table_rows) + 38, 200),
        )

    st.divider()
