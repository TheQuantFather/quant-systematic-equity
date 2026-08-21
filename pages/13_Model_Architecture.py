"""
13_Model_Architecture.py — Top-to-bottom visualisation of how the alpha model
is built: Alpha composite → base models → factors → factor formulas.

Designed to be readable by someone with zero knowledge of this infrastructure:
each layer is introduced with one plain-English line before any numbers appear.

FULLY DYNAMIC. Every node, weight, formula and applicability tag is read at
runtime from:
    data/models_reference.csv   (Alpha → models → factor weights)
    data/factors_reference.csv  (factor formula, inputs, direction, sector)
Re-weighting a model, adding a factor, renaming, or changing a formula is
reflected automatically — nothing on this page is hardcoded.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_REF, FACTORS_REF
from utils import inject_css

st.set_page_config(page_title="Model Architecture", layout="wide")
inject_css()

# ---------------------------------------------------------------------------
# Page-local styling — cards, pills, tables (matches the dark dashboard theme)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .ma-hero { font-size: 0.95rem; line-height: 1.7; opacity: 0.85; max-width: 880px; }
    .ma-step { display:flex; align-items:center; gap:10px; margin: 2px 0 6px; }
    .ma-step .dot { width:26px; height:26px; border-radius:50%; display:flex;
        align-items:center; justify-content:center; font-weight:700; font-size:0.8rem;
        background: rgba(59,130,246,0.18); border:1px solid rgba(59,130,246,0.4); color:#93c5fd; }
    .ma-pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.72rem;
        font-weight:600; letter-spacing:0.02em; }
    .ma-tag { display:inline-block; padding:1px 8px; border-radius:6px; font-size:0.68rem;
        font-weight:600; border:1px solid rgba(255,255,255,0.12); white-space:nowrap; }
    .ma-tbl { border-collapse:collapse; width:100%; font-size:0.8rem; }
    .ma-tbl th { text-align:left; padding:7px 10px; font-size:0.68rem; text-transform:uppercase;
        letter-spacing:0.04em; opacity:0.6; border-bottom:1px solid rgba(255,255,255,0.12); }
    .ma-tbl td { padding:8px 10px; border-bottom:1px solid rgba(255,255,255,0.06);
        vertical-align:top; }
    .ma-tbl td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
    .ma-tbl tr:hover td { background: rgba(255,255,255,0.03); }
    .ma-formula { font-family: 'SF Mono', ui-monospace, Menlo, monospace; font-size:0.76rem;
        opacity:0.92; }
    .ma-mut { opacity:0.55; font-size:0.74rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Model Architecture")
st.markdown(
    '<div class="ma-hero">A guided, top-to-bottom tour of how a single stock score is built. '
    'We start at the very top with the <b>Alpha</b> model — the one number used to rank every '
    'company — then unfold it layer by layer: the <b>base models</b> that compose Alpha, the '
    '<b>factors</b> that compose each model, and finally the <b>raw financial inputs</b> and '
    'formula behind every factor.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load reference data (single source of truth)
# ---------------------------------------------------------------------------
mref = pd.read_csv(MODELS_REF)
fref = pd.read_csv(FACTORS_REF)

mref["IsComposite"] = mref["IsComposite"].astype(int)
mref["Weights"] = pd.to_numeric(mref["Weights"], errors="coerce")
mref["sector_type"] = mref["sector_type"].fillna("").astype(str)

# id → display name for every model
MODEL_NAME = (
    mref.drop_duplicates("ModelID").set_index("ModelID")["Model"].to_dict()
)
# factor_id → row of metadata
fmeta = fref.set_index("factor_id")
fref["direction"] = pd.to_numeric(fref["direction"], errors="coerce")


def fname(fid: str) -> str:
    return str(fmeta.loc[fid, "factor_name"]) if fid in fmeta.index else fid


def fdir(fid: str) -> int:
    if fid in fmeta.index and not pd.isna(fmeta.loc[fid, "direction"]):
        return int(fmeta.loc[fid, "direction"])
    return 1


# Composite (top) models and the base models they reference
composite_ids = mref.loc[mref["IsComposite"] == 1, "ModelID"].unique().tolist()
base_ids = mref.loc[mref["IsComposite"] == 0, "ModelID"].unique().tolist()

if not composite_ids:
    st.warning(
        "No composite model (IsComposite = 1) found in models_reference.csv — "
        "there is no top-level Alpha blend to unfold."
    )
    st.stop()

# If there is more than one composite, let the user choose which to unfold.
if len(composite_ids) > 1:
    composite_id = st.selectbox(
        "Composite model", composite_ids,
        format_func=lambda c: f"{MODEL_NAME.get(c, c)} ({c})",
    )
else:
    composite_id = composite_ids[0]

ALPHA_NAME = MODEL_NAME.get(composite_id, composite_id)

# Alpha → base-model weights (normalised so they read as % of Alpha)
alpha_rows = mref[(mref["ModelID"] == composite_id) & (mref["IsComposite"] == 1)].copy()
alpha_rows = alpha_rows.rename(columns={"Factors": "child_id", "Weights": "raw_w"})
alpha_rows["w"] = alpha_rows["raw_w"] / alpha_rows["raw_w"].sum()
alpha_rows = alpha_rows.sort_values("w", ascending=False).reset_index(drop=True)

in_alpha_ids = alpha_rows["child_id"].tolist()
standalone_ids = [b for b in base_ids if b not in in_alpha_ids]


# ---------------------------------------------------------------------------
# Colour system — one stable colour per base model, derived from CSV order so
# adding a model just extends the palette (no hardcoded model→colour map).
# ---------------------------------------------------------------------------
PALETTE = [
    "#3B82F6", "#22C55E", "#F59E0B", "#EF4444", "#A855F7", "#06B6D4",
    "#EC4899", "#84CC16", "#F97316", "#14B8A6", "#8B5CF6", "#EAB308",
    "#10B981", "#6366F1",
]
MODEL_COLOR = {mid: PALETTE[i % len(PALETTE)] for i, mid in enumerate(base_ids)}
ALPHA_COLOR = "#E2E8F0"


def hex_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def sector_group(sector_type: str) -> str:
    """Plain-English applicability label from a model-row sector_type token.

    sector_type is a pipe-separated set of scopes (e.g. 'general|financial|bank').
    Only a *single* pure scope tells a clean substitution story; any multi-scope
    or empty/'all' value means the factor applies broadly. Unknown single scopes
    are surfaced verbatim rather than mislabelled, so new sectors degrade safely.
    """
    toks = {t.strip().lower() for t in str(sector_type or "").split("|") if t.strip()}
    toks.discard("all")
    if not toks:
        return "All companies"
    if toks == {"general"}:
        return "Ex-banks/REITs"
    if len(toks) == 1:
        only = next(iter(toks))
        return {"bank": "Banks only", "reit": "REITs only"}.get(only, f"{only.title()} only")
    return "All companies"  # broad multi-scope factor


# Colours for the known single-scope buckets; anything else falls back to grey.
SECTOR_COLOR = {
    "All companies": "#64748B",
    "Ex-banks/REITs": "#3B82F6",
    "Banks only": "#F59E0B",
    "REITs only": "#22C55E",
}
SECTOR_FALLBACK = "#A855F7"


def model_factor_rows(model_id: str) -> pd.DataFrame:
    """Factor-level rows for a base model, enriched with factor metadata."""
    df = mref[(mref["ModelID"] == model_id) & (mref["IsComposite"] == 0)].copy()
    df = df.rename(columns={"Factors": "factor_id", "Weights": "weight"})
    df["factor_name"] = df["factor_id"].map(fname)
    df["direction"] = df["factor_id"].map(fdir)
    df["applies"] = df["sector_type"].map(sector_group)
    return df.sort_values("weight", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------
n_factor_links = mref[mref["IsComposite"] == 0].shape[0]
inputs = set()
for c in fref["constituents"].dropna():
    for tok in str(c).split(","):
        if tok.strip():
            inputs.add(tok.strip())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"Models in {ALPHA_NAME}", len(in_alpha_ids))
c2.metric("Standalone models", len(standalone_ids))
c3.metric("Factors defined", fref["factor_id"].nunique())
c4.metric("Model · factor links", n_factor_links)
c5.metric("Raw financial inputs", len(inputs))

st.divider()

# ===========================================================================
# STEP 1 — The Alpha composite (top)
# ===========================================================================
st.markdown(
    '<div class="ma-step"><div class="dot">1</div>'
    f'<h2 style="margin:0;">{ALPHA_NAME} — the model that ranks every stock</h2></div>',
    unsafe_allow_html=True,
)
st.caption(
    f"{ALPHA_NAME} is a weighted blend of the base models below. A higher {ALPHA_NAME} score "
    "means the stock looks more attractive across all the dimensions we care about at once. "
    f"The ribbons flow top-to-bottom: width = how much each model contributes to {ALPHA_NAME}."
)

# ---- Vertical Sankey: Alpha (top) → base models (bottom) ----
labels = [f"{ALPHA_NAME}"]
node_colors = [hex_rgba(ALPHA_COLOR, 0.9)]
src, tgt, val, link_colors = [], [], [], []

for i, r in alpha_rows.iterrows():
    cid = r["child_id"]
    labels.append(f"{MODEL_NAME.get(cid, cid)}  ·  {r['w']*100:.1f}%")
    node_colors.append(hex_rgba(MODEL_COLOR.get(cid, "#888"), 0.92))
    src.append(0)
    tgt.append(i + 1)
    val.append(float(r["w"]))
    link_colors.append(hex_rgba(MODEL_COLOR.get(cid, "#888"), 0.45))

# Note: with orientation="v" Plotly swaps the x/y axes, so explicit node
# coordinates collapse the children onto one line — let snap auto-place them.
fig = go.Figure(
    go.Sankey(
        orientation="v",
        arrangement="snap",
        node=dict(
            label=labels, color=node_colors,
            pad=18, thickness=22,
            line=dict(color="rgba(255,255,255,0.15)", width=0.5),
        ),
        link=dict(source=src, target=tgt, value=val, color=link_colors),
    )
)
fig.update_layout(
    template="plotly_dark", height=360,
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=13),
)
st.plotly_chart(fig, use_container_width=True)

# ---- Top effective drivers: alpha_w(model) × factor share within model ----
contrib = []
for _, r in alpha_rows.iterrows():
    cid = r["child_id"]
    fr = model_factor_rows(cid)
    if fr.empty:
        continue
    share = fr["weight"] / fr["weight"].sum()
    for j, fr_row in fr.iterrows():
        contrib.append({
            "factor": fr_row["factor_name"],
            "model": MODEL_NAME.get(cid, cid),
            "model_id": cid,
            "eff": float(r["w"] * share.iloc[j]),
        })
contrib_df = pd.DataFrame(contrib).sort_values("eff", ascending=False).head(12)

with st.expander(f"Which individual factors move {ALPHA_NAME} the most?  (effective weight)", expanded=False):
    st.caption(
        f"Effective weight = the model's share of {ALPHA_NAME} × the factor's share of that model. "
        "It answers: across the whole pipeline, how much does one factor actually drive the final score."
    )
    bar = go.Figure(
        go.Bar(
            x=contrib_df["eff"][::-1] * 100,
            y=contrib_df["factor"][::-1],
            orientation="h",
            marker_color=[hex_rgba(MODEL_COLOR.get(m, "#888"), 0.85)
                          for m in contrib_df["model_id"][::-1]],
            customdata=contrib_df["model"][::-1],
            hovertemplate="<b>%{y}</b><br>via %{customdata}<br>%{x:.1f}% of " + ALPHA_NAME + "<extra></extra>",
            text=[f"{v*100:.1f}%" for v in contrib_df["eff"][::-1]],
            textposition="outside",
        )
    )
    bar.update_layout(
        template="plotly_dark", height=420,
        margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=f"Effective weight in {ALPHA_NAME} (%)",
    )
    st.plotly_chart(bar, use_container_width=True)

# ---- Optional: full three-tier flow in one picture ----
with st.expander(f"Show the complete flow in one diagram  ({ALPHA_NAME} → models → factors)", expanded=False):
    st.caption(
        "The entire formation in a single Sankey. Factor weights are normalised within "
        "each model for a clean flow; bank/REIT-only factors substitute for the general "
        "ones on a per-company basis (so they appear as alternative branches)."
    )
    f_labels = [ALPHA_NAME]
    f_colors = [hex_rgba(ALPHA_COLOR, 0.9)]
    fsrc, ftgt, fval, flink = [], [], [], []
    node_idx = {composite_id: 0}
    for _, r in alpha_rows.iterrows():
        cid = r["child_id"]
        node_idx[cid] = len(f_labels)
        f_labels.append(MODEL_NAME.get(cid, cid))
        f_colors.append(hex_rgba(MODEL_COLOR.get(cid, "#888"), 0.92))
        fsrc.append(0); ftgt.append(node_idx[cid]); fval.append(float(r["w"]))
        flink.append(hex_rgba(MODEL_COLOR.get(cid, "#888"), 0.4))
    for _, r in alpha_rows.iterrows():
        cid = r["child_id"]
        fr = model_factor_rows(cid)
        if fr.empty:
            continue
        share = fr["weight"] / fr["weight"].sum()
        col = MODEL_COLOR.get(cid, "#888")
        for j, fr_row in fr.iterrows():
            leaf = len(f_labels)
            arrow = "↑" if fr_row["direction"] > 0 else "↓"
            f_labels.append(f"{arrow} {fr_row['factor_name']}")
            f_colors.append(hex_rgba(col, 0.6))
            fsrc.append(node_idx[cid]); ftgt.append(leaf)
            fval.append(float(r["w"] * share.iloc[j]))
            flink.append(hex_rgba(col, 0.28))
    big = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=f_labels, color=f_colors, pad=10, thickness=16,
                      line=dict(color="rgba(255,255,255,0.12)", width=0.5)),
            link=dict(source=fsrc, target=ftgt, value=fval, color=flink),
        )
    )
    big.update_layout(
        template="plotly_dark", height=max(560, 16 * (len(f_labels) - len(alpha_rows))),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )
    st.plotly_chart(big, use_container_width=True)

st.divider()


# ===========================================================================
# STEP 2 + 3 — Inside each model: factor weights and factor formulas
# ===========================================================================
def render_model(model_id: str, weight_in_alpha: float | None) -> None:
    name = MODEL_NAME.get(model_id, model_id)
    color = MODEL_COLOR.get(model_id, "#888")
    fr = model_factor_rows(model_id)

    if weight_in_alpha is not None:
        badge = (f'<span class="ma-pill" style="background:{hex_rgba(color,0.18)};'
                 f'border:1px solid {hex_rgba(color,0.5)};color:{color};">'
                 f'{weight_in_alpha*100:.1f}% of {ALPHA_NAME}</span>')
    else:
        badge = ('<span class="ma-pill" style="background:rgba(148,163,184,0.15);'
                 'border:1px solid rgba(148,163,184,0.4);color:#94a3b8;">standalone</span>')

    header = (f'<div style="display:flex;align-items:center;gap:12px;">'
              f'<span style="width:12px;height:12px;border-radius:3px;background:{color};'
              f'display:inline-block;"></span>'
              f'<span style="font-size:1.05rem;font-weight:600;">{name}</span>'
              f'<span class="ma-mut">{model_id} · {len(fr)} factors</span>'
              f'{badge}</div>')

    with st.expander(f"{name}  ·  {len(fr)} factors", expanded=False):
        st.markdown(header, unsafe_allow_html=True)

        col_chart, col_tbl = st.columns([1, 1.45])

        # ---- weight bars, coloured by applicability, ↑/↓ = direction ----
        with col_chart:
            plot_df = fr.iloc[::-1]
            ylabels = [f"{'↑' if d > 0 else '↓'} {n}"
                       for n, d in zip(plot_df["factor_name"], plot_df["direction"])]
            bar = go.Figure(
                go.Bar(
                    x=plot_df["weight"], y=ylabels, orientation="h",
                    marker_color=[SECTOR_COLOR.get(a, SECTOR_FALLBACK) for a in plot_df["applies"]],
                    customdata=plot_df["applies"],
                    hovertemplate="<b>%{y}</b><br>weight %{x:.2f}<br>%{customdata}<extra></extra>",
                )
            )
            bar.update_layout(
                template="plotly_dark", height=max(190, 34 * len(fr)),
                margin=dict(l=10, r=10, t=6, b=6),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Weight within model", font=dict(size=11),
                showlegend=False,
            )
            st.plotly_chart(bar, use_container_width=True, key=f"bar_{model_id}")
            st.markdown(
                '<span class="ma-mut">↑ higher is better · ↓ lower is better. '
                'Bar colour = which companies the factor applies to.</span>',
                unsafe_allow_html=True,
            )

        # ---- factor formulation table ----
        with col_tbl:
            rows_html = []
            for _, r in fr.iterrows():
                fid = r["factor_id"]
                meta = fmeta.loc[fid] if fid in fmeta.index else None
                formula = str(meta["formula"]) if meta is not None and not pd.isna(meta.get("formula")) else "—"
                applies = r["applies"]
                sec_col = SECTOR_COLOR.get(applies, SECTOR_FALLBACK)
                tag = (f'<span class="ma-tag" style="color:{sec_col};'
                       f'border-color:{hex_rgba(sec_col,0.5)};">'
                       f'{applies}</span>')
                arrow = ("↑" if r["direction"] > 0 else "↓")
                rows_html.append(
                    f'<tr><td><b>{arrow} {r["factor_name"]}</b><br>'
                    f'<span class="ma-formula">{formula}</span></td>'
                    f'<td class="num">{r["weight"]:.2f}</td>'
                    f'<td>{tag}</td></tr>'
                )
            table = (
                '<table class="ma-tbl"><thead><tr>'
                '<th>Factor &amp; formula</th><th style="text-align:right;">Weight</th>'
                '<th>Applies to</th></tr></thead><tbody>'
                + "".join(rows_html) + "</tbody></table>"
            )
            st.markdown(table, unsafe_allow_html=True)


st.markdown(
    '<div class="ma-step"><div class="dot">2</div>'
    '<h2 style="margin:0;">Inside each model — factor weights & formulas</h2></div>',
    unsafe_allow_html=True,
)
st.caption(
    "Each base model is itself a weighted blend of factors. Open a model to see exactly "
    "which factors it uses, how they are weighted, whether higher or lower is better, and "
    f"the formula behind every one. Models are ordered by their weight in {ALPHA_NAME}."
)

for _, r in alpha_rows.iterrows():
    render_model(r["child_id"], float(r["w"]))

if standalone_ids:
    st.divider()
    st.markdown(
        '<div class="ma-step"><div class="dot">3</div>'
        f'<h2 style="margin:0;">Standalone models — not in {ALPHA_NAME}</h2></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"These models are computed and tracked but are not part of the {ALPHA_NAME} blend. "
        "They are used directly by specific strategies and/or as risk factors in the Barra model."
    )
    for mid in standalone_ids:
        render_model(mid, None)

# ===========================================================================
# STEP 4 — Factor glossary: every factor, its formula and raw inputs
# ===========================================================================
st.divider()
st.markdown(
    '<div class="ma-step"><div class="dot">4</div>'
    '<h2 style="margin:0;">Factor glossary — formulas & raw inputs</h2></div>',
    unsafe_allow_html=True,
)
st.caption(
    "The bottom layer: every factor in the library with its formula, the raw financial "
    "line items it consumes, its category and signal direction. This is where the chain "
    "finally touches the underlying accounting data."
)

q = st.text_input("Filter factors", placeholder="search by name, formula, category or input…").strip().lower()

gl = fref.copy()
gl["direction_lbl"] = gl["direction"].map(lambda d: "↑ higher better" if d > 0 else "↓ lower better")
show_cols = ["factor_name", "category", "direction_lbl", "formula", "constituents", "sector_type"]
gl = gl[show_cols].rename(columns={
    "factor_name": "Factor", "category": "Category", "direction_lbl": "Signal",
    "formula": "Formula", "constituents": "Raw inputs", "sector_type": "Scope",
})
if q:
    mask = gl.apply(lambda row: q in " ".join(str(v).lower() for v in row.values), axis=1)
    gl = gl[mask]

st.dataframe(gl, use_container_width=True, hide_index=True, height=560)
st.caption(f"{len(gl)} factors shown. All values read live from factors_reference.csv.")
