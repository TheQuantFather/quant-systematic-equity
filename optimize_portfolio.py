"""
optimize_portfolio.py — Mean-variance portfolio optimizer.

Reads strategy parameters from data/strategy_params.xlsx, runs CVXPY
optimization, and saves results to data/portfolio_output/.

Usage:
    python optimize_portfolio.py                         # all active strategies
    python optimize_portfolio.py --strategy core_active  # single strategy
    python optimize_portfolio.py --list                  # list available strategies

Objectives
----------
maximize_alpha      Benchmark-aware: maximize alpha subject to tracking-error and
                    active-weight constraints. Requires a benchmark index or file.

maximize_sharpe     Absolute return: maximize Sharpe via Charnes-Cooper transform.
                    No benchmark required.

minimize_variance   Pure risk minimisation: no alpha signal, just minimize w'Σw.
                    Good for capital preservation / low-vol mandates.

Risk model: Barra factor model (risk.db factor_covariance / factor_exposures /
            idiosyncratic_vars).
"""

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# MOSEK bootstrap — ensure Python 3.13 can load MOSEK 10.x (which ships .so
# files only for 3.7–3.12).  We symlink the 3.12 extension as 3.13 once.
# Harmless when Python ≤3.12 or when MOSEK 11+ (with native 3.13 wheel) is
# installed.
# ---------------------------------------------------------------------------
def _ensure_mosek_symlink() -> None:
    try:
        import mosek  # noqa: F401 — already works

        return
    except ImportError:
        pass
    ver = sys.version_info
    if ver < (3, 13):
        return
    try:
        import importlib.util

        spec = importlib.util.find_spec("mosek")
        if spec is None:
            return
        pkg_dir = Path(spec.submodule_search_locations[0])
        src = pkg_dir / "_msk.cpython-312-darwin.so"
        dst = pkg_dir / f"_msk.cpython-{ver.major}{ver.minor}-darwin.so"
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
    except Exception:
        pass


_ensure_mosek_symlink()

import db
from classification.register import (
    CLASSIFICATION_VIEWS,
    LEVELS,
    UNCLASSIFIED,
    BusinessLabels,
    view_labels,
)
from config import (
    BENCHMARK_DIR,
    CONSTITUENTS_DB,
    INCUBATION_PORTFOLIO_ID,
    MODELS_DB,
    OUTPUT_DIR,
    PARAMS_FILE,
    RETURNS_DB,
    RISK_DB,
    UNIVERSE_DB,
)
from portfolio_analytics import load_latest_positions
from securities.data import (
    company_observations,
    feed_identifiers,
    metadata_by_sid,
    returns_by_sid,
    validate_research_rows,
)
from securities.universe import load_clean_universe
from utils import get_db, get_logger, isin_to_sid_map, sid_to_isin_map

log = get_logger("optimize_portfolio")


# ── Excel params ──────────────────────────────────────────────────────────────


def _date_is_latest(raw: object) -> bool:
    val = str(raw or "").strip().lower()
    return val in ("", "latest", "auto")


def _latest_alpha_date(model_ids: list[str]) -> str:
    if not model_ids:
        raise ValueError("Cannot resolve alpha_date=latest without alpha models.")

    placeholders = ",".join("?" * len(model_ids))
    with get_db(MODELS_DB) as conn:
        row = conn.execute(
            f"""
            SELECT data_date
            FROM models
            WHERE model_id IN ({placeholders})
            GROUP BY data_date
            HAVING COUNT(DISTINCT model_id) = ?
            ORDER BY data_date DESC
            LIMIT 1
            """,
            [*model_ids, len(model_ids)],
        ).fetchone()

    if row is None or not row[0]:
        raise ValueError(f"No model snapshot found for alpha models: {', '.join(model_ids)}")
    return str(row[0])


def _latest_risk_date() -> str:
    with get_db(RISK_DB) as conn:
        row = conn.execute("SELECT MAX(snapshot_date) FROM factor_covariance").fetchone()
    if row is None or not row[0]:
        raise ValueError("No Barra risk snapshot found.")
    return str(row[0])


def _param_str(row: pd.Series, key: str, default: str = "") -> str:
    value = row.get(key, default)
    if pd.isna(value):
        return default
    text = str(value).strip()
    if text.lower() == "nan":
        return default
    return text


def load_strategy_params(strategy_id: str | None = None) -> list[dict]:
    xl = pd.ExcelFile(PARAMS_FILE)

    strats = pd.read_excel(xl, sheet_name="Strategies", dtype=str)
    strats = strats[strats["active"].str.strip().str.upper() == "TRUE"]
    if strategy_id:
        strats = strats[strats["strategy_id"].str.strip() == strategy_id]
    if strats.empty:
        raise ValueError(f"No active strategy found: {strategy_id!r}")

    cons_df = pd.read_excel(xl, sheet_name="Constraints", dtype=str)
    alpha_df = pd.read_excel(xl, sheet_name="Alpha_Weights", dtype=str)

    result = []
    for _, row in strats.iterrows():
        sid = row["strategy_id"].strip()

        c_rows = cons_df[cons_df["strategy_id"].str.strip() == sid]
        constraints = {}
        for _, c in c_rows.iterrows():
            if str(c["enabled"]).strip().upper() != "TRUE":
                continue
            name = c["constraint"].strip()
            val = str(c["value"]).strip()
            try:
                constraints[name] = float(val)
            except ValueError:
                upper = val.upper()
                if upper in ("TRUE", "FALSE"):
                    constraints[name] = upper == "TRUE"
                else:
                    constraints[name] = val  # keep as string (e.g. excluded_tickers)

        a_rows = alpha_df[alpha_df["strategy_id"].str.strip() == sid]
        alpha_weights = {}
        for _, a in a_rows.iterrows():
            alpha_weights[a["model_id"].strip()] = float(a["weight"])

        benchmark_file = _param_str(row, "benchmark_file")
        benchmark_index = _param_str(row, "benchmark_index")
        universe_index = _param_str(row, "universe_index")
        objective = _param_str(row, "objective", "maximize_alpha")
        try:
            risk_aversion = float(_param_str(row, "risk_aversion", "0") or 0.0)
        except ValueError:
            risk_aversion = 0.0
        alpha_date_raw = row.get("alpha_date", "")
        risk_date_raw = row.get("risk_date", "")
        alpha_date = (
            _latest_alpha_date(list(alpha_weights))
            if _date_is_latest(alpha_date_raw)
            else str(alpha_date_raw).strip()
        )
        risk_date = (
            _latest_risk_date() if _date_is_latest(risk_date_raw) else str(risk_date_raw).strip()
        )

        result.append(
            {
                "strategy_id": sid,
                "name": row["name"].strip(),
                "benchmark_file": benchmark_file,
                "benchmark_index": benchmark_index,
                "universe_index": universe_index,
                "alpha_date": alpha_date,
                "risk_date": risk_date,
                "solver": row["solver"].strip(),
                "objective": objective,
                "risk_aversion": risk_aversion,
                "investable_universe": row["investable_universe"].strip(),
                "constraints": constraints,
                "alpha_weights": alpha_weights,
            }
        )
    return result


# ── Data loaders ─────────────────────────────────────────────────────────────


def load_benchmark(benchmark_file: str) -> pd.DataFrame:
    path = BENCHMARK_DIR / benchmark_file
    df = pd.read_csv(path, skiprows=2)
    df = df.dropna(subset=["Ticker"])
    df = df[df["Asset Class"].str.strip() == "Equity"]
    df["weight"] = pd.to_numeric(df["Weight (%)"], errors="coerce") / 100.0
    df = df.dropna(subset=["weight"])
    df = df.rename(columns={"Ticker": "ticker", "Sector": "benchmark_sector"})
    df["weight"] /= df["weight"].sum()
    return df[["ticker", "benchmark_sector", "weight"]].copy()


def load_index_benchmark(index_name: str, snapshot_date: str) -> tuple[pd.DataFrame, dict]:
    result = load_clean_universe(index_name, snapshot_date, mode="live", tradable_only=False)
    members = result.tradable
    if members.empty:
        return pd.DataFrame(columns=["isin", "ticker", "benchmark_sector", "weight"]), {
            "benchmark_index": index_name,
            "benchmark_source_snapshot_date": result.source_snapshot_date,
            "benchmark_tradable": 0,
            "benchmark_blocked": int((~result.members["is_tradable"]).sum()),
        }

    df = members[["isin", "ticker", "gics_sector", "raw_weight"]].copy()
    df = df.rename(columns={"gics_sector": "benchmark_sector", "raw_weight": "weight"})
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    if df["weight"].sum() > 0:
        df["weight"] /= df["weight"].sum()

    meta = {
        "benchmark_index": index_name,
        "benchmark_source_snapshot_date": result.source_snapshot_date,
        "benchmark_tradable": int(len(members)),
        "benchmark_blocked": int((~result.members["is_tradable"]).sum()),
        "identity_aliases": {
            str(row["isin"]): str(row["mapped_from_isin"])
            for _, row in members.iterrows()
            if str(row.get("mapped_from_isin") or "").strip()
        },
    }
    return df, meta


def load_universe_metadata() -> pd.DataFrame:
    with get_db(UNIVERSE_DB) as conn:
        df = company_observations(conn)
        df["sid"] = df["isin"].map(isin_to_sid_map(conn)).astype("Int64")
        issuer_by_sid = dict(conn.execute("SELECT security_id, issuer_id FROM security_master"))
        df["issuer_id"] = df["sid"].map(issuer_by_sid).astype("Int64")
    return df


def load_latest_market_caps(
    meta_df: pd.DataFrame, investable: list, *, by_sid: bool = False
) -> dict:
    """Return latest raw-close × latest reported shares outstanding by ISIN."""
    if not investable:
        return {}

    if by_sid:
        with get_db(RETURNS_DB) as conn, get_db(UNIVERSE_DB) as master:
            px = returns_by_sid(conn, master, value="close", sids=investable)
        px = px[px["close"] > 0].sort_values("date").drop_duplicates("sid", keep="last")
        with get_db(CONSTITUENTS_DB) as conn:
            shares = pd.read_sql_query(
                f"SELECT sid, constituent_value, publish_date FROM constituents "
                f"WHERE sid IN ({','.join('?' for _ in investable)}) "
                "AND constituent_id='B3C4D5E6' AND constituent_value>0 "
                "ORDER BY publish_date",
                conn,
                params=investable,
            )
        shares = shares.drop_duplicates("sid", keep="last")
        caps = px.merge(shares, on="sid")
        return dict(zip(caps["sid"], caps["close"] * caps["constituent_value"]))

    ph = ",".join("?" * len(investable))
    with get_db(RETURNS_DB) as conn:
        px = pd.read_sql(
            f"""
            SELECT r.isin, r.close
            FROM returns r
            JOIN (
                SELECT isin, MAX(date) AS max_date
                FROM returns
                WHERE isin IN ({ph}) AND close IS NOT NULL AND close > 0
                GROUP BY isin
            ) latest
              ON latest.isin = r.isin AND latest.max_date = r.date
            """,
            conn,
            params=investable,
        )
    if px.empty:
        return {}

    meta = meta_df[meta_df["isin"].isin(investable)][["isin"]].copy()
    isin_sid = isin_to_sid_map()
    meta["sid"] = meta["isin"].map(isin_sid)
    sids = sorted({int(s) for s in meta["sid"].dropna()})
    if not sids:
        return {}

    share_ids = (
        "B3C4D5E6",
    )  # IIG88888 excluded: collides with InterestAndDividendIncome in EDGAR concept map
    sid_ph = ",".join("?" * len(sids))
    cid_ph = ",".join("?" * len(share_ids))
    with get_db(CONSTITUENTS_DB) as conn:
        shares_raw = pd.read_sql(
            f"""
            SELECT sid, constituent_value, publish_date
            FROM constituents
            WHERE sid IN ({sid_ph})
              AND constituent_id IN ({cid_ph})
              AND constituent_value IS NOT NULL
              AND constituent_value > 0
            ORDER BY sid, publish_date
            """,
            conn,
            params=sids + list(share_ids),
        )
    if shares_raw.empty:
        return {}

    latest_shares = shares_raw.groupby("sid", as_index=False).tail(1)
    shares_by_sid = dict(zip(latest_shares["sid"], latest_shares["constituent_value"]))
    meta["shares"] = meta["sid"].map(shares_by_sid)
    out = meta.merge(px, on="isin", how="inner").dropna(subset=["shares", "close"])
    out["market_cap"] = out["shares"].astype(float) * out["close"].astype(float)
    return dict(zip(out["isin"], out["market_cap"]))


def map_tickers_to_isins(tickers: list[str]) -> dict[str, str]:
    if not tickers:
        return {}
    with get_db(UNIVERSE_DB) as conn:
        return feed_identifiers(conn, set(tickers))


def load_alpha_scores(model_id: str, date: str, *, by_sid: bool = False) -> dict:
    with get_db(MODELS_DB) as conn:
        rows = conn.execute(
            "SELECT sid, model_value_z FROM models WHERE model_id=? AND data_date=?",
            (model_id, date),
        ).fetchall()
    mapping = sid_to_isin_map()
    frame = pd.DataFrame(rows, columns=["sid", "score"])
    frame["data_date"] = date
    frame["model_id"] = model_id
    frame = validate_research_rows(
        frame, ["data_date", "model_id", "sid"], mapping, context="models alpha scores"
    ).dropna(subset="score")
    labels = frame["sid"] if by_sid else frame["sid"].map(mapping)
    return dict(zip(labels, frame["score"]))


def load_blended_alpha(alpha_weights: dict[str, float], date: str, *, by_sid: bool = False) -> dict:
    """Load and blend alpha z-scores from one or more models."""
    if not alpha_weights:
        return {}
    total_w = sum(alpha_weights.values())
    blended: dict[str, float] = {}
    for model_id, w in alpha_weights.items():
        scores = load_alpha_scores(model_id, date, by_sid=by_sid)
        norm_w = w / total_w
        for isin, z in scores.items():
            blended[isin] = blended.get(isin, 0.0) + z * norm_w
    return blended


def _latest_barra_date() -> str | None:
    """Return the most recent snapshot_date in risk.db (Barra tables), or None if unavailable."""
    try:
        with get_db(RISK_DB) as conn:
            row = conn.execute("SELECT MAX(snapshot_date) FROM factor_covariance").fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def load_barra_universe(barra_date: str, *, by_sid: bool = False) -> list:
    """Return security_ids with Barra factor exposure coverage for a snapshot."""
    with get_db(RISK_DB) as conn:
        row = conn.execute(
            "SELECT factor_names FROM factor_covariance WHERE snapshot_date=?",
            (barra_date,),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"No Barra factor covariance for {barra_date}. Run create_barra.py first."
            )
        factor_names = json.loads(row[0])
        factor_id = "market" if "market" in factor_names else factor_names[0]
        rows = conn.execute(
            "SELECT sid FROM factor_exposures WHERE snapshot_date=? AND factor_id=?",
            (barra_date, factor_id),
        ).fetchall()
    if not rows:
        raise ValueError(f"No Barra factor exposures for {barra_date}. Run create_barra.py first.")
    mapping = sid_to_isin_map()
    frame = pd.DataFrame(rows, columns=["sid"])
    frame["snapshot_date"] = barra_date
    frame["factor_id"] = factor_id
    frame = validate_research_rows(
        frame, ["snapshot_date", "sid", "factor_id"], mapping, context="factor_exposures"
    )
    labels = frame["sid"] if by_sid else frame["sid"].map(mapping)
    return sorted(labels)


def load_barra_L(
    barra_date: str,
    investable: list,
    return_exposures: bool = False,
    *,
    by_sid: bool = False,
) -> np.ndarray | None | tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build stacked-L matrix for the Barra risk model.

    Returns L_barra of shape (N, K+N) such that
        ||L_barra.T @ w||² = w' (X F X' + Δ) w = w' Σ_barra w.
    Drop-in replacement for Cholesky L in all cp.norm(L.T @ w, 2) expressions.
    Returns None if risk.db has no Barra snapshot for barra_date.

    When ``return_exposures=True`` returns ``(L, X, factor_names)`` where ``X`` is
    the raw (N×K) factor-exposure matrix and ``factor_names`` its column labels —
    used to impose linear exposure constraints (e.g. beta-neutrality via the
    ``beta_60d`` column). Returns ``None`` unchanged on failure.

    Exposures/variances come from db.load_barra_components, whose sid seam already
    maps a row stored under a superseded ISIN to the security's current ISIN — so
    the old identity-alias routing is no longer needed (investables are current).
    """
    try:
        comps = db.load_barra_components(barra_date, by_sid=by_sid)
        if comps is None:
            return None
        F, factor_names, X_df, delta_s = comps
        F = np.array(F, dtype=np.float64, copy=True)  # may spectral-floor; keep cache clean
        N = len(investable)

        # Factor exposures for the investable names (current ISINs).
        missing_x = [isin for isin in investable if isin not in X_df.index]
        if missing_x:
            sample = ", ".join(map(str, missing_x[:10]))
            more = "" if len(missing_x) <= 10 else f", ... +{len(missing_x) - 10}"
            log.warning(
                "Barra snapshot %s is missing factor exposures for %d/%d investable names "
                "(%s%s); Barra risk unavailable.",
                barra_date,
                len(missing_x),
                N,
                sample,
                more,
            )
            return None
        X = X_df.reindex(index=investable, columns=factor_names, fill_value=0.0).to_numpy(
            dtype=np.float64
        )

        # Idiosyncratic variances (annualised). Missing names receive the snapshot
        # median instead of a hardcoded 20% annual vol so coverage holes do not look
        # artificially low-risk. The median query reads no ISIN column.
        with get_db(RISK_DB) as conn:
            all_d = conn.execute(
                "SELECT idio_var FROM idiosyncratic_vars WHERE snapshot_date=?",
                (barra_date,),
            ).fetchall()
        snapshot_delta = [
            float(r[0]) for r in all_d if r[0] is not None and np.isfinite(float(r[0]))
        ]
        delta_default = float(np.median(snapshot_delta)) if snapshot_delta else 0.04
        delta = delta_s.reindex(investable).to_numpy(dtype=np.float64)
        missing_mask = ~np.isfinite(delta)
        if missing_mask.any():
            missing_delta = [investable[i] for i in np.nonzero(missing_mask)[0]]
            delta = np.where(missing_mask, delta_default, delta)
            sample = ", ".join(map(str, missing_delta[:10]))
            more = "" if len(missing_delta) <= 10 else f", ... +{len(missing_delta) - 10}"
            log.warning(
                "Barra snapshot %s is missing idiosyncratic variance for %d/%d names "
                "(%s%s); using snapshot median %.6f.",
                barra_date,
                len(missing_delta),
                N,
                sample,
                more,
                delta_default,
            )

        # Cholesky of factor covariance (apply spectral floor if needed)
        try:
            L_F = np.linalg.cholesky(F)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(F)
            F = eigvecs @ np.diag(np.maximum(eigvals, 1e-6)) @ eigvecs.T
            L_F = np.linalg.cholesky(F)

        A = L_F.T @ X.T  # (K, N)
        B = np.diag(np.sqrt(np.maximum(delta, 0)))  # (N, N)
        L_barra = np.vstack([A, B]).T  # (N, K+N)
        if return_exposures:
            return L_barra, X, factor_names
        return L_barra

    except Exception as exc:
        log.warning("Barra load failed (%s).", exc)
        return None


def _variance(w: np.ndarray, Sigma, L: np.ndarray) -> float:
    """
    Portfolio variance w' Σ w.
    When Sigma is None (Barra mode) uses ||L.T @ w||² which equals w' Σ_barra w.
    """
    if Sigma is not None:
        return float(w @ Sigma @ w)
    return float(np.sum((L.T @ w) ** 2))


def _covariance(a: np.ndarray, b: np.ndarray, Sigma, L: np.ndarray) -> float:
    """
    Portfolio covariance a' Σ b.
    When Sigma is None (Barra mode) uses (L.T @ a)'(L.T @ b).
    """
    if Sigma is not None:
        return float(a @ Sigma @ b)
    return float((L.T @ a) @ (L.T @ b))


# ── Shared setup helpers ──────────────────────────────────────────────────────


def _load_clean_investable(index_name: str, snapshot_date: str) -> tuple[list[str], dict]:
    result = load_clean_universe(index_name, snapshot_date, mode="live", tradable_only=False)
    tradable = result.tradable
    aliases = {
        str(row["isin"]): str(row["mapped_from_isin"])
        for _, row in tradable.iterrows()
        if str(row.get("mapped_from_isin") or "").strip()
    }
    meta = {
        "universe_index": index_name,
        "universe_source_snapshot_date": result.source_snapshot_date,
        "universe_tradable": int(len(tradable)),
        "universe_blocked": int((~result.members["is_tradable"]).sum()),
        "identity_aliases": aliases,
    }
    return sorted(set(tradable["isin"].dropna().astype(str))), meta


def _build_investable(
    strategy,
    bm_df,
    risk_isins,
    risk_isin_idx,
    universe_isins=None,
    identity_aliases: dict[str, str] | None = None,
):
    identity_aliases = identity_aliases or {}

    def _has_risk(isin: str) -> bool:
        alias = identity_aliases.get(isin)
        return isin in risk_isin_idx or (alias is not None and alias in risk_isin_idx)

    if strategy["investable_universe"] == "benchmark_only":
        investable = sorted({i for i in bm_df["isin"] if _has_risk(i)})
    elif universe_isins is not None:
        investable = sorted({i for i in universe_isins if _has_risk(i)})
    else:
        investable = sorted(set(risk_isins))

    c = strategy.get("constraints", {})
    excluded_tickers = {
        s.strip().upper() for s in str(c.get("excluded_tickers", "") or "").split("|") if s.strip()
    }
    if excluded_tickers:
        meta_df = load_universe_metadata()
        ticker_excluded_isins = set(
            meta_df.loc[
                meta_df["ticker"].str.upper().isin(excluded_tickers),
                "isin",
            ]
        )
        investable = [isin for isin in investable if isin not in ticker_excluded_isins]
    return investable


def _with_business_labels(meta: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Add in-house business_* label columns (by sid) when CLASSIFICATION_SOURCE='inhouse'.

    Vendor mode returns ``meta`` unchanged, so every vendor path reads GICS/SimFin.
    """
    if not BusinessLabels.enabled():
        return meta
    sids = meta["sid"].dropna().astype(int).unique()
    labels = BusinessLabels().labels(as_of, sids).set_index("sid")
    out = meta.drop(columns=[c for c in meta.columns if c.startswith("business_")])
    out["business_status"] = out["sid"].map(labels["status"].to_dict()).fillna("missing")
    for level in LEVELS:
        out[f"business_{level}"] = out["sid"].map(labels[level].to_dict()).fillna(UNCLASSIFIED)
    return out


@dataclass
class ClassificationExtras:
    """In-house constraints beyond sector/industry.

    ``levels`` are (constraint key, groups, membership matrix, benchmark targets,
    tolerance) for industry groups and sub-industries, with benchmark targets from the
    full universe. ``blocked`` names
    (unresolved/missing classification) are held at zero weight.
    """

    levels: list[tuple[str, list[str], np.ndarray, np.ndarray, float]]
    blocked: np.ndarray

    def apply(self, cvx: list, var, scale) -> None:
        if self.blocked.any():
            cvx.append(var[np.flatnonzero(self.blocked)] == 0)
        for _key, _names, B, targets, tol in self.levels:
            for g in range(len(B)):
                weight = B[g] @ var
                cvx.append(weight <= (targets[g] + tol) * scale)
                cvx.append(weight >= max(0.0, targets[g] - tol) * scale)


_EXTRA_LEVELS = {
    "max_industry_group_active_weight": "business_industry_group",
    "max_sub_industry_active_weight": "business_sub_industry",
}


def _classification_extras(
    investable: list, meta: pd.DataFrame, b: np.ndarray, c: dict
) -> ClassificationExtras | None:
    """Extras for in-house metadata; None for vendor metadata (no change to its paths)."""
    if "business_status" not in meta.columns:
        if any(c.get(key) is not None for key in _EXTRA_LEVELS):
            raise ValueError(
                "Industry-group/sub-industry limits need CLASSIFICATION_SOURCE='inhouse'"
            )
        return None
    rows = meta.reindex(investable)
    blocked = rows["business_status"].ne("accepted").to_numpy()
    b = np.asarray(b, dtype=float)
    levels = []
    for key, column in _EXTRA_LEVELS.items():
        if c.get(key) is None:
            continue
        labels = rows[column].fillna(UNCLASSIFIED)
        names = sorted(set(labels) - {UNCLASSIFIED})
        B = np.array([labels.eq(n).to_numpy(dtype=float) for n in names]).reshape(len(names), -1)
        levels.append((key, names, B, B @ b, float(c[key])))
    return ClassificationExtras(levels, blocked)


def _output_labels(investable: list, gics_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Display labels per ISIN for the results file, under their metadata column names.

    Always GICS sector / SimFin industry; in-house mode adds the four business_*
    levels. Constraints read the active scheme (_sector_industry_matrices), never these.
    """
    labels = view_labels(gics_df.reindex(investable))
    labels.columns = [CLASSIFICATION_VIEWS[scheme][level] for scheme, level in labels.columns]
    return labels.astype(object).to_dict("index")


def _sector_industry_matrices(investable, gics_df):
    # In-house metadata carries business_* columns; vendor metadata never does.
    sector_column, industry_column = (
        ("business_sector", "business_industry")
        if "business_sector" in gics_df.columns
        else ("gics_sector", "simfin_industry")
    )

    def _sector(isin):
        try:
            v = gics_df.loc[isin, sector_column]
            return str(v).strip() if pd.notna(v) and str(v).strip() else "Unknown"
        except KeyError:
            return "Unknown"

    def _industry(isin):
        try:
            v = gics_df.loc[isin, industry_column]
            return str(v).strip() if pd.notna(v) and str(v).strip() else "Unknown"
        except KeyError:
            return "Unknown"

    sectors = sorted({_sector(i) for i in investable})
    industries = sorted({_industry(i) for i in investable})

    B_sector = np.zeros((len(sectors), len(investable)))
    B_ind = np.zeros((len(industries), len(investable)))
    for j, isin in enumerate(investable):
        B_sector[sectors.index(_sector(isin)), j] = 1.0
        B_ind[industries.index(_industry(isin)), j] = 1.0

    return sectors, industries, B_sector, B_ind, _sector, _industry


def _issuer_matrices(
    investable: list,
    metadata: pd.DataFrame,
    *,
    required: bool = True,
) -> tuple[list[int] | None, np.ndarray | None, Callable[[str | int], int] | None]:
    """Use master issuer IDs for either ISIN- or SID-indexed securities.

    Incomplete identities are an error when an issuer cap is requested. Without
    that constraint, return no membership rather than inventing issuer groups.
    """
    if not metadata.index.is_unique:
        raise ValueError("Issuer metadata must have one row per security.")
    raw = metadata.reindex(index=investable, columns=["issuer_id"])["issuer_id"]
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid = numeric.isna() | ~np.isfinite(numeric) | numeric.le(0) | numeric.mod(1).ne(0)
    if invalid.any():
        if required:
            raise ValueError(
                "max_issuer_weight requires security-master issuer_id for every security; "
                f"missing or invalid for {raw.index[invalid].tolist()[:10]}"
            )
        return None, None, None

    issuer_by_security = dict(zip(investable, numeric.astype("int64")))
    issuers = sorted(set(issuer_by_security.values()))
    issuer_positions = {issuer_id: i for i, issuer_id in enumerate(issuers)}
    B_issuer = np.zeros((len(issuers), len(investable)))
    for j, security in enumerate(investable):
        B_issuer[issuer_positions[issuer_by_security[security]], j] = 1.0
    return issuers, B_issuer, issuer_by_security.__getitem__


def _market_cap_bucket_matrices(
    investable: list[str],
    market_caps: dict[str, float],
    c: dict,
) -> tuple[list[str], np.ndarray, dict[str, str]]:
    """Build large/mid/small bucket membership from dynamic market caps."""
    large_min = float(c.get("large_cap_min_market_cap", 10_000_000_000.0))
    mid_min = float(c.get("mid_cap_min_market_cap", 2_000_000_000.0))
    buckets = ["large", "mid", "small"]
    bucket_by_isin: dict[str, str] = {}

    for isin in investable:
        mcap = market_caps.get(isin)
        if mcap is None or not np.isfinite(mcap):
            bucket = "small"
        elif mcap >= large_min:
            bucket = "large"
        elif mcap >= mid_min:
            bucket = "mid"
        else:
            bucket = "small"
        bucket_by_isin[isin] = bucket

    B_cap = np.zeros((len(buckets), len(investable)))
    for j, isin in enumerate(investable):
        B_cap[buckets.index(bucket_by_isin[isin]), j] = 1.0
    return buckets, B_cap, bucket_by_isin


def _add_market_cap_constraints(cvx, var, scale, buckets, B_cap, c):
    """Apply absolute single-name and total weight caps by market-cap bucket."""
    cap_cfg = {
        "large": (
            c.get("max_large_cap_position", None),
            c.get("min_large_cap_weight", None),
            c.get("max_large_cap_weight", None),
        ),
        "mid": (
            c.get("max_mid_cap_position", None),
            c.get("min_mid_cap_weight", None),
            c.get("max_mid_cap_weight", None),
        ),
        "small": (
            c.get("max_small_cap_position", None),
            c.get("min_small_cap_weight", None),
            c.get("max_small_cap_weight", None),
        ),
    }

    for bucket, (max_pos, min_weight, max_weight) in cap_cfg.items():
        if bucket not in buckets:
            continue
        mask = B_cap[buckets.index(bucket)].astype(bool)
        if max_pos is not None and mask.any():
            cvx.append(var[mask] <= float(max_pos) * scale)
        if min_weight is not None:
            cvx.append(B_cap[buckets.index(bucket)] @ var >= float(min_weight) * scale)
        if max_weight is not None:
            cvx.append(B_cap[buckets.index(bucket)] @ var <= float(max_weight) * scale)


def _covariance_submatrix(risk_cov, risk_isin_idx, investable):
    idx = [risk_isin_idx[isin] for isin in investable]
    Sigma = risk_cov[np.ix_(idx, idx)]
    Sigma = (Sigma + Sigma.T) / 2.0
    eigmin = np.linalg.eigvalsh(Sigma).min()
    if eigmin < 1e-8:
        Sigma += (abs(eigmin) + 1e-8) * np.eye(len(investable))
    return Sigma, np.linalg.cholesky(Sigma)


def _add_benchmark_relative_group_constraints(
    cvx, var, scale, groups, B_group, benchmark, tolerance
):
    """
    Keep portfolio group weights within +/- tolerance of benchmark group weights.

    In Charnes-Cooper space `var` is y and `scale` is sum(y), so the benchmark
    target must be scaled by the same amount as absolute weight constraints.
    """
    if tolerance is None or benchmark is None:
        return
    benchmark = np.asarray(benchmark, dtype=float)
    if benchmark.size == 0 or benchmark.sum() <= 1e-10:
        return

    tol = float(tolerance)
    for g in range(len(groups)):
        bm_weight = float(B_group[g] @ benchmark)
        group_weight = B_group[g] @ var
        cvx.append(group_weight <= (bm_weight + tol) * scale)
        cvx.append(group_weight >= max(0.0, bm_weight - tol) * scale)


def _add_issuer_constraints(cvx, var, scale, issuers, B_issuer, c):
    """Apply absolute issuer-level caps across multiple securities."""
    max_iw = c.get("max_issuer_weight", None)
    if max_iw is None:
        return
    if issuers is None or B_issuer is None:
        raise ValueError("max_issuer_weight requires issuer membership data.")

    cap = float(max_iw)
    for issuer_idx in range(len(issuers)):
        cvx.append(B_issuer[issuer_idx] @ var <= cap * scale)


# ── Integer constraint helpers ────────────────────────────────────────────────

# Minimum assumed portfolio vol; used as a denominator for big-M bounds in the
# Charnes-Cooper (maximize_sharpe) formulation where the decision variable is
# y = w/σ_p rather than w directly.
_SIGMA_FLOOR = 0.05


def _has_integer_constraints(c: dict) -> bool:
    return (
        c.get("max_positions") is not None
        or c.get("min_positions") is not None
        or c.get("min_position_if_held") is not None
    )


def _ensure_mosek(strategy: dict) -> None:
    """Auto-promote solver to MOSEK when integer constraints are present."""
    if _has_integer_constraints(strategy["constraints"]):
        if strategy.get("solver", "CLARABEL").upper() != "MOSEK":
            log.info(
                "Integer constraints detected — switching solver to MOSEK (was %s)",
                strategy["solver"],
            )
            strategy["solver"] = "MOSEK"


# ── Objective 1: maximize_alpha (benchmark-aware) ─────────────────────────────


def _optimize_alpha_long_short(
    strategy,
    investable,
    alpha,
    L,
    sectors,
    B_sector,
    prev_weights_arr: np.ndarray | None = None,
    max_turnover: float | None = None,
    beta_exposure: np.ndarray | None = None,
    short_eligible: np.ndarray | None = None,
    extras: ClassificationExtras | None = None,
):
    """Dollar + beta-neutral long/short optimiser (maximize_alpha, no benchmark).

    Signed weights ``w = w_long − w_short`` (both legs ≥ 0):

        maximize  α·w  [− ½·risk_aversion·‖Lᵀw‖²]
        s.t.  Σw = net_exposure                         (dollar neutrality, default 0)
              Σw_long + Σw_short ≤ gross_exposure        (default 2.0; legs balanced)
              |β·w| ≤ beta_tolerance                     (beta neutrality)
              w_long ≤ max_long_position, w_short ≤ max_short_position
              w_short[¬short_eligible] = 0               (liquid short universe)
              cardinality / min_position_if_held via boolean z (MOSEK MIP)
              Σ|w − w_prev| ≤ max_turnover

    Sigma is implicit in the low-rank ``L`` (Barra), so risk terms reuse ‖Lᵀw‖.
    """
    c = strategy["constraints"]
    N = len(investable)

    net_exposure = float(c.get("net_exposure", 0.0))
    net_tol = float(c.get("net_exposure_tolerance", 0.0))
    gross_exposure = float(c.get("gross_exposure", 2.0))
    beta_tol = float(c.get("beta_tolerance", 0.05))
    max_long = float(c.get("max_long_position", 0.04))
    max_short = float(c.get("max_short_position", 0.04))
    leg_tol = float(c.get("leg_balance_tolerance", 0.05))

    wl = cp.Variable(N, nonneg=True, name="w_long")
    ws = cp.Variable(N, nonneg=True, name="w_short")
    w = wl - ws

    # Dollar (net) neutrality
    if net_tol > 0:
        cvx = [cp.sum(w) <= net_exposure + net_tol, cp.sum(w) >= net_exposure - net_tol]
    else:
        cvx = [cp.sum(w) == net_exposure]
    if extras is not None and extras.blocked.any():
        held_out = np.flatnonzero(extras.blocked)
        cvx += [wl[held_out] == 0, ws[held_out] == 0]
    # Gross exposure with balanced legs (pins ~half long / half short)
    half = gross_exposure / 2.0
    cvx += [
        cp.sum(wl) + cp.sum(ws) <= gross_exposure,
        cp.sum(wl) <= half + leg_tol,
        cp.sum(ws) <= half + leg_tol,
    ]
    # Per-name leg caps
    cvx += [wl <= max_long, ws <= max_short]
    # Liquid short universe: ineligible names cannot be shorted
    if short_eligible is not None:
        inelig = np.where(~np.asarray(short_eligible, dtype=bool))[0]
        if len(inelig):
            cvx.append(ws[inelig] == 0)
    # Beta neutrality
    if beta_exposure is not None:
        cvx += [beta_exposure @ w <= beta_tol, beta_exposure @ w >= -beta_tol]
    else:
        log.warning("Long/short: no beta exposures supplied — beta-neutral constraint skipped.")
    # Optional sector dollar-neutrality (off by default per dollar+beta design)
    if bool(c.get("sector_neutral", False)):
        sec_tol = float(c.get("sector_neutral_tolerance", 0.02))
        for s in range(len(sectors)):
            cvx += [B_sector[s] @ w <= sec_tol, B_sector[s] @ w >= -sec_tol]

    # Absolute portfolio vol cap (annualised, same L as risk model)
    max_vol = c.get("max_portfolio_vol", None)
    if max_vol is not None:
        cvx.append(cp.norm(L.T @ w, 2) <= float(max_vol))

    # Cardinality + minimum position, per leg (MIP; needs MOSEK)
    max_pos_n = c.get("max_positions")
    min_pos_n = c.get("min_positions")
    min_held = c.get("min_position_if_held")
    zl = zs = None
    if max_pos_n is not None or min_pos_n is not None or min_held is not None:
        zl = cp.Variable(N, boolean=True)
        zs = cp.Variable(N, boolean=True)
        cvx += [wl <= max_long * zl, ws <= max_short * zs, zl + zs <= 1]
        if max_pos_n is not None:
            cvx.append(cp.sum(zl) + cp.sum(zs) <= int(max_pos_n))
        if min_pos_n is not None:
            cvx.append(cp.sum(zl) + cp.sum(zs) >= int(min_pos_n))
        if min_held is not None:
            cvx += [wl >= min_held * zl, ws >= min_held * zs]

    # Two-way turnover vs previous (signed) weights
    if prev_weights_arr is not None and max_turnover is not None:
        cvx.append(cp.sum(cp.abs(w - prev_weights_arr)) <= max_turnover)

    risk_aversion = float(strategy.get("risk_aversion", 0.0) or 0.0)
    objective = alpha @ w
    if risk_aversion > 0.0:
        objective = objective - 0.5 * risk_aversion * cp.sum_squares(L.T @ w)
    prob = cp.Problem(cp.Maximize(objective), cvx)
    _solve(prob, strategy["solver"])

    weights = np.array(w.value)
    if zl is not None:
        held = (np.array(zl.value) > 0.5) | (np.array(zs.value) > 0.5)
        weights = np.where(held, weights, 0.0)

    gross = float(np.abs(weights).sum())
    net = float(weights.sum())
    port_vol = float(np.sqrt(max(0.0, _variance(weights, None, L))))
    port_beta = float(beta_exposure @ weights) if beta_exposure is not None else 0.0
    exp_alpha = float(alpha @ weights)
    n_long = int((weights > 1e-4).sum())
    n_short = int((weights < -1e-4).sum())
    info_ratio = exp_alpha / port_vol if port_vol > 0 else 0.0

    log.info(
        "L/S alpha: %+.4f | Gross: %.2f | Net: %+.3f | Beta: %+.3f | Vol: %.2f%% | Long: %d Short: %d",
        exp_alpha,
        gross,
        net,
        port_beta,
        port_vol * 100,
        n_long,
        n_short,
    )
    return weights, {
        "expected_alpha": round(exp_alpha, 4),
        "portfolio_vol": round(port_vol, 4),
        "active_risk": round(port_vol, 4),
        "beta": round(port_beta, 4),
        "gross": round(gross, 4),
        "net": round(net, 4),
        "n_long": n_long,
        "n_short": n_short,
        "info_ratio": round(info_ratio, 4),
    }


def _optimize_alpha(
    strategy,
    investable,
    alpha,
    b,
    Sigma,
    L,
    sectors,
    industries,
    B_sector,
    B_ind,
    cap_buckets,
    B_cap,
    issuers=None,
    B_issuer=None,
    prev_weights_arr: np.ndarray | None = None,
    max_turnover: float | None = None,
    beta_exposure: np.ndarray | None = None,
    short_eligible: np.ndarray | None = None,
    extras: ClassificationExtras | None = None,
):
    c = strategy["constraints"]
    N = len(investable)

    # Long/short (market-neutral) path — fully gated behind long_only=False so the
    # long-only optimiser below is byte-for-byte unchanged when the key is absent.
    if not bool(c.get("long_only", True)):
        return _optimize_alpha_long_short(
            strategy,
            investable,
            alpha,
            L,
            sectors,
            B_sector,
            prev_weights_arr=prev_weights_arr,
            max_turnover=max_turnover,
            beta_exposure=beta_exposure,
            short_eligible=short_eligible,
            extras=extras,
        )

    # Cash buffer: an explicit max_cash_weight allows the optimizer to hold cash
    # (weights sum in [1 - buffer, 1]) so a binding max_portfolio_vol can de-risk
    # by scaling down instead of distorting the alpha portfolio toward low-vol
    # names. Cash has zero alpha, so it is only used when a risk cap binds.
    max_cash = c.get("max_cash_weight", None)
    cash_buffer = float(max_cash) if max_cash is not None else None

    w = cp.Variable(N, name="weights")
    aw = w - b

    if cash_buffer is not None:
        cvx = [cp.sum(w) <= 1.0, cp.sum(w) >= 1.0 - cash_buffer, w >= 0]
    else:
        cvx = [cp.sum(w) == 1.0, w >= 0]
    max_saw = c.get("max_stock_active_weight", None)

    # ── Integer: cardinality + minimum position ────────────────────────────
    max_pos_n = c.get("max_positions")
    min_pos_n = c.get("min_positions")
    min_held = c.get("min_position_if_held")
    z = None
    if max_pos_n is not None or min_pos_n is not None or min_held is not None:
        z = cp.Variable(N, boolean=True)
        if max_pos_n is not None:
            cvx.append(cp.sum(z) <= int(max_pos_n))
        if min_pos_n is not None:
            cvx.append(cp.sum(z) >= int(min_pos_n))
        # w_i > 0 requires z_i = 1. The stock active cap already bounds w_i by
        # b_i + cap; using it (not 1) as the big-M tightens the relaxation MOSEK
        # branches on without changing the feasible portfolios.
        upper = np.minimum(1.0, b + float(max_saw)) if max_saw is not None else np.ones(N)
        cvx.append(w <= cp.multiply(upper, z))
        if min_held is not None:
            cvx.append(w >= min_held * z)  # if held, at least min_held

    # Active-weight constraints
    if max_saw is not None:
        cvx += [aw <= float(max_saw), aw >= -float(max_saw)]

    # Active-risk expression in factor (square-root) space; reused by both the
    # active-risk cap and the risk-aversion penalty so CVXPY builds it once.
    # L is the low-rank Barra factor/idio loading (or LW Cholesky), so this stays
    # an efficient SOCP term and never materialises the dense N×N covariance.
    Lt_aw = L.T @ aw

    max_ar = c.get("max_active_risk", None)
    if max_ar is not None:
        cvx.append(cp.norm(Lt_aw, 2) <= float(max_ar))

    # Absolute portfolio vol cap (annualised, same L as active risk). With a
    # cash buffer enabled this de-risks by scaling into cash; without one it
    # can only rotate the composition toward lower-vol names.
    max_vol = c.get("max_portfolio_vol", None)
    if max_vol is not None:
        cvx.append(cp.norm(L.T @ w, 2) <= float(max_vol))

    max_secaw = c.get("max_sector_active_weight", None)
    if max_secaw is not None:
        for s in range(len(sectors)):
            s_aw = B_sector[s] @ aw
            cvx += [s_aw <= max_secaw, s_aw >= -max_secaw]

    max_indaw = c.get("max_industry_active_weight", None)
    if max_indaw is not None:
        for g in range(len(industries)):
            g_aw = B_ind[g] @ aw
            cvx += [g_aw <= max_indaw, g_aw >= -max_indaw]

    _add_market_cap_constraints(cvx, w, 1.0, cap_buckets, B_cap, c)
    _add_issuer_constraints(cvx, w, 1.0, issuers, B_issuer, c)
    if extras is not None:
        extras.apply(cvx, w, 1.0)

    # Two-way turnover constraint: sum(|w - w_prev|) ≤ max_turnover
    if prev_weights_arr is not None and max_turnover is not None:
        cvx.append(cp.sum(cp.abs(w - prev_weights_arr)) <= max_turnover)

    # Objective. `risk_aversion` (Strategies sheet, default 0) turns the pure
    # linear alpha objective into a mean-variance objective in active space:
    #   maximize  α·w − ½·risk_aversion·(w−b)'Σ(w−b)
    # With risk_aversion=0 this is the legacy pure-linear alpha objective (loads
    # the highest point estimates to their caps, lets beta/active risk drift to
    # the max_active_risk boundary). With risk_aversion>0 the optimizer trades
    # alpha off against active variance, so noisy high-alpha/high-risk names are
    # pulled back and the book de-concentrates; the max_active_risk cap stays as
    # a backstop. Note: walk-forward backtests show a positive penalty *hurts*
    # the current ALP001 alpha (its edge is in the conviction tilts), so it is
    # left at 0 for live strategies — kept as an available knob.
    risk_aversion = float(strategy.get("risk_aversion", 0.0) or 0.0)
    objective = alpha @ w
    if risk_aversion > 0.0:
        objective = objective - 0.5 * risk_aversion * cp.sum_squares(Lt_aw)
    prob = cp.Problem(cp.Maximize(objective), cvx)
    _solve(prob, strategy["solver"])

    weights = np.array(w.value).clip(0)
    if z is not None:
        weights *= (np.array(z.value) > 0.5).astype(float)
    if cash_buffer is None:
        weights /= weights.sum()

    act = weights - b
    active_risk = float(np.sqrt(_variance(act, Sigma, L)))
    port_vol = float(np.sqrt(_variance(weights, Sigma, L)))
    bench_var = float(_variance(b, Sigma, L)) if b.sum() > 1e-10 else 0.0
    beta = float(_covariance(weights, b, Sigma, L) / bench_var) if bench_var > 1e-12 else 0.0
    exp_alpha = float(alpha @ weights)
    n_pos = int((weights > 1e-4).sum())
    info_ratio = exp_alpha / active_risk if active_risk > 0 else 0.0

    log.info(
        "Expected alpha: %+.4f | Active risk: %.2f%% | Portfolio vol: %.2f%% | "
        "Beta: %.2f | Info ratio: %.2f | Positions: %d",
        exp_alpha,
        active_risk * 100,
        port_vol * 100,
        beta,
        info_ratio,
        n_pos,
    )

    return weights, {
        "expected_alpha": round(exp_alpha, 4),
        "active_risk": round(active_risk, 4),
        "portfolio_vol": round(port_vol, 4),
        "beta": round(beta, 4),
        "info_ratio": round(info_ratio, 4),
    }


# ── Objective 2: maximize_sharpe (Charnes-Cooper) ────────────────────────────


def _optimize_sharpe(
    strategy,
    investable,
    alpha,
    b,
    Sigma,
    L,
    sectors,
    industries,
    B_sector,
    B_ind,
    cap_buckets,
    B_cap,
    issuers=None,
    B_issuer=None,
    prev_weights_arr: np.ndarray | None = None,
    max_turnover: float | None = None,
    extras: ClassificationExtras | None = None,
):
    """
    Charnes-Cooper: let y = w/σ_p.
    Maximize alpha @ y  s.t.  ||L.T @ y||₂ ≤ 1, y ≥ 0, per-stock/sector bounds.
    Recover weights:  w = y / sum(y),  σ_p = ||L.T @ w||₂.

    For MIP (cardinality / min position): binary z_i added with big-M bounds.
    The zero-out big-M is max_position / _SIGMA_FLOOR (tight bound on y_i).
    Minimum position uses a linear big-M relaxation of the bilinear term y_i ≥ p*t*z_i.
    """
    c = strategy["constraints"]
    N = len(investable)

    y = cp.Variable(N, name="scaled_weights", nonneg=True)
    t = cp.sum(y)  # proportional to 1/σ_p

    cvx = [cp.norm(L.T @ y, 2) <= 1]

    max_pos = c.get("max_position", 0.05)
    cvx.append(y <= max_pos * t)

    # ── Integer: cardinality + minimum position ────────────────────────────
    max_pos_n = c.get("max_positions")
    min_pos_n = c.get("min_positions")
    min_held = c.get("min_position_if_held")
    z = None
    if max_pos_n is not None or min_pos_n is not None or min_held is not None:
        z = cp.Variable(N, boolean=True)
        big_M = max_pos / _SIGMA_FLOOR  # upper bound on y_i when held
        if max_pos_n is not None:
            cvx.append(cp.sum(z) <= int(max_pos_n))
        if min_pos_n is not None:
            cvx.append(cp.sum(z) >= int(min_pos_n))
        cvx.append(y <= big_M * z)  # y_i > 0 requires z_i = 1
        if min_held is not None:
            # y_i >= min_held * t * z_i  linearised via big-M on (1-z_i):
            #   y_i >= min_held * t - (min_held/σ_floor) * (1 - z_i)
            M_lo = min_held / _SIGMA_FLOOR
            cvx.append(y >= min_held * t - M_lo * (1 - z))

    # Sector constraints (y-space: scale by t)
    _add_benchmark_relative_group_constraints(
        cvx, y, t, sectors, B_sector, b, c.get("max_sector_active_weight", None)
    )
    _add_market_cap_constraints(cvx, y, t, cap_buckets, B_cap, c)
    _add_issuer_constraints(cvx, y, t, issuers, B_issuer, c)
    if extras is not None:
        extras.apply(cvx, y, t)

    _add_benchmark_relative_group_constraints(
        cvx, y, t, industries, B_ind, b, c.get("max_industry_active_weight", None)
    )

    # Max portfolio vol: σ_p ≤ v  →  ||L.T @ y||₂ ≤ v * t
    max_vol = c.get("max_portfolio_vol", None)
    if max_vol is not None:
        cvx.append(cp.norm(L.T @ y, 2) <= max_vol * t)

    # Two-way turnover in Charnes-Cooper space:
    # sum(|w - w_prev|) ≤ T  ⟺  sum(|y - w_prev·t|) ≤ T·t
    if prev_weights_arr is not None and max_turnover is not None:
        cvx.append(cp.sum(cp.abs(y - prev_weights_arr * t)) <= max_turnover * t)

    prob = cp.Problem(cp.Maximize(alpha @ y), cvx)
    _solve(prob, strategy["solver"])

    y_val = np.array(y.value).clip(0)
    if z is not None:
        y_val *= (np.array(z.value) > 0.5).astype(float)
    if not np.isfinite(y_val).all() or y_val.sum() <= 1e-10:
        raise RuntimeError(
            "Optimization failed: solver returned no usable Sharpe solution "
            "(likely MIP time limit without incumbent)."
        )
    weights = y_val / y_val.sum()

    port_vol = float(np.sqrt(_variance(weights, Sigma, L)))
    exp_return = float(alpha @ weights)
    sharpe = exp_return / port_vol if port_vol > 0 else 0.0
    n_pos = int((weights > 1e-4).sum())

    log.info(
        "Expected return: %+.4f | Portfolio vol: %.2f%% | Sharpe: %.2f | Positions: %d",
        exp_return,
        port_vol * 100,
        sharpe,
        n_pos,
    )

    return weights, {
        "expected_alpha": round(exp_return, 4),
        "portfolio_vol": round(port_vol, 4),
        "active_risk": round(port_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "info_ratio": round(sharpe, 4),
    }


# ── Objective 3: minimize_variance ───────────────────────────────────────────


def _optimize_min_variance(
    strategy,
    investable,
    b,
    Sigma,
    L,
    sectors,
    industries,
    B_sector,
    B_ind,
    cap_buckets,
    B_cap,
    issuers=None,
    B_issuer=None,
    prev_weights_arr: np.ndarray | None = None,
    max_turnover: float | None = None,
    extras: ClassificationExtras | None = None,
):
    c = strategy["constraints"]
    N = len(investable)

    w = cp.Variable(N, name="weights")

    cvx = [cp.sum(w) == 1.0, w >= 0]

    max_pos = c.get("max_position", 0.05)

    # ── Integer: cardinality + minimum position ────────────────────────────
    max_pos_n = c.get("max_positions")
    min_pos_n = c.get("min_positions")
    min_held = c.get("min_position_if_held")
    z = None
    if max_pos_n is not None or min_pos_n is not None or min_held is not None:
        z = cp.Variable(N, boolean=True)
        if max_pos_n is not None:
            cvx.append(cp.sum(z) <= int(max_pos_n))
        if min_pos_n is not None:
            cvx.append(cp.sum(z) >= int(min_pos_n))
        cvx.append(w <= max_pos * z)  # zero-out + upper bound combined
        if min_held is not None:
            cvx.append(w >= min_held * z)
    else:
        cvx.append(w <= max_pos)

    # Sector constraints (w-space)
    _add_benchmark_relative_group_constraints(
        cvx, w, 1.0, sectors, B_sector, b, c.get("max_sector_active_weight", None)
    )
    _add_market_cap_constraints(cvx, w, 1.0, cap_buckets, B_cap, c)
    _add_issuer_constraints(cvx, w, 1.0, issuers, B_issuer, c)
    if extras is not None:
        extras.apply(cvx, w, 1.0)

    _add_benchmark_relative_group_constraints(
        cvx, w, 1.0, industries, B_ind, b, c.get("max_industry_active_weight", None)
    )

    # Two-way turnover constraint: sum(|w - w_prev|) ≤ max_turnover
    if prev_weights_arr is not None and max_turnover is not None:
        cvx.append(cp.sum(cp.abs(w - prev_weights_arr)) <= max_turnover)

    prob = cp.Problem(cp.Minimize(cp.sum_squares(L.T @ w)), cvx)
    _solve(prob, strategy["solver"])

    weights = np.array(w.value).clip(0)
    if z is not None:
        weights *= (np.array(z.value) > 0.5).astype(float)
    weights /= weights.sum()

    port_vol = float(np.sqrt(_variance(weights, Sigma, L)))
    n_pos = int((weights > 1e-4).sum())

    log.info("Portfolio vol: %.2f%% | Positions: %d", port_vol * 100, n_pos)

    return weights, {
        "expected_alpha": 0.0,
        "portfolio_vol": round(port_vol, 4),
        "active_risk": round(port_vol, 4),
        "sharpe_ratio": 0.0,
        "info_ratio": 0.0,
    }


# ── Solver ────────────────────────────────────────────────────────────────────

_MOSEK_PARAMS = {
    "MSK_IPAR_NUM_THREADS": 4,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-6,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-6,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-6,
}

# Extra params applied only when the problem contains integer variables.
# Allow MOSEK up to 120 s of branch-and-bound; it returns the best incumbent
# when time expires (status = optimal_inaccurate → still accepted). Cardinality
# problems rarely prove the 0.1% gap, but the incumbent stops improving well
# before the cap (core_active 2026-10-30: 99.6% of the 900 s result at 63 s).
_MOSEK_PARAMS_MIP = {
    **_MOSEK_PARAMS,
    "MSK_DPAR_OPTIMIZER_MAX_TIME": 120.0,
    "MSK_DPAR_MIO_MAX_TIME": 120.0,
    "MSK_DPAR_MIO_TOL_REL_GAP": 1e-3,  # 0.1% optimality gap is fine in practice
}


class OptimizationError(RuntimeError):
    """An unsuccessful solve with its explicit CVXPY termination status."""

    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__(f"Optimization failed: {status}")


def _solve(prob, solver_name: str):
    solver_map = {
        "ECOS": cp.ECOS,
        "SCS": cp.SCS,
        "CLARABEL": cp.CLARABEL,
        "MOSEK": cp.MOSEK,
    }
    key = solver_name.upper()
    solver = solver_map.get(key, cp.CLARABEL)
    n_vars = sum(v.size for v in prob.variables())
    log.info("Solving (%s, vars=%d) ...", solver_name, n_vars)

    is_mip = any(v.attributes.get("boolean") for v in prob.variables())
    kwargs: dict = {"verbose": False}
    if solver is cp.MOSEK:
        kwargs["mosek_params"] = _MOSEK_PARAMS_MIP if is_mip else _MOSEK_PARAMS

    prob.solve(solver=solver, **kwargs)
    val = f"  obj={prob.value:.4f}" if prob.value is not None else ""
    log.info("status=%s%s", prob.status, val)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise OptimizationError(prob.status)


# ── Backtest entry point ──────────────────────────────────────────────────────


def optimize_for_backtest(
    alpha_weights: dict[str, float],
    objective: str,
    constraints: dict,
    alpha_date: str,
    barra_date: str | None,
    risk_date: str,
    candidate_sids: list[int],
    bm_weights: dict[int, float],
    prev_weights: dict[int, float] | None,
    max_turnover: float,
    solver: str = "CLARABEL",
    min_weight: float = 0.0,
    risk_aversion: float = 0.0,
    metadata: pd.DataFrame | None = None,
    market_caps_lookup: dict[int, float] | None = None,
    failure_details: list[str] | None = None,
    failure_statuses: list[str] | None = None,
) -> tuple[dict[int, float], dict] | None:
    """
    Single-period optimizer for walk-forward backtest. No file I/O or console output.

    Parameters
    ----------
    alpha_weights : model_id → blend weight (from Alpha_Weights sheet)
    objective     : "maximize_alpha" | "maximize_sharpe" | "minimize_variance"
    constraints   : dict of constraint name → value (from Constraints sheet)
    alpha_date    : model snapshot date used to load alpha scores
    barra_date    : Barra snapshot to use (required)
    risk_date     : retained for logging compatibility; Barra supplies risk
    candidate_sids: stable security IDs in this period's historical universe
    bm_weights    : {sid: weight} for benchmark vector (maximize_alpha only);
                    pass {} for equal-weight fallback
    prev_weights  : {sid: weight} from previous period (None → first period,
                    no turnover constraint applied)
    max_turnover  : two-way turnover fraction, e.g. 0.60 for 60% round-trip
    solver        : CVXPY solver name
    min_weight    : minimum position weight; positions below this are zeroed and
                    the portfolio is renormalised (post-processing, no MIP required)
    failure_statuses: optional output list of explicit failure statuses; only
                    "infeasible" establishes infeasibility for turnover retries

    Returns
    -------
    ({sid: weight}, metrics_dict) or None on failure/infeasibility.
    """
    import contextlib
    import io as _io

    try:
        if barra_date is None:
            raise ValueError("Barra snapshot is required.")

        risk_isins = load_barra_universe(barra_date, by_sid=True)
        investable = sorted(set(candidate_sids) & set(risk_isins))
        if len(investable) < 50:
            if failure_statuses is not None:
                failure_statuses.append("data_error")
            if failure_details is not None:
                failure_details.append(
                    f"Only {len(investable)} candidate SIDs have risk coverage on {barra_date}; need 50"
                )
            return None

        # Alpha scores
        alpha_lookup = load_blended_alpha(alpha_weights, alpha_date, by_sid=True)
        alpha_arr = np.array([alpha_lookup.get(isin, 0.0) for isin in investable])

        # Sector metadata for sector constraints
        if metadata is None:
            with get_db(UNIVERSE_DB) as conn:
                metadata = metadata_by_sid(conn)
        meta_df = _with_business_labels(metadata, alpha_date)
        gics_df = meta_df.set_index("sid")

        # Long/short (market-neutral) strategies need the raw factor exposures so
        # the optimiser can impose beta-neutrality; long-only loads L only.
        is_long_short = not bool(constraints.get("long_only", True))
        beta_exposure: np.ndarray | None = None
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            if is_long_short:
                loaded = load_barra_L(barra_date, investable, return_exposures=True, by_sid=True)
                if loaded is not None:
                    L, X_exp, factor_names = loaded
                    if "beta_60d" in factor_names:
                        beta_exposure = X_exp[:, factor_names.index("beta_60d")]
                else:
                    L = None
            else:
                L = load_barra_L(barra_date, investable, by_sid=True)
        if L is None:
            raise RuntimeError(f"Barra risk model unavailable for {barra_date}.")
        Sigma = None
        used_barra = True

        N = len(investable)

        # Benchmark vector (only used in maximize_alpha long-only objective). A
        # market-neutral book has no benchmark (b = 0 / inert).
        if is_long_short:
            b = np.zeros(N)
        elif bm_weights:
            b_raw = np.array([bm_weights.get(isin, 0.0) for isin in investable])
            b_sum = b_raw.sum()
            b = b_raw / b_sum if b_sum > 1e-10 else np.full(N, 1.0 / N)
        else:
            b = np.full(N, 1.0 / N)  # equal-weight fallback

        # Previous-period weights aligned to current investable universe. Long-only
        # weights are renormalised to sum 1; signed L/S weights are passed through
        # as-is (they sum to ~net ≈ 0, so renormalising would wipe them out).
        prev_w_arr: np.ndarray | None = None
        if prev_weights is not None:
            raw = np.array([prev_weights.get(isin, 0.0) for isin in investable])
            if is_long_short:
                if np.abs(raw).sum() > 1e-10:
                    prev_w_arr = raw
            else:
                total = raw.sum()
                if total > 1e-10:
                    prev_w_arr = raw / total

        sectors, industries, B_sector, B_ind, _sector_fn, _industry_fn = _sector_industry_matrices(
            investable, gics_df
        )
        issuers, B_issuer, _issuer_fn = _issuer_matrices(
            investable, gics_df, required=constraints.get("max_issuer_weight") is not None
        )
        extras = _classification_extras(investable, gics_df, b, constraints)
        market_caps = (
            market_caps_lookup
            if market_caps_lookup is not None
            else load_latest_market_caps(meta_df, investable, by_sid=True)
        )
        cap_buckets, B_cap, _bucket_by_isin = _market_cap_bucket_matrices(
            investable, market_caps, constraints
        )

        # Liquid short universe: only names above min_short_market_cap may be
        # shorted (borrow realism). Longs are unrestricted. Missing-cap names are
        # treated as ineligible to short.
        short_eligible: np.ndarray | None = None
        if is_long_short:
            min_short_cap = constraints.get("min_short_market_cap")
            if min_short_cap is not None:
                thr = float(min_short_cap)
                short_eligible = np.array(
                    [float(market_caps.get(isin, 0.0) or 0.0) >= thr for isin in investable]
                )

        # Integer constraints (max_positions, min_position_if_held) require a MIP-capable
        # solver (MOSEK). Strip them when CLARABEL is selected; pass through for MOSEK.
        # max_cash_weight is NOT stripped here: it is a continuous constraint
        # (cash buffer for the vol cap), not an integer one.
        _INTEGER_CONSTRAINT_KEYS = (
            "max_positions",
            "min_positions",
            "min_position_if_held",
        )
        has_integer = any(constraints.get(k) is not None for k in _INTEGER_CONSTRAINT_KEYS)
        if solver.upper() != "MOSEK" and has_integer:
            relaxed_integer = True
            effective_constraints = {
                k: v for k, v in constraints.items() if k not in _INTEGER_CONSTRAINT_KEYS
            }
        else:
            relaxed_integer = False
            effective_constraints = constraints

        strategy = {
            "constraints": effective_constraints,
            "solver": solver,
            "objective": objective,
            "risk_aversion": risk_aversion,
            "identity_key": "sid",
        }

        # Run optimizer with stdout suppressed (internal prints not relevant in backtest)
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            if objective == "maximize_sharpe":
                weights, extra = _optimize_sharpe(
                    strategy,
                    investable,
                    alpha_arr,
                    b,
                    Sigma,
                    L,
                    sectors,
                    industries,
                    B_sector,
                    B_ind,
                    cap_buckets,
                    B_cap,
                    issuers,
                    B_issuer,
                    extras=extras,
                    prev_weights_arr=prev_w_arr,
                    max_turnover=max_turnover,
                )
            elif objective == "minimize_variance":
                weights, extra = _optimize_min_variance(
                    strategy,
                    investable,
                    b,
                    Sigma,
                    L,
                    sectors,
                    industries,
                    B_sector,
                    B_ind,
                    cap_buckets,
                    B_cap,
                    issuers,
                    B_issuer,
                    extras=extras,
                    prev_weights_arr=prev_w_arr,
                    max_turnover=max_turnover,
                )
            else:  # maximize_alpha
                weights, extra = _optimize_alpha(
                    strategy,
                    investable,
                    alpha_arr,
                    b,
                    Sigma,
                    L,
                    sectors,
                    industries,
                    B_sector,
                    B_ind,
                    cap_buckets,
                    B_cap,
                    issuers,
                    B_issuer,
                    extras=extras,
                    prev_weights_arr=prev_w_arr,
                    max_turnover=max_turnover,
                    beta_exposure=beta_exposure,
                    short_eligible=short_eligible,
                )

        # Keep signed (short) weights for the L/S path; long-only drops the tiny
        # residuals below 1e-5 as before.
        if is_long_short:
            result_weights = {
                isin: float(w) for isin, w in zip(investable, weights) if abs(w) > 1e-5
            }
        else:
            result_weights = {isin: float(w) for isin, w in zip(investable, weights) if w > 1e-5}

        # Drop positions below operational minimum and renormalise (long-only only;
        # renormalising a net-0 book by its ~0 sum is undefined).
        if min_weight > 0.0 and not is_long_short:
            result_weights = {isin: w for isin, w in result_weights.items() if w >= min_weight}
            total = sum(result_weights.values())
            if total > 1e-10:
                result_weights = {isin: w / total for isin, w in result_weights.items()}

        extra.update(
            {
                "used_barra": used_barra,
                "relaxed_integer": relaxed_integer,
                "n_positions": len(result_weights),
                "alpha_date": alpha_date,
                "barra_date": barra_date,
                "risk_date": risk_date,
            }
        )
        return result_weights, extra

    except Exception as exc:
        # Caller decides how to handle (e.g. carry forward previous weights)
        if failure_statuses is not None:
            failure_statuses.append(exc.status if isinstance(exc, OptimizationError) else "error")
        if failure_details is not None:
            failure_details.append(str(exc))
        log.warning("optimize_for_backtest failed (%s): %s", alpha_date, exc)
        return None


# ── Main orchestration ────────────────────────────────────────────────────────


def _load_turnover_anchor(
    c_pre: dict,
    investable: list[str],
    identity_aliases: dict[str, str],
) -> tuple[np.ndarray | None, float | None]:
    """Resolve the optional two-way turnover cap and its prev-weights anchor.

    Returns ``(prev_weights_arr, max_turnover)`` aligned to ``investable``, or
    ``(None, None)`` when the strategy has no enabled ``max_turnover`` — the
    default, leaving the optimiser unconstrained and byte-for-byte unchanged. The
    anchor is the latest live holdings of the incubation book (weights as a
    fraction of net-liq, so deploying held cash into the target counts as genuine
    turnover). Held names outside the investable set are logged as forced exits
    the in-universe cap does not count.
    """
    raw = c_pre.get("max_turnover")
    try:
        max_turnover = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        max_turnover = None
    if max_turnover is None or max_turnover <= 0:
        return None, None

    book, book_meta = load_latest_positions(INCUBATION_PORTFOLIO_ID)
    if book.empty:
        log.warning(
            "max_turnover=%.3f set but no live holdings for %s — turnover constraint skipped.",
            max_turnover,
            INCUBATION_PORTFOLIO_ID,
        )
        return None, None

    book_w = dict(zip(book["isin"], book["weight"]))
    prev = np.array(
        [
            float(book_w.get(isin, book_w.get(identity_aliases.get(isin, ""), 0.0)))
            for isin in investable
        ]
    )
    in_universe = float(prev.sum())
    off_universe = float(book["weight"].sum()) - in_universe
    log.info(
        "Turnover cap %.1f%% two-way | anchor %s @ %s: %d/%d holdings in universe "
        "(%.1f%% of book; %.1f%% off-universe forced-exit).",
        max_turnover * 100,
        INCUBATION_PORTFOLIO_ID,
        book_meta.get("data_date"),
        int((prev > 0).sum()),
        len(book),
        in_universe * 100,
        off_universe * 100,
    )
    return prev, max_turnover


def run_optimization(strategy: dict) -> tuple[pd.DataFrame, dict]:
    log.info(
        "=== Strategy: %s  (%s)  Objective: %s ===",
        strategy["name"],
        strategy["strategy_id"],
        strategy["objective"],
    )

    objective = strategy["objective"]

    log.info("Loading Barra risk coverage ...")
    used_barra_date = strategy["risk_date"]
    risk_isins = load_barra_universe(used_barra_date)
    risk_isin_idx = {isin: i for i, isin in enumerate(risk_isins)}
    log.info("  %d stocks in Barra risk model (%s)", len(risk_isins), used_barra_date)

    universe_isins: list[str] | None = None
    universe_meta: dict = {}
    identity_aliases: dict[str, str] = {}
    universe_index = strategy.get("universe_index", "")
    if universe_index:
        log.info("Loading clean investable universe: %s ...", universe_index)
        universe_isins, universe_meta = _load_clean_investable(
            universe_index, strategy["risk_date"]
        )
        identity_aliases = universe_meta.get("identity_aliases", {})
        log.info(
            "  %d tradable stocks from %s (%d blocked)",
            universe_meta["universe_tradable"],
            universe_meta["universe_source_snapshot_date"],
            universe_meta["universe_blocked"],
        )

    # Benchmark — always load for display if benchmark_index/benchmark_file is set;
    # only fed into the optimiser's b vector for maximize_alpha.
    bm_meta: dict = {}
    benchmark_index = strategy.get("benchmark_index", "")
    if benchmark_index:
        log.info("Loading clean benchmark: %s ...", benchmark_index)
        bm_df, bm_meta = load_index_benchmark(benchmark_index, strategy["risk_date"])
        identity_aliases.update(bm_meta.get("identity_aliases", {}))
        log.info(
            "  %d benchmark stocks from %s (%d blocked)",
            bm_meta["benchmark_tradable"],
            bm_meta["benchmark_source_snapshot_date"],
            bm_meta["benchmark_blocked"],
        )
    elif strategy["benchmark_file"]:
        log.info("Loading benchmark file ...")
        bm_df = load_benchmark(strategy["benchmark_file"])
        ticker_to_isin = map_tickers_to_isins(list(bm_df["ticker"]))
        bm_df["isin"] = bm_df["ticker"].map(ticker_to_isin)
        bm_df = bm_df.dropna(subset=["isin"])
        if bm_df["weight"].sum() > 0:
            bm_df["weight"] /= bm_df["weight"].sum()
        log.info("  %d benchmark stocks", len(bm_df))
        bm_meta = {"benchmark_file": strategy["benchmark_file"]}
    else:
        bm_df = pd.DataFrame(columns=["isin", "weight"])

    # Alpha scores (blended from Alpha_Weights sheet)
    log.info("Loading alpha scores ...")
    alpha_lookup = load_blended_alpha(strategy["alpha_weights"], strategy["alpha_date"])
    models_str = ", ".join(f"{m}×{w:.2g}" for m, w in strategy["alpha_weights"].items())
    log.info("  %d scores  [%s]", len(alpha_lookup), models_str)

    meta_df = _with_business_labels(load_universe_metadata(), strategy["alpha_date"])
    gics_df = meta_df.set_index("isin")

    investable = _build_investable(
        strategy, bm_df, risk_isins, risk_isin_idx, universe_isins, identity_aliases
    )
    alpha_arr = np.array(
        [
            alpha_lookup.get(isin, alpha_lookup.get(identity_aliases.get(isin, ""), 0.0))
            for isin in investable
        ]
    )

    c_pre = strategy["constraints"]

    N = len(investable)
    isin_to_pos = {isin: i for i, isin in enumerate(investable)}
    log.info("Investable universe: %d stocks", N)

    # ── Risk model: Barra only ───────────────────────────────────────────────
    is_long_short = not bool(c_pre.get("long_only", True))
    beta_exposure: np.ndarray | None = None
    if is_long_short:
        loaded = load_barra_L(used_barra_date, investable, return_exposures=True)
        if loaded is not None:
            L, X_exp, factor_names = loaded
            if "beta_60d" in factor_names:
                beta_exposure = X_exp[:, factor_names.index("beta_60d")]
        else:
            L = None
    else:
        L = load_barra_L(used_barra_date, investable)
    if L is None:
        raise RuntimeError(
            f"Barra risk model unavailable for {used_barra_date}; optimizer aborted."
        )
    Sigma = None  # _variance() uses ||L.T @ w||² for Barra.
    log.info("Risk model: Barra (%s)  [%d factors+stocks]", used_barra_date, L.shape[1])

    alpha = alpha_arr

    # bm_display: benchmark weights for results CSV (all strategies)
    bm_lookup = dict(zip(bm_df["isin"], bm_df["weight"])) if not bm_df.empty else {}
    bm_display = np.array([bm_lookup.get(isin, 0.0) for isin in investable])
    if bm_display.sum() > 0:
        bm_display /= bm_display.sum()

    # b: benchmark vector. maximize_alpha uses it in the objective constraints;
    # other objectives may use it for benchmark-relative exposure bands.
    b = bm_display.copy() if bm_display.sum() > 0 else np.zeros(N)

    sectors, industries, B_sector, B_ind, _sector_fn, _industry_fn = _sector_industry_matrices(
        investable, gics_df
    )
    issuers, B_issuer, _issuer_fn = _issuer_matrices(
        investable, gics_df, required=c_pre.get("max_issuer_weight") is not None
    )
    extras = _classification_extras(investable, gics_df, b, c_pre)
    market_caps = load_latest_market_caps(meta_df, investable)
    cap_buckets, B_cap, bucket_by_isin = _market_cap_bucket_matrices(investable, market_caps, c_pre)
    # Liquid short universe (L/S only): names below min_short_market_cap can be
    # held long but not shorted.
    short_eligible: np.ndarray | None = None
    if is_long_short and c_pre.get("min_short_market_cap") is not None:
        _thr = float(c_pre["min_short_market_cap"])
        short_eligible = np.array(
            [float(market_caps.get(isin, 0.0) or 0.0) >= _thr for isin in investable]
        )
    log.info(
        "Market caps: %d/%d available | buckets large=%d mid=%d small=%d",
        len(market_caps),
        len(investable),
        sum(1 for bkt in bucket_by_isin.values() if bkt == "large"),
        sum(1 for bkt in bucket_by_isin.values() if bkt == "mid"),
        sum(1 for bkt in bucket_by_isin.values() if bkt == "small"),
    )

    _ensure_mosek(strategy)

    # ── Turnover anchor (optional) ───────────────────────────────────────────
    # When a strategy enables max_turnover > 0, constrain Σ|w − w_prev| ≤
    # max_turnover (two-way) against the latest live holdings of the incubation
    # book. Disabled by default (no enabled max_turnover row) → both stay None and
    # the optimiser behaves exactly as before. Aligned to the investable set; held
    # names outside it are forced exits the cap does
    # not count — reported below and surfaced in the pre/post view.
    prev_weights_arr, max_turnover = _load_turnover_anchor(c_pre, investable, identity_aliases)

    # Dispatch to objective
    try:
        if objective == "maximize_sharpe":
            weights, extra = _optimize_sharpe(
                strategy,
                investable,
                alpha,
                b,
                Sigma,
                L,
                sectors,
                industries,
                B_sector,
                B_ind,
                cap_buckets,
                B_cap,
                issuers,
                B_issuer,
                extras=extras,
                prev_weights_arr=prev_weights_arr,
                max_turnover=max_turnover,
            )
        elif objective == "minimize_variance":
            weights, extra = _optimize_min_variance(
                strategy,
                investable,
                b,
                Sigma,
                L,
                sectors,
                industries,
                B_sector,
                B_ind,
                cap_buckets,
                B_cap,
                issuers,
                B_issuer,
                extras=extras,
                prev_weights_arr=prev_weights_arr,
                max_turnover=max_turnover,
            )
        else:
            weights, extra = _optimize_alpha(
                strategy,
                investable,
                alpha,
                b,
                Sigma,
                L,
                sectors,
                industries,
                B_sector,
                B_ind,
                cap_buckets,
                B_cap,
                issuers,
                B_issuer,
                extras=extras,
                prev_weights_arr=prev_weights_arr,
                max_turnover=max_turnover,
                beta_exposure=beta_exposure,
                short_eligible=short_eligible,
            )
    except RuntimeError as exc:
        # A too-tight turnover cap is a common infeasibility cause — surface it.
        if max_turnover is not None:
            raise RuntimeError(
                f"{exc} — the max_turnover cap ({max_turnover:.0%} two-way) may be "
                "too tight against the current book; loosen it (or disable the "
                "max_turnover row) and re-run."
            ) from exc
        raise

    # Build results DataFrame
    isin_to_ticker = dict(zip(meta_df["isin"], meta_df["ticker"]))
    isin_to_name = dict(zip(meta_df["isin"], meta_df["company_name"]))
    labels = _output_labels(investable, gics_df)

    rows = []
    for j, isin in enumerate(investable):
        rows.append(
            {
                "isin": isin,
                "ticker": isin_to_ticker.get(isin, ""),
                "company_name": isin_to_name.get(isin, ""),
                **labels[isin],
                "market_cap": round(float(market_caps.get(isin, np.nan)), 2)
                if market_caps.get(isin) is not None
                else np.nan,
                "market_cap_bucket": bucket_by_isin.get(isin, "small"),
                "benchmark_weight": round(float(bm_display[j]), 6),
                "portfolio_weight": round(float(weights[j]), 6),
                "active_weight": round(float(weights[j]) - float(bm_display[j]), 6),
                "alpha_score": round(float(alpha[j]), 4),
            }
        )
    results_df = pd.DataFrame(rows).sort_values("active_weight", ascending=False)

    turnover_summary: dict = {}
    if prev_weights_arr is not None:
        two_way = float(np.abs(np.asarray(weights, dtype=float) - prev_weights_arr).sum())
        turnover_summary = {
            "turnover_cap": round(float(max_turnover), 4),
            "turnover_two_way": round(two_way, 4),
        }
        log.info(
            "Realized turnover: %.1f%% two-way (cap %.1f%%).", two_way * 100, max_turnover * 100
        )

    summary = {
        "strategy_id": strategy["strategy_id"],
        "name": strategy["name"],
        "objective": objective,
        "status": "optimal",
        "alpha_date": strategy["alpha_date"],
        "risk_date": strategy["risk_date"],
        "n_positions": int((np.abs(weights) > 1e-4).sum()),
        "n_benchmark": int((bm_display > 0).sum()),
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        # Which labels the sector/industry/group bands constrained.
        "classification": "inhouse" if "business_sector" in gics_df.columns else "vendor",
        **universe_meta,
        **bm_meta,
        **turnover_summary,
        **extra,
    }
    if used_barra_date:
        summary["barra_date"] = used_barra_date
    return results_df, summary


# ── Output ────────────────────────────────────────────────────────────────────


def save_results(results_df: pd.DataFrame, summary: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sid = summary["strategy_id"]

    latest = OUTPUT_DIR / f"{sid}_latest.csv"
    results_df.to_csv(latest, index=False)

    summary_path = OUTPUT_DIR / f"{sid}_latest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Saved → %s", latest)
    return latest


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Portfolio optimizer")
    parser.add_argument("--strategy", help="Strategy ID (default: all active)")
    parser.add_argument("--list", action="store_true", help="List strategies")
    parser.add_argument(
        "--solver", help="Override solver for all strategies (e.g. MOSEK, CLARABEL, SCS)"
    )
    args = parser.parse_args()

    if not PARAMS_FILE.exists():
        log.error("ERROR: %s not found. Run create_strategy_params.py first.", PARAMS_FILE)
        return

    if args.list:
        xl = pd.ExcelFile(PARAMS_FILE)
        df = pd.read_excel(xl, sheet_name="Strategies")
        # print() intentional here — tabular display for CLI listing only
        print(df[["strategy_id", "name", "active", "objective"]].to_string(index=False))
        return

    strategies = load_strategy_params(args.strategy)
    if args.solver:
        for s in strategies:
            s["solver"] = args.solver.upper()
    log.info("Running %d strategy/strategies ...", len(strategies))

    for s in strategies:
        try:
            results_df, summary = run_optimization(s)
            save_results(results_df, summary)
        except Exception as exc:
            log.error("ERROR in %s: %s", s["strategy_id"], exc)
            raise

    log.info("Done.")


if __name__ == "__main__":
    main()
