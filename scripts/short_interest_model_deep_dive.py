#!/usr/bin/env python3
"""Read-only audit for the consolidated short-interest model."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODELS_DB, RETURNS_DB, UNIVERSE_DB
from scripts.growth_variant_experiments import forward_returns, load_dates, load_returns_matrix, load_universe


BASE_MODELS = ["PROF001", "DEF001", "VAL001", "GRO001", "MOM001", "SIZ001", "SHI001", "ALP001"]


def load_models(start: str) -> pd.DataFrame:
    with sqlite3.connect(MODELS_DB) as conn:
        return pd.read_sql_query(
            "SELECT data_date, security_id, model_id, model_value, model_value_z "
            f"FROM models WHERE data_date >= ? AND model_id IN ({','.join('?' * len(BASE_MODELS))})",
            conn,
            params=[start, *BASE_MODELS],
        )


def evaluate_model(
    scores: pd.DataFrame, universe: pd.DataFrame, fwd: dict[str, pd.Series]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pit = universe[["data_date", "security_id", "gics_sector"]].drop_duplicates()
    scores = scores.merge(pit, on=["data_date", "security_id"], how="inner")
    rows = []
    sector_rows = []
    for data_date, fr in fwd.items():
        snap = scores[scores["data_date"] == data_date]
        df = snap.set_index("security_id")[["model_value_z", "gics_sector"]].join(fr.rename("fwd_ret")).dropna()
        if len(df) < 30:
            continue
        neutral = df.copy()
        neutral["model_value_z"] = neutral["model_value_z"] - neutral.groupby("gics_sector")["model_value_z"].transform("mean")
        neutral["fwd_ret"] = neutral["fwd_ret"] - neutral.groupby("gics_sector")["fwd_ret"].transform("mean")
        ordered = df.sort_values("model_value_z", ascending=False)
        n = max(int(len(df) * 0.2), 1)
        rows.append(
            {
                "data_date": data_date,
                "ic": df["model_value_z"].corr(df["fwd_ret"], method="spearman"),
                "neutral_ic": neutral["model_value_z"].corr(neutral["fwd_ret"], method="spearman"),
                "spread": ordered.head(n)["fwd_ret"].mean() - ordered.tail(n)["fwd_ret"].mean(),
                "n": len(df),
            }
        )
        for sector, group in df.groupby("gics_sector"):
            if len(group) >= 15:
                sector_rows.append(
                    {
                        "data_date": data_date,
                        "sector": sector,
                        "ic": group["model_value_z"].corr(group["fwd_ret"], method="spearman"),
                        "n": len(group),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(sector_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="sp500_3pct_capped")
    parser.add_argument("--start", default="2021-06-30")
    args = parser.parse_args()

    universe = load_universe(args.index, args.start)
    dates = load_dates(args.start)
    models = load_models(args.start)
    returns = load_returns_matrix()
    fwd = forward_returns(returns, dates)

    shi = models[models["model_id"] == "SHI001"]
    pit = universe[["data_date", "security_id", "ticker", "company_name", "gics_sector"]].drop_duplicates()
    cov = pit.merge(shi[["data_date", "security_id", "model_value_z"]], on=["data_date", "security_id"], how="left")
    cov_by_date = cov.groupby("data_date").agg(
        names=("security_id", "count"),
        covered=("model_value_z", lambda s: s.notna().sum()),
    )
    cov_by_date["coverage"] = cov_by_date["covered"] / cov_by_date["names"]

    summary, sector = evaluate_model(shi, universe, fwd)
    summary["year"] = pd.to_datetime(summary["data_date"]).dt.year

    base = models.merge(pit[["data_date", "security_id"]], on=["data_date", "security_id"], how="inner")
    wide = base.pivot_table(index=["data_date", "security_id"], columns="model_id", values="model_value_z")

    latest = sorted(cov["data_date"].unique())[-2]
    latest_scores = (
        cov[cov["data_date"] == latest]
        .dropna(subset=["model_value_z"])
        .sort_values("model_value_z", ascending=False)
    )

    print(f"Index={args.index} start={args.start} latest={latest}")
    print("\n=== Coverage ===")
    print(
        pd.Series(
            {
                "dates": len(cov_by_date),
                "avg_coverage": cov_by_date["coverage"].mean(),
                "min_coverage": cov_by_date["coverage"].min(),
                "latest_coverage": cov_by_date.loc[latest, "coverage"],
            }
        )
        .round(4)
        .to_string()
    )

    print("\n=== Standalone SHI001 ===")
    print(
        pd.Series(
            {
                "n_months": len(summary),
                "ic": summary["ic"].mean(),
                "hit": (summary["ic"] > 0).mean(),
                "neutral_ic": summary["neutral_ic"].mean(),
                "spread": summary["spread"].mean(),
            }
        )
        .round(4)
        .to_string()
    )

    print("\n=== SHI001 IC by Year ===")
    print(summary.groupby("year")[["ic", "neutral_ic", "spread"]].mean().round(4).to_string())

    print("\n=== SHI001 Sector IC ===")
    print(sector.groupby("sector")["ic"].mean().sort_values(ascending=False).round(4).to_string())

    print("\n=== Base Model Correlations ===")
    cols = [c for c in BASE_MODELS if c in wide.columns]
    print(wide[cols].corr()["SHI001"].drop("SHI001").sort_values(ascending=False).round(3).to_string())

    print("\n=== Latest Lowest Short-Interest Scores ===")
    print(latest_scores.head(20)[["ticker", "company_name", "gics_sector", "model_value_z"]].round(3).to_string(index=False))

    print("\n=== Latest Highest Short-Pressure Scores ===")
    print(latest_scores.tail(20)[["ticker", "company_name", "gics_sector", "model_value_z"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
