"""
Validate the Kalshi research feature table.

This is intentionally lightweight and local-only:
- coverage by factor
- missingness for paper-aligned fields
- event timing sanity checks
- distribution-moment ranges
- delta feature availability

Output
- out_csv/kalshi_research_features_validation.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"
INPUT = OUT / "kalshi_research_features_daily.csv"
OUTPUT = OUT / "kalshi_research_features_validation.csv"


FIELDS = [
    "mean",
    "median",
    "mode",
    "variance",
    "std",
    "skewness",
    "kurtosis",
    "num_strikes",
    "strike_min",
    "strike_max",
    "range_width",
    "mass_proxy",
    "days_to_release",
    "event_proximity",
    "delta_mean_1d",
    "delta_mean_3d",
    "delta_mean_5d",
    "delta_median_1d",
    "delta_mode_1d",
    "delta_variance_1d",
    "delta_skewness_1d",
    "delta_kurtosis_1d",
]


def _load() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing research feature table: {INPUT}")
    df = pd.read_csv(INPUT)
    if df.empty:
        raise RuntimeError("Kalshi research feature table is empty")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in FIELDS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for factor, g in df.groupby("factor", dropna=False):
        row: Dict[str, object] = {
            "factor": factor,
            "rows": int(len(g)),
            "date_min": g["date"].min(),
            "date_max": g["date"].max(),
            "n_days": int(g["date"].nunique(dropna=True)),
            "mean_num_strikes": float(g["num_strikes"].mean(skipna=True)) if "num_strikes" in g else np.nan,
            "median_num_strikes": float(g["num_strikes"].median(skipna=True)) if "num_strikes" in g else np.nan,
        }
        for f in ["mean", "median", "mode", "variance", "skewness", "kurtosis", "days_to_release"]:
            if f in g.columns:
                row[f"missing_{f}"] = float(g[f].isna().mean())
        row["pct_days_with_delta_mean_1d"] = float(g["delta_mean_1d"].notna().mean()) if "delta_mean_1d" in g else np.nan
        row["pct_days_with_delta_variance_1d"] = float(g["delta_variance_1d"].notna().mean()) if "delta_variance_1d" in g else np.nan
        row["pct_days_with_event_proximity"] = float(g["event_proximity"].notna().mean()) if "event_proximity" in g else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rows", "factor"], ascending=[False, True])


def _sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor, g in df.groupby("factor", dropna=False):
        checks = {
            "factor": factor,
            "pct_mean_outside_range": float(((g["mean"] < g["strike_min"]) | (g["mean"] > g["strike_max"])).fillna(False).mean())
            if {"mean", "strike_min", "strike_max"}.issubset(g.columns)
            else np.nan,
            "pct_negative_variance": float((g["variance"] < 0).mean()) if "variance" in g else np.nan,
            "pct_zero_num_strikes": float((g["num_strikes"] <= 1).mean()) if "num_strikes" in g else np.nan,
            "pct_nonfinite_mass_proxy": float((~np.isfinite(g["mass_proxy"])).mean()) if "mass_proxy" in g else np.nan,
            "pct_missing_days_to_release": float(g["days_to_release"].isna().mean()) if "days_to_release" in g else np.nan,
        }
        rows.append(checks)
    return pd.DataFrame(rows).sort_values("factor")


def main() -> None:
    df = _load()
    cov = _coverage(df)
    san = _sanity_checks(df)
    out = cov.merge(san, on="factor", how="outer")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print(f"Saved validation report to {OUTPUT}")
    print("\nCoverage summary:")
    print(out[["factor", "rows", "n_days", "mean_num_strikes", "pct_days_with_delta_mean_1d"]].to_string(index=False))
    print("\nSanity summary:")
    print(out[["factor", "pct_zero_num_strikes", "pct_negative_variance", "pct_missing_days_to_release"]].to_string(index=False))


if __name__ == "__main__":
    main()
