"""
Build a research-grade Kalshi feature table aligned to the paper.

Inputs
- out_csv/kalshi_ffr_distributions_daily.csv
- out_csv/kalshi_cpi_distributions_daily.csv (optional, if available)
- out_csv/kalshi_u3_distributions_daily.csv (optional, if available)
- out_csv/kalshi_pcecore_distributions_daily.csv (optional, if available)
- out_csv/kalshi_ratecutcount_distributions_daily.csv (optional, if available)

Output
- out_csv/kalshi_research_features_daily.csv

Design goals
- Keep the first version lightweight and fully reproducible on local compute.
- Prioritize paper-aligned fields: level, shape, deltas, event timing, and basic liquidity proxies.
- Avoid expensive joins or external data dependencies unless the files already exist locally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"

DATASETS = {
    "ffr": OUT / "kalshi_ffr_distributions_daily.csv",
    "cpi": OUT / "kalshi_cpi_distributions_daily.csv",
    "u3": OUT / "kalshi_u3_distributions_daily.csv",
    "pcecore": OUT / "kalshi_pcecore_distributions_daily.csv",
    "ratecutcount": OUT / "kalshi_ratecutcount_distributions_daily.csv",
}

PRIMARY = ["ffr", "cpi", "u3", "pcecore", "ratecutcount"]


def _read_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    return df


def _normalize_frame(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    out = df.copy()
    out["factor"] = factor

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    elif "trade_date" in out.columns:
        out["date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    else:
        out["date"] = pd.NaT

    for col in ["mean", "median", "mode", "variance", "std", "skewness", "kurtosis"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "num_strikes" not in out.columns:
        out["num_strikes"] = np.nan
    out["num_strikes"] = pd.to_numeric(out["num_strikes"], errors="coerce")

    if "strike_min" not in out.columns:
        out["strike_min"] = np.nan
    if "strike_max" not in out.columns:
        out["strike_max"] = np.nan

    if "days_to_release" not in out.columns:
        out["days_to_release"] = np.nan
    out["days_to_release"] = pd.to_numeric(out["days_to_release"], errors="coerce")

    if "trade_date" not in out.columns:
        out["trade_date"] = out["date"]
    else:
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")

    return out


def _add_time_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["factor", "date"]).copy()
    group = out.groupby("factor", dropna=False)

    for col in ["mean", "median", "mode", "variance", "skewness", "kurtosis", "std"]:
        out[f"delta_{col}_1d"] = group[col].diff(1)
        out[f"delta_{col}_3d"] = group[col].diff(3)
        out[f"delta_{col}_5d"] = group[col].diff(5)

    out["range_width"] = out["strike_max"] - out["strike_min"]
    out["mass_proxy"] = out["num_strikes"] / out["range_width"].replace(0, np.nan)
    out["mass_proxy"] = out["mass_proxy"].replace([np.inf, -np.inf], np.nan)

    # Paper-aligned simple regime flags; these are cheap and interpretable.
    out["high_uncertainty"] = (out["std"] > out["std"].median(skipna=True)).astype(float)
    out["high_asymmetry"] = (out["skewness"].abs() > out["skewness"].abs().median(skipna=True)).astype(float)

    return out


def _add_event_proximity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "factor" not in out.columns:
        return out

    # Keep it conservative and local: no external calendars here.
    # For FFR and inflation factors, days_to_release already gives the event proximity.
    out["event_proximity"] = np.nan
    mask = out["days_to_release"].notna()
    out.loc[mask, "event_proximity"] = -out.loc[mask, "days_to_release"]
    return out


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "factor",
        "date",
        "trade_date",
        "days_to_release",
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
        "high_uncertainty",
        "high_asymmetry",
        "delta_mean_1d",
        "delta_mean_3d",
        "delta_mean_5d",
        "delta_median_1d",
        "delta_mode_1d",
        "delta_variance_1d",
        "delta_skewness_1d",
        "delta_kurtosis_1d",
        "event_proximity",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols].sort_values(["factor", "date"]).reset_index(drop=True)


def main() -> None:
    frames: List[pd.DataFrame] = []

    for factor in PRIMARY:
        df = _read_optional_csv(DATASETS[factor])
        if df is None:
            continue
        norm = _normalize_frame(df, factor)
        frames.append(norm)

    if not frames:
        raise FileNotFoundError("No Kalshi distribution CSVs found in out_csv")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = _add_time_deltas(combined)
    combined = _add_event_proximity(combined)
    combined = _select_columns(combined)

    OUT.mkdir(parents=True, exist_ok=True)
    output_path = OUT / "kalshi_research_features_daily.csv"
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved {len(combined):,} rows to {output_path}")
    print("\nPreview:")
    print(combined.head(10).to_string(index=False))
    print("\nCoverage by factor:")
    print(combined.groupby("factor").size().to_string())


if __name__ == "__main__":
    main()
