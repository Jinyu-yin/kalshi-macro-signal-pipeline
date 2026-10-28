"""
Fast validation pack for Kalshi CPI / U3 / PCE-style factors.

Purpose
- Quickly compare factor quality across the available daily distribution files.
- Keep the first pass lightweight and local-only.

Inputs
- out_csv/kalshi_cpi_distributions_daily.csv
- out_csv/kalshi_u3_distributions_daily.csv
- out_csv/kalshi_pcecore_distributions_daily.csv
- out_csv/kalshi_ratecutcount_distributions_daily.csv (if available)

Outputs
- out_csv/kalshi_factor_validation_pack.csv
- out_csv/kalshi_factor_validation_pack_top.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"
FILES = {
    "cpi": OUT / "kalshi_cpi_distributions_daily.csv",
    "u3": OUT / "kalshi_u3_distributions_daily.csv",
    "pcecore": OUT / "kalshi_pcecore_distributions_daily.csv",
    "ratecutcount": OUT / "kalshi_ratecutcount_distributions_daily.csv",
}


def _load(path: Path, factor: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    df = df.copy()
    if "trade_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"trade_date": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["factor"] = factor
    for c in ["mean", "median", "mode", "std", "variance", "skewness", "kurtosis", "days_to_release", "num_strikes"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    if "strike_min" not in df.columns:
        df["strike_min"] = np.nan
    if "strike_max" not in df.columns:
        df["strike_max"] = np.nan
    return df


def _summarize(df: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {
        "rows": int(len(df)),
        "n_days": int(df["date"].nunique(dropna=True)),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "mean_num_strikes": float(df["num_strikes"].mean(skipna=True)),
        "median_num_strikes": float(df["num_strikes"].median(skipna=True)),
        "pct_missing_mean": float(df["mean"].isna().mean()),
        "pct_missing_median": float(df["median"].isna().mean()),
        "pct_missing_mode": float(df["mode"].isna().mean()),
        "pct_missing_std": float(df["std"].isna().mean()),
        "pct_missing_days_to_release": float(df["days_to_release"].isna().mean()),
        "pct_zero_or_one_strike": float((df["num_strikes"] <= 1).mean()),
        "pct_nonfinite_std": float((~np.isfinite(df["std"])).mean()),
        "pct_mean_outside_range": float(((df["mean"] < df["strike_min"]) | (df["mean"] > df["strike_max"])).fillna(False).mean()),
        "avg_abs_delta_mean_1d": float(df["mean"].diff().abs().mean()) if len(df) > 1 else np.nan,
        "avg_abs_delta_std_1d": float(df["std"].diff().abs().mean()) if len(df) > 1 else np.nan,
    }
    return out


def main() -> None:
    frames: List[pd.DataFrame] = []
    for factor, path in FILES.items():
        df = _load(path, factor)
        if df is not None:
            frames.append(df)

    if not frames:
        raise FileNotFoundError("No factor distribution files found")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    rows = []
    for factor, g in combined.groupby("factor", dropna=False):
        rows.append({"factor": factor, **_summarize(g)})

    out = pd.DataFrame(rows).sort_values(["rows", "factor"], ascending=[False, True])
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "kalshi_factor_validation_pack.csv", index=False, encoding="utf-8-sig")
    out.head(20).to_csv(OUT / "kalshi_factor_validation_pack_top.csv", index=False, encoding="utf-8-sig")

    print(f"Saved validation pack to {OUT / 'kalshi_factor_validation_pack.csv'}")
    print("\nSummary:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
