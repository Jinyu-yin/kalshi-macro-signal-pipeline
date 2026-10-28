"""
Kalshi FFR event study around FOMC dates.

Purpose
- Test whether the FFR distribution moments move systematically before FOMC decisions.
- Keep the first implementation local-only and paper-aligned:
  * level moments: mean / median / mode
  * shape moments: variance / std / skewness / kurtosis
  * short-horizon deltas: 1d changes in selected moments

Inputs
- out_csv/kalshi_research_features_daily.csv
- out_csv/fomc_realized_decisions.csv

Outputs
- out_csv/kalshi_ffr_event_study.csv
- out_csv/kalshi_ffr_event_study_summary.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"
FEATURES = OUT / "kalshi_research_features_daily.csv"
REALIZED = OUT / "fomc_realized_decisions.csv"
EVENT_OUT = OUT / "kalshi_ffr_event_study.csv"
SUMMARY_OUT = OUT / "kalshi_ffr_event_study_summary.csv"

WINDOWS = [1, 3, 5, 10]
MOMENTS = ["mean", "median", "mode", "variance", "std", "skewness", "kurtosis"]
DELTA_MOMENTS = ["delta_mean_1d", "delta_variance_1d", "delta_skewness_1d"]


def _load_features() -> pd.DataFrame:
    if not FEATURES.exists():
        raise FileNotFoundError(f"Missing Kalshi feature table: {FEATURES}")
    df = pd.read_csv(FEATURES)
    if df.empty:
        raise RuntimeError("Kalshi feature table is empty")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["days_to_release"] + MOMENTS + DELTA_MOMENTS + ["num_strikes", "range_width", "mass_proxy"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_realized() -> pd.DataFrame:
    if not REALIZED.exists():
        raise FileNotFoundError(f"Missing realized FOMC file: {REALIZED}")
    df = pd.read_csv(REALIZED)
    if df.empty:
        raise RuntimeError("Realized FOMC file is empty")
    df["fomc_date"] = pd.to_datetime(df["fomc_date"], errors="coerce")
    for c in ["ffr_upper", "ffr_lower", "change_bps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["fomc_date"])


def _event_window_snapshot(feats: pd.DataFrame, event_date: pd.Timestamp, window: int) -> pd.DataFrame:
    """Get all rows in [event_date - window, event_date)."""
    start = event_date - pd.Timedelta(days=int(window))
    sub = feats[(feats["date"] >= start) & (feats["date"] < event_date)].copy()
    return sub


def _summarize_window(sub: pd.DataFrame, event_date: pd.Timestamp, window: int, realized_row: pd.Series) -> Dict[str, object]:
    row: Dict[str, object] = {
        "fomc_date": event_date,
        "window_days": int(window),
        "n_rows": int(len(sub)),
        "n_dates": int(sub["date"].nunique(dropna=True)),
        "realized_ffr_upper": float(realized_row.get("ffr_upper", np.nan)),
        "realized_ffr_lower": float(realized_row.get("ffr_lower", np.nan)),
        "realized_change_bps": float(realized_row.get("change_bps", np.nan)),
    }

    if sub.empty:
        for c in MOMENTS + DELTA_MOMENTS + ["num_strikes", "range_width", "mass_proxy"]:
            row[f"{c}_mean"] = np.nan
            row[f"{c}_last"] = np.nan
            row[f"{c}_trend"] = np.nan
        return row

    sub = sub.sort_values(["factor", "date"]).copy()
    # Use the FFR factor only; this is the cleanest paper-aligned target.
    ffr = sub[sub["factor"] == "ffr"].copy()
    if ffr.empty:
        for c in MOMENTS + DELTA_MOMENTS + ["num_strikes", "range_width", "mass_proxy"]:
            row[f"{c}_mean"] = np.nan
            row[f"{c}_last"] = np.nan
            row[f"{c}_trend"] = np.nan
        return row

    for c in MOMENTS + DELTA_MOMENTS + ["num_strikes", "range_width", "mass_proxy"]:
        if c not in ffr.columns:
            row[f"{c}_mean"] = np.nan
            row[f"{c}_last"] = np.nan
            row[f"{c}_trend"] = np.nan
            continue
        vals = pd.to_numeric(ffr[c], errors="coerce")
        row[f"{c}_mean"] = float(vals.mean(skipna=True)) if vals.notna().any() else np.nan
        row[f"{c}_last"] = float(vals.dropna().iloc[-1]) if vals.notna().any() else np.nan
        if vals.notna().sum() >= 2:
            x = np.arange(len(vals.dropna()), dtype=float)
            y = vals.dropna().to_numpy(dtype=float)
            slope = np.polyfit(x, y, 1)[0]
            row[f"{c}_trend"] = float(slope)
        else:
            row[f"{c}_trend"] = np.nan
    return row


def build_event_study() -> pd.DataFrame:
    feats = _load_features()
    realized = _load_realized()
    feats = feats[feats["factor"] == "ffr"].copy()
    if feats.empty:
        raise RuntimeError("No FFR rows found in Kalshi research features")

    rows: List[Dict[str, object]] = []
    realized = realized.sort_values("fomc_date")
    for _, r in realized.iterrows():
        event_date = r["fomc_date"]
        for w in WINDOWS:
            sub = _event_window_snapshot(feats, event_date, w)
            rows.append(_summarize_window(sub, event_date, w, r))

    return pd.DataFrame(rows)


def _simple_relation_stats(df: pd.DataFrame, xcol: str, ycol: str) -> Dict[str, float]:
    d = df[[xcol, ycol]].dropna()
    if len(d) < 3:
        return {"n_obs": int(len(d)), "pearson": np.nan, "spearman": np.nan, "mae": np.nan, "rmse": np.nan}
    x = d[xcol].astype(float)
    y = d[ycol].astype(float)
    return {
        "n_obs": int(len(d)),
        "pearson": float(x.corr(y, method="pearson")),
        "spearman": float(x.corr(y, method="spearman")),
        "mae": float(np.mean(np.abs(x - y))),
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
    }


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    targets = {
        "realized_ffr_lower": "ffr_lower",
        "realized_ffr_upper": "ffr_upper",
        "realized_change_bps": "change_bps",
    }

    for window in sorted(df["window_days"].unique()):
        sub = df[df["window_days"] == window].copy()
        for moment in ["mean_last", "median_last", "mode_last", "variance_last", "std_last", "skewness_last", "kurtosis_last"]:
            for target_name, target_col in targets.items():
                stats = _simple_relation_stats(sub, moment, target_name)
                rows.append(
                    {
                        "window_days": int(window),
                        "feature": moment,
                        "target": target_col,
                        **stats,
                    }
                )
        for delta_feature in ["delta_mean_1d_mean", "delta_variance_1d_mean", "delta_skewness_1d_mean", "delta_mean_1d_last", "delta_variance_1d_last"]:
            if delta_feature not in sub.columns:
                continue
            for target_name, target_col in targets.items():
                stats = _simple_relation_stats(sub, delta_feature, target_name)
                rows.append(
                    {
                        "window_days": int(window),
                        "feature": delta_feature,
                        "target": target_col,
                        **stats,
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target", "window_days", "spearman", "pearson"], ascending=[True, True, False, False])
    return out


def main() -> None:
    event_df = build_event_study()
    OUT.mkdir(parents=True, exist_ok=True)
    event_df.to_csv(EVENT_OUT, index=False, encoding="utf-8-sig")

    summary_df = build_summary(event_df)
    summary_df.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

    print(f"Saved event study to {EVENT_OUT}")
    print(f"Saved summary to {SUMMARY_OUT}")
    if not summary_df.empty:
        print("\nTop summary rows:")
        print(summary_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
