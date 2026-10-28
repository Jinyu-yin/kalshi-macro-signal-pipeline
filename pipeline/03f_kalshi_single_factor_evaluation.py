"""
Kalshi-specific single-factor evaluation against realized FOMC outcomes.

Goal
- Keep the evaluation close to the paper's spirit: ask whether Kalshi distribution features
  anticipate realized FOMC outcomes, especially the realized policy rate and hike/cut magnitude.
- Stay local-only and lightweight.

Inputs
- out_csv/kalshi_research_features_daily.csv
- out_csv/fomc_realized_decisions.csv

Outputs
- out_csv/kalshi_single_factor_evaluation.csv
- out_csv/kalshi_single_factor_top_hits.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"
FEATURES = OUT / "kalshi_research_features_daily.csv"
REALIZED = OUT / "fomc_realized_decisions.csv"
EVAL_OUT = OUT / "kalshi_single_factor_evaluation.csv"
TOP_OUT = OUT / "kalshi_single_factor_top_hits.csv"


FEATURE_COLS = [
    "mean",
    "median",
    "mode",
    "variance",
    "std",
    "skewness",
    "kurtosis",
    "num_strikes",
    "range_width",
    "mass_proxy",
    "delta_mean_1d",
    "delta_mean_3d",
    "delta_mean_5d",
    "delta_median_1d",
    "delta_mode_1d",
    "delta_variance_1d",
    "delta_skewness_1d",
    "delta_kurtosis_1d",
]

TARGET_COLS = ["ffr_upper", "ffr_lower", "change_bps"]


def _load_features() -> pd.DataFrame:
    if not FEATURES.exists():
        raise FileNotFoundError(f"Missing research features file: {FEATURES}")
    df = pd.read_csv(FEATURES)
    if df.empty:
        raise RuntimeError("Kalshi research features table is empty")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in FEATURE_COLS + ["days_to_release"]:
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
    for c in TARGET_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["fomc_date"])


def _pick_snapshot_before_event(feats: pd.DataFrame, event_date: pd.Timestamp) -> pd.DataFrame:
    """Pick the most recent feature row strictly before the event date for each factor."""
    sub = feats[feats["date"] < event_date].copy()
    if sub.empty:
        return sub
    idx = sub.sort_values("date").groupby("factor", dropna=False).tail(1).index
    return sub.loc[idx].copy()


def _factor_stats(x: pd.Series, y: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    x = x[m]
    y = y[m]
    out = {
        "n_obs": int(len(x)),
        "corr_pearson": np.nan,
        "corr_spearman": np.nan,
        "mae_rank": np.nan,
        "rmse_rank": np.nan,
        "directional_hit_rate": np.nan,
    }
    if len(x) < 3:
        return out

    out["corr_pearson"] = float(x.corr(y, method="pearson"))
    out["corr_spearman"] = float(x.corr(y, method="spearman"))
    out["mae_rank"] = float(np.mean(np.abs(x - y)))
    out["rmse_rank"] = float(np.sqrt(np.mean((x - y) ** 2)))
    out["directional_hit_rate"] = float(np.mean(np.sign(x - x.median()) == np.sign(y - y.median())))
    return out


def evaluate_against_target(feats: pd.DataFrame, realized: pd.DataFrame, target: str) -> pd.DataFrame:
    rows: List[dict] = []
    for _, ev in realized.iterrows():
        event_date = ev["fomc_date"]
        snap = _pick_snapshot_before_event(feats, event_date)
        if snap.empty:
            continue

        # Paper-aligned comparison target:
        # - realized policy rate upper bound, lower bound, or change in bps
        if target == "change_bps":
            y_value = ev["change_bps"] / 100.0  # scale to rate-like units
        elif target == "ffr_upper":
            y_value = ev["ffr_upper"]
        else:
            y_value = ev["ffr_lower"]

        # Aggregate factor-level feature snapshots into a single event-level row.
        # This keeps the first pass lightweight while still allowing factor ranking.
        for factor, g in snap.groupby("factor", dropna=False):
            g = g.sort_values("date")
            last = g.iloc[-1]
            row = {
                "fomc_date": event_date,
                "factor": factor,
                "target": target,
                "target_value": y_value,
                "snapshot_date": last["date"],
                "days_to_release": last.get("days_to_release", np.nan),
                "mean": last.get("mean", np.nan),
                "median": last.get("median", np.nan),
                "mode": last.get("mode", np.nan),
                "variance": last.get("variance", np.nan),
                "std": last.get("std", np.nan),
                "skewness": last.get("skewness", np.nan),
                "kurtosis": last.get("kurtosis", np.nan),
                "num_strikes": last.get("num_strikes", np.nan),
                "range_width": last.get("range_width", np.nan),
                "mass_proxy": last.get("mass_proxy", np.nan),
                "delta_mean_1d": last.get("delta_mean_1d", np.nan),
                "delta_mean_3d": last.get("delta_mean_3d", np.nan),
                "delta_mean_5d": last.get("delta_mean_5d", np.nan),
                "delta_variance_1d": last.get("delta_variance_1d", np.nan),
                "delta_skewness_1d": last.get("delta_skewness_1d", np.nan),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Construct a simple factor score rank per target.
    score_rows = []
    for factor, g in df.groupby("factor", dropna=False):
        g = g.copy()
        # Primary paper-aligned intuition:
        # closer mean/median/mode to realized policy rate is better;
        # tighter variance and positive delta_mean before hikes should be informative.
        for feature in ["mean", "median", "mode", "variance", "std", "skewness", "kurtosis", "delta_mean_1d", "delta_variance_1d"]:
            if feature not in g.columns:
                continue
            stats = _factor_stats(g[feature], g["target_value"])
            score_rows.append({
                "factor": factor,
                "feature": feature,
                "target": target,
                **stats,
            })
    return pd.DataFrame(score_rows)


def main() -> None:
    feats = _load_features()
    realized = _load_realized()

    # Focus on FFR-related rows for the first pass; this is the paper's cleanest target.
    feats = feats[feats["factor"] == "ffr"].copy()
    if feats.empty:
        raise RuntimeError("No FFR rows found in Kalshi research features")

    all_scores = []
    for target in ["ffr_upper", "ffr_lower", "change_bps"]:
        score_df = evaluate_against_target(feats, realized, target)
        if not score_df.empty:
            all_scores.append(score_df)

    if not all_scores:
        raise RuntimeError("No factor evaluation results produced")

    eval_df = pd.concat(all_scores, ignore_index=True)
    eval_df = eval_df.sort_values(["target", "corr_spearman", "corr_pearson"], ascending=[True, False, False])
    OUT.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(EVAL_OUT, index=False, encoding="utf-8-sig")

    # Top hits across targets/features.
    top = eval_df.sort_values(["corr_spearman", "corr_pearson"], ascending=[False, False]).head(20)
    top.to_csv(TOP_OUT, index=False, encoding="utf-8-sig")

    print(f"Saved evaluation to {EVAL_OUT}")
    print(f"Saved top hits to {TOP_OUT}")
    print("\nTop results:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
