"""
Quick predictiveness scan for Kalshi CPI/U3/PCE factors.

Idea (fast, local, no external API dependency)
- For each release_date and factor, treat release-day market estimate (days_to_release == 0)
  as a local anchor/target.
- Evaluate pre-release snapshots (1/3/5/10 days before release) for mean/median/mode/std.
- Report correlation and error metrics by factor + horizon.

This is a fast diagnostic for predictiveness structure, not a final out-of-sample truth test.
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
}

HORIZONS = [1, 3, 5, 10]
FEATURES = ["mean", "median", "mode", "std"]


def _load_factor(path: Path, factor: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None

    df = df.copy()
    if "trade_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"trade_date": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["factor"] = factor

    for c in FEATURES + ["days_to_release", "num_strikes"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["date", "release_date"])


def _stats(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(d) < 3:
        return {"n_obs": int(len(d)), "pearson": np.nan, "spearman": np.nan, "mae": np.nan, "rmse": np.nan}
    xx = d["x"].to_numpy(dtype=float)
    yy = d["y"].to_numpy(dtype=float)
    return {
        "n_obs": int(len(d)),
        "pearson": float(pd.Series(xx).corr(pd.Series(yy), method="pearson")),
        "spearman": float(pd.Series(xx).corr(pd.Series(yy), method="spearman")),
        "mae": float(np.mean(np.abs(xx - yy))),
        "rmse": float(np.sqrt(np.mean((xx - yy) ** 2))),
    }


def main() -> None:
    frames: List[pd.DataFrame] = []
    for factor, path in FILES.items():
        d = _load_factor(path, factor)
        if d is not None:
            frames.append(d)

    if not frames:
        raise FileNotFoundError("No CPI/U3/PCE distribution files found")

    df = pd.concat(frames, ignore_index=True, sort=False)

    # release-day anchor: per factor + release_date use days_to_release==0, fallback nearest >=0 smallest.
    anchors = []
    for (factor, rdate), g in df.groupby(["factor", "release_date"], dropna=False):
        g = g.sort_values(["days_to_release", "date"])
        g0 = g[g["days_to_release"] == 0]
        if g0.empty:
            g0 = g[g["days_to_release"] >= 0]
        if g0.empty:
            continue
        last = g0.iloc[-1]
        anchors.append(
            {
                "factor": factor,
                "release_date": rdate,
                "anchor_mean": last.get("mean", np.nan),
                "anchor_median": last.get("median", np.nan),
                "anchor_mode": last.get("mode", np.nan),
                "anchor_std": last.get("std", np.nan),
            }
        )

    anchor_df = pd.DataFrame(anchors)
    if anchor_df.empty:
        raise RuntimeError("No valid release anchors found")

    rows = []
    for horizon in HORIZONS:
        for (factor, rdate), g in df.groupby(["factor", "release_date"], dropna=False):
            # pick latest snapshot with days_to_release >= horizon (i.e., at least h days before release)
            cand = g[g["days_to_release"] >= horizon].sort_values("date")
            if cand.empty:
                continue
            snap = cand.iloc[-1]
            anc = anchor_df[(anchor_df["factor"] == factor) & (anchor_df["release_date"] == rdate)]
            if anc.empty:
                continue
            anc = anc.iloc[0]
            rows.append(
                {
                    "factor": factor,
                    "release_date": rdate,
                    "horizon_days": int(horizon),
                    "pred_mean": snap.get("mean", np.nan),
                    "pred_median": snap.get("median", np.nan),
                    "pred_mode": snap.get("mode", np.nan),
                    "pred_std": snap.get("std", np.nan),
                    "anchor_mean": anc.get("anchor_mean", np.nan),
                    "anchor_median": anc.get("anchor_median", np.nan),
                    "anchor_mode": anc.get("anchor_mode", np.nan),
                    "anchor_std": anc.get("anchor_std", np.nan),
                    "num_strikes": snap.get("num_strikes", np.nan),
                }
            )

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        raise RuntimeError("No pre-release snapshots found for selected horizons")

    summary_rows = []
    for (factor, horizon), g in pair_df.groupby(["factor", "horizon_days"], dropna=False):
        for f in ["mean", "median", "mode", "std"]:
            s = _stats(g[f"pred_{f}"], g[f"anchor_{f}"])
            summary_rows.append(
                {
                    "factor": factor,
                    "horizon_days": int(horizon),
                    "feature": f,
                    **s,
                    "avg_num_strikes": float(pd.to_numeric(g["num_strikes"], errors="coerce").mean()),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(["factor", "horizon_days", "spearman"], ascending=[True, True, False])

    OUT.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(OUT / "kalshi_factor_predictiveness_pairs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "kalshi_factor_predictiveness_quickscan.csv", index=False, encoding="utf-8-sig")

    top = summary.sort_values(["spearman", "pearson"], ascending=[False, False]).head(30)
    top.to_csv(OUT / "kalshi_factor_predictiveness_top30.csv", index=False, encoding="utf-8-sig")

    print(f"Saved pairs to {OUT / 'kalshi_factor_predictiveness_pairs.csv'}")
    print(f"Saved summary to {OUT / 'kalshi_factor_predictiveness_quickscan.csv'}")
    print("\nTop rows:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
