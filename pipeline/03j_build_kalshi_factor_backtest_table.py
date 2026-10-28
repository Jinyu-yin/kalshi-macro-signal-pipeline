"""
Build a compact backtest-ready table from Kalshi macro factors (CPI/U3 primary, PCE optional).

Outputs
- out_csv/kalshi_factor_backtest_table.csv
- out_csv/kalshi_factor_backtest_summary.csv
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"
INP = OUT / "kalshi_factor_predictiveness_pairs.csv"
OUT_TABLE = OUT / "kalshi_factor_backtest_table.csv"
OUT_SUM = OUT / "kalshi_factor_backtest_summary.csv"

PRIMARY = {"cpi", "u3"}
SECONDARY = {"pcecore"}


def main() -> None:
    if not INP.exists():
        raise FileNotFoundError(f"Missing input: {INP}")
    df = pd.read_csv(INP)
    if df.empty:
        raise RuntimeError("Predictiveness pairs file is empty")

    # Fast rule-based qualification for backtest pool.
    df["tier"] = np.where(df["factor"].isin(PRIMARY), "primary", np.where(df["factor"].isin(SECONDARY), "secondary", "exclude"))
    df = df[df["tier"] != "exclude"].copy()

    df["qualified"] = (
        (df["n_obs"] >= 10)
        & (df["spearman"] >= 0.75)
        & (df["avg_num_strikes"] >= 2.5)
    )

    # score for ranking backtest candidates
    df["signal_score"] = (
        0.45 * df["spearman"].fillna(0)
        + 0.35 * df["pearson"].fillna(0)
        + 0.20 * (1.0 / (1.0 + df["mae"].fillna(1.0)))
    )

    df = df.sort_values(["qualified", "tier", "signal_score"], ascending=[False, True, False]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

    summary = (
        df.groupby(["factor", "tier", "qualified"], dropna=False)
        .agg(
            rows=("factor", "size"),
            avg_spearman=("spearman", "mean"),
            avg_pearson=("pearson", "mean"),
            avg_mae=("mae", "mean"),
            avg_obs=("n_obs", "mean"),
        )
        .reset_index()
        .sort_values(["factor", "qualified"], ascending=[True, False])
    )
    summary.to_csv(OUT_SUM, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUT_TABLE}")
    print(f"Saved: {OUT_SUM}")
    print("\nTop qualified rows:")
    print(df[df["qualified"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
