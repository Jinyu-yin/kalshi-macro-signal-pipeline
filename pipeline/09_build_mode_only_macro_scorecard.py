from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"
REPORT = ROOT / "output" / "report" / "mode_only_macro"
REPORT.mkdir(parents=True, exist_ok=True)

FRED_SERIES = {
    "cpi": "CPIAUCSL",
    "u3": "UNRATE",
    "pcecore": "PCEPILFE",
}

WINDOWS = [1, 3, 5, 7, 10]
FOMC_SIGNAL_FILE = OUT / "kalshi_event_signals_T3.csv"
FOMC_REALIZED_FILE = OUT / "fomc_realized_decisions.csv"

# Tolerance for within-tolerance hit rate
# CPI / PCE Core are YoY percent series; 0.1 means 0.1 percentage points.
# U3 is unemployment rate in percent, same tolerance.
# FOMC change is in bps.
TOLERANCE = {"cpi": 0.1, "u3": 0.1, "pcecore": 0.1, "fomc": 25.0}


def _fred_csv_path(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _load_fred_series(series_id: str) -> pd.DataFrame:
    df = pd.read_csv(_fred_csv_path(series_id))
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else "observation_date" if "observation_date" in df.columns else None
    if date_col is None:
        raise ValueError(f"Unexpected FRED columns for {series_id}: {df.columns.tolist()}")
    value_col = [c for c in df.columns if c not in {date_col}][0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col].replace(".", np.nan), errors="coerce")
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
    return df.rename(columns={date_col: "date", value_col: "realized"})[["date", "realized"]]


def _load_targets() -> Dict[str, pd.DataFrame]:
    targets: Dict[str, pd.DataFrame] = {}
    for target, series_id in FRED_SERIES.items():
        df = _load_fred_series(series_id)
        if target in {"cpi", "pcecore"}:
            df = df.sort_values("date").copy()
            df["realized"] = (df["realized"] / df["realized"].shift(12) - 1.0) * 100.0
        targets[target] = df.dropna(subset=["realized"])
    return targets


def _load_predictions(target: str) -> pd.DataFrame:
    mapping = {
        "cpi": OUT / "kalshi_cpi_distributions_daily.csv",
        "u3": OUT / "kalshi_u3_distributions_daily.csv",
        "pcecore": OUT / "kalshi_pcecore_distributions_daily.csv",
    }
    p = mapping[target]
    if not p.exists():
        raise FileNotFoundError(f"Missing prediction file for {target}: {p}")
    df = pd.read_csv(p)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    for c in ["mode", "days_to_release", "std", "num_strikes"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["release_date", "trade_date", "mode"]).sort_values(["release_date", "trade_date"])


def _load_fomc_predictions() -> pd.DataFrame:
    if not FOMC_SIGNAL_FILE.exists():
        raise FileNotFoundError(f"Missing FOMC signal file: {FOMC_SIGNAL_FILE}")
    df = pd.read_csv(FOMC_SIGNAL_FILE)
    df["fomc_date"] = pd.to_datetime(df["fomc_date"], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    for c in ["mode", "days_to_fomc_ffr", "ffr_pred_change_bps", "expected_change_bps", "prob_hike", "prob_cut", "prob_hold"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["fomc_date", "trade_date", "mode"]).sort_values(["fomc_date", "trade_date"])


def _pick_snapshot_by_window(pred: pd.DataFrame, key: str, window: int, window_col: str) -> pd.DataFrame:
    p = pred.copy()
    p = p.assign(abs_gap=(p[window_col] - window).abs())
    p = p.sort_values([key, "abs_gap", "trade_date"]).groupby(key, as_index=False).first()
    p = p.rename(columns={"mode": "prediction_mode"})
    return p[[key, "trade_date", window_col, "prediction_mode"]]


def _score_event_level(pred: pd.DataFrame, realized: pd.DataFrame, key: str, realized_col: str, window_col: str, target: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    realized = realized.copy()
    realized[key] = pd.to_datetime(realized[key], errors="coerce")
    realized = realized.dropna(subset=[key]).sort_values(key)

    for w in WINDOWS:
        p = _pick_snapshot_by_window(pred, key, w, window_col)
        merged = p.merge(realized, on=key, how="inner")
        merged = merged.dropna(subset=["prediction_mode", realized_col])
        if merged.empty:
            rows.append({"target": target, "window": w, "n": 0, "MAE": np.nan, "RMSE": np.nan, "IC": np.nan, "within-tolerance hit rate": np.nan})
            continue

        merged["error"] = merged["prediction_mode"] - merged[realized_col]
        merged["abs_error"] = merged["error"].abs()
        merged["sq_error"] = merged["error"] ** 2
        rows.append(
            {
                "target": target,
                "window": w,
                "n": int(len(merged)),
                "MAE": float(merged["abs_error"].mean()),
                "RMSE": float(np.sqrt(merged["sq_error"].mean())),
                "IC": np.nan,
                "within-tolerance hit rate": float((merged["abs_error"] <= TOLERANCE[target]).mean()),
            }
        )
    return pd.DataFrame(rows)


def _score_fomc(pred: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    realized = realized.copy()
    realized["fomc_date"] = pd.to_datetime(realized["fomc_date"], errors="coerce")
    realized = realized.dropna(subset=["fomc_date"]).sort_values("fomc_date")

    for w in WINDOWS:
        p = _pick_snapshot_by_window(pred, "fomc_date", w, "days_to_fomc_ffr")
        merged = p.merge(realized, on="fomc_date", how="inner")
        merged = merged.dropna(subset=["prediction_mode", "change_bps"])
        if merged.empty:
            rows.append({"target": "fomc", "window": w, "n": 0, "MAE": np.nan, "RMSE": np.nan, "IC": np.nan, "within-tolerance hit rate": np.nan})
            continue

        merged["error"] = merged["prediction_mode"] - merged["change_bps"]
        merged["abs_error"] = merged["error"].abs()
        merged["sq_error"] = merged["error"] ** 2
        rows.append(
            {
                "target": "fomc",
                "window": w,
                "n": int(len(merged)),
                "MAE": float(merged["abs_error"].mean()),
                "RMSE": float(np.sqrt(merged["sq_error"].mean())),
                "IC": np.nan,
                "within-tolerance hit rate": float((merged["abs_error"] <= TOLERANCE["fomc"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    realized_map = _load_targets()
    all_rows = []
    all_rows.append(_score_fomc(_load_fomc_predictions(), pd.read_csv(FOMC_REALIZED_FILE)))
    for target in ["cpi", "u3", "pcecore"]:
        all_rows.append(_score_event_level(_load_predictions(target), realized_map[target].rename(columns={"date": "release_date"}), "release_date", "realized", "days_to_release", target))

    out = pd.concat(all_rows, ignore_index=True).sort_values(["target", "window"]).reset_index(drop=True)
    out_path = REPORT / "mode_only_macro_scorecard.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Summary page
    best_target = out.groupby("target", dropna=False)["within-tolerance hit rate"].mean().sort_values(ascending=False).head(1)
    best_window = out.groupby("window", dropna=False)["within-tolerance hit rate"].mean().sort_values(ascending=False).head(1)
    summary_path = REPORT / "mode_only_macro_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Mode-only macro scorecard summary\n")
        f.write("================================\n\n")
        f.write(f"Best target by mean within-tolerance hit rate: {best_target.index[0]} ({best_target.iloc[0]:.4f})\n")
        f.write(f"Best window by mean within-tolerance hit rate: {int(best_window.index[0])} days ({best_window.iloc[0]:.4f})\n\n")
        f.write(out.to_string(index=False))

    print(f"Saved -> {out_path}")
    print(f"Saved -> {summary_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
