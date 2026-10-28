from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"
REPORT = ROOT / "output" / "report" / "yield_proxy_timing_tdd"
REPORT.mkdir(parents=True, exist_ok=True)

ENTRY_OFFSETS = [1, 3, 5, 7]
HOLD_DAYS = 1
YIELD_FILES = ["DGS5.csv", "DGS10.csv", "DGS20.csv"]
SIGNAL_FILE = OUT / "kalshi_event_signals_T3.csv"


def _load_signals() -> pd.DataFrame:
    if not SIGNAL_FILE.exists():
        raise FileNotFoundError(f"Missing signal file: {SIGNAL_FILE}")
    sig = pd.read_csv(SIGNAL_FILE)
    sig["fomc_date"] = pd.to_datetime(sig["fomc_date"], errors="coerce")
    sig["trade_date"] = pd.to_datetime(sig["trade_date"], errors="coerce")
    sig = sig.dropna(subset=["fomc_date", "trade_date"]).sort_values("trade_date")
    # True signal: higher expected change / more hike pressure => long yield.
    # On a yield series, positive PnL comes from rates rising when signal is up.
    for c in ["expected_change_bps", "ffr_pred_change_bps", "mean", "median", "mode", "prob_hike", "prob_cut", "prob_hold"]:
        if c in sig.columns:
            sig[c] = pd.to_numeric(sig[c], errors="coerce")
    if "ffr_pred_change_bps" in sig.columns and sig["ffr_pred_change_bps"].notna().any():
        base = sig["ffr_pred_change_bps"]
    elif "expected_change_bps" in sig.columns and sig["expected_change_bps"].notna().any():
        base = sig["expected_change_bps"]
    else:
        # Fallback only if direct change proxies are unavailable: probability imbalance.
        base = (sig.get("prob_hike", 0).fillna(0) - sig.get("prob_cut", 0).fillna(0)) * 100.0
    sig["signal_strength"] = pd.Series(base).astype(float)
    sig["signal_dir"] = np.where(sig["signal_strength"] > 0, 1, np.where(sig["signal_strength"] < 0, -1, 0))
    return sig[["fomc_date", "trade_date", "signal_strength", "signal_dir", "mode_action"] + [c for c in ["mean", "median", "mode", "expected_change_bps", "ffr_pred_change_bps", "prob_hike", "prob_cut", "prob_hold"] if c in sig.columns]]


def _load_series(path: Path) -> pd.DataFrame:
    col = path.stem
    df = pd.read_csv(path)
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["observation_date", col]).sort_values("observation_date")
    return df[["observation_date", col]]


def _backtest(series: pd.DataFrame, col: str, events: pd.DataFrame, entry_offset: int) -> pd.DataFrame:
    d2i = {pd.Timestamp(d).floor("D"): i for i, d in enumerate(series["observation_date"])}
    rows = []
    for _, r in events.iterrows():
        event_date = pd.Timestamp(r["fomc_date"]).floor("D")
        if event_date not in d2i:
            continue
        i = d2i[event_date]
        entry_i = i - entry_offset
        exit_i = i + HOLD_DAYS
        if entry_i < 0 or exit_i >= len(series):
            continue
        entry_date = series.iloc[entry_i]["observation_date"]
        exit_date = series.iloc[exit_i]["observation_date"]
        entry_y = float(series.iloc[entry_i][col])
        exit_y = float(series.iloc[exit_i][col])
        yield_change_bps = (exit_y - entry_y) * 100.0
        strategy_ret_bps = float(r["signal_dir"]) * yield_change_bps
        rows.append(
            {
                "series": col,
                "fomc_date": event_date,
                "trade_date": r["trade_date"],
                "entry_offset": entry_offset,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_yield": entry_y,
                "exit_yield": exit_y,
                "yield_change_bps": yield_change_bps,
                "signal_strength": float(r["signal_strength"]),
                "signal_dir": int(r["signal_dir"]),
                "strategy_ret_bps": strategy_ret_bps,
                "strategy_ret_pct": strategy_ret_bps / 100.0,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["entry_date", "series", "entry_offset"]).reset_index(drop=True)
        out["cum_nav"] = (1.0 + out["strategy_ret_pct"].fillna(0.0)).cumprod()
    return out


def _summary(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {"n_trades": 0, "win_rate": np.nan, "direction_hit_rate": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "ic": np.nan}
    r = bt["strategy_ret_pct"].astype(float)
    nav = (1.0 + r).cumprod()
    win_rate = float((r > 0).mean())
    direction_hit_rate = float(((bt["signal_strength"] * bt["yield_change_bps"]) > 0).mean())
    mu = float(r.mean())
    sd = float(r.std())
    sharpe = (mu / (sd + 1e-12)) * np.sqrt(252.0) if sd > 0 else np.nan
    max_dd = float((nav / nav.cummax() - 1.0).min())
    ic = float(bt["signal_strength"].corr(bt["yield_change_bps"], method="spearman")) if len(bt) >= 3 else np.nan
    return {"n_trades": int(len(bt)), "win_rate": win_rate, "direction_hit_rate": direction_hit_rate, "sharpe": sharpe, "max_drawdown": max_dd, "ic": ic}


def main() -> None:
    events = _load_signals()
    series_paths = [ROOT / p for p in YIELD_FILES if (ROOT / p).exists()]
    if not series_paths:
        raise FileNotFoundError("No DGS5/DGS10/DGS20 files found")

    all_trades = []
    summary_rows = []
    for path in series_paths:
        series = _load_series(path)
        col = path.stem
        per_series_dir = REPORT / col.lower()
        per_series_dir.mkdir(parents=True, exist_ok=True)
        for off in ENTRY_OFFSETS:
            bt = _backtest(series, col, events, off)
            bt.to_csv(per_series_dir / f"trades_offset_{off}.csv", index=False, encoding="utf-8-sig")
            summary_rows.append({"series": col, "entry_offset": off, **_summary(bt)})
            if not bt.empty:
                all_trades.append(bt)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary = pd.DataFrame(summary_rows).sort_values(["series", "entry_offset"])
    trades.to_csv(REPORT / "yield_proxy_trades_all.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(REPORT / "yield_proxy_summary_all.csv", index=False, encoding="utf-8-sig")
    print(f"Saved trades -> {REPORT / 'yield_proxy_trades_all.csv'}")
    print(f"Saved summary -> {REPORT / 'yield_proxy_summary_all.csv'}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
