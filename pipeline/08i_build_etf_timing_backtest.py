from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"
REPORT = ROOT / "output" / "report" / "etf_timing"
REPORT.mkdir(parents=True, exist_ok=True)

ENTRY_OFFSETS = [1, 3, 5, 7]
HOLD_DAYS = 1
TICKERS = ["TLT", "IEF"]
SIGNAL_FILE = OUT / "kalshi_event_signals_T3.csv"


def _load_signals() -> pd.DataFrame:
    if not SIGNAL_FILE.exists():
        raise FileNotFoundError(f"Missing signal file: {SIGNAL_FILE}")
    sig = pd.read_csv(SIGNAL_FILE)
    sig["fomc_date"] = pd.to_datetime(sig["fomc_date"], errors="coerce")
    sig["trade_date"] = pd.to_datetime(sig["trade_date"], errors="coerce")
    sig = sig.dropna(subset=["fomc_date", "trade_date"]).sort_values("trade_date")
    for c in ["expected_change_bps", "ffr_pred_change_bps", "mean", "median", "mode", "prob_hike", "prob_cut", "prob_hold"]:
        if c in sig.columns:
            sig[c] = pd.to_numeric(sig[c], errors="coerce")
    if sig.get("ffr_pred_change_bps", pd.Series(dtype=float)).notna().any():
        base = sig["ffr_pred_change_bps"]
    elif sig.get("expected_change_bps", pd.Series(dtype=float)).notna().any():
        base = sig["expected_change_bps"]
    else:
        base = (sig.get("prob_hike", 0).fillna(0) - sig.get("prob_cut", 0).fillna(0)) * 100.0
    sig["signal_strength"] = pd.Series(base).astype(float)
    # Kalshi signal up => short bond ETF (yields up / bond prices down)
    sig["signal_dir"] = np.where(sig["signal_strength"] > 0, -1, np.where(sig["signal_strength"] < 0, 1, 0))
    return sig[["fomc_date", "trade_date", "signal_strength", "signal_dir", "mode_action"] + [c for c in ["mean", "median", "mode", "expected_change_bps", "ffr_pred_change_bps", "prob_hike", "prob_cut", "prob_hold"] if c in sig.columns]]


def _download_ticker(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2019-01-01", progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")
    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "date" not in df.columns:
        raise RuntimeError(f"Unexpected columns for {ticker}: {df.columns.tolist()}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df["close"] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=["close"])
    return df[["date", "close"]]


def _backtest(price: pd.DataFrame, events: pd.DataFrame, entry_offset: int) -> pd.DataFrame:
    d2i = {pd.Timestamp(d).floor("D"): i for i, d in enumerate(price["date"])}
    rows = []
    for _, r in events.iterrows():
        event_date = pd.Timestamp(r["fomc_date"]).floor("D")
        if event_date not in d2i:
            continue
        i = d2i[event_date]
        entry_i = i - entry_offset
        exit_i = i + HOLD_DAYS
        if entry_i < 0 or exit_i >= len(price):
            continue
        entry_date = price.iloc[entry_i]["date"]
        exit_date = price.iloc[exit_i]["date"]
        entry_px = float(price.iloc[entry_i]["close"])
        exit_px = float(price.iloc[exit_i]["close"])
        raw_ret_pct = exit_px / entry_px - 1.0
        strategy_ret_pct = float(r["signal_dir"]) * raw_ret_pct
        rows.append(
            {
                "fomc_date": event_date,
                "trade_date": r["trade_date"],
                "entry_offset": entry_offset,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_px": entry_px,
                "exit_px": exit_px,
                "raw_ret_pct": raw_ret_pct,
                "signal_strength": float(r["signal_strength"]),
                "signal_dir": int(r["signal_dir"]),
                "strategy_ret_pct": strategy_ret_pct,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["entry_date", "entry_offset"]).reset_index(drop=True)
        out["cum_nav"] = (1.0 + out["strategy_ret_pct"]).cumprod()
    return out


def _summary(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {"n_trades": 0, "win_rate": np.nan, "direction_hit_rate": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "ic": np.nan}
    r = bt["strategy_ret_pct"].astype(float)
    nav = (1.0 + r).cumprod()
    win_rate = float((r > 0).mean())
    direction_hit_rate = float(((bt["signal_strength"] * bt["raw_ret_pct"]) > 0).mean())
    mu = float(r.mean())
    sd = float(r.std())
    sharpe = (mu / (sd + 1e-12)) * np.sqrt(252.0) if sd > 0 else np.nan
    max_dd = float((nav / nav.cummax() - 1.0).min())
    ic = float(bt["signal_strength"].corr(bt["raw_ret_pct"], method="spearman")) if len(bt) >= 3 else np.nan
    return {"n_trades": int(len(bt)), "win_rate": win_rate, "direction_hit_rate": direction_hit_rate, "sharpe": sharpe, "max_drawdown": max_dd, "ic": ic}


def main() -> None:
    events = _load_signals()
    all_trades = []
    summary_rows = []
    for ticker in TICKERS:
        price = _download_ticker(ticker)
        tdir = REPORT / ticker.lower()
        tdir.mkdir(parents=True, exist_ok=True)
        for off in ENTRY_OFFSETS:
            bt = _backtest(price, events, off)
            bt.to_csv(tdir / f"trades_offset_{off}.csv", index=False, encoding="utf-8-sig")
            summary_rows.append({"ticker": ticker, "entry_offset": off, **_summary(bt)})
            if not bt.empty:
                bt = bt.copy()
                bt["ticker"] = ticker
                all_trades.append(bt)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary = pd.DataFrame(summary_rows).sort_values(["ticker", "entry_offset"])
    trades.to_csv(REPORT / "etf_trades_all.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(REPORT / "etf_summary_all.csv", index=False, encoding="utf-8-sig")
    print(f"Saved trades -> {REPORT / 'etf_trades_all.csv'}")
    print(f"Saved summary -> {REPORT / 'etf_summary_all.csv'}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
