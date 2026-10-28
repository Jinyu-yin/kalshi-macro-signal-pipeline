from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"
REPORT = ROOT / "output" / "report" / "zt_timing"
REPORT.mkdir(parents=True, exist_ok=True)

HOLD_DAYS = 1
ENTRY_OFFSETS = [1, 3, 5, 7]


def _load_price() -> pd.DataFrame:
    # Use existing daily ZT series if available from 1m data.
    p = ROOT / "data" / "macro" / "ZT_1min.csv"
    if not p.exists():
        p = ROOT / "src" / "analysis" / "ZT_1min.csv"
    if not p.exists():
        raise FileNotFoundError("ZT_1min.csv not found in data/macro or src/analysis")
    df = pd.read_csv(p, header=None)
    df.columns = ["ts", "c2", "c3", "c4", "open", "high", "low", "close", "volume", "symbol"]
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["ts", "close"]).sort_values("ts")
    df["date"] = df["ts"].dt.floor("D")
    daily = df.groupby("date", as_index=False)["close"].last().sort_values("date")
    daily["fwd1"] = daily["close"].shift(-HOLD_DAYS)
    daily["ret1"] = daily["fwd1"] / daily["close"] - 1.0
    return daily


def _load_signals() -> pd.DataFrame:
    # Current project already produces these forecast scorecards.
    p = REPORT.parent / "paper_bundle" / "tables" / "manager_onepage_scorecard.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing scorecard: {p}")
    return pd.read_csv(p)


def _build_event_table(price: pd.DataFrame) -> pd.DataFrame:
    # Use the paper bundle summary as the signal source and map to a simple direction.
    # We keep the first version intentionally transparent.
    events_path = OUT / "fomc_realized_decisions.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"missing events: {events_path}")
    ev = pd.read_csv(events_path)
    ev["fomc_date"] = pd.to_datetime(ev["fomc_date"], errors="coerce")
    ev = ev.dropna(subset=["fomc_date"]).sort_values("fomc_date")

    # Signal proxy based on the strongest FFR/CPI tags already validated earlier.
    # If FFR/CPI outlook rises => short ZT, else long ZT.
    # Here we use realized event change as a placeholder direction proxy for the first cut.
    # Positive change_bps = hikes -> bearish ZT.
    ev["signal_dir"] = np.where(ev["change_bps"].fillna(0) > 0, -1, 1)
    return ev[["fomc_date", "signal_dir", "change_bps", "ffr_upper", "ffr_lower"]]


def _backtest(price: pd.DataFrame, events: pd.DataFrame, entry_offset: int) -> pd.DataFrame:
    rows = []
    idx = price["date"].to_list()
    date_to_idx = {d: i for i, d in enumerate(idx)}
    for _, r in events.iterrows():
        d = r["fomc_date"]
        if pd.isna(d):
            continue
        d = pd.Timestamp(d).floor("D")
        if d not in date_to_idx:
            continue
        i = date_to_idx[d]
        entry_i = i - entry_offset
        exit_i = i + HOLD_DAYS
        if entry_i < 0 or exit_i >= len(price):
            continue
        entry_px = float(price.iloc[entry_i]["close"])
        exit_px = float(price.iloc[exit_i]["close"])
        raw_ret = exit_px / entry_px - 1.0
        strat_ret = float(r["signal_dir"]) * raw_ret
        rows.append(
            {
                "fomc_date": d,
                "entry_offset": entry_offset,
                "entry_date": price.iloc[entry_i]["date"],
                "exit_date": price.iloc[exit_i]["date"],
                "entry_px": entry_px,
                "exit_px": exit_px,
                "raw_ret": raw_ret,
                "signal_dir": int(r["signal_dir"]),
                "strat_ret": strat_ret,
                "change_bps": r["change_bps"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["cum_nav"] = (1.0 + out["strat_ret"]).cumprod()
    return out


def _summarize(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {"n_trades": 0, "win_rate": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "ic": np.nan}
    ret = bt["strat_ret"].astype(float)
    nav = (1.0 + ret).cumprod()
    win_rate = float((ret > 0).mean())
    mean_r = float(ret.mean())
    std_r = float(ret.std())
    sharpe = (mean_r / (std_r + 1e-12)) * np.sqrt(252.0) if std_r > 0 else np.nan
    dd = float((nav / nav.cummax() - 1.0).min())
    # IC against realized change_bps direction proxy (simple and transparent).
    ic = float(bt["signal_dir"].corr(bt["raw_ret"], method="spearman")) if len(bt) >= 3 else np.nan
    return {"n_trades": int(len(bt)), "win_rate": win_rate, "sharpe": sharpe, "max_drawdown": dd, "ic": ic}


def main() -> None:
    price = _load_price()
    events = _build_event_table(price)

    all_bt = []
    summary_rows = []
    for off in ENTRY_OFFSETS:
        bt = _backtest(price, events, off)
        if bt.empty:
            continue
        bt["holding_days"] = HOLD_DAYS
        bt["entry_offset"] = off
        all_bt.append(bt)
        summary_rows.append({"entry_offset": off, **_summarize(bt)})

    out_bt = pd.concat(all_bt, ignore_index=True) if all_bt else pd.DataFrame()
    out_sum = pd.DataFrame(summary_rows)

    out_bt.to_csv(REPORT / "zt_timing_trades.csv", index=False, encoding="utf-8-sig")
    out_sum.to_csv(REPORT / "zt_timing_summary.csv", index=False, encoding="utf-8-sig")

    print(f"Saved trades -> {REPORT / 'zt_timing_trades.csv'}")
    print(f"Saved summary -> {REPORT / 'zt_timing_summary.csv'}")
    if not out_sum.empty:
        print(out_sum.to_string(index=False))


if __name__ == "__main__":
    main()
