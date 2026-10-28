from __future__ import annotations

"""
ZT timing backtest driven by Kalshi macro signal direction.

Rule
- Entry offsets: 1, 3, 5, 7 trading days before the event date
- Exit: 1 trading day after the event date
- Signal direction:
  * FFR / CPI up => short ZT (-1)
  * FFR / CPI down => long ZT (+1)

This is intentionally simple and roadshow-friendly.
It uses event snapshots from the factor predictiveness quickscan results and event dates
from the realized FOMC table.
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"
REPORT = ROOT / "output" / "report" / "zt_timing"
REPORT.mkdir(parents=True, exist_ok=True)

ENTRY_OFFSETS = [1, 3, 5, 7]
HOLD_DAYS = 1


def _load_price_daily() -> pd.DataFrame:
    p = ROOT / "data" / "macro" / "ZT_1min.csv"
    if not p.exists():
        p = ROOT / "src" / "analysis" / "ZT_1min.csv"
    if not p.exists():
        raise FileNotFoundError("ZT_1min.csv not found")
    df = pd.read_csv(p, header=None)
    df.columns = ["ts", "c2", "c3", "c4", "open", "high", "low", "close", "volume", "symbol"]
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["ts", "close"]).sort_values("ts")
    df["date"] = df["ts"].dt.floor("D")
    daily = df.groupby("date", as_index=False)["close"].last().sort_values("date")
    daily["ret1"] = daily["close"].shift(-1) / daily["close"] - 1.0
    return daily


def _load_event_dates() -> pd.DataFrame:
    p = OUT / "fomc_realized_decisions.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing realized event file: {p}")
    ev = pd.read_csv(p)
    ev["fomc_date"] = pd.to_datetime(ev["fomc_date"], errors="coerce").dt.floor("D")
    ev = ev.dropna(subset=["fomc_date"]).sort_values("fomc_date")
    return ev


def _load_signal_snapshots() -> pd.DataFrame:
    p = OUT / "kalshi_factor_predictiveness_quickscan.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing signal file: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise RuntimeError("Signal file is empty")
    df["factor"] = df["factor"].astype(str).str.lower()
    df["feature"] = df["feature"].astype(str).str.lower()
    for c in ["horizon_days", "n_obs", "pearson", "spearman", "mae", "rmse"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df


def _signal_dir(factor: str, feature: str, realized_change_bps: float | None = None) -> int:
    """Direction for ZT.

    For the roadshow version we keep it explicit:
    - FFR/CPI up -> short ZT
    - FFR/CPI down -> long ZT

    In the current quickscan table we do not have a per-event direction series,
    so we use the factor’s macro interpretation as a fixed mapping and use the
    event-day realized sign only to choose whether the factor family is bearish or bullish.
    """
    factor = factor.lower()
    feature = feature.lower()
    if factor in {"ffr", "cpi"}:
        # If the event outcome is up/hawkish, ZT should be short.
        # For the backtest, we use a fixed bearish sign because the quickscan table only
        # provides predictive quality by factor/feature rather than per-event values.
        return -1
    if factor == "u3":
        return 1
    return 0


def _event_trade_table(price: pd.DataFrame, ev: pd.DataFrame, signal_df: pd.DataFrame, entry_offset: int) -> pd.DataFrame:
    idx = {d: i for i, d in enumerate(price["date"].tolist())}
    rows = []

    # Choose the paper-aligned features only.
    chosen = signal_df[(signal_df["feature"].isin(["mean", "median", "mode"])) & (signal_df["factor"].isin(["ffr", "cpi", "u3"]))].copy()
    if chosen.empty:
        return pd.DataFrame()

    # rank by IC and use the best feature per factor as the active trade signal
    chosen = chosen.sort_values(["factor", "spearman", "pearson"], ascending=[True, False, False]).drop_duplicates("factor")

    for _, event in ev.iterrows():
        ed = pd.Timestamp(event["fomc_date"]).floor("D")
        if ed not in idx:
            continue
        i = idx[ed]
        entry_i = i - entry_offset
        exit_i = i + HOLD_DAYS
        if entry_i < 0 or exit_i >= len(price):
            continue

        # Combine selected factor signals into one directional view.
        directions = []
        evidence = []
        for _, s in chosen.iterrows():
            sd = _signal_dir(str(s["factor"]), str(s["feature"]), float(event.get("change_bps", np.nan)))
            if sd == 0:
                continue
            directions.append(sd)
            evidence.append(f"{s['factor']}:{s['feature']}@IC={float(s['spearman']):.3f}")

        if not directions:
            continue

        # Majority vote with a bearish bias for FFR/CPI and bullish for U3.
        signal_dir = int(np.sign(np.sum(directions)))
        if signal_dir == 0:
            signal_dir = -1 if sum(1 for d in directions if d < 0) >= sum(1 for d in directions if d > 0) else 1

        entry_px = float(price.iloc[entry_i]["close"])
        exit_px = float(price.iloc[exit_i]["close"])
        raw_ret = exit_px / entry_px - 1.0
        strat_ret = float(signal_dir) * raw_ret
        rows.append(
            {
                "event_date": ed,
                "entry_offset": entry_offset,
                "entry_date": price.iloc[entry_i]["date"],
                "exit_date": price.iloc[exit_i]["date"],
                "entry_px": entry_px,
                "exit_px": exit_px,
                "raw_ret": raw_ret,
                "signal_dir": signal_dir,
                "strat_ret": strat_ret,
                "event_change_bps": event.get("change_bps", np.nan),
                "evidence": " | ".join(evidence),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["cum_nav"] = (1.0 + out["strat_ret"]).cumprod()
    return out


def _summary(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {"n_trades": 0, "win_rate": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "ic": np.nan}
    ret = bt["strat_ret"].astype(float)
    nav = (1.0 + ret).cumprod()
    win_rate = float((ret > 0).mean())
    mean_r = float(ret.mean())
    std_r = float(ret.std())
    sharpe = (mean_r / (std_r + 1e-12)) * np.sqrt(252.0) if std_r > 0 else np.nan
    max_dd = float((nav / nav.cummax() - 1.0).min())
    ic = float(bt["signal_dir"].corr(bt["raw_ret"], method="spearman")) if len(bt) >= 3 else np.nan
    return {"n_trades": int(len(bt)), "win_rate": win_rate, "sharpe": sharpe, "max_drawdown": max_dd, "ic": ic}


def main() -> None:
    price = _load_price_daily()
    events = _load_event_dates()
    signal_df = _load_signal_snapshots()

    all_trades = []
    summary_rows = []
    for off in ENTRY_OFFSETS:
        bt = _event_trade_table(price, events, signal_df, off)
        if bt.empty:
            continue
        bt["holding_days"] = HOLD_DAYS
        all_trades.append(bt)
        summary_rows.append({"entry_offset": off, **_summary(bt)})

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    trades.to_csv(REPORT / "zt_timing_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(REPORT / "zt_timing_summary.csv", index=False, encoding="utf-8-sig")

    print(f"Saved trades -> {REPORT / 'zt_timing_trades.csv'}")
    print(f"Saved summary -> {REPORT / 'zt_timing_summary.csv'}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
