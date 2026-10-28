"""Rebuild Kalshi mode from raw markets + trades.

This script reconstructs the most likely interval (mode) from raw Kalshi
binary contract markets and trade prints.

Principle
---------
For cumulative "above strike" markets, each contract price approximates
P(outcome >= strike). Sorting strikes ascending allows us to recover the
probability mass of each interval by differencing adjacent cumulative prices.
The mode is the interval with the largest recovered mass.

Outputs
-------
- out_csv/kalshi_modes_rebuilt_daily.csv
- out_csv/kalshi_modes_rebuilt_fomc_daily.csv
- out_csv/kalshi_modes_rebuilt_cpi_daily.csv
- out_csv/kalshi_modes_rebuilt_u3_daily.csv
- out_csv/kalshi_modes_rebuilt_pcecore_daily.csv
- out_csv/kalshi_modes_rebuilt_summary.csv

Notes
-----
- Reads parquet files one-by-one to stay memory-safe.
- Uses only raw `data/kalshi/markets` and `data/kalshi/trades`.
- Does not assume a precomputed distribution table.
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MARKETS_DIR = ROOT / "data" / "kalshi" / "markets"
TRADES_DIR = ROOT / "data" / "kalshi" / "trades"
OUT = ROOT / "out_csv"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_PATTERNS = {
    "fomc": [r"fed funds", r"fomc", r"federal funds", r"target rate"],
    # Avoid broad inflation matches like ARGINFLATIONM, which are not US CPI.
    "cpi": [r"\bcpi\b", r"cpiyoy", r"inflation rate"],
    "u3": [r"unemployment", r"\bu3\b"],
    "pcecore": [r"core pce", r"pce core", r"pcecore"],
}

# Conservative tolerance when later validating against realized labels.
# Not used in reconstruction itself.
MODE_COLUMNS = [
    "target",
    "event_ticker",
    "market_ticker",
    "trade_date",
    "mode_interval_low",
    "mode_interval_high",
    "mode_interval_mass",
    "mode_probability_mode",
    "n_contracts",
]

STRIKE_RE = re.compile(r"-T(?P<strike>\d+(?:\.\d+)?)$")


def _match_target(text: str) -> Optional[str]:
    t = (text or "").lower()
    for target, pats in TARGET_PATTERNS.items():
        if any(re.search(p, t) for p in pats):
            return target
    return None


def _parse_strike(ticker: str) -> Optional[float]:
    m = STRIKE_RE.search(ticker or "")
    if not m:
        return None
    try:
        return float(m.group("strike"))
    except Exception:
        return None


def _safe_read_parquet(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        # Fallback to full read if column projection isn't supported by engine.
        return pd.read_parquet(path)


def _scan_markets() -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for p in sorted(MARKETS_DIR.glob("*.parquet")):
        df = _safe_read_parquet(p, columns=["ticker", "event_ticker", "title", "status", "close_time", "expected_expiration_time"])
        if df.empty:
            continue
        text = (
            df["ticker"].astype(str).str.lower().fillna("") + " "
            + df["title"].astype(str).str.lower().fillna("") + " "
            + df["event_ticker"].astype(str).str.lower().fillna("")
        )
        target = pd.Series(np.nan, index=df.index, dtype="object")
        for k, pats in TARGET_PATTERNS.items():
            mask = pd.Series(False, index=df.index)
            for pat in pats:
                mask |= text.str.contains(pat, regex=True, na=False)
            target.loc[mask & target.isna()] = k
        # Keep CPI as headline CPI only. Exclude core-CPI and non-outcome text/release markets.
        bad_cpi = (
            (target == "cpi")
            & (
                text.str.contains(r"corecpi|cpicore|core cpi|releasecpi|will bureau of labor statistics|press secretary|fedmention", regex=True, na=False)
            )
        )
        target.loc[bad_cpi] = np.nan
        keep = target.notna()
        if not keep.any():
            continue
        cols = [c for c in ["ticker", "event_ticker", "title", "status", "close_time", "expected_expiration_time"] if c in df.columns]
        out = df.loc[keep, cols].copy()
        for c in ["ticker", "event_ticker", "title", "status", "close_time", "expected_expiration_time"]:
            if c not in out.columns:
                out[c] = pd.NA
        out["target"] = target.loc[keep].values
        out["strike"] = out["ticker"].astype(str).map(_parse_strike)
        out = out.dropna(subset=["strike"])
        expected = out["expected_expiration_time"].astype("string")
        close = out["close_time"].astype("string")
        event_base = expected.where(expected.notna() & (expected != ""), close)
        out["event_ts"] = pd.to_datetime(event_base, errors="coerce", utc=True)
        out = out.dropna(subset=["event_ts"])
        if out.empty:
            continue
        parts.append(out[["target", "ticker", "event_ticker", "title", "status", "event_ts", "strike"]])
    if not parts:
        raise FileNotFoundError("No target markets found in data/kalshi/markets")
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(subset=["ticker"]).sort_values(["target", "event_ticker", "strike", "ticker"]).reset_index(drop=True)


def _scan_trades(needed_tickers: set[str]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for p in sorted(TRADES_DIR.glob("*.parquet")):
        try:
            df = _safe_read_parquet(p, columns=["ticker", "yes_price", "created_time"])
        except Exception:
            continue
        if df.empty:
            continue
        df = df[df["ticker"].isin(needed_tickers)].copy()
        if df.empty:
            continue
        df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce", utc=True)
        df = df.dropna(subset=["created_time", "yes_price"])
        if df.empty:
            continue
        df["trade_date"] = df["created_time"].dt.floor("D").dt.tz_convert(None)
        df = df.sort_values(["ticker", "created_time"]).groupby(["ticker", "trade_date"], as_index=False).tail(1)
        df["trade_ts"] = pd.to_datetime(df["created_time"], errors="coerce", utc=True)
        parts.append(df[["ticker", "trade_ts", "trade_date", "yes_price"]])
    if not parts:
        raise FileNotFoundError("No matching trades found in data/kalshi/trades")
    out = pd.concat(parts, ignore_index=True)
    out["yes_price"] = pd.to_numeric(out["yes_price"], errors="coerce")
    return out.dropna(subset=["yes_price"]).drop_duplicates(subset=["ticker", "trade_date"]).reset_index(drop=True)


def _build_mode_for_group(group: pd.DataFrame) -> pd.DataFrame:
    # group columns: target, event_ticker, event_ts, trade_date, ticker, strike, yes_price
    g = group.sort_values("strike").copy()
    g["cum_p"] = g["yes_price"] / 100.0
    # In a cumulative "above strike" ladder, the probability mass for each interval
    # is the drop in cumulative probability between adjacent strikes.
    strikes = g["strike"].to_list()
    cum_p = g["cum_p"].to_list()

    # Build interval masses:
    # (-inf, s1): 1 - P(>= s1)
    # [s_i, s_{i+1}): P(>= s_i) - P(>= s_{i+1})
    # [s_last, +inf): P(>= s_last)
    intervals: List[Tuple[float, float, float]] = []
    if len(strikes) == 1:
        p = float(np.clip(cum_p[0], 0.0, 1.0))
        intervals.append((float("-inf"), strikes[0], max(0.0, 1.0 - p)))
        intervals.append((strikes[0], float("inf"), p))
    else:
        first_mass = max(0.0, 1.0 - float(cum_p[0]))
        intervals.append((float("-inf"), strikes[0], first_mass))
        for i in range(len(strikes) - 1):
            mass = float(cum_p[i] - cum_p[i + 1])
            intervals.append((strikes[i], strikes[i + 1], max(0.0, mass)))
        last_mass = max(0.0, float(cum_p[-1]))
        intervals.append((strikes[-1], float("inf"), last_mass))

    mode_interval = max(intervals, key=lambda x: x[2])
    mode_mass = mode_interval[2]
    mode_prob = float(g.loc[(g["strike"] >= mode_interval[0]) & (g["strike"] < mode_interval[1]), "cum_p"].max()) if np.isfinite(mode_interval[1]) else float(g["cum_p"].iloc[-1])
    interval_payload = json.dumps(
        [
            {"low": low, "high": high, "mass": mass}
            for low, high, mass in intervals
            if mass > 0
        ],
        allow_nan=True,
        separators=(",", ":"),
    )

    return pd.DataFrame(
        {
            "target": [group["target"].iloc[0]],
            "event_ticker": [group["event_ticker"].iloc[0]],
            "market_ticker": ["|".join(sorted(g["ticker"].unique().tolist()))],
            "trade_ts": [group["trade_ts"].max()],
            "trade_date": [group["trade_date"].iloc[0]],
            "event_ts": [group["event_ts"].iloc[0]],
            "mode_interval_low": [mode_interval[0]],
            "mode_interval_high": [mode_interval[1]],
            "mode_interval_mass": [mode_mass],
            "mode_probability_mode": [mode_prob],
            "interval_distribution": [interval_payload],
            "n_contracts": [len(g)],
        }
    )


def _rebuild_modes(markets: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    market_map = markets.set_index("ticker")[["target", "event_ticker", "event_ts", "strike"]].to_dict("index")
    trade_rows: List[Dict[str, object]] = []
    for row in trades.itertuples(index=False):
        meta = market_map.get(row.ticker)
        if meta is None:
            continue
        trade_rows.append(
            {
                "ticker": row.ticker,
                "target": meta["target"],
                "event_ticker": meta["event_ticker"],
                "event_ts": meta["event_ts"],
                "strike": meta["strike"],
                "trade_ts": row.trade_ts,
                "trade_date": row.trade_date,
                "yes_price": row.yes_price,
            }
        )
    if not trade_rows:
        raise RuntimeError("No overlap between markets and trades after mapping")
    merged = pd.DataFrame(trade_rows)
    out_rows: List[pd.DataFrame] = []
    for (target, event_ticker, event_ts, trade_date), g in merged.groupby(["target", "event_ticker", "event_ts", "trade_date"], sort=True):
        if len(g) < 2:
            continue
        g = g.sort_values("strike").drop_duplicates(subset=["strike"], keep="last")
        if len(g) < 2:
            continue
        row = _build_mode_for_group(g)
        row["event_ts"] = event_ts
        out_rows.append(row)
    if not out_rows:
        raise RuntimeError("No mode rows reconstructed")
    return pd.concat(out_rows, ignore_index=True)


def main() -> None:
    markets = _scan_markets()
    needed_tickers = set(markets["ticker"].tolist())
    trades = _scan_trades(needed_tickers)
    rebuilt = _rebuild_modes(markets, trades)

    rebuilt["trade_date"] = pd.to_datetime(rebuilt["trade_date"], errors="coerce")
    rebuilt = rebuilt.dropna(subset=["trade_date"]).sort_values(["target", "event_ticker", "event_ts", "trade_ts"])

    # Write one general table + per-target slices.
    rebuilt_out = OUT / "kalshi_modes_rebuilt_daily.csv"
    rebuilt.to_csv(rebuilt_out, index=False, encoding="utf-8-sig")
    for target in ["fomc", "cpi", "u3", "pcecore"]:
        sub = rebuilt[rebuilt["target"] == target].copy()
        sub.to_csv(OUT / f"kalshi_modes_rebuilt_{target}_daily.csv", index=False, encoding="utf-8-sig")

    summary = (
        rebuilt.groupby("target")
        .agg(n_rows=("event_ticker", "size"), n_events=("event_ticker", "nunique"), n_days=("trade_date", "nunique"))
        .reset_index()
    )
    summary.to_csv(OUT / "kalshi_modes_rebuilt_summary.csv", index=False, encoding="utf-8-sig")

    print(f"Saved -> {rebuilt_out}")
    print(f"Saved -> {OUT / 'kalshi_modes_rebuilt_summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
