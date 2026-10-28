from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "kalshi"
MARKETS_DIR = DATA / "markets"
TRADES_DIR = DATA / "trades"
OUT = ROOT / "out_csv"
REPORT = ROOT / "output" / "report" / "kalshi_mode_rebuilt"
REPORT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "fomc": "ffr",
    "cpi": "cpi",
    "u3": "u3",
    "pcecore": "pcecore",
}

# Conservative, structured parsing of Kalshi titles.
# We only keep markets with clear macro contracts.
FOMC_TITLE_PATTERNS = [r"federal funds rate", r"fomc", r"fed funds rate", r"target rate"]
MACRO_TITLE_PATTERNS = {
    "cpi": [r"headline cpi", r"cpi yo?y", r"cpi mom", r"cpi for year"],
    "u3": [r"unemployment rate", r"u3"],
    "pcecore": [r"core pce", r"pce core", r"pce inflation"],
}

WINDOWS = [1, 3, 5, 7]
TOLERANCE = {"fomc": 0.25, "cpi": 0.20, "u3": 0.20, "pcecore": 0.20}


def _glob_parquets(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.parquet"))


def _safe_read_parquet(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.read_parquet(path)


def _iter_frames(folder: Path) -> Iterable[pd.DataFrame]:
    for p in _glob_parquets(folder):
        try:
            yield pd.read_parquet(p)
        except Exception:
            continue


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _match_patterns(text: str, patterns: List[str]) -> bool:
    t = _normalize_text(text)
    return any(re.search(p, t) for p in patterns)


def _extract_fomc_release_date(title: str) -> Optional[pd.Timestamp]:
    t = _normalize_text(title)
    m = re.search(r"(\d{4})[- ]([a-z]{3,9})(?:[- ](\d{1,2}))?", t)
    if m:
        # Not robust enough for all titles; prefer event_ticker when available.
        return None
    return None


def _identify_target(title: str) -> Optional[str]:
    if _match_patterns(title, FOMC_TITLE_PATTERNS):
        return "fomc"
    for target, pats in MACRO_TITLE_PATTERNS.items():
        if _match_patterns(title, pats):
            return target
    return None


def _load_markets() -> pd.DataFrame:
    rows = []
    for df in _iter_frames(MARKETS_DIR):
        cols = {c.lower(): c for c in df.columns}
        if "ticker" not in cols or "title" not in cols:
            continue
        sub = df.copy()
        sub["ticker"] = sub[cols["ticker"]].astype(str)
        sub["title"] = sub[cols["title"]].astype(str)
        if "event_ticker" in cols:
            sub["event_ticker"] = sub[cols["event_ticker"]].astype(str)
        else:
            sub["event_ticker"] = None
        if "close_time" in cols:
            sub["close_time"] = pd.to_datetime(sub[cols["close_time"]], errors="coerce", utc=True)
        else:
            sub["close_time"] = pd.NaT
        if "open_time" in cols:
            sub["open_time"] = pd.to_datetime(sub[cols["open_time"]], errors="coerce", utc=True)
        else:
            sub["open_time"] = pd.NaT
        sub["target"] = sub["title"].map(_identify_target)
        rows.append(sub[["ticker", "event_ticker", "title", "target", "open_time", "close_time"]])
    if not rows:
        raise FileNotFoundError("No market parquet files found")
    markets = pd.concat(rows, ignore_index=True)
    markets = markets.dropna(subset=["target", "ticker"])
    markets = markets.drop_duplicates(subset=["ticker"])
    return markets


def _load_trades(tickers: set[str]) -> pd.DataFrame:
    rows = []
    for p in _glob_parquets(TRADES_DIR):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        if "ticker" not in cols or "yes_price" not in cols or "created_time" not in cols:
            continue
        sub = df[[cols["ticker"], cols["yes_price"], cols["created_time"]]].copy()
        sub.columns = ["ticker", "yes_price", "created_time"]
        sub["ticker"] = sub["ticker"].astype(str)
        sub = sub[sub["ticker"].isin(tickers)]
        if sub.empty:
            continue
        sub["yes_price"] = pd.to_numeric(sub["yes_price"], errors="coerce")
        sub["created_time"] = pd.to_datetime(sub["created_time"], errors="coerce", utc=True)
        sub = sub.dropna(subset=["ticker", "yes_price", "created_time"])
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["ticker", "yes_price", "created_time"])
    return pd.concat(rows, ignore_index=True)


def _daily_last_price(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    t = trades.copy()
    t["trade_date"] = t["created_time"].dt.tz_convert(None).dt.floor("D")
    t = t.sort_values(["ticker", "trade_date", "created_time"])
    idx = t.groupby(["ticker", "trade_date"])['created_time'].idxmax()
    return t.loc[idx, ["ticker", "trade_date", "yes_price"]].reset_index(drop=True)


def _build_fomc_mode(markets: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    # For each event_ticker and day, get the latest price across strikes.
    # The mode is the strike with maximum marginal probability from the CDF deltas.
    out_rows = []
    for event_ticker, g in markets.groupby("event_ticker", dropna=False):
        gg = g.copy()
        # Need only FOMC-ish markets with same event_ticker.
        if gg.empty:
            continue
        # Extract strike from ticker suffix T4.50 / T3.25 etc.
        gg["strike"] = gg["ticker"].str.extract(r"T([0-9]+(?:\.[0-9]+)?)")[0]
        gg["strike"] = pd.to_numeric(gg["strike"], errors="coerce")
        gg = gg.dropna(subset=["strike"]).sort_values("strike")
        if gg.empty:
            continue
        tt = trades[trades["ticker"].isin(set(gg["ticker"]))].copy()
        if tt.empty:
            continue
        daily = _daily_last_price(tt)
        if daily.empty:
            continue
        pivot = daily.pivot(index="trade_date", columns="ticker", values="yes_price")
        for dt, row in pivot.iterrows():
            prices = gg[["ticker", "strike"]].merge(row.rename("yes_price"), left_on="ticker", right_index=True, how="left")
            prices["yes_price"] = prices["yes_price"].ffill().bfill()
            # convert yes price [0,100] to probability [0,1]
            prices = prices.sort_values("strike")
            probs = prices["yes_price"] / 100.0
            # tail to pdf: margin between cumulative yes prices.
            pdf = []
            prev = 1.0
            for p in probs.tolist():
                pdf.append(max(prev - p, 0.0))
                prev = p
            pdf.append(max(prev, 0.0))
            pdf = np.array(pdf, dtype=float)
            if pdf.sum() <= 0:
                continue
            mode_idx = int(np.argmax(pdf))
            # interval corresponding to mode_idx
            if mode_idx == 0:
                low = -np.inf
                high = float(prices.iloc[0]["strike"])
            elif mode_idx >= len(prices):
                low = float(prices.iloc[-1]["strike"])
                high = np.inf
            else:
                low = float(prices.iloc[mode_idx - 1]["strike"])
                high = float(prices.iloc[mode_idx]["strike"])
            mode_center = (low + high) / 2 if np.isfinite(low) and np.isfinite(high) else (high if np.isfinite(high) else low)
            out_rows.append(
                {
                    "target": "fomc",
                    "event_ticker": event_ticker,
                    "trade_date": dt,
                    "mode": mode_center,
                    "mode_low": low,
                    "mode_high": high,
                    "n_strikes": int(len(prices)),
                }
            )
    return pd.DataFrame(out_rows)


def _build_simple_mode(markets: pd.DataFrame, trades: pd.DataFrame, target: str) -> pd.DataFrame:
    # Lightweight first pass for monthly macro markets: use market title grouping and strike extraction if available.
    tg = markets[markets["target"] == target].copy()
    if tg.empty:
        return pd.DataFrame(columns=["target", "ticker", "trade_date", "mode", "mode_low", "mode_high", "n_strikes"])
    # Try to extract a numeric strike from ticker title or ticker string.
    tg["strike"] = tg["ticker"].str.extract(r"T([0-9]+(?:\.[0-9]+)?)")[0]
    tg["strike"] = pd.to_numeric(tg["strike"], errors="coerce")
    tg = tg.dropna(subset=["strike"])
    if tg.empty:
        return pd.DataFrame(columns=["target", "ticker", "trade_date", "mode", "mode_low", "mode_high", "n_strikes"])
    tt = trades[trades["ticker"].isin(set(tg["ticker"]))].copy()
    if tt.empty:
        return pd.DataFrame(columns=["target", "ticker", "trade_date", "mode", "mode_low", "mode_high", "n_strikes"])
    daily = _daily_last_price(tt)
    out_rows = []
    for tkr, g in tg.groupby("ticker"):
        # Skip if no matching trades.
        if tkr not in set(daily["ticker"]):
            continue
        # For now, only a minimal market-level mode based on latest price.
        prices = daily[daily["ticker"] == tkr].sort_values("trade_date")
        if prices.empty:
            continue
        last = float(prices.iloc[-1]["yes_price"]) / 100.0
        out_rows.append(
            {
                "target": target,
                "ticker": tkr,
                "trade_date": prices.iloc[-1]["trade_date"],
                "mode": last,
                "mode_low": np.nan,
                "mode_high": np.nan,
                "n_strikes": int(g.shape[0]),
            }
        )
    return pd.DataFrame(out_rows)


def main() -> None:
    markets = _load_markets()
    tset = set(markets["ticker"].astype(str))
    trades = _load_trades(tset)

    fomc_mode = _build_fomc_mode(markets, trades)
    fomc_mode.to_csv(OUT / "kalshi_fomc_mode_rebuilt.csv", index=False, encoding="utf-8-sig")

    # Also write simple market-level mode outputs for macro targets to replace prior proxies.
    macro_rows = []
    for target in ["cpi", "u3", "pcecore"]:
        macro_rows.append(_build_simple_mode(markets, trades, target))
    macro = pd.concat(macro_rows, ignore_index=True) if macro_rows else pd.DataFrame()
    macro.to_csv(OUT / "kalshi_macro_mode_rebuilt.csv", index=False, encoding="utf-8-sig")

    print(f"Saved -> {OUT / 'kalshi_fomc_mode_rebuilt.csv'}")
    print(f"Saved -> {OUT / 'kalshi_macro_mode_rebuilt.csv'}")
    print(f"Markets used: {len(markets):,}; Trades used: {len(trades):,}; FOMC mode rows: {len(fomc_mode):,}; Macro rows: {len(macro):,}")


if __name__ == "__main__":
    main()
