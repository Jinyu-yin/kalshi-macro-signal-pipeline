"""
构建 Kalshi Fed Funds Rate 分布（日频）。

对齐论文的关键点：
1. Kalshi 的利率合约 ticker 通常表示“FFR >= strike”的累积概率。
2. 用相邻 strike 的 yes_price 差分，得到区间概率（PDF）。
3. 从 PDF 计算 mean / median / mode / variance / skewness / kurtosis。
4. 输出每日快照，便于后续预测评估与事件研究对齐。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "out_csv"
DEFAULT_MARKETS_PATH = OUT_CSV / "kalshi_fed_markets.csv"
DEFAULT_OUTPUT_PATH = OUT_CSV / "kalshi_ffr_distributions_daily.csv"


def parse_fed_ticker(ticker: str) -> Tuple[Optional[int], Optional[str], Optional[float]]:
    """Parse a Kalshi FFR ticker.

    Supported examples:
    - FED-25MAY-T4.50
    - KXFED-25MAY-T4.50
    """
    if not isinstance(ticker, str):
        return None, None, None
    pattern = r"^(?:KX)?FED-(\d{2})([A-Z]{3})-T(\d+(?:\.\d+)?)$"
    match = re.match(pattern, ticker.strip().upper())
    if not match:
        return None, None, None
    return int("20" + match.group(1)), match.group(2), float(match.group(3))


def _as_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=False, errors="coerce")


def locate_trade_files() -> List[Path]:
    candidates = [
        ROOT / "data" / "kalshi" / "trades",
        ROOT / "data" / "kalshi" / "trades" / "parquet",
        ROOT / "data" / "kalshi",
        ROOT / "data",
    ]
    files: List[Path] = []
    for base in candidates:
        if base.exists() and base.is_dir():
            files.extend(sorted(base.glob("**/*.parquet")))
    # de-duplicate while keeping order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def normalize_trade_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize likely Kalshi trade columns to a common schema."""
    col_map = {}
    for cand in ["created_time", "created_at", "ts", "timestamp"]:
        if cand in df.columns:
            col_map[cand] = "created_time"
            break
    for cand in ["ticker", "symbol", "market_ticker"]:
        if cand in df.columns:
            col_map[cand] = "ticker"
            break
    for cand in ["yes_price", "price", "prob", "probability"]:
        if cand in df.columns:
            col_map[cand] = "yes_price"
            break
    df = df.rename(columns=col_map).copy()
    required = {"created_time", "ticker", "yes_price"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["created_time", "ticker", "yes_price"])
    df = df.loc[:, ["created_time", "ticker", "yes_price"]].copy()
    df["created_time"] = _as_dt(df["created_time"])
    df["yes_price"] = pd.to_numeric(df["yes_price"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str)
    return df.dropna(subset=["created_time", "ticker", "yes_price"])


def load_markets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing markets file: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "ticker" not in df.columns:
        raise RuntimeError("`kalshi_fed_markets.csv` must contain a `ticker` column")
    return df


def load_trades(trade_files: Iterable[Path], tickers: set[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for i, f in enumerate(trade_files):
        if i % 500 == 0:
            print(f"  scanning {i}/{len(trade_files)} parquet files")
        try:
            raw = pd.read_parquet(f)
            df = normalize_trade_frame(raw)
            if not df.empty:
                df = df[df["ticker"].isin(tickers)]
                if not df.empty:
                    frames.append(df)
        except Exception as e:
            print(f"  skip {f.name}: {e}")
    if not frames:
        return pd.DataFrame(columns=["created_time", "ticker", "yes_price"])
    return pd.concat(frames, ignore_index=True)


def build_pdf_from_cdf(cdf: Dict[float, float]) -> Tuple[Dict[float, float], List[float]]:
    """Difference a cumulative curve into interval probabilities.

    For strikes s1 < s2 < ... < sn, if cdf[s] = P(FFR >= s), then
    P(si <= FFR < s(i+1)) = cdf[si] - cdf[s(i+1)].
    """
    strikes = sorted(cdf)
    pdf: Dict[float, float] = {}
    if len(strikes) < 2:
        return pdf, strikes
    for left, right in zip(strikes[:-1], strikes[1:]):
        p = float(cdf[left]) - float(cdf[right])
        if p > 0:
            pdf[(left + right) / 2.0] = p
    return pdf, strikes


def moments_from_pdf(pdf: Dict[float, float]) -> Optional[Dict[str, float]]:
    if not pdf:
        return None
    x = np.array(sorted(pdf.keys()), dtype=float)
    p = np.array([pdf[v] for v in x], dtype=float)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    p = p / total
    mean = float(np.sum(x * p))
    cdf = np.cumsum(p)
    median = float(x[np.searchsorted(cdf, 0.5, side="left")])
    mode = float(x[np.argmax(p)])
    var = float(np.sum(((x - mean) ** 2) * p))
    std = float(np.sqrt(var))
    skew = float(np.sum(((x - mean) ** 3) * p) / (std**3)) if std > 0 else 0.0
    kurt = float(np.sum(((x - mean) ** 4) * p) / (std**4) - 3.0) if std > 0 else 0.0
    return {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": var,
        "std": std,
        "skewness": skew,
        "kurtosis": kurt,
    }


def process_event_day(event_day_trades: pd.DataFrame) -> Optional[Dict[str, float]]:
    # last yes_price per strike on that date
    cdf: Dict[float, float] = {}
    for ticker, g in event_day_trades.groupby("ticker"):
        _, _, strike = parse_fed_ticker(ticker)
        if strike is None:
            continue
        last_trade = g.sort_values("created_time").iloc[-1]
        price = float(last_trade["yes_price"])
        # Kalshi yes-price is a probability; clip to sane bounds only for robustness.
        cdf[strike] = min(max(price, 0.0), 1.0)

    if len(cdf) < 2:
        return None

    # Ensure monotonicity of cumulative probabilities from low strike to high strike.
    ordered_strikes = sorted(cdf)
    cum = np.array([cdf[s] for s in ordered_strikes], dtype=float)
    cum = np.maximum.accumulate(cum[::-1])[::-1]  # enforce non-increasing with strike
    cdf = {s: float(v) for s, v in zip(ordered_strikes, cum)}

    pdf, strikes = build_pdf_from_cdf(cdf)
    moments = moments_from_pdf(pdf)
    if not moments:
        return None
    moments.update({
        "num_strikes": int(len(strikes)),
        "strike_min": float(min(strikes)),
        "strike_max": float(max(strikes)),
    })
    return moments


def main() -> None:
    print("=== Kalshi Fed Funds Rate 分布构建（日频）===\n")

    markets = load_markets(DEFAULT_MARKETS_PATH)
    markets["ticker"] = markets["ticker"].astype(str)
    markets[["year", "month", "strike"]] = markets["ticker"].apply(lambda x: pd.Series(parse_fed_ticker(x)))
    markets = markets.dropna(subset=["year", "month", "strike"]).copy()
    if markets.empty:
        raise RuntimeError("No FFR tickers found in kalshi_fed_markets.csv")

    markets["event"] = markets["year"].astype(int).astype(str) + "-" + markets["month"].astype(str)
    event_map = markets.groupby("event", sort=True)["ticker"].apply(list).to_dict()
    ffr_tickers = set(markets["ticker"].astype(str))

    trade_files = locate_trade_files()
    if not trade_files:
        raise FileNotFoundError("No Kalshi parquet trade files found under data/kalshi or data")

    print(f"Found {len(ffr_tickers):,} FFR tickers across {len(event_map):,} events")
    print(f"Found {len(trade_files):,} parquet files")

    trades = load_trades(trade_files, ffr_tickers)
    if trades.empty:
        raise RuntimeError("No FFR trades matched the selected tickers")

    print(f"Loaded {len(trades):,} matching trades")
    print(f"Time span: {trades['created_time'].min()} -> {trades['created_time'].max()}")

    trades["date"] = trades["created_time"].dt.date
    rows: List[Dict[str, object]] = []

    for event, tickers in event_map.items():
        event_trades = trades[trades["ticker"].isin(tickers)].copy()
        if event_trades.empty:
            continue
        for d, g in event_trades.groupby("date", sort=True):
            moments = process_event_day(g)
            if moments:
                moments.update({"event": event, "date": d})
                rows.append(moments)

    if not rows:
        print("No daily distributions could be built.")
        return

    df = pd.DataFrame(rows).sort_values(["event", "date"]).reset_index(drop=True)
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\nDone. Built {len(df):,} daily distributions across {df['event'].nunique():,} events")
    print(f"Saved to {DEFAULT_OUTPUT_PATH}")
    print("\nPreview:")
    print(df.head(10).to_string(index=False))
    print("\nSummary:")
    print(df[["mean", "median", "mode", "std", "skewness", "kurtosis"]].describe().to_string())


if __name__ == "__main__":
    main()
