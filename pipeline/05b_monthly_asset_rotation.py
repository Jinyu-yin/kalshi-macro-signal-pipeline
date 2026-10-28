from __future__ import annotations

import argparse
import math
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.utils_paths import output_dir, output_topic_signals_dir, zt_1min_path
from pipeline.utils_topics import discover_poly_topic_files

_YAHOO_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
_FRED_PROXY = {
    "SPY": "SP500",
    "GLD": "GOLDAMGBD228NLBM",
    "UUP": "DTWEXBGS",
}


def _load_local_future_daily(symbol: str) -> pd.Series:
    p = zt_1min_path(symbol)
    if p is None:
        raise FileNotFoundError(f"{symbol} 1min csv not found under data/macro or src/analysis.")
    df = pd.read_csv(p, header=None, low_memory=False)
    df.columns = ["ts", "c2", "c3", "c4", "open", "high", "low", "close", "volume", "symbol"]
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["ts", "close"]).sort_values("ts")
    s = df.set_index("ts")["close"]
    if not s.index.is_unique:
        s = s.groupby(level=0).last().sort_index()
    daily = s.groupby(s.index.floor("D")).last().sort_index()
    daily.name = symbol
    return daily


def _yahoo_chart_daily(symbol: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.Series:
    start_utc = pd.Timestamp(start_utc)
    end_utc = pd.Timestamp(end_utc)
    if start_utc.tz is None:
        start_utc = start_utc.tz_localize("UTC")
    if end_utc.tz is None:
        end_utc = end_utc.tz_localize("UTC")
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(
        url,
        params={"period1": int(start_utc.timestamp()), "period2": int(end_utc.timestamp()), "interval": "1d"},
        headers={
            "User-Agent": _YAHOO_UA,
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://finance.yahoo.com/",
            "Origin": "https://finance.yahoo.com",
        },
        timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    res = j["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True, errors="coerce")
    close = pd.Series(res["indicators"]["quote"][0]["close"], index=idx).dropna()
    close = close[~close.index.duplicated(keep="last")].astype(float).sort_index()
    return close


def _fred_daily(series_id: str) -> pd.Series:
    r = requests.get(_FRED_CSV.format(id=series_id), timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    col = series_id if series_id in df.columns else df.columns[-1]
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    s = pd.to_numeric(df[col], errors="coerce")
    out = pd.Series(s.values, index=df[date_col]).dropna().sort_index()
    return out


def _download_yf_daily(symbol: str, cache_dir: Path) -> pd.Series:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol.replace('=','_')}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if {"Date", "Close"}.issubset(cached.columns):
            cached["Date"] = pd.to_datetime(cached["Date"], utc=True, errors="coerce")
            cached["Close"] = pd.to_numeric(cached["Close"], errors="coerce")
            s = cached.dropna(subset=["Date", "Close"]).set_index("Date")["Close"].sort_index()
            s.name = symbol
            return s

    s = pd.Series(dtype=float)
    try:
        s = _yahoo_chart_daily(symbol, pd.Timestamp("1990-01-01", tz="UTC"), pd.Timestamp.now(tz="UTC"))
    except Exception:
        s = pd.Series(dtype=float)

    if s.empty and symbol.upper() in _FRED_PROXY:
        sid = _FRED_PROXY[symbol.upper()]
        s = _fred_daily(sid)

    if s.empty:
        raise RuntimeError(f"Failed to download symbol={symbol} from yahoo/fred")
    s.name = symbol
    pd.DataFrame({"Date": s.index, "Close": s.values}).to_csv(cache_path, index=False)
    return s


def load_asset_prices(asset_symbols: List[str], cache_dir: Path) -> pd.DataFrame:
    rows = []
    for sym in asset_symbols:
        sym_u = sym.strip().upper()
        try:
            if sym_u in {"ZT", "ZN"}:
                s = _load_local_future_daily(sym_u)
            else:
                s = _download_yf_daily(sym, cache_dir)
            rows.append(s.rename(sym_u))
        except Exception as e:
            print(f"[05b][WARN] skip symbol={sym} reason={e}")
            continue
    if not rows:
        raise RuntimeError("No asset price series available. Check local csv or yfinance/network.")
    return pd.concat(rows, axis=1).sort_index()


def load_factor_signals(strategies: List[str], freqs: List[str]) -> pd.DataFrame:
    topic_files = discover_poly_topic_files()
    series_list = []
    for tf in topic_files:
        topic = tf.topic
        for strategy in strategies:
            for freq in freqs:
                path = output_topic_signals_dir(topic) / strategy / freq / "event_signal.parquet"
                if not path.exists():
                    continue
                df = pd.read_parquet(path)
                if "ts" not in df.columns or "signal_volume_weighted" not in df.columns:
                    continue
                df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                df["signal_volume_weighted"] = pd.to_numeric(df["signal_volume_weighted"], errors="coerce")
                df = df.dropna(subset=["ts", "signal_volume_weighted"]).sort_values("ts")
                if df.empty:
                    continue
                daily = (
                    df.assign(d=df["ts"].dt.floor("D"))
                    .groupby("d", as_index=True)["signal_volume_weighted"]
                    .mean()
                    .sort_index()
                )
                factor_name = f"{topic}__{strategy}__{freq}"
                series_list.append(daily.rename(factor_name))
    if not series_list:
        return pd.DataFrame()
    fac = pd.concat(series_list, axis=1).sort_index()
    return fac.fillna(0.0)


def _to_daily_index(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out[~out.index.isna()]
    out["_d"] = out.index.floor("D")
    if how == "last":
        out = out.groupby("_d", as_index=True).last()
    else:
        out = out.groupby("_d", as_index=True).mean()
    out.index.name = None
    return out.sort_index()


def _corr_with_lag(signal: pd.Series, ret: pd.Series) -> float:
    x = signal.shift(1)
    joined = pd.concat([x.rename("x"), ret.rename("y")], axis=1).dropna()
    if len(joined) < 30:
        return 0.0
    std_x = float(joined["x"].std())
    std_y = float(joined["y"].std())
    if std_x <= 1e-12 or std_y <= 1e-12:
        return 0.0
    return float(joined["x"].corr(joined["y"]))


def monthly_rotation(
    factor_daily: pd.DataFrame,
    asset_ret: pd.DataFrame,
    top_pct: float,
    lookback_days: int,
    min_abs_corr: float,
    max_asset_weight: float,
    min_active_assets: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_idx = factor_daily.index.intersection(asset_ret.index)
    factor_daily = factor_daily.loc[common_idx].sort_index()
    asset_ret = asset_ret.loc[common_idx].sort_index()
    if factor_daily.empty or asset_ret.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if len(common_idx) < 40:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Auto-adjust long lookback when data window is short.
    eff_lookback = min(int(lookback_days), max(30, int(len(common_idx) * 0.6)))
    min_train_days = max(30, min(60, eff_lookback // 2))

    months = pd.Series(common_idx).dt.to_period("M")
    month_starts = (
        pd.DataFrame({"d": common_idx, "m": months})
        .groupby("m", as_index=False)["d"]
        .min()["d"]
        .tolist()
    )

    weight_rows = []
    map_rows = []
    perf_rows = []

    for m_start in month_starts:
        train_end = m_start - pd.Timedelta(days=1)
        train_start = m_start - pd.Timedelta(days=eff_lookback)
        train_idx = common_idx[(common_idx >= train_start) & (common_idx <= train_end)]
        if len(train_idx) < min_train_days:
            continue

        # factor -> best asset by |corr|
        best = []
        for f in factor_daily.columns:
            f_sig = factor_daily.loc[train_idx, f]
            best_asset = None
            best_corr = 0.0
            for a in asset_ret.columns:
                a_ret = asset_ret.loc[train_idx, a]
                c = _corr_with_lag(f_sig, a_ret)
                if abs(c) > abs(best_corr):
                    best_corr = c
                    best_asset = a
            if best_asset is None or abs(best_corr) < min_abs_corr:
                continue
            best.append((f, best_asset, best_corr, abs(best_corr)))

        if not best:
            continue

        n_pick = max(1, int(math.ceil(len(best) * top_pct)))
        best_sorted = sorted(best, key=lambda x: x[3], reverse=True)[:n_pick]

        # factor weights by signed score
        factor_scores = np.array([x[2] for x in best_sorted], dtype=float)
        denom = float(np.sum(np.abs(factor_scores)))
        if denom <= 1e-12:
            continue
        factor_w = factor_scores / denom

        asset_w: Dict[str, float] = {a: 0.0 for a in asset_ret.columns}
        for (f, a, c, _), w in zip(best_sorted, factor_w):
            asset_w[a] += float(w)
            map_rows.append(
                {
                    "month_start": m_start,
                    "factor": f,
                    "mapped_asset": a,
                    "corr_score": c,
                    "factor_weight": float(w),
                }
            )

        raw_asset_w = dict(asset_w)

        # Cap concentration. Do NOT renormalize gross exposure back to 1 blindly,
        # otherwise a single surviving asset becomes 100% and defeats the cap.
        for a in list(asset_w.keys()):
            asset_w[a] = float(np.clip(asset_w[a], -max_asset_weight, max_asset_weight))
        gross = float(np.sum(np.abs(list(asset_w.values()))))
        if gross <= 1e-12:
            continue
        if gross > 1.0:
            scale = 1.0 / gross
            for a in list(asset_w.keys()):
                asset_w[a] = float(asset_w[a] * scale)

        # Enforce minimum active assets to avoid accidental single-asset concentration.
        active_now = [a for a, v in asset_w.items() if abs(v) > 1e-9]
        k = max(1, int(min_active_assets))
        if len(active_now) < k:
            # Fallback: use top-k (factor, asset) pairs by |corr| from the selected set.
            ranked_pairs = sorted(best, key=lambda x: abs(x[2]), reverse=True)
            picked_pairs = ranked_pairs[:k]
            if len(picked_pairs) < k:
                continue
            per_abs = min(float(max_asset_weight), 1.0 / float(k))
            asset_w = {a: 0.0 for a in asset_ret.columns}
            for _, a, c, _ in picked_pairs:
                sgn = 1.0 if c >= 0 else -1.0
                asset_w[a] += float(per_abs * sgn)
            for a in list(asset_w.keys()):
                asset_w[a] = float(np.clip(asset_w[a], -max_asset_weight, max_asset_weight))
            gross2 = float(np.sum(np.abs(list(asset_w.values()))))
            if gross2 > 1.0:
                scale2 = 1.0 / gross2
                for a in list(asset_w.keys()):
                    asset_w[a] = float(asset_w[a] * scale2)

        row = {"month_start": m_start}
        row.update(asset_w)
        weight_rows.append(row)

        # realized monthly pnl with static monthly weights
        m_end = (m_start + pd.offsets.MonthEnd(0)).normalize()
        test_idx = common_idx[(common_idx >= m_start) & (common_idx <= m_end)]
        if len(test_idx) == 0:
            continue
        pnl = asset_ret.loc[test_idx].mul(pd.Series(asset_w), axis=1).sum(axis=1)
        perf_rows.append(
            {
                "month_start": m_start,
                "n_days": int(len(test_idx)),
                "month_return": float((1.0 + pnl).prod() - 1.0),
                "month_vol": float(pnl.std()),
                "active_assets": int(sum(abs(v) > 1e-9 for v in asset_w.values())),
                "n_factors_selected": int(len(best_sorted)),
            }
        )

    return pd.DataFrame(weight_rows), pd.DataFrame(map_rows), pd.DataFrame(perf_rows)


def _evaluate_monthly_perf(p_df: pd.DataFrame) -> dict:
    if p_df.empty or "month_return" not in p_df.columns:
        return {
            "n_months": 0,
            "cum_return": np.nan,
            "avg_month_return": np.nan,
            "std_month_return": np.nan,
            "sharpe_monthly": np.nan,
        }
    r = pd.to_numeric(p_df["month_return"], errors="coerce").dropna()
    if r.empty:
        return {
            "n_months": 0,
            "cum_return": np.nan,
            "avg_month_return": np.nan,
            "std_month_return": np.nan,
            "sharpe_monthly": np.nan,
        }
    avg = float(r.mean())
    std = float(r.std())
    sharpe = float((avg / (std + 1e-12)) * np.sqrt(12.0)) if std > 0 else 0.0
    cum = float((1.0 + r).prod() - 1.0)
    return {
        "n_months": int(len(r)),
        "cum_return": cum,
        "avg_month_return": avg,
        "std_month_return": std,
        "sharpe_monthly": sharpe,
    }


def _run_grid_search(
    factor_daily: pd.DataFrame,
    returns: pd.DataFrame,
    report_dir: Path,
    min_active_assets: int,
) -> pd.DataFrame:
    grid_top_pct = [0.10, 0.15, 0.20, 0.25]
    grid_lookback = [63, 126, 189, 252]
    grid_min_corr = [0.00, 0.02, 0.05]
    grid_max_w = [0.25, 0.35, 0.50]
    rows = []
    for top_pct in grid_top_pct:
        for lookback in grid_lookback:
            for min_corr in grid_min_corr:
                for max_w in grid_max_w:
                    w_df, m_df, p_df = monthly_rotation(
                        factor_daily=factor_daily,
                        asset_ret=returns,
                        top_pct=float(top_pct),
                        lookback_days=int(lookback),
                        min_abs_corr=float(min_corr),
                        max_asset_weight=float(max_w),
                        min_active_assets=int(min_active_assets),
                    )
                    met = _evaluate_monthly_perf(p_df)
                    rows.append(
                        {
                            "top_pct": float(top_pct),
                            "lookback_days": int(lookback),
                            "min_abs_corr": float(min_corr),
                            "max_asset_weight": float(max_w),
                            "n_months": int(met["n_months"]),
                            "cum_return": float(met["cum_return"]) if pd.notna(met["cum_return"]) else np.nan,
                            "avg_month_return": float(met["avg_month_return"]) if pd.notna(met["avg_month_return"]) else np.nan,
                            "std_month_return": float(met["std_month_return"]) if pd.notna(met["std_month_return"]) else np.nan,
                            "sharpe_monthly": float(met["sharpe_monthly"]) if pd.notna(met["sharpe_monthly"]) else np.nan,
                            "n_weight_months": int(len(w_df)),
                            "n_map_rows": int(len(m_df)),
                        }
                    )
    res = pd.DataFrame(rows).sort_values(
        ["sharpe_monthly", "cum_return", "n_months"],
        ascending=[False, False, False],
    )
    out = report_dir / "monthly_asset_rotation_grid_search.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly matrix rotation with expanded asset pool.")
    parser.add_argument(
        "--asset-symbols",
        type=str,
        default="ZT,ZN,SPY,GLD,USO,UUP,FXI",
        help="comma-separated tradable proxy assets",
    )
    parser.add_argument("--strategies", type=str, default="jump,momentum")
    parser.add_argument("--signal-freqs", type=str, default="30min,60min,1d")
    parser.add_argument("--top-pct", type=float, default=0.15, help="select top pct factors each month")
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--min-abs-corr", type=float, default=0.05)
    parser.add_argument("--max-asset-weight", type=float, default=0.35)
    parser.add_argument(
        "--min-active-assets",
        type=int,
        default=1,
        help="minimum number of active assets each month",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="run parameter grid search and use best params",
    )
    args = parser.parse_args()

    asset_symbols = [x.strip() for x in args.asset_symbols.split(",") if x.strip()]
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    signal_freqs = [x.strip().lower() for x in args.signal_freqs.split(",") if x.strip()]

    report_dir = output_dir() / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir() / "cache" / "asset_prices"

    factor_daily = load_factor_signals(strategies, signal_freqs)
    if factor_daily.empty:
        raise RuntimeError("No factor signals found. Run step 04 first.")

    prices = load_asset_prices(asset_symbols, cache_dir)
    prices = prices.sort_index()
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    # Align both factor and asset returns on daily calendar index.
    factor_daily = _to_daily_index(factor_daily, how="mean")
    returns = _to_daily_index(returns, how="mean")

    if args.grid_search:
        gs = _run_grid_search(factor_daily, returns, report_dir, min_active_assets=int(args.min_active_assets))
        best = gs.iloc[0].to_dict() if not gs.empty else {}
        if best:
            args.top_pct = float(best["top_pct"])
            args.lookback_days = int(best["lookback_days"])
            args.min_abs_corr = float(best["min_abs_corr"])
            args.max_asset_weight = float(best["max_asset_weight"])
            print(
                "[05b] best grid params:",
                f"top_pct={args.top_pct}",
                f"lookback_days={args.lookback_days}",
                f"min_abs_corr={args.min_abs_corr}",
                f"max_asset_weight={args.max_asset_weight}",
            )

    w_df, m_df, p_df = monthly_rotation(
        factor_daily=factor_daily,
        asset_ret=returns,
        top_pct=float(args.top_pct),
        lookback_days=int(args.lookback_days),
        min_abs_corr=float(args.min_abs_corr),
        max_asset_weight=float(args.max_asset_weight),
        min_active_assets=int(args.min_active_assets),
    )

    if w_df.empty:
        raise RuntimeError("No monthly allocation produced. Try lower min-abs-corr or longer lookback.")

    w_path = report_dir / "monthly_asset_rotation_weights.csv"
    m_path = report_dir / "monthly_factor_to_asset_map.csv"
    p_path = report_dir / "monthly_asset_rotation_performance.csv"
    nav_path = report_dir / "monthly_asset_rotation_nav.csv"

    w_df.to_csv(w_path, index=False, encoding="utf-8-sig")
    m_df.to_csv(m_path, index=False, encoding="utf-8-sig")
    p_df.to_csv(p_path, index=False, encoding="utf-8-sig")

    nav = p_df.copy()
    nav["cum_nav"] = (1.0 + nav["month_return"]).cumprod()
    nav.to_csv(nav_path, index=False, encoding="utf-8-sig")

    print(f"[05b] factors={factor_daily.shape[1]} assets={returns.shape[1]}")
    print(f"[05b] months={len(w_df)} selected_top_pct={args.top_pct}")
    print(f"[05b] saved -> {w_path}")
    print(f"[05b] saved -> {m_path}")
    print(f"[05b] saved -> {p_path}")
    print(f"[05b] saved -> {nav_path}")
    print("[05b] latest monthly weights:")
    print(w_df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
