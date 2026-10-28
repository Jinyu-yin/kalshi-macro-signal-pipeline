from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pipeline.utils_paths import output_dir, output_pipeline_dir, output_topic_signals_dir, zt_1min_path
from pipeline.utils_topics import discover_poly_topic_files

HORIZONS_D = [5, 10, 20]
HORIZONS_MIN = [60, 240, 1440]
COOLDOWN_DAYS = 1
COOLDOWN_MIN = 30
MIN_RETURN_THRESHOLD = 0.00001
OUTLIER_STD_MULTIPLE = 3
STRATEGIES = ["jump", "reversal"]
SIGNAL_FREQS = ["30min", "60min", "1d"]
ENTRY_UPPER_Q = 0.85
ENTRY_LOWER_Q = 0.15
ENTRY_Z_BY_FREQ = {"30min": 1.6, "60min": 1.45, "1d": 1.2}
ROLL_WINDOW_BY_FREQ = {"30min": 96, "60min": 80, "1d": 63}
ROLL_MINP_BY_FREQ = {"30min": 24, "60min": 20, "1d": 15}
MIN_CONSENSUS_BY_FREQ = {"30min": 0.45, "60min": 0.35, "1d": 0.2}
MIN_COUNT_BY_FREQ = {"30min": 2, "60min": 2, "1d": 1}
MIN_RUN_BY_FREQ = {"30min": 2, "60min": 2, "1d": 1}
MIN_TRADES_QUALIFIED = 40
MIN_SHARPE_QUALIFIED = 0.3
MIN_PROFIT_FACTOR_QUALIFIED = 1.05
ASSET_BUCKETS_BY_SYMBOL = {
    "ZT": ["USD", "CPI"],
    "ZN": ["USD", "CPI"],
    "SPY": ["SPX"],
    "GLD": ["GOLD"],
    "USO": ["OIL", "GAS"],
    "UUP": ["USD", "USDCNY"],
    "FXI": ["CHNA_EQ", "USDCNY"],
}


def _load_asset_daily_return(asset_symbol: str) -> pd.Series:
    sym = asset_symbol.strip().upper()
    try:
        p1m = _load_price_1min(sym)
        s = p1m.groupby(p1m.index.floor("D")).last().sort_index()
    except Exception:
        s = _load_cached_daily_price(sym)
        s = s.groupby(s.index.floor("D")).last().sort_index()
    ret = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ret.name = sym
    return ret


def _run_monthly_rotation_portfolio_backtest(
    weight_csv: Path,
    out_csv: Path,
    out_metrics_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not weight_csv.exists():
        return pd.DataFrame(), pd.DataFrame()
    w = pd.read_csv(weight_csv)
    if w.empty or "month_start" not in w.columns:
        return pd.DataFrame(), pd.DataFrame()

    w["month_start"] = pd.to_datetime(w["month_start"], utc=True, errors="coerce")
    w = w.dropna(subset=["month_start"]).sort_values("month_start")
    asset_cols = [c for c in w.columns if c != "month_start"]
    if not asset_cols:
        return pd.DataFrame(), pd.DataFrame()

    returns_map = {}
    for a in asset_cols:
        try:
            returns_map[a] = _load_asset_daily_return(a)
        except Exception:
            continue
    if not returns_map:
        return pd.DataFrame(), pd.DataFrame()
    ret_df = pd.concat(returns_map.values(), axis=1).sort_index()
    ret_df = ret_df[[c for c in asset_cols if c in ret_df.columns]]
    if ret_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for i, row in w.iterrows():
        start = row["month_start"]
        end = w.iloc[i + 1]["month_start"] - pd.Timedelta(days=1) if i + 1 < len(w) else ret_df.index.max()
        month_idx = ret_df.index[(ret_df.index >= start.floor("D")) & (ret_df.index <= end.floor("D"))]
        if len(month_idx) == 0:
            continue
        weights = pd.to_numeric(row[ret_df.columns], errors="coerce").fillna(0.0)
        if float(weights.abs().sum()) <= 1e-12:
            continue
        # Respect capped monthly weights from 05b; keep residual as cash.
        gross = float(weights.abs().sum())
        if gross > 1.0:
            weights = weights / gross
        pnl = ret_df.loc[month_idx].mul(weights, axis=1).sum(axis=1)
        for ts, r in pnl.items():
            rows.append({"ts": ts, "daily_return": float(r), "month_start": start})

    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    nav = pd.DataFrame(rows).sort_values("ts")
    nav["nav"] = (1.0 + nav["daily_return"]).cumprod()
    nav.to_csv(out_csv, index=False)

    mean_r = float(nav["daily_return"].mean())
    std_r = float(nav["daily_return"].std())
    sharpe = (mean_r / (std_r + 1e-12)) * np.sqrt(252.0) if std_r > 0 else 0.0
    mdd = float((nav["nav"] / nav["nav"].cummax() - 1.0).min())
    metrics = pd.DataFrame(
        [
            {
                "strategy": "monthly_rotation_portfolio",
                "n_days": int(len(nav)),
                "cum_return": float(nav["nav"].iloc[-1] - 1.0),
                "avg_daily_return": mean_r,
                "std_daily_return": std_r,
                "sharpe": float(sharpe),
                "max_drawdown": mdd,
            }
        ]
    )
    metrics.to_csv(out_metrics_csv, index=False)
    return nav, metrics


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_freq_list(raw: str) -> list[str]:
    out = []
    for x in raw.split(","):
        v = x.strip().lower()
        if not v:
            continue
        if v in {"1d", "day", "daily"}:
            out.append("1d")
        elif v in {"30min", "60min"}:
            out.append(v)
        else:
            raise ValueError(f"Unsupported signal freq: {x}")
    return out


def _load_price_1min(asset_symbol: str) -> pd.Series:
    asset = asset_symbol.strip().upper()
    p = zt_1min_path(asset)
    if p is None:
        raise FileNotFoundError(f"{asset} price file not found under data/macro or src/analysis.")
    df = pd.read_csv(p, header=None, low_memory=False)
    df.columns = ["ts", "C2", "C3", "C4", "open", "high", "low", "close", "volume", "symbol"]
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["ts", "close"]).sort_values("ts")
    s = df.set_index("ts")["close"].sort_index()
    if not s.index.is_unique:
        s = s.groupby(level=0).last().sort_index()
    return s


def load_price_daily() -> pd.Series:
    s = _load_price_1min("ZT")
    return s.groupby(s.index.floor("D")).last().sort_index()


def _load_cached_daily_price(asset_symbol: str) -> pd.Series:
    """
    Fallback for non-futures assets using daily cache from 05b:
      output/cache/asset_prices/<SYMBOL>.csv with columns Date,Close
    """
    sym = asset_symbol.strip().upper()
    p = output_dir() / "cache" / "asset_prices" / f"{sym}.csv"
    if not p.exists():
        raise FileNotFoundError(f"daily cache not found: {p}")
    d = pd.read_csv(p)
    if not {"Date", "Close"}.issubset(d.columns):
        raise ValueError(f"invalid daily cache schema: {p}")
    d["Date"] = pd.to_datetime(d["Date"], utc=True, errors="coerce")
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    s = d.dropna(subset=["Date", "Close"]).set_index("Date")["Close"].sort_index()
    if s.empty:
        raise ValueError(f"empty daily cache: {p}")
    return s


def _ensure_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "signal_volume_weighted" not in out.columns:
        if "signal_mean" in out.columns:
            out["signal_volume_weighted"] = out["signal_mean"]
        elif "signal_sum" in out.columns:
            out["signal_volume_weighted"] = out["signal_sum"]
        else:
            out["signal_volume_weighted"] = 0.0

    out["signal_volume_weighted"] = pd.to_numeric(out["signal_volume_weighted"], errors="coerce")
    if "signal_sum" not in out.columns:
        out["signal_sum"] = out["signal_volume_weighted"]
    if "signal_abs_sum" not in out.columns:
        out["signal_abs_sum"] = pd.to_numeric(out["signal_sum"], errors="coerce").abs()
    if "signal_sq_sum" not in out.columns:
        out["signal_sq_sum"] = pd.to_numeric(out["signal_sum"], errors="coerce") ** 2
    if "weighted_score_sum" not in out.columns:
        out["weighted_score_sum"] = out["signal_volume_weighted"]
    if "volume_weight_sum" not in out.columns:
        out["volume_weight_sum"] = np.where(out["signal_volume_weighted"].notna(), 1.0, 0.0)
    if "signal_count" not in out.columns:
        out["signal_count"] = np.where(out["signal_volume_weighted"].notna(), 1, 0)
    if "signal_pos_count" not in out.columns:
        out["signal_pos_count"] = (pd.to_numeric(out["signal_sum"], errors="coerce") > 0).astype(int)
    if "signal_neg_count" not in out.columns:
        out["signal_neg_count"] = (pd.to_numeric(out["signal_sum"], errors="coerce") < 0).astype(int)
    if "active_markets" not in out.columns:
        out["active_markets"] = np.where(out["signal_count"] > 0, 1, 0)

    numeric_cols = [
        "signal_sum",
        "signal_abs_sum",
        "signal_sq_sum",
        "weighted_score_sum",
        "volume_weight_sum",
        "signal_count",
        "signal_pos_count",
        "signal_neg_count",
        "active_markets",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _finalize_signal_frame(df: pd.DataFrame, signal_freq: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ts",
                "signal_sum",
                "signal_mean",
                "signal_abs_sum",
                "signal_abs_mean",
                "signal_sq_sum",
                "signal_rms",
                "weighted_score_sum",
                "volume_weight_sum",
                "signal_volume_weighted",
                "signal_count",
                "signal_pos_count",
                "signal_neg_count",
                "active_markets",
                "signal_breadth",
                "signal_consensus",
            ]
        )

    out = _ensure_signal_columns(df)
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    if signal_freq == "1d":
        out["ts"] = out["ts"].dt.floor("D")
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if out.empty:
        return out

    out["signal_count"] = out["signal_count"].clip(lower=1)
    out["active_markets"] = out["active_markets"].clip(lower=1)
    out["signal_mean"] = out["signal_sum"] / out["signal_count"]
    out["signal_abs_mean"] = out["signal_abs_sum"] / out["signal_count"]
    out["signal_rms"] = np.sqrt(out["signal_sq_sum"] / out["signal_count"])
    out["signal_volume_weighted"] = out["weighted_score_sum"] / out["volume_weight_sum"].replace(0.0, np.nan)
    out["signal_volume_weighted"] = out["signal_volume_weighted"].fillna(out["signal_mean"])
    out["signal_breadth"] = ((out["signal_pos_count"] - out["signal_neg_count"]) / out["signal_count"]).clip(-1.0, 1.0)
    out["signal_consensus"] = out["signal_breadth"].abs()
    return out


def _collapse_signal_rows(df: pd.DataFrame, signal_freq: str) -> pd.DataFrame:
    if df.empty:
        return _finalize_signal_frame(df, signal_freq)

    base = _ensure_signal_columns(df)
    base["ts"] = pd.to_datetime(base["ts"], utc=True, errors="coerce")
    if signal_freq == "1d":
        base["ts"] = base["ts"].dt.floor("D")
    base = base.dropna(subset=["ts"])
    if base.empty:
        return _finalize_signal_frame(base, signal_freq)

    grouped = (
        base.groupby("ts", as_index=False)[
            [
                "signal_sum",
                "signal_abs_sum",
                "signal_sq_sum",
                "weighted_score_sum",
                "volume_weight_sum",
                "signal_count",
                "signal_pos_count",
                "signal_neg_count",
                "active_markets",
            ]
        ]
        .sum()
    )
    return _finalize_signal_frame(grouped, signal_freq)


def load_signals(topic: str, strategy: str, signal_freq: str, asset_symbol: str | None = None) -> pd.DataFrame:
    path = output_topic_signals_dir(topic) / strategy / signal_freq / "event_signal.parquet"
    if not path.exists():
        path = output_topic_signals_dir(topic) / strategy / "event_signal.parquet"
    bucket_frames: list[pd.DataFrame] = []
    if asset_symbol:
        for b in ASSET_BUCKETS_BY_SYMBOL.get(asset_symbol.upper(), []):
            b_path = output_topic_signals_dir(topic) / strategy / signal_freq / "by_asset" / b / "event_signal.parquet"
            if b_path.exists():
                try:
                    bucket_frames.append(pd.read_parquet(b_path))
                except Exception:
                    continue
    if bucket_frames:
        df = pd.concat(bucket_frames, ignore_index=True)
        return _collapse_signal_rows(df, signal_freq)
    else:
        if not path.exists():
            return _finalize_signal_frame(pd.DataFrame(), signal_freq)
        df = pd.read_parquet(path)
    return _collapse_signal_rows(df, signal_freq)


def _load_topic_asset_routing(
    min_asset_score: float = 0.2,
    min_assets_per_topic: int = 2,
) -> dict[str, set[str]]:
    """
    Build topic -> {asset symbols} routing from Step01 outputs.
    Expected map rows like top_asset in {OIL,GOLD,SPX,USD,CPI,USDCNY,CHNA_EQ,GAS}.
    """
    p = output_pipeline_dir() / "topic_asset_map.csv"
    if not p.exists():
        return {}
    try:
        d = pd.read_csv(p)
    except Exception:
        return {}
    if d.empty or "topic" not in d.columns or "top_asset" not in d.columns:
        return {}

    to_proxy = {
        "USD": {"UUP"},
        "GOLD": {"GLD"},
        "OIL": {"USO"},
        "SPX": {"SPY"},
        "CHNA_EQ": {"FXI"},
        "USDCNY": {"FXI", "UUP"},
        "CPI": {"ZN", "ZT", "UUP"},
        "GAS": {"USO"},
    }
    out: dict[str, set[str]] = {}
    for topic, g in d.groupby("topic"):
        gg = g.copy()
        if "asset_score_mean" in gg.columns:
            gg["asset_score_mean"] = pd.to_numeric(gg["asset_score_mean"], errors="coerce").fillna(0.0)
            gg = gg.sort_values("asset_score_mean", ascending=False)
            keep = gg[gg["asset_score_mean"] >= float(min_asset_score)]
            if len(keep) < int(min_assets_per_topic):
                keep = gg.head(int(min_assets_per_topic))
        else:
            keep = gg
        aset = set()
        for a in keep["top_asset"].astype(str):
            aset.update(to_proxy.get(a, set()))
        if aset:
            out[str(topic)] = aset
    return out


def get_fwd_return(price_daily: pd.Series, ts: pd.Timestamp, horizon_days: int) -> float | None:
    if ts in price_daily.index:
        idx = price_daily.index.get_loc(ts)
    else:
        idx = price_daily.index.get_indexer([ts], method="pad")[0]
    if idx < 0:
        return None
    fwd = idx + horizon_days
    if fwd >= len(price_daily):
        return None
    p0 = float(price_daily.iloc[idx])
    p1 = float(price_daily.iloc[fwd])
    if p0 <= 0:
        return None
    ret = (p1 / p0) - 1.0
    return None if abs(ret) < MIN_RETURN_THRESHOLD else ret


def get_fwd_return_minute(price_1m: pd.Series, ts: pd.Timestamp, horizon_min: int) -> float | None:
    if ts in price_1m.index:
        idx = price_1m.index.get_loc(ts)
    else:
        idx = price_1m.index.get_indexer([ts], method="pad")[0]
    if idx < 0:
        return None
    fwd = idx + horizon_min
    if fwd >= len(price_1m):
        return None
    p0 = float(price_1m.iloc[idx])
    p1 = float(price_1m.iloc[fwd])
    if p0 <= 0:
        return None
    ret = (p1 / p0) - 1.0
    return None if abs(ret) < MIN_RETURN_THRESHOLD else ret


def _ann_factor(horizon: int, unit: str) -> float:
    return np.sqrt(max(1.0, (252.0 / float(horizon)) if unit == "d" else ((252.0 * 390.0) / float(horizon))))


def _timing_params(signal_freq: str) -> tuple[int, int, float, float, int, int]:
    key = signal_freq.strip().lower()
    if key in {"day", "daily"}:
        key = "1d"
    return (
        ROLL_WINDOW_BY_FREQ.get(key, 63),
        ROLL_MINP_BY_FREQ.get(key, 15),
        ENTRY_Z_BY_FREQ.get(key, 1.2),
        MIN_CONSENSUS_BY_FREQ.get(key, 0.2),
        MIN_COUNT_BY_FREQ.get(key, 1),
        MIN_RUN_BY_FREQ.get(key, 1),
    )


def _direction_streak(direction: pd.Series) -> pd.Series:
    vals = pd.to_numeric(direction, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    streak = np.zeros(len(vals), dtype=int)
    prev = 0
    run = 0
    for i, val in enumerate(vals):
        cur = int(np.sign(val))
        if cur == 0:
            run = 0
            prev = 0
        elif cur == prev:
            run += 1
            prev = cur
        else:
            run = 1
            prev = cur
        streak[i] = run
    return pd.Series(streak, index=direction.index)


def _build_entry_candidates(signals: pd.DataFrame, signal_freq: str) -> tuple[pd.DataFrame, str]:
    s = _finalize_signal_frame(signals, signal_freq)
    if s.empty:
        s = s.assign(
            entry_trigger=pd.Series(dtype=bool),
            entry_direction=pd.Series(dtype=float),
            entry_z=pd.Series(dtype=float),
            entry_score=pd.Series(dtype=float),
        )
        return s, "none"

    roll_window, min_periods, entry_z, min_consensus, min_count, min_run = _timing_params(signal_freq)
    s = s.sort_values("ts").reset_index(drop=True)

    s["directional_score"] = (
        0.7 * s["signal_volume_weighted"] + 0.3 * s["signal_breadth"] * s["signal_rms"]
    )
    zero_dir = np.isclose(s["directional_score"], 0.0)
    s.loc[zero_dir, "directional_score"] = s.loc[zero_dir, "signal_sum"]
    s["entry_direction"] = np.sign(s["directional_score"])

    participation = np.sqrt(s["active_markets"].clip(lower=1.0))
    breadth_weight = 0.5 + s["signal_consensus"].clip(0.0, 1.0)
    density_weight = np.sqrt(s["signal_count"].clip(lower=1.0))
    s["entry_score"] = s["directional_score"].abs() * participation * breadth_weight * density_weight

    score_hist = s["entry_score"].rolling(roll_window, min_periods=min_periods)
    score_mu = score_hist.mean().shift(1)
    score_sd = score_hist.std().shift(1).replace(0.0, np.nan)
    s["entry_z"] = ((s["entry_score"] - score_mu) / score_sd).replace([np.inf, -np.inf], np.nan)
    s["direction_run"] = _direction_streak(s["entry_direction"])
    s["entry_accel"] = s["entry_score"].diff()

    count_floor = s["signal_count"].rolling(roll_window, min_periods=1).median().shift(1)
    count_floor = count_floor.fillna(float(min_count))
    count_floor = np.maximum(count_floor.to_numpy(dtype=float) * 0.75, float(min_count))
    count_ok = s["signal_count"].to_numpy(dtype=float) >= count_floor
    consensus_ok = s["signal_consensus"].to_numpy(dtype=float) >= float(min_consensus)
    breadth_sign = np.sign(s["signal_breadth"].to_numpy(dtype=float))
    direction_vals = np.sign(s["entry_direction"].to_numpy(dtype=float))
    sign_ok = (breadth_sign == 0.0) | (breadth_sign == direction_vals)
    accel_ok = (s["entry_accel"].fillna(s["entry_score"]).to_numpy(dtype=float) > 0.0) | (
        s["entry_z"].fillna(-np.inf).to_numpy(dtype=float) >= float(entry_z + 0.75)
    )
    crossed = (s["entry_z"].fillna(-np.inf) >= float(entry_z)) & (
        (s["entry_z"].shift(1).fillna(-np.inf) < float(entry_z))
        | (s["entry_direction"] != s["entry_direction"].shift(1))
    )

    s["entry_trigger"] = (
        s["entry_direction"].ne(0.0)
        & s["entry_z"].notna()
        & pd.Series(count_ok, index=s.index)
        & pd.Series(consensus_ok, index=s.index)
        & pd.Series(sign_ok, index=s.index)
        & pd.Series(accel_ok, index=s.index)
        & crossed
        & (s["direction_run"] >= int(min_run))
    )

    rule = "rolling_zscore_confirmation"
    if int(s["entry_trigger"].sum()) == 0 or int(s["entry_z"].notna().sum()) < int(min_periods):
        upper = float(s["signal_volume_weighted"].quantile(ENTRY_UPPER_Q))
        lower = float(s["signal_volume_weighted"].quantile(ENTRY_LOWER_Q))
        if np.isclose(upper, lower):
            fallback_dir = np.sign(s["signal_volume_weighted"])
        else:
            fallback_dir = np.where(
                s["signal_volume_weighted"] >= upper,
                1.0,
                np.where(s["signal_volume_weighted"] <= lower, -1.0, 0.0),
            )
        s["entry_direction"] = fallback_dir
        s["entry_trigger"] = (
            pd.Series(fallback_dir, index=s.index).ne(0.0)
            & pd.Series(count_ok, index=s.index)
            & pd.Series(consensus_ok, index=s.index)
        )
        s["entry_z"] = s["entry_z"].fillna(0.0)
        rule = "quantile_fallback"

    return s, rule


def event_backtest(
    price_daily: pd.Series,
    price_1m: pd.Series,
    signals: pd.DataFrame,
    topic: str,
    strategy: str,
    signal_freq: str,
    asset_symbol: str,
    horizons_d: list[int],
    horizons_min: list[int],
) -> List[dict]:
    results = []
    upper = float(signals["signal_volume_weighted"].quantile(ENTRY_UPPER_Q))
    lower = float(signals["signal_volume_weighted"].quantile(ENTRY_LOWER_Q))
    horizons, unit = (horizons_d, "d") if signal_freq == "1d" else (horizons_min, "min")
    candidates, entry_rule = _build_entry_candidates(signals, signal_freq)
    if candidates.empty:
        return results

    for horizon in horizons:
        trades = []
        last_trade_ts = None
        for _, row in candidates[candidates["entry_trigger"]].iterrows():
            ts = row["ts"]
            sig = float(row["signal_volume_weighted"])
            direction = int(np.sign(float(row["entry_direction"])))
            if direction == 0:
                continue

            if last_trade_ts is not None:
                delta = ts - last_trade_ts
                if (unit == "d" and delta.days < COOLDOWN_DAYS) or (unit == "min" and delta.total_seconds() < COOLDOWN_MIN * 60):
                    continue

            ret = get_fwd_return(price_daily, ts, horizon) if unit == "d" else get_fwd_return_minute(price_1m, ts, horizon)
            if ret is None:
                continue
            strat_ret = direction * ret
            trades.append(
                {
                    "ts": ts,
                    "signal": sig,
                    "ret": ret,
                    "strategy_ret": strat_ret,
                    "entry_z": float(row.get("entry_z", np.nan)),
                    "signal_consensus": float(row.get("signal_consensus", np.nan)),
                    "signal_count": float(row.get("signal_count", np.nan)),
                    "active_markets": float(row.get("active_markets", np.nan)),
                }
            )
            last_trade_ts = ts

        if not trades:
            continue
        trades_df = pd.DataFrame(trades)
        std0 = float(trades_df["strategy_ret"].std())
        if std0 > 0:
            trades_df = trades_df[np.abs(trades_df["strategy_ret"]) < OUTLIER_STD_MULTIPLE * std0]
        if trades_df.empty:
            continue

        avg_ret = float(trades_df["strategy_ret"].mean())
        med_ret = float(trades_df["strategy_ret"].median())
        std_ret = float(trades_df["strategy_ret"].std())
        win_rate = float((trades_df["strategy_ret"] > 0).mean())
        pos = float(trades_df.loc[trades_df["strategy_ret"] > 0, "strategy_ret"].sum())
        neg = float(-trades_df.loc[trades_df["strategy_ret"] < 0, "strategy_ret"].sum())
        profit_factor = pos / (neg + 1e-12)
        sharpe = (avg_ret / (std_ret + 1e-12)) * _ann_factor(horizon, unit) if std_ret > 0 else 0.0
        cum_return = float(np.prod(np.maximum(1.0 + trades_df["strategy_ret"].to_numpy(dtype=float), 1e-9)) - 1.0)
        results.append(
            {
                "topic": topic,
                "asset_symbol": asset_symbol,
                "strategy": strategy,
                "signal_freq": signal_freq,
                "signal": "signal_volume_weighted",
                "entry_rule": entry_rule,
                "horizon": horizon,
                "horizon_unit": unit,
                "n_trades": int(len(trades_df)),
                "upper_q97": upper,
                "lower_q03": lower,
                "avg_entry_z": float(trades_df["entry_z"].mean()),
                "avg_signal_consensus": float(trades_df["signal_consensus"].mean()),
                "avg_signal_count": float(trades_df["signal_count"].mean()),
                "avg_active_markets": float(trades_df["active_markets"].mean()),
                "win_rate": win_rate,
                "avg_strat_ret": avg_ret,
                "med_strat_ret": med_ret,
                "std_strat_ret": std_ret,
                "sharpe": float(sharpe),
                "cum_return": cum_return,
                "profit_factor": profit_factor,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Step05 backtest with freq-aware signals")
    parser.add_argument(
        "--signal-freqs",
        type=str,
        default="30min,60min,1d",
        help="comma-separated signal frequencies; default runs all",
    )
    parser.add_argument(
        "--horizons-d",
        type=str,
        default="5,10,20",
        help="comma-separated day horizons for daily signals",
    )
    parser.add_argument(
        "--horizons-min",
        type=str,
        default="60,240,1440",
        help="comma-separated minute horizons for minute signals",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="jump,momentum",
        help="comma-separated strategies to run, e.g. jump,momentum,reversal",
    )
    parser.add_argument("--min-trades-qualified", type=int, default=MIN_TRADES_QUALIFIED)
    parser.add_argument("--min-sharpe-qualified", type=float, default=MIN_SHARPE_QUALIFIED)
    parser.add_argument("--min-pf-qualified", type=float, default=MIN_PROFIT_FACTOR_QUALIFIED)
    parser.add_argument(
        "--max-drawdown-qualified",
        type=float,
        default=0.0,
        help="max allowed drawdown for qualified table (<=0 disables this filter if metric absent)",
    )
    parser.add_argument(
        "--asset-symbols",
        type=str,
        default="ZT",
        help="comma-separated asset symbols to backtest, e.g. ZT,ZN",
    )
    parser.add_argument(
        "--ignore-topic-asset-routing",
        action="store_true",
        help="if set, do not filter topic-asset pairs by Step01 routing map",
    )
    parser.add_argument(
        "--topic-asset-min-score",
        type=float,
        default=0.2,
        help="minimum asset_score_mean in topic_asset_map to keep route",
    )
    parser.add_argument(
        "--topic-asset-min-count",
        type=int,
        default=2,
        help="minimum number of top assets kept per topic even if below score threshold",
    )
    parser.add_argument(
        "--with-monthly-rotation",
        action="store_true",
        help="also evaluate portfolio using 05b monthly rotation weights",
    )
    args = parser.parse_args()

    signal_freqs = _parse_freq_list(args.signal_freqs)
    horizons_d = _parse_int_list(args.horizons_d)
    horizons_min = _parse_int_list(args.horizons_min)

    asset_symbols = [x.strip().upper() for x in args.asset_symbols.split(",") if x.strip()]
    if not asset_symbols:
        asset_symbols = ["ZT"]
    topic_files = discover_poly_topic_files()
    strategies = [s.strip().lower() for s in args.strategies.split(",") if s.strip()]
    if not strategies:
        strategies = ["jump", "momentum"]
    routing = (
        {}
        if args.ignore_topic_asset_routing
        else _load_topic_asset_routing(
            min_asset_score=float(args.topic_asset_min_score),
            min_assets_per_topic=int(args.topic_asset_min_count),
        )
    )
    if routing:
        print(f"[05] loaded topic-asset routing for {len(routing)} topics")
    all_results: List[dict] = []
    for asset_symbol in asset_symbols:
        print(f"[05] loading asset={asset_symbol}")
        has_intraday = True
        try:
            price_1m = _load_price_1min(asset_symbol)
            price_daily = price_1m.groupby(price_1m.index.floor("D")).last().sort_index()
        except Exception as e:
            # Fallback: use cached daily-only series for assets like SPY/GLD/USO/UUP/FXI
            has_intraday = False
            try:
                price_daily = _load_cached_daily_price(asset_symbol)
                price_1m = pd.Series(dtype=float)
                print(f"[05][INFO] asset={asset_symbol} use daily-only cache")
            except Exception:
                print(f"[05][WARN] skip asset={asset_symbol} reason={e}")
                continue
        for tf in topic_files:
            topic = tf.topic
            allowed_assets = routing.get(topic, None)
            for strategy in strategies:
                for signal_freq in signal_freqs:
                    if (not has_intraday) and signal_freq != "1d":
                        continue
                    if allowed_assets is not None and asset_symbol not in allowed_assets:
                        continue
                    print(f"[05] backtesting asset={asset_symbol} topic={topic} strategy={strategy} freq={signal_freq}")
                    signals = load_signals(topic, strategy, signal_freq, asset_symbol=asset_symbol)
                    if signals.empty:
                        continue
                    all_results.extend(
                        event_backtest(
                            price_daily,
                            price_1m,
                            signals,
                            topic,
                            strategy,
                            signal_freq,
                            asset_symbol,
                            horizons_d,
                            horizons_min,
                        )
                    )

    df = pd.DataFrame(all_results)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "topic",
                "asset_symbol",
                "strategy",
                "signal_freq",
                "signal",
                "entry_rule",
                "horizon",
                "horizon_unit",
                "n_trades",
                "upper_q97",
                "lower_q03",
                "avg_entry_z",
                "avg_signal_consensus",
                "avg_signal_count",
                "avg_active_markets",
                "win_rate",
                "avg_strat_ret",
                "med_strat_ret",
                "std_strat_ret",
                "sharpe",
                "cum_return",
                "profit_factor",
            ]
        )
    print(df)
    out_main = output_dir() / "event_backtest_results.csv"
    df.to_csv(out_main, index=False)
    report_dir = output_dir() / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_dir / "backtest_summary_table.csv", index=False)

    # Build a strict shortlist to avoid "looks good by chance" combinations.
    qualified = df.copy()
    if not qualified.empty:
        qualified = qualified[qualified["n_trades"] >= int(args.min_trades_qualified)]
        qualified = qualified[qualified["sharpe"] >= float(args.min_sharpe_qualified)]
        qualified = qualified[qualified["profit_factor"] >= float(args.min_pf_qualified)]
        if args.max_drawdown_qualified > 0 and "max_drawdown" in qualified.columns:
            qualified = qualified[qualified["max_drawdown"] >= -float(args.max_drawdown_qualified)]
        qualified = qualified.sort_values(
            ["sharpe", "profit_factor", "cum_return", "n_trades"],
            ascending=[False, False, False, False],
        )
    qualified_out = report_dir / "backtest_qualified_table.csv"
    qualified.to_csv(qualified_out, index=False)
    print(f"[05] qualified rows={len(qualified)} -> {qualified_out}")
    print(f"[05] saved -> {out_main}")

    if args.with_monthly_rotation:
        w_path = report_dir / "monthly_asset_rotation_weights.csv"
        nav_path = report_dir / "monthly_rotation_portfolio_nav.csv"
        m_path = report_dir / "monthly_rotation_portfolio_metrics.csv"
        nav_df, m_df = _run_monthly_rotation_portfolio_backtest(w_path, nav_path, m_path)
        if nav_df.empty:
            print(f"[05][WARN] monthly rotation portfolio not produced. missing/invalid weights: {w_path}")
        else:
            print(f"[05] monthly rotation portfolio saved -> {nav_path}")
            print(f"[05] monthly rotation metrics saved -> {m_path}")
            print(m_df.to_string(index=False))


if __name__ == "__main__":
    main()
