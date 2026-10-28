from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from pipeline.utils_paths import (
    output_topic_dir,
    output_topic_market_prob_dir,
    output_topic_signals_dir,
)

from pipeline.utils_topics import discover_poly_topic_files


# 默认参数（会被 topic 的 single_factor/thresholds.csv 覆盖）
DEFAULT_PROB_JUMP_THRESHOLD = 0.02
DEFAULT_REVERSAL_MIN_DISTANCE = 0.02
DEFAULT_FREQS = ("30min", "60min", "1D")
ASSET_BUCKETS = {"OIL", "GOLD", "SPX", "USD", "CPI", "USDCNY", "CHNA_EQ", "GAS"}
MOM_TOP_PCT = 0.15
MOM_SHORT_WIN = 10
MOM_LONG_WIN = 252
MOM_Z_WIN_LEVEL = 252
MOM_Z_MINP_LEVEL = 60
MOM_Z_WIN_DIFF = 63
MOM_Z_MINP_DIFF = 20


def _normalize_question_stem(question: str) -> str:
    q = (question or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    # Remove common option tail markers to identify sibling options of same event.
    for sep in [" - ", " | ", " / ", ": "]:
        if sep in q:
            q = q.split(sep)[0].strip()
    q = re.sub(r"\b(yes|no|over|under)\b", "", q).strip()
    return q[:160]


def _freq_subdir(prob_dir: Path, freq: str) -> Path:
    f = freq.strip().lower()
    if f in {"1d", "day", "daily"}:
        return prob_dir
    if f in {"60min", "1h"}:
        return prob_dir / "60min"
    if f == "30min":
        return prob_dir / "30min"
    raise ValueError(f"Unsupported freq: {freq}")


def _load_market_prob(prob_dir: Path, freq: str) -> pd.DataFrame:
    prob_files: List[Path] = sorted(_freq_subdir(prob_dir, freq).glob("polymarket_*.parquet"))
    if not prob_files:
        return pd.DataFrame()

    prob_parts: List[pd.DataFrame] = []
    for p in prob_files:
        try:
            df_p = pd.read_parquet(p)
        except Exception:
            continue
        if "market_id" not in df_p.columns:
            df_p["market_id"] = p.stem.replace("polymarket_", "")
        prob_parts.append(df_p)

    if not prob_parts:
        return pd.DataFrame()

    df = pd.concat(prob_parts, ignore_index=True)
    if df.empty:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0)
    df["market_id"] = df["market_id"].astype(str)
    df = df.dropna(subset=["ts", "prob", "market_id"])
    df = df[(df["prob"] > 0.01) & (df["prob"] < 0.99)]
    return df


def _enrich_with_event_meta(topic: str, df: pd.DataFrame) -> pd.DataFrame:
    events_path = output_topic_dir(topic) / "events.csv"
    markets_path = output_topic_dir(topic) / "markets.csv"

    if events_path.exists():
        events = pd.read_csv(events_path)
        events["market_id"] = events["market_id"].astype(str)
        if "direction" not in events.columns:
            events["direction"] = 1
        if "magnitude" not in events.columns:
            events["magnitude"] = 1.0
        df = df.merge(
            events[["market_id", "direction", "magnitude"]].drop_duplicates("market_id"),
            on="market_id",
            how="left",
        )
    else:
        df["direction"] = 1
        df["magnitude"] = 1.0

    if markets_path.exists():
        markets = pd.read_csv(markets_path)
        markets["market_id"] = markets["market_id"].astype(str)
        markets["question"] = markets.get("question", "").astype(str)
        markets["group_key"] = markets["question"].map(_normalize_question_stem)
        keep_cols = ["market_id", "question", "group_key"]
        if "top_asset" in markets.columns:
            keep_cols.append("top_asset")
        if "top_asset_score" in markets.columns:
            keep_cols.append("top_asset_score")
        df = df.merge(markets[keep_cols], on="market_id", how="left")
    else:
        df["question"] = ""
        df["group_key"] = df["market_id"]
        df["top_asset"] = "UNK"
        df["top_asset_score"] = 0.0

    df["direction"] = pd.to_numeric(df["direction"], errors="coerce").fillna(1).clip(-1, 1)
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce").fillna(1.0).clip(0, 1)
    df["group_key"] = df["group_key"].fillna(df["market_id"]).astype(str)
    if "top_asset" not in df.columns:
        df["top_asset"] = "UNK"
    df["top_asset"] = df["top_asset"].fillna("UNK").astype(str).str.upper()
    if "top_asset_score" not in df.columns:
        df["top_asset_score"] = 0.0
    df["top_asset_score"] = pd.to_numeric(df["top_asset_score"], errors="coerce").fillna(0.0)
    return df


def _load_eligible_markets(topic: str) -> set[str] | None:
    candidates = [
        output_topic_dir(topic) / "quality" / "eligible_markets.csv",
        output_topic_dir(topic) / "single_factor" / "eligible_markets.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if "market_id" not in d.columns:
            continue
        ids = set(d["market_id"].astype(str).dropna().unique())
        return ids
    return None


def _aggregate_signal(df: pd.DataFrame, out_dir: Path, freq: str) -> None:
    market_signal_path = out_dir / "signal_market.parquet"
    agg_path = out_dir / "event_signal.parquet"
    if df.empty:
        return

    out = df.copy()
    out.to_parquet(market_signal_path, index=False)
    floor_unit = "D" if freq.strip().lower() in {"1d", "day", "daily"} else freq
    out["ts_day"] = out["ts"].dt.floor(floor_unit)
    out["abs_score"] = out["score"].abs()
    out["signal_sq"] = out["score"] ** 2
    out["volume_weight"] = np.maximum(pd.to_numeric(out["volume"], errors="coerce").fillna(0.0), 1e-12)
    out["weighted_score"] = out["score"] * out["volume_weight"]
    out["pos_flag"] = (out["score"] > 0).astype(int)
    out["neg_flag"] = (out["score"] < 0).astype(int)

    agg = (
        out.groupby("ts_day", as_index=False)
        .agg(
            signal_sum=("score", "sum"),
            signal_mean=("score", "mean"),
            signal_abs_sum=("abs_score", "sum"),
            signal_sq_sum=("signal_sq", "sum"),
            weighted_score_sum=("weighted_score", "sum"),
            volume_weight_sum=("volume_weight", "sum"),
            signal_count=("score", "size"),
            signal_pos_count=("pos_flag", "sum"),
            signal_neg_count=("neg_flag", "sum"),
            active_markets=("market_id", "nunique"),
        )
        .rename(columns={"ts_day": "ts"})
        .sort_values("ts")
    )
    agg["signal_count"] = pd.to_numeric(agg["signal_count"], errors="coerce").fillna(0).clip(lower=1)
    agg["active_markets"] = pd.to_numeric(agg["active_markets"], errors="coerce").fillna(0).clip(lower=1)
    agg["signal_abs_mean"] = agg["signal_abs_sum"] / agg["signal_count"]
    agg["signal_rms"] = np.sqrt(agg["signal_sq_sum"] / agg["signal_count"])
    agg["signal_volume_weighted"] = agg["weighted_score_sum"] / agg["volume_weight_sum"].replace(0.0, np.nan)
    agg["signal_volume_weighted"] = agg["signal_volume_weighted"].fillna(agg["signal_mean"])
    agg["signal_breadth"] = (
        (agg["signal_pos_count"] - agg["signal_neg_count"]) / agg["signal_count"]
    ).clip(-1.0, 1.0)
    agg["signal_consensus"] = agg["signal_breadth"].abs()
    agg.to_parquet(agg_path, index=False)


def _aggregate_signal_by_asset(df: pd.DataFrame, out_dir: Path, freq: str) -> None:
    if df.empty or "top_asset" not in df.columns:
        return
    by_asset_root = out_dir / "by_asset"
    by_asset_root.mkdir(parents=True, exist_ok=True)
    for asset in sorted(ASSET_BUCKETS):
        sub = df[df["top_asset"] == asset].copy()
        if sub.empty:
            continue
        a_dir = by_asset_root / asset
        a_dir.mkdir(parents=True, exist_ok=True)
        _aggregate_signal(sub, a_dir, freq)


def _load_topic_thresholds(topic: str) -> tuple[float, float]:
    th_path = output_topic_dir(topic) / "single_factor" / "thresholds.csv"
    if not th_path.exists():
        return DEFAULT_PROB_JUMP_THRESHOLD, DEFAULT_REVERSAL_MIN_DISTANCE

    try:
        th = pd.read_csv(th_path)
        kv = {str(k): float(v) for k, v in zip(th["key"], th["value"])}
    except Exception:
        return DEFAULT_PROB_JUMP_THRESHOLD, DEFAULT_REVERSAL_MIN_DISTANCE

    jump_thr = max(0.005, kv.get("prob_jump_threshold", DEFAULT_PROB_JUMP_THRESHOLD))
    rev_thr = max(0.005, kv.get("reversal_min_distance", DEFAULT_REVERSAL_MIN_DISTANCE))
    return jump_thr, rev_thr


def _build_jump_signal(df: pd.DataFrame, prob_jump_threshold: float) -> pd.DataFrame:
    out = df.sort_values(["market_id", "ts"]).copy()
    out["dprob"] = out.groupby("market_id")["prob"].diff()
    out["prev_prob"] = out.groupby("market_id")["prob"].shift(1)
    out["dlogit"] = np.log(out["prob"] / (1 - out["prob"])) - np.log(out["prev_prob"] / (1 - out["prev_prob"]))
    out = out[out["dprob"].abs() > prob_jump_threshold].copy()
    out["score"] = (
        out["direction"] * out["magnitude"] * np.sign(out["dlogit"]) * np.sqrt(np.abs(out["dlogit"]))
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _build_reversal_signal(df: pd.DataFrame, reversal_min_distance: float) -> pd.DataFrame:
    out = df.sort_values(["market_id", "ts"]).copy()
    out["group_size"] = out.groupby("group_key")["market_id"].transform("nunique").clip(lower=1)
    # Dynamic pivot: multi-option groups use 1/N; standalone markets use 0.5.
    out["pivot"] = np.where(out["group_size"] > 1, 1.0 / out["group_size"], 0.5)
    out["prev_prob"] = out.groupby("market_id")["prob"].shift(1)
    out["prev_rel"] = out["prev_prob"] - out["pivot"]
    out["curr_rel"] = out["prob"] - out["pivot"]
    out["crossed"] = (out["prev_rel"] * out["curr_rel"] <= 0) & out["prev_rel"].notna()
    out["move_mag"] = (out["curr_rel"] - out["prev_rel"]).abs()
    out = out[out["crossed"] & (out["move_mag"] >= reversal_min_distance)].copy()

    # Positive when crossing upward through pivot; negative for downward reversal.
    out["reversal_sign"] = np.sign(out["curr_rel"] - out["prev_rel"])
    out["score"] = (
        out["direction"] * out["magnitude"] * out["reversal_sign"] * np.sqrt(np.maximum(out["move_mag"], 1e-12))
    )
    out["score"] = out["score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _build_momentum_signal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["market_id", "ts"]).copy()
    if out.empty:
        return out

    g = out.groupby("market_id", group_keys=False)

    # Reproduce-style single-factor direction:
    # (trend_change > 0) OR (trend_strength > 0) => short(-1), else long(+1).
    level_mu = g["prob"].transform(lambda s: s.rolling(MOM_Z_WIN_LEVEL, min_periods=MOM_Z_MINP_LEVEL).mean())
    level_sd = g["prob"].transform(lambda s: s.rolling(MOM_Z_WIN_LEVEL, min_periods=MOM_Z_MINP_LEVEL).std()).replace(0.0, np.nan)
    strength = (out["prob"] - level_mu) / level_sd

    d1 = g["prob"].diff(1)
    diff_mu = d1.groupby(out["market_id"]).transform(
        lambda s: s.rolling(MOM_Z_WIN_DIFF, min_periods=MOM_Z_MINP_DIFF).mean()
    )
    diff_sd = d1.groupby(out["market_id"]).transform(
        lambda s: s.rolling(MOM_Z_WIN_DIFF, min_periods=MOM_Z_MINP_DIFF).std()
    ).replace(0.0, np.nan)
    change = (d1 - diff_mu) / diff_sd
    base_sig = np.where((change > 0) | (strength > 0), -1.0, 1.0)
    out["base_sig"] = np.where(out["prob"].notna(), base_sig, 0.0)

    # Build per-market factor NAV for momentum ranking.
    out["dprob"] = g["prob"].diff(1).fillna(0.0)
    out["factor_ret"] = g["base_sig"].shift(1).fillna(0.0) * out["dprob"]
    out["factor_nav"] = (1.0 + out["factor_ret"]).groupby(out["market_id"]).cumprod()

    # Monthly first timestamp rebalance, select top 15% by composite (10, 252) momentum rank.
    ts_list = pd.Index(out["ts"].sort_values().unique())
    reb_ts = pd.Series(ts_list).groupby(pd.Series(ts_list).dt.to_period("M")).min().tolist()
    selected_flag = pd.Series(False, index=out.index)

    for i, start_ts in enumerate(reb_ts):
        end_ts = reb_ts[i + 1] if i + 1 < len(reb_ts) else ts_list.max() + pd.Timedelta(seconds=1)
        period_mask = (out["ts"] >= start_ts) & (out["ts"] < end_ts)
        current = out.loc[period_mask]
        if current.empty:
            continue

        asof = out[out["ts"] < start_ts]
        if asof.empty:
            continue
        latest_nav = asof.sort_values("ts").groupby("market_id").tail(1)[["market_id", "factor_nav"]]
        nav_map_now = latest_nav.set_index("market_id")["factor_nav"]

        asof_sorted = asof.sort_values(["market_id", "ts"])
        nav_short = asof_sorted.groupby("market_id")["factor_nav"].apply(
            lambda s: s.iloc[-(MOM_SHORT_WIN + 1)] if len(s) > MOM_SHORT_WIN else np.nan
        )
        nav_long = asof_sorted.groupby("market_id")["factor_nav"].apply(
            lambda s: s.iloc[-(MOM_LONG_WIN + 1)] if len(s) > MOM_LONG_WIN else np.nan
        )

        common_ids = nav_map_now.index.intersection(nav_short.index).intersection(nav_long.index)
        if len(common_ids) == 0:
            continue

        mom_s = (nav_map_now.reindex(common_ids) / nav_short.reindex(common_ids) - 1.0).replace([np.inf, -np.inf], np.nan)
        mom_l = (nav_map_now.reindex(common_ids) / nav_long.reindex(common_ids) - 1.0).replace([np.inf, -np.inf], np.nan)
        rank_df = pd.DataFrame({"mom_s": mom_s, "mom_l": mom_l}).dropna()
        rank_df = rank_df[rank_df["mom_l"] >= 0.0]
        if rank_df.empty:
            continue

        r1 = rank_df["mom_s"].rank(ascending=False, method="average")
        r2 = rank_df["mom_l"].rank(ascending=False, method="average")
        rank_df["comp_rank"] = (r1 + r2) / 2.0

        n_pick = max(1, int(np.ceil(MOM_TOP_PCT * len(rank_df))))
        chosen_ids = set(rank_df.sort_values("comp_rank").head(n_pick).index.astype(str))
        selected_flag.loc[current[current["market_id"].astype(str).isin(chosen_ids)].index] = True

    out["selected"] = selected_flag
    out["score"] = out["direction"] * out["magnitude"] * out["base_sig"] * out["selected"].astype(float)
    return out[out["score"] != 0].copy()


def build_topic_signals(topic: str):

    prob_dir = output_topic_market_prob_dir(topic)
    base_out_dir = output_topic_signals_dir(topic)
    print(f"[04] topic={topic:<16} computing signals")
    for freq in DEFAULT_FREQS:
        freq_key = freq.lower()
        jump_out_dir = base_out_dir / "jump" / freq_key
        reversal_out_dir = base_out_dir / "reversal" / freq_key
        momentum_out_dir = base_out_dir / "momentum" / freq_key
        jump_out_dir.mkdir(parents=True, exist_ok=True)
        reversal_out_dir.mkdir(parents=True, exist_ok=True)
        momentum_out_dir.mkdir(parents=True, exist_ok=True)

        for d in [jump_out_dir, reversal_out_dir, momentum_out_dir]:
            for fname in ["signal_market.parquet", "event_signal.parquet"]:
                p = d / fname
                if p.exists():
                    p.unlink()

        df = _load_market_prob(prob_dir, freq)
        if df.empty:
            print(f"[04] topic={topic:<16} freq={freq:<5} no market_prob files, skipped")
            continue

        df = _enrich_with_event_meta(topic, df)
        eligible_ids = _load_eligible_markets(topic)
        if eligible_ids is not None:
            df = df[df["market_id"].isin(eligible_ids)].copy()

        if df.empty:
            print(f"[04] topic={topic:<16} freq={freq:<5} no eligible market data, skipped")
            continue

        jump_threshold, reversal_threshold = _load_topic_thresholds(topic)
        jump_df = _build_jump_signal(df, jump_threshold)
        reversal_df = _build_reversal_signal(df, reversal_threshold)
        momentum_df = _build_momentum_signal(df)

        _aggregate_signal(jump_df, jump_out_dir, freq)
        _aggregate_signal(reversal_df, reversal_out_dir, freq)
        _aggregate_signal(momentum_df, momentum_out_dir, freq)
        _aggregate_signal_by_asset(jump_df, jump_out_dir, freq)
        _aggregate_signal_by_asset(reversal_df, reversal_out_dir, freq)
        _aggregate_signal_by_asset(momentum_df, momentum_out_dir, freq)

        print(
            f"[04] topic={topic:<16} freq={freq:<5} finished | jump={len(jump_df):,} "
            f"reversal={len(reversal_df):,} momentum={len(momentum_df):,}"
        )


def main():

    topic_files = discover_poly_topic_files()

    if not topic_files:
        print("[04] no topics found")
        return

    for tf in topic_files:
        build_topic_signals(tf.topic)

    print("[04] finished")


if __name__ == "__main__":
    main()
