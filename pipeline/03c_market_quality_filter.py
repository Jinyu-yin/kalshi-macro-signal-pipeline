from __future__ import annotations

"""
Step 03c: strict market quality filter and validity score.

This step builds topic-level market validity scores (0-100) from available local data
and exports a strict eligible market list for downstream signal generation.

Outputs per topic:
  output/pipeline/topics/{topic}/quality/market_quality_scores.csv
  output/pipeline/topics/{topic}/quality/eligible_markets.csv
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from pipeline.utils_paths import output_topic_dir, output_topic_market_prob_dir
from pipeline.utils_topics import discover_poly_topic_files


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _runs_test_z(s: pd.Series) -> float:
    x = np.sign(pd.to_numeric(s, errors="coerce").dropna().to_numpy())
    x = x[x != 0]
    n = len(x)
    if n < 20:
        return float("nan")
    runs = 1 + np.sum(x[1:] != x[:-1])
    n_pos = int(np.sum(x > 0))
    n_neg = int(np.sum(x < 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    mean_runs = 1 + (2 * n_pos * n_neg) / (n_pos + n_neg)
    var_runs = (
        2
        * n_pos
        * n_neg
        * (2 * n_pos * n_neg - n_pos - n_neg)
        / (((n_pos + n_neg) ** 2) * (n_pos + n_neg - 1))
    )
    if var_runs <= 0:
        return float("nan")
    return float((runs - mean_runs) / math.sqrt(var_runs))


def _is_binary_preferred_question(q: str) -> bool:
    t = (q or "").lower()
    bad_markers = [
        "who will win",
        "which team",
        "which player",
        "winner of",
        "top scorer",
        "夺冠",
        "冠军",
    ]
    return not any(m in t for m in bad_markers)


def _market_design_score(question: str, start_time: pd.Timestamp, end_time: pd.Timestamp) -> tuple[float, bool]:
    q = (question or "").strip()
    clarity = 1.0 if len(q) >= 20 else 0.6
    binary_ok = _is_binary_preferred_question(q)
    binary_score = 1.0 if binary_ok else 0.2
    if pd.isna(start_time) or pd.isna(end_time):
        dur_score = 0.3
        hard_duration_ok = False
    else:
        dur_days = (end_time - start_time).total_seconds() / 86400.0
        if dur_days < 1:
            dur_score = 0.4
        elif dur_days <= 90:
            dur_score = 1.0
        elif dur_days <= 365:
            dur_score = 0.5
        else:
            dur_score = 0.0
        hard_duration_ok = (dur_days >= 1) and (dur_days <= 365)
    return 0.5 * clarity + 0.3 * binary_score + 0.2 * dur_score, hard_duration_ok and binary_ok


def _calc_market_row(
    topic: str,
    row: pd.Series,
    prob_path: Path,
    *,
    volume_min_usd: float,
    acf_max_thr: float,
    runs_abs_z_thr: float,
    vol_proxy_thr: float,
    eff_min_checks: int,
    max_duration_days_hard: float,
) -> dict:
    out = {
        "topic": topic,
        "market_id": str(row["market_id"]),
        "question": str(row.get("question", "")),
        "validity_score": 0.0,
        "hard_pass": False,
        "total_volume_usd": 0.0,
        "active_bars": 0,
        "autocorr_abs_max_lag5": np.nan,
        "runs_test_abs_z": np.nan,
        "volatility_daily_proxy": np.nan,
        "design_score": 0.0,
        "liquidity_score": 0.0,
        "efficiency_score": 0.0,
        "history_score": 0.0,
        "manipulation_score": 0.0,
        "fail_reason": "",
    }

    if not prob_path.exists():
        out["fail_reason"] = "missing_market_prob"
        return out

    try:
        df = pd.read_parquet(prob_path)
    except Exception:
        out["fail_reason"] = "bad_market_prob_file"
        return out

    if df.empty or "prob" not in df.columns:
        out["fail_reason"] = "empty_or_no_prob"
        return out

    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0.0)
    df = df.dropna(subset=["prob"])
    if len(df) < 20:
        out["fail_reason"] = "too_few_bars"
        return out

    total_volume = float(df["volume"].sum())
    out["total_volume_usd"] = total_volume
    out["active_bars"] = int(len(df))

    ret = df["prob"].diff().dropna()
    acfs = []
    for lag in range(1, 6):
        if len(ret) > lag + 2:
            acfs.append(abs(float(ret.autocorr(lag=lag))))
    acf_max = max(acfs) if acfs else np.nan
    runs_z_abs = abs(_runs_test_z(ret))
    vol_proxy = float(ret.std()) if len(ret) > 5 else np.nan
    out["autocorr_abs_max_lag5"] = acf_max
    out["runs_test_abs_z"] = runs_z_abs
    out["volatility_daily_proxy"] = vol_proxy

    st = pd.to_datetime(row.get("start_time"), errors="coerce", utc=True)
    et = pd.to_datetime(row.get("end_time"), errors="coerce", utc=True)
    design_score, hard_design_ok = _market_design_score(str(row.get("question", "")), st, et)
    if not pd.isna(st) and not pd.isna(et):
        dur_days = (et - st).total_seconds() / 86400.0
        if (dur_days < 1) or (dur_days > max_duration_days_hard):
            hard_design_ok = False

    # liquidity proxy (hard threshold aligned to your strict rule)
    liq_score = _clip01((math.log10(total_volume + 1) - 5.0) / 1.0)  # >=1 at ~$1m
    hard_liq_ok = total_volume >= float(volume_min_usd)

    # efficiency proxy: require at least 2 checks pass
    eff_checks = 0
    if not np.isnan(acf_max) and acf_max < float(acf_max_thr):
        eff_checks += 1
    if not np.isnan(runs_z_abs) and runs_z_abs < float(runs_abs_z_thr):
        eff_checks += 1
    if not np.isnan(vol_proxy) and vol_proxy < float(vol_proxy_thr):
        eff_checks += 1
    eff_score = eff_checks / 3.0
    hard_eff_ok = eff_checks >= int(eff_min_checks)

    # unavailable historical Brier/accuracy -> conservative neutral
    history_score = 0.5

    # anti-manipulation proxy: active bars + moderate vol + enough volume
    manip_score = 0.4 * _clip01((len(df) - 40) / 200) + 0.3 * _clip01((0.06 - (vol_proxy if not np.isnan(vol_proxy) else 0.06)) / 0.06) + 0.3 * liq_score

    validity = (
        0.3 * liq_score
        + 0.3 * eff_score
        + 0.2 * history_score
        + 0.1 * design_score
        + 0.1 * manip_score
    ) * 100.0

    hard_pass = hard_design_ok and hard_liq_ok and hard_eff_ok
    fail_reasons = []
    if not hard_design_ok:
        fail_reasons.append("design")
    if not hard_liq_ok:
        fail_reasons.append("liquidity")
    if not hard_eff_ok:
        fail_reasons.append("efficiency")

    out.update(
        {
            "validity_score": float(validity),
            "hard_pass": bool(hard_pass),
            "design_score": float(design_score),
            "liquidity_score": float(liq_score),
            "efficiency_score": float(eff_score),
            "history_score": float(history_score),
            "manipulation_score": float(manip_score),
            "fail_reason": ";".join(fail_reasons),
        }
    )
    return out


def process_topic(
    topic: str,
    *,
    strict_keep: float,
    warn_keep: float,
    allow_soft: bool,
    volume_min_usd: float,
    acf_max_thr: float,
    runs_abs_z_thr: float,
    vol_proxy_thr: float,
    eff_min_checks: int,
    max_duration_days_hard: float,
) -> pd.DataFrame:
    tdir = output_topic_dir(topic)
    mpath = tdir / "markets.csv"
    if not mpath.exists():
        return pd.DataFrame()
    markets = pd.read_csv(mpath)
    if markets.empty:
        return pd.DataFrame()

    prob_dir = output_topic_market_prob_dir(topic)
    rows: List[dict] = []
    for _, r in markets.iterrows():
        mid = str(r["market_id"])
        p = prob_dir / f"polymarket_{mid}.parquet"
        rows.append(
            _calc_market_row(
                topic,
                r,
                p,
                volume_min_usd=volume_min_usd,
                acf_max_thr=acf_max_thr,
                runs_abs_z_thr=runs_abs_z_thr,
                vol_proxy_thr=vol_proxy_thr,
                eff_min_checks=eff_min_checks,
                max_duration_days_hard=max_duration_days_hard,
            )
        )

    out = pd.DataFrame(rows)
    qdir = tdir / "quality"
    qdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(qdir / "market_quality_scores.csv", index=False, encoding="utf-8")

    keep_mask = (out["validity_score"] >= float(strict_keep)) & (out["hard_pass"] | bool(allow_soft))
    warn_mask = (
        (out["validity_score"] >= float(warn_keep))
        & (out["validity_score"] < float(strict_keep))
        & (out["hard_pass"] | bool(allow_soft))
    )

    keep = out[keep_mask].copy()
    warn = out[warn_mask].copy()

    eligible = keep[["market_id"]].drop_duplicates()
    cautious = warn[["market_id"]].drop_duplicates()

    eligible.to_csv(qdir / "eligible_markets.csv", index=False, encoding="utf-8")
    cautious.to_csv(qdir / "cautious_markets.csv", index=False, encoding="utf-8")

    # keep legacy path for compatibility with existing Step04 loader
    sf_dir = tdir / "single_factor"
    sf_dir.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(sf_dir / "eligible_markets.csv", index=False, encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                "topic": topic,
                "n_markets": int(len(out)),
                "n_hard_pass": int(out["hard_pass"].sum()),
                "n_keep": int(len(eligible)),
                "n_cautious": int(len(cautious)),
                "keep_ratio": float(len(eligible) / max(1, len(out))),
                "strict_keep": float(strict_keep),
                "warn_keep": float(warn_keep),
                "allow_soft": bool(allow_soft),
                "volume_min_usd": float(volume_min_usd),
                "acf_max_thr": float(acf_max_thr),
                "runs_abs_z_thr": float(runs_abs_z_thr),
                "vol_proxy_thr": float(vol_proxy_thr),
                "eff_min_checks": int(eff_min_checks),
                "max_duration_days_hard": float(max_duration_days_hard),
            }
        ]
    )
    summary.to_csv(qdir / "quality_summary.csv", index=False, encoding="utf-8")
    print(
        f"[03c] topic={topic:<18} total={len(out):,} hard_pass={int(out['hard_pass'].sum()):,} keep={len(eligible):,}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step03c strict market quality filter")
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="strict",
        choices=["strict", "relaxed"],
        help="strict: hard gates + score>=strict-keep; relaxed: softer gates + allow-soft",
    )
    parser.add_argument("--strict-keep", type=float, default=70.0)
    parser.add_argument("--warn-keep", type=float, default=60.0)
    parser.add_argument("--allow-soft", action="store_true", help="allow score-based keep even if hard gates fail")
    parser.add_argument("--volume-min-usd", type=float, default=None)
    parser.add_argument("--acf-max-thr", type=float, default=None)
    parser.add_argument("--runs-abs-z-thr", type=float, default=None)
    parser.add_argument("--vol-proxy-thr", type=float, default=None)
    parser.add_argument("--eff-min-checks", type=int, default=None)
    parser.add_argument("--max-duration-days-hard", type=float, default=None)
    args = parser.parse_args()

    if args.mode == "strict":
        default_params = dict(
            allow_soft=False,
            volume_min_usd=500_000.0,
            acf_max_thr=0.12,
            runs_abs_z_thr=1.96,
            vol_proxy_thr=0.05,
            eff_min_checks=2,
            max_duration_days_hard=365.0,
        )
    else:
        default_params = dict(
            allow_soft=True,
            volume_min_usd=100_000.0,
            acf_max_thr=0.20,
            runs_abs_z_thr=2.58,
            vol_proxy_thr=0.08,
            eff_min_checks=1,
            max_duration_days_hard=730.0,
        )

    allow_soft = bool(args.allow_soft) if args.allow_soft else bool(default_params["allow_soft"])
    volume_min_usd = float(args.volume_min_usd) if args.volume_min_usd is not None else float(default_params["volume_min_usd"])
    acf_max_thr = float(args.acf_max_thr) if args.acf_max_thr is not None else float(default_params["acf_max_thr"])
    runs_abs_z_thr = float(args.runs_abs_z_thr) if args.runs_abs_z_thr is not None else float(default_params["runs_abs_z_thr"])
    vol_proxy_thr = float(args.vol_proxy_thr) if args.vol_proxy_thr is not None else float(default_params["vol_proxy_thr"])
    eff_min_checks = int(args.eff_min_checks) if args.eff_min_checks is not None else int(default_params["eff_min_checks"])
    max_duration_days_hard = (
        float(args.max_duration_days_hard) if args.max_duration_days_hard is not None else float(default_params["max_duration_days_hard"])
    )

    topics = discover_poly_topic_files()
    if args.topic:
        topics = [t for t in topics if t.topic == args.topic]
    if not topics:
        print("[03c] no topics found")
        return

    parts = []
    for t in topics:
        s = process_topic(
            t.topic,
            strict_keep=args.strict_keep,
            warn_keep=args.warn_keep,
            allow_soft=allow_soft,
            volume_min_usd=volume_min_usd,
            acf_max_thr=acf_max_thr,
            runs_abs_z_thr=runs_abs_z_thr,
            vol_proxy_thr=vol_proxy_thr,
            eff_min_checks=eff_min_checks,
            max_duration_days_hard=max_duration_days_hard,
        )
        if not s.empty:
            parts.append(s)

    if parts:
        summary_all = pd.concat(parts, ignore_index=True)
        summary_all.to_csv(
            ROOT_FOR_IMPORT / "output" / "pipeline" / "market_quality_summary.csv",
            index=False,
            encoding="utf-8",
        )
        print("[03c] saved output/pipeline/market_quality_summary.csv")


if __name__ == "__main__":
    main()

