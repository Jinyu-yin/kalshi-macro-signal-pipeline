"""Kalshi 宏观分布 -> 点预测 -> 严格提前 N 天评估的主实验脚本。

这份脚本是当前 Kalshi 研究链路里最核心的一层，职责可以概括成四件事：

1. 读取已经构建好的 Kalshi 事件分布快照，并对齐真实 realized 标签。
2. 把区间分布压缩成一组候选点预测信号，如 mode / topN / shrunk / stack。
3. 在严格 point-in-time 条件下，评估不同窗口、不同点模型、不同 reject 策略。
4. 产出 best-by-window、baseline 对比、事件级明细和验证矩阵，供报告直接使用。

推荐阅读顺序：
- `_load_prediction_frame`：理解输入表如何被标准化
- `_pick_snapshot_strict_pre`：理解“提前 N 天”是怎么严格实现的
- `_distribution_features`：理解从分布里抽了哪些形状/不确定性特征
- `_selective_cv_score`：理解 selective prediction 与误差估计器
- `grid_search_selective_point_forecasts`：理解整条网格搜索主流程
- `export_best_event_level_predictions_selective`：理解最终报告表是怎么落地的

如果你是第一次接手这份代码，最重要的不是立刻读完整个文件，
而是先把“输入数据 schema -> snapshot 选取 -> predictor 构造 -> 评估输出”这四步串起来。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline._yield_proxy_timing import backtest_one_series, load_yield

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"
REPORT = ROOT / "output" / "report" / "kalshi_grid_search"
REPORT.mkdir(parents=True, exist_ok=True)

WINDOWS = [1, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50]
TOPNS = [1, 2, 3, 5, 10]
ENTRY_OFFSETS = [1, 3, 5, 7]
SIGNAL_THRESHOLDS = [0.0, 0.05, 0.10, 0.15]
YIELD_FILES = ["DGS5.csv", "DGS10.csv", "DGS20.csv"]
# Selective prediction (reject option) budgets. User-requested: reject <= 0.10.
REJECT_RATES = [0.0, 0.05, 0.10]
# Note: using alpha=0 with Ridge is numerically fragile; prefer small positive alphas.
RIDGE_ALPHAS = [0.1, 1.0, 10.0]
# Direct MAE optimization via median regression (QuantileRegressor @ q=0.5).
# Keep this small to avoid exploding grid size; alpha=0 is allowed and often works well on small data.
QR_ALPHAS = [0.0, 0.1, 1.0]
SELECTIVE_MIN_COVERAGE = 0.90
# Trade-off selector: maximize window while staying within (1+eps) of the best MAE per target.
SELECTIVE_EPS_FRAC = 0.10
MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def _parse_distribution(value: object) -> list[dict[str, float]]:
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(str(value))
    except Exception:
        return []
    out: list[dict[str, float]] = []
    for item in parsed if isinstance(parsed, list) else []:
        try:
            low_raw = item["low"]
            high_raw = item["high"]
            low = -np.inf if str(low_raw).lower() in {"-inf", "-infinity"} else float(low_raw)
            high = np.inf if str(high_raw).lower() in {"inf", "infinity"} else float(high_raw)
            mass = max(float(item["mass"]), 0.0)
        except Exception:
            continue
        if mass <= 0:
            continue
        if np.isfinite(low) and np.isfinite(high):
            mid = (low + high) / 2.0
        elif np.isfinite(low):
            mid = low
        elif np.isfinite(high):
            mid = high
        else:
            mid = np.nan
        out.append({"low": low, "high": high, "mid": mid, "mass": mass})
    return out


def _topn_prediction(value: object, topn: int) -> float:
    dist = _parse_distribution(value)
    if not dist:
        return float("nan")
    selected = sorted(dist, key=lambda x: x["mass"], reverse=True)[:topn]
    mids = np.array([x["mid"] for x in selected], dtype=float)
    masses = np.array([x["mass"] for x in selected], dtype=float)
    valid = np.isfinite(mids) & np.isfinite(masses) & (masses > 0)
    if not valid.any():
        return float("nan")
    mids = mids[valid]
    masses = masses[valid]
    return float((mids * masses).sum() / masses.sum())


def _mode_midpoint(frame: pd.DataFrame) -> pd.Series:
    low = pd.to_numeric(frame.get("mode_interval_low"), errors="coerce")
    high = pd.to_numeric(frame.get("mode_interval_high"), errors="coerce")
    both = low.notna() & high.notna() & np.isfinite(low) & np.isfinite(high)
    low_only = low.notna() & np.isfinite(low) & ~(high.notna() & np.isfinite(high))
    high_only = high.notna() & np.isfinite(high) & ~(low.notna() & np.isfinite(low))
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    out.loc[both] = ((low.loc[both] + high.loc[both]) / 2.0).to_numpy()
    out.loc[low_only] = low.loc[low_only].to_numpy()
    out.loc[high_only] = high.loc[high_only].to_numpy()
    return out


def _event_period_from_ticker(event_ticker: object) -> str | None:
    x = str(event_ticker)
    if len(x) < 5:
        return None

    # Kalshi macro tickers end with YYMON where MON is a 3-letter month, e.g.:
    #   CPI-21DEC, KXU3-24NOV, KXPCECORE-25JAN, FED-23MAR
    suffix = x[-5:]
    year = suffix[:2]
    month = suffix[2:]
    if not year.isdigit():
        return None
    if month not in MONTHS:
        return None
    return f"20{year}-{MONTHS[month]:02d}"


def _load_macro_realized_from_report_cache(series_id: str, *, yoy: bool = False, mom: bool = False) -> pd.DataFrame | None:
    """Fallback realized-loader when FRED is unavailable.

    Uses previously exported report artifacts (event-level predictions) as a lightweight
    realized cache keyed by `event_period`.
    """
    key = (str(series_id), bool(yoy), bool(mom))
    target_map: dict[tuple[str, bool, bool], str] = {
        ("CPIAUCSL", False, True): "cpi_mom",
        ("CPIAUCNS", True, False): "cpi_yoy",
        ("UNRATE", False, False): "u3",
        ("PCEPILFE", False, True): "pcecore_mom",
    }
    target = target_map.get(key)
    if target is None:
        return None

    frames: list[pd.DataFrame] = []
    for path in [REPORT / "best_event_level_predictions_selective.csv", REPORT / "best_event_level_predictions.csv"]:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not {"target", "event_period", "realized"}.issubset(set(df.columns)):
            continue
        g = df[df["target"].astype(str) == target][["event_period", "realized"]].copy()
        g["event_period"] = g["event_period"].astype(str)
        g["realized"] = pd.to_numeric(g["realized"], errors="coerce")
        g = g.dropna(subset=["event_period", "realized"]).drop_duplicates()
        if not g.empty:
            frames.append(g)

    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    return out.sort_values(["event_period"]).reset_index(drop=True)[["event_period", "realized"]]


@lru_cache(maxsize=None)
def _load_macro_realized(series_id: str, *, yoy: bool = False, mom: bool = False) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    # FRED occasionally fails with SSL handshake / EOF issues in some environments.
    # Keep a lightweight local cache so grid searches are reproducible and don't depend
    # on transient network failures.
    cache_dir = ROOT / "cache" / "fred"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{series_id}.csv"
    try:
        df = pd.read_csv(url)
        try:
            df.to_csv(cache_path, index=False, encoding="utf-8")
        except Exception:
            pass
    except Exception:
        if cache_path.exists():
            df = pd.read_csv(cache_path)
        else:
            cached = _load_macro_realized_from_report_cache(series_id, yoy=yoy, mom=mom)
            if cached is not None and not cached.empty:
                return cached
            raise
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else "observation_date"
    value_col = [c for c in df.columns if c != date_col][0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["realized"] = pd.to_numeric(df[value_col].replace(".", np.nan), errors="coerce")
    # Keep NaNs before computing YoY / MoM so the lag stays aligned to calendar months.
    # Some series may have blank values for specific months in `fredgraph.csv`.
    df = df.dropna(subset=[date_col]).sort_values(date_col).copy()
    if yoy:
        df["realized"] = (df["realized"] / df["realized"].shift(12) - 1.0) * 100.0
    if mom:
        df["realized"] = (df["realized"] / df["realized"].shift(1) - 1.0) * 100.0

    # Contract settlement values for these macro series are single-decimal in Kalshi contract terms,
    # so match that scale when scoring point forecasts.
    df["realized"] = df["realized"].round(1)
    df["event_period"] = df[date_col].dt.to_period("M").astype(str)
    return df.dropna(subset=["event_period", "realized"])[["event_period", "realized"]]


def _load_fomc_realized() -> pd.DataFrame:
    df = pd.read_csv(OUT / "fomc_realized_decisions.csv")
    df["event_date"] = pd.to_datetime(df["fomc_date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    for col in ["change_bps", "ffr_upper", "ffr_lower"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["event_date"])


def _load_prediction_frame(path: Path, *, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trade_ts"] = pd.to_datetime(df["trade_ts"], utc=True, errors="coerce")
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["trade_ts", "event_ts"]).copy()
    df["trade_date"] = df["trade_ts"].dt.tz_convert(None).dt.normalize()
    df["event_date"] = df["event_ts"].dt.tz_convert(None).dt.normalize()
    df["days_to_event"] = (df["event_date"] - df["trade_date"]).dt.days.astype(float)
    if "event_ticker" in df.columns:
        df["event_period"] = df["event_ticker"].map(_event_period_from_ticker)
    else:
        df["event_period"] = None

    df["pred_mode_mid"] = _mode_midpoint(df)
    for n in TOPNS:
        df[f"pred_top{n}"] = df["interval_distribution"].apply(lambda x, topn=n: _topn_prediction(x, topn)) if "interval_distribution" in df.columns else np.nan
    if "mode_interval_mass" in df.columns:
        mass = pd.to_numeric(df["mode_interval_mass"], errors="coerce").clip(0.0, 1.0)
        top2 = pd.to_numeric(df.get("pred_top2"), errors="coerce")
        df["pred_shrunk"] = mass * pd.to_numeric(df["pred_mode_mid"], errors="coerce") + (1.0 - mass) * top2
        df["pred_conf_weighted"] = mass * pd.to_numeric(df["pred_mode_mid"], errors="coerce")
    else:
        df["pred_shrunk"] = pd.to_numeric(df["pred_mode_mid"], errors="coerce")
        df["pred_conf_weighted"] = pd.to_numeric(df["pred_mode_mid"], errors="coerce")
    df["target"] = target
    return df


def _pick_snapshot(pred: pd.DataFrame, *, key: str, window: int) -> pd.DataFrame:
    work = pred.copy()
    work["window_gap"] = (work["days_to_event"] - float(window)).abs()
    work = work.sort_values([key, "window_gap", "trade_date"], kind="mergesort")
    return work.groupby(key, as_index=False).first()


def _pick_snapshot_strict_pre(pred: pd.DataFrame, *, key: str, window: int) -> pd.DataFrame:
    """Pick the last observation at least `window` days before the event (no lookahead).

    For each event key, we select the latest trade that occurred on or before:
        event_ts - window days

    This matches the practical question "can we know the value N days in advance?".
    """
    work = pred.copy()
    cutoff = work["event_ts"] - pd.to_timedelta(int(window), unit="D")
    work = work[work["trade_ts"] <= cutoff].copy()
    if work.empty:
        return work
    work = work.sort_values([key, "trade_ts"], kind="mergesort")
    return work.groupby(key, as_index=False).tail(1)


def _iter_predictor_cols(frame: pd.DataFrame) -> Iterable[str]:
    for col in ["pred_mode_mid", "pred_top1", "pred_top2", "pred_top3", "pred_top5", "pred_top10", "pred_shrunk", "pred_conf_weighted"]:
        if col in frame.columns:
            yield col


def _distribution_features(value: object) -> dict[str, float]:
    """Extract uncertainty / shape features from the interval distribution.

    Kalshi interval distributions nearly always include one or two "infinite tail" bins.
    We still compute entropy / concentration on masses (well-defined), and compute
    midpoint/width stats only on finite endpoints where possible.
    """
    dist = _parse_distribution(value)
    if not dist:
        return {
            "feat_bins": float("nan"),
            "feat_entropy": float("nan"),
            "feat_hhi": float("nan"),
            "feat_eff_bins": float("nan"),
            "feat_top2_mass": float("nan"),
            "feat_top3_mass": float("nan"),
            "feat_tail_mass": float("nan"),
            "feat_tail_low_mass": float("nan"),
            "feat_tail_high_mass": float("nan"),
            "feat_mid_mean": float("nan"),
            "feat_mid_std": float("nan"),
            "feat_width_mean": float("nan"),
            "feat_width_mass": float("nan"),
        }

    masses = np.array([float(x["mass"]) for x in dist], dtype=float)
    mass_sum = float(np.nansum(masses))
    if not np.isfinite(mass_sum) or mass_sum <= 0:
        return {
            "feat_bins": float(len(dist)),
            "feat_entropy": float("nan"),
            "feat_hhi": float("nan"),
            "feat_eff_bins": float("nan"),
            "feat_top2_mass": float("nan"),
            "feat_top3_mass": float("nan"),
            "feat_tail_mass": float("nan"),
            "feat_tail_low_mass": float("nan"),
            "feat_tail_high_mass": float("nan"),
            "feat_mid_mean": float("nan"),
            "feat_mid_std": float("nan"),
            "feat_width_mean": float("nan"),
            "feat_width_mass": float("nan"),
        }

    p = masses / mass_sum
    p = np.clip(p, 0.0, 1.0)
    # Shannon entropy and concentration (HHI) over interval masses.
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    hhi = float((p**2).sum())
    eff_bins = float(1.0 / hhi) if hhi > 0 else float("nan")
    p_sorted = np.sort(p)
    top2 = float(p_sorted[-2:].sum()) if len(p_sorted) >= 2 else float(p_sorted[-1])
    top3 = float(p_sorted[-3:].sum()) if len(p_sorted) >= 3 else float(p_sorted[-1])

    lows = np.array([float(x["low"]) for x in dist], dtype=float)
    highs = np.array([float(x["high"]) for x in dist], dtype=float)
    mids = np.array([float(x["mid"]) for x in dist], dtype=float)

    tail_low = ~np.isfinite(lows)
    tail_high = ~np.isfinite(highs)
    tail_mass = float(p[tail_low | tail_high].sum())
    tail_low_mass = float(p[tail_low].sum())
    tail_high_mass = float(p[tail_high].sum())

    mid_ok = np.isfinite(mids) & np.isfinite(p) & (p > 0)
    if mid_ok.any():
        p_mid = p[mid_ok]
        p_mid = p_mid / p_mid.sum()
        mids_ok = mids[mid_ok]
        mid_mean = float((mids_ok * p_mid).sum())
        mid_var = float(((mids_ok - mid_mean) ** 2 * p_mid).sum())
        mid_std = float(np.sqrt(max(mid_var, 0.0)))
    else:
        mid_mean = float("nan")
        mid_std = float("nan")

    widths = highs - lows
    width_ok = np.isfinite(widths) & np.isfinite(p) & (p > 0)
    if width_ok.any():
        p_w = p[width_ok]
        p_w = p_w / p_w.sum()
        widths_ok = widths[width_ok]
        width_mean = float((widths_ok * p_w).sum())
        width_mass = float(p[width_ok].sum())
    else:
        width_mean = float("nan")
        width_mass = 0.0

    return {
        "feat_bins": float(len(dist)),
        "feat_entropy": entropy,
        "feat_hhi": hhi,
        "feat_eff_bins": eff_bins,
        "feat_top2_mass": top2,
        "feat_top3_mass": top3,
        "feat_tail_mass": tail_mass,
        "feat_tail_low_mass": tail_low_mass,
        "feat_tail_high_mass": tail_high_mass,
        "feat_mid_mean": mid_mean,
        "feat_mid_std": mid_std,
        "feat_width_mean": width_mean,
        "feat_width_mass": width_mass,
    }


def _add_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    # Interval-distribution derived features
    if "interval_distribution" in work.columns:
        feats = work["interval_distribution"].apply(_distribution_features).apply(pd.Series)
        work = pd.concat([work.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    else:
        # Keep schema consistent
        for col in [
            "feat_bins",
            "feat_entropy",
            "feat_hhi",
            "feat_eff_bins",
            "feat_top2_mass",
            "feat_top3_mass",
            "feat_tail_mass",
            "feat_tail_low_mass",
            "feat_tail_high_mass",
            "feat_mid_mean",
            "feat_mid_std",
            "feat_width_mean",
            "feat_width_mass",
        ]:
            work[col] = np.nan

    # Mode interval width and one-sidedness
    low = pd.to_numeric(work.get("mode_interval_low"), errors="coerce")
    high = pd.to_numeric(work.get("mode_interval_high"), errors="coerce")
    both_finite = low.notna() & high.notna() & np.isfinite(low) & np.isfinite(high)
    work["feat_mode_width"] = np.where(both_finite, (high - low).to_numpy(), np.nan)
    work["feat_mode_one_sided"] = np.where((~np.isfinite(low)) | (~np.isfinite(high)), 1.0, 0.0)

    # Always keep these numeric columns present.
    for col in ["mode_interval_mass", "mode_probability_mode", "n_contracts", "days_to_event"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        else:
            work[col] = np.nan
    return work


def _feature_columns_for_model() -> list[str]:
    # Keep this small and robust across targets.
    return [
        "mode_interval_mass",
        "mode_probability_mode",
        "n_contracts",
        "days_to_event",
        "feat_entropy",
        "feat_hhi",
        "feat_eff_bins",
        "feat_top2_mass",
        "feat_top3_mass",
        "feat_tail_mass",
        "feat_mid_std",
        "feat_width_mean",
        "feat_width_mass",
        "feat_mode_width",
        "feat_mode_one_sided",
    ]


def _to_matrix(frame: pd.DataFrame, *, predictor: str, include_features: bool = True) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    work = _add_feature_columns(frame) if include_features else frame.copy()
    y = pd.to_numeric(work["realized"], errors="coerce")
    base_pred = pd.to_numeric(work[predictor], errors="coerce") if predictor in work.columns else pd.Series(np.nan, index=work.index)
    base_col = "_base_pred"
    work[base_col] = base_pred
    cols = [base_col] + (_feature_columns_for_model() if include_features else [])
    X = work[cols].copy()
    # Simple imputation: median per column.
    for c in cols:
        med = pd.to_numeric(X[c], errors="coerce").median(skipna=True)
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med if pd.notna(med) else 0.0)
    mask = base_pred.notna() & y.notna()
    return X.loc[mask].to_numpy(dtype=float), y.loc[mask].to_numpy(dtype=float), work.loc[mask].reset_index(drop=True)


def _to_matrix_multi_preds(
    frame: pd.DataFrame,
    *,
    predictors: list[str],
    include_features: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Like _to_matrix, but include multiple base predictor columns."""
    work = _add_feature_columns(frame) if include_features else frame.copy()
    y = pd.to_numeric(work["realized"], errors="coerce")
    base_cols: list[str] = []
    for p in predictors:
        if p in work.columns:
            col = f"_pred_{p}"
            work[col] = pd.to_numeric(work[p], errors="coerce")
            base_cols.append(col)
    if not base_cols:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float), work.iloc[0:0].copy(), []

    cols = base_cols + (_feature_columns_for_model() if include_features else [])
    X = work[cols].copy()
    for c in cols:
        med = pd.to_numeric(X[c], errors="coerce").median(skipna=True)
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(med if pd.notna(med) else 0.0)

    mask = y.notna()
    # require at least one base predictor present on the row
    base_ok = pd.Series(False, index=work.index)
    for c in base_cols:
        base_ok = base_ok | pd.to_numeric(work[c], errors="coerce").notna()
    mask = mask & base_ok
    return X.loc[mask].to_numpy(dtype=float), y.loc[mask].to_numpy(dtype=float), work.loc[mask].reset_index(drop=True), cols


def _quantize_prediction(pred: np.ndarray, *, step: float | None) -> np.ndarray:
    """Round predictions to the contract settlement grid.

    Many Kalshi macro contracts settle on a coarse grid (e.g. FOMC upper bound in 0.25 increments,
    CPI/PCE/U3 reported to 0.1). Quantizing helps MAE align to how outcomes are actually scored.
    """
    if step is None:
        return pred
    s = float(step)
    if not np.isfinite(s) or s <= 0:
        return pred
    return np.round(pred / s) * s


def _time_order_key(frame: pd.DataFrame) -> pd.Series:
    if "event_date" in frame.columns:
        return pd.to_datetime(frame["event_date"], errors="coerce")
    if "event_period" in frame.columns:
        return pd.to_datetime(frame["event_period"].astype(str) + "-01", errors="coerce")
    return pd.to_datetime(pd.Series([pd.NaT] * len(frame)))


def _selective_cv_score(
    merged: pd.DataFrame,
    *,
    point_model_type: str = "ridge",
    predictor: str,
    predictor_mode: str = "single",
    reject_rate: float,
    point_alpha: float,
    err_alpha: float,
    quantize_step: float | None = None,
) -> dict[str, float]:
    # Local import to keep base script usable even without sklearn.
    from sklearn.linear_model import QuantileRegressor, Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if point_model_type not in {"ridge", "qr", "simplex"}:
        return {}
    if predictor_mode not in {"single", "stack"}:
        return {}
    if point_model_type == "simplex" and predictor_mode != "stack":
        return {}

    if predictor_mode == "single":
        # For MAE minimization, extra distribution-shape features tend to overfit and
        # can worsen CV MAE. Default to base predictor only.
        X, y, meta = _to_matrix(merged, predictor=predictor, include_features=False)
    else:
        stack_preds = ["pred_mode_mid", "pred_top1", "pred_top2", "pred_top3", "pred_top5", "pred_top10", "pred_shrunk"]
        # In practice, the base predictors already encode the distribution. Adding extra
        # shape features can overfit and *increase* CV MAE on small samples.
        X, y, meta, _ = _to_matrix_multi_preds(merged, predictors=stack_preds, include_features=False)

    if len(y) < 10 or X.size == 0:
        return {}

    order = _time_order_key(meta)
    order = order.fillna(pd.Timestamp.min)
    idx = np.argsort(order.to_numpy())
    X = X[idx]
    y = y[idx]

    # Pooled CV predictions (test-fold only) to estimate out-of-sample performance.
    pooled_y: list[float] = []
    pooled_pred: list[float] = []
    pooled_accept: list[bool] = []

    pooled_y_all: list[float] = []
    pooled_pred_all: list[float] = []

    tscv = TimeSeriesSplit(n_splits=3)
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Point model:
        # - ridge: squared loss + L2 (stable, good default)
        # - qr: median quantile regression (MAE-oriented)
        # - simplex: convex combination over base predictors (weights >=0, sum=1), MAE-oriented
        if point_model_type == "simplex":
            from scipy.optimize import minimize

            # Convex weights are more stable than unconstrained regression on small samples and avoid
            # extrapolation. Works best when X contains only the base predictors (include_features=False).
            k = int(X_tr.shape[1])
            if k <= 0:
                return {}

            # Initialize with uniform weights.
            w0 = np.full((k,), 1.0 / k, dtype=float)
            bounds = [(0.0, 1.0)] * k
            cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)

            def obj(w: np.ndarray) -> float:
                # Mean absolute error
                pred = _quantize_prediction(X_tr @ w, step=quantize_step)
                return float(np.mean(np.abs(pred - y_tr)))

            res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
            w = np.asarray(res.x, dtype=float) if res.success else w0
            # Hard project in case of numerical drift.
            w = np.clip(w, 0.0, 1.0)
            s = float(w.sum())
            w = w / s if s > 0 else w0

            # NOTE: keep predictions in the same scale as y (do NOT StandardScale here), since X already
            # contains base point-estimates. StandardScaler would change the meaning of convex weights.
            yhat_tr = _quantize_prediction(X_tr @ w, step=quantize_step)
            yhat_te = _quantize_prediction(X_te @ w, step=quantize_step)
        else:
            # Either Ridge (squared loss) or QuantileRegressor at median (L1/MAE-oriented).
            if point_model_type == "qr":
                # QuantileRegressor minimizes pinball loss; q=0.5 corresponds to median regression (MAE).
                # It includes L1 regularization controlled by alpha.
                point_model = make_pipeline(StandardScaler(), QuantileRegressor(quantile=0.5, alpha=float(point_alpha), solver="highs"))
            else:
                point_model = make_pipeline(StandardScaler(), Ridge(alpha=float(point_alpha), solver="svd"))
            point_model.fit(X_tr, y_tr)
            yhat_tr = _quantize_prediction(point_model.predict(X_tr), step=quantize_step)
            yhat_te = _quantize_prediction(point_model.predict(X_te), step=quantize_step)

        # Error model: predict absolute residual.
        abs_resid = np.abs(y_tr - yhat_tr)
        err_model = make_pipeline(StandardScaler(), Ridge(alpha=float(err_alpha), solver="svd"))
        err_model.fit(X_tr, abs_resid)

        # Calibrate rejection threshold on *train* predicted error.
        errhat_tr = np.clip(err_model.predict(X_tr), 0.0, np.inf)
        if reject_rate <= 0.0:
            thr = np.inf
        else:
            q = 1.0 - float(reject_rate)
            thr = float(np.quantile(errhat_tr, q))

        errhat_te = np.clip(err_model.predict(X_te), 0.0, np.inf)
        accept = errhat_te <= thr

        pooled_y_all.extend(y_te.tolist())
        pooled_pred_all.extend(yhat_te.tolist())

        pooled_y.extend(y_te[accept].tolist())
        pooled_pred.extend(yhat_te[accept].tolist())
        pooled_accept.extend(accept.tolist())

    if not pooled_y_all:
        return {}

    y_all = np.array(pooled_y_all, dtype=float)
    pred_all = np.array(pooled_pred_all, dtype=float)
    err_all = pred_all - y_all
    mae_all = float(np.mean(np.abs(err_all)))
    rmse_all = float(np.sqrt(np.mean(err_all**2)))

    if not pooled_y:
        return {
            "mae_all": mae_all,
            "rmse_all": rmse_all,
            "mae": float("nan"),
            "rmse": float("nan"),
            "coverage": 0.0,
        }

    y_acc = np.array(pooled_y, dtype=float)
    pred_acc = np.array(pooled_pred, dtype=float)
    err_acc = pred_acc - y_acc
    mae = float(np.mean(np.abs(err_acc)))
    rmse = float(np.sqrt(np.mean(err_acc**2)))
    coverage = float(len(y_acc) / len(y_all))
    return {
        "mae_all": mae_all,
        "rmse_all": rmse_all,
        "mae": mae,
        "rmse": rmse,
        "coverage": coverage,
    }


def grid_search_selective_point_forecasts(
    *,
    strict_pre: bool = True,
    min_n: int = 10,
    max_reject: float = 0.10,
    include_qr: bool = False,
    include_simplex: bool = False,
) -> pd.DataFrame:
    """Grid search: point model + reject option (<= max_reject) to reduce MAE at larger windows.

    Uses strictly pre-event snapshots by default (no lookahead) and evaluates out-of-sample via
    a small time-series cross-validation.
    """
    results: list[dict[str, object]] = []

    macro_specs = [
        (
            "cpi_mom",
            OUT / "kalshi_modes_rebuilt_cpi_daily.csv",
            _load_macro_realized("CPIAUCSL", mom=True),
            "event_period",
            "realized",
            {"CPI", "KXCPI"},
        ),
        (
            "cpi_yoy",
            OUT / "kalshi_modes_rebuilt_cpi_daily.csv",
            _load_macro_realized("CPIAUCNS", yoy=True),
            "event_period",
            "realized",
            {"CPIYOY", "KXCPIYOY"},
        ),
        ("u3", OUT / "kalshi_modes_rebuilt_u3_daily.csv", _load_macro_realized("UNRATE", yoy=False), "event_period", "realized", None),
        ("pcecore_mom", OUT / "kalshi_modes_rebuilt_pcecore_daily.csv", _load_macro_realized("PCEPILFE", mom=True), "event_period", "realized", {"PCECORE", "KXPCECORE"}),
    ]

    # FOMC: use ffr_upper as realized level. (Note: this is not a macro release but a decision date.)
    fomc_realized = _load_fomc_realized()
    fomc_pred = _load_prediction_frame(OUT / "kalshi_modes_rebuilt_fomc_daily.csv", target="fomc")
    for window in WINDOWS:
        snap = _pick_snapshot_strict_pre(fomc_pred, key="event_date", window=window) if strict_pre else _pick_snapshot(fomc_pred, key="event_date", window=window)
        merged = snap.merge(fomc_realized[["event_date", "ffr_upper"]].rename(columns={"ffr_upper": "realized"}), on="event_date", how="inner")
        # Evaluate both: single predictor and stacked combination.
        predictor_modes = [("single", None), ("stack", "stack")]
        for mode, predictor_name in predictor_modes:
            if mode == "single":
                predictors = list(_iter_predictor_cols(merged))
            else:
                predictors = ["stack"]

            for predictor in predictors:
                base = merged.copy()
                if mode == "single":
                    valid = pd.to_numeric(base[predictor], errors="coerce").notna() & pd.to_numeric(base["realized"], errors="coerce").notna()
                else:
                    # Require realized and at least one base prediction present.
                    y_ok = pd.to_numeric(base["realized"], errors="coerce").notna()
                    pred_ok = pd.Series(False, index=base.index)
                    for c in ["pred_mode_mid", "pred_top1", "pred_top2", "pred_top3", "pred_top5", "pred_top10", "pred_shrunk"]:
                        if c in base.columns:
                            pred_ok = pred_ok | pd.to_numeric(base[c], errors="coerce").notna()
                    valid = y_ok & pred_ok
                if int(valid.sum()) < int(min_n):
                    continue

                for reject_rate in REJECT_RATES:
                    if float(reject_rate) > float(max_reject):
                        continue
                    for alpha in RIDGE_ALPHAS:
                        score = _selective_cv_score(
                            base.loc[valid],
                            point_model_type="ridge",
                            predictor=predictor,
                            predictor_mode=mode,
                            reject_rate=float(reject_rate),
                            point_alpha=float(alpha),
                            err_alpha=float(alpha),
                            quantize_step=0.25,
                        )
                        if not score:
                            continue
                        results.append(
                            {
                                "target": "fomc",
                                "strict_pre": bool(strict_pre),
                                "window": int(window),
                                "predictor": str(predictor_name if predictor_name is not None else predictor),
                                "predictor_mode": str(mode),
                                "point_model": "ridge",
                                "reject_rate": float(reject_rate),
                                "alpha": float(alpha),
                                "n": int(valid.sum()),
                                "coverage": float(score["coverage"]),
                                "MAE_all": float(score["mae_all"]),
                                "RMSE_all": float(score["rmse_all"]),
                                    "MAE": float(score["mae"]),
                                    "RMSE": float(score["rmse"]),
                                }
                            )
                    if include_simplex and mode == "stack":
                        # Simplex-constrained convex combo over base predictors (MAE-oriented).
                        # alpha is not used by this point model, but we keep schema consistent.
                        score = _selective_cv_score(
                            base.loc[valid],
                            point_model_type="simplex",
                            predictor=predictor,
                            predictor_mode=mode,
                            reject_rate=float(reject_rate),
                            point_alpha=0.0,
                            err_alpha=float(alpha),
                            quantize_step=0.25,
                        )
                        if score:
                            results.append(
                                {
                                    "target": "fomc",
                                    "strict_pre": bool(strict_pre),
                                    "window": int(window),
                                    "predictor": str(predictor_name if predictor_name is not None else predictor),
                                    "predictor_mode": str(mode),
                                    "point_model": "simplex",
                                    "reject_rate": float(reject_rate),
                                    "alpha": 0.0,
                                    "n": int(valid.sum()),
                                    "coverage": float(score["coverage"]),
                                    "MAE_all": float(score["mae_all"]),
                                    "RMSE_all": float(score["rmse_all"]),
                                    "MAE": float(score["mae"]),
                                    "RMSE": float(score["rmse"]),
                                }
                            )
                    if include_qr:
                        # QuantileRegressor (median regression) for MAE-friendly point forecasts.
                        for alpha in QR_ALPHAS:
                            score = _selective_cv_score(
                                base.loc[valid],
                                point_model_type="qr",
                                predictor=predictor,
                                predictor_mode=mode,
                                reject_rate=float(reject_rate),
                                point_alpha=float(alpha),
                                err_alpha=float(alpha),
                                quantize_step=0.25,
                            )
                            if not score:
                                continue
                            results.append(
                                {
                                    "target": "fomc",
                                    "strict_pre": bool(strict_pre),
                                    "window": int(window),
                                    "predictor": str(predictor_name if predictor_name is not None else predictor),
                                    "predictor_mode": str(mode),
                                    "point_model": "qr",
                                    "reject_rate": float(reject_rate),
                                    "alpha": float(alpha),
                                    "n": int(valid.sum()),
                                    "coverage": float(score["coverage"]),
                                    "MAE_all": float(score["mae_all"]),
                                    "RMSE_all": float(score["rmse_all"]),
                                    "MAE": float(score["mae"]),
                                    "RMSE": float(score["rmse"]),
                                }
                            )

    for spec in macro_specs:
        target, path, realized, key, y_col, prefixes = spec
        pred = _load_prediction_frame(path, target=target)
        if prefixes and "event_ticker" in pred.columns:
            prefix = pred["event_ticker"].astype(str).str.split("-", n=1).str[0]
            pred = pred[prefix.isin(set(prefixes))].copy()
        for window in WINDOWS:
            snap = _pick_snapshot_strict_pre(pred, key=key, window=window) if strict_pre else _pick_snapshot(pred, key=key, window=window)
            merged = snap.merge(realized.rename(columns={y_col: "realized"}), on=key, how="inner")
            predictor_modes = [("single", None), ("stack", "stack")]
            for mode, predictor_name in predictor_modes:
                if mode == "single":
                    predictors = list(_iter_predictor_cols(merged))
                else:
                    predictors = ["stack"]

                for predictor in predictors:
                    base = merged.copy()
                    if mode == "single":
                        valid = pd.to_numeric(base[predictor], errors="coerce").notna() & pd.to_numeric(base["realized"], errors="coerce").notna()
                    else:
                        y_ok = pd.to_numeric(base["realized"], errors="coerce").notna()
                        pred_ok = pd.Series(False, index=base.index)
                        for c in ["pred_mode_mid", "pred_top1", "pred_top2", "pred_top3", "pred_top5", "pred_top10", "pred_shrunk"]:
                            if c in base.columns:
                                pred_ok = pred_ok | pd.to_numeric(base[c], errors="coerce").notna()
                        valid = y_ok & pred_ok
                    if int(valid.sum()) < int(min_n):
                        continue
                    for reject_rate in REJECT_RATES:
                        if float(reject_rate) > float(max_reject):
                            continue
                        for alpha in RIDGE_ALPHAS:
                            score = _selective_cv_score(
                                base.loc[valid],
                                point_model_type="ridge",
                                predictor=predictor,
                                predictor_mode=mode,
                                reject_rate=float(reject_rate),
                                point_alpha=float(alpha),
                                err_alpha=float(alpha),
                                quantize_step=0.1,
                            )
                            if not score:
                                continue
                            results.append(
                                {
                                    "target": str(target),
                                    "strict_pre": bool(strict_pre),
                                    "window": int(window),
                                    "predictor": str(predictor_name if predictor_name is not None else predictor),
                                    "predictor_mode": str(mode),
                                    "point_model": "ridge",
                                    "reject_rate": float(reject_rate),
                                    "alpha": float(alpha),
                                    "n": int(valid.sum()),
                                    "coverage": float(score["coverage"]),
                                    "MAE_all": float(score["mae_all"]),
                                    "RMSE_all": float(score["rmse_all"]),
                                    "MAE": float(score["mae"]),
                                        "RMSE": float(score["rmse"]),
                                    }
                                )
                        if include_simplex and mode == "stack":
                            score = _selective_cv_score(
                                base.loc[valid],
                                point_model_type="simplex",
                                predictor=predictor,
                                predictor_mode=mode,
                                reject_rate=float(reject_rate),
                                point_alpha=0.0,
                                err_alpha=float(alpha),
                                quantize_step=0.1,
                            )
                            if score:
                                results.append(
                                    {
                                        "target": str(target),
                                        "strict_pre": bool(strict_pre),
                                        "window": int(window),
                                        "predictor": str(predictor_name if predictor_name is not None else predictor),
                                        "predictor_mode": str(mode),
                                        "point_model": "simplex",
                                        "reject_rate": float(reject_rate),
                                        "alpha": 0.0,
                                        "n": int(valid.sum()),
                                        "coverage": float(score["coverage"]),
                                        "MAE_all": float(score["mae_all"]),
                                        "RMSE_all": float(score["rmse_all"]),
                                        "MAE": float(score["mae"]),
                                        "RMSE": float(score["rmse"]),
                                    }
                                )
                        if include_qr:
                            for alpha in QR_ALPHAS:
                                score = _selective_cv_score(
                                    base.loc[valid],
                                    point_model_type="qr",
                                    predictor=predictor,
                                    predictor_mode=mode,
                                    reject_rate=float(reject_rate),
                                    point_alpha=float(alpha),
                                    err_alpha=float(alpha),
                                    quantize_step=0.1,
                                )
                                if not score:
                                    continue
                                results.append(
                                    {
                                        "target": str(target),
                                        "strict_pre": bool(strict_pre),
                                        "window": int(window),
                                        "predictor": str(predictor_name if predictor_name is not None else predictor),
                                        "predictor_mode": str(mode),
                                        "point_model": "qr",
                                        "reject_rate": float(reject_rate),
                                        "alpha": float(alpha),
                                        "n": int(valid.sum()),
                                        "coverage": float(score["coverage"]),
                                        "MAE_all": float(score["mae_all"]),
                                        "RMSE_all": float(score["rmse_all"]),
                                        "MAE": float(score["mae"]),
                                        "RMSE": float(score["rmse"]),
                                    }
                                )

    return pd.DataFrame(results)


def _run_targeted_simplex_test_cpi_mom_window7() -> None:
    """Quick sanity check: does simplex-constrained stack reduce CPI MoM MAE at window=7?

    This is not a full report run; it prints a small table to stdout and is meant for fast iteration.
    """
    strict_pre = True
    window = 7
    base = _load_prediction_frame(OUT / "kalshi_modes_rebuilt_cpi_daily.csv", target="cpi_mom")
    realized = _load_macro_realized("CPIAUCSL", mom=True)

    # Match grid_search() behavior: filter to CPI MoM tickers only (exclude CPIYOY, shelter, gas, ...)
    prefix = base["event_ticker"].astype(str).str.split("-", n=1).str[0]
    base = base[prefix.isin({"CPI", "KXCPI"})].copy()

    snap = _pick_snapshot_strict_pre(base, key="event_period", window=window) if strict_pre else _pick_snapshot(base, key="event_period", window=window)
    merged = snap.merge(realized[["event_period", "realized"]], on="event_period", how="inner")

    # stack uses only base predictors as features (see _selective_cv_score include_features=False)
    y_ok = pd.to_numeric(merged["realized"], errors="coerce").notna()
    pred_ok = pd.Series(False, index=merged.index)
    for c in ["pred_mode_mid", "pred_top1", "pred_top2", "pred_top3", "pred_top5", "pred_top10", "pred_shrunk"]:
        if c in merged.columns:
            pred_ok = pred_ok | pd.to_numeric(merged[c], errors="coerce").notna()
    valid = y_ok & pred_ok
    merged = merged.loc[valid].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    # Compare: current best (ridge stack) vs simplex-constrained convex stack.
    for point_model_type, alpha in [("ridge", 10.0), ("simplex", 0.0)]:
        for reject_rate in [0.0, 0.05, 0.10]:
            score = _selective_cv_score(
                merged,
                point_model_type=point_model_type,
                predictor="stack",
                predictor_mode="stack",
                reject_rate=float(reject_rate),
                point_alpha=float(alpha),
                err_alpha=10.0,
                quantize_step=0.1,
            )
            if not score:
                continue
            rows.append(
                {
                    "point_model": point_model_type,
                    "reject_rate": float(reject_rate),
                    "alpha": float(alpha),
                    "coverage": float(score["coverage"]),
                    "MAE": float(score["mae"]),
                    "RMSE": float(score["rmse"]),
                    "MAE_all": float(score["mae_all"]),
                    "RMSE_all": float(score["rmse_all"]),
                    "n": int(len(merged)),
                }
            )

    if not rows:
        print("No results (insufficient data?)")
        return
    out = pd.DataFrame(rows).sort_values(["MAE", "reject_rate", "point_model"])
    print("Targeted test: cpi_mom window=7 (stack predictors)")
    print(out.to_string(index=False))
    # For convenience when iterating on MAE <= 0.06:
    best = out.iloc[0].to_dict()
    print("Best:", best)


def export_best_event_level_predictions_selective(selective_best: pd.DataFrame, *, strict_pre: bool = True) -> pd.DataFrame:
    """Export per-event predictions for the best selective configs (per target).

    This fits the (point + error) models on all available realized events and then
    marks each event as accepted/rejected based on the configured reject_rate.
    """
    if selective_best.empty:
        return pd.DataFrame()

    from sklearn.linear_model import QuantileRegressor, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    stack_base_predictors = ["pred_mode_mid", "pred_top1", "pred_top2", "pred_top3", "pred_top5", "pred_top10", "pred_shrunk"]

    pred_sources = {
        "fomc": OUT / "kalshi_modes_rebuilt_fomc_daily.csv",
        "cpi_mom": OUT / "kalshi_modes_rebuilt_cpi_daily.csv",
        "cpi_yoy": OUT / "kalshi_modes_rebuilt_cpi_daily.csv",
        "u3": OUT / "kalshi_modes_rebuilt_u3_daily.csv",
        "pcecore_mom": OUT / "kalshi_modes_rebuilt_pcecore_daily.csv",
    }
    realized_sources = {
        "fomc": ("fomc", _load_fomc_realized(), "event_date", "ffr_upper"),
        "cpi_mom": ("cpi_mom", _load_macro_realized("CPIAUCSL", mom=True), "event_period", "realized"),
        "cpi_yoy": ("cpi_yoy", _load_macro_realized("CPIAUCNS", yoy=True), "event_period", "realized"),
        "u3": ("u3", _load_macro_realized("UNRATE"), "event_period", "realized"),
        "pcecore_mom": ("pcecore_mom", _load_macro_realized("PCEPILFE", mom=True), "event_period", "realized"),
    }
    prefix_filters = {
        "cpi_mom": {"CPI", "KXCPI"},
        "cpi_yoy": {"CPIYOY", "KXCPIYOY"},
        "pcecore_mom": {"PCECORE", "KXPCECORE"},
    }

    rows: list[pd.DataFrame] = []
    for r in selective_best.itertuples(index=False):
        target = str(r.target)
        window = int(r.window)
        predictor = str(r.predictor)
        reject_rate = float(r.reject_rate)
        alpha = float(r.alpha)
        predictor_mode = getattr(r, "predictor_mode", "stack" if predictor == "stack" else "single")
        pm = getattr(r, "point_model", "ridge")
        point_model_type = str(pm) if pm is not None else "ridge"
        if point_model_type not in {"ridge", "qr"}:
            point_model_type = "ridge"
        quantize_step = 0.25 if target == "fomc" else 0.1

        src = pred_sources.get(target)
        if src is None or not src.exists():
            continue
        pred = _load_prediction_frame(src, target=target)
        if target in prefix_filters and "event_ticker" in pred.columns:
            prefix = pred["event_ticker"].astype(str).str.split("-", n=1).str[0]
            pred = pred[prefix.isin(prefix_filters[target])].copy()
        if predictor_mode != "stack" and predictor not in pred.columns:
            continue

        _, realized, key, y_col = realized_sources[target]
        snap = _pick_snapshot_strict_pre(pred, key=key, window=window) if strict_pre else _pick_snapshot(pred, key=key, window=window)
        if snap.empty:
            continue

        # Fit on realized-known rows only.
        merged = snap.merge(realized.rename(columns={y_col: "realized"}), on=key, how="left")
        train = merged.dropna(subset=["realized"]).copy()
        if train.empty:
            continue

        if predictor_mode == "stack":
            X_tr, y_tr, _, cols = _to_matrix_multi_preds(train, predictors=stack_base_predictors, include_features=False)
            if X_tr.size == 0 or len(y_tr) == 0:
                continue

            if point_model_type == "qr":
                point_model = make_pipeline(StandardScaler(), QuantileRegressor(quantile=0.5, alpha=float(alpha), solver="highs"))
            else:
                point_model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha), solver="svd"))
            point_model.fit(X_tr, y_tr)
            yhat_tr = _quantize_prediction(point_model.predict(X_tr), step=quantize_step)
            abs_resid = np.abs(y_tr - yhat_tr)
            err_model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha), solver="svd"))
            err_model.fit(X_tr, abs_resid)
            errhat_tr = np.clip(err_model.predict(X_tr), 0.0, np.inf)
            thr = float(np.quantile(errhat_tr, 1.0 - reject_rate)) if reject_rate > 0 else np.inf

            full = merged.copy()
            for pcol in stack_base_predictors:
                if pcol in full.columns:
                    full[f"_pred_{pcol}"] = pd.to_numeric(full[pcol], errors="coerce")
            X_full = full[cols].copy()
            for c in cols:
                med = pd.to_numeric(X_full[c], errors="coerce").median(skipna=True)
                X_full[c] = pd.to_numeric(X_full[c], errors="coerce").fillna(med if pd.notna(med) else 0.0)
            X_full_np = X_full.to_numpy(dtype=float)
        else:
            X_tr, y_tr, _ = _to_matrix(train, predictor=predictor, include_features=False)
            if X_tr.size == 0 or len(y_tr) == 0:
                continue

            if point_model_type == "qr":
                point_model = make_pipeline(StandardScaler(), QuantileRegressor(quantile=0.5, alpha=float(alpha), solver="highs"))
            else:
                point_model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha), solver="svd"))
            point_model.fit(X_tr, y_tr)
            yhat_tr = _quantize_prediction(point_model.predict(X_tr), step=quantize_step)
            abs_resid = np.abs(y_tr - yhat_tr)
            err_model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha), solver="svd"))
            err_model.fit(X_tr, abs_resid)
            errhat_tr = np.clip(err_model.predict(X_tr), 0.0, np.inf)
            thr = float(np.quantile(errhat_tr, 1.0 - reject_rate)) if reject_rate > 0 else np.inf

            # Predict for all events (including future/unrealized).
            full = merged.copy()
            full["_base_pred"] = pd.to_numeric(full[predictor], errors="coerce")
            cols = ["_base_pred"]
            X_full = full[cols].copy()
            for c in cols:
                med = pd.to_numeric(X_full[c], errors="coerce").median(skipna=True)
                X_full[c] = pd.to_numeric(X_full[c], errors="coerce").fillna(med if pd.notna(med) else 0.0)
            X_full_np = X_full.to_numpy(dtype=float)

        pred_point = _quantize_prediction(point_model.predict(X_full_np), step=quantize_step)
        pred_err = np.clip(err_model.predict(X_full_np), 0.0, np.inf)
        accept = pred_err <= thr

        out = merged[["target", "event_ticker", "market_ticker", "trade_date", "event_date", "event_period", "days_to_event", "mode_interval_mass", "n_contracts"]].copy()
        out["window"] = window
        out["predictor"] = predictor
        out["predictor_mode"] = predictor_mode
        out["point_model"] = point_model_type
        out["reject_rate"] = reject_rate
        out["alpha"] = alpha
        out["pred"] = pred_point
        out["pred_abs_err_hat"] = pred_err
        out["accepted"] = accept.astype(bool)
        out = out.merge(realized.rename(columns={y_col: "realized"}), on=key, how="left")
        out["abs_error"] = (pd.to_numeric(out["pred"], errors="coerce") - pd.to_numeric(out["realized"], errors="coerce")).abs()
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    cols = [
        c
        for c in [
            "target",
            "event_period",
            "event_date",
            "trade_date",
            "window",
            "predictor",
            "predictor_mode",
            "point_model",
            "reject_rate",
            "alpha",
            "pred",
            "pred_abs_err_hat",
            "accepted",
            "realized",
            "abs_error",
            "mode_interval_mass",
            "n_contracts",
            "event_ticker",
            "market_ticker",
        ]
        if c in out.columns
    ]
    extra = [c for c in out.columns if c not in cols]
    return out[cols + extra].sort_values(["target", "event_date", "event_period", "trade_date"], na_position="last").reset_index(drop=True)


def _pareto_frontier_by_window(best_by_window: pd.DataFrame) -> pd.DataFrame:
    """Compute a simple 2D Pareto frontier: maximize window, minimize MAE.

    Input should already be best-per-(target,window) (one row per window).
    """
    if best_by_window.empty:
        return best_by_window

    out_rows: list[pd.DataFrame] = []
    for target, grp in best_by_window.groupby("target", sort=False):
        g = grp.sort_values(["window", "MAE", "RMSE"], ascending=[False, True, True]).copy()
        best_mae_so_far = float("inf")
        keep: list[int] = []
        for idx, row in g.iterrows():
            mae = float(row["MAE"]) if pd.notna(row["MAE"]) else float("inf")
            if mae < best_mae_so_far - 1e-12:
                keep.append(idx)
                best_mae_so_far = mae
        out_rows.append(g.loc[keep].sort_values(["window"], ascending=[False]))
    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


def export_best_event_level_predictions(point_best: pd.DataFrame, *, strict_pre: bool = True) -> pd.DataFrame:
    """Export per-event point predictions using the best (target, window, predictor) configs.

    This is the artifact you can directly use to "predict the macro value at expiry/release".
    """
    if point_best.empty:
        return pd.DataFrame()

    # Build a loader map for the prediction sources.
    pred_sources = {
        "fomc": OUT / "kalshi_modes_rebuilt_fomc_daily.csv",
        "cpi_mom": OUT / "kalshi_modes_rebuilt_cpi_daily.csv",
        "cpi_yoy": OUT / "kalshi_modes_rebuilt_cpi_daily.csv",
        "u3": OUT / "kalshi_modes_rebuilt_u3_daily.csv",
        "pcecore_mom": OUT / "kalshi_modes_rebuilt_pcecore_daily.csv",
    }
    realized_sources = {
        "fomc": ("fomc", _load_fomc_realized(), "event_date", "ffr_upper"),
        "cpi_mom": ("cpi_mom", _load_macro_realized("CPIAUCSL", mom=True), "event_period", "realized"),
        # YOY CPI: CPI-U all items, not seasonally adjusted
        "cpi_yoy": ("cpi_yoy", _load_macro_realized("CPIAUCNS", yoy=True), "event_period", "realized"),
        "u3": ("u3", _load_macro_realized("UNRATE"), "event_period", "realized"),
        "pcecore_mom": ("pcecore_mom", _load_macro_realized("PCEPILFE", mom=True), "event_period", "realized"),
    }
    prefix_filters = {
        "cpi_mom": {"CPI", "KXCPI"},
        "cpi_yoy": {"CPIYOY", "KXCPIYOY"},
        "pcecore_mom": {"PCECORE", "KXPCECORE"},
    }

    rows: list[pd.DataFrame] = []
    for r in point_best.itertuples(index=False):
        target = str(r.target)
        window = int(r.window)
        predictor = str(r.predictor)
        src = pred_sources.get(target)
        if src is None or not src.exists():
            continue
        pred = _load_prediction_frame(src, target=target)
        if target in prefix_filters and "event_ticker" in pred.columns:
            prefix = pred["event_ticker"].astype(str).str.split("-", n=1).str[0]
            pred = pred[prefix.isin(prefix_filters[target])].copy()
        if predictor not in pred.columns:
            continue

        _, realized, key, y_col = realized_sources[target]
        snap = _pick_snapshot_strict_pre(pred, key=key, window=window) if strict_pre else _pick_snapshot(pred, key=key, window=window)
        if snap.empty:
            continue
        snap = snap.rename(columns={predictor: "pred"})
        out = snap[["target", "event_ticker", "market_ticker", "trade_date", "event_date", "event_period", "days_to_event", "mode_interval_mass", "n_contracts", "pred"]].copy()
        out["window"] = window
        out["predictor"] = predictor
        # attach realized + errors when available
        merged = out.merge(realized.rename(columns={y_col: "realized"}), on=key, how="left")
        merged["abs_error"] = (pd.to_numeric(merged["pred"], errors="coerce") - pd.to_numeric(merged["realized"], errors="coerce")).abs()
        rows.append(merged)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # Put the join-key columns early for readability
    cols = [c for c in ["target", "event_period", "event_date", "trade_date", "window", "predictor", "pred", "realized", "abs_error", "mode_interval_mass", "n_contracts", "event_ticker", "market_ticker"] if c in out.columns]
    extra = [c for c in out.columns if c not in cols]
    return out[cols + extra].sort_values(["target", "event_date", "event_period", "trade_date"], na_position="last").reset_index(drop=True)


def grid_search_point_forecasts(*, strict_pre: bool = True, min_n: int = 10) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    macro_specs = [
        # These rebuilt files contain multiple CPI-like markets (CPI MoM, CPI YoY, etc).
        # To keep scoring meaningful, map each Kalshi ticker prefix to the right realized transform.
        (
            "cpi_mom",
            OUT / "kalshi_modes_rebuilt_cpi_daily.csv",
            _load_macro_realized("CPIAUCSL", mom=True),
            "event_period",
            "realized",
            {"CPI", "KXCPI"},
        ),
        (
            "cpi_yoy",
            OUT / "kalshi_modes_rebuilt_cpi_daily.csv",
            # Kalshi YOY CPI uses the CPI-U all-items index before seasonal adjustment.
            _load_macro_realized("CPIAUCNS", yoy=True),
            "event_period",
            "realized",
            {"CPIYOY", "KXCPIYOY"},
        ),
        ("u3", OUT / "kalshi_modes_rebuilt_u3_daily.csv", _load_macro_realized("UNRATE", yoy=False), "event_period", "realized"),
        ("pcecore_mom", OUT / "kalshi_modes_rebuilt_pcecore_daily.csv", _load_macro_realized("PCEPILFE", mom=True), "event_period", "realized", {"PCECORE", "KXPCECORE"}),
    ]
    fomc_realized = _load_fomc_realized()
    fomc_pred = _load_prediction_frame(OUT / "kalshi_modes_rebuilt_fomc_daily.csv", target="fomc")
    for window in WINDOWS:
        snap = _pick_snapshot_strict_pre(fomc_pred, key="event_date", window=window) if strict_pre else _pick_snapshot(fomc_pred, key="event_date", window=window)
        merged = snap.merge(fomc_realized[["event_date", "ffr_upper"]], on="event_date", how="inner")
        for predictor in _iter_predictor_cols(merged):
            pred = pd.to_numeric(merged[predictor], errors="coerce")
            y = pd.to_numeric(merged["ffr_upper"], errors="coerce")
            valid = pred.notna() & y.notna()
            if int(valid.sum()) < int(min_n):
                continue
            err = pred[valid] - y[valid]
            results.append(
                {
                    "target": "fomc",
                    "strict_pre": bool(strict_pre),
                    "window": window,
                    "predictor": predictor,
                    "n": int(valid.sum()),
                    "MAE": float(err.abs().mean()),
                    "RMSE": float(np.sqrt((err**2).mean())),
                }
            )
    for spec in macro_specs:
        # backward-compatible unpack: (target, path, realized, key, y_col, prefixes?)
        if len(spec) == 5:
            target, path, realized, key, y_col = spec
            prefixes = None
        else:
            target, path, realized, key, y_col, prefixes = spec

        pred = _load_prediction_frame(path, target=target)
        if prefixes and "event_ticker" in pred.columns:
            prefix = pred["event_ticker"].astype(str).str.split("-", n=1).str[0]
            pred = pred[prefix.isin(set(prefixes))].copy()
        for window in WINDOWS:
            snap = _pick_snapshot_strict_pre(pred, key=key, window=window) if strict_pre else _pick_snapshot(pred, key=key, window=window)
            merged = snap.merge(realized, on=key, how="inner")
            for predictor in _iter_predictor_cols(merged):
                score = pd.to_numeric(merged[predictor], errors="coerce")
                y = pd.to_numeric(merged[y_col], errors="coerce")
                valid = score.notna() & y.notna()
                if int(valid.sum()) < int(min_n):
                    continue
                err = score[valid] - y[valid]
                results.append(
                    {
                        "target": target,
                        "strict_pre": bool(strict_pre),
                        "window": window,
                        "predictor": predictor,
                        "n": int(valid.sum()),
                        "MAE": float(err.abs().mean()),
                        "RMSE": float(np.sqrt((err**2).mean())),
                    }
                )
    return pd.DataFrame(results)


def _load_fomc_signal_grid_source() -> pd.DataFrame:
    pred = _load_prediction_frame(OUT / "kalshi_modes_rebuilt_fomc_daily.csv", target="fomc")
    realized = _load_fomc_realized()[["event_date", "change_bps"]].copy()
    realized["signal_dir_truth"] = np.where(realized["change_bps"] > 0, -1, np.where(realized["change_bps"] < 0, 1, 0))
    return pred.merge(realized, on="event_date", how="inner")


def grid_search_timing() -> pd.DataFrame:
    base = _load_fomc_signal_grid_source()
    rows: list[dict[str, object]] = []
    # These yield files live at repo root (not under out_csv) and match the schema expected by load_yield().
    price_frames = {name.replace(".csv", ""): load_yield(ROOT / name) for name in YIELD_FILES}
    # Keep timing grid small: restrict to the best predictors/windows from the point-forecast grid.
    point = grid_search_point_forecasts()
    best = (
        point.sort_values(["RMSE", "MAE", "window"], ascending=[True, True, True])
        .head(6)[["predictor", "window"]]
        .drop_duplicates()
    )
    predictors = best["predictor"].astype(str).tolist()
    windows = sorted({int(x) for x in best["window"].tolist()})

    for predictor in predictors:
        if predictor not in base.columns:
            continue
        for window in windows:
            snap = _pick_snapshot_strict_pre(base, key="event_date", window=window)
            if snap.empty:
                continue
            signal_strength = pd.to_numeric(snap[predictor], errors="coerce")
            for threshold in SIGNAL_THRESHOLDS:
                sig = snap.copy()
                centered = signal_strength - signal_strength.median(skipna=True)
                sig["signal_strength"] = centered
                sig["signal_dir"] = np.where(centered > threshold, -1, np.where(centered < -threshold, 1, 0))
                sig = sig[sig["signal_dir"] != 0].copy()
                if sig.empty:
                    continue
                dir_hit = float((sig["signal_dir"] == sig["signal_dir_truth"]).mean())
                for series, price in price_frames.items():
                    for entry_offset in ENTRY_OFFSETS:
                        bt = backtest_one_series(
                            price,
                            series,
                            sig[["event_date", "signal_dir", "change_bps"]].rename(columns={"event_date": "fomc_date"}),
                            entry_offset,
                        )
                        if bt.empty:
                            continue
                        r = pd.to_numeric(bt["strat_ret"], errors="coerce").dropna()
                        raw = pd.to_numeric(bt["raw_ret"], errors="coerce")
                        sharpe = float((r.mean() / (r.std() + 1e-12)) * np.sqrt(252.0)) if len(r) > 1 else float("nan")
                        ic = float(pd.Series(sig["signal_strength"].to_numpy()[: len(bt)]).corr(raw.reset_index(drop=True), method="spearman")) if len(bt) >= 3 else float("nan")
                        rows.append(
                            {
                                "predictor": predictor,
                                "window": window,
                                "threshold": threshold,
                                "series": series,
                                "entry_offset": entry_offset,
                                "n_trades": int(len(bt)),
                                "direction_hit_rate": dir_hit,
                                "mean_strat_ret": float(r.mean()) if len(r) else float("nan"),
                                "cum_strat_ret": float((1.0 + r).prod() - 1.0) if len(r) else float("nan"),
                                "sharpe": sharpe,
                                "ic_spearman": ic,
                            }
                        )
    return pd.DataFrame(rows)


def main() -> None:
    # Use strict pre-event selection by default: "can we know the value N days in advance?"
    point = grid_search_point_forecasts(strict_pre=True, min_n=10).sort_values(["target", "RMSE", "MAE", "window"]).reset_index(drop=True)
    point.to_csv(REPORT / "grid_point_forecasts.csv", index=False, encoding="utf-8-sig")
    # For each target + window (N days pre-event), keep the best predictor.
    point_best_by_window = (
        point.sort_values(["target", "window", "RMSE", "MAE"], ascending=[True, True, True, True])
        .groupby(["target", "window"], as_index=False)
        .first()
        .sort_values(["target", "window"])
        .reset_index(drop=True)
    )
    point_best_by_window.to_csv(REPORT / "grid_point_forecasts_best_by_window.csv", index=False, encoding="utf-8-sig")
    point_best = point.sort_values(["target", "RMSE", "MAE"]).groupby("target", as_index=False).first()
    point_best.to_csv(REPORT / "grid_point_forecasts_best.csv", index=False, encoding="utf-8-sig")

    event_level = export_best_event_level_predictions(point_best, strict_pre=True)
    if not event_level.empty:
        event_level.to_csv(REPORT / "best_event_level_predictions.csv", index=False, encoding="utf-8-sig")
        # Convenience extract: "future" here means after the last trade_date in the dataset (not wall-clock today).
        cut = pd.to_datetime(event_level["trade_date"], errors="coerce").max()
        fut = event_level[pd.to_datetime(event_level["event_date"], errors="coerce") > cut].copy() if pd.notna(cut) else pd.DataFrame()
        if not fut.empty:
            fut = fut.sort_values(["event_date", "target"]).reset_index(drop=True)
            fut.to_csv(REPORT / "future_targets_snapshot.csv", index=False, encoding="utf-8-sig")

    # Selective prediction grid search: allow rejecting up to 10% lowest-confidence cases
    # to improve MAE while pushing the forecast horizon (window) earlier.
    selective = grid_search_selective_point_forecasts(strict_pre=True, min_n=10, max_reject=0.10, include_qr=False)
    if not selective.empty:
        selective = selective.sort_values(["target", "window", "reject_rate", "MAE"], ascending=[True, False, True, True]).reset_index(drop=True)
        selective.to_csv(REPORT / "grid_point_forecasts_selective.csv", index=False, encoding="utf-8-sig")

        # Feasible subset: reject<=10% and coverage>=90%
        feasible = selective[(selective["coverage"] >= SELECTIVE_MIN_COVERAGE) & (selective["reject_rate"] <= 0.10)].copy()
        if not feasible.empty:
            feasible.to_csv(REPORT / "grid_point_forecasts_selective_feasible.csv", index=False, encoding="utf-8-sig")

            # For each (target, window): best MAE.
            sel_best_by_window = (
                feasible.sort_values(["target", "window", "MAE", "RMSE"], ascending=[True, False, True, True])
                .groupby(["target", "window"], as_index=False)
                .first()
                .sort_values(["target", "window"], ascending=[True, True])
                .reset_index(drop=True)
            )
            sel_best_by_window.to_csv(REPORT / "grid_point_forecasts_selective_best_by_window.csv", index=False, encoding="utf-8-sig")

            # Pareto frontier over windows for each target.
            pareto = _pareto_frontier_by_window(sel_best_by_window)
            if not pareto.empty:
                pareto.to_csv(REPORT / "grid_point_forecasts_selective_pareto.csv", index=False, encoding="utf-8-sig")

            # Recommended config A: minimize MAE (tie-break: larger window).
            sel_best_min_mae = (
                feasible.sort_values(["target", "MAE", "RMSE", "window"], ascending=[True, True, True, False])
                .groupby("target", as_index=False)
                .first()
                .reset_index(drop=True)
            )
            sel_best_min_mae.to_csv(REPORT / "grid_point_forecasts_selective_best_min_mae.csv", index=False, encoding="utf-8-sig")

            # Recommended config B: push window as far as possible while staying near the best MAE.
            rows: list[pd.DataFrame] = []
            for target, grp in feasible.groupby("target", sort=False):
                mae_min = float(pd.to_numeric(grp["MAE"], errors="coerce").min())
                if not np.isfinite(mae_min):
                    continue
                cap = mae_min * (1.0 + float(SELECTIVE_EPS_FRAC))
                cand = grp[pd.to_numeric(grp["MAE"], errors="coerce") <= cap].copy()
                if cand.empty:
                    cand = grp.copy()
                best = cand.sort_values(["window", "MAE", "RMSE"], ascending=[False, True, True]).head(1)
                rows.append(best)
            sel_best_tradeoff = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            if not sel_best_tradeoff.empty:
                sel_best_tradeoff.to_csv(REPORT / "grid_point_forecasts_selective_best_tradeoff.csv", index=False, encoding="utf-8-sig")

                event_level_sel = export_best_event_level_predictions_selective(sel_best_tradeoff, strict_pre=True)
                if not event_level_sel.empty:
                    event_level_sel.to_csv(REPORT / "best_event_level_predictions_selective.csv", index=False, encoding="utf-8-sig")

    # Extended selective grid: add MAE-oriented median regression (QuantileRegressor @ q=0.5) as point model.
    selective_qr = grid_search_selective_point_forecasts(strict_pre=True, min_n=10, max_reject=0.10, include_qr=True)
    if not selective_qr.empty and "point_model" in selective_qr.columns:
        selective_qr = selective_qr.sort_values(["target", "window", "reject_rate", "MAE"], ascending=[True, False, True, True]).reset_index(drop=True)
        selective_qr.to_csv(REPORT / "grid_point_forecasts_selective_with_qr.csv", index=False, encoding="utf-8-sig")

        feasible_qr = selective_qr[(selective_qr["coverage"] >= SELECTIVE_MIN_COVERAGE) & (selective_qr["reject_rate"] <= 0.10)].copy()
        if not feasible_qr.empty:
            feasible_qr.to_csv(REPORT / "grid_point_forecasts_selective_with_qr_feasible.csv", index=False, encoding="utf-8-sig")
            best_by_window_qr = (
                feasible_qr.sort_values(["target", "window", "MAE", "RMSE"], ascending=[True, False, True, True])
                .groupby(["target", "window"], as_index=False)
                .first()
                .sort_values(["target", "window"], ascending=[True, True])
                .reset_index(drop=True)
            )
            best_by_window_qr.to_csv(REPORT / "grid_point_forecasts_selective_with_qr_best_by_window_feasible.csv", index=False, encoding="utf-8-sig")

            # Trade-off: maximize window while staying within eps of the best MAE per target.
            rows2: list[pd.DataFrame] = []
            for target, grp in feasible_qr.groupby("target", sort=False):
                mae_min = float(pd.to_numeric(grp["MAE"], errors="coerce").min())
                if not np.isfinite(mae_min):
                    continue
                cap = mae_min * (1.0 + float(SELECTIVE_EPS_FRAC))
                cand = grp[pd.to_numeric(grp["MAE"], errors="coerce") <= cap].copy()
                if cand.empty:
                    cand = grp.copy()
                best = cand.sort_values(["window", "MAE", "RMSE"], ascending=[False, True, True]).head(1)
                rows2.append(best)
            best_tradeoff_qr = pd.concat(rows2, ignore_index=True) if rows2 else pd.DataFrame()
            if not best_tradeoff_qr.empty:
                best_tradeoff_qr.to_csv(REPORT / "grid_point_forecasts_selective_with_qr_best_tradeoff_feasible.csv", index=False, encoding="utf-8-sig")
                event_level_sel_qr = export_best_event_level_predictions_selective(best_tradeoff_qr, strict_pre=True)
                if not event_level_sel_qr.empty:
                    event_level_sel_qr.to_csv(REPORT / "best_event_level_predictions_selective_with_qr_best_tradeoff_feasible.csv", index=False, encoding="utf-8-sig")

    timing = grid_search_timing().sort_values(["series", "cum_strat_ret", "direction_hit_rate"], ascending=[True, False, False]).reset_index(drop=True)
    timing.to_csv(REPORT / "grid_timing.csv", index=False, encoding="utf-8-sig")
    timing_best = timing.sort_values(["series", "cum_strat_ret", "direction_hit_rate"], ascending=[True, False, False]).groupby("series", as_index=False).first()
    timing_best.to_csv(REPORT / "grid_timing_best.csv", index=False, encoding="utf-8-sig")

    print("Saved ->", REPORT / "grid_point_forecasts.csv")
    print("Saved ->", REPORT / "grid_point_forecasts_best_by_window.csv")
    print("Saved ->", REPORT / "grid_point_forecasts_best.csv")
    if not event_level.empty:
        print("Saved ->", REPORT / "best_event_level_predictions.csv")
    if not selective.empty:
        print("Saved ->", REPORT / "grid_point_forecasts_selective.csv")
        if "feasible" in locals() and not feasible.empty:
            print("Saved ->", REPORT / "grid_point_forecasts_selective_feasible.csv")
            print("Saved ->", REPORT / "grid_point_forecasts_selective_best_by_window.csv")
            if "pareto" in locals() and not pareto.empty:
                print("Saved ->", REPORT / "grid_point_forecasts_selective_pareto.csv")
            print("Saved ->", REPORT / "grid_point_forecasts_selective_best_min_mae.csv")
            if "sel_best_tradeoff" in locals() and not sel_best_tradeoff.empty:
                print("Saved ->", REPORT / "grid_point_forecasts_selective_best_tradeoff.csv")
                if "event_level_sel" in locals() and not event_level_sel.empty:
                    print("Saved ->", REPORT / "best_event_level_predictions_selective.csv")
    print("Saved ->", REPORT / "grid_timing.csv")
    print("Saved ->", REPORT / "grid_timing_best.csv")
    print("\nBest point forecast configs")
    print(point_best.to_string(index=False))
    if not selective.empty and "sel_best_tradeoff" in locals() and not sel_best_tradeoff.empty:
        print(
            f"\nSelective best tradeoff (coverage>={SELECTIVE_MIN_COVERAGE:.2f}, reject<=0.10, window as large as possible within {SELECTIVE_EPS_FRAC:.0%} of best MAE)"
        )
        print(sel_best_tradeoff.to_string(index=False))
    print("\nBest timing configs")
    print(timing_best.to_string(index=False))


if __name__ == "__main__":
    main()
