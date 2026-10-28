from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

# ====================== 配置 ======================
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"
REPORT = ROOT / "output" / "report" / "mode_final"
REPORT.mkdir(parents=True, exist_ok=True)

WINDOWS = [1, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50]
POINT_TOLERANCES = [0.10, 0.15, 0.20]
TOPN_VALUES = [1, 2, 3, 5, 10]

PRED_FILES = {
    "fomc": OUT / "kalshi_modes_rebuilt_fomc_daily.csv",
    "cpi": OUT / "kalshi_modes_rebuilt_cpi_daily.csv",
    "u3": OUT / "kalshi_modes_rebuilt_u3_daily.csv",
    "pcecore": OUT / "kalshi_modes_rebuilt_pcecore_daily.csv",
}

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

# ====================== 过滤 SHELTER ======================
def is_shelter(ticker):
    return re.search(r"SHELTER", str(ticker).upper()) is not None

# ====================== 真实值加载 ======================
def load_fomc_realized():
    df = pd.read_csv(OUT / "fomc_realized_decisions.csv")
    df["event_date"] = pd.to_datetime(df["fomc_date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    return df.dropna(subset=["event_date", "ffr_upper"])

def load_macro_realized(series_id, mom=False):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else "observation_date"
    value_col = [c for c in df.columns if c != date_col][0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["event_period"] = df[date_col].dt.to_period("M").astype(str)
    df["realized"] = pd.to_numeric(df[value_col].replace(".", np.nan), errors="coerce")
    if mom:
        df["realized"] = (df["realized"] / df["realized"].shift(1) - 1) * 100
    return df.dropna(subset=["event_period", "realized"])

# ====================== 预测数据加载 ======================
def load_prediction(path):
    df = pd.read_csv(path)
    df["trade_ts"] = pd.to_datetime(df["trade_ts"], utc=True, errors="coerce")
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["trade_ts", "event_ts", "mode_interval_low", "mode_interval_high"])
    df["event_date"] = df["event_ts"].dt.tz_convert(None).dt.normalize()
    df["event_period"] = df["event_ticker"].apply(lambda x: None if pd.isna(x) else
        f"20{x[-5:-3]}-{MONTHS[x[-2:]]}" if len(x) >= 5 and x[-2:] in MONTHS else None)
    return df

# ====================== ✅ 严格取事件前 N 天（你要的逻辑） ======================
def get_window_pred(pred, window_days):
    results = []
    for ticker, group in pred.groupby("event_ticker"):
        g = group.sort_values("trade_ts")
        event_ts = g["event_ts"].iloc[0]
        cutoff_ts = event_ts - pd.Timedelta(days=window_days)

        # 【严格按你的要求】取事件前N天之前最后一条
        valid = g[g["trade_ts"] <= cutoff_ts]
        if valid.empty:
            continue
        results.append(valid.iloc[-1])
    return pd.DataFrame(results)

def parse_interval_distribution(value):
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    out = []
    for item in parsed if isinstance(parsed, list) else []:
        try:
            low = float(item["low"])
            high = float(item["high"])
            mass = max(float(item["mass"]), 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(low) or not np.isfinite(high) or mass <= 0:
            continue
        out.append({"low": low, "high": high, "mass": mass, "mid": (low + high) / 2.0})
    return out


def topn_prediction(value, topn):
    dist = parse_interval_distribution(value)
    if not dist:
        return np.nan
    selected = sorted(dist, key=lambda x: x["mass"], reverse=True)[:topn]
    mass_sum = sum(x["mass"] for x in selected)
    if mass_sum <= 0:
        return np.nan
    return sum(x["mid"] * x["mass"] for x in selected) / mass_sum


# ====================== SPAN 区间评估（正确、不暴跌） ======================
def evaluate_span(target, pred, realized):
    all_rows = []
    for w in WINDOWS:
        snap = get_window_pred(pred, w)
        if snap.empty:
            all_rows.append({"target": target, "window": w, "n": 0, "span_hit_rate": np.nan})
            continue

        # 合并真实值
        if target == "fomc":
            merged = snap.merge(realized, on="event_date", how="inner")
            y = merged["ffr_upper"]
        else:
            merged = snap.merge(realized, on="event_period", how="inner")
            y = merged["realized"]

        if merged.empty:
            all_rows.append({"target": target, "window": w, "n": 0, "span_hit_rate": np.nan})
            continue

        # 区间 SPAN 命中
        L = merged["mode_interval_low"].astype(float)
        R = merged["mode_interval_high"].astype(float)
        hit = (y >= L) & (y <= R)
        span_hit_rate = hit.mean()
        n = len(merged)

        # 容错胜率
        mid = (L + R) / 2
        abs_error = abs(mid - y)
        row = {"target": target, "window": w, "n": n, "span_hit_rate": span_hit_rate}
        for tol in POINT_TOLERANCES:
            row[f"win_±{tol}"] = (abs_error <= tol).mean()
        all_rows.append(row)
    return pd.DataFrame(all_rows)


def evaluate_topn_matrix(target, pred, realized):
    rows = []
    for w in WINDOWS:
        snap = get_window_pred(pred, w)
        if snap.empty:
            continue

        if target == "fomc":
            merged = snap.merge(realized, on="event_date", how="inner")
            y = pd.to_numeric(merged["ffr_upper"], errors="coerce")
        else:
            merged = snap.merge(realized, on="event_period", how="inner")
            y = pd.to_numeric(merged["realized"], errors="coerce")

        if merged.empty or "interval_distribution" not in merged.columns:
            continue

        for topn in TOPN_VALUES:
            pred_point = merged["interval_distribution"].apply(lambda x: topn_prediction(x, topn))
            error = pred_point - y
            valid = error.dropna()
            if valid.empty:
                continue
            row = {
                "target": target,
                "window": w,
                "topn": topn,
                "n": int(valid.shape[0]),
                "MAE": float(valid.abs().mean()),
                "RMSE": float(np.sqrt((valid ** 2).mean())),
            }
            for tol in POINT_TOLERANCES:
                row[f"win_pm{str(tol).replace('.', 'p')}"] = float((valid.abs() <= tol).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def save_topn_matrices(topn_scores):
    if topn_scores.empty:
        return
    topn_scores.to_csv(REPORT / "topn_point_forecast_long.csv", index=False, encoding="utf-8-sig")
    metrics = ["MAE", "RMSE"] + [f"win_pm{str(tol).replace('.', 'p')}" for tol in POINT_TOLERANCES]
    for target, group in topn_scores.groupby("target"):
        for metric in metrics:
            matrix = group.pivot(index="window", columns="topn", values=metric).sort_index()
            matrix.columns = [f"top{int(c)}" for c in matrix.columns]
            matrix.to_csv(REPORT / f"topn_matrix_{target}_{metric}.csv", encoding="utf-8-sig")


# ====================== 主函数 ======================
def main():
    # 真实数据
    fomc_real = load_fomc_realized()
    cpi_real = load_macro_realized("CPIAUCSL", mom=True)
    u3_real = load_macro_realized("UNRATE")
    pce_real = load_macro_realized("PCEPILFE", mom=True)

    # 预测数据
    fomc_pred = load_prediction(PRED_FILES["fomc"])
    cpi_pred = load_prediction(PRED_FILES["cpi"])
    u3_pred = load_prediction(PRED_FILES["u3"])
    pce_pred = load_prediction(PRED_FILES["pcecore"])

    # 筛选 shelter
    cpi_no_shelter = cpi_pred[~cpi_pred["event_ticker"].apply(is_shelter)].copy()
    cpi_only_shelter = cpi_pred[cpi_pred["event_ticker"].apply(is_shelter)].copy()

    # 评估
    res = []
    res.append(evaluate_span("fomc", fomc_pred, fomc_real))
    res.append(evaluate_span("cpi_no_shelter", cpi_no_shelter, cpi_real))
    res.append(evaluate_span("cpi_shelter", cpi_only_shelter, cpi_real))
    res.append(evaluate_span("u3", u3_pred, u3_real))
    res.append(evaluate_span("pcecore", pce_pred, pce_real))

    final = pd.concat(res).sort_values(["target", "window"])
    final.to_csv(REPORT / "final_span_metrics.csv", index=False, encoding="utf-8-sig")

    topn_res = []
    topn_res.append(evaluate_topn_matrix("fomc", fomc_pred, fomc_real))
    topn_res.append(evaluate_topn_matrix("cpi_no_shelter", cpi_no_shelter, cpi_real))
    topn_res.append(evaluate_topn_matrix("cpi_shelter", cpi_only_shelter, cpi_real))
    topn_res.append(evaluate_topn_matrix("u3", u3_pred, u3_real))
    topn_res.append(evaluate_topn_matrix("pcecore", pce_pred, pce_real))
    topn_scores = pd.concat([x for x in topn_res if not x.empty], ignore_index=True) if any(not x.empty for x in topn_res) else pd.DataFrame()
    save_topn_matrices(topn_scores)

    print("✅ 运行完成！严格前N天 + SPAN区间 + 筛shelter + TopN矩阵")
    print(final.to_string(index=False))
    if not topn_scores.empty:
        print("\nTopN point forecast scores")
        print(topn_scores.to_string(index=False))

if __name__ == "__main__":
    main()