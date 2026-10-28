from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "out_csv"
REPORT_ROOT = ROOT / "output" / "report" / "paper_bundle"
FIG_DIR = REPORT_ROOT / "figures"
TAB_DIR = REPORT_ROOT / "tables"


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_mapping_table() -> pd.DataFrame:
    rows: List[Dict[str, object]] = [
        {
            "paper_item": "Figure 1 / 14 style forecast errors",
            "source_file": str(OUT_CSV / "kalshi_factor_predictiveness_quickscan.csv"),
            "status": (OUT_CSV / "kalshi_factor_predictiveness_quickscan.csv").exists(),
        },
        {
            "paper_item": "Table 3 style forecast accuracy summary",
            "source_file": str(OUT_CSV / "kalshi_single_factor_evaluation.csv"),
            "status": (OUT_CSV / "kalshi_single_factor_evaluation.csv").exists(),
        },
        {
            "paper_item": "Figure 16/17 style event dynamics",
            "source_file": str(OUT_CSV / "kalshi_ffr_event_study_summary.csv"),
            "status": (OUT_CSV / "kalshi_ffr_event_study_summary.csv").exists(),
        },
        {
            "paper_item": "Factor coverage/quality audit",
            "source_file": str(OUT_CSV / "kalshi_factor_validation_pack.csv"),
            "status": (OUT_CSV / "kalshi_factor_validation_pack.csv").exists(),
        },
    ]
    return pd.DataFrame(rows)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def build_unified_metrics() -> pd.DataFrame:
    quick = _safe_read(OUT_CSV / "kalshi_factor_predictiveness_quickscan.csv")
    eval_df = _safe_read(OUT_CSV / "kalshi_single_factor_evaluation.csv")

    rows: List[Dict[str, object]] = []

    if not quick.empty:
        q = quick.copy()
        q["ic_spearman"] = pd.to_numeric(q.get("spearman"), errors="coerce")
        q["ic_pearson"] = pd.to_numeric(q.get("pearson"), errors="coerce")
        q["n_obs"] = pd.to_numeric(q.get("n_obs"), errors="coerce")
        q["win_rate"] = np.nan
        q["sharpe"] = q["ic_spearman"] / q["mae"].replace(0, np.nan)
        q["backtest_tag"] = np.where((q["n_obs"] >= 10) & (q["ic_spearman"] >= 0.75), "candidate", "watch")

        for _, r in q.iterrows():
            rows.append(
                {
                    "source": "quickscan",
                    "factor": r.get("factor"),
                    "feature": r.get("feature"),
                    "horizon": r.get("horizon_days"),
                    "n_obs": r.get("n_obs"),
                    "win_rate": r.get("win_rate"),
                    "ic_spearman": r.get("ic_spearman"),
                    "ic_pearson": r.get("ic_pearson"),
                    "sharpe": r.get("sharpe"),
                    "backtest_tag": r.get("backtest_tag"),
                }
            )

    if not eval_df.empty:
        e = eval_df.copy()
        e["n_obs"] = pd.to_numeric(e.get("n_obs"), errors="coerce")
        for _, r in e.iterrows():
            rows.append(
                {
                    "source": "ffr_eval",
                    "factor": r.get("factor"),
                    "feature": r.get("feature"),
                    "horizon": r.get("target"),
                    "n_obs": r.get("n_obs"),
                    "win_rate": r.get("directional_hit_rate"),
                    "ic_spearman": r.get("corr_spearman"),
                    "ic_pearson": r.get("corr_pearson"),
                    "sharpe": np.nan,
                    "backtest_tag": "ffr_core",
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["source", "ic_spearman", "ic_pearson"], ascending=[True, False, False]).reset_index(drop=True)


def build_forecast_scorecard(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    d = metrics.copy()
    d["n_obs"] = pd.to_numeric(d["n_obs"], errors="coerce")
    d["ic_spearman"] = pd.to_numeric(d["ic_spearman"], errors="coerce")
    d["win_rate"] = pd.to_numeric(d["win_rate"], errors="coerce")

    d["ic_tstat_approx"] = d["ic_spearman"] * np.sqrt((d["n_obs"] - 2).clip(lower=1) / (1 - d["ic_spearman"].pow(2).clip(upper=0.9999)))
    d["ic_pvalue_approx"] = d["ic_tstat_approx"].abs().apply(lambda t: 2 * (1 - _norm_cdf(float(t))))

    wr = d["win_rate"].dropna()
    if len(wr):
        z = (wr - 0.5) / np.sqrt(0.25 / d.loc[wr.index, "n_obs"].clip(lower=1))
        d.loc[wr.index, "hitrate_pvalue_approx"] = z.abs().apply(lambda t: 2 * (1 - _norm_cdf(float(t))))

    d["adopt_flag"] = (
        (d["n_obs"] >= 12)
        & (d["ic_spearman"] >= 0.75)
        & (d["ic_pvalue_approx"] <= 0.10)
    )

    cols = [
        "source", "factor", "feature", "horizon", "n_obs", "ic_spearman", "ic_pearson",
        "ic_tstat_approx", "ic_pvalue_approx", "win_rate", "hitrate_pvalue_approx", "backtest_tag", "adopt_flag",
    ]
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    return d[cols].sort_values(["adopt_flag", "ic_spearman"], ascending=[False, False])


def build_strategy_scorecard(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    d = metrics.copy()
    d["n_obs"] = pd.to_numeric(d["n_obs"], errors="coerce")
    d["ic_spearman"] = pd.to_numeric(d["ic_spearman"], errors="coerce")
    d["sharpe"] = pd.to_numeric(d["sharpe"], errors="coerce")

    d["strategy_grade"] = np.select(
        [
            (d["n_obs"] >= 12) & (d["ic_spearman"] >= 0.80) & (d["sharpe"] >= 15),
            (d["n_obs"] >= 10) & (d["ic_spearman"] >= 0.70) & (d["sharpe"] >= 8),
        ],
        ["A", "B"],
        default="C",
    )

    cols = ["source", "factor", "feature", "horizon", "n_obs", "ic_spearman", "sharpe", "backtest_tag", "strategy_grade"]
    return d[cols].sort_values(["strategy_grade", "ic_spearman"], ascending=[True, False])


def build_manager_onepage(forecast: pd.DataFrame, strategy: pd.DataFrame) -> pd.DataFrame:
    if forecast.empty and strategy.empty:
        return pd.DataFrame()

    key = ["source", "factor", "feature", "horizon"]
    if forecast.empty:
        merged = strategy.copy()
    elif strategy.empty:
        merged = forecast.copy()
    else:
        merged = forecast.merge(strategy[key + ["sharpe", "strategy_grade"]], on=key, how="left")

    if "adopt_flag" not in merged.columns:
        merged["adopt_flag"] = False
    if "strategy_grade" not in merged.columns:
        merged["strategy_grade"] = np.nan

    merged["headline_status"] = np.where(
        (merged["adopt_flag"] == True) & (merged["strategy_grade"].isin(["A", "B"]) | merged["strategy_grade"].isna()),
        "ADOPT",
        "WATCH",
    )
    return merged.sort_values(["headline_status", "ic_spearman"], ascending=[True, False])


def build_qa_checklist(mapping: pd.DataFrame, metrics: pd.DataFrame, forecast: pd.DataFrame, manager: pd.DataFrame) -> pd.DataFrame:
    checks: List[Dict[str, object]] = []
    checks.append({"check": "paper_mapping_coverage", "passed": bool(mapping["status"].all()) if not mapping.empty else False, "detail": f"available={int(mapping['status'].sum())}/{len(mapping)}" if not mapping.empty else "mapping empty"})
    checks.append({"check": "metrics_nonempty", "passed": not metrics.empty, "detail": f"rows={len(metrics)}"})
    checks.append({"check": "forecast_scorecard_nonempty", "passed": not forecast.empty, "detail": f"rows={len(forecast)}"})
    checks.append({"check": "manager_onepage_nonempty", "passed": not manager.empty, "detail": f"rows={len(manager)}"})
    if not manager.empty:
        adopt_n = int((manager["headline_status"] == "ADOPT").sum())
        checks.append({"check": "adopt_count_positive", "passed": adopt_n > 0, "detail": f"adopt_rows={adopt_n}"})
    return pd.DataFrame(checks)


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    mapping = build_mapping_table()
    metrics = build_unified_metrics()
    forecast = build_forecast_scorecard(metrics)
    strategy = build_strategy_scorecard(metrics)
    manager = build_manager_onepage(forecast, strategy)
    qa = build_qa_checklist(mapping, metrics, forecast, manager)

    mapping.to_csv(TAB_DIR / "paper_mapping_status.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(TAB_DIR / "unified_metrics_winrate_ic_backtest_sharpe.csv", index=False, encoding="utf-8-sig")
    forecast.to_csv(TAB_DIR / "research_forecast_scorecard.csv", index=False, encoding="utf-8-sig")
    strategy.to_csv(TAB_DIR / "strategy_backtest_scorecard.csv", index=False, encoding="utf-8-sig")
    manager.to_csv(TAB_DIR / "manager_onepage_scorecard.csv", index=False, encoding="utf-8-sig")
    qa.to_csv(TAB_DIR / "qa_checklist.csv", index=False, encoding="utf-8-sig")

    index_rows = [
        {"artifact": "paper_mapping_status", "path": str(TAB_DIR / "paper_mapping_status.csv")},
        {"artifact": "unified_metrics", "path": str(TAB_DIR / "unified_metrics_winrate_ic_backtest_sharpe.csv")},
        {"artifact": "research_forecast_scorecard", "path": str(TAB_DIR / "research_forecast_scorecard.csv")},
        {"artifact": "strategy_backtest_scorecard", "path": str(TAB_DIR / "strategy_backtest_scorecard.csv")},
        {"artifact": "manager_onepage_scorecard", "path": str(TAB_DIR / "manager_onepage_scorecard.csv")},
        {"artifact": "qa_checklist", "path": str(TAB_DIR / "qa_checklist.csv")},
    ]
    pd.DataFrame(index_rows).to_csv(REPORT_ROOT / "report_index.csv", index=False, encoding="utf-8-sig")

    print(f"Saved bundle tables to: {TAB_DIR}")
    print(f"Saved index to: {REPORT_ROOT / 'report_index.csv'}")


if __name__ == "__main__":
    main()
