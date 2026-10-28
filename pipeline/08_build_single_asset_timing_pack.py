from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"
REPORT = ROOT / "output" / "report" / "single_asset_timing"
TAB = REPORT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

# First implementation: ZT as the primary route-ready asset.
PRIMARY_ASSET = "ZT"


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_signal_spec() -> pd.DataFrame:
    rows = [
        {
            "theme": "ffr",
            "signal": "event_eve_mode",
            "asset": "ZT",
            "direction": "short if mode/mean implies higher future rates, long if lower",
            "entry": "event_eve",
            "exit": "1d/3d/5d after event",
            "rationale": "front-end Treasury is most sensitive to FFR repricing",
        },
        {
            "theme": "cpi",
            "signal": "event_eve_mean_mode",
            "asset": "ZT",
            "direction": "short on hotter inflation, long on cooler inflation",
            "entry": "event_eve",
            "exit": "1d/3d/5d after event",
            "rationale": "CPI surprise reprices Fed path and front-end yields",
        },
        {
            "theme": "u3",
            "signal": "event_eve_mean_mode",
            "asset": "ZT",
            "direction": "long on weaker labor / lower growth, short on stronger labor",
            "entry": "event_eve",
            "exit": "1d/3d/5d after event",
            "rationale": "labor surprises affect policy expectations, but more weakly than CPI",
        },
    ]
    return pd.DataFrame(rows)


def build_timing_backtest() -> pd.DataFrame:
    # Bridge from existing Kalshi evaluation to a route-ready first-pass table.
    ffr_eval = _safe_csv(OUT / "kalshi_single_factor_evaluation.csv")
    quick = _safe_csv(OUT / "kalshi_factor_predictiveness_quickscan.csv")
    rows = []

    if not ffr_eval.empty:
        sub = ffr_eval[ffr_eval["factor"] == "ffr"].copy()
        for _, r in sub.iterrows():
            rows.append(
                {
                    "asset": PRIMARY_ASSET,
                    "theme": "ffr",
                    "feature": r.get("feature"),
                    "target": r.get("target"),
                    "n_obs": r.get("n_obs"),
                    "ic_spearman": r.get("corr_spearman"),
                    "ic_pearson": r.get("corr_pearson"),
                    "hit_rate": r.get("directional_hit_rate"),
                    "sharpe": np.nan,
                    "max_drawdown": np.nan,
                    "hold": "event_eve -> event+1d",
                    "status": "research",
                }
            )

    # Add short-form CPI/U3 readiness markers based on current quickscan only.
    if not quick.empty:
        q = quick[quick["factor"].isin(["cpi", "u3"]) & quick["feature"].isin(["mean", "median", "mode"])].copy()
        for _, r in q.iterrows():
            rows.append(
                {
                    "asset": PRIMARY_ASSET,
                    "theme": r.get("factor"),
                    "feature": r.get("feature"),
                    "target": f"{r.get('horizon_days')}d_forward_return_proxy",
                    "n_obs": r.get("n_obs"),
                    "ic_spearman": r.get("spearman"),
                    "ic_pearson": r.get("pearson"),
                    "hit_rate": np.nan,
                    "sharpe": np.nan,
                    "max_drawdown": np.nan,
                    "hold": f"{int(r.get('horizon_days'))}d",
                    "status": "candidate",
                }
            )

    return pd.DataFrame(rows)


def build_risk_report(backtest: pd.DataFrame) -> pd.DataFrame:
    if backtest.empty:
        return pd.DataFrame()
    d = backtest.copy()
    d["ic_spearman"] = pd.to_numeric(d["ic_spearman"], errors="coerce")
    d["ic_pearson"] = pd.to_numeric(d["ic_pearson"], errors="coerce")
    d["hit_rate"] = pd.to_numeric(d["hit_rate"], errors="coerce")
    d["n_obs"] = pd.to_numeric(d["n_obs"], errors="coerce")
    d["route_ready"] = np.where((d["theme"] == "ffr") | (d["ic_spearman"] >= 0.75), "yes", "watch")
    return d


def main() -> None:
    spec = build_signal_spec()
    backtest = build_timing_backtest()
    risk = build_risk_report(backtest)

    spec.to_csv(TAB / "single_asset_timing_signal_spec.csv", index=False, encoding="utf-8-sig")
    backtest.to_csv(TAB / "single_asset_timing_backtest_scorecard.csv", index=False, encoding="utf-8-sig")
    risk.to_csv(TAB / "single_asset_timing_risk_report.csv", index=False, encoding="utf-8-sig")

    print(f"Saved: {TAB / 'single_asset_timing_signal_spec.csv'}")
    print(f"Saved: {TAB / 'single_asset_timing_backtest_scorecard.csv'}")
    print(f"Saved: {TAB / 'single_asset_timing_risk_report.csv'}")


if __name__ == "__main__":
    main()
