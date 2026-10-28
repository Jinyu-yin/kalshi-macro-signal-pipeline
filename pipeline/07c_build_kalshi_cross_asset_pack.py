from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"
REPORT = ROOT / "output" / "report" / "paper_bundle"
TAB = REPORT / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_mapping() -> pd.DataFrame:
    rows = [
        {"kalshi_theme": "fed_funds_path", "factor_group": "ffr", "asset": "ZT", "rationale": "front-end rate sensitivity"},
        {"kalshi_theme": "fed_funds_path", "factor_group": "ffr", "asset": "ZN", "rationale": "curve-duration transmission"},
        {"kalshi_theme": "inflation_surprise", "factor_group": "cpi", "asset": "GC", "rationale": "inflation hedge / real-rate sensitivity"},
        {"kalshi_theme": "inflation_surprise", "factor_group": "cpi", "asset": "UUP", "rationale": "usd inflation-policy channel"},
        {"kalshi_theme": "growth_labor", "factor_group": "u3", "asset": "SPY", "rationale": "growth risk appetite"},
        {"kalshi_theme": "inflation_growth_mix", "factor_group": "pcecore", "asset": "USO", "rationale": "commodity inflation-growth linkage"},
    ]
    return pd.DataFrame(rows)


def build_ic_matrix(quickscan: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if quickscan.empty:
        return pd.DataFrame()
    q = quickscan.copy()
    q["spearman"] = pd.to_numeric(q.get("spearman"), errors="coerce")
    q["pearson"] = pd.to_numeric(q.get("pearson"), errors="coerce")
    q["n_obs"] = pd.to_numeric(q.get("n_obs"), errors="coerce")
    q = q.merge(mapping[["factor_group", "asset"]], left_on="factor", right_on="factor_group", how="inner")
    if q.empty:
        return pd.DataFrame()
    g = (
        q.groupby(["asset", "factor", "feature"], dropna=False)
        .agg(ic_spearman=("spearman", "mean"), ic_pearson=("pearson", "mean"), n_obs=("n_obs", "mean"))
        .reset_index()
        .sort_values(["asset", "ic_spearman"], ascending=[True, False])
    )
    return g


def build_backtest_table(ic_matrix: pd.DataFrame) -> pd.DataFrame:
    if ic_matrix.empty:
        return pd.DataFrame()
    d = ic_matrix.copy()
    # conservative proxy score until full realized-asset return backtest is attached
    d["hit_rate_proxy"] = (0.5 + 0.35 * d["ic_spearman"].clip(-1, 1)).clip(0, 1)
    d["sharpe_proxy"] = d["ic_spearman"] / 0.06
    d["status"] = np.where((d["ic_spearman"] >= 0.75) & (d["n_obs"] >= 10), "adopt", np.where(d["ic_spearman"] >= 0.55, "watch", "reject"))
    return d[["asset", "factor", "feature", "n_obs", "ic_spearman", "ic_pearson", "hit_rate_proxy", "sharpe_proxy", "status"]]


def main() -> None:
    quickscan = _safe_csv(OUT / "kalshi_factor_predictiveness_quickscan.csv")

    mapping = build_mapping()
    ic_matrix = build_ic_matrix(quickscan, mapping)
    backtest = build_backtest_table(ic_matrix)

    mapping.to_csv(TAB / "asset_mapping_kalshi_macro.csv", index=False, encoding="utf-8-sig")
    ic_matrix.to_csv(TAB / "kalshi_factor_asset_ic_matrix.csv", index=False, encoding="utf-8-sig")
    backtest.to_csv(TAB / "kalshi_factor_asset_backtest.csv", index=False, encoding="utf-8-sig")

    print(f"Saved: {TAB / 'asset_mapping_kalshi_macro.csv'}")
    print(f"Saved: {TAB / 'kalshi_factor_asset_ic_matrix.csv'}")
    print(f"Saved: {TAB / 'kalshi_factor_asset_backtest.csv'}")


if __name__ == "__main__":
    main()
