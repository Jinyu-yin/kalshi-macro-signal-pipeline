from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "output" / "report" / "paper_bundle"
TAB = BUNDLE / "tables"
FIG = BUNDLE / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def plot_ic_hitrate_panel(df: pd.DataFrame) -> None:
    d = df.copy()
    d = d[d["source"].isin(["quickscan", "ffr_eval"])].copy()
    d = d[pd.to_numeric(d["n_obs"], errors="coerce") >= 8]
    d["ic_spearman"] = pd.to_numeric(d["ic_spearman"], errors="coerce")
    d["win_rate"] = pd.to_numeric(d["win_rate"], errors="coerce")

    top_ic = d.sort_values("ic_spearman", ascending=False).head(12)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(range(len(top_ic)), top_ic["ic_spearman"].fillna(0.0))
    axes[0].set_yticks(range(len(top_ic)))
    axes[0].set_yticklabels((top_ic["factor"].astype(str) + "-" + top_ic["feature"].astype(str)).tolist(), fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title("Top IC (Spearman)")
    axes[0].grid(alpha=0.3)

    top_wr = d.dropna(subset=["win_rate"]).sort_values("win_rate", ascending=False).head(12)
    if not top_wr.empty:
        axes[1].barh(range(len(top_wr)), top_wr["win_rate"].fillna(0.0), color="tab:orange")
        axes[1].set_yticks(range(len(top_wr)))
        axes[1].set_yticklabels((top_wr["factor"].astype(str) + "-" + top_wr["feature"].astype(str)).tolist(), fontsize=8)
        axes[1].invert_yaxis()
    axes[1].set_title("Top Hit Rate")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Manager Dashboard: Predictive Quality")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "manager_dashboard_ic_hitrate.png", dpi=180)
    plt.close(fig)


def plot_candidate_matrix(df: pd.DataFrame) -> None:
    d = df.copy()
    d["ic_spearman"] = pd.to_numeric(d["ic_spearman"], errors="coerce")
    d["sharpe"] = pd.to_numeric(d["sharpe"], errors="coerce")
    d["n_obs"] = pd.to_numeric(d["n_obs"], errors="coerce")
    d = d.dropna(subset=["ic_spearman", "sharpe", "n_obs"])
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(d["ic_spearman"], d["sharpe"], s=d["n_obs"] * 4, alpha=0.7)
    for _, r in d.head(30).iterrows():
        ax.text(r["ic_spearman"], r["sharpe"], f"{r['factor']}-{r['feature']}", fontsize=7)
    ax.set_xlabel("IC (Spearman)")
    ax.set_ylabel("Sharpe (proxy)")
    ax.set_title("Candidate Matrix: IC vs Sharpe")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "manager_dashboard_candidate_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    metrics = _safe(TAB / "unified_metrics_winrate_ic_backtest_sharpe.csv")
    if metrics.empty:
        print("No metrics table found.")
        return
    plot_ic_hitrate_panel(metrics)
    plot_candidate_matrix(metrics)
    print(f"Saved manager dashboard figures to: {FIG}")


if __name__ == "__main__":
    main()
