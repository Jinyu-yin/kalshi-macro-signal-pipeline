from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_csv"
FIG = ROOT / "output" / "report" / "paper_style_figures"
FIG.mkdir(parents=True, exist_ok=True)


def plot_factor_error_curves() -> None:
    p = OUT / "kalshi_factor_predictiveness_quickscan.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    keep = df[df["factor"].isin(["cpi", "u3", "pcecore"]) & df["feature"].isin(["mean", "median", "mode"])].copy()
    if keep.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
    for ax, metric in zip(axes, ["mae", "rmse", "spearman"]):
        for (factor, feature), g in keep.groupby(["factor", "feature"]):
            g = g.sort_values("horizon_days")
            ax.plot(g["horizon_days"], g[metric], marker="o", label=f"{factor}-{feature}")
        ax.set_title(metric.upper())
        ax.set_xlabel("Horizon (days)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Value")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Kalshi Macro Forecast Accuracy Curves (Paper-style)")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(FIG / "fig_kalshi_macro_error_curves.png", dpi=180)
    plt.close(fig)


def plot_ffr_event_summary() -> None:
    p = OUT / "kalshi_ffr_event_study_summary.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    sub = df[df["feature"].isin(["mean_last", "median_last", "mode_last", "variance_last", "std_last"]) & (df["target"] == "change_bps")]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for feature, g in sub.groupby("feature"):
        g = g.sort_values("window_days")
        ax.plot(g["window_days"], g["spearman"], marker="o", label=feature)
    ax.set_title("FFR Event Study: Spearman vs Window (Target: change_bps)")
    ax.set_xlabel("Window days before FOMC")
    ax.set_ylabel("Spearman")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ffr_event_spearman_by_window.png", dpi=180)
    plt.close(fig)


def plot_factor_heatmap_table() -> None:
    p = OUT / "kalshi_factor_predictiveness_quickscan.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    sub = df[df["feature"].isin(["mean", "median", "mode"])].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(index=["factor", "feature"], columns="horizon_days", values="spearman", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{i[0]}-{i[1]}" for i in pivot.index], fontsize=8)
    ax.set_title("Kalshi Factor Predictiveness Heatmap (Spearman)")
    ax.set_xlabel("Horizon days")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_kalshi_factor_spearman_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_factor_error_curves()
    plot_ffr_event_summary()
    plot_factor_heatmap_table()
    print(f"Saved paper-style figures to: {FIG}")


if __name__ == "__main__":
    main()
