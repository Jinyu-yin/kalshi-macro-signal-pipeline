from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "output" / "report" / "zt_timing"
REPORT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    p = REPORT / "zt_timing_trades.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing trades file: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise RuntimeError("No trades to plot")

    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df = df.dropna(subset=["entry_date"]).sort_values(["entry_date", "entry_offset"])
    df["cum_nav"] = (1.0 + pd.to_numeric(df["strat_ret"], errors="coerce").fillna(0.0)).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["entry_date"], df["cum_nav"], linewidth=2, label="Strategy NAV")

    colors = {1: "tab:green", -1: "tab:red"}
    for _, r in df.iterrows():
        c = colors.get(int(r["signal_dir"]), "tab:gray")
        ax.scatter(r["entry_date"], r["cum_nav"], color=c, s=35, zorder=3)
        ax.annotate(
            f"{int(r['entry_offset'])}d",
            (r["entry_date"], r["cum_nav"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
            color=c,
        )

    ax.set_title("ZT Timing Strategy Equity Curve")
    ax.set_xlabel("Entry date")
    ax.set_ylabel("Cumulative NAV")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT / "zt_timing_equity_curve.png", dpi=180)
    plt.close(fig)

    # Also print a compact label map for interpretation.
    print(f"Saved -> {REPORT / 'zt_timing_equity_curve.png'}")


if __name__ == "__main__":
    main()
