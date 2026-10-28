from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "output" / "report" / "yield_proxy_timing"
REPORT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    p = REPORT / "yield_proxy_trades.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing trades file: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise RuntimeError("No yield proxy trades to plot")

    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df = df.dropna(subset=["entry_date"]).sort_values(["series", "entry_date", "entry_offset"])
    df["nav"] = (1.0 + pd.to_numeric(df["strat_ret"], errors="coerce").fillna(0.0) / 100.0).groupby(df["series"]).cumprod()

    for series, g in df.groupby("series"):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(g["entry_date"], g["nav"], linewidth=2, label=f"{series} Strategy NAV")
        for _, r in g.iterrows():
            color = "tab:green" if int(r["signal_dir"]) > 0 else "tab:red"
            ax.scatter(r["entry_date"], r["nav"], color=color, s=28, zorder=3)
            ax.annotate(f"{int(r['entry_offset'])}d", (r["entry_date"], r["nav"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color=color)
        ax.set_title(f"{series} Timing Strategy Equity Curve")
        ax.set_xlabel("Entry date")
        ax.set_ylabel("Cumulative NAV")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(REPORT / f"{series.lower()}_equity_curve.png", dpi=180)
        plt.close(fig)
        print(f"Saved -> {REPORT / f'{series.lower()}_equity_curve.png'}")


if __name__ == "__main__":
    main()
