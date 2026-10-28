from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "output" / "report" / "yield_proxy_timing_tdd"
REPORT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    p = REPORT / "yield_proxy_trades_all.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing trades file: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise RuntimeError("No trades to plot")

    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df = df.dropna(subset=["entry_date"]).sort_values(["series", "entry_date", "entry_offset"])
    df["cum_nav"] = df.groupby(["series", "entry_offset"]) ["strategy_ret_pct"].transform(lambda s: (1.0 + s.fillna(0.0)).cumprod())

    fig, ax = plt.subplots(figsize=(12, 6))
    for (series, off), g in df.groupby(["series", "entry_offset"]):
        ax.plot(g["entry_date"], g["cum_nav"], linewidth=1.8, label=f"{series} {off}d")
    ax.set_title("Yield Proxy Timing Strategy Equity Curves")
    ax.set_xlabel("Entry date")
    ax.set_ylabel("Cumulative NAV")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT / "yield_proxy_equity_curve_all.png", dpi=180)
    plt.close(fig)
    print(f"Saved -> {REPORT / 'yield_proxy_equity_curve_all.png'}")


if __name__ == "__main__":
    main()
