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
    df = df.dropna(subset=["entry_date"]).sort_values(["series", "entry_offset", "entry_date"])
    df["strategy_ret_pct"] = pd.to_numeric(df["strategy_ret_pct"], errors="coerce").fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    for (series, off), g in df.groupby(["series", "entry_offset"]):
        nav = (1.0 + g["strategy_ret_pct"]).cumprod()
        ax.plot(g["entry_date"], nav, linewidth=1.8, label=f"{series}-{off}d")

    ax.set_title("Cumulative Return Curves by Series and Entry Offset")
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
