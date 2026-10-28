from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from pipeline._yield_proxy_timing import ENTRY_OFFSETS, HOLD_DAYS, ROOT, YIELD_FILES, backtest_one_series, load_events, load_yield

OUT = ROOT / "out_csv"
REPORT = ROOT / "output" / "report" / "yield_proxy_timing"
REPORT.mkdir(parents=True, exist_ok=True)


def _summary(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {"n_trades": 0, "hit_rate": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "ic": np.nan}
    r = bt["strat_ret"].astype(float) / 100.0
    nav = (1.0 + r).cumprod()
    hit_rate = float((r > 0).mean())
    mu = float(r.mean())
    sd = float(r.std())
    sharpe = (mu / (sd + 1e-12)) * np.sqrt(252.0) if sd > 0 else np.nan
    mdd = float((nav / nav.cummax() - 1.0).min())
    ic = float(bt["signal_dir"].corr(bt["raw_ret"], method="spearman")) if len(bt) >= 3 else np.nan
    return {"n_trades": int(len(bt)), "hit_rate": hit_rate, "sharpe": sharpe, "max_drawdown": mdd, "ic": ic}


def main() -> None:
    ev = load_events()
    series_paths = [ROOT / p for p in YIELD_FILES if (ROOT / p).exists()]
    if not series_paths:
        raise FileNotFoundError("No DGS5/DGS10/DGS20 files found")

    all_bt = []
    summary_rows = []
    for path in series_paths:
        y = load_yield(path)
        for off in ENTRY_OFFSETS:
            bt = backtest_one_series(y, path.stem, ev, off)
            if bt.empty:
                continue
            all_bt.append(bt)
            summary_rows.append({"series": path.stem, "entry_offset": off, **_summary(bt)})

    trades = pd.concat(all_bt, ignore_index=True) if all_bt else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    trades.to_csv(REPORT / "yield_proxy_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(REPORT / "yield_proxy_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Saved trades -> {REPORT / 'yield_proxy_trades.csv'}")
    print(f"Saved summary -> {REPORT / 'yield_proxy_summary.csv'}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
