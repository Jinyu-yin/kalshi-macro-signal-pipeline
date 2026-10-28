from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from pipeline.utils_paths import output_dir, output_pipeline_dir, output_topic_dir
from pipeline.utils_topics import discover_poly_topic_files

plt.style.use("seaborn-v0_8-whitegrid")


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_table_csv(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return path.name


def _save_table_png(df: pd.DataFrame, title: str, path: Path) -> str:
    if df.empty:
        return ""
    fig_h = max(2.2, 0.42 * (len(df) + 2))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def _single_factor_outputs(report_dir: Path) -> Dict[str, List[str]]:
    out = {"figs": [], "tables": []}
    p = output_pipeline_dir() / "single_factor_summary.csv"
    if not p.exists():
        return out

    sf = pd.read_csv(p)
    if sf.empty:
        return out

    pivot = sf.pivot_table(index="topic", columns="factor", values="ic_spearman", aggfunc="mean")
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.65 * len(pivot.index))))
        im = ax.imshow(pivot.fillna(0).to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-0.08, vmax=0.08)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("Single-factor IC (Spearman, daily)")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("IC")
        fig.tight_layout()
        out_path = report_dir / "single_factor_ic_heatmap.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        out["figs"].append(out_path.name)

    rank = (
        sf.groupby("factor", as_index=False)
        .agg(avg_ic=("ic_spearman", "mean"), avg_sharpe=("factor_sharpe", "mean"), avg_win_rate=("factor_win_rate", "mean"), avg_cum_ret=("cum_factor_ret", "mean"))
        .sort_values(["avg_ic", "avg_sharpe"], ascending=False)
        .round(4)
    )
    out["tables"].append(_save_table_csv(rank, report_dir / "table_single_factor_summary.csv"))
    png = _save_table_png(rank, "Single-factor summary (strict, daily)", report_dir / "table_single_factor_summary.png")
    if png:
        out["figs"].append(png)

    top = sf.sort_values(["factor_sharpe", "ic_spearman"], ascending=False).head(20)
    cols = ["topic", "factor", "n_obs", "ic_spearman", "factor_sharpe", "factor_win_rate", "cum_factor_ret", "pass_topic"]
    cols = [c for c in cols if c in top.columns]
    out["tables"].append(_save_table_csv(top[cols].round(4), report_dir / "table_single_factor_top20.csv"))
    return out


def _backtest_outputs(report_dir: Path) -> Dict[str, List[str]]:
    out = {"figs": [], "tables": []}
    p = output_dir() / "event_backtest_results.csv"
    if not p.exists():
        return out
    df = pd.read_csv(p)
    if df.empty:
        return out
    if "horizon_d" in df.columns and "horizon" not in df.columns:
        df["horizon"] = df["horizon_d"]
        df["horizon_unit"] = "d"
    if "signal_freq" not in df.columns:
        df["signal_freq"] = "1d"
    df["horizon_label"] = df["horizon"].astype(str) + df["horizon_unit"].astype(str)

    strat = (
        df.groupby(["strategy", "signal_freq", "horizon_label"], as_index=False)
        .agg(n_topics=("topic", "nunique"), avg_win_rate=("win_rate", "mean"), avg_sharpe=("sharpe", "mean"), avg_cum_return=("cum_return", "mean"), avg_profit_factor=("profit_factor", "mean"), avg_n_trades=("n_trades", "mean"))
        .sort_values(["strategy", "signal_freq", "horizon_label"])
        .round(4)
    )
    out["tables"].append(_save_table_csv(strat, report_dir / "table_backtest_strategy_horizon.csv"))
    png = _save_table_png(strat, "Backtest summary by strategy/horizon (daily)", report_dir / "table_backtest_strategy_horizon.png")
    if png:
        out["figs"].append(png)

    t_focus = (
        df.sort_values(["strategy", "signal_freq", "sharpe"], ascending=[True, True, False])
        [["topic", "strategy", "signal_freq", "horizon_label", "n_trades", "win_rate", "sharpe", "cum_return", "profit_factor"]]
        .head(30)
        .round(4)
    )
    out["tables"].append(_save_table_csv(t_focus, report_dir / "table_backtest_topic_strategy_5d.csv"))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for (strategy, signal_freq), g in strat.groupby(["strategy", "signal_freq"]):
        ax.plot(range(len(g)), g["avg_win_rate"], marker="o", linewidth=2, label=f"{strategy}-{signal_freq}")
    ax.set_xticks(range(len(strat["horizon_label"].unique())))
    ax.set_xticklabels(sorted(strat["horizon_label"].unique()))
    ax.set_ylim(0, 1)
    ax.set_title("Average win-rate by horizon (freq-aware)")
    ax.set_xlabel("horizon")
    ax.set_ylabel("win-rate")
    ax.legend()
    fig.tight_layout()
    p1 = report_dir / "backtest_winrate_by_horizon.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    out["figs"].append(p1.name)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for (strategy, signal_freq), g in strat.groupby(["strategy", "signal_freq"]):
        ax.plot(range(len(g)), g["avg_sharpe"], marker="o", linewidth=2, label=f"{strategy}-{signal_freq}")
    ax.set_xticks(range(len(strat["horizon_label"].unique())))
    ax.set_xticklabels(sorted(strat["horizon_label"].unique()))
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Average Sharpe by horizon (freq-aware)")
    ax.set_xlabel("horizon")
    ax.set_ylabel("Sharpe")
    ax.legend()
    fig.tight_layout()
    p2 = report_dir / "backtest_sharpe_by_horizon.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    out["figs"].append(p2.name)

    return out


def _topic_case_outputs(report_dir: Path) -> Dict[str, List[str]]:
    out = {"figs": [], "tables": []}
    for tf in discover_poly_topic_files():
        topic = tf.topic
        topic_dir = output_topic_dir(topic)
        jump_event = _safe_read_parquet(topic_dir / "signals" / "jump" / "1d" / "event_signal.parquet")
        rev_event = _safe_read_parquet(topic_dir / "signals" / "reversal" / "1d" / "event_signal.parquet")
        mom_event = _safe_read_parquet(topic_dir / "signals" / "momentum" / "1d" / "event_signal.parquet")
        if jump_event.empty:
            jump_event = _safe_read_parquet(topic_dir / "signals" / "jump" / "event_signal.parquet")
        if rev_event.empty:
            rev_event = _safe_read_parquet(topic_dir / "signals" / "reversal" / "event_signal.parquet")
        if jump_event.empty and rev_event.empty and mom_event.empty:
            continue

        if not jump_event.empty:
            jump_event["ts"] = pd.to_datetime(jump_event["ts"], utc=True, errors="coerce")
            jump_event = jump_event.dropna(subset=["ts"]).sort_values("ts")
        if not rev_event.empty:
            rev_event["ts"] = pd.to_datetime(rev_event["ts"], utc=True, errors="coerce")
            rev_event = rev_event.dropna(subset=["ts"]).sort_values("ts")
        if not mom_event.empty:
            mom_event["ts"] = pd.to_datetime(mom_event["ts"], utc=True, errors="coerce")
            mom_event = mom_event.dropna(subset=["ts"]).sort_values("ts")

        fig, ax = plt.subplots(figsize=(10, 4.5))
        if not jump_event.empty:
            ax.plot(jump_event["ts"], jump_event["signal_volume_weighted"], label="jump", linewidth=1.0)
        if not rev_event.empty:
            ax.plot(rev_event["ts"], rev_event["signal_volume_weighted"], label="reversal", linewidth=1.0)
        if not mom_event.empty:
            ax.plot(mom_event["ts"], mom_event["signal_volume_weighted"], label="momentum", linewidth=1.0)
        ax.set_title(f"{topic}: daily event signal")
        ax.set_xlabel("date")
        ax.set_ylabel("signal")
        ax.legend()
        fig.tight_layout()
        p = report_dir / f"{topic}_signal_timeseries.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out["figs"].append(p.name)

        parts = []
        if not jump_event.empty:
            t = jump_event[["ts", "signal_volume_weighted"]].copy(); t["strategy"] = "jump"; parts.append(t)
        if not rev_event.empty:
            t = rev_event[["ts", "signal_volume_weighted"]].copy(); t["strategy"] = "reversal"; parts.append(t)
        if not mom_event.empty:
            t = mom_event[["ts", "signal_volume_weighted"]].copy(); t["strategy"] = "momentum"; parts.append(t)
        if parts:
            all_sig = pd.concat(parts, ignore_index=True)
            all_sig["abs_signal"] = all_sig["signal_volume_weighted"].abs()
            top = all_sig.nlargest(12, "abs_signal")[["ts", "strategy", "signal_volume_weighted", "abs_signal"]].round(6)
            out["tables"].append(_save_table_csv(top, report_dir / f"{topic}_case_top_signals.csv"))

    return out


def _write_report_md(report_dir: Path, sections: Dict[str, Dict[str, List[str]]]) -> None:
    lines: List[str] = []
    lines.append("# US Treasury Rate Prediction-Market Report (Daily)")
    lines.append("")
    lines.append("## 1) Strict single-factor diagnostics")
    for f in sections["single_factor"]["figs"]:
        lines.extend(["", f"![]({f})"])
    lines.append("")
    lines.append("### CSV tables")
    for t in sections["single_factor"]["tables"]:
        lines.append(f"- `{t}`")

    lines.append("")
    lines.append("## 2) Daily strategy backtest")
    for f in sections["backtest"]["figs"]:
        lines.extend(["", f"![]({f})"])
    lines.append("")
    lines.append("### CSV tables")
    for t in sections["backtest"]["tables"]:
        lines.append(f"- `{t}`")

    lines.append("")
    lines.append("## 3) Case studies")
    for f in sections["topics"]["figs"]:
        lines.extend(["", f"![]({f})"])
    lines.append("")
    lines.append("### CSV tables")
    for t in sections["topics"]["tables"]:
        lines.append(f"- `{t}`")

    (report_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    report_dir = _ensure_dir(output_dir() / "report")
    sections = {
        "single_factor": _single_factor_outputs(report_dir),
        "backtest": _backtest_outputs(report_dir),
        "topics": _topic_case_outputs(report_dir),
    }
    _write_report_md(report_dir, sections)
    print(f"[report] saved chart+table pack -> {report_dir}")


if __name__ == "__main__":
    main()
