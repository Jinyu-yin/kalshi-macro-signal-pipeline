from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT_THEMES: Dict[str, List[str]] = {
    "俄乌战争": ["ukraine", "russia", "俄乌", "俄罗斯", "乌克兰"],
    "美联储加息": ["fed", "fomc", "rate hike", "rate cut", "加息", "降息", "美联储"],
    "关税与贸易战": ["tariff", "trade war", "trade-war", "关税", "贸易战"],
    "伊朗-美国冲突": ["iran", "middle east", "伊朗", "中东", "israel", "以色列"],
    "美国大选": ["election", "trump", "biden", "harris", "大选", "选举"],
    "欧美银行业危机": ["banking crisis", "credit suisse", "svb", "银行危机"],
    "美国政府停摆": ["shutdown", "government shutdown", "停摆"],
    "红海航运危机": ["red sea", "suez", "shipping", "红海", "航运"],
    "英国迷你预算危机": ["liz truss", "mini budget", "gilt", "英国", "迷你预算"],
    "AI革命": ["chatgpt", "openai", "ai", "nvidia", "人工智能"],
}

GENERIC_PATTERNS = [
    "by end of",
    "by year end",
    "this year",
    "this month",
    "anytime",
    "at any point",
    "before end of",
    "will there be",
    "yes or no",
]


def _match_theme(text: str) -> str:
    t = (text or "").lower()
    for theme, keys in ROOT_THEMES.items():
        if any(k.lower() in t for k in keys):
            return theme
    return "其他"


def _is_too_generic(question: str) -> bool:
    q = (question or "").lower()
    if len(q) < 25:
        return True
    return any(p in q for p in GENERIC_PATTERNS)


def build_core_pool(
    input_path: Path,
    out_path: Path,
    summary_path: Path,
    min_top_score: float,
    top_n_per_theme: int,
    drop_generic: bool,
) -> None:
    df = pd.read_csv(input_path)
    if df.empty:
        raise RuntimeError(f"Input is empty: {input_path}")

    work = df.copy()
    work["event_title"] = work["event_title"].fillna("").astype(str)
    work["question"] = work["question"].fillna("").astype(str)
    work["topic"] = work["topic"].fillna("").astype(str)
    if "top_score" not in work.columns:
        work["top_score"] = 1.0
    if "confidence" not in work.columns:
        work["confidence"] = 1.0
    if "method" not in work.columns:
        work["method"] = "rule_only"
    if "top_asset" not in work.columns:
        work["top_asset"] = ""
    for c in ["OIL", "GOLD", "SPX", "USD", "CPI", "USDCNY", "CHNA_EQ", "GAS"]:
        if c not in work.columns:
            work[c] = 0.0

    work["top_score"] = pd.to_numeric(work["top_score"], errors="coerce").fillna(0.0)
    work["confidence"] = pd.to_numeric(work["confidence"], errors="coerce").fillna(0.0)

    work["theme"] = (work["event_title"] + " | " + work["question"]).map(_match_theme)
    work = work[work["theme"] != "其他"].copy()
    work = work[work["top_score"] >= min_top_score].copy()
    if drop_generic:
        work = work[~work["question"].map(_is_too_generic)].copy()

    # Keep higher-confidence and more informative rows first.
    work = work.sort_values(
        by=["theme", "top_score", "confidence", "event_id", "market_id"],
        ascending=[True, False, False, True, True],
    )

    core = (
        work.groupby("theme", dropna=False)
        .head(top_n_per_theme)
        .drop_duplicates(subset=["theme", "event_id", "question"])
        .drop_duplicates(subset=["market_id"])
        .copy()
    )

    keep_cols = [
        "theme",
        "market_id",
        "event_id",
        "event_title",
        "question",
        "topic",
        "method",
        "confidence",
        "top_asset",
        "top_score",
        "OIL",
        "GOLD",
        "SPX",
        "USD",
        "CPI",
        "USDCNY",
        "CHNA_EQ",
        "GAS",
    ]
    core = core[[c for c in keep_cols if c in core.columns]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    core.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = (
        core.groupby("theme", dropna=False)
        .agg(
            n_markets=("market_id", "nunique"),
            n_events=("event_id", "nunique"),
            avg_top_score=("top_score", "mean"),
        )
        .reset_index()
        .sort_values("n_markets", ascending=False)
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[saved] {out_path}")
    print(f"[saved] {summary_path}")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build high-discriminability core market pool from low-cost event-asset mapping."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("output/report/major_themes_markets_from_events.csv"),
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("output/report/core_market_pool_no_ai.csv"),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("output/report/core_market_pool_no_ai_summary.csv"),
    )
    parser.add_argument("--min-top-score", type=float, default=0.55)
    parser.add_argument("--top-n-per-theme", type=int, default=50)
    parser.add_argument("--drop-generic", action="store_true")
    args = parser.parse_args()

    build_core_pool(
        input_path=args.input_path,
        out_path=args.out_path,
        summary_path=args.summary_path,
        min_top_score=args.min_top_score,
        top_n_per_theme=args.top_n_per_theme,
        drop_generic=args.drop_generic,
    )


if __name__ == "__main__":
    main()
