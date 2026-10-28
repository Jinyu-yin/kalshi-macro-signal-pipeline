from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from pipeline.utils_paths import output_dir, project_root
from pipeline.wheels_market_data import ensure_polymarket_markets

ROOT = project_root()
OUT_MARKETS_DIR = output_dir() / "markets"
CATEGORIES_DIR = OUT_MARKETS_DIR / "categories"
CATEGORIES_DIR.mkdir(parents=True, exist_ok=True)

# 基于 Polymarket 官网筛选口径扩展主题（可继续迭代关键词）
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "trade_war": ["trade war", "tariff", "tariffs", "china tariff", "import duty"],
    "fed_rates": ["fed", "fomc", "fed funds", "rate cut", "rate hike", "interest rate", "powell"],
    "inflation": ["inflation", "cpi", "ppi", "core pce", "deflation"],
    "macro_indicators": ["nfp", "unemployment", "jobless", "payrolls", "retail sales", "pmi", "ism", "consumer confidence"],
    "gdp": ["gdp", "gross domestic product", "recession"],
    "global_rates": ["ecb", "boe", "boj", "central bank", "policy rate", "yield", "government bond"],
    "taxes": ["tax", "taxes", "corporate tax", "income tax", "tax bill"],
    "treasuries": ["treasury", "2-year yield", "5-year yield", "10-year yield", "30-year yield", "10y", "2y", "yield curve", "ust", "bond yield", "fed funds rate", "fomc", "rate cut", "rate hike"],
    "tsa": ["tsa", "transportation security administration"],
    "housing": ["housing", "home price", "mortgage rate", "real estate", "housing starts"],
    "oil": ["oil", "wti", "brent", "crude", "opec"],
    "commodities": ["commodities", "commodity", "copper", "lithium", "uranium", "natural gas"],
    "equities": ["equities", "stock market", "stocks", "index", "indices", "spx", "s&p", "nasdaq"],
    "ipo": ["ipo", "initial public offering", "listing day"],
    "gold_silver": ["gold", "silver", "xau", "xag"],
    "big_tech": ["apple", "aapl", "amazon", "amzn", "meta", "nvidia", "nvda", "tesla", "tsla", "netflix", "nflx"],
    "acquisitions": ["acquisition", "acquire", "merger", "m&a", "takeover", "buyout"],
    "ath": ["all-time high", "ath", "record high"],
}

KALSHI_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "fed_rates": ["fed funds rate", "fomc", "fed decision", "rate cut", "rate hike", "interest rate", "powell"],
    "inflation": ["cpi", "inflation", "ppi", "core pce", "pce"],
    "macro_indicators": ["unemployment", "jobless", "payroll", "nfp", "nonfarm", "retail sales", "pmi", "ism"],
    "gdp": ["gdp", "gross domestic product", "recession"],
    "global_rates": ["ecb", "boe", "boj", "central bank", "policy rate", "yield"],
    "taxes": ["tax", "taxes"],
    "treasuries": ["treasury", "2-year", "5-year", "10-year", "30-year", "yield curve", "ust", "bond yield"],
    "housing": ["housing", "mortgage", "home price", "real estate"],
    "oil": ["oil", "wti", "brent", "crude", "opec"],
    "commodities": ["commodity", "commodities", "copper", "lithium", "uranium", "natural gas"],
    "equities": ["stock market", "stocks", "index", "spx", "s&p", "nasdaq"],
    "ipo": ["ipo", "initial public offering"],
    "gold_silver": ["gold", "silver", "xau", "xag"],
    "big_tech": ["apple", "amazon", "meta", "nvidia", "tesla", "netflix"],
    "acquisitions": ["acquisition", "acquire", "merger", "takeover", "buyout"],
    "ath": ["all-time high", "record high"],
}


def _cleanup_old_topic_files() -> None:
    for p in CATEGORIES_DIR.glob("poly_*_markets.jsonl"):
        p.unlink(missing_ok=True)
    for p in CATEGORIES_DIR.glob("poly_*_ids.csv"):
        p.unlink(missing_ok=True)


def _read_polymarket_markets() -> pd.DataFrame:
    # Do not hard-cap market count; load full local parquet or full Gamma pagination.
    df, source = ensure_polymarket_markets(max_markets=None)
    if df.empty:
        raise FileNotFoundError("No polymarket markets found locally and Gamma API fetch failed.")
    print(f"[00] markets source: {source} | rows={len(df):,}")

    return pd.DataFrame(
        {
            "id": df["id"].astype(str),
            "question": df.get("question", "").astype(str),
            "slug": df.get("slug", "").astype(str),
            "end_date": df.get("end_date"),
            "created_at": df.get("created_at"),
            "volume": df.get("volume"),
            "liquidity": df.get("liquidity"),
            "active": df.get("active"),
            "closed": df.get("closed"),
            "category": df.get("category"),
            "subcategory": df.get("subcategory"),
            "tags": df.get("tags"),
            "events": df.get("events"),
            "series": df.get("series"),
            "event_titles": df.get("event_titles"),
            "event_slugs": df.get("event_slugs"),
            "series_titles": df.get("series_titles"),
            "series_slugs": df.get("series_slugs"),
        }
    )


def _compile_topic_patterns(topic_keywords: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
    patterns: Dict[str, re.Pattern] = {}
    for topic, kws in topic_keywords.items():
        safe = [re.escape(k.lower()) for k in kws if k and k.strip()]
        if safe:
            patterns[topic] = re.compile("|".join(safe))
    return patterns


def _compose_tag_text(markets: pd.DataFrame) -> pd.Series:
    base = markets.get("official_tag_slugs", pd.Series([""] * len(markets), index=markets.index)).fillna("")
    labels = markets.get("official_tag_labels", pd.Series([""] * len(markets), index=markets.index)).fillna("")
    text = (base.astype(str) + " " + labels.astype(str)).str.lower()
    if "platform" in markets.columns:
        kalshi_mask = markets["platform"].astype(str).str.lower().eq("kalshi")
        kalshi_text = (
            markets.get("question", pd.Series([""] * len(markets), index=markets.index)).fillna("")
            + " "
            + markets.get("slug", pd.Series([""] * len(markets), index=markets.index)).fillna("")
            + " "
            + markets.get("category", pd.Series([""] * len(markets), index=markets.index)).fillna("")
            + " "
            + markets.get("subcategory", pd.Series([""] * len(markets), index=markets.index)).fillna("")
            + " "
            + markets.get("tags", pd.Series([""] * len(markets), index=markets.index)).fillna("")
        ).astype(str).str.lower()
        text = text.where(~kalshi_mask, kalshi_text)
    return text


def classify_topics(markets: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    poly_patterns = _compile_topic_patterns(TOPIC_KEYWORDS)
    kalshi_patterns = _compile_topic_patterns(KALSHI_TOPIC_KEYWORDS)

    if "platform" not in markets.columns:
        raise RuntimeError("Missing `platform` in markets")

    tag_text = _compose_tag_text(markets)
    topic_map: Dict[str, pd.DataFrame] = {}
    any_mask = pd.Series(False, index=markets.index)

    for topic, pat in poly_patterns.items():
        mask = tag_text.str.contains(pat, na=False)
        topic_map[topic] = markets.loc[mask].copy()
        any_mask |= mask

    # Add Kalshi-only topic bins, keyed by the same topic names where possible.
    kalshi_only = markets[markets["platform"].astype(str).str.lower().eq("kalshi")].copy()
    if not kalshi_only.empty:
        kalshi_text = _compose_tag_text(kalshi_only)
        for topic, pat in kalshi_patterns.items():
            mask = kalshi_text.str.contains(pat, na=False)
            if topic in topic_map:
                topic_map[topic] = pd.concat([topic_map[topic], kalshi_only.loc[mask]], ignore_index=True)
            else:
                topic_map[topic] = kalshi_only.loc[mask].copy()
            any_mask.loc[kalshi_only.index] |= mask.reindex(kalshi_only.index, fill_value=False)

    return markets.loc[any_mask].copy(), topic_map


def save_topic_jsons(topic_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary_rows = []

    def _jsonable(v):
        if v is None:
            return None
        if isinstance(v, pd.Timestamp):
            return v.isoformat() if pd.notna(v) else None
        if isinstance(v, (str, int, float, bool)):
            return v
        return str(v)

    for topic, df in topic_map.items():
        jsonl_path = CATEGORIES_DIR / f"poly_{topic}_markets.jsonl"
        ids_path = CATEGORIES_DIR / f"poly_{topic}_ids.csv"

        if df.empty:
            summary_rows.append({"topic": topic, "markets": 0})
            continue

        with jsonl_path.open("w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                rec = {
                    "id": row["id"],
                    "question": row["question"],
                    "slug": row.get("slug", ""),
                    "end_date": _jsonable(row.get("end_date", None)),
                    "created_at": _jsonable(row.get("created_at", None)),
                    "volume": _jsonable(row.get("volume", None)),
                    "liquidity": _jsonable(row.get("liquidity", None)),
                    "active": _jsonable(row.get("active", None)),
                    "closed": _jsonable(row.get("closed", None)),
                    "category": _jsonable(row.get("category", None)),
                    "subcategory": _jsonable(row.get("subcategory", None)),
                    "tags": _jsonable(row.get("tags", None)),
                    "events": _jsonable(row.get("events", None)),
                    "series": _jsonable(row.get("series", None)),
                    "event_titles": _jsonable(row.get("event_titles", None)),
                    "event_slugs": _jsonable(row.get("event_slugs", None)),
                    "series_titles": _jsonable(row.get("series_titles", None)),
                    "series_slugs": _jsonable(row.get("series_slugs", None)),
                    "platform": "polymarket",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        df[["id"]].drop_duplicates().to_csv(ids_path, index=False, encoding="utf-8")
        summary_rows.append({"topic": topic, "markets": int(len(df))})
        print(f"[00] topic={topic:<18} markets={len(df):,} -> {jsonl_path.name}")

    return pd.DataFrame(summary_rows).sort_values("markets", ascending=False)


def main() -> None:
    _cleanup_old_topic_files()

    markets = _read_polymarket_markets()
    markets = markets.dropna(subset=["id", "question"])
    markets = markets[markets["question"].astype(str).str.strip() != ""]

    # Merge official Gamma event tag mapping (Event.tags -> market_id) for tag-first classification.
    tag_map_path = OUT_MARKETS_DIR / "official_event_tag_map.csv"
    if not tag_map_path.exists():
        raise FileNotFoundError(
            f"Missing {tag_map_path}. Run `python pipeline/00b_build_official_event_tag_map.py` first."
        )

    tag_map = pd.read_csv(tag_map_path)
    if "market_id" not in tag_map.columns:
        raise RuntimeError("official_event_tag_map.csv missing `market_id` column")
    tag_map = tag_map.rename(columns={"market_id": "id"})
    for c in ["official_tag_slugs", "official_tag_labels"]:
        if c not in tag_map.columns:
            tag_map[c] = None

    markets["id"] = markets["id"].astype(str)
    tag_map["id"] = tag_map["id"].astype(str)
    markets = markets.merge(tag_map[["id", "official_tag_slugs", "official_tag_labels"]], on="id", how="left")

    _, topic_map = classify_topics(markets)
    summary = save_topic_jsons(topic_map)

    summary_path = OUT_MARKETS_DIR / "poly_topics_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"[00] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
