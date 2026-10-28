from __future__ import annotations

# 允许直接用 `python pipeline/xx.py` 运行（确保项目根目录在 sys.path）
import sys
from pathlib import Path

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

"""
Step 1: 市场读取（按主题分别存储）

优先读取 Step0 的主题分类结果：
    output/markets/categories/poly_*_markets.jsonl

并分别输出：
    output/pipeline/topics/{topic}/markets.csv

同时输出一个全量汇总（方便调试/回退）：
    output/pipeline/markets.csv

如果 Step0 结果不存在，则 fallback 到读取 data/*/markets 的 parquet（全量）。
"""

from typing import List

import pandas as pd

from pipeline.utils_paths import data_dir, output_dir, output_pipeline_dir
from pipeline.event_asset_mapper import ASSETS as MAPPER_ASSETS
from pipeline.event_asset_mapper import EventAssetMapper, parse_json_list
from pipeline.utils_topics import discover_poly_topic_files, read_topic_markets_jsonl
from pipeline.wheels_market_data import ensure_polymarket_markets

def _attach_asset_map(df: pd.DataFrame, mapper: EventAssetMapper) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    score_rows = []
    for _, r in out.iterrows():
        text = " ".join(
            [
                str(r.get("question", "") or ""),
                str(r.get("slug", "") or ""),
                str(r.get("category", "") or ""),
                str(r.get("subcategory", "") or ""),
                str(r.get("event_title", "") or ""),
            ]
        ).strip()
        tags = parse_json_list(r.get("official_tag_labels")) + parse_json_list(r.get("official_tag_slugs"))
        score_rows.append(mapper.map_asset_scores(text=text, tags=tags))
    score_df = pd.DataFrame(score_rows, index=out.index)
    for a in MAPPER_ASSETS:
        out[f"asset_score_{a}"] = pd.to_numeric(score_df.get(a, 0.0), errors="coerce").fillna(0.0)
    out["top_asset"] = score_df.idxmax(axis=1)
    out["top_asset_score"] = score_df.max(axis=1).astype(float)
    return out


def _load_official_market_tag_map() -> pd.DataFrame:
    p = output_dir() / "markets" / "official_event_tag_map.csv"
    if not p.exists():
        print("[01] official event-tag cache missing; run pipeline/00b_build_official_event_tag_map.py if needed")
        return pd.DataFrame()
    try:
        d = pd.read_csv(p)
    except Exception as e:
        print(f"[01] WARN failed to read official event-tag cache: {e}")
        return pd.DataFrame()
    if d.empty or "market_id" not in d.columns:
        return pd.DataFrame()
    d["market_id"] = d["market_id"].astype(str)
    return d


def load_polymarket_markets() -> pd.DataFrame:
    df, source = ensure_polymarket_markets(max_markets=None)
    if df.empty:
        return pd.DataFrame()
    print(f"[01] polymarket markets source: {source} | rows={len(df):,}")

    out = pd.DataFrame(
        {
            "market_id": df["id"].astype(str),
            "question": df.get("question", "").astype(str),
            "slug": df.get("slug", "").astype(str),
            "start_time": df.get("created_at"),
            "end_time": df.get("end_date"),
            "platform": "polymarket",
            "volume": pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0.0),
            "liquidity": pd.to_numeric(df.get("liquidity", 0.0), errors="coerce").fillna(0.0),
            "active": df.get("active", False),
            "closed": df.get("closed", False),
            "category": df.get("category", None),
            "subcategory": df.get("subcategory", None),
            "tags": df.get("tags", None),
            "events": df.get("events", None),
            "series": df.get("series", None),
            "event_titles": df.get("event_titles", None),
            "event_slugs": df.get("event_slugs", None),
            "series_titles": df.get("series_titles", None),
            "series_slugs": df.get("series_slugs", None),
        }
    )
    return out


def load_kalshi_markets() -> pd.DataFrame:
    markets_dir = data_dir() / "kalshi" / "markets"
    if not markets_dir.exists():
        return pd.DataFrame()

    import duckdb

    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet('{str(markets_dir / 'markets_*.parquet')}')").df()
    finally:
        con.close()
    out = pd.DataFrame(
        {
            "market_id": df["ticker"].astype(str),
            "question": df.get("title", "").astype(str),
            "slug": df.get("event_ticker", "").astype(str),
            "start_time": df.get("created_time"),
            "end_time": df.get("expected_expiration_time", df.get("close_time")),
            "platform": "kalshi",
            "volume": pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0.0),
            "liquidity": pd.to_numeric(df.get("open_interest", 0.0), errors="coerce").fillna(0.0),
            "active": ~df.get("status", pd.Series([""] * len(df))).astype(str).str.lower().isin(["finalized", "settled", "closed"]),
            "closed": df.get("status", pd.Series([""] * len(df))).astype(str).str.lower().isin(["finalized", "settled", "closed"]),
            "category": df.get("category", None),
            "subcategory": df.get("subcategory", None),
            "tags": df.get("tags", None),
            "events": None,
            "series": None,
            "event_titles": None,
            "event_slugs": None,
            "series_titles": None,
            "series_slugs": None,
        }
    )
    return out


def load_markets() -> pd.DataFrame:
    parts = []

    poly = load_polymarket_markets()
    if not poly.empty:
        parts.append(poly)

    kalshi = load_kalshi_markets()
    if not kalshi.empty:
        parts.append(kalshi)

    if not parts:
        return pd.DataFrame(
            columns=[
                "market_id",
                "question",
                "slug",
                "start_time",
                "end_time",
                "platform",
                "volume",
                "liquidity",
                "active",
                "closed",
                "category",
                "subcategory",
                "tags",
                "events",
                "series",
                "event_titles",
                "event_slugs",
                "series_titles",
                "series_slugs",
            ]
        )

    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["market_id", "question"])
    df = df[df["question"].astype(str).str.strip() != ""]
    return df


def main() -> None:
    out_dir = output_pipeline_dir()
    topic_files = discover_poly_topic_files()
    official_tag_map = _load_official_market_tag_map()
    mapper = EventAssetMapper(graph_threshold=0.5, vector_threshold=0.4)
    if not official_tag_map.empty:
        print(f"[01] loaded official event-tag map | rows={len(official_tag_map):,}")

    all_parts: List[pd.DataFrame] = []

    if topic_files:
        topics_root = out_dir / "topics"
        topics_root.mkdir(parents=True, exist_ok=True)

        print(f"[01] discovered topics: {len(topic_files)} under output/markets/categories")

        for tf in topic_files:
            df_t = read_topic_markets_jsonl(tf.path)
            df_t["topic"] = tf.topic
            df_t = df_t.dropna(subset=["market_id", "question"])
            df_t = df_t[df_t["question"].astype(str).str.strip() != ""]
            if not official_tag_map.empty:
                # Avoid merge suffixes: let official cache override these fields.
                for c in ["category", "subcategory", "tags"]:
                    if c in df_t.columns:
                        df_t = df_t.drop(columns=[c])
                df_t["market_id"] = df_t["market_id"].astype(str)
                df_t = df_t.merge(
                    official_tag_map[
                        [
                            "market_id",
                            "event_id",
                            "event_title",
                            "event_slug",
                            "official_tags",
                            "official_tag_slugs",
                            "official_tag_labels",
                            "category",
                            "subcategory",
                            "tags",
                        ]
                    ],
                    on="market_id",
                    how="left",
                )
            df_t = _attach_asset_map(df_t, mapper)

            tdir = topics_root / tf.topic
            tdir.mkdir(parents=True, exist_ok=True)
            tpath = tdir / "markets.csv"
            df_t.to_csv(tpath, index=False, encoding="utf-8")
            print(f"[01] topic={tf.topic:<16} markets={len(df_t):,} saved={tpath}")

            all_parts.append(df_t)

        markets = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    else:
        print("[01] no topic jsonl found. fallback to parquet markets load.")
        markets = load_markets()
        markets = _attach_asset_map(markets, mapper)

    out_path = out_dir / "markets.csv"
    markets.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[01] saved combined: {out_path} | rows={len(markets):,}")

    if not markets.empty and {"topic", "top_asset", "top_asset_score"}.issubset(markets.columns):
        topic_asset = (
            markets.groupby(["topic", "top_asset"], dropna=False)["top_asset_score"]
            .mean()
            .reset_index(name="asset_score_mean")
            .sort_values(["topic", "asset_score_mean"], ascending=[True, False])
        )
        map_path = out_dir / "topic_asset_map.csv"
        topic_asset.to_csv(map_path, index=False, encoding="utf-8")
        print(f"[01] saved topic-asset map: {map_path} | rows={len(topic_asset):,}")


if __name__ == "__main__":
    main()

