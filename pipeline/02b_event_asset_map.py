from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from pipeline.utils_paths import output_dir

ASSETS: List[str] = ["OIL", "GOLD", "SPX", "USD", "CPI", "USDCNY", "CHNA_EQ", "GAS"]

# Knowledge graph rules: 0-cost, deterministic mapping.
EVENT_ASSET_GRAPH: Dict[str, Dict[str, float]] = {
    "iran": {"OIL": 0.85, "GAS": 0.55, "USD": 0.35, "GOLD": 0.45},
    "middle east": {"OIL": 0.80, "GAS": 0.60, "GOLD": 0.40},
    "opec": {"OIL": 0.90, "GAS": 0.65, "USD": 0.20},
    "red sea": {"OIL": 0.55, "GAS": 0.45, "CPI": 0.35, "SPX": 0.20},
    "ukraine": {"OIL": 0.65, "GAS": 0.80, "GOLD": 0.45, "CPI": 0.35},
    "russia": {"OIL": 0.60, "GAS": 0.75, "GOLD": 0.35},
    "fed": {"USD": 0.90, "SPX": 0.70, "GOLD": 0.55, "CPI": 0.40},
    "fomc": {"USD": 0.90, "SPX": 0.70, "GOLD": 0.55, "CPI": 0.40},
    "rate hike": {"USD": 0.85, "SPX": 0.60, "GOLD": 0.55},
    "rate cut": {"USD": 0.65, "SPX": 0.70, "GOLD": 0.50},
    "cpi": {"CPI": 1.00, "USD": 0.55, "SPX": 0.35, "GOLD": 0.40},
    "inflation": {"CPI": 0.95, "USD": 0.50, "SPX": 0.35, "GOLD": 0.45},
    "tariff": {"USDCNY": 0.65, "CHNA_EQ": 0.55, "SPX": 0.45, "CPI": 0.30},
    "trade war": {"USDCNY": 0.75, "CHNA_EQ": 0.65, "SPX": 0.55, "CPI": 0.35},
    "shutdown": {"USD": 0.45, "SPX": 0.45, "GOLD": 0.35},
    "election": {"USD": 0.45, "SPX": 0.50, "CHNA_EQ": 0.20},
    "banking crisis": {"GOLD": 0.75, "USD": 0.60, "SPX": 0.70},
    "credit suisse": {"GOLD": 0.60, "USD": 0.55, "SPX": 0.65},
    "chatgpt": {"SPX": 0.75, "CHNA_EQ": 0.35, "USD": 0.15},
    "ai": {"SPX": 0.65, "CHNA_EQ": 0.30, "USD": 0.10},
}

# Chinese aliases
ZH_ALIAS = {
    "伊朗": "iran",
    "中东": "middle east",
    "欧佩克": "opec",
    "红海": "red sea",
    "俄乌": "ukraine",
    "俄罗斯": "russia",
    "美联储": "fed",
    "加息": "rate hike",
    "降息": "rate cut",
    "关税": "tariff",
    "贸易战": "trade war",
    "停摆": "shutdown",
    "大选": "election",
    "银行危机": "banking crisis",
    "通胀": "inflation",
}

ASSET_PROTOTYPES: Dict[str, List[str]] = {
    "OIL": ["oil price", "crude price", "brent", "wti", "energy shock"],
    "GOLD": ["gold price", "safe haven", "bullion", "flight to quality"],
    "SPX": ["sp500 index", "us equities", "risk assets", "stocks drawdown"],
    "USD": ["us dollar index", "dxy", "usd strength", "global liquidity"],
    "CPI": ["inflation cpi ppi", "price pressure", "consumer inflation"],
    "USDCNY": ["usd cny exchange rate", "renminbi pressure", "yuan depreciation"],
    "CHNA_EQ": ["china equities", "a shares", "hang seng china enterprises"],
    "GAS": ["natural gas price", "lng market", "europe gas crisis"],
}


@dataclass
class MapResult:
    scores: Dict[str, float]
    method: str
    confidence: float


def _parse_json_list(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x]
    text = str(value).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data if x]
    except Exception:
        pass
    return [text]


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    out = {k: float(np.clip(v, 0.0, 1.0)) for k, v in scores.items() if k in ASSETS}
    for a in ASSETS:
        out.setdefault(a, 0.0)
    return out


class EventAssetMapper:
    def __init__(
        self,
        graph_threshold: float = 0.5,
        vector_threshold: float = 0.4,
        enable_ai_fallback: bool = False,
        vector_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.graph_threshold = graph_threshold
        self.vector_threshold = vector_threshold
        self.enable_ai_fallback = enable_ai_fallback
        self.model = SentenceTransformer(vector_model_name)

        self.prototype_texts = []
        self.prototype_assets = []
        for asset, phrases in ASSET_PROTOTYPES.items():
            for phrase in phrases:
                self.prototype_assets.append(asset)
                self.prototype_texts.append(phrase)
        self.prototype_emb = self.model.encode(self.prototype_texts, normalize_embeddings=True)

        self.ai_base = os.getenv("N1N_BASE_URL", "https://api.n1n.ai/v1")
        self.ai_key = os.getenv("N1N_API_KEY", "")
        self.ai_model = os.getenv("N1N_MODEL", "gpt-5.2")

    def graph_match(self, text: str, tags: List[str]) -> Dict[str, float]:
        merged = f"{text} {' '.join(tags)}".lower()
        scores = {a: 0.0 for a in ASSETS}

        for zh, en in ZH_ALIAS.items():
            if zh in merged:
                merged += f" {en}"

        for key, asset_map in EVENT_ASSET_GRAPH.items():
            if key in merged:
                for asset, value in asset_map.items():
                    scores[asset] = max(scores[asset], value)

        return _normalize(scores)

    def vector_match(self, text: str) -> Dict[str, float]:
        emb = self.model.encode([text], normalize_embeddings=True)[0]
        sim = np.dot(self.prototype_emb, emb)
        scores = {a: 0.0 for a in ASSETS}
        for idx, raw in enumerate(sim):
            asset = self.prototype_assets[idx]
            val = float(max(raw, 0.0))
            scores[asset] = max(scores[asset], val)
        return _normalize(scores)

    def ai_reason(self, title: str) -> Dict[str, float]:
        if not self.enable_ai_fallback or not self.ai_key:
            return {a: 0.0 for a in ASSETS}

        prompt = (
            "你是金融事件-资产关联引擎。"
            "输入事件标题，输出JSON：{资产: 0~1相关性}。"
            "只输出标准资产：OIL, GOLD, SPX, USD, CPI, USDCNY, CHNA_EQ, GAS。"
            "无关联则为0。不要解释，只输出JSON。"
        )
        payload = {
            "model": self.ai_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": title},
            ],
            "temperature": 0,
            "max_tokens": 120,
        }
        headers = {"Authorization": f"Bearer {self.ai_key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(
                f"{self.ai_base.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            data = json.loads(text)
            return _normalize({k: float(data.get(k, 0.0)) for k in ASSETS})
        except Exception:
            return {a: 0.0 for a in ASSETS}

    def map_asset(self, title: str, tags: List[str]) -> MapResult:
        graph_scores = self.graph_match(title, tags)
        graph_conf = max(graph_scores.values())
        if graph_conf > self.graph_threshold:
            return MapResult(graph_scores, "graph", graph_conf)

        vector_scores = self.vector_match(title)
        vector_conf = max(vector_scores.values())
        if vector_conf > self.vector_threshold:
            return MapResult(vector_scores, "vector", vector_conf)

        ai_scores = self.ai_reason(title)
        ai_conf = max(ai_scores.values())
        return MapResult(ai_scores, "ai_fallback", ai_conf)


def _load_source(event_map_path: Path, markets_path: Path) -> pd.DataFrame:
    event_df = pd.read_csv(event_map_path)
    mk_df = pd.read_csv(markets_path)
    event_df["market_id"] = event_df["market_id"].astype(str)
    mk_df["market_id"] = mk_df["market_id"].astype(str)
    merged = event_df.merge(mk_df[["market_id", "question", "topic"]], on="market_id", how="left")
    return merged


def build_market_asset_map(
    df: pd.DataFrame,
    mapper: EventAssetMapper,
    max_rows: int | None = None,
) -> pd.DataFrame:
    work = df.copy()
    if max_rows is not None and max_rows > 0:
        work = work.head(max_rows)

    rows = []
    for _, row in work.iterrows():
        tags = _parse_json_list(row.get("official_tag_labels")) + _parse_json_list(row.get("official_tag_slugs"))
        event_title = str(row.get("event_title", "") or "")
        question = str(row.get("question", "") or "")
        title = (event_title + " | " + question).strip(" |")
        result = mapper.map_asset(title=title, tags=tags)
        top_asset = max(result.scores, key=result.scores.get)
        rows.append(
            {
                "market_id": row.get("market_id"),
                "event_id": row.get("event_id"),
                "event_title": event_title,
                "question": question,
                "topic": row.get("topic"),
                "method": result.method,
                "confidence": result.confidence,
                "top_asset": top_asset,
                "top_score": result.scores[top_asset],
                "asset_scores_json": json.dumps(result.scores, ensure_ascii=False),
            }
        )
    out = pd.DataFrame(rows)
    for asset in ASSETS:
        out[asset] = out["asset_scores_json"].map(lambda x, a=asset: float(json.loads(x).get(a, 0.0)))
    return out


def build_event_asset_summary(market_asset_df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = ["OIL", "GOLD", "SPX", "USD", "CPI", "USDCNY", "CHNA_EQ", "GAS"]
    grouped = (
        market_asset_df.groupby(["event_id", "event_title"], dropna=False)[agg_cols]
        .mean()
        .reset_index()
    )
    grouped["top_asset"] = grouped[agg_cols].idxmax(axis=1)
    grouped["top_score"] = grouped[agg_cols].max(axis=1)
    grouped["n_markets"] = (
        market_asset_df.groupby(["event_id", "event_title"], dropna=False)["market_id"].nunique().values
    )
    return grouped.sort_values(["top_score", "n_markets"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Low-cost event->asset mapping: graph first, vectors second, AI fallback last."
    )
    parser.add_argument(
        "--event-map-path",
        type=Path,
        default=output_dir() / "markets" / "official_event_tag_map.csv",
    )
    parser.add_argument(
        "--markets-path",
        type=Path,
        default=output_dir() / "pipeline" / "markets.csv",
    )
    parser.add_argument(
        "--out-market-path",
        type=Path,
        default=output_dir() / "report" / "market_asset_map_low_cost.csv",
    )
    parser.add_argument(
        "--out-event-path",
        type=Path,
        default=output_dir() / "report" / "event_asset_map_low_cost.csv",
    )
    parser.add_argument("--graph-threshold", type=float, default=0.5)
    parser.add_argument("--vector-threshold", type=float, default=0.4)
    parser.add_argument("--enable-ai-fallback", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    source_df = _load_source(args.event_map_path, args.markets_path)
    mapper = EventAssetMapper(
        graph_threshold=args.graph_threshold,
        vector_threshold=args.vector_threshold,
        enable_ai_fallback=args.enable_ai_fallback,
    )
    market_map = build_market_asset_map(
        source_df,
        mapper,
        max_rows=args.max_rows if args.max_rows > 0 else None,
    )
    event_map = build_event_asset_summary(market_map)

    args.out_market_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_event_path.parent.mkdir(parents=True, exist_ok=True)
    market_map.to_csv(args.out_market_path, index=False, encoding="utf-8-sig")
    event_map.to_csv(args.out_event_path, index=False, encoding="utf-8-sig")

    method_stats = market_map["method"].value_counts(dropna=False).to_dict()
    print(f"[saved] {args.out_market_path}")
    print(f"[saved] {args.out_event_path}")
    print(f"[rows] market={len(market_map)} event={len(event_map)}")
    print(f"[method_stats] {method_stats}")


if __name__ == "__main__":
    main()
