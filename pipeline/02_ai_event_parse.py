from __future__ import annotations

import sys
import json
import time
import os
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from pipeline.utils_paths import output_pipeline_dir

# ================= API CONFIG =================

N1N_API_KEY = os.getenv("N1N_API_KEY", "")
N1N_BASE_URL = "https://api.n1n.ai/v1"
N1N_MODEL = "gpt-5.2"

PROXIES = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

MAX_WORKERS = 30

SYSTEM_PROMPT = """
你是专业的金融债券/利率事件分析助手，仅输出JSON。

规则：
1. **若无法明确判断对美债/利率（如 ZT/收益率曲线）的方向影响，必须视为不相关** → direction=0 magnitude=0
   - 例如：黄金库存、个股涨跌、娱乐体育、公司并购等，通常与利率/债券无直接关系，应输出 0。
2. direction:
   +1 看多债券
   -1 看空债券
3. magnitude: 0~1
4. weight = direction*magnitude

输出：
{"event_type":"xxx","direction":0,"magnitude":0.0,"weight":0.0}
"""

session = requests.Session()
session.proxies = PROXIES


def _is_rates_bonds_relevant(topic: str, question: str) -> bool:
    """
    Gatekeeper to avoid obvious mislabeling (e.g., gold inventory -> nonzero).
    We only allow non-zero scores when the text contains rate/bond/macro triggers.
    """
    q = (question or "").lower()
    t = (topic or "").lower()
    # topic can be a hint, but we still require textual triggers
    allow_topics = {
        "fed_rates",
        "inflation",
        "macro_indicators",
        "gdp",
        "global_rates",
        "treasuries",
        "taxes",
        "trade_war",
    }
    triggers = [
        "fed",
        "fomc",
        "interest rate",
        "rate cut",
        "rate hike",
        "yield",
        "treasury",
        "bond",
        "2y",
        "5y",
        "10y",
        "30y",
        "cpi",
        "ppi",
        "pce",
        "inflation",
        "gdp",
        "recession",
        "unemployment",
        "payroll",
        "jobless",
        "nfp",
    ]
    has_trigger = any(w in q for w in triggers)
    return has_trigger and (t in allow_topics or True)


def _heuristic_event_score(question: str) -> Dict[str, float]:
    q = (question or "").lower()
    hawkish = ["higher", "hike", "increase", "strong", "hot", "surge", "above"]
    dovish = ["lower", "cut", "decrease", "weak", "cool", "below", "miss"]
    strong_words = ["fomc", "cpi", "payroll", "gdp", "yield", "inflation", "fed", "treasury"]

    hk = sum(w in q for w in hawkish)
    dv = sum(w in q for w in dovish)
    signal_strength = min(1.0, 0.15 * (hk + dv) + 0.1 * sum(w in q for w in strong_words))

    if hk > dv:
        direction = -1
    elif dv > hk:
        direction = 1
    else:
        direction = 0

    magnitude = float(signal_strength if direction != 0 else 0.0)
    direction = int(np.sign(direction))
    magnitude = float(np.clip(magnitude, 0.0, 1.0))
    return {"direction": direction, "magnitude": magnitude, "weight": float(direction * magnitude)}

# ================= LLM CALL =================

def call_llm(topic: str, question: str):
    # Hard relevance gate: if clearly unrelated to rates/bonds, force 0.
    if not _is_rates_bonds_relevant(topic, question):
        return {"direction": 0, "magnitude": 0.0, "weight": 0.0}

    if not N1N_API_KEY:
        return _heuristic_event_score(question)

    url = f"{N1N_BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {N1N_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": N1N_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"topic:{topic}\nquestion:{question}"}
        ],
        "temperature": 0,
        "max_tokens": 100
    }

    try:

        r = session.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        r.raise_for_status()

        content = r.json()["choices"][0]["message"]["content"]

        data = json.loads(content)

        direction = int(np.sign(int(data.get("direction", 0))))
        magnitude = float(np.clip(float(data.get("magnitude", 0)), 0.0, 1.0))
        weight = float(direction * magnitude)

        return {
            "direction": direction,
            "magnitude": magnitude,
            "weight": weight
        }

    except Exception:

        return {
            "direction": 0,
            "magnitude": 0,
            "weight": 0
        }


# ================= EMBEDDING =================

def build_embeddings(questions):

    print("[Embedding] loading model...")

    model = SentenceTransformer("BAAI/bge-small-en")

    embeddings = model.encode(
        questions,
        batch_size=128,
        show_progress_bar=True
    )

    return embeddings


# ================= CLUSTER =================

def build_clusters(embeddings):

    print("[Clustering] start")

    cluster = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=0.1
    )

    labels = cluster.fit_predict(embeddings)

    return labels


# ================= LLM BATCH =================

def llm_parallel(topic, questions):

    results = {}

    with ThreadPoolExecutor(MAX_WORKERS) as executor:

        futures = {
            executor.submit(call_llm, topic, q): q
            for q in questions
        }

        for future in as_completed(futures):

            q = futures[future]
            results[q] = future.result()

    return results


# ================= PROCESS TOPIC =================

def build_events_for_topic(topic_dir: Path):

    markets_path = topic_dir / "markets.csv"

    if not markets_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(markets_path)

    topic = topic_dir.name
    df["topic"] = topic

    questions = df["question"].fillna("").tolist()

    print(f"\n[Topic] {topic} | markets={len(df)}")

    # ---------- embedding ----------
    embeddings = build_embeddings(questions)

    # ---------- cluster ----------
    labels = build_clusters(embeddings)

    df["cluster"] = labels

    n_cluster = df["cluster"].nunique()

    print(f"[Cluster] {n_cluster} clusters")

    # ---------- representatives ----------
    rep_df = (
        df.groupby("cluster")
        .first()
        .reset_index()
    )

    rep_questions = rep_df["question"].tolist()

    print(f"[LLM] calling {len(rep_questions)} representatives")

    # ---------- LLM ----------
    llm_results = llm_parallel(topic, rep_questions)

    rep_df["direction"] = rep_df["question"].map(
        lambda q: llm_results[q]["direction"]
    )

    rep_df["magnitude"] = rep_df["question"].map(
        lambda q: llm_results[q]["magnitude"]
    )

    rep_df["weight"] = rep_df["question"].map(
        lambda q: llm_results[q]["weight"]
    )

    # ---------- merge back ----------
    df = df.merge(
        rep_df[
            ["cluster", "direction", "magnitude", "weight"]
        ],
        on="cluster",
        how="left"
    )

    df["event_type"] = topic

    out = df[
        [
            "market_id",
            "topic",
            "question",
            "event_type",
            "direction",
            "magnitude",
            "weight"
        ]
    ].copy()

    # Final safety normalization
    out["direction"] = out["direction"].astype(float).fillna(0).clip(-1, 1).round().astype(int)
    out["magnitude"] = out["magnitude"].astype(float).fillna(0.0).clip(0.0, 1.0)
    out["weight"] = out["direction"] * out["magnitude"]
    return out


def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Windows can lock CSVs if opened (Excel/preview). Write to temp then replace.
    If replacement fails, keep a side-by-side *.new.csv for inspection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    alt = path.with_suffix(".new.csv")
    df.to_csv(tmp, index=False)
    try:
        tmp.replace(path)
    except PermissionError:
        tmp.replace(alt)
        print(f"[WARN] cannot overwrite locked file, wrote -> {alt}")


# ================= MAIN =================

def main():

    out_dir = output_pipeline_dir()

    topics_root = out_dir / "topics"

    topic_dirs = sorted([p for p in topics_root.iterdir() if p.is_dir() and (p / "markets.csv").exists()])

    combined = []

    print(f"[Start] topics={len(topic_dirs)}")

    for td in topic_dirs:

        events = build_events_for_topic(td)

        if events.empty:
            continue

        out_path = td / "events.csv"

        _safe_write_csv(events, out_path)

        print(f"[Saved] {out_path}")

        combined.append(events)

    if combined:

        final = pd.concat(combined, ignore_index=True)

        final_path = out_dir / "events.csv"

        _safe_write_csv(final, final_path)

        print(f"\n[Done] total events={len(final)}")


if __name__ == "__main__":
    main()