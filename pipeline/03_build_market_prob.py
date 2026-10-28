from __future__ import annotations

# 允许直接用 `python pipeline/xx.py` 运行（确保项目根目录在 sys.path）
import sys
from pathlib import Path

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

"""
Step 3: 交易数据 → 日频概率序列（按主题分别处理）

输入:
    data/polymarket/trades/*.parquet
    output/pipeline/topics/{topic}/markets.csv  （由 Step 1 生成）

输出:
    output/pipeline/topics/{topic}/market_prob/polymarket_{market_id}.parquet

字段:
    ts, prob, volume, trades

特点:
    • 按主题分别处理，每个主题只处理该主题的市场
    • DuckDB vectorized aggregation
    • 自动解析 timestamp
    • 单次扫描 parquet
    • 默认日频（day）
"""

import argparse
import shutil
from typing import Iterable, Set

import duckdb
import pandas as pd

from pipeline.utils_paths import data_dir, output_topic_dir, output_topic_market_prob_dir
from pipeline.utils_topics import discover_poly_topic_files
from pipeline.wheels_market_data import build_prob_snapshot_from_markets, ensure_polymarket_markets


def get_topic_market_ids(topic: str) -> Set[str]:
    """从 topics/{topic}/markets.csv 读取该主题的市场 ID 列表"""
    markets_path = output_topic_dir(topic) / "markets.csv"
    if not markets_path.exists():
        return set()
    
    df = pd.read_csv(markets_path)
    if "market_id" not in df.columns:
        return set()
    
    return set(df["market_id"].astype(str).unique())


def _duckdb_bar_freq(freq: str) -> str:
    f = freq.strip().lower()
    mapping = {
        "1d": "day",
        "1": "day",
        "day": "day",
        "daily": "day",
        "60min": "hour",
        "1h": "hour",
        "30min": "30min",
    }
    if f not in mapping:
        raise ValueError(f"Unsupported bar frequency: {freq}")
    return mapping[f]


def _freq_folder(freq: str) -> str:
    f = freq.strip().lower()
    if f in {"1d", "1", "day", "daily"}:
        return "1d"
    if f in {"60min", "1h"}:
        return "60min"
    if f == "30min":
        return "30min"
    raise ValueError(f"Unsupported bar frequency: {freq}")


def build_topic_market_prob(topic: str, market_ids: Set[str], bar_freqs: Iterable[str]) -> None:
    """
    为指定主题构建市场概率序列（默认日频）

    Args:
        topic: 主题名称
        market_ids: 该主题的市场 ID 集合
        bar_freq: 时间频率（day）
    """
    if not market_ids:
        print(f"[03] topic={topic:<16} no markets, skipped")
        return
    
    out_dir = output_topic_market_prob_dir(topic)
    trades_dir = data_dir() / "polymarket" / "trades"

    if not trades_dir.exists():
        # Fallback wheel: build a one-point probability snapshot from Gamma market metadata.
        markets_df, source = ensure_polymarket_markets(max_markets=None)
        topic_markets = markets_df[markets_df["id"].astype(str).isin(market_ids)].copy()
        snap = build_prob_snapshot_from_markets(topic_markets)
        if snap.empty:
            print(f"[03] topic={topic:<16} no local trades and no gamma snapshot data")
            return
        saved = 0
        for market_id, group in snap.groupby("market_id"):
            output_path = out_dir / f"polymarket_{market_id}.parquet"
            group[["market_id", "ts", "prob", "volume", "trades"]].to_parquet(output_path, index=False)
            saved += 1
        print(
            f"[03] topic={topic:<16} no local trades -> snapshot source={source} saved={saved:,} files"
        )
        return
    
    con = duckdb.connect()
    # Avoid oversubscription on Windows; high thread counts can stall parquet I/O.
    con.execute("PRAGMA threads=8")
    
    # 使用参数化查询避免 SQL 注入和超长字符串
    market_ids_list = list(market_ids)
    
    print(f"[03] topic={topic:<16} markets={len(market_ids):,} processing...")
    
    # 创建临时表存储 market_ids（更安全且高效）
    con.execute("CREATE TEMP TABLE topic_market_ids (market_id VARCHAR)")
    con.executemany(
        "INSERT INTO topic_market_ids VALUES (?)",
        [(mid,) for mid in market_ids_list]
    )
    
    for freq in bar_freqs:
        freq_bucket = _duckdb_bar_freq(freq)
        freq_dir_name = _freq_folder(freq)
        temp_file = out_dir / f"_temp_all_{topic}_{freq_dir_name}.parquet"
        if freq_bucket == "30min":
            ts_expr = "time_bucket(INTERVAL '30 minutes', CAST(t.timestamp AS TIMESTAMP))"
        else:
            ts_expr = f"date_trunc('{freq_bucket}', CAST(t.timestamp AS TIMESTAMP))"

        query = f"""
        COPY (
            SELECT
                t.market_id,
                {ts_expr} AS ts,
                SUM(t.price * ABS(t.usd_amount)) / SUM(ABS(t.usd_amount)) AS prob,
                SUM(ABS(t.usd_amount)) AS volume,
                COUNT(*) AS trades
            FROM read_parquet('{trades_dir}/*.parquet') AS t
            INNER JOIN topic_market_ids AS m ON t.market_id = m.market_id
            WHERE
                t.price BETWEEN 0 AND 1
                AND t.usd_amount IS NOT NULL
            GROUP BY t.market_id, ts
            ORDER BY t.market_id, ts
        )
        TO '{temp_file}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """

        try:
            con.execute(query)
        except Exception as e:
            print(f"[03] topic={topic:<16} query failed freq={freq}: {e}")
            continue

        if not temp_file.exists():
            print(f"[03] topic={topic:<16} temp file not created freq={freq}, skipped")
            continue

        try:
            # Use DuckDB split instead of loading a giant dataframe into pandas.
            market_rows = con.execute(
                f"SELECT DISTINCT market_id FROM read_parquet('{temp_file}') ORDER BY market_id"
            ).fetchall()
            market_ids_in_temp = [str(r[0]) for r in market_rows if r and r[0] is not None]
            if not market_ids_in_temp:
                print(f"[03] topic={topic:<16} no data in temp file freq={freq}, skipped")
                temp_file.unlink(missing_ok=True)
                continue

            saved_count = 0
            freq_out_dir = out_dir if freq_dir_name == "1d" else (out_dir / freq_dir_name)
            freq_out_dir.mkdir(parents=True, exist_ok=True)
            for idx, market_id in enumerate(market_ids_in_temp, 1):
                output_path = freq_out_dir / f"polymarket_{market_id}.parquet"
                output_path_sql = str(output_path).replace("'", "''")
                con.execute(
                    f"""
                    COPY (
                        SELECT market_id, ts, prob, volume, trades
                        FROM read_parquet('{temp_file}')
                        WHERE market_id = ?
                        ORDER BY ts
                    )
                    TO '{output_path_sql}'
                    (FORMAT PARQUET, COMPRESSION ZSTD);
                    """,
                    [market_id],
                )
                saved_count += 1
                if idx % 500 == 0:
                    print(
                        f"[03] topic={topic:<16} freq={freq:<6} split_progress={idx:,}/{len(market_ids_in_temp):,}"
                    )

            temp_file.unlink(missing_ok=True)
            print(f"[03] topic={topic:<16} freq={freq:<6} saved={saved_count:,} files -> {freq_out_dir}")
        except Exception as e:
            print(f"[03] topic={topic:<16} split failed freq={freq}: {e}")
            temp_file.unlink(missing_ok=True)

    con.execute("DROP TABLE IF EXISTS topic_market_ids")
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step3: build market probability time series (daily, per topic)"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="process only this topic (default: all topics)"
    )
    parser.add_argument(
        "--bar-freqs",
        type=str,
        default="30min,60min,1D",
        help="comma-separated bar frequencies, e.g. 30min,60min,1D",
    )

    args = parser.parse_args()
    
    topic_files = discover_poly_topic_files()
    if not topic_files:
        print("[03] no topic files found. Run Step 0 & 1 first.")
        return
    
    # 如果指定了单个 topic，只处理那个
    if args.topic:
        topic_files = [tf for tf in topic_files if tf.topic == args.topic]
        if not topic_files:
            print(f"[03] topic '{args.topic}' not found")
            return
    
    bar_freqs = [x.strip() for x in args.bar_freqs.split(",") if x.strip()]
    print(f"[03] processing {len(topic_files)} topics... freqs={bar_freqs}")
    
    for tf in topic_files:
        market_ids = get_topic_market_ids(tf.topic)
        build_topic_market_prob(tf.topic, market_ids, bar_freqs)
    
    print("[03] finished all topics")


if __name__ == "__main__":
    main()
