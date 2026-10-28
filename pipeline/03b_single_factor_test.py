from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import duckdb
import pandas as pd

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from pipeline.utils_paths import output_pipeline_dir, output_topic_dir, output_topic_market_prob_dir, zt_1min_path
from pipeline.utils_topics import discover_poly_topic_files

HORIZON_DAYS = 5
MIN_SAMPLE_TOPIC = 60
MIN_SAMPLE_MARKET = 25
PASS_IC = 0.05
PASS_SHARPE = 0.6
PASS_WIN_RATE = 0.55
PASS_CUM_RET = 0.0


def _build_price_forward_returns_daily(horizon_days: int) -> pd.DataFrame:
    zt_path = zt_1min_path("ZT")
    if zt_path is None:
        raise FileNotFoundError("ZT price file not found under data/macro or src/analysis")

    df = pd.read_csv(zt_path, header=None)
    df.columns = ["ts", "c2", "c3", "c4", "open", "high", "low", "close", "volume", "symbol"]
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["ts", "close"]).sort_values("ts")
    df["ts"] = df["ts"].dt.tz_convert(None)
    df["day"] = df["ts"].dt.floor("D")

    daily = df.groupby("day", as_index=False)["close"].last()
    daily["close_fwd"] = daily["close"].shift(-horizon_days)
    daily["fwd_ret"] = (daily["close_fwd"] / daily["close"]) - 1.0
    out = daily.dropna(subset=["fwd_ret"]).rename(columns={"day": "ts"})[["ts", "fwd_ret"]]
    print(f"[03b] price_ret rows={len(out):,}")
    return out


def _load_events(topic: str) -> pd.DataFrame:
    p = output_topic_dir(topic) / "events.csv"
    if not p.exists():
        return pd.DataFrame(columns=["market_id", "direction", "magnitude"])

    d = pd.read_csv(p)
    if d.empty or "market_id" not in d.columns:
        return pd.DataFrame(columns=["market_id", "direction", "magnitude"])

    d["market_id"] = d["market_id"].astype(str)
    d["direction"] = pd.to_numeric(d.get("direction", 1), errors="coerce").fillna(1).clip(-1, 1)
    d["magnitude"] = pd.to_numeric(d.get("magnitude", 1.0), errors="coerce").fillna(1.0).clip(0, 1)
    return d[["market_id", "direction", "magnitude"]].drop_duplicates("market_id")


def _has_column(path: Path, col: str) -> bool:
    try:
        pd.read_parquet(path, columns=[col])
        return True
    except Exception:
        return False


def _create_factor_table(con: duckdb.DuckDBPyConnection, topic: str, prob_dir: Path) -> bool:
    files = sorted(prob_dir.glob("polymarket_*.parquet"))
    if not files:
        print(f"[03b] topic={topic:<20} no market_prob files")
        return False

    con.register("events_df", _load_events(topic))
    con.execute("DROP TABLE IF EXISTS factor_obs")
    con.execute(
        "CREATE TEMP TABLE factor_obs (ts TIMESTAMP, market_id VARCHAR, prob DOUBLE, volume DOUBLE, dprob DOUBLE, dlogit DOUBLE, factor_jump DOUBLE, factor_reversal DOUBLE, factor_vol_jump DOUBLE, fwd_ret DOUBLE)"
    )

    for p in files:
        p_sql = str(p.as_posix())
        mid = p.stem.replace("polymarket_", "")
        mid_expr = "CAST(src.market_id AS VARCHAR)" if _has_column(p, "market_id") else f"'{mid}'"
        con.execute(
            f"""
            INSERT INTO factor_obs
            WITH base AS (
                SELECT {mid_expr} AS market_id, date_trunc('day', CAST(src.ts AS TIMESTAMP)) AS ts, CAST(src.prob AS DOUBLE) AS prob, COALESCE(CAST(src.volume AS DOUBLE), 0.0) AS volume
                FROM read_parquet('{p_sql}') src
                WHERE src.prob IS NOT NULL AND src.prob > 0.001 AND src.prob < 0.999
            ), day_bar AS (
                SELECT market_id, ts, AVG(prob) AS prob, SUM(volume) AS volume FROM base GROUP BY market_id, ts
            ), seq AS (
                SELECT market_id, ts, prob, volume, LAG(prob) OVER (PARTITION BY market_id ORDER BY ts) AS prev_prob FROM day_bar
            ), fac AS (
                SELECT s.market_id, s.ts, s.prob, s.volume, (s.prob - s.prev_prob) AS dprob, (LN(s.prob / (1 - s.prob)) - LN(s.prev_prob / (1 - s.prev_prob))) AS dlogit
                FROM seq s WHERE s.prev_prob IS NOT NULL
            ), fac2 AS (
                SELECT f.market_id, f.ts, f.prob, f.volume, f.dprob, f.dlogit, COALESCE(e.direction, 1.0) AS direction, COALESCE(e.magnitude, 1.0) AS magnitude
                FROM fac f LEFT JOIN events_df e USING (market_id)
            )
            SELECT f2.ts, f2.market_id, f2.prob, f2.volume, f2.dprob, f2.dlogit,
                   (f2.direction * f2.magnitude * f2.dlogit) AS factor_jump,
                   (f2.direction * f2.magnitude * (0.5 - f2.prob) * f2.dprob) AS factor_reversal,
                   (f2.direction * f2.magnitude * f2.dlogit * LN(1 + GREATEST(f2.volume, 0))) AS factor_vol_jump,
                   p.fwd_ret
            FROM fac2 f2 INNER JOIN price_ret p ON f2.ts = p.ts WHERE p.fwd_ret IS NOT NULL
            """
        )

    n = con.execute("SELECT COUNT(*) FROM factor_obs").fetchone()[0]
    print(f"[03b] topic={topic:<20} factor_obs={n:,}")
    return True


def _topic_stats_sql(factor_col: str) -> str:
    return f"""
    WITH t AS (SELECT {factor_col} AS x, fwd_ret AS y FROM factor_obs WHERE {factor_col} IS NOT NULL AND fwd_ret IS NOT NULL),
    r AS (SELECT x,y,percent_rank() OVER (ORDER BY x) AS rx, percent_rank() OVER (ORDER BY y) AS ry FROM t),
    q AS (
      SELECT COUNT(*) AS n_obs, corr(rx, ry) AS ic_spearman, corr(x,y) AS ic_pearson,
             AVG(CASE WHEN sign(x)=sign(y) THEN 1.0 ELSE 0.0 END) AS hit_rate,
             AVG(x*y) AS avg_factor_ret, stddev_samp(x*y) AS std_factor_ret,
             AVG(CASE WHEN x*y>0 THEN 1.0 ELSE 0.0 END) AS factor_win_rate,
             EXP(SUM(LN(GREATEST(1 + (x*y), 1e-9)))) - 1 AS cum_factor_ret,
             quantile_cont(x, 0.2) AS q20, quantile_cont(x, 0.8) AS q80
      FROM r
    )
    SELECT '{factor_col}' AS factor, q.n_obs, q.ic_spearman, q.ic_pearson,
           ((SELECT AVG(y) FROM t WHERE x>=q.q80) - (SELECT AVG(y) FROM t WHERE x<=q.q20)) AS long_short_spread,
           q.hit_rate, q.avg_factor_ret, q.std_factor_ret,
           (q.avg_factor_ret/(q.std_factor_ret+1e-12))*SQRT(252.0/{HORIZON_DAYS}) AS factor_sharpe,
           q.factor_win_rate, q.cum_factor_ret
    FROM q
    """


def _market_scores_sql(factor_col: str) -> str:
    return f"""
    WITH t AS (SELECT market_id, {factor_col} AS x, fwd_ret AS y FROM factor_obs WHERE {factor_col} IS NOT NULL AND fwd_ret IS NOT NULL),
    r AS (SELECT market_id,x,y,percent_rank() OVER (PARTITION BY market_id ORDER BY x) AS rx, percent_rank() OVER (PARTITION BY market_id ORDER BY y) AS ry FROM t)
    SELECT market_id, '{factor_col}' AS factor, COUNT(*) AS n_obs, corr(rx, ry) AS ic_spearman,
           AVG(x*y) AS avg_factor_ret, stddev_samp(x*y) AS std_factor_ret,
           (AVG(x*y)/(stddev_samp(x*y)+1e-12))*SQRT(252.0/{HORIZON_DAYS}) AS factor_sharpe,
           AVG(CASE WHEN x*y>0 THEN 1.0 ELSE 0.0 END) AS factor_win_rate,
           EXP(SUM(LN(GREATEST(1 + (x*y), 1e-9)))) - 1 AS cum_factor_ret
    FROM r GROUP BY market_id
    """


def process_topic(con: duckdb.DuckDBPyConnection, topic: str, sample_rows: int) -> pd.DataFrame:
    out_dir = output_topic_dir(topic) / "single_factor"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not _create_factor_table(con, topic, output_topic_market_prob_dir(topic)):
        return pd.DataFrame()

    n_obs = con.execute("SELECT COUNT(*) FROM factor_obs").fetchone()[0]
    if n_obs < MIN_SAMPLE_TOPIC:
        print(f"[03b] topic={topic:<20} too few obs: {n_obs:,}")
        return pd.DataFrame()

    factors = ["dprob", "dlogit", "factor_jump", "factor_reversal", "factor_vol_jump"]
    topic_df = pd.concat([con.execute(_topic_stats_sql(c)).df() for c in factors], ignore_index=True)
    topic_df.insert(0, "topic", topic)
    topic_df["pass_topic"] = (
        (topic_df["n_obs"] >= MIN_SAMPLE_TOPIC)
        & (topic_df["ic_spearman"] >= PASS_IC)
        & (topic_df["factor_sharpe"] >= PASS_SHARPE)
        & (topic_df["factor_win_rate"] >= PASS_WIN_RATE)
        & (topic_df["cum_factor_ret"] > PASS_CUM_RET)
    )
    topic_df.to_csv(out_dir / "factor_scores.csv", index=False)

    mk = pd.concat([con.execute(_market_scores_sql("factor_jump")).df(), con.execute(_market_scores_sql("factor_reversal")).df()], ignore_index=True)
    mk["topic"] = topic
    mk["pass_market"] = (
        (mk["n_obs"] >= MIN_SAMPLE_MARKET)
        & (mk["ic_spearman"] >= PASS_IC)
        & (mk["factor_sharpe"] >= PASS_SHARPE)
        & (mk["factor_win_rate"] >= PASS_WIN_RATE)
        & (mk["cum_factor_ret"] > PASS_CUM_RET)
    )
    mk_best = mk.sort_values(["market_id", "factor_sharpe"], ascending=[True, False]).drop_duplicates("market_id")
    mk_best.to_csv(out_dir / "market_scores.csv", index=False)
    eligible = mk_best[mk_best["pass_market"]].copy()
    eligible[["market_id"]].to_csv(out_dir / "eligible_markets.csv", index=False)

    if not eligible.empty:
        con.register("eligible_df", eligible[["market_id"]])
        thresholds = con.execute(
            """
            SELECT * FROM (
                SELECT 'prob_jump_threshold' AS key, GREATEST(0.01, quantile_cont(abs(dprob), 0.80)) AS value FROM factor_obs o INNER JOIN eligible_df e ON o.market_id=e.market_id
                UNION ALL
                SELECT 'reversal_min_distance' AS key, GREATEST(0.01, quantile_cont(abs(dprob), 0.70)) AS value FROM factor_obs o INNER JOIN eligible_df e ON o.market_id=e.market_id
            )
            """
        ).df()
    else:
        thresholds = pd.DataFrame([{"key": "prob_jump_threshold", "value": 0.03}, {"key": "reversal_min_distance", "value": 0.02}])

    thresholds.to_csv(out_dir / "thresholds.csv", index=False)
    con.execute(f"""COPY (SELECT * FROM factor_obs USING SAMPLE reservoir({sample_rows} ROWS)) TO '{str((out_dir / 'factor_observations.parquet').as_posix())}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    print(f"[03b] topic={topic:<20} eligible_markets={len(eligible):,}")
    return topic_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Step03b daily strict filter")
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--sample-rows", type=int, default=60000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    topic_files = discover_poly_topic_files()
    if args.topic:
        topic_files = [t for t in topic_files if t.topic == args.topic]

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")
    con.register("price_ret", _build_price_forward_returns_daily(HORIZON_DAYS))

    outs: List[pd.DataFrame] = []
    for tf in topic_files:
        d = process_topic(con, tf.topic, args.sample_rows)
        if not d.empty:
            outs.append(d)
    con.close()

    if outs:
        summary = pd.concat(outs, ignore_index=True)
        summary.to_csv(output_pipeline_dir() / "single_factor_summary.csv", index=False)


if __name__ == "__main__":
    main()
