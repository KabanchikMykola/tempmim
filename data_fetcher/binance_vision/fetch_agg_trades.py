"""Загрузка исторических aggTrades (агрегированные сделки) с data.binance.vision.

Поддерживает спот и фьючерсы. Данные пишутся в DuckDB кеш и Parquet.

Использование:
    python -m data_fetcher agg-trades --symbol BTCUSDT --days 7
    python -m data_fetcher agg-trades --symbol SOLUSDT --start 2024-01-01 --end 2024-01-31
"""

import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd

from data_fetcher import config

BASE_URL = "https://data.binance.vision"
S3_SPOT = "data/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date}.zip"
S3_FUTURES = "data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date}.zip"
AGGTRADES_COLS = [
    "agg_trade_id", "price", "qty", "first_trade_id",
    "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match",
]


def _db_path():
    return Path(config.CACHE_DIR) / "binance_vision.db"


def _parquet_path(symbol):
    return Path(config.DATA_DIR) / f"{symbol}_agg_trades.parquet"


def read_csv_zip(url, names):
    """Скачать zip-архив с CSV и прочитать как DataFrame."""
    try:
        return pd.read_csv(url, compression="zip", header=None, names=names, dtype=str)
    except Exception:
        return None


def download_agg_trades(symbol, date_str):
    """Скачать aggTrades за один день (спот и фьючерсы).

    Returns:
        list[pd.DataFrame]: [spot_df, futures_df] или пустой список.
    """
    results = []
    for path_template, source in [
        (S3_SPOT, "spot"),
        (S3_FUTURES, "futures"),
    ]:
        url = f"{BASE_URL}/{path_template.format(symbol=symbol, date=date_str)}"
        df = read_csv_zip(url, AGGTRADES_COLS)
        if df is not None and not df.empty:
            df["symbol"] = symbol
            df["source"] = source
            df["date"] = date_str
            df["agg_trade_id"] = pd.to_numeric(df["agg_trade_id"], errors="coerce")
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
            df["first_trade_id"] = pd.to_numeric(df["first_trade_id"], errors="coerce")
            df["last_trade_id"] = pd.to_numeric(df["last_trade_id"], errors="coerce")
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df["is_buyer_maker"] = df["is_buyer_maker"].map({"True": True, "False": False})
            results.append(df[[
                "symbol", "source", "agg_trade_id", "price", "qty",
                "first_trade_id", "last_trade_id", "timestamp",
                "is_buyer_maker", "date",
            ]])
    return results


def fetch_range(symbol, start, end):
    """Загрузить aggTrades за диапазон дат. DuckDB кеш + Parquet экспорт.

    Args:
        symbol: Тикер (BTCUSDT, SOLUSDT...)
        start: datetime начала
        end: datetime конца

    Returns:
        pd.DataFrame
    """
    db = _db_path()
    pq = _parquet_path(symbol)
    db.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agg_trades_hist (
            symbol VARCHAR,
            source VARCHAR,
            agg_trade_id BIGINT,
            price DOUBLE,
            qty DOUBLE,
            first_trade_id BIGINT,
            last_trade_id BIGINT,
            timestamp BIGINT,
            is_buyer_maker BOOLEAN,
            date VARCHAR
        )
    """)

    current = start
    total_rows = 0
    days_fetched = 0
    days_skipped = 0

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        existing = conn.execute(
            "SELECT COUNT(*) FROM agg_trades_hist WHERE symbol=? AND date=?",
            [symbol, date_str]
        ).fetchone()[0]

        if existing > 0:
            days_skipped += 1
            current += timedelta(days=1)
            continue

        dfs = download_agg_trades(symbol, date_str)
        if dfs:
            for df in dfs:
                clean = df.dropna(subset=["timestamp"])
                if not clean.empty:
                    conn.execute("INSERT INTO agg_trades_hist SELECT * FROM clean")
                    total_rows += len(clean)
            days_fetched += 1
            time.sleep(0.1)
        else:
            days_skipped += 1

        current += timedelta(days=1)

    df = conn.execute(
        "SELECT * FROM agg_trades_hist WHERE symbol=? ORDER BY timestamp",
        [symbol]
    ).fetchdf()
    conn.close()

    if not df.empty:
        pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq, index=False)

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Загрузка aggTrades с data.binance.vision")
    parser.add_argument("--symbol", default="SOLUSDT", help="Тикер")
    parser.add_argument("--days", type=int, help="Последние N дней")
    parser.add_argument("--start", help="Дата начала (YYYY-MM-DD)")
    parser.add_argument("--end", help="Дата конца (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.days:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)
    elif args.start and args.end:
        start_ = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_ = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start, end = start_, end_
    else:
        parser.error("Укажите --days или --start/--end")
        return

    print(f"Загрузка aggTrades: {args.symbol} {start.date()} - {end.date()}")
    t0 = time.time()
    df = fetch_range(args.symbol, start, end)
    elapsed = time.time() - t0
    print(f"  Строк: {len(df):,}")
    print(f"  Время: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
