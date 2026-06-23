"""Загрузка исторических bookDepth (снапшоты стакана) с data.binance.vision.

Данные только для фьючерсов (UM). Пишутся в DuckDB кеш и Parquet.

Использование:
    python -m data_fetcher book-depth --symbol BTCUSDT --days 7
    python -m data_fetcher book-depth --symbol SOLUSDT --start 2024-01-01 --end 2024-01-31
"""

import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd

from data_fetcher import config

BASE_URL = "https://data.binance.vision"
S3_FUTURES_BOOKDEPTH = "data/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date}.zip"
BOOKDEPTH_COLS = ["timestamp", "percentage", "depth", "notional"]


def _db_path():
    return Path(config.CACHE_DIR) / "binance_vision.db"


def _parquet_path(symbol):
    return Path(config.DATA_DIR) / f"{symbol}_book_depth.parquet"


def read_csv_zip(url, names):
    """Скачать zip-архив. Авто-определение header."""
    for skip in [0, 1]:
        try:
            df = pd.read_csv(url, compression="zip", skiprows=skip, header=None, names=names, dtype=str)
            if df is not None and not df.empty:
                first_val = df.iloc[0, 0]
                if first_val == names[0]:
                    continue
                return df
        except Exception:
            return None
    return None


def download_book_depth(symbol, date_str):
    """Скачать bookDepth за один день (только фьючерсы)."""
    url = f"{BASE_URL}/{S3_FUTURES_BOOKDEPTH.format(symbol=symbol, date=date_str)}"
    df = read_csv_zip(url, BOOKDEPTH_COLS)
    if df is not None and not df.empty:
        df["symbol"] = symbol
        df["date"] = date_str
        df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")
        df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
        df["notional"] = pd.to_numeric(df["notional"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["percentage", "timestamp"])
        df["percentage"] = df["percentage"].astype(int)
        return df[["symbol", "timestamp", "percentage", "depth", "notional", "date"]]
    return None


def fetch_range(symbol, start, end):
    """Загрузить bookDepth за диапазон дат. DuckDB кеш + Parquet экспорт.

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
        CREATE TABLE IF NOT EXISTS book_depth_hist (
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            percentage INTEGER,
            depth DOUBLE,
            notional DOUBLE,
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
            "SELECT COUNT(*) FROM book_depth_hist WHERE symbol=? AND date=?",
            [symbol, date_str]
        ).fetchone()[0]

        if existing > 0:
            days_skipped += 1
            current += timedelta(days=1)
            continue

        df = download_book_depth(symbol, date_str)
        if df is not None and not df.empty:
            conn.execute("INSERT INTO book_depth_hist SELECT * FROM df")
            total_rows += len(df)
            days_fetched += 1
            time.sleep(0.1)
        else:
            days_skipped += 1

        current += timedelta(days=1)

    result = conn.execute(
        "SELECT * FROM book_depth_hist WHERE symbol=? ORDER BY timestamp",
        [symbol]
    ).fetchdf()
    conn.close()

    if not result.empty:
        pq.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(pq, index=False)

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Загрузка bookDepth с data.binance.vision")
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

    print(f"Загрузка bookDepth: {args.symbol} {start.date()} - {end.date()}")
    t0 = time.time()
    df = fetch_range(args.symbol, start, end)
    elapsed = time.time() - t0
    print(f"  Строк: {len(df):,}")
    print(f"  Время: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
