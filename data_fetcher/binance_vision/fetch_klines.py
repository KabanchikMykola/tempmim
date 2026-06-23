"""Загрузка OHLCV klines с data.binance.vision.

Трёхуровневый кеш:
1. Parquet файл (data/{symbol}_{interval}_{source}.parquet)
2. DuckDB кеш (data/cache/binance_vision.db)
3. Скачивание с data.binance.vision (S3 архивы)

Использование:
    python -m data_fetcher ohlcv vision --symbol BTCUSDT --interval 1h
    python -m data_fetcher ohlcv vision --symbol SOLUSDT --interval 5m --perp
    python -m data_fetcher ohlcv vision --symbol BTCUSDT ETHUSDT --years 2
"""

import io
import time
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import duckdb

from data_fetcher import config


def _db_path():
    return Path(config.CACHE_DIR) / "binance_vision.db"


def _parquet_path(symbol, interval, source):
    return Path(config.DATA_DIR) / f"{symbol}_{interval}_{source}.parquet"


def _safe_name(symbol):
    return symbol.replace("/", "_").replace(":", "_")


def download_monthly(symbol, interval, year, month, perp=False):
    """Скачать один месячный архив klines с data.binance.vision."""
    base = "futures/um" if perp else "spot"
    url = (f"https://data.binance.vision/data/{base}/monthly/klines/"
           f"{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip")
    try:
        resp = requests.get(url, timeout=config.VISION_TIMEOUT)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    content = f.read().decode("utf-8")
                    lines = content.strip().split("\n")
                    if lines[0].startswith("open_time"):
                        lines = lines[1:]
                    if not lines:
                        return pd.DataFrame()
                    data = [line.split(",") for line in lines]
                    df = pd.DataFrame(data, columns=[
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "count",
                        "taker_buy_base", "taker_buy_quote", "ignore"
                    ])
                    for col in ["open_time", "close_time", "count"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                                "taker_buy_base", "taker_buy_quote"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    return df
    except Exception:
        pass
    return pd.DataFrame()


def download_daily(symbol, interval, date_str, perp=False):
    """Скачать один дневной архив klines."""
    base = "futures/um" if perp else "spot"
    url = (f"https://data.binance.vision/data/{base}/daily/klines/"
           f"{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip")
    try:
        resp = requests.get(url, timeout=config.VISION_TIMEOUT)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    content = f.read().decode("utf-8")
                    lines = content.strip().split("\n")
                    if lines[0].startswith("open_time"):
                        lines = lines[1:]
                    if not lines:
                        return pd.DataFrame()
                    data = [line.split(",") for line in lines]
                    df = pd.DataFrame(data, columns=[
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "count",
                        "taker_buy_base", "taker_buy_quote", "ignore"
                    ])
                    for col in ["open_time", "close_time", "count"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                                "taker_buy_base", "taker_buy_quote"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    return df
    except Exception:
        pass
    return pd.DataFrame()


def _ensure_parquet(symbol, interval, source):
    """Экспортировать данные из DuckDB в Parquet."""
    db = _db_path()
    pq = _parquet_path(symbol, interval, source)
    db.parent.mkdir(parents=True, exist_ok=True)
    pq.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db))
    existing = conn.execute(
        "SELECT COUNT(*) FROM klines WHERE symbol=? AND source=? AND interval=?",
        [symbol, source, interval]
    ).fetchone()[0]
    if existing > 0:
        df = conn.execute(
            "SELECT * FROM klines WHERE symbol=? AND source=? AND interval=? ORDER BY ts",
            [symbol, source, interval]
        ).fetchdf()
        df.to_parquet(pq, index=False)
    conn.close()
    return pq


def fetch_symbol(symbol, interval="1h", years=3, perp=False, export_parquet=True):
    """Загрузить klines для одного символа.

    Сначала проверяет Parquet, потом DuckDB кеш, потом качает с S3.

    Args:
        symbol: Тикер (BTCUSDT, ETHUSDT...)
        interval: Таймфрейм (1m, 5m, 15m, 1h, 4h, 1d)
        years: Сколько лет качать (если нет в кеше)
        perp: True = перпетуалы, False = спот
        export_parquet: Экспортировать в Parquet после загрузки

    Returns:
        pd.DataFrame с колонками open_time, open, high, low, close, volume, ...
    """
    source = "perp" if perp else "spot"
    pq = _parquet_path(symbol, interval, source)
    db = _db_path()

    # 1. Проверить Parquet
    if pq.exists():
        df = pd.read_parquet(pq)
        if len(df) > 1000:
            return df

    # 2. Проверить DuckDB кеш
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS klines (
            symbol VARCHAR, source VARCHAR, interval VARCHAR,
            open_time BIGINT, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume DOUBLE, close_time BIGINT, quote_volume DOUBLE,
            count INTEGER, taker_buy_base DOUBLE, taker_buy_quote DOUBLE, ts BIGINT
        )
    """)

    existing = conn.execute(
        "SELECT COUNT(*) FROM klines WHERE symbol=? AND source=? AND interval=?",
        [symbol, source, interval]
    ).fetchone()[0]

    if existing > 1000:
        df = conn.execute(
            "SELECT * FROM klines WHERE symbol=? AND source=? AND interval=? ORDER BY ts",
            [symbol, source, interval]
        ).fetchdf()
        conn.close()
        if export_parquet:
            df.to_parquet(pq, index=False)
        return df

    # 3. Скачать с S3 и записать в DuckDB
    now = datetime.now()
    tasks = []
    for year in range(now.year - years, now.year + 1):
        for month in range(1, 13):
            if year == now.year and month > now.month:
                break
            tasks.append((year, month))

    def _fetch_one(args):
        yr, mo = args
        df = download_monthly(symbol, interval, yr, mo, perp)
        if df.empty:
            return pd.DataFrame()
        df["symbol"] = symbol
        df["source"] = source
        df["interval"] = interval
        df["ts"] = df["open_time"] // 1000 if df["open_time"].max() > 1e14 else df["open_time"]
        return df

    total = 0
    with ThreadPoolExecutor(max_workers=config.VISION_WORKERS) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in tasks}
        for f in as_completed(futures):
            df = f.result()
            if not df.empty:
                conn.execute("""
                    INSERT INTO klines
                    SELECT symbol, source, interval, open_time, open, high, low, close,
                           volume, close_time, quote_volume, count, taker_buy_base,
                           taker_buy_quote, ts FROM df
                """)
                total += len(df)

    df = conn.execute(
        "SELECT * FROM klines WHERE symbol=? AND source=? AND interval=? ORDER BY ts",
        [symbol, source, interval]
    ).fetchdf()
    conn.close()

    if export_parquet and not df.empty:
        pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq, index=False)

    return df


def export_all():
    """Экспортировать все данные из DuckDB в Parquet."""
    db = _db_path()
    if not db.exists():
        return
    conn = duckdb.connect(str(db))
    rows = conn.execute("SELECT DISTINCT symbol, source, interval FROM klines").fetchall()
    conn.close()
    for symbol, source, interval in rows:
        pq = _parquet_path(symbol, interval, source)
        print(f"  Экспорт {symbol} {source} {interval} -> {pq}")
        _ensure_parquet(symbol, interval, source)


def summary():
    """Вывести сводку по данным в DuckDB."""
    db = _db_path()
    if not db.exists():
        print("  Кеш пуст")
        return
    conn = duckdb.connect(str(db), read_only=True)
    result = conn.execute("""
        SELECT symbol, source, interval, COUNT(*), MIN(ts), MAX(ts)
        FROM klines
        GROUP BY symbol, source, interval
        ORDER BY symbol, source, interval
    """).fetchall()
    conn.close()

    for symbol, source, interval, cnt, t_min, t_max in result:
        def _fmt(ts):
            if not ts or ts <= 0:
                return "?"
            try:
                return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            except (OSError, ValueError):
                return "?"
        d_min = _fmt(t_min)
        d_max = _fmt(t_max)
        print(f"    {symbol:>12} {source:>5} {interval:>4}: {cnt:>8,} баров ({d_min} - {d_max})")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Загрузка klines с data.binance.vision")
    parser.add_argument("--symbol", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Тикеры")
    parser.add_argument("--interval", default="1h", help="Таймфрейм")
    parser.add_argument("--years", type=int, default=3, help="Сколько лет")
    parser.add_argument("--perp", action="store_true", help="Перпетуалы вместо спота")
    parser.add_argument("--export-all", action="store_true", help="Экспорт всего кеша в Parquet")
    parser.add_argument("--summary", action="store_true", help="Сводка по кешу")
    args = parser.parse_args()

    if args.export_all:
        export_all()
        return

    if args.summary:
        summary()
        return

    t0 = time.time()
    for symbol in args.symbol:
        df = fetch_symbol(symbol, args.interval, args.years, args.perp, export_parquet=True)
        print(f"  {symbol} ({'perp' if args.perp else 'spot'}): {len(df):,} баров")

    summary()
    print(f"\n  Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
