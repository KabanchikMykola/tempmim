"""Загрузка исторических funding rate с data.binance.vision.

Фандинг: месячные архивы (fundingRate).
Перп klines — в fetch_klines.py (тот же скрипт что и спот).

Кеш: HuggingFace Bucket → Parquet (data/) → DuckDB (data/cache/) → S3 (data.binance.vision).
Если BUCKET_ID задан в config.py — синхронизация с HuggingFace Bucket.

Использование:
    python -m data_fetcher funding --symbol BTCUSDT
    python -m data_fetcher funding --symbol ETHUSDT --years 2
"""

import io
import zipfile
import time
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import duckdb

from data_fetcher import config


def _db_path():
    return Path(config.CACHE_DIR) / "binance_vision.db"


def _parquet_path_funding(symbol):
    return Path(config.DATA_DIR) / f"{symbol}_funding.parquet"


def _bucket_parquet_uri(symbol):
    return f"hf://buckets/{config.BUCKET_ID}/data/{symbol}_funding.parquet"


def _sync_from_bucket(symbol):
    """Скачать parquet из bucket в локальный кеш."""
    if not config.BUCKET_ID:
        return None
    pq = _parquet_path_funding(symbol)
    if pq.exists():
        return None
    try:
        uri = _bucket_parquet_uri(symbol)
        df = pd.read_parquet(uri)
        if df.empty:
            return None
        pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq, index=False)
        return df
    except Exception as e:
        print(f"  bucket sync error: {e}")
        return None


def _sync_to_bucket(symbol, df=None):
    """Загрузить локальный parquet в bucket."""
    if not config.BUCKET_ID:
        return
    try:
        if df is None:
            pq = _parquet_path_funding(symbol)
            if not pq.exists():
                return
            df = pd.read_parquet(pq)
        uri = _bucket_parquet_uri(symbol)
        df.to_parquet(uri, index=False)
    except Exception as e:
        print(f"  bucket sync error: {e}")


def download_funding_monthly(symbol, year, month):
    """Скачать месячный архив funding rate."""
    url = (f"https://data.binance.vision/data/futures/um/monthly/fundingRate/"
           f"{symbol}/{symbol}-fundingRate-{year}-{month:02d}.zip")
    try:
        resp = requests.get(url, timeout=config.VISION_TIMEOUT)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    df = pd.read_csv(f)
                    df["symbol"] = symbol
                    return df
    except Exception:
        pass
    return pd.DataFrame()


def _db_max_ts(conn, symbol):
    result = conn.execute(
        "SELECT COALESCE(MAX(ts), 0) FROM funding_rates WHERE symbol=?", [symbol]
    ).fetchone()
    return result[0] if result else 0


def fetch_funding(symbol, years=3, export_parquet=True):
    """Загрузить funding rate для символа.

    Кеш: HuggingFace Bucket → Parquet (data/) → DuckDB (data/cache/) → S3 (data.binance.vision).

    Args:
        symbol: Тикер (BTCUSDT, ETHUSDT...)
        years: Сколько лет
        export_parquet: Экспорт в Parquet

    Returns:
        pd.DataFrame с колонками symbol, calc_time, funding_interval_hours, last_funding_rate
    """
    pq = _parquet_path_funding(symbol)
    db = _db_path()
    all_warnings = []

    # ── 1. Загрузить из bucket в локальный кеш ──
    if config.BUCKET_ID and not pq.exists():
        bucket_df = _sync_from_bucket(symbol)
        if bucket_df is not None and len(bucket_df) > 1000:
            return bucket_df

    # ── 2. Проверить локальный Parquet ──
    if pq.exists():
        df = pd.read_parquet(pq)
        if len(df) > 1000:
            return df

    # ── 3. DuckDB ──
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS funding_rates (
            symbol VARCHAR, calc_time BIGINT, funding_interval_hours INTEGER,
            last_funding_rate DOUBLE, ts BIGINT
        )
    """)

    existing = conn.execute(
        "SELECT COUNT(*) FROM funding_rates WHERE symbol=?", [symbol]
    ).fetchone()[0]

    if existing > 1000:
        df = conn.execute(
            "SELECT * FROM funding_rates WHERE symbol=? ORDER BY ts", [symbol]
        ).fetchdf()
        conn.close()
        return df

    # ── 4. Скачать с S3 ──
    now = datetime.now()
    tasks = []
    for year in range(now.year - years, now.year + 1):
        for month in range(1, 13):
            if year == now.year and month > now.month:
                break
            tasks.append((year, month))

    max_ts = _db_max_ts(conn, symbol)
    total = 0

    def _fetch_one(yr, mo):
        df = download_funding_monthly(symbol, yr, mo)
        if df.empty:
            return pd.DataFrame()
        df["ts"] = pd.to_numeric(df["calc_time"], errors="coerce")
        return df[df["ts"] > max_ts]

    with ThreadPoolExecutor(max_workers=config.VISION_WORKERS) as ex:
        futures = {ex.submit(_fetch_one, yr, mo): (yr, mo) for yr, mo in tasks}
        for f in as_completed(futures):
            try:
                df = f.result()
            except Exception:
                continue
            if df.empty:
                continue
            conn.execute("""
                INSERT INTO funding_rates
                SELECT symbol, calc_time, funding_interval_hours, last_funding_rate, ts FROM df
            """)
            total += len(df)
            time.sleep(0.1)

    # ── 5. Выгрузить результат ──
    df = conn.execute(
        "SELECT * FROM funding_rates WHERE symbol=? ORDER BY ts", [symbol]
    ).fetchdf()
    conn.close()

    if export_parquet and not df.empty:
        pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq, index=False)
        _sync_to_bucket(symbol, df)

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Загрузка funding rate с data.binance.vision")
    parser.add_argument("--symbol", nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="Тикеры (по умолчанию BTCUSDT ETHUSDT)")
    parser.add_argument("--years", type=int, default=3, help="Сколько лет")
    args = parser.parse_args()

    t0 = time.time()
    for symbol in args.symbol:
        print(f"  {symbol}: загрузка funding...", end=" ", flush=True)
        df = fetch_funding(symbol, args.years)
        print(f"{len(df):,} записей")

    print(f"\n  Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
