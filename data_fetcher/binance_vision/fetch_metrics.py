"""Загрузка исторических derivatives metrics с data.binance.vision.

Metrics: open interest, long/short ratios (5-мин интервал).
Только дневные архивы (monthly не доступен).

Колонки CSV:
  create_time, symbol, sum_open_interest, sum_open_interest_value,
  count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
  count_long_short_ratio, sum_taker_long_short_vol_ratio

Поддержка HuggingFace Bucket как основного хранилища:
  Если BUCKET_ID задан в config.py, данные читаются/пишутся в hf://buckets/{BUCKET_ID}/data/

Использование:
    python -m data_fetcher metrics --symbol BTCUSDT
    python -m data_fetcher metrics --symbol BTCUSDT ETHUSDT --years 2
    python -m data_fetcher metrics --symbol BTCUSDT --summary
"""

import io
import time
import zipfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import duckdb

from data_fetcher import config

# ── Константы ────────────────────────────────────────────────────

METRICS_COLS = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]

S3_BASE = "https://data.binance.vision/data/futures/um/daily/metrics"


# ── Path helpers ─────────────────────────────────────────────────


def _db_path():
    return Path(config.CACHE_DIR) / "binance_vision.db"


def _parquet_path(symbol):
    """Локальный путь к parquet (для DuckDB и кеша)."""
    return Path(config.DATA_DIR) / f"{symbol}_metrics.parquet"


def _bucket_parquet_uri(symbol):
    """URI к parquet в HuggingFace Bucket."""
    return f"hf://buckets/{config.BUCKET_ID}/data/{symbol}_metrics.parquet"


# ── DuckDB helpers ───────────────────────────────────────────────


def _init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            symbol VARCHAR, create_time BIGINT,
            sum_open_interest DOUBLE, sum_open_interest_value DOUBLE,
            count_toptrader_long_short_ratio DOUBLE,
            sum_toptrader_long_short_ratio DOUBLE,
            count_long_short_ratio DOUBLE,
            sum_taker_long_short_vol_ratio DOUBLE,
            ts BIGINT
        )
    """)


def _db_max_ts(conn, symbol):
    result = conn.execute(
        "SELECT COALESCE(MAX(ts), 0) FROM metrics WHERE symbol=?", [symbol]
    ).fetchone()
    return result[0] if result else 0


def _db_count(conn, symbol):
    result = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE symbol=?", [symbol]
    ).fetchone()
    return result[0] if result else 0


# ── Date helpers ─────────────────────────────────────────────────


def _generate_dates(years):
    """Сгенерировать даты (строки YYYY-MM-DD) от начала до вчера."""
    now = datetime.now(timezone.utc)
    start = datetime(now.year - years, 1, 1, tzinfo=timezone.utc)
    end = now - timedelta(days=1)
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


# ── S3 download ──────────────────────────────────────────────────


def download_daily(symbol, date_str):
    """Скачать metrics за один день (ZIP с data.binance.vision).

    create_time конвертируется в миллисекунды для совместимости
    с остальными fetcher-ами (klines, funding и т.д.).
    """
    url = f"{S3_BASE}/{symbol}/{symbol}-metrics-{date_str}.zip"
    try:
        resp = requests.get(url, timeout=config.VISION_TIMEOUT)
        if resp.status_code != 200:
            return pd.DataFrame()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f, header=0)
                df = df[["create_time"] + METRICS_COLS[2:]]
                # Timestamp → миллисекунды (как open_time в klines)
                df["create_time"] = (
                    pd.to_datetime(df["create_time"], utc=True)
                    .astype("int64") // 10**6
                )
                for col in METRICS_COLS[2:]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
    except Exception:
        pass
    return pd.DataFrame()


# ── HuggingFace Bucket helpers ───────────────────────────────────


def _sync_from_bucket(symbol):
    """Скачать parquet из bucket в локальный кеш (если есть). Возвращает DataFrame или None."""
    if not config.BUCKET_ID:
        return None
    pq = _parquet_path(symbol)
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
        print(f"  ошибка bucket sync: {e}")
        return None


def _sync_to_bucket(symbol, df=None):
    """Загрузить локальный parquet в bucket. Принимает DataFrame для избежания повторного чтения."""
    if not config.BUCKET_ID:
        return
    try:
        if df is None:
            pq = _parquet_path(symbol)
            if not pq.exists():
                return
            df = pd.read_parquet(pq)
        uri = _bucket_parquet_uri(symbol)
        df.to_parquet(uri, index=False)
    except Exception as e:
        print(f"  ошибка bucket sync: {e}")


def _load_parquet(symbol):
    """Прочитать локальный parquet (если есть)."""
    pq = _parquet_path(symbol)
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception as e:
            print(f"  ошибка чтения parquet: {e}")
    return pd.DataFrame()


# ── Валидация ────────────────────────────────────────────────────


def _validate_metrics(df):
    """Проверить и почистить metrics данные."""
    warnings = []
    if df.empty:
        return df, warnings

    before = len(df)
    df = df.drop_duplicates(subset=["symbol", "ts"])
    dropped = before - len(df)
    if dropped > 0:
        warnings.append(f"  {dropped} дубликатов удалено")

    df = df.sort_values("ts").reset_index(drop=True)
    return df, warnings


# ── Core: fetch_metrics ──────────────────────────────────────────


def fetch_metrics(symbol, years=3, export_parquet=True, force=False):
    """Загрузить derivatives metrics для одного символа.

    Кеш: Parquet (data/) → DuckDB (data/cache/) → S3 (data.binance.vision).
    Если BUCKET_ID задан — синхронизация с HuggingFace Bucket.

    Архитектура данных:
      DuckDB — промежуточный слой для мержа данных.
      Parquet — локальный кеш для быстрого чтения.
      Bucket — долгосрочное хранилище (облачные среды).

    Args:
        symbol: Тикер (BTCUSDT, ETHUSDT...)
        years: Сколько лет
        export_parquet: Экспорт в Parquet
        force: Принудительная перезагрузка

    Returns:
        (pd.DataFrame, warnings)
    """
    all_warnings = []
    pq = _parquet_path(symbol)

    # ── 1. Загрузить из bucket в локальный кеш ──
    if config.BUCKET_ID and not pq.exists():
        bucket_df = _sync_from_bucket(symbol)
        if bucket_df is not None and len(bucket_df) > 1000:
            return bucket_df, all_warnings

    # ── 2. Проверить локальный Parquet ──
    if pq.exists() and not force:
        try:
            df = pd.read_parquet(pq)
            if len(df) > 1000:
                # Проверяем свежесть: если данные устарели > 2 дней —
                # автоматически докачиваем
                max_ts = df["ts"].max()
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                yesterday_ms = now_ms - 86400000
                gap_days = (yesterday_ms - max_ts) / 86400000

                if gap_days <= 2:
                    return df, all_warnings

                # Данные устарели — нужна докачка
                all_warnings.append(
                    f"  кеш устарел на {gap_days:.0f}д — докачка"
                )
                force = True
        except Exception:
            pass

    # ── 3. Подготовить DuckDB ──
    db = _db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db))
    _init_db(conn)

    # Загрузить существующий parquet в DuckDB (если DuckDB пустой)
    if _db_count(conn, symbol) == 0:
        existing = _load_parquet(symbol)
        if not existing.empty:
            existing["symbol"] = symbol
            conn.execute("""
                INSERT INTO metrics
                SELECT symbol, create_time, sum_open_interest,
                       sum_open_interest_value,
                       count_toptrader_long_short_ratio,
                       sum_toptrader_long_short_ratio,
                       count_long_short_ratio,
                       sum_taker_long_short_vol_ratio,
                       ts FROM existing
            """)

    # При force-режиме очищаем старые данные для переимпорта
    if force:
        conn.execute("DELETE FROM metrics WHERE symbol=?", [symbol])

    existing_count = _db_count(conn, symbol)
    if existing_count > 10000 and not force:
        df = conn.execute(
            "SELECT * FROM metrics WHERE symbol=? ORDER BY ts", [symbol]
        ).fetchdf()
        conn.close()
        if export_parquet and not df.empty:
            pq.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(pq, index=False)
            _sync_to_bucket(symbol, df)
        return df, all_warnings

    # ── 4. Определить недостающие даты ──
    all_dates = _generate_dates(years)
    db_max = _db_max_ts(conn, symbol)

    if db_max > 0:
        db_dt = datetime.fromtimestamp(db_max / 1000, tz=timezone.utc)
        cutoff = db_dt.strftime("%Y-%m-%d")
        dates_to_fetch = [d for d in all_dates if d > cutoff]
    else:
        dates_to_fetch = all_dates

    if not dates_to_fetch:
        # Данные уже полные — экспорт и возврат
        df = conn.execute(
            "SELECT * FROM metrics WHERE symbol=? ORDER BY ts", [symbol]
        ).fetchdf()
        conn.close()
        if not df.empty:
            df, vw = _validate_metrics(df)
            all_warnings.extend(vw)
        if export_parquet and not df.empty:
            pq.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(pq, index=False)
            _sync_to_bucket(symbol)
        return df, all_warnings

    # ── 5. Параллельная загрузка с S3 ──
    def _fetch_one(date_str):
        df = download_daily(symbol, date_str)
        if not df.empty:
            df["ts"] = df["create_time"]
        return date_str, df

    total_inserted = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=config.VISION_WORKERS) as ex:
        futures = {ex.submit(_fetch_one, d): d for d in dates_to_fetch}
        for f in as_completed(futures):
            date_str = futures[f]
            try:
                _, df = f.result()
            except Exception:
                errors += 1
                continue
            if df.empty:
                continue
            df["symbol"] = symbol
            conn.execute("""
                INSERT INTO metrics
                SELECT symbol, create_time, sum_open_interest,
                       sum_open_interest_value,
                       count_toptrader_long_short_ratio,
                       sum_toptrader_long_short_ratio,
                       count_long_short_ratio,
                       sum_taker_long_short_vol_ratio,
                       ts FROM df
            """)
            total_inserted += len(df)

    if errors > 0:
        all_warnings.append(f"  {errors} ошибок загрузки")

    if total_inserted == 0:
        all_warnings.append("  Нет новых данных с S3")
        conn.close()
        if export_parquet and pq.exists():
            df = pd.read_parquet(pq)
            return df, all_warnings
        return pd.DataFrame(), all_warnings

    # ── 6. Выгрузить результат ──
    df = conn.execute(
        "SELECT * FROM metrics WHERE symbol=? ORDER BY ts", [symbol]
    ).fetchdf()
    conn.close()

    if not df.empty:
        df, vw = _validate_metrics(df)
        all_warnings.extend(vw)

    if export_parquet and not df.empty:
        pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq, index=False)
        _sync_to_bucket(symbol, df)

    return df, all_warnings


# ── Export / Summary ─────────────────────────────────────────────


def export_all():
    """Экспортировать все metrics из DuckDB в Parquet."""
    db = _db_path()
    if not db.exists():
        print("  Кеш пуст")
        return
    conn = duckdb.connect(str(db), read_only=True)
    rows = conn.execute(
        "SELECT DISTINCT symbol FROM metrics"
    ).fetchall()
    conn.close()
    for (symbol,) in rows:
        print(f"  Экспорт {symbol} metrics")
        fetch_metrics(symbol, export_parquet=True)


def summary():
    """Вывести сводку по metrics в DuckDB."""
    db = _db_path()
    if not db.exists():
        print("  Кеш пуст")
        return
    conn = duckdb.connect(str(db), read_only=True)
    result = conn.execute("""
        SELECT symbol, COUNT(*), MIN(ts), MAX(ts)
        FROM metrics
        GROUP BY symbol
        ORDER BY symbol
    """).fetchall()
    conn.close()

    for symbol, cnt, t_min, t_max in result:
        def _fmt(ts):
            if not ts or ts <= 0:
                return "?"
            try:
                return datetime.fromtimestamp(
                    ts / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d")
            except (OSError, ValueError):
                return "?"

        print(f"    {symbol:>12} metrics: {cnt:>8,} записей ({_fmt(t_min)} - {_fmt(t_max)})")


# ── CLI ───────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Загрузка derivatives metrics с data.binance.vision"
    )
    parser.add_argument(
        "--symbol", nargs="+", default=["BTCUSDT", "ETHUSDT"],
        help="Тикеры (по умолчанию BTCUSDT ETHUSDT)",
    )
    parser.add_argument("--years", type=int, default=3, help="Сколько лет")
    parser.add_argument("--force", action="store_true", help="Принудительная перезагрузка")
    parser.add_argument("--summary", action="store_true", help="Сводка по кешу")
    args = parser.parse_args()

    if args.summary:
        summary()
        return

    t0 = time.time()
    for symbol in args.symbol:
        print(f"  {symbol}: загрузка metrics...", flush=True)
        df, warnings = fetch_metrics(symbol, args.years, force=args.force)
        for w in warnings:
            print(w)
        print(f"  {symbol}: {len(df):,} записей")

    summary()
    print(f"\n  Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
