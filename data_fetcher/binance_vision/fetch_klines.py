"""Загрузка OHLCV klines с data.binance.vision.

Трёхуровневый кеш:
1. Parquet файл (data/{symbol}_{interval}_{source}.parquet)
2. DuckDB кеш (data/cache/binance_vision.db)
3. Скачивание с data.binance.vision (S3 архивы)

Использование:
    python -m data_fetcher ohlcv vision --symbol BTCUSDT --interval 1h
    python -m data_fetcher ohlcv vision --symbol SOLUSDT --interval 5m --perp
    python -m data_fetcher ohlcv vision --symbol BTCUSDT ETHUSDT --years 2
    python -m data_fetcher ohlcv vision --symbol BTCUSDT --tail       # с хвостом через API
"""

import io
import time
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import requests
import pandas as pd
import duckdb

from data_fetcher import config

# ── Timestamp helpers ────────────────────────────────────────────


def _is_microseconds(series):
    """Определить есть ли значения в микросекундах (> 1e14, т.е. > 5138 год)."""
    return series > 1e14


def _normalize_timestamps(df):
    """Привести open_time/close_time к миллисекундам (per-row).

    Binance S3 spot отдаёт open_time в μs после ~2025-01-01,
    а perp — всегда в ms. В одном файле/месяце могут быть оба формата.
    Нормализуем построчно: если > 1e14 → делим на 1000.
    """
    if df.empty:
        return df

    for col in ("open_time", "close_time"):
        if col not in df.columns:
            continue
        mask = _is_microseconds(df[col])
        if mask.any():
            df.loc[mask, col] = df.loc[mask, col] // 1000

    # ts всегда равен open_time (после нормализации оба в ms)
    df["ts"] = df["open_time"].copy()

    # Удалить служебную колонку из S3 CSV
    if "ignore" in df.columns:
        df = df.drop(columns=["ignore"])

    return df


def _validate_ohlcv(df, interval="1h", silent=False):
    """Проверить и почистить скачанные OHLCV данные.

    Returns:
        (df_clean, warnings) — DataFrame и список строк-предупреждений.
    """
    warnings = []
    if df.empty:
        return df, warnings

    before = len(df)

    # 1. Удалить zero-volume строки (техобслуживание биржи)
    df = df[df["volume"] > 0].copy()
    dropped = before - len(df)
    if dropped > 0:
        msg = f"    {dropped} zero-volume строк удалено"
        warnings.append(msg)

    if df.empty:
        return df, warnings

    # 2. Удалить дубликаты по ts
    before = len(df)
    df = df.drop_duplicates(subset=["ts"])
    dropped = before - len(df)
    if dropped > 0:
        msg = f"    {dropped} дубликатов удалено"
        warnings.append(msg)

    # 3. Сортировка
    df = df.sort_values("ts").reset_index(drop=True)

    # 4. OHLC consistency
    bad = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    if bad.any():
        msg = f"    OHLC inconsistency: {bad.sum()} строк"
        warnings.append(msg)

    # 5. Негативные цены
    for col in ("open", "high", "low", "close"):
        if (df[col] <= 0).any():
            msg = f"    {col} <= 0: {(df[col] <= 0).sum()} строк"
            warnings.append(msg)

    # 6. Проверка на пропуски (gaps) — только если > 2 строк
    if len(df) > 2:
        tf_ms = {"1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000,
                 "4h": 14400000, "1d": 86400000}
        expected = tf_ms.get(interval, 3600000)
        gaps = (df["ts"].diff() > expected * 1.5).sum()
        if gaps > 0:
            msg = f"    Гэпов (>1.5x интервал): {gaps}"
            warnings.append(msg)

    return df, warnings


# ── Binance REST API tail ────────────────────────────────────────


from data_fetcher.binance_api.klines import fetch_tail


# ── S3 download helpers ──────────────────────────────────────────


def _db_path():
    return Path(config.CACHE_DIR) / "binance_vision.db"


def _parquet_path(symbol, interval, source):
    return Path(config.DATA_DIR) / f"{symbol}_{interval}_{source}.parquet"


def download_monthly(symbol, interval, year, month, perp=False):
    """Скачать один месячный архив klines с data.binance.vision."""
    src = "futures/um" if perp else "spot"
    url = (f"https://data.binance.vision/data/{src}/monthly/klines/"
           f"{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip")
    try:
        resp = requests.get(url, timeout=config.VISION_TIMEOUT)
        if resp.status_code != 200:
            return pd.DataFrame()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                content = f.read().decode("utf-8")
                lines = content.strip().split("\n")
                if not lines:
                    return pd.DataFrame()
                if lines[0].startswith("open_time"):
                    lines = lines[1:]
                data = [line.split(",") for line in lines]
                df = pd.DataFrame(data, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "count",
                    "taker_buy_base", "taker_buy_quote", "ignore",
                ])
                for col in ("open_time", "close_time", "count"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                for col in ("open", "high", "low", "close", "volume",
                            "quote_volume", "taker_buy_base", "taker_buy_quote"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return _normalize_timestamps(df)
    except Exception:
        pass
    return pd.DataFrame()


def download_daily(symbol, interval, date_str, perp=False):
    """Скачать один дневной архив klines."""
    src = "futures/um" if perp else "spot"
    url = (f"https://data.binance.vision/data/{src}/daily/klines/"
           f"{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip")
    try:
        resp = requests.get(url, timeout=config.VISION_TIMEOUT)
        if resp.status_code != 200:
            return pd.DataFrame()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                content = f.read().decode("utf-8")
                lines = content.strip().split("\n")
                if not lines:
                    return pd.DataFrame()
                if lines[0].startswith("open_time"):
                    lines = lines[1:]
                data = [line.split(",") for line in lines]
                df = pd.DataFrame(data, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "count",
                    "taker_buy_base", "taker_buy_quote", "ignore",
                ])
                for col in ("open_time", "close_time", "count"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                for col in ("open", "high", "low", "close", "volume",
                            "quote_volume", "taker_buy_base", "taker_buy_quote"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return _normalize_timestamps(df)
    except Exception:
        pass
    return pd.DataFrame()


# ── DuckDB helpers ───────────────────────────────────────────────


def _init_duckdb(conn):
    """Создать таблицу klines если не существует."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS klines (
            symbol VARCHAR, source VARCHAR, interval VARCHAR,
            open_time BIGINT, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume DOUBLE, close_time BIGINT, quote_volume DOUBLE,
            count INTEGER, taker_buy_base DOUBLE, taker_buy_quote DOUBLE, ts BIGINT
        )
    """)


def _db_max_ts(conn, symbol, source, interval):
    """Максимальный ts в DuckDB для (symbol, source, interval), 0 если нет данных."""
    result = conn.execute(
        "SELECT COALESCE(MAX(ts), 0) FROM klines WHERE symbol=? AND source=? AND interval=?",
        [symbol, source, interval],
    ).fetchone()
    return result[0] if result else 0


def _db_count(conn, symbol, source, interval):
    result = conn.execute(
        "SELECT COUNT(*) FROM klines WHERE symbol=? AND source=? AND interval=?",
        [symbol, source, interval],
    ).fetchone()
    return result[0] if result else 0


def _db_insert_batch(conn, df):
    """Вставить только строки с ts > чем уже есть в DuckDB для данной (symbol, source, interval)."""
    if df.empty:
        return 0

    row = df.iloc[0]
    symbol, source, interval = row["symbol"], row["source"], row["interval"]

    max_ts = conn.execute(
        "SELECT COALESCE(MAX(ts), 0) FROM klines WHERE symbol=? AND source=? AND interval=?",
        [symbol, source, interval],
    ).fetchone()[0]

    df_new = df[df["ts"] > max_ts]
    if df_new.empty:
        return 0

    conn.execute("INSERT INTO klines SELECT * FROM df_new")
    return len(df_new)


def _export_to_parquet(conn, symbol, source, interval):
    """Экспортировать данные из DuckDB в Parquet с колонкой timestamp."""
    pq = _parquet_path(symbol, interval, source)
    df = conn.execute(
        "SELECT *, ts AS timestamp FROM klines WHERE symbol=? AND source=? AND interval=? ORDER BY ts",
        [symbol, source, interval],
    ).fetchdf()
    if not df.empty:
        pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq, index=False)
    return df


def _normalize_parquet_on_read(pq):
    """Прочитать parquet, нормализовать timestamp, перезаписать если были µs."""
    df = pd.read_parquet(pq)
    if df.empty:
        return df

    fixed = False
    for col in ("open_time", "close_time", "ts"):
        if col in df.columns and _is_microseconds(df[col]).any():
            df = _normalize_timestamps(df)
            fixed = True
            break

    # Добавить timestamp если нет (для старых файлов)
    if "timestamp" not in df.columns and "ts" in df.columns:
        df["timestamp"] = df["ts"]
        fixed = True

    if fixed and not df.empty:
        df.to_parquet(pq, index=False)

    return df


# ── Core: fetch_symbol ───────────────────────────────────────────


def _generate_months(years):
    """Сгенерировать список (year, month) от (now - years) до текущего месяца.

    Текущий месяц включён — monthly архив 404 → fallback на daily.
    """
    now = datetime.now(timezone.utc)
    months = []
    start_yr = now.year - years
    for yr in range(start_yr, now.year + 1):
        end_m = now.month if yr == now.year else 12
        for mo in range(1, end_m + 1):
            months.append((yr, mo))
    return months


def fetch_symbol(symbol, interval="1h", years=3, perp=False,
                 export_parquet=True, tail=False, tail_only=False, validate=True):
    """Загрузить klines для одного символа.

    Args:
        symbol: Тикер (BTCUSDT, ETHUSDT...)
        interval: Таймфрейм (1m, 5m, 15m, 1h, 4h, 1d)
        years: Сколько лет качать с S3 (если нет в кеше)
        perp: True = перпетуалы, False = спот
        export_parquet: Экспортировать в Parquet после загрузки
        tail: Докачать хвост через Binance REST API
        tail_only: ТОЛЬКО хвост, без S3 и кеша
        validate: Запустить валидацию после загрузки

    Returns:
        (pd.DataFrame, warnings) — данные и список предупреждений.
    """
    source = "perp" if perp else "spot"
    pq = _parquet_path(symbol, interval, source)
    db = _db_path()
    all_warnings = []

    # ── tail_only: только REST API, без S3 и кеша ──
    if tail_only:
        tail_df = fetch_tail(symbol, interval, perp)
        if tail_df.empty:
            return pd.DataFrame(), ["tail: нет данных от API"]
        tail_df["symbol"] = symbol
        tail_df["source"] = source
        tail_df["interval"] = interval
        tail_df, tw = _validate_ohlcv(tail_df, interval)
        all_warnings.extend(tw)
        if export_parquet:
            pq.parent.mkdir(parents=True, exist_ok=True)
            # Мерж с существующим parquet если есть (не перезаписывать!)
            if pq.exists():
                old = _normalize_parquet_on_read(pq)
                old_new = old[old["ts"] < tail_df["ts"].min()]
                merged = pd.concat([old_new, tail_df], ignore_index=True)
                merged = merged.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
                merged["timestamp"] = merged["ts"]
                merged.to_parquet(pq, index=False)
                n_new = len(merged) - len(old_new)
                if n_new > 0:
                    all_warnings.append(f"   tail: +{n_new} баров (мерж с существующими)")
                return merged, all_warnings
            tail_df["timestamp"] = tail_df["ts"]
            tail_df.to_parquet(pq, index=False)
        return tail_df, all_warnings

    # ── 1. Проверить Parquet с авто-нормализацией старых данных ──
    if pq.exists():
        df = _normalize_parquet_on_read(pq)
        if len(df) > 1000:
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            max_ts = df["ts"].max()
            staleness_days = (now_ms - max_ts) / 86400000

            if staleness_days < 2:
                # Данные свежие — опционально докинуть только tail
                if tail:
                    tail_df = fetch_tail(symbol, interval, perp)
                    if not tail_df.empty and tail_df["ts"].max() > max_ts:
                        tail_df["symbol"] = symbol
                        tail_df["source"] = source
                        tail_df["interval"] = interval
                        tail_new = tail_df[tail_df["ts"] > max_ts]
                        if not tail_new.empty:
                            df = pd.concat([df, tail_new], ignore_index=True)
                            df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
                            df = df.reset_index(drop=True)
                            df["timestamp"] = df["ts"]
                            df.to_parquet(pq, index=False)
                            all_warnings.append(f"   tail: +{len(tail_new)} баров от API")
                return df, all_warnings

            # Данные устарели — нужно докачать недостающие месяцы с S3
            all_warnings.append(f"   кеш устарел на {staleness_days:.0f}д — докачка")

    # ── 2. Подготовить DuckDB ──
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db))
    _init_duckdb(conn)

    existing_count = _db_count(conn, symbol, source, interval)
    db_max = _db_max_ts(conn, symbol, source, interval)

    # ── 3. Определить какие месяцы нужно скачать ──
    if db_max > 0 and existing_count > 1000:
        # Есть данные в DuckDB — качаем только начиная с db_max
        from datetime import datetime as dt
        db_dt = dt.fromtimestamp(db_max / 1000, tz=timezone.utc)
        all_months = _generate_months(years)
        months_to_fetch = [(y, m) for y, m in all_months
                           if (y > db_dt.year) or (y == db_dt.year and m >= db_dt.month)]
    else:
        months_to_fetch = _generate_months(years)

    if not months_to_fetch:
        conn.close()
        if export_parquet:
            df, _ = _export_to_parquet(conn, symbol, source, interval)
            return (df, all_warnings) if not df.empty else (pd.DataFrame(), all_warnings)
        df = conn.execute(
            "SELECT * FROM klines WHERE symbol=? AND source=? AND interval=? ORDER BY ts",
            [symbol, source, interval],
        ).fetchdf()
        conn.close()
        return df, all_warnings

    # ── 4. Скачать с S3 параллельно ──
    import calendar

    now = datetime.now(timezone.utc)

    def _fetch_month(yr, mo):
        df = download_monthly(symbol, interval, yr, mo, perp)
        if not df.empty:
            return df
        days_in_month = calendar.monthrange(yr, mo)[1]
        max_day = now.day if (yr == now.year and mo == now.month) else days_in_month
        daily_dfs = []
        for day in range(1, max_day + 1):
            date_str = f"{yr}-{mo:02d}-{day:02d}"
            df_d = download_daily(symbol, interval, date_str, perp)
            if not df_d.empty:
                daily_dfs.append(df_d)
        if daily_dfs:
            return pd.concat(daily_dfs, ignore_index=True)
        return pd.DataFrame()

    total_inserted = 0
    with ThreadPoolExecutor(max_workers=config.VISION_WORKERS) as ex:
        futures = {ex.submit(_fetch_month, yr, mo): (yr, mo)
                   for yr, mo in months_to_fetch}
        for f in as_completed(futures):
            yr, mo = futures[f]
            try:
                df = f.result()
            except Exception:
                continue
            if df.empty:
                continue
            df["symbol"] = symbol
            df["source"] = source
            df["interval"] = interval
            n = _db_insert_batch(conn, df)
            total_inserted += n

    # ── 5. Tail через REST API ──
    if tail:
        tail_df = fetch_tail(symbol, interval, perp)
        if not tail_df.empty:
            tail_df["symbol"] = symbol
            tail_df["source"] = source
            tail_df["interval"] = interval
            n = _db_insert_batch(conn, tail_df)
            if n > 0:
                all_warnings.append(f"   tail API: +{n} баров")
            total_inserted += n

    # ── 6. Выгрузить результат ──
    df = conn.execute(
        "SELECT * FROM klines WHERE symbol=? AND source=? AND interval=? ORDER BY ts",
        [symbol, source, interval],
    ).fetchdf()

    if validate and not df.empty:
        df, vw = _validate_ohlcv(df, interval)
        all_warnings.extend(vw)

    conn.close()

    if export_parquet and not df.empty:
        pq.parent.mkdir(parents=True, exist_ok=True)
        df["timestamp"] = df["ts"]
        df.to_parquet(pq, index=False)

    return df, all_warnings


# ── Export / Summary ─────────────────────────────────────────────


def export_all():
    """Экспортировать все данные из DuckDB в Parquet."""
    db = _db_path()
    if not db.exists():
        print("  Кеш пуст")
        return
    conn = duckdb.connect(str(db))
    rows = conn.execute(
        "SELECT DISTINCT symbol, source, interval FROM klines"
    ).fetchall()
    for symbol, source, interval in rows:
        pq = _parquet_path(symbol, interval, source)
        print(f"  Экспорт {symbol} {source} {interval} -> {pq}")
        _export_to_parquet(conn, symbol, source, interval)
    conn.close()


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
                return datetime.fromtimestamp(
                    ts / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d")
            except (OSError, ValueError):
                return "?"
        print(f"    {symbol:>12} {source:>5} {interval:>4}: "
              f"{cnt:>8,} баров ({_fmt(t_min)} - {_fmt(t_max)})")


# ── CLI ───────────────────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Загрузка klines с data.binance.vision"
    )
    parser.add_argument("--symbol", nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="Тикеры")
    parser.add_argument("--interval", default="1h", help="Таймфрейм")
    parser.add_argument("--years", type=int, default=3, help="Сколько лет")
    parser.add_argument("--perp", action="store_true",
                        help="Перпетуалы вместо спота")
    parser.add_argument("--tail", action="store_true",
                        help="Докачать последние бары через API")
    parser.add_argument("--export-all", action="store_true",
                        help="Экспорт всего кеша в Parquet")
    parser.add_argument("--summary", action="store_true",
                        help="Сводка по кешу")
    args = parser.parse_args()

    if args.export_all:
        export_all()
        return

    if args.summary:
        summary()
        return

    t0 = time.time()
    for symbol in args.symbol:
        src_name = "perp" if args.perp else "spot"
        print(f"  {symbol} ({src_name}): загрузка...", flush=True)
        df, warnings = fetch_symbol(
            symbol, args.interval, args.years, args.perp,
            export_parquet=True, tail=args.tail,
        )
        for w in warnings:
            print(w)
        print(f"  {symbol} ({src_name}): {len(df):,} баров")

    summary()
    print(f"\n  Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()