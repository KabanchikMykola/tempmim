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
from data_fetcher import config
from data_fetcher.contracts import validate_ohlcv

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


# ── Binance REST API tail ────────────────────────────────────────


from data_fetcher.binance_api.klines import fetch_tail


# ── S3 download helpers ──────────────────────────────────────────


def _bucket_parquet_uri(symbol, interval, source):
    market_dir = "ohlcv_perp" if source == "perp" else "ohlcv_spot"
    return f"{config.BUCKET_URI}/{market_dir}/{symbol}_{interval}.parquet"


def _load_from_bucket(symbol, interval, source):
    """Прочитать единый parquet из bucket."""
    if not config.BUCKET_ID:
        return None
    uri = _bucket_parquet_uri(symbol, interval, source)
    try:
        df = pd.read_parquet(uri)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _upload_to_bucket(symbol, interval, source, df):
    """Загрузить DataFrame как единый parquet в bucket."""
    if df is None or df.empty or not config.BUCKET_ID:
        return
    uri = _bucket_parquet_uri(symbol, interval, source)
    df.to_parquet(uri, index=False)


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





# ── Core: fetch_symbol ───────────────────────────────────────────


def _generate_months(years, end_time=None):
    """Сгенерировать список (year, month) от (end_time - years) до end_time.

    end_time по умолчанию = сейчас. Текущий месяц включён.
    """
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    months = []
    start_yr = end_time.year - years
    for yr in range(start_yr, end_time.year + 1):
        end_m = end_time.month if yr == end_time.year else 12
        for mo in range(1, end_m + 1):
            months.append((yr, mo))
    return months


def fetch_symbol(symbol, interval="1h", years=3, perp=False,
                 export_parquet=True, tail=False, tail_only=False, validate=True,
                 end_time=None):
    """Загрузить klines для одного символа.

    Args:
        symbol: Тикер (BTCUSDT, ETHUSDT...)
        interval: Таймфрейм (1m, 5m, 15m, 1h, 4h, 1d)
        years: Сколько лет качать с S3
        perp: True = перпетуалы, False = спот
        export_parquet: Экспортировать в Parquet после загрузки
        tail: Докачать хвост через Binance REST API
        tail_only: ТОЛЬКО хвост, без S3 и кеша
        validate: Запустить валидацию после загрузки
        end_time: Верхняя граница (datetime, UTC). None = сейчас.

    Returns:
        (pd.DataFrame, warnings) — данные и список предупреждений.
    """
    source = "perp" if perp else "spot"
    all_warnings = []
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    end_ms = int(end_time.timestamp() * 1000)

    # ── tail_only: только REST API ──
    if tail_only:
        tail_df = fetch_tail(symbol, interval, perp, end_time=end_ms)
        if tail_df.empty:
            return pd.DataFrame(), ["tail: нет данных от API"]
        tail_df["symbol"] = symbol
        tail_df["source"] = source
        tail_df["interval"] = interval
        tail_df, tw = validate_ohlcv(tail_df, interval)
        all_warnings.extend(tw)
        _upload_to_bucket(symbol, interval, source, tail_df)
        return tail_df, all_warnings

    # ── 1. Загрузить существующие данные из bucket ──
    existing = _load_from_bucket(symbol, interval, source)
    if existing is not None:
        existing = existing[existing["ts"] <= end_ms]
    if existing is not None and len(existing) > 1000:
        staleness_days = (end_ms - existing["ts"].max()) / 86400000

        if staleness_days < 2:
            if tail:
                tail_df = fetch_tail(symbol, interval, perp, end_time=end_ms)
                if not tail_df.empty and tail_df["ts"].max() > existing["ts"].max():
                    tail_df["symbol"] = symbol
                    tail_df["source"] = source
                    tail_df["interval"] = interval
                    tail_new = tail_df[tail_df["ts"] > existing["ts"].max()]
                    if not tail_new.empty:
                        existing = pd.concat([existing, tail_new], ignore_index=True)
                        existing = existing.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
                        existing = existing[existing["ts"] <= end_ms]
                        existing["timestamp"] = existing["ts"]
                        _upload_to_bucket(symbol, interval, source, existing)
                        all_warnings.append(f"   tail: +{len(tail_new)} баров от API")
            return existing, all_warnings

        all_warnings.append(f"   данные устарели на {staleness_days:.0f}д — докачка")

    # ── 2. Определить какие месяцы нужно скачать ──
    if existing is not None and not existing.empty:
        db_max = existing["ts"].max()
        db_dt = datetime.fromtimestamp(db_max / 1000, tz=timezone.utc)
        all_months = _generate_months(years, end_time)
        months_to_fetch = [(y, m) for y, m in all_months
                           if (y > db_dt.year) or (y == db_dt.year and m >= db_dt.month)]
    else:
        months_to_fetch = _generate_months(years, end_time)

    if not months_to_fetch:
        if existing is not None and not existing.empty:
            return existing, all_warnings
        return pd.DataFrame(), all_warnings

    # ── 3. Скачать с S3 параллельно ──
    import calendar

    def _fetch_month(yr, mo):
        df = download_monthly(symbol, interval, yr, mo, perp)
        if not df.empty:
            return df
        days_in_month = calendar.monthrange(yr, mo)[1]
        max_day = end_time.day if (yr == end_time.year and mo == end_time.month) else days_in_month
        daily_dfs = []
        for day in range(1, max_day + 1):
            date_str = f"{yr}-{mo:02d}-{day:02d}"
            df_d = download_daily(symbol, interval, date_str, perp)
            if not df_d.empty:
                daily_dfs.append(df_d)
        if daily_dfs:
            return pd.concat(daily_dfs, ignore_index=True)
        return pd.DataFrame()

    s3_dfs = []
    with ThreadPoolExecutor(max_workers=config.VISION_WORKERS) as ex:
        futures = {ex.submit(_fetch_month, yr, mo): (yr, mo)
                   for yr, mo in months_to_fetch}
        for f in as_completed(futures):
            try:
                df = f.result()
            except Exception:
                continue
            if df.empty:
                continue
            df["symbol"] = symbol
            df["source"] = source
            df["interval"] = interval
            s3_dfs.append(df)

    # ── 4. Склеить существующие и новые данные ──
    all_dfs = []
    if existing is not None and not existing.empty:
        all_dfs.append(existing)
    all_dfs.extend(s3_dfs)
    if not all_dfs:
        return pd.DataFrame(), all_warnings

    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    merged = merged[merged["ts"] <= end_ms]
    merged["timestamp"] = merged["ts"]

    # ── 5. Tail через REST API ──
    if tail:
        tail_df = fetch_tail(symbol, interval, perp, end_time=end_ms)
        if not tail_df.empty:
            tail_df["symbol"] = symbol
            tail_df["source"] = source
            tail_df["interval"] = interval
            tail_new = tail_df[tail_df["ts"] > merged["ts"].max()]
            if not tail_new.empty:
                merged = pd.concat([merged, tail_new], ignore_index=True)
                merged = merged.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
                merged = merged[merged["ts"] <= end_ms]
                merged["timestamp"] = merged["ts"]
                all_warnings.append(f"   tail API: +{len(tail_new)} баров")

    # ── 6. Валидация ──
    if validate and not merged.empty:
        merged, vw = validate_ohlcv(merged, interval)
        all_warnings.extend(vw)

    # ── 7. Загрузить в bucket ──
    _upload_to_bucket(symbol, interval, source, merged)

    return merged, all_warnings


# ── Bucket summary (на основе данных в bucket) ────────────────────


def list_bucket_data():
    """Вывести сводку данных в bucket."""
    from data_fetcher.config import list_bucket
    for subdir in ["binance/ohlcv_spot/", "binance/ohlcv_perp/"]:
        files = list_bucket(subdir)
        if not files:
            continue
        total_mb = sum(f["size_bytes"] for f in files) / 1024 / 1024
        print(f"  Bucket: {subdir.strip('/')}  ({len(files)} файлов, {total_mb:.0f} MB)")
        for f in sorted(files, key=lambda x: x["path"]):
            size_kb = f["size_bytes"] / 1024
            print(f"    {f['path']:68s} {size_kb:>8.0f} KB")
    print()


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
    args = parser.parse_args()

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

    list_bucket_data()
    print(f"\n  Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()