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

from data_fetcher import config
from data_fetcher.contracts import validate_funding
from data_fetcher.binance_api.tail import tail_funding


def _bucket_parquet_uri(symbol):
    return f"{config.BUCKET_URI}/funding/{symbol}_funding.parquet"


def _load_from_bucket(symbol):
    """Прочитать funding parquet из bucket."""
    if not config.BUCKET_ID:
        return None
    try:
        uri = _bucket_parquet_uri(symbol)
        df = pd.read_parquet(uri)
        return df if not df.empty else None
    except Exception:
        return None


def _upload_to_bucket(symbol, df):
    """Загрузить funding parquet в bucket."""
    if df is None or df.empty or not config.BUCKET_ID:
        return
    uri = _bucket_parquet_uri(symbol)
    df.to_parquet(uri, index=False)


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


def fetch_funding(symbol, years=3, tail=True):
    """Загрузить funding rate для символа.

    Args:
        symbol: Тикер (BTCUSDT, ETHUSDT...)
        years: Сколько лет
        tail: Докачать хвост через API

    Returns:
        pd.DataFrame с колонками symbol, calc_time, funding_interval_hours, last_funding_rate
    """
    all_warnings = []

    # ── 1. Загрузить существующие данные из bucket ──
    existing = _load_from_bucket(symbol)
    if existing is not None and len(existing) > 1000:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        max_ts = existing["ts"].max()
        staleness_days = (now_ms - max_ts) / 86400000

        if staleness_days < 2:
            if tail:
                tail_df = tail_funding(symbol, start_time=max_ts + 1, limit=1000)
                if not tail_df.empty:
                    tail_new = tail_df[tail_df["ts"] > max_ts]
                    if not tail_new.empty:
                        existing = pd.concat([existing, tail_new], ignore_index=True)
                        existing = existing.sort_values("ts").reset_index(drop=True)
                        _upload_to_bucket(symbol, existing)
                        all_warnings.append(f"   tail API: +{len(tail_new)} записей")
            if all_warnings:
                for w in all_warnings:
                    print(w)
            return existing

    # ── 2. Скачать историю с S3 ──
    now = datetime.now()
    tasks = []
    for year in range(now.year - years, now.year + 1):
        for month in range(1, 13):
            if year == now.year and month > now.month:
                break
            tasks.append((year, month))

    max_ts = existing["ts"].max() if existing is not None and not existing.empty else 0

    def _fetch_one(yr, mo):
        df = download_funding_monthly(symbol, yr, mo)
        if df.empty:
            return pd.DataFrame()
        df["ts"] = pd.to_numeric(df["calc_time"], errors="coerce")
        return df[df["ts"] > max_ts]

    s3_dfs = []
    with ThreadPoolExecutor(max_workers=config.VISION_WORKERS) as ex:
        futures = {ex.submit(_fetch_one, yr, mo): (yr, mo) for yr, mo in tasks}
        for f in as_completed(futures):
            try:
                df = f.result()
            except Exception:
                continue
            if df.empty:
                continue
            s3_dfs.append(df)

    # ── 3. Склеить ──
    all_dfs = []
    if existing is not None and not existing.empty:
        all_dfs.append(existing)
    all_dfs.extend(s3_dfs)

    if not all_dfs:
        return pd.DataFrame()

    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # ── 4. Tail ──
    if tail:
        max_ts = merged["ts"].max()
        tail_df = tail_funding(symbol, start_time=max_ts + 1, limit=1000)
        if not tail_df.empty:
            tail_new = tail_df[tail_df["ts"] > max_ts]
            if not tail_new.empty:
                merged = pd.concat([merged, tail_new], ignore_index=True)
                merged = merged.sort_values("ts").reset_index(drop=True)
                all_warnings.append(f"   tail API: +{len(tail_new)} записей")

    # ── 5. Валидация ──
    if not merged.empty:
        merged, vw = validate_funding(merged)

    # ── 6. Загрузить в bucket ──
    _upload_to_bucket(symbol, merged)

    if all_warnings:
        for w in all_warnings:
            print(w)

    return merged


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Загрузка funding rate с data.binance.vision")
    parser.add_argument("--symbol", nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="Тикеры (по умолчанию BTCUSDT ETHUSDT)")
    parser.add_argument("--years", type=int, default=3, help="Сколько лет")
    parser.add_argument("--tail", action="store_true", help="Докачать хвост через API")
    args = parser.parse_args()

    t0 = time.time()
    for symbol in args.symbol:
        print(f"  {symbol}: загрузка funding...", end=" ", flush=True)
        df = fetch_funding(symbol, args.years, tail=args.tail)
        print(f"{len(df):,} записей")

    print(f"\n  Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
