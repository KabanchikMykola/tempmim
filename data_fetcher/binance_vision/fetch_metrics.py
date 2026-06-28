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

from data_fetcher import config
from data_fetcher.contracts import validate_metrics
from data_fetcher.binance_api.tail import tail_open_interest_hist, tail_taker_vol_ratio

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


def _bucket_parquet_uri(symbol):
    return f"{config.BUCKET_URI}/metrics/{symbol}_metrics.parquet"


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
            print(f"    S3 HTTP {resp.status_code} для {symbol}/{date_str}")
            return pd.DataFrame()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f, header=0)
                df = df[["create_time"] + METRICS_COLS[2:]]
                # Timestamp → миллисекунды (как open_time в klines)
                df["create_time"] = (
                    pd.to_datetime(df["create_time"], utc=True)
                    .astype("int64") // 10**3
                )
                for col in METRICS_COLS[2:]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
    except Exception as e:
        print(f"    S3 ошибка {symbol}/{date_str}: {type(e).__name__}: {e}")
    return pd.DataFrame()


# ── HuggingFace Bucket helpers ───────────────────────────────────


def _load_from_bucket(symbol):
    """Прочитать metrics parquet из bucket."""
    if not config.BUCKET_ID:
        return None
    try:
        uri = _bucket_parquet_uri(symbol)
        df = pd.read_parquet(uri)
        return df if not df.empty else None
    except Exception:
        return None


def _upload_to_bucket(symbol, df):
    """Загрузить metrics parquet в bucket."""
    if df is None or df.empty or not config.BUCKET_ID:
        return
    uri = _bucket_parquet_uri(symbol)
    df.to_parquet(uri, index=False)


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


def fetch_metrics(symbol, years=3, force=False, tail=True):
    """Загрузить derivatives metrics для одного символа.

    Args:
        symbol: Тикер (BTCUSDT, ETHUSDT...)
        years: Сколько лет
        force: Принудительная перезагрузка с S3
        tail: Докачать хвост через API

    Returns:
        (pd.DataFrame, warnings)
    """
    all_warnings = []

    # ── 1. Загрузить существующие данные из bucket ──
    existing = _load_from_bucket(symbol)
    if existing is not None and len(existing) > 1000 and not force:
        max_ts = existing["ts"].max()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        gap_days = (now_ms - max_ts) / 86400000

        if gap_days <= 2:
            if tail:
                # Tail OI
                oi_df = tail_open_interest_hist(symbol, start_time=max_ts + 1)
                if not oi_df.empty:
                    oi_new = oi_df[oi_df["ts"] > max_ts]
                    if not oi_new.empty:
                        oi_new["symbol"] = symbol
                        oi_new["create_time"] = oi_new["ts"]
                        for col in ["count_toptrader_long_short_ratio",
                                     "sum_toptrader_long_short_ratio",
                                     "count_long_short_ratio",
                                     "sum_taker_long_short_vol_ratio"]:
                            oi_new[col] = None
                        existing = pd.concat([existing, oi_new], ignore_index=True)
                        all_warnings.append(f"   tail OI: +{len(oi_new)} записей")

                # Tail taker vol
                tv_df = tail_taker_vol_ratio(symbol, start_time=max_ts + 1)
                if not tv_df.empty:
                    tv_new = tv_df[tv_df["ts"] > max_ts]
                    if not tv_new.empty:
                        tv_new["symbol"] = symbol
                        tv_new["create_time"] = tv_new["ts"]
                        for col in ["sum_open_interest", "sum_open_interest_value",
                                     "count_toptrader_long_short_ratio",
                                     "sum_toptrader_long_short_ratio",
                                     "count_long_short_ratio"]:
                            tv_new[col] = None
                        existing = pd.concat([existing, tv_new], ignore_index=True)
                        all_warnings.append(f"   tail taker vol: +{len(tv_new)} записей")

                if all_warnings:
                    existing = existing.drop_duplicates(subset=["symbol", "ts"]).sort_values("ts").reset_index(drop=True)
                    _upload_to_bucket(symbol, existing)

            existing, vw = _validate_metrics(existing)
            all_warnings.extend(vw)
            return existing, all_warnings

        all_warnings.append(f"  данные устарели на {gap_days:.0f}д — докачка")

    # ── 2. Определить недостающие даты ──
    all_dates = _generate_dates(years)
    max_ts = existing["ts"].max() if existing is not None and not existing.empty else 0

    if max_ts > 0:
        cutoff = datetime.fromtimestamp(max_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        dates_to_fetch = [d for d in all_dates if d > cutoff]
    else:
        dates_to_fetch = all_dates

    if not dates_to_fetch and existing is not None and not existing.empty:
        existing, vw = _validate_metrics(existing)
        all_warnings.extend(vw)
        return existing, all_warnings

    # ── 3. Параллельная загрузка с S3 ──
    def _fetch_one(date_str):
        df = download_daily(symbol, date_str)
        if not df.empty:
            df["ts"] = df["create_time"]
        return date_str, df

    s3_dfs = []
    errors = 0
    with ThreadPoolExecutor(max_workers=config.VISION_WORKERS) as ex:
        futures = {ex.submit(_fetch_one, d): d for d in dates_to_fetch}
        for f in as_completed(futures):
            try:
                _, df = f.result()
            except Exception:
                errors += 1
                continue
            if df.empty:
                continue
            s3_dfs.append(df)

    if errors > 0:
        all_warnings.append(f"  {errors} ошибок загрузки")

    # ── 4. Склеить ──
    all_dfs = []
    if existing is not None and not existing.empty:
        all_dfs.append(existing)
    all_dfs.extend(s3_dfs)

    if not all_dfs:
        return pd.DataFrame(), all_warnings

    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["symbol", "ts"]).sort_values("ts").reset_index(drop=True)

    # ── 5. Tail ──
    if tail:
        max_ts = merged["ts"].max()

        oi_df = tail_open_interest_hist(symbol, start_time=max_ts + 1)
        if not oi_df.empty:
            oi_new = oi_df[oi_df["ts"] > max_ts]
            if not oi_new.empty:
                oi_new["symbol"] = symbol
                oi_new["create_time"] = oi_new["ts"]
                for col in ["count_toptrader_long_short_ratio",
                             "sum_toptrader_long_short_ratio",
                             "count_long_short_ratio",
                             "sum_taker_long_short_vol_ratio"]:
                    oi_new[col] = None
                merged = pd.concat([merged, oi_new], ignore_index=True)
                all_warnings.append(f"   tail OI: +{len(oi_new)} записей")

        tv_df = tail_taker_vol_ratio(symbol, start_time=max_ts + 1)
        if not tv_df.empty:
            tv_new = tv_df[tv_df["ts"] > max_ts]
            if not tv_new.empty:
                tv_new["symbol"] = symbol
                tv_new["create_time"] = tv_new["ts"]
                for col in ["sum_open_interest", "sum_open_interest_value",
                             "count_toptrader_long_short_ratio",
                             "sum_toptrader_long_short_ratio",
                             "count_long_short_ratio"]:
                    tv_new[col] = None
                merged = pd.concat([merged, tv_new], ignore_index=True)
                all_warnings.append(f"   tail taker vol: +{len(tv_new)} записей")

        merged = merged.drop_duplicates(subset=["symbol", "ts"]).sort_values("ts").reset_index(drop=True)

    # ── 6. Валидация ──
    if not merged.empty:
        merged, vw = _validate_metrics(merged)
        all_warnings.extend(vw)

    # ── 7. Загрузить в bucket ──
    _upload_to_bucket(symbol, merged)

    return merged, all_warnings


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
    parser.add_argument("--tail", action="store_true", help="Докачать хвост через API")
    args = parser.parse_args()

    t0 = time.time()
    for symbol in args.symbol:
        print(f"  {symbol}: загрузка metrics...", flush=True)
        df, warnings = fetch_metrics(symbol, args.years, force=args.force, tail=args.tail)
        for w in warnings:
            print(w)
        print(f"  {symbol}: {len(df):,} записей")
    print(f"\n  Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
