"""Загрузка исторических funding rate и перпетуальных klines с data.binance.vision.

Фандинг: месячные архивы (fundingRate).
Перп klines: дневные архивы (нужны для анализа базы/спреда).

Использование:
    python -m data_fetcher funding --symbol BTCUSDT
    python -m data_fetcher funding --symbol ETHUSDT --years 2
"""

import io
import zipfile
import time
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd
import duckdb

from data_fetcher import config


def _db_path():
    return Path(config.CACHE_DIR) / "binance_vision.db"


def _parquet_path_funding(symbol):
    return Path(config.DATA_DIR) / f"{symbol}_funding.parquet"


def _parquet_path_perp(symbol, interval):
    return Path(config.DATA_DIR) / f"{symbol}_perp_{interval}.parquet"


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


def download_perp_klines_daily(symbol, interval, year, month):
    """Скачать дневные архивы klines для перпетуалов за месяц.

    data.binance.vision не хранит месячные архивы perp klines,
    поэтому качаем поденно.
    """
    import calendar
    all_dfs = []
    days_in_month = calendar.monthrange(year, month)[1]
    for day in range(1, days_in_month + 1):
        date_str = f"{year}-{month:02d}-{day:02d}"
        url = (f"https://data.binance.vision/data/futures/um/daily/klines/"
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
                        if lines:
                            data = [line.split(",") for line in lines]
                            df = pd.DataFrame(data, columns=[
                                "open_time", "open", "high", "low", "close", "volume",
                                "close_time", "quote_volume", "count",
                                "taker_buy_base", "taker_buy_quote", "ignore"
                            ])
                            for col in ["open_time", "close_time", "count"]:
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                            for col in ["open", "high", "low", "close", "volume",
                                        "quote_volume", "taker_buy_base", "taker_buy_quote"]:
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                            all_dfs.append(df)
        except Exception:
            pass
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


def fetch_funding(symbol, years=3, export_parquet=True):
    """Загрузить funding rate для символа. DuckDB кеш + Parquet.

    Args:
        symbol: Тикер (BTCUSDT, ETHUSDT...)
        years: Сколько лет
        export_parquet: Экспорт в Parquet

    Returns:
        pd.DataFrame с колонками symbol, calc_time, funding_interval_hours, last_funding_rate
    """
    db = _db_path()
    pq = _parquet_path_funding(symbol)
    db.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS funding_rates (
            symbol VARCHAR, calc_time BIGINT, funding_interval_hours INTEGER,
            last_funding_rate DOUBLE, ts BIGINT
        )
    """)

    existing = conn.execute(
        "SELECT COUNT(*) FROM funding_rates WHERE symbol=?",
        [symbol]
    ).fetchone()[0]

    if existing > 1000:
        df = conn.execute(
            "SELECT * FROM funding_rates WHERE symbol=? ORDER BY ts",
            [symbol]
        ).fetchdf()
        conn.close()
        return df

    now = datetime.now()
    for year in range(now.year - years, now.year + 1):
        for month in range(1, 13):
            if year == now.year and month > now.month:
                break
            df_m = download_funding_monthly(symbol, year, month)
            if not df_m.empty:
                df_m["ts"] = pd.to_numeric(df_m["calc_time"], errors="coerce")
                conn.execute("""
                    INSERT INTO funding_rates
                    SELECT symbol, calc_time, funding_interval_hours, last_funding_rate, ts FROM df_m
                """)
        time.sleep(0.2)

    df = conn.execute(
        "SELECT * FROM funding_rates WHERE symbol=? ORDER BY ts",
        [symbol]
    ).fetchdf()
    conn.close()

    if export_parquet and not df.empty:
        pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq, index=False)

    return df


def fetch_perp_klines(symbol, interval="15m", years=3, export_parquet=True):
    """Загрузить перпетуальные klines. DuckDB кеш + Parquet.

    Args:
        symbol: Тикер
        interval: Таймфрейм (15m, 1h, 4h...)
        years: Сколько лет
        export_parquet: Экспорт в Parquet

    Returns:
        pd.DataFrame с OHLCV данными
    """
    db = _db_path()
    pq = _parquet_path_perp(symbol, interval)
    db.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS perp_klines (
            symbol VARCHAR, interval VARCHAR, open_time BIGINT,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume DOUBLE, close_time BIGINT, quote_volume DOUBLE,
            count INTEGER, taker_buy_base DOUBLE, taker_buy_quote DOUBLE, ts BIGINT
        )
    """)

    existing = conn.execute(
        "SELECT COUNT(*) FROM perp_klines WHERE symbol=? AND interval=?",
        [symbol, interval]
    ).fetchone()[0]

    if existing > 1000:
        df = conn.execute(
            "SELECT * FROM perp_klines WHERE symbol=? AND interval=? ORDER BY ts",
            [symbol, interval]
        ).fetchdf()
        conn.close()
        return df

    now = datetime.now()
    for year in range(now.year - years, now.year + 1):
        for month in range(1, 13):
            if year == now.year and month > now.month:
                break
            df_m = download_perp_klines_daily(symbol, interval, year, month)
            if not df_m.empty:
                df_m["symbol"] = symbol
                df_m["interval"] = interval
                df_m["ts"] = df_m["open_time"] * 1000
                conn.execute("""
                    INSERT INTO perp_klines
                    SELECT symbol, interval, open_time, open, high, low, close, volume,
                           close_time, quote_volume, count, taker_buy_base, taker_buy_quote, ts
                    FROM df_m
                """)

    df = conn.execute(
        "SELECT * FROM perp_klines WHERE symbol=? AND interval=? ORDER BY ts",
        [symbol, interval]
    ).fetchdf()
    conn.close()

    if export_parquet and not df.empty:
        pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq, index=False)

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Загрузка funding rate с data.binance.vision")
    parser.add_argument("--symbol", nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="Тикеры (по умолчанию BTCUSDT ETHUSDT)")
    parser.add_argument("--years", type=int, default=3, help="Сколько лет")
    parser.add_argument("--klines", action="store_true",
                        help="Также загрузить перп klines (15m)")
    args = parser.parse_args()

    t0 = time.time()
    for symbol in args.symbol:
        print(f"  {symbol}: загрузка funding...", end=" ", flush=True)
        df = fetch_funding(symbol, args.years)
        print(f"{len(df):,} записей")
        if args.klines:
            print(f"  {symbol}: загрузка perp klines...", end=" ", flush=True)
            df_k = fetch_perp_klines(symbol, "15m", args.years)
            print(f"{len(df_k):,} баров")

    print(f"\n  Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
