"""Получение OHLCV klines через Binance REST API (спот + перп).

Без API-ключа, бесплатно. Для хвоста, которого ещё нет в S3 архивах.

Использование (через fetch_klines.py):
    python -m data_fetcher ohlcv vision --symbol BTCUSDT --tail
"""

import pandas as pd


def fetch_tail(symbol, interval="1h", perp=False):
    """Достать последние 48 часов 1h-свечей через Binance REST API.

    Returns:
        pd.DataFrame с колонками open_time..taker_buy_quote (ts в ms).
        Пустой DataFrame при ошибке или отсутствии данных.
    """
    try:
        import requests as req
    except ImportError:
        return pd.DataFrame()

    base = "https://fapi.binance.com" if perp else "https://api.binance.com"
    endpoint = "/fapi/v1/klines" if perp else "/api/v3/klines"
    url = f"{base}{endpoint}?symbol={symbol}&interval={interval}&limit=48"

    try:
        resp = req.get(url, timeout=10)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
    except Exception:
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(data, columns=cols)
    for c in ("open_time", "close_time", "count"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_base", "taker_buy_quote"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Binance API всегда возвращает ms, но страхуемся от µs
    for col in ("open_time", "close_time"):
        mask = df[col] > 1e14
        df.loc[mask, col] = df.loc[mask, col] // 1000
    df["ts"] = df["open_time"]
    if "ignore" in df.columns:
        df = df.drop(columns=["ignore"])
    return df
