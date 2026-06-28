"""Докачка агрегированных данных через Binance REST API.

Каждая функция возвращает DataFrame с колонками, совместимыми с архивами Binance Vision.
"""
import time
import pandas as pd
from data_fetcher import config

API_SPOT = "https://api.binance.com"
API_FUTURES = "https://fapi.binance.com"

RETRIES = 3
TIMEOUT = 15


def _request(url, params):
    for attempt in range(RETRIES):
        try:
            import requests as req
            resp = req.get(url, params=params, timeout=TIMEOUT)
            if resp.status_code == 429:
                time.sleep(int(resp.headers.get("Retry-After", 5)))
                continue
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            if attempt == RETRIES - 1:
                return None
            time.sleep(1)
    return None


# ── OHLCV ──────────────────────────────────────────────────────────

def tail_klines(symbol, interval, perp=False, start_time=None, end_time=None, limit=1000):
    """Докачка OHLCV через Binance REST API.

    Возвращает DataFrame с колонками, совместимыми с архивами:
      open_time, open, high, low, close, volume, close_time,
      quote_volume, count, taker_buy_base, taker_buy_quote, ts
    """
    base = API_FUTURES if perp else API_SPOT
    endpoint = "/fapi/v1/klines" if perp else "/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500 if perp else 1000)}
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)

    data = _request(f"{base}{endpoint}", params)
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

    for col in ("open_time", "close_time"):
        mask = df[col] > 1e14
        df.loc[mask, col] = df.loc[mask, col] // 1000
    df["ts"] = df["open_time"]
    if "ignore" in df.columns:
        df = df.drop(columns=["ignore"])
    return df


# ── Funding Rate ───────────────────────────────────────────────────

_FUNDING_INTERVAL_CACHE = {}

def _funding_interval(symbol):
    """Получить fundingIntervalHours для символа (из кеша или API)."""
    if symbol in _FUNDING_INTERVAL_CACHE:
        return _FUNDING_INTERVAL_CACHE[symbol]

    data = _request(f"{API_FUTURES}/fapi/v1/fundingInfo", {})
    if data:
        for entry in data:
            if entry.get("symbol") == symbol:
                val = entry.get("fundingIntervalHours", 8)
                _FUNDING_INTERVAL_CACHE[symbol] = val
                return val
    _FUNDING_INTERVAL_CACHE[symbol] = 8
    return 8


def tail_funding(symbol, start_time=None, end_time=None, limit=1000):
    """Докачка funding rate через Binance API.

    Возвращает DataFrame с колонками, совместимыми с архивами:
      symbol, calc_time, funding_interval_hours, last_funding_rate, ts
    """
    params = {"symbol": symbol, "limit": min(limit, 1000)}
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)

    data = _request(f"{API_FUTURES}/fapi/v1/fundingRate", params)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["calc_time"] = pd.to_numeric(df["fundingTime"], errors="coerce")
    df["last_funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["ts"] = df["calc_time"]
    df["funding_interval_hours"] = _funding_interval(symbol)
    return df[["symbol", "calc_time", "funding_interval_hours", "last_funding_rate", "ts"]]


# ── Metrics ────────────────────────────────────────────────────────

def tail_open_interest_hist(symbol, period="5m", start_time=None, end_time=None, limit=500):
    """Докачка Open Interest History через Binance API.

    Возвращает DataFrame с колонками, совместимыми с архивами:
      symbol, sum_open_interest, sum_open_interest_value, create_time (как ts)
    """
    params = {"symbol": symbol, "period": period, "limit": min(limit, 500)}
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)

    data = _request(f"{API_FUTURES}/futures/data/openInterestHist", params)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["sum_open_interest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    df["sum_open_interest_value"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
    df["ts"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["symbol"] = symbol
    return df[["symbol", "sum_open_interest", "sum_open_interest_value", "ts"]]


def tail_taker_vol_ratio(symbol, period="5m", start_time=None, end_time=None, limit=500):
    """Докачка Taker Long/Short Volume Ratio через Binance API.

    Возвращает DataFrame с колонками, совместимыми с архивами:
      symbol, sum_taker_long_short_vol_ratio, ts
    """
    params = {"symbol": symbol, "period": period, "limit": min(limit, 500)}
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)

    data = _request(f"{API_FUTURES}/futures/data/takerlongshortRatio", params)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["sum_taker_long_short_vol_ratio"] = pd.to_numeric(df["buySellRatio"], errors="coerce")
    df["ts"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["symbol"] = symbol
    # buyVol/sellVol — дополнительные колонки (нет в архивах, но полезны)
    df["taker_buy_vol"] = pd.to_numeric(df["buyVol"], errors="coerce")
    df["taker_sell_vol"] = pd.to_numeric(df["sellVol"], errors="coerce")
    return df[["symbol", "sum_taker_long_short_vol_ratio", "taker_buy_vol", "taker_sell_vol", "ts"]]


def tail_top_long_short_ratio(symbol, period="5m", start_time=None, end_time=None, limit=500):
    """Докачка Top Trader Long/Short Account Ratio через Binance API.

    Возвращает DataFrame с колонками (отличными от архивов):
      symbol, top_long_account, top_short_account, top_long_short_ratio, ts
    """
    params = {"symbol": symbol, "period": period, "limit": min(limit, 500)}
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)

    data = _request(f"{API_FUTURES}/futures/data/topLongShortAccountRatio", params)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["top_long_account"] = pd.to_numeric(df["longAccount"], errors="coerce")
    df["top_short_account"] = pd.to_numeric(df["shortAccount"], errors="coerce")
    df["top_long_short_ratio"] = pd.to_numeric(df["longShortRatio"], errors="coerce")
    df["ts"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["symbol"] = symbol
    return df[["symbol", "top_long_account", "top_short_account", "top_long_short_ratio", "ts"]]
