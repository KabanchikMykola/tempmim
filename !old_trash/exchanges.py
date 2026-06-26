import ccxt
import pandas as pd
from typing import Literal

ExchangeName = Literal["binance", "bybit", "okx"]

EXCHANGES: dict[ExchangeName, type] = {
    "binance": ccxt.binance,
    "bybit": ccxt.bybit,
    "okx": ccxt.okx,
}


def fetch_markets(exchange_name: ExchangeName) -> pd.DataFrame:
    """Загрузить рынки с биржи."""
    ex_cls = EXCHANGES[exchange_name]
    ex = ex_cls({"enableRateLimit": True})
    markets = ex.fetch_markets()

    rows = []
    for m in markets:
        rows.append({
            "exchange": exchange_name,
            "symbol": m["symbol"],
            "base": m["base"],
            "quote": m["quote"],
            "market_type": m["type"],
            "active": m.get("active", True),
            "id": m.get("id", ""),
        })
    return pd.DataFrame(rows)


def fetch_tickers(exchange_name: ExchangeName) -> pd.DataFrame:
    """Загрузить тикеры (цены + объём) с биржи."""
    ex_cls = EXCHANGES[exchange_name]
    ex = ex_cls({"enableRateLimit": True})
    tickers = ex.fetch_tickers()

    rows = []
    for symbol, t in tickers.items():
        base = symbol.split("/")[0].split(":")[0]
        rows.append({
            "exchange": exchange_name,
            "symbol": symbol,
            "base": base,
            "last": t.get("last", 0),
            "quote_volume_24h": t.get("quoteVolume", 0) or 0,
            "base_volume_24h": t.get("baseVolume", 0) or 0,
        })
    return pd.DataFrame(rows)


def fetch_all_markets() -> pd.DataFrame:
    """Загрузить рынки со всех 3 бирж."""
    frames = []
    for name in EXCHANGES:
        try:
            frames.append(fetch_markets(name))
        except Exception as e:
            print(f"Ошибка загрузки рынков {name}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_all_tickers() -> pd.DataFrame:
    """Загрузить тикеры со всех 3 бирж."""
    frames = []
    for name in EXCHANGES:
        try:
            frames.append(fetch_tickers(name))
        except Exception as e:
            print(f"Ошибка загрузки тикеров {name}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
