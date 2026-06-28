"""Data fetcher package.

Подпакеты:
  ccxt_api/         -- OHLCV через ccxt API (спот + перпы)
  binance_vision/   -- Исторические данные с data.binance.vision (S3)
  binance_api/      -- Binance REST API (exchangeInfo)
"""

__all__ = [
    "fetch_klines",
    "fetch_funding",
    "fetch_metrics",
    "fetch_symbols",
]


def __getattr__(name):
    if name == "fetch_klines":
        from data_fetcher.binance_vision.fetch_klines import fetch_symbol as fetch_klines
        return fetch_klines
    if name == "fetch_funding":
        from data_fetcher.binance_vision.fetch_funding import fetch_funding
        return fetch_funding
    if name == "fetch_metrics":
        from data_fetcher.binance_vision.fetch_metrics import fetch_metrics
        return fetch_metrics
    if name == "fetch_symbols":
        from data_fetcher.binance_api.fetch_symbols import fetch_all as fetch_symbols
        return fetch_symbols
    raise AttributeError(f"{__name__!r} has no attribute {name!r}")
