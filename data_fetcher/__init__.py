"""Data fetcher package.

Подпакеты:
  ccxt_api/         -- OHLCV через ccxt API (спот + перпы)
  binance_vision/   -- Исторические данные с data.binance.vision (S3)
  binance_api/      -- Binance REST API (exchangeInfo)
  websocket/        -- Реалтайм WebSocket pipeline (ccxt.pro)
"""

__all__ = [
    "discover_common_symbols",
    "run_download",
    "upload_to_huggingface",
    "get_exchange",
    "fetch_klines",
    "fetch_agg_trades",
    "fetch_book_depth",
    "fetch_funding",
    "fetch_symbols",
]


def __getattr__(name):
    if name in ("discover_common_symbols", "run_download", "upload_to_huggingface", "upload_to_bucket", "get_exchange"):
        from data_fetcher.ccxt_api.fetcher import (
            discover_common_symbols, run_download, upload_to_huggingface, upload_to_bucket, get_exchange,
        )
        _map = {
            "discover_common_symbols": discover_common_symbols,
            "run_download": run_download,
            "upload_to_huggingface": upload_to_huggingface,
            "upload_to_bucket": upload_to_bucket,
            "get_exchange": get_exchange,
        }
        return _map[name]
    if name == "fetch_klines":
        from data_fetcher.binance_vision.fetch_klines import fetch_symbol as fetch_klines
        return fetch_klines
    if name == "fetch_agg_trades":
        from data_fetcher.binance_vision.fetch_agg_trades import fetch_range as fetch_agg_trades
        return fetch_agg_trades
    if name == "fetch_book_depth":
        from data_fetcher.binance_vision.fetch_book_depth import fetch_range as fetch_book_depth
        return fetch_book_depth
    if name == "fetch_funding":
        from data_fetcher.binance_vision.fetch_funding import fetch_funding
        return fetch_funding
    if name == "fetch_symbols":
        from data_fetcher.binance_api.fetch_symbols import fetch_all as fetch_symbols
        return fetch_symbols
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
