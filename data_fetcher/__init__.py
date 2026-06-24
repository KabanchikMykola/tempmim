"""Data fetcher package.

Подпакеты:
  ccxt_api/         -- OHLCV через ccxt API (спот + перпы)
  binance_vision/   -- Исторические данные с data.binance.vision (S3)
  binance_api/      -- Binance REST API (exchangeInfo)
  websocket/        -- Реалтайм WebSocket pipeline (ccxt.pro)
"""

from data_fetcher.ccxt_api.fetcher import (
    discover_common_symbols,
    run_download,
    upload_to_huggingface,
    upload_to_bucket,
    get_exchange,
)
from data_fetcher.binance_vision.fetch_klines import fetch_symbol as fetch_klines
from data_fetcher.binance_vision.fetch_agg_trades import fetch_range as fetch_agg_trades
from data_fetcher.binance_vision.fetch_book_depth import fetch_range as fetch_book_depth
from data_fetcher.binance_vision.fetch_funding import fetch_funding
from data_fetcher.binance_api.fetch_symbols import fetch_all as fetch_symbols

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
