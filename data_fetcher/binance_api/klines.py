"""Получение OHLCV klines через Binance REST API (спот + перп).

Без API-ключа, бесплатно. Для хвоста, которого ещё нет в S3 архивах.

Использование (через fetch_klines.py):
    python -m data_fetcher ohlcv vision --symbol BTCUSDT --tail
"""

from data_fetcher.binance_api.tail import tail_klines


def fetch_tail(symbol, interval="1h", perp=False, limit=1000, end_time=None):
    """Достать klines через Binance REST API.

    Args:
        symbol: Тикер (BTCUSDT)
        interval: Таймфрейм (1h, 5m, 1d...)
        perp: True = перпетуалы, False = спот
        limit: Максимум баров (1000 spot / 1500 perp)
        end_time: Верхняя граница в ms (None = сейчас)

    Returns:
        pd.DataFrame с колонками open_time..taker_buy_quote (ts в ms).
        Пустой DataFrame при ошибке.
    """
    return tail_klines(symbol=symbol, interval=interval, perp=perp, limit=limit, end_time=end_time)
