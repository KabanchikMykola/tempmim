"""Конфигурация загрузки данных."""

from pathlib import Path
from datetime import datetime


# === Фильтрация символов ===
MIN_VOLUME_USD = 1_000_000          # Мин. суммарный 24h объём (спот + перп)
QUOTE = "USDT"                       # Quote валюта

# === Временные параметры ===
SINCE = "2026-01-01"                 # Дата начала
TIMEFRAME = "1h"                     # Таймфрейм

# === Параллелизм ===
WORKERS = 8                          # Количество параллельных воркеров (ccxt API)

# === Хранение ===
DATA_DIR = Path("fin_data")          # Корневая папка для всех данных
HUGGINGFACE_REPO = None             # Репозиторий HF (None = не загружать)
BUCKET_ID = "Kabanchik/mimo"          # HuggingFace Bucket ID (можно переопределить через env HF_BUCKET_ID)
BUCKET_PREFIX = "fin_data"           # Префикс внутри bucket (папка)
CACHE_DIR = DATA_DIR / "cache"       # DuckDB кеш (binance_vision.db)

# === Binance Vision (data.binance.vision S3 архивы) ===
VISION_WORKERS = 16                  # Потоков для параллельной загрузки
VISION_TIMEOUT = 30                  # Таймаут HTTP запроса (сек)


# ── Helpers для построения путей ────────────────────────────────────


def _market_dir(source: str) -> str:
    """ohlcv_spot или ohlcv_perp."""
    return "ohlcv_perp" if source == "perp" else "ohlcv_spot"


def ohlcv_path(symbol: str, interval: str, source: str, year: int | None = None) -> Path:
    """Путь к годовому parquet OHLCV.

    fin_data/binance/ohlcv_spot/BTCUSDT_1h_2025.parquet
    """
    if year is None:
        year = datetime.now().year
    return DATA_DIR / "binance" / _market_dir(source) / f"{symbol}_{interval}_{year}.parquet"


def ohlcv_pattern(symbol: str, interval: str, source: str) -> str:
    """Glob-шаблон для чтения всех годов.

    fin_data/binance/ohlcv_spot/BTCUSDT_1h_*.parquet
    """
    return str(DATA_DIR / "binance" / _market_dir(source) / f"{symbol}_{interval}_*.parquet")


def ohlcv_bucket_uri(symbol: str, interval: str, source: str, year: int) -> str:
    """URI в HF Bucket."""
    return f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}/binance/{_market_dir(source)}/{symbol}_{interval}_{year}.parquet"


def funding_path(symbol: str) -> Path:
    """Путь к parquet funding rate.

    fin_data/binance/funding/BTCUSDT_funding.parquet
    """
    return DATA_DIR / "binance" / "funding" / f"{symbol}_funding.parquet"


def funding_bucket_uri(symbol: str) -> str:
    return f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}/binance/funding/{symbol}_funding.parquet"


def metrics_path(symbol: str) -> Path:
    """Путь к parquet metrics.

    fin_data/binance/metrics/BTCUSDT_metrics.parquet
    """
    return DATA_DIR / "binance" / "metrics" / f"{symbol}_metrics.parquet"


def metrics_bucket_uri(symbol: str) -> str:
    return f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}/binance/metrics/{symbol}_metrics.parquet"

