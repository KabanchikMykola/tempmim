"""Конфигурация загрузки данных."""

from pathlib import Path

# === Фильтрация символов ===
MIN_VOLUME_USD = 1_000_000          # Мин. суммарный 24h объём (спот + перп)
QUOTE = "USDT"                       # Quote валюта

# === Временные параметры ===
SINCE = "2026-01-01"                 # Дата начала
TIMEFRAME = "1h"                     # Таймфрейм

# === Параллелизм ===
WORKERS = 8                          # Количество параллельных воркеров (ccxt API)

# === Хранение ===
DATA_DIR = Path("data")              # Корневая папка для всех данных
HUGGINGFACE_REPO = None             # Репозиторий HF (None = не загружать)

CACHE_DIR = DATA_DIR / "cache"       # DuckDB кеш (binance_vision.db)

# === Binance Vision (data.binance.vision S3 архивы) ===
VISION_WORKERS = 16                  # Потоков для параллельной загрузки
VISION_TIMEOUT = 30                  # Таймаут HTTP запроса (сек)

