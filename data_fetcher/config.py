"""Конфигурация проекта — только переменные."""

import os
from pathlib import Path

# HF токен для записи в bucket (задаётся через HF_TOKEN env, иначе чтение без записи)
# Если не задан, bucket работает только на чтение.


# === Фильтрация символов ===
MIN_VOLUME_USD = 1_000_000
QUOTE = "USDT"

# === Параллелизм ===
WORKERS = 8          # REST API tail (бенчмарк: 8 оптимально/стабильно)
VISION_WORKERS = 32  # S3 архивы (бенчмарк: 32 оптимально)
VISION_TIMEOUT = 30

# === Хранение ===
DATA_DIR = Path("fin_data")
BUCKET_ID = "Kabanchik/mimo"
BUCKET_PREFIX = "fin_data"
CACHE_DIR = DATA_DIR / "cache"

# === Binance market dirs ===
MARKET_DIRS = {
    "spot": DATA_DIR / "binance" / "ohlcv_spot",
    "perp": DATA_DIR / "binance" / "ohlcv_perp",
    "funding": DATA_DIR / "binance" / "funding",
    "metrics": DATA_DIR / "binance" / "metrics",
}

# === HF bucket URIs ===
BUCKET_URI = f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}/binance"


def list_bucket(prefix: str = "") -> list[dict]:
    """Список файлов в HF bucket."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}/"
    pattern = base + prefix + "**/*.parquet"
    results = []
    for f in fs.glob(pattern):
        info = fs.info(f)
        rel = f.replace(base, "")
        results.append({"path": rel, "size_bytes": info["size"]})
    return results
