"""Бенчмарк: оптимальное количество воркеров для S3 архивов и REST API."""

import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

TIMEOUT = 30
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
WORKER_VALUES = [1, 2, 4, 8, 16, 32]

# ── S3 архивы ─────────────────────────────────────────────────


def _download_monthly(symbol, year, month):
    url = (f"https://data.binance.vision/data/spot/monthly/klines/"
           f"{symbol}/1h/{symbol}-1h-{year}-{month:02d}.zip")
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        if resp.status_code != 200:
            return 0
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                lines = f.read().decode("utf-8").strip().split("\n")
        return len(lines)
    except Exception:
        return 0


def benchmark_s3(symbols, months, max_workers):
    """Скачать monthly архивы для всех symbols × months."""
    tasks = [(s, y, m) for s in symbols for y, m in months]
    t0 = time.time()
    results = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_download_monthly, s, y, m): (s, y, m)
                   for s in symbols for y, m in months}
        for f in as_completed(futures):
            rows = f.result()
            results += rows
            done += 1
    elapsed = time.time() - t0
    return elapsed, results


# ── REST API (tail) ────────────────────────────────────────────


def _fetch_klines(symbol, limit=100):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "1h", "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        if resp.status_code == 200:
            return len(resp.json())
        return 0
    except Exception:
        return 0


def benchmark_rest(symbols, limit, max_workers):
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_klines, s, limit): s for s in symbols}
        for f in as_completed(futures):
            results.append(f.result())
    elapsed = time.time() - t0
    return elapsed, sum(results)


# ── Main ────────────────────────────────────────────────────────


def main():
    print("Бенчмарк параллельных запросов\n")

    # ── 1. S3 архивы ──
    months = [(y, m) for y in range(2025, 2026) for m in range(1, 13)]
    n_files = len(SYMBOLS) * len(months)
    print(f"S3: {len(SYMBOLS)} символов × {len(months)} месяцев = {n_files} monthly архивов")
    print(f"Таймаут {TIMEOUT}s\n")
    print(f"{'workers':>8} | {'time':>8} | {'rows':>8} | {'speed':>10}")
    print("-" * 40)
    best_s3 = (0, float("inf"))
    for w in WORKER_VALUES:
        elapsed, rows = benchmark_s3(SYMBOLS, months, w)
        speed = rows / elapsed if elapsed > 0 else 0
        print(f"{w:>8} | {elapsed:>7.1f}s | {rows:>8} | {speed:>8.0f} rows/s")
        if elapsed < best_s3[1] and elapsed > 0:
            best_s3 = (w, elapsed)

    # ── 2. REST API ──
    print(f"\nREST API: {len(SYMBOLS)} символов, klines(limit=100)\n")
    print(f"{'workers':>8} | {'time':>8} | {'bars':>8} | {'speed':>10}")
    print("-" * 40)
    best_rest = (0, float("inf"))
    for w in WORKER_VALUES:
        elapsed, bars = benchmark_rest(SYMBOLS, 100, w)
        speed = bars / elapsed if elapsed > 0 else 0
        print(f"{w:>8} | {elapsed:>7.1f}s | {bars:>8} | {speed:>8.0f} bars/s")
        if elapsed < best_rest[1] and elapsed > 0:
            best_rest = (w, elapsed)

    # ── Итоги ──
    print(f"\n{'=' * 40}")
    print(f"S3:     оптимально {best_s3[0]} воркеров ({best_s3[1]:.1f}s)")
    print(f"REST:   оптимально {best_rest[0]} воркеров ({best_rest[1]:.1f}s)")
    print(f"{'=' * 40}")
    print(f"\nРекомендация для config.py:")
    print(f"  VISION_WORKERS = {best_s3[0]}")
    print(f"  WORKERS = {best_rest[0]}")


if __name__ == "__main__":
    main()
