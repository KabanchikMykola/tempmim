"""Бенчмарк: сколько параллельных загрузок выдерживает Binance API."""

import ccxt
import time
import sys
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

TEST_SYMBOLS_SPOT = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
TEST_SYMBOLS_PERP = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT"]
SINCE_MS = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
TIMEFRAME = "1h"


def make_exchange(label: str, default_type: str = "spot") -> ccxt.binance:
    config = {"enableRateLimit": True}
    if default_type != "spot":
        config["options"] = {"defaultType": default_type}
    return ccxt.binance(config)


def fetch_one(exchange: ccxt.binance, symbol: str) -> tuple[str, int, float]:
    """Скачать одну пару. Возвращает (symbol, bars, elapsed_sec)."""
    t0 = time.time()
    all_candles = []
    cursor = SINCE_MS
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=cursor, limit=1000)
        except Exception as e:
            return symbol, 0, time.time() - t0
        if not candles:
            break
        all_candles.extend(candles)
        if len(candles) < 1000:
            break
        cursor = candles[-1][0] + 1
        time.sleep(0.1)
    return symbol, len(all_candles), time.time() - t0


def benchmark_sequential(exchange: ccxt.binance, symbols: list[str], label: str) -> float:
    """Последовательная загрузка. Возвращает общее время."""
    print(f"\n{'='*60}")
    print(f"[{label}] Последовательно: {len(symbols)} символов")
    print(f"{'='*60}")
    t0 = time.time()
    for s in symbols:
        _, bars, elapsed = fetch_one(exchange, s)
        print(f"  {s:>20}: {bars:>5} баров | {elapsed:.1f}s")
    total = time.time() - t0
    print(f"  ИТОГО: {total:.1f}s")
    return total


def benchmark_parallel(
    exchanges: list[ccxt.binance],
    symbols: list[str],
    max_workers: int,
    label: str,
) -> float:
    """Параллельная загрузка. Возвращает общее время."""
    print(f"\n{'='*60}")
    print(f"[{label}] Параллельно ({max_workers} воркеров): {len(symbols)} символов")
    print(f"{'='*60}")

    symbol_exchanges = []
    for i, s in enumerate(symbols):
        ex = exchanges[i % len(exchanges)]
        symbol_exchanges.append((ex, s))

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_one, ex, s): s for ex, s in symbol_exchanges}
        for future in as_completed(futures):
            symbol, bars, elapsed = future.result()
            results.append((symbol, bars, elapsed))
            print(f"  {symbol:>20}: {bars:>5} баров | {elapsed:.1f}s")

    total = time.time() - t0
    print(f"  ИТОГО: {total:.1f}s")
    return total


def main():
    print("Бенчмарк Binance API: параллелизм и лимиты\n")
    print(f"Таймфрейм: {TIMEFRAME}, с: 2026-01-01")
    print(f"Символов для теста: {len(TEST_SYMBOLS_SPOT)} spot + {len(TEST_SYMBOLS_PERP)} perp")

    ex_spot = make_exchange("spot", "spot")
    ex_spot.load_markets()
    ex_perp = make_exchange("perp", "future")
    ex_perp.load_markets()

    results = {}

    # 1. Спот последовательно
    results["spot_seq"] = benchmark_sequential(ex_spot, TEST_SYMBOLS_SPOT, "SPOT sequential")

    # 2. Перпы последовательно
    results["perp_seq"] = benchmark_sequential(ex_perp, TEST_SYMBOLS_PERP, "PERP sequential")

    # 3. Спот+перпы через один exchange (как сейчас)
    all_symbols_mixed = TEST_SYMBOLS_SPOT + TEST_SYMBOLS_PERP
    results["mixed_seq"] = benchmark_sequential(ex_spot, all_symbols_mixed, "MIXED sequential (one exchange)")

    # 4. Параллельно спот, 2 воркера
    results["spot_par2"] = benchmark_parallel([ex_spot], TEST_SYMBOLS_SPOT, 2, "SPOT parallel-2")

    # 5. Параллельно спот, 4 воркера
    results["spot_par4"] = benchmark_parallel([ex_spot], TEST_SYMBOLS_SPOT, 4, "SPOT parallel-4")

    # 6. Параллельно спот, 8 воркеров
    results["spot_par8"] = benchmark_parallel([ex_spot], TEST_SYMBOLS_SPOT, 8, "SPOT parallel-8")

    # 7. Параллельно спот+перпы, разные exchange, 4 воркера
    results["split_par4"] = benchmark_parallel(
        [ex_spot, ex_perp], TEST_SYMBOLS_SPOT + TEST_SYMBOLS_PERP, 4, "SPLIT parallel-4 (spot+perp)"
    )

    # 8. Параллельно спот+перпы, разные exchange, 8 воркеров
    results["split_par8"] = benchmark_parallel(
        [ex_spot, ex_perp], TEST_SYMBOLS_SPOT + TEST_SYMBOLS_PERP, 8, "SPLIT parallel-8 (spot+perp)"
    )

    # Итоги
    print(f"\n{'='*60}")
    print("ИТОГИ")
    print(f"{'='*60}")
    for name, t in results.items():
        symbols_count = len(TEST_SYMBOLS_SPOT) if "spot" in name and "perp" not in name else (
            len(TEST_SYMBOLS_PERP) if "perp" in name else len(TEST_SYMBOLS_SPOT) + len(TEST_SYMBOLS_PERP)
        )
        speed = symbols_count / t if t > 0 else 0
        print(f"  {name:<20}: {t:>6.1f}s | {speed:.1f} символов/сек")


if __name__ == "__main__":
    main()
