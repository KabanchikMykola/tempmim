"""Тест глубины истории: OHLCV, Funding Rate, Long/Short Ratio."""

import ccxt
from datetime import datetime, timezone

EXCHANGES = {
    "binance": ccxt.binance,
    "bybit": ccxt.bybit,
    "okx": ccxt.okx,
}

SYMBOLS = {
    "binance": "BTC/USDT:USDT",
    "bybit": "BTC/USDT:USDT",
    "okx": "BTC/USDT:USDT",
}

TIMEFRAMES = ["1h", "4h", "1d"]


def find_oldest_ohlcv(ex, symbol, timeframe, ex_name=""):
    # OKX не отдаёт данные с 2019, начинаем с 2020
    start_year = 2020 if ex_name == "okx" else 2019
    start_ts = int(datetime(start_year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    try:
        candles = ex.fetch_ohlcv(symbol, timeframe, since=start_ts, limit=1)
        if candles:
            dt = datetime.fromtimestamp(candles[0][0] / 1000, tz=timezone.utc)
            return True, dt.strftime("%Y-%m-%d")
        return False, "нет данных"
    except Exception as e:
        return False, f"{type(e).__name__}"


def find_funding_range(ex, symbol):
    """Найти диапазон фандинга, пробуя разные лимиты."""
    for limit in [1000, 500, 200, 100, 50, 20, 10, 5]:
        try:
            results = ex.fetch_funding_rate_history(symbol, limit=limit)
            if results and len(results) > 0:
                oldest = min(r["timestamp"] for r in results)
                newest = max(r["timestamp"] for r in results)
                dt_o = datetime.fromtimestamp(oldest / 1000, tz=timezone.utc)
                dt_n = datetime.fromtimestamp(newest / 1000, tz=timezone.utc)
                return True, f"{dt_o.strftime('%Y-%m-%d')} — {dt_n.strftime('%Y-%m-%d')} ({len(results)} записей, limit={limit})"
        except Exception:
            continue
    return False, "нет данных"


def find_lsr_range(ex, symbol):
    """Найти диапазон Long/Short Ratio."""
    for limit in [1000, 500, 200, 100, 50, 20, 10, 5]:
        try:
            results = ex.fetch_long_short_ratio_history(symbol, limit=limit)
            if results and len(results) > 0:
                oldest = min(r["timestamp"] for r in results)
                newest = max(r["timestamp"] for r in results)
                dt_o = datetime.fromtimestamp(oldest / 1000, tz=timezone.utc)
                dt_n = datetime.fromtimestamp(newest / 1000, tz=timezone.utc)
                return True, f"{dt_o.strftime('%Y-%m-%d')} — {dt_n.strftime('%Y-%m-%d')} ({len(results)} записей, limit={limit})"
        except Exception:
            continue
    return False, "нет данных"


def main():
    print(f"Тест глубины истории")
    print("=" * 80)

    for ex_name, ex_cls in EXCHANGES.items():
        symbol = SYMBOLS[ex_name]
        print(f"\n{'━' * 80}")
        print(f"  {ex_name.upper()} | {symbol}")
        print(f"{'━' * 80}")

        try:
            ex = ex_cls({"enableRateLimit": True})
            ex.load_markets()
            if symbol not in ex.markets:
                print(f"  ❌ {symbol} не найден")
                continue
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            continue

        print(f"\n  📊 OHLCV:")
        for tf in TIMEFRAMES:
            ok, info = find_oldest_ohlcv(ex, symbol, tf, ex_name)
            icon = "✅" if ok else "❌"
            print(f"     {icon} {tf:>4}: {info}")

        print(f"\n  💰 Funding Rate:")
        ok, info = find_funding_range(ex, symbol)
        icon = "✅" if ok else "❌"
        print(f"     {icon} {info}")

        print(f"\n  📈 Long/Short Ratio:")
        ok, info = find_lsr_range(ex, symbol)
        icon = "✅" if ok else "❌"
        print(f"     {icon} {info}")

    print(f"\n{'=' * 80}")
    print("Готово.")


if __name__ == "__main__":
    main()
