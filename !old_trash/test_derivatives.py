"""Тест доступности Open Interest, Funding Rate, Long/Short Ratio через ccxt."""

import ccxt
import traceback
from datetime import datetime, timedelta, timezone

EXCHANGES = {
    "binance": ccxt.binance,
    "bybit": ccxt.bybit,
    "okx": ccxt.okx,
}

SYMBOL = "BTC/USDT:USDT"  # perpetual


def test_method(ex, method_name, symbol, **kwargs):
    """Вызвать метод и вернуть результат или ошибку."""
    try:
        fn = getattr(ex, method_name)
        result = fn(symbol, **kwargs)
        if isinstance(result, list):
            count = len(result)
            sample = result[0] if result else {}
            return True, f"{count} записей", sample
        elif isinstance(result, dict):
            return True, "OK", result
        else:
            return True, str(type(result).__name__), result
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", {}


def main():
    since = datetime.now(timezone.utc) - timedelta(days=7)
    since_ms = int(since.timestamp() * 1000)

    tests = [
        # (method, description, extra_kwargs)
        ("fetchOpenInterest", "Open Interest", {}),
        ("fetchFundingRate", "Funding Rate (текущая)", {}),
        ("fetchFundingRateHistory", "Funding Rate History (7 дней)", {"since": since_ms, "limit": 5}),
    ]

    # Long/Short Ratio — только OKX имеет этот метод в unified API
    lsr_tests = [
        ("fetchLongShortRatioHistory", "Long/Short Ratio (7 дней)", {"since": since_ms, "limit": 5}),
    ]

    print(f"Тест: {SYMBOL}")
    print(f"Период: {since.strftime('%Y-%m-%d')} — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    print("=" * 70)

    for ex_name, ex_cls in EXCHANGES.items():
        print(f"\n{'─' * 70}")
        print(f"  {ex_name.upper()}")
        print(f"{'─' * 70}")

        try:
            ex = ex_cls({"enableRateLimit": True})
            ex.load_markets()

            # Проверить что символ существует
            if SYMBOL not in ex.markets:
                print(f"  ❌ Символ {SYMBOL} не найден на {ex_name}")
                continue
        except Exception as e:
            print(f"  ❌ Ошибка инициализации: {e}")
            continue

        # Основные тесты
        for method, desc, kwargs in tests:
            ok, info, sample = test_method(ex, method, SYMBOL, **kwargs)
            icon = "✅" if ok else "❌"
            print(f"  {icon} {desc}: {info}")
            if ok and sample and isinstance(sample, dict):
                keys = list(sample.keys())[:5]
                if keys:
                    print(f"     Пример полей: {keys}")

        # Long/Short Ratio
        for method, desc, kwargs in lsr_tests:
            ok, info, sample = test_method(ex, method, SYMBOL, **kwargs)
            icon = "✅" if ok else "❌"
            print(f"  {icon} {desc}: {info}")
            if ok and sample and isinstance(sample, dict):
                keys = list(sample.keys())[:5]
                if keys:
                    print(f"     Пример полей: {keys}")

        # Дополнительно: попробовать long/short ratio под другими именами
        # OKX использует свой API для этого
        if ex_name == "okx":
            try:
                result = ex.fetch_long_short_ratio_history(SYMBOL, timeframe="1h", since=since_ms, limit=5)
                print(f"  ✅ Long/Short Ratio (OKX specific): {len(result)} записей")
            except Exception as e:
                print(f"  ❌ Long/Short Ratio (OKX specific): {type(e).__name__}: {e}")

    print(f"\n{'=' * 70}")
    print("Готово.")


if __name__ == "__main__":
    main()
