"""Единая точка входа для загрузки данных Binance.

Фиксирует end_time на старте батча, включает tail,
управляет upload в bucket. Вызывается из menu.py и __main__.py.
"""

import time
from datetime import datetime, timezone, timedelta

from data_fetcher import config


def run_download(symbols, years, types, upload=True, tail=True):
    """Загрузить данные по списку символов и типов.

    Args:
        symbols: список тикеров (BTCUSDT, ETHUSDT...)
        years: сколько лет
        types: {"spot": bool, "perp": bool, "funding": bool, "metrics": bool}
        upload: загружать в bucket
        tail: докачать хвост через REST API
    """
    from data_fetcher.binance_vision.fetch_klines import fetch_symbol as fetch_klines
    from data_fetcher.binance_vision.fetch_funding import fetch_funding
    from data_fetcher.binance_vision.fetch_metrics import fetch_metrics

    # зафиксировать границы один раз на весь батч
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=365 * years)
    bucket_id = config.BUCKET_ID if upload else None

    active = [k for k, v in types.items() if v]
    print(f"\n  {'='*55}")
    print(f"  Загрузка: {len(symbols)} символов × {', '.join(active)}")
    print(f"  Период: {start.date()} → {now.date()}")
    print(f"  Bucket: {bucket_id or 'нет'}")
    print(f"  Tail: {'да' if tail else 'нет'}")
    print(f"  {'='*55}")

    t0 = time.time()
    for sym in symbols:
        if types.get("spot"):
            print(f"\n  OHLCV spot: {sym}")
            df, warns = fetch_klines(sym, "1h", years, perp=False, tail=tail, end_time=now)
            print(f"    {len(df):,} баров")
            for w in warns:
                print(f"    ⚠ {w}")

        if types.get("perp"):
            print(f"\n  OHLCV perp: {sym}")
            df, warns = fetch_klines(sym, "1h", years, perp=True, tail=tail, end_time=now)
            print(f"    {len(df):,} баров")
            for w in warns:
                print(f"    ⚠ {w}")

        if types.get("funding"):
            print(f"\n  Funding: {sym}", end=" ", flush=True)
            df = fetch_funding(sym, years=years, tail=tail)
            print(f"{len(df):,} записей")

        if types.get("metrics"):
            print(f"\n  Metrics: {sym}", end=" ", flush=True)
            df, warns = fetch_metrics(sym, years=years, tail=tail)
            print(f"{len(df):,} записей")
            for w in warns:
                print(f"    ⚠ {w}")

    print(f"\n  ✅ Готово за {time.time()-t0:.1f}s")
