"""Загрузка дневных OHLCV данных с Binance (с 2022 года)."""

import ccxt
import pandas as pd
import time
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("data/binance")
DATA_DIR.mkdir(parents=True, exist_ok=True)

EXCHANGE = ccxt.binance({"enableRateLimit": True})
EXCHANGE.load_markets()

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
    "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT",
    "UNI/USDT", "ATOM/USDT", "FIL/USDT", "LTC/USDT", "NEAR/USDT",
    "AAVE/USDT", "FET/USDT", "INJ/USDT", "GALA/USDT", "IMX/USDT",
    "STX/USDT", "OP/USDT", "ARB/USDT",
]

SINCE = "2022-01-01"
TIMEFRAME = "1d"


def fetch_ohlcv_all(symbol: str, timeframe: str, since_str: str) -> pd.DataFrame:
    since = int(datetime.strptime(since_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    all_candles = []

    while True:
        try:
            candles = EXCHANGE.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        except Exception as e:
            print(f"    Ошибка: {e}, повтор...")
            time.sleep(5)
            continue

        if not candles:
            break

        all_candles.extend(candles)
        if len(candles) < 1000:
            break
        since = candles[-1][0] + 1
        time.sleep(0.1)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["symbol"] = symbol
    df["timeframe"] = timeframe
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    print(f"Загрузка OHLCV | Binance | {TIMEFRAME} | с {SINCE}")
    print(f"Символов: {len(SYMBOLS)}")
    print("=" * 60)

    total_rows = 0
    success = 0

    for symbol in SYMBOLS:
        df = fetch_ohlcv_all(symbol, TIMEFRAME, SINCE)
        if df.empty:
            print(f"  ❌ {symbol}: нет данных")
            continue

        clean = symbol.replace("/", "_")
        fp = DATA_DIR / f"{clean}_{TIMEFRAME}.parquet"
        df.to_parquet(fp, index=False)

        sz = fp.stat().st_size / 1024
        bars = len(df)
        total_rows += bars
        success += 1
        print(f"  ✅ {symbol:>12}: {bars:>5} баров | {df['datetime'].iloc[0].strftime('%Y-%m-%d')} — {df['datetime'].iloc[-1].strftime('%Y-%m-%d')} | {sz:.0f} KB")

    print(f"\n{'=' * 60}")
    print(f"Готово: {success}/{len(SYMBOLS)} символов, {total_rows} строк")
    total = sum(f.stat().st_size for f in DATA_DIR.glob("*.parquet"))
    print(f"Общий размер: {total / 1024:.0f} KB")


if __name__ == "__main__":
    main()
