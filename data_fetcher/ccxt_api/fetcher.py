"""Получение OHLCV данных с Binance через ccxt API — спот и перпы, любой таймфрейм."""

import ccxt
import pandas as pd
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

MarketType = Literal["spot", "swap"]


def get_exchange(market_type: MarketType = "spot") -> ccxt.binance:
    config = {"enableRateLimit": True}
    if market_type == "swap":
        config["options"] = {"defaultType": "future"}
    return ccxt.binance(config)


def discover_common_symbols(min_volume: float = 1_000_000) -> dict[str, dict]:
    """Найти пары, которые есть И на споте, И на перпах Binance."""
    print("Получение спотовых пар...")
    spot_ex = get_exchange("spot")
    spot_ex.load_markets()
    spot_data = _discover(spot_ex)

    print("Получение перп-пар...")
    perp_ex = get_exchange("swap")
    perp_ex.load_markets()
    perp_data = _discover(perp_ex)

    spot_by_base = {s["base"]: s for s in spot_data}
    perp_by_base = {s["base"]: s for s in perp_data}

    common_bases = set(spot_by_base.keys()) & set(perp_by_base.keys())
    print(f"Спот: {len(spot_by_base)}, Перпы: {len(perp_by_base)}, Пересечение: {len(common_bases)}")

    result = {}
    for base in common_bases:
        total_vol = spot_by_base[base]["volume_usd"] + perp_by_base[base]["volume_usd"]
        if total_vol < min_volume:
            continue
        result[base] = {
            "spot_symbol": spot_by_base[base]["symbol"],
            "perp_symbol": perp_by_base[base]["symbol"],
            "total_volume": total_vol,
        }

    result = dict(sorted(result.items(), key=lambda x: x[1]["total_volume"], reverse=True))
    print(f"После фильтрации ${min_volume:,.0f}: {len(result)} пар")
    return result


def _discover(exchange: ccxt.binance, quote: str = "USDT") -> list[dict]:
    """Получить все активные пары. Дедупликация по base — лучший по объёму."""
    tickers = exchange.fetch_tickers()
    by_base = {}
    for symbol, ticker in tickers.items():
        m = exchange.markets.get(symbol)
        if not m or m["quote"] != quote or not m.get("active", False):
            continue
        vol = ticker.get("quoteVolume") or 0
        base = m["base"]
        if base not in by_base or vol > by_base[base]["volume_usd"]:
            by_base[base] = {"symbol": symbol, "base": base, "volume_usd": vol}
    return list(by_base.values())


def _fetch_ohlcv(exchange: ccxt.binance, symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    """Скачать все свечи с since_ms. С retry."""
    all_candles = []
    cursor = since_ms
    for _ in range(100):
        for attempt in range(5):
            try:
                candles = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
                break
            except (ccxt.RateLimitExceeded, ccxt.NetworkError):
                time.sleep(2 ** (attempt + 1))
                candles = []
            except ccxt.BadSymbol:
                return pd.DataFrame()
            except Exception:
                time.sleep(2)
                candles = []
        else:
            break

        if not candles:
            break
        all_candles.extend(candles)
        if len(candles) < 1000:
            break
        cursor = candles[-1][0] + 1
        time.sleep(0.1)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["symbol"] = symbol
    df["timeframe"] = timeframe
    return df


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Чистка: дедупликация, NaN, нулевые цены, high/low проверки."""
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df[df["low"] <= df["high"]]
    return df


def _load_existing(filepath: Path) -> pd.DataFrame | None:
    if not filepath.exists():
        return None
    try:
        return pd.read_parquet(filepath)
    except Exception:
        return None


def _safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def run_download(
    symbols: list[str],
    timeframe: str,
    since: str,
    data_dir: Path,
    workers: int = 8,
) -> None:
    """Параллельная загрузка. Каждый воркер — свой exchange."""
    data_dir.mkdir(parents=True, exist_ok=True)
    since_ms = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    print(f"\nЗагрузка OHLCV | {timeframe} | с {since}")
    print(f"Символов: {len(symbols)} | Воркеров: {workers}")
    print("=" * 60)

    def worker(symbol: str) -> tuple[str, int, str]:
        is_perp = ":" in symbol
        ex = get_exchange("swap" if is_perp else "spot")

        fname = f"{_safe_name(symbol)}_{timeframe}.parquet"
        filepath = data_dir / fname

        existing = _load_existing(filepath)
        s = since_ms
        if existing is not None and len(existing) > 0:
            s = int(existing["timestamp"].max()) + 1

        df = _fetch_ohlcv(ex, symbol, timeframe, s)
        if df.empty:
            return symbol, 0, "уже актуно" if existing is not None else "нет данных"

        df = _validate(df)
        if df.empty:
            return symbol, 0, "невалидно"

        if existing is not None and len(existing) > 0:
            df = pd.concat([existing, df]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        df.to_parquet(filepath, index=False)
        bars = len(df)
        dr = f"{df['datetime'].iloc[0].strftime('%Y-%m-%d')} — {df['datetime'].iloc[-1].strftime('%Y-%m-%d')}"
        return symbol, bars, dr

    total_rows = 0
    success = 0
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, s): s for s in symbols}
        done = 0
        for f in as_completed(futures):
            done += 1
            try:
                symbol, bars, info = f.result()
                if bars == 0 and info in ("нет данных", "невалидно"):
                    print(f"  [{done:>4}/{len(symbols)}] {symbol:>20}: {info}")
                    failed += 1
                elif bars == 0:
                    print(f"  [{done:>4}/{len(symbols)}] {symbol:>20}: {info}")
                    success += 1
                else:
                    fname = f"{_safe_name(symbol)}_{timeframe}.parquet"
                    sz = (data_dir / fname).stat().st_size / 1024
                    total_rows += bars
                    success += 1
                    print(f"  [{done:>4}/{len(symbols)}] {symbol:>20}: {bars:>5} баров | {info} | {sz:.0f}KB")
            except Exception as e:
                print(f"  [{done:>4}/{len(symbols)}] {futures[f]:>20}: ОШИБКА {e}")
                failed += 1

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Готово: {success}/{len(symbols)} | ошибок: {failed} | строк: {total_rows}")
    print(f"Время: {elapsed:.1f}s | {success/elapsed:.1f} символов/сек")
    total = sum(f.stat().st_size for f in data_dir.glob("*.parquet"))
    print(f"Размер: {total / 1024 / 1024:.1f} MB")


def upload_to_huggingface(data_dir: Path, repo_id: str, token: str | None = None) -> None:
    from huggingface_hub import HfApi, upload_folder
    api = HfApi(token=token)
    if not api.repo_exists(repo_id=repo_id, repo_type="dataset"):
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(data_dir), path_in_repo=data_dir.name, token=token)
    print(f"Загружено: https://huggingface.co/datasets/{repo_id}/tree/main/{data_dir.name}")
