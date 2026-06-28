"""HyperLiquid OHLCV из Chainticks/perp-data + native API tail."""

import pandas as pd
import numpy as np
import requests
import time
import io
from datetime import datetime, timezone, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CHAIN_TICKS = "https://huggingface.co/datasets/Chainticks/perp-data/resolve/main"
HYPER_API = "https://api.hyperliquid.xyz"
INTERVAL_MS = {"1h": 3600000, "15m": 900000, "1m": 60000}


def _fetch_markets_file(date_str: str) -> pd.DataFrame:
    """Скачать один markets parquet за дату."""
    url = f"{CHAIN_TICKS}/hyperliquid_chain/markets/date={date_str}/part-0000.parquet"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.read_parquet(io.BytesIO(resp.content))


def list_symbols() -> list[str]:
    """Получить список всех символов из Chainticks."""
    dates = _get_available_dates()
    if not dates:
        return []
    latest = dates[-1]
    try:
        df = _fetch_markets_file(latest)
        return sorted(df["symbol"].unique().tolist())
    except Exception:
        return []


def _get_available_dates() -> list[str]:
    """Получить список доступных дат из HF метадаты."""
    resp = requests.get("https://huggingface.co/api/datasets/Chainticks/perp-data", timeout=15)
    data = resp.json()
    dates = set()
    for sib in data.get("siblings", []):
        f = sib["rfilename"]
        if f.startswith("hyperliquid_chain/markets/date="):
            d = f.split("date=")[1].split("/")[0]
            dates.add(d)
    return sorted(dates)


def _resample_to_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Из 1-min snapshots в OHLCV через mid_price."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["exchange_time"], utc=True)

    if timeframe == "1m":
        freq = "1min"
    elif timeframe == "15m":
        freq = "15min"
    else:
        freq = "1h"

    df = df.set_index("time")
    ohlcv = df["mid_price"].resample(freq).agg(["first", "max", "min", "last"])
    ohlcv = ohlcv.dropna()
    ohlcv.columns = ["open", "high", "low", "close"]
    ohlcv = ohlcv.reset_index()
    ohlcv["timestamp"] = ohlcv["time"].values.astype("datetime64[ms]").astype(np.int64)
    return ohlcv


def _fetch_tail(symbol: str, timeframe: str, since_ms: int | None = None) -> pd.DataFrame:
    """Догрузить хвост через native API.

    API возвращает до 5000 свечей за запрос.
    Если since_ms указан, качаем с этой точки до текущего момента.
    """
    rows = []
    end_ms = int(time.time() * 1000)
    interval_ms = INTERVAL_MS.get(timeframe, 3600000)

    while True:
        start = since_ms if since_ms else (end_ms - 5000 * interval_ms)
        start = max(start, end_ms - 5000 * interval_ms)
        payload = {
            "type": "candleSnapshot",
            "req": {"coin": symbol, "interval": timeframe, "startTime": start, "endTime": end_ms},
        }
        resp = requests.post(f"{HYPER_API}/info", json=payload, timeout=15)
        if resp.status_code != 200:
            break
        candles = resp.json()
        if not candles:
            break
        rows = candles + rows
        if len(candles) < 5000:
            break
        end_ms = candles[0]["t"] - 1

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    cols = {"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns=cols)
    df = df[list(cols.values())]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["timestamp"] = df["timestamp"].astype(np.int64)
    return df


def fetch_symbol(
    symbol: str,
    timeframe: str = "1h",
    years: int = 3,
    upload_bucket: str | None = None,
    use_chain: bool = False,
) -> pd.DataFrame:
    """Скачать OHLCV для символа.

    Стратегия:
    1. Native API tail (гарантированно непрерывный, ~7 мес для 1h)
    2. Chainticks дополняет историю 2023-2024 (если нужный период попадает туда)

    Args:
        symbol: BTC, ETH, SOL...
        timeframe: 1h, 15m, 1m
        years: сколько лет данных нужно (считая от сегодня)
        upload_bucket: если указан — загрузить результат в HF bucket

    Returns:
        DataFrame с колонками: timestamp (ms), open, high, low, close, volume
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=365 * years)
    cutoff_ms = int(cutoff.timestamp() * 1000)

    # ── 1. Всегда качаем tail из native API ──
    tail = _fetch_tail(symbol, timeframe)
    if tail.empty:
        print(f"  {symbol}: tail API не вернул данных")
        return pd.DataFrame()

    result = tail

    # ── 2. Chainticks (опционально, медленно) ──
    tail_start = tail["timestamp"].min()
    if use_chain and cutoff_ms < tail_start:
        dates = _get_available_dates()
        chain_dates = [d for d in dates if d < datetime.fromtimestamp(tail_start / 1000, tz=timezone.utc).strftime("%Y-%m-%d")]
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        chain_dates = [d for d in chain_dates if d >= cutoff_str]

        if chain_dates:
            print(f"  {symbol}: {len(chain_dates)} days from Chainticks + tail...", end=" ", flush=True)

            chunks = []
            def load(d):
                try:
                    df = _fetch_markets_file(d)
                    df_sym = df[df["symbol"] == symbol]
                    if df_sym.empty:
                        return None
                    return _resample_to_ohlcv(df_sym, timeframe)
                except Exception:
                    return None

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(load, d): d for d in chain_dates}
                for f in as_completed(futures):
                    try:
                        chunk = f.result(timeout=180)
                        if chunk is not None and not chunk.empty:
                            chunks.append(chunk)
                    except Exception:
                        pass

            if chunks:
                chain_df = pd.concat(chunks, ignore_index=True)
                chain_df = chain_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                chain_to_tail = pd.concat([chain_df, tail], ignore_index=True)
                result = chain_to_tail.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if not use_chain or cutoff_ms >= tail_start:
        print(f"  {symbol}: tail only...", end=" ", flush=True)

    # ── 3. Обрезать до нужного количества лет ──
    result = result[result["timestamp"] >= cutoff_ms].reset_index(drop=True)

    # ── 4. Если нет volume (из mid_price), заполняем 0 ──
    if "volume" not in result.columns:
        result["volume"] = 0.0
    result["volume"] = result["volume"].fillna(0.0)

    result["datetime"] = pd.to_datetime(result["timestamp"], unit="ms", utc=True)
    print(f"{len(result)} баров | {result['datetime'].iloc[0].strftime('%Y-%m-%d')} — {result['datetime'].iloc[-1].strftime('%Y-%m-%d')}")

    # ── 5. Upload в HF bucket ──
    if upload_bucket and not result.empty:
        out = result.copy()
        out["symbol"] = symbol
        out["timeframe"] = timeframe
        year = now.year
        local_path = Path(f"/tmp/{symbol}_{timeframe}_{year}.parquet")
        out.to_parquet(local_path, index=False)
        from huggingface_hub import sync_bucket
        remote = f"hf://buckets/{upload_bucket}/fin_data/hyperliquid/ohlcv/{symbol}_{timeframe}_{year}.parquet"
        print(f"  Upload: {remote}")
        sync_bucket(str(local_path), remote)

    return result
