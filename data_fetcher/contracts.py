"""Data Contracts — схема, типы, правила качества для каждого типа данных.

Каждая функция validate_* возвращает (df, warnings):
  df — очищенный DataFrame (с удалёнными проблемными строками)
  warnings — список строк-предупреждений
"""

from datetime import datetime, timezone

import pandas as pd


# ── OHLCV ──────────────────────────────────────────────────────────

def validate_ohlcv(df, interval="1h", silent=False):
    """Проверить и почистить OHLCV данные.

    Контракт:
      - timestamp: int64, ms, monotonic, NOT null
      - open/high/low/close: float64, > 0
      - volume: float64, >= 0
      - high >= low, high >= open, high >= close
      - low <= open, low <= close
      - no gaps > 1.5 × interval
      - no duplicates by ts
    """
    warnings = []
    if df.empty:
        return df, warnings

    # 1. Zero-volume — предупреждение, строки НЕ удаляются
    n_zero = (df["volume"] == 0).sum()
    if n_zero > 0:
        warnings.append(f"    {n_zero} zero-volume строк (оставлены)")

    # 2. Дубликаты по ts — предупреждение, строки НЕ удаляются
    n_dup = df.duplicated(subset=["ts"]).sum()
    if n_dup > 0:
        warnings.append(f"    {n_dup} дубликатов (оставлены)")

    # 3. Сортировка
    df = df.sort_values("ts").reset_index(drop=True)

    # 4. OHLC consistency
    bad = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    if bad.any():
        warnings.append(f"    OHLC inconsistency: {bad.sum()} строк")

    # 5. Цены <= 0
    for col in ("open", "high", "low", "close"):
        n = (df[col] <= 0).sum()
        if n:
            warnings.append(f"    {col} <= 0: {n} строк")

    # 6. Гэпы (> 1.5x интервал)
    if len(df) > 2:
        tf_ms = {"1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000,
                 "4h": 14400000, "1d": 86400000}
        expected = tf_ms.get(interval, 3600000)
        gaps = (df["ts"].diff() > expected * 1.5).sum()
        if gaps > 0:
            warnings.append(f"    Гэпов (>1.5x интервал): {gaps}")

    return df, warnings


# ── Funding Rate ───────────────────────────────────────────────────

def validate_funding(df):
    """Проверить funding rate данные.

    Контракт:
      - symbol: string, NOT null
      - calc_time: int64, ms
      - last_funding_rate: float64
      - ts: int64, ms, monotonic
    """
    warnings = []
    if df.empty:
        return df, warnings

    before = len(df)
    df = df.drop_duplicates(subset=["symbol", "ts"])
    dropped = before - len(df)
    if dropped > 0:
        warnings.append(f"    {dropped} дубликатов удалено")

    df = df.sort_values("ts").reset_index(drop=True)

    if "last_funding_rate" in df.columns:
        n = df["last_funding_rate"].isna().sum()
        if n:
            warnings.append(f"    {n} NaN в last_funding_rate")

    return df, warnings


# ── Metrics ────────────────────────────────────────────────────────

def validate_metrics(df):
    """Проверить derivatives metrics данные.

    Контракт:
      - symbol: string, NOT null
      - create_time: int64, ms
      - sum_open_interest: float64, >= 0
      - ts: int64, ms, monotonic
    """
    warnings = []
    if df.empty:
        return df, warnings

    before = len(df)
    df = df.drop_duplicates(subset=["symbol", "ts"])
    dropped = before - len(df)
    if dropped > 0:
        warnings.append(f"    {dropped} дубликатов удалено")

    df = df.sort_values("ts").reset_index(drop=True)

    if "sum_open_interest" in df.columns:
        n = (df["sum_open_interest"] < 0).sum()
        if n:
            warnings.append(f"    {n} negative sum_open_interest")

    return df, warnings
