"""Аудит данных: проверка качества скачанных OHLCV.

Поддерживает форматы:
  - ccxt:   {SYMBOL}_{TF}.parquet  (колонки: timestamp, datetime, timeframe)
  - vision: {SYMBOL}_{TF}_{spot|perp}.parquet  (колонки: ts, interval)
"""

import sys
import io
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TF_MS = {"1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000, "1d": 86400000}


def audit_file(filepath: Path) -> dict:
    """Проверить один parquet файл. Возвращает dict с проблемами и метриками."""
    result = {"issues": [], "bars": 0, "start": None, "end": None,
              "daily_completeness": None, "extreme_returns": None}

    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        result["issues"].append(f"Не удалось прочитать: {e}")
        return result

    if df.empty:
        result["issues"].append("Пустой файл")
        return result

    result["bars"] = len(df)

    # Определить колонку timestamp (vision → ts, ccxt → timestamp)
    ts_col = "timestamp" if "timestamp" in df.columns else ("ts" if "ts" in df.columns else None)
    if ts_col is None:
        result["issues"].append("Нет колонки timestamp или ts")
        return result

    ts = df[ts_col].values
    result["start"] = _fmt_date(ts[0])
    result["end"] = _fmt_date(ts[-1])

    # Определить колонку timeframe
    tf_col = "timeframe" if "timeframe" in df.columns else ("interval" if "interval" in df.columns else None)
    timeframe = df[tf_col].iloc[0] if tf_col else "1h"

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        result["issues"].append(f"Нет колонок: {missing}")
        return result

    if df.isnull().any().any():
        result["issues"].append(f"NaN: {dict(df.isnull().sum()[df.isnull().sum() > 0])}")

    dups = df.duplicated(subset=[ts_col]).sum()
    if dups > 0:
        result["issues"].append(f"Дубликатов: {dups}")

    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            result["issues"].append(f"{col} <= 0: {(df[col] <= 0).sum()}")

    if (df["volume"] < 0).any():
        result["issues"].append(f"volume < 0: {(df['volume'] < 0).sum()}")
    if (df["low"] > df["high"]).any():
        result["issues"].append(f"low > high: {(df['low'] > df['high']).sum()}")
    bad_oh = ((df["open"] > df["high"]) | (df["close"] > df["high"])).any()
    bad_ol = ((df["open"] < df["low"]) | (df["close"] < df["low"])).any()
    if bad_oh:
        result["issues"].append(f"open/close > high")
    if bad_ol:
        result["issues"].append(f"open/close < low")

    # Gap detection
    diffs = pd.Series(ts).diff().dropna()
    expected = TF_MS.get(timeframe, 3600000)
    gaps = (diffs > expected * 1.5).sum()
    if gaps > 0:
        result["issues"].append(f"Гэпов: {gaps}")

    # Daily completeness
    if ts_col == "timestamp" or ts_col == "ts":
        if ts_col in df.columns:
            dt_col = pd.to_datetime(df[ts_col], unit="ms", utc=True)
        else:
            dt_col = None
        if dt_col is not None:
            today = pd.Timestamp.now(tz="UTC").date()
            df_tmp = df.copy()
            df_tmp["day"] = dt_col.dt.date
            df_complete = df_tmp[df_tmp["day"] < today]
            bars_per_day = 24 if timeframe == "1h" else 6 if timeframe == "4h" else 1 if timeframe == "1d" else 24
            daily = df_complete.groupby("day").size()
            incomplete = daily[daily < bars_per_day * 0.8]
            if len(incomplete) > 0:
                result["daily_completeness"] = {str(k): int(v) for k, v in incomplete.items()}
                result["issues"].append(f"Неполных дней: {len(incomplete)}")

    if "close" in df.columns and len(df) > 1:
        rets = df["close"].pct_change().dropna()
        top_abs = rets.abs().nlargest(5)
        if len(top_abs) > 0 and top_abs.iloc[0] > 0.3:
            extreme = []
            for idx in top_abs.index:
                r = rets[idx]
                if abs(r) > 0.3:
                    if dt_col is not None:
                        d = dt_col.iloc[idx].strftime("%Y-%m-%d %H:%M")
                    else:
                        d = str(idx)
                    extreme.append({"date": d, "return": round(r * 100, 2)})
            result["extreme_returns"] = extreme

    return result


def _fmt_date(ts: int) -> str:
    if not ts or ts <= 0:
        return "?"
    try:
        import datetime as dt
        return dt.datetime.fromtimestamp(ts / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d")
    except (OSError, ValueError):
        return "?"


def audit_folder(data_dir: Path) -> None:
    """Аудит всех parquet файлов в папке."""
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        print(f"Нет parquet файлов в {data_dir}")
        return

    print(f"Аудит {data_dir}: {len(files)} файлов\n")

    clean = 0
    problems = []

    for f in files:
        r = audit_file(f)
        if r["issues"]:
            problems.append((f.name, r))
        else:
            clean += 1

    print(f"Чистых: {clean}/{len(files)}")
    print(f"С проблемами: {len(problems)}/{len(files)}")

    if problems:
        print(f"\n{'=' * 60}")
        for name, r in problems:
            print(f"\n  {name} ({r['bars']} баров, {r['start']} → {r['end']}):")
            for issue in r["issues"]:
                print(f"    - {issue}")
            if r["extreme_returns"]:
                print(f"    Экстремальные возвраты:")
                for er in r["extreme_returns"]:
                    print(f"      {er['date']}: {er['return']:+.1f}%")
    else:
        print("\nВсе файлы чистые!")

    # Pair completeness (vision naming: {SYMBOL}_{TF}_{spot|perp}.parquet)
    spots = {}
    perps = {}
    for f in files:
        name = f.stem  # e.g. BTCUSDT_1h_spot
        parts = name.rsplit("_", 1)  # ["BTCUSDT_1h", "spot"]
        if len(parts) != 2:
            continue
        base_tf, src = parts
        if src == "spot":
            spots[base_tf] = f
        elif src == "perp":
            perps[base_tf] = f

    common_bases = set(spots.keys()) & set(perps.keys())
    incomplete = sorted(set(spots.keys()) ^ set(perps.keys()))
    if incomplete:
        print(f"\nНеполные пары ({len(incomplete)}):")
        for b in incomplete:
            sides = []
            if b in spots:
                sides.append("spot")
            if b in perps:
                sides.append("perp")
            print(f"    {b}: только {', '.join(sides)}")
    print(f"Полных пар: {len(common_bases)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Аудит OHLCV данных")
    parser.add_argument("data_dir", type=Path, nargs="?", default=Path("data"))
    args = parser.parse_args()
    audit_folder(args.data_dir)
