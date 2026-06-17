"""Аудит данных: проверка качества скачанных OHLCV."""

import sys
import io
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TF_MS = {"1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000, "1d": 86400000}


def audit_file(filepath: Path) -> dict:
    """Проверить один parquet файл. Возвращает dict с проблемами и метриками."""
    result = {"issues": [], "bars": 0, "start": None, "end": None, "daily_completeness": None, "extreme_returns": None}

    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        result["issues"].append(f"Не удалось прочитать: {e}")
        return result

    if df.empty:
        result["issues"].append("Пустой файл")
        return result

    result["bars"] = len(df)
    result["start"] = df["datetime"].iloc[0].strftime("%Y-%m-%d")
    result["end"] = df["datetime"].iloc[-1].strftime("%Y-%m-%d")

    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        result["issues"].append(f"Нет колонок: {missing}")
        return result

    if df.isnull().any().any():
        result["issues"].append(f"NaN: {dict(df.isnull().sum()[df.isnull().sum() > 0])}")

    dups = df.duplicated(subset=["timestamp"]).sum()
    if dups > 0:
        result["issues"].append(f"Дубликатов: {dups}")

    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            result["issues"].append(f"{col} <= 0: {(df[col] <= 0).sum()}")

    if (df["volume"] < 0).any():
        result["issues"].append(f"volume < 0: {(df['volume'] < 0).sum()}")
    if (df["low"] > df["high"]).any():
        result["issues"].append(f"low > high: {(df['low'] > df['high']).sum()}")
    if ((df["open"] > df["high"]) | (df["close"] > df["high"])).any():
        result["issues"].append(f"open/close > high")
    if ((df["open"] < df["low"]) | (df["close"] < df["low"])).any():
        result["issues"].append(f"open/close < low")

    ts = df["timestamp"].sort_values().values
    diffs = pd.Series(ts).diff().dropna()
    tf = df["timeframe"].iloc[0] if "timeframe" in df.columns else "1h"
    expected = TF_MS.get(tf, 3600000)
    gaps = (diffs > expected * 1.5).sum()
    if gaps > 0:
        result["issues"].append(f"Гэпов: {gaps}")

    today = pd.Timestamp.now(tz="UTC").date()
    df["day"] = df["datetime"].dt.date
    df_complete = df[df["day"] < today]
    bars_per_day = expected // (24 * 3600000) * 24 if tf == "1d" else 24 if tf in ("1h", "4h") else 24 * 60 // (expected // 60000)
    if tf == "1h":
        bars_per_day = 24
    elif tf == "4h":
        bars_per_day = 6
    elif tf == "1d":
        bars_per_day = 1

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
                    d = df.loc[idx, "datetime"].strftime("%Y-%m-%d %H:%M")
                    extreme.append({"date": d, "return": round(r * 100, 2)})
            result["extreme_returns"] = extreme

    return result


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

    pairs = {}
    for f in files:
        name = f.stem.replace("_1h", "").replace("_4h", "").replace("_1d", "")
        if "_USDT_USDT" in name:
            base = name.replace("_USDT_USDT", "")
            pairs.setdefault(base, {})["perp"] = f
        elif "_USDT" in name:
            base = name.replace("_USDT", "")
            pairs.setdefault(base, {})["spot"] = f

    incomplete = [b for b, s in sorted(pairs.items()) if "spot" not in s or "perp" not in s]
    full = {b: s for b, s in pairs.items() if "spot" in s and "perp" in s}

    if incomplete:
        print(f"\nНеполные пары: {incomplete}")
    print(f"Полных пар: {len(full)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Аудит OHLCV данных")
    parser.add_argument("data_dir", type=Path, nargs="?", default=Path("data/top5_2026"))
    args = parser.parse_args()
    audit_folder(args.data_dir)
