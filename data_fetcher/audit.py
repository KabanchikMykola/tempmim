"""Аудит данных в HuggingFace Bucket: проверка качества."""

import sys
import io
import pandas as pd
from datetime import datetime, timezone
from huggingface_hub import HfFileSystem

from data_fetcher import config

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")

TF_MS = {"1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000, "1d": 86400000}
BUCKET_BASE = f"hf://buckets/{config.BUCKET_ID}/{config.BUCKET_PREFIX}/binance"


def _bucket_files(subdir: str) -> list[tuple[str, str]]:
    fs = HfFileSystem()
    pattern = f"{BUCKET_BASE}/{subdir}/**/*.parquet"
    results = []
    for f in fs.glob(pattern):
        uri = f"hf://{f}"
        name = f.replace(f"{BUCKET_BASE.replace('hf://', '')}/{subdir}/", "")
        results.append((name, uri))
    return results


def _fmt_date(ts: int) -> str:
    if not ts or ts <= 0:
        return "?"
    try:
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except (OSError, ValueError):
        return "?"


def audit_df(df: pd.DataFrame, name: str, ts_col=None) -> dict:
    """Проверить DataFrame. Возвращает dict с метриками."""
    result = {"name": name, "issues": [], "warnings": [], "bars": 0,
              "start": None, "end": None, "gap_days": None, "gaps": 0,
              "has_ohlcv": False, "verdict": "ok"}

    if df.empty:
        result["issues"].append("пустой файл")
        result["verdict"] = "fail"
        return result

    result["bars"] = len(df)

    if ts_col is None:
        ts_col = "timestamp" if "timestamp" in df.columns else ("ts" if "ts" in df.columns else None)

    if ts_col:
        ts = df[ts_col].values
        result["start"] = _fmt_date(ts[0])
        result["end"] = _fmt_date(ts[-1])

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        result["gap_days"] = round((now_ms - ts[-1]) / 86400000, 1)

        if result["gap_days"] > 3:
            result["warnings"].append(f"устарело на {result['gap_days']}д")
        if result["gap_days"] > 30:
            result["issues"].append(f"сильно устарело ({result['gap_days']}д) — непригодно")

        required = ["open", "high", "low", "close", "volume"]
        has_ohlcv = all(c in df.columns for c in required)
        result["has_ohlcv"] = has_ohlcv

        if has_ohlcv:
            for col in required:
                if (df[col] <= 0).any():
                    result["issues"].append(f"{col} <= 0: {(df[col] <= 0).sum()}")

            if (df["low"] > df["high"]).any():
                result["issues"].append(f"low > high: {(df['low'] > df['high']).sum()}")

            # Гэпы
            diffs = pd.Series(ts).diff().dropna()
            tf_col = "timeframe" if "timeframe" in df.columns else ("interval" if "interval" in df.columns else None)
            tf = df[tf_col].iloc[0] if tf_col else "1h"
            expected = TF_MS.get(tf, 3600000)
            result["gaps"] = int((diffs > expected * 1.5).sum())
            if result["gaps"] > 0:
                result["warnings"].append(f"гэпов: {result['gaps']}")

        dups = df.duplicated(subset=[ts_col]).sum()
        if dups > 0:
            result["issues"].append(f"дубликатов: {dups}")

        if result["bars"] < 1000:
            result["issues"].append(f"мало данных ({result['bars']} баров)")
    else:
        # Нет ts — просто проверяем непустоту
        result["start"] = "?"
        result["end"] = "?"

    if result["issues"]:
        result["verdict"] = "fail"
    elif result["warnings"]:
        result["verdict"] = "warn"

    return result


def run_audit(subdirs: list[str] = None) -> list[dict]:
    """Аудит всех файлов в bucket. Возвращает список результатов."""
    if subdirs is None:
        subdirs = ["ohlcv_spot", "ohlcv_perp", "funding", "metrics"]

    results = []
    for subdir in subdirs:
        files = _bucket_files(subdir)
        for name, uri in files:
            try:
                df = pd.read_parquet(uri)
            except Exception as e:
                results.append({"name": f"{subdir}/{name}", "issues": [f"ошибка чтения"],
                                "bars": 0, "start": None, "end": None, "gap_days": None,
                                "gaps": 0, "has_ohlcv": False, "verdict": "fail",
                                "warnings": []})
                continue
            r = audit_df(df, f"{subdir}/{name}")
            results.append(r)
    return results


def print_report(results: list[dict]):
    """Красивый вывод аудита."""
    by_subdir = {}
    for r in results:
        parts = r["name"].split("/", 1)
        subdir = parts[0] if len(parts) > 1 else "other"
        by_subdir.setdefault(subdir, []).append(r)

    for label in ["ohlcv_spot", "ohlcv_perp", "funding", "metrics"]:
        group = by_subdir.get(label, [])
        if not group:
            continue
        names = {
            "ohlcv_spot": "OHLCV Spot",
            "ohlcv_perp": "OHLCV Perp",
            "funding": "Funding Rate",
            "metrics": "Metrics",
        }
        print(f"\n  {names.get(label, label)}:")
        print(f"  {'─'*60}")

        for r in sorted(group, key=lambda x: x["name"]):
            short = r["name"].split("/", 1)[1] if "/" in r["name"] else r["name"]
            bars = f"{r['bars']:>7,}" if r['bars'] else "      0"
            dates = f"{r['start']} → {r['end']}" if r['start'] else "—"

            if r["verdict"] == "fail":
                icon = "❌"
            elif r["verdict"] == "warn":
                icon = "⚠"
            else:
                icon = "✅"

            age = f"  (отставание: {r['gap_days']}д)" if r["gap_days"] is not None else ""

            print(f"  {icon} {short:<40s} {bars} баров")
            print(f"     {dates}{age}")
            if r["gaps"] > 0:
                print(f"     гэпы: {r['gaps']}")
            for w in r.get("warnings", []):
                print(f"     ⚠ {w}")
            for issue in r.get("issues", []):
                print(f"     ✗ {issue}")


def main():
    print("Аудит данных в HuggingFace Bucket\n")
    results = run_audit()
    print_report(results)

    clean = sum(1 for r in results if r["verdict"] == "ok")
    warn = sum(1 for r in results if r["verdict"] == "warn")
    fail = sum(1 for r in results if r["verdict"] == "fail")
    print(f"\n  {'='*60}")
    print(f"  ✅ {clean} чистых  ⚠ {warn} с предупреждениями  ❌ {fail} непригодны")
    print(f"  Всего: {len(results)} файлов")


if __name__ == "__main__":
    main()
