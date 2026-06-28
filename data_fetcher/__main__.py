"""CLI для загрузки данных Binance.

Подкоманды:
  ohlcv vision   — OHLCV из S3 архивов data.binance.vision
  funding        — Funding rate из S3
  metrics        — Derivatives metrics (OI, long/short ratios)
  hyperliquid    — HyperLiquid OHLCV из native API
  list           — Список файлов в HuggingFace Bucket
  menu           — Интерактивное меню
"""

import sys
import io
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Загрузка данных с Binance")
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # --- ohlcv vision ---
    p_vision = sub.add_parser("ohlcv", help="OHLCV свечи")
    ohlcv_sub = p_vision.add_subparsers(dest="source")
    ohlcv_sub.required = True
    p_vis = ohlcv_sub.add_parser("vision", help="Из S3 архивов (data.binance.vision)")
    p_vis.add_argument("--symbol", nargs="+", help="Тикеры")
    p_vis.add_argument("--all", action="store_true", help="Все общие спот+перп пары из symbols/")
    p_vis.add_argument("--interval", default="1h", help="Таймфрейм")
    p_vis.add_argument("--years", type=int, default=3, help="Сколько лет")
    p_vis.add_argument("--perp", action="store_true", help="Перпетуалы вместо спота")
    p_vis.add_argument("--tail", action="store_true", help="Докачать последние бары через REST API")
    p_vis.add_argument("--tail-only", action="store_true", help="Только последние бары (без S3 истории)")

    # --- funding ---
    p_fund = sub.add_parser("funding", help="Funding rate из S3")
    p_fund.add_argument("--symbol", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Тикеры")
    p_fund.add_argument("--years", type=int, default=3, help="Сколько лет")
    p_fund.add_argument("--tail", action="store_true", help="Докачать хвост через API")

    # --- metrics ---
    p_metrics = sub.add_parser("metrics", help="Derivatives metrics (OI, long/short ratios)")
    p_metrics.add_argument("--symbol", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Тикеры")
    p_metrics.add_argument("--years", type=int, default=3, help="Сколько лет")
    p_metrics.add_argument("--tail", action="store_true", help="Докачать хвост через API")
    p_metrics.add_argument("--force", action="store_true", help="Принудительная перезагрузка")

    # --- hyperliquid ---
    p_hl = sub.add_parser("hyperliquid", help="HyperLiquid OHLCV")
    p_hl.add_argument("--symbol", nargs="+", help="Символы (BTC, ETH, SOL...)")
    p_hl.add_argument("--all", action="store_true", help="Все доступные символы")
    p_hl.add_argument("--timeframe", default="1h", help="Таймфрейм (default: 1h)")
    p_hl.add_argument("--years", type=int, default=3, help="Сколько лет (default: 3)")
    p_hl.add_argument("--bucket", type=str, default="Kabanchik/mimo", help="HF Bucket для загрузки")
    p_hl.add_argument("--list", action="store_true", help="Список символов")
    p_hl.add_argument("--chain", action="store_true", help="Добавить Chainticks 2023-2024 историю (медленно)")

    # --- list ---
    sub.add_parser("list", help="Список файлов в HuggingFace Bucket")

    # --- menu ---
    p_menu = sub.add_parser("menu", help="Интерактивное меню загрузки")
    p_menu.add_argument("--run", action="store_true", help="Запустить с сохранённым конфигом (без интерактива)")

    args = parser.parse_args()

    # --- dispatch ---
    if args.command == "ohlcv" and args.source == "vision":
        _run_vision(args)
    elif args.command == "funding":
        _run_funding(args)
    elif args.command == "metrics":
        _run_metrics(args)
    elif args.command == "hyperliquid":
        _run_hyperliquid(args)
    elif args.command == "list":
        _run_list(args)
    elif args.command == "menu":
        _run_menu(args)


def _run_vision(args):
    from data_fetcher.binance_vision.fetch_klines import fetch_symbol
    import json

    if args.all:
        common_path = Path("symbols/spot_perpetual_common_usdt.json")
        if not common_path.exists():
            print("Сначала запусти: python -m data_fetcher symbols")
            return
        with open(common_path) as f:
            data = json.load(f)
        symbols = [p["symbol"] for p in data["pairs"]]
        print(f"Загрузка {len(symbols)} символов из {common_path}")
    elif args.symbol:
        symbols = args.symbol
    else:
        symbols = ["BTCUSDT", "ETHUSDT"]

    import time as ttime
    t0 = ttime.time()
    for i, symbol in enumerate(symbols, 1):
        src = "perp" if args.perp else "spot"
        tail_label = " +tail" if args.tail else ""
        print(f"[{i}/{len(symbols)}] {symbol} ({src}{tail_label})...", flush=True)
        df, warnings = fetch_symbol(
            symbol, args.interval, args.years, args.perp,
            tail=args.tail, tail_only=args.tail_only,
        )
        for w in warnings:
            print(f"  {w}")
        print(f"  {symbol} ({src}): {len(df):,} баров")
    print(f"\n  Время: {ttime.time()-t0:.1f}s")


def _run_funding(args):
    from data_fetcher.binance_vision.fetch_funding import fetch_funding

    import time as ttime
    t0 = ttime.time()
    for symbol in args.symbol:
        print(f"  {symbol}: загрузка funding...", end=" ", flush=True)
        df = fetch_funding(symbol, args.years, tail=args.tail)
        print(f"{len(df):,} записей")
    print(f"\n  Время: {ttime.time()-t0:.1f}s")


def _run_metrics(args):
    from data_fetcher.binance_vision.fetch_metrics import fetch_metrics

    import time as ttime
    t0 = ttime.time()
    for symbol in args.symbol:
        print(f"  {symbol}: загрузка metrics...", flush=True)
        df, warnings = fetch_metrics(symbol, args.years, force=args.force, tail=args.tail)
        for w in warnings:
            print(w)
        print(f"  {symbol}: {len(df):,} записей")
    print(f"\n  Время: {ttime.time()-t0:.1f}s")


def _run_hyperliquid(args):
    from data_fetcher.hyperliquid import fetch_symbol, list_symbols

    if args.list:
        symbols = list_symbols()
        print(f"Доступно символов: {len(symbols)}")
        for s in symbols:
            print(f"  {s}")
        return

    if args.all:
        symbols = list_symbols()
        if not symbols:
            print("Не удалось получить список символов")
            return
        print(f"Загрузка {len(symbols)} символов\n")
    elif args.symbol:
        symbols = args.symbol
    else:
        symbols = ["BTC", "ETH"]

    import time as ttime
    t0 = ttime.time()
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}]", end=" ", flush=True)
        df = fetch_symbol(sym, args.timeframe, args.years, upload_bucket=args.bucket, use_chain=args.chain)
        if df.empty:
            print(f"  {sym}: нет данных")
    print(f"\n  Время: {ttime.time()-t0:.1f}s")


def _run_list(args):
    from data_fetcher.config import list_bucket, DATA_DIR

    for subdir in ["binance/ohlcv_spot/", "binance/ohlcv_perp/",
                    "binance/funding/", "binance/metrics/", "hyperliquid/"]:
        files = list_bucket(subdir)
        if not files:
            continue
        total_mb = sum(f["size_bytes"] for f in files) / 1024 / 1024
        print(f"\n  HF Bucket: {subdir.strip('/')}  ({len(files)} файлов, {total_mb:.0f} MB)")
        print(f"  {'─'*60}")
        for f in sorted(files, key=lambda x: x["path"]):
            size_kb = f["size_bytes"] / 1024
            print(f"    {f['path']:68s} {size_kb:>8.0f} KB")

    local_patterns = sorted(DATA_DIR.rglob("*.parquet"))
    if local_patterns:
        total_mb = sum(f.stat().st_size for f in local_patterns) / 1024 / 1024
        print(f"\n  Local: ({len(local_patterns)} файлов, {total_mb:.0f} MB)")
        print(f"  {'─'*60}")
        for f in local_patterns:
            rel = f.relative_to(DATA_DIR)
            size_kb = f.stat().st_size / 1024
            print(f"    {str(rel):68s} {size_kb:>8.0f} KB")
    if not local_patterns:
        print("\n  Локально данных нет. Запусти fetcher для загрузки.")
    print()


def _run_menu(args):
    """Интерактивное меню."""
    from data_fetcher.menu import main as menu_main
    menu_main()


if __name__ == "__main__":
    main()
