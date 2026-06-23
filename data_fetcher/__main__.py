"""CLI для загрузки данных.

Подкоманды:
  ohlcv ccxt      — OHLCV через ccxt API (спот + перпы)
  ohlcv vision    — OHLCV из S3 архивов data.binance.vision
  agg-trades       — Исторические aggTrades из S3
  book-depth       — Исторические bookDepth из S3
  funding          — Funding rate + перп klines из S3
  symbols          — Символы с Binance (exchangeInfo)
  stream           — Реалтайм WebSocket pipeline
"""

import sys
import io
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def _parse_date_args(args):
    """Преобразовать --days или --start/--end в (start, end)."""
    if args.days:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)
    elif args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        raise ValueError("Укажите --days или --start/--end")
    return start, end


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Загрузка данных с Binance")
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # --- ohlcv ---
    p_ohlcv = sub.add_parser("ohlcv", help="Загрузка OHLCV свечей")
    ohlcv_sub = p_ohlcv.add_subparsers(dest="source")
    ohlcv_sub.required = True

    p_ccxt = ohlcv_sub.add_parser("ccxt", help="Через ccxt API (спот + перпы)")
    p_ccxt.add_argument("--common", action="store_true", help="Пары, общие для спота и перпов")
    p_ccxt.add_argument("--list-only", action="store_true", help="Только список символов")
    p_ccxt.add_argument("--timeframe", default="1h", help="Таймфрейм (default: 1h)")
    p_ccxt.add_argument("--since", default="2022-03-01", help="Дата начала (default: 2022-03-01)")
    p_ccxt.add_argument("--min-volume", type=float, default=1_000_000, help="Мин. объём (default: 1M)")
    p_ccxt.add_argument("--data-dir", type=Path, default=Path("data/common_1h"), help="Папка данных")
    p_ccxt.add_argument("--workers", type=int, default=8, help="Воркеров (default: 8)")
    p_ccxt.add_argument("--upload", action="store_true", help="Загрузить на HuggingFace")
    p_ccxt.add_argument("--repo", type=str, help="HuggingFace repo ID")

    p_vision = ohlcv_sub.add_parser("vision", help="Из S3 архивов (data.binance.vision)")
    p_vision.add_argument("--symbol", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Тикеры")
    p_vision.add_argument("--interval", default="1h", help="Таймфрейм")
    p_vision.add_argument("--years", type=int, default=3, help="Сколько лет")
    p_vision.add_argument("--perp", action="store_true", help="Перпетуалы вместо спота")

    # --- agg-trades ---
    p_agg = sub.add_parser("agg-trades", help="Исторические aggTrades из S3")
    p_agg.add_argument("--symbol", default="SOLUSDT", help="Тикер")
    p_agg.add_argument("--days", type=int, help="Последние N дней")
    p_agg.add_argument("--start", help="Дата начала (YYYY-MM-DD)")
    p_agg.add_argument("--end", help="Дата конца (YYYY-MM-DD)")

    # --- book-depth ---
    p_book = sub.add_parser("book-depth", help="Исторические bookDepth из S3")
    p_book.add_argument("--symbol", default="SOLUSDT", help="Тикер")
    p_book.add_argument("--days", type=int, help="Последние N дней")
    p_book.add_argument("--start", help="Дата начала (YYYY-MM-DD)")
    p_book.add_argument("--end", help="Дата конца (YYYY-MM-DD)")

    # --- funding ---
    p_fund = sub.add_parser("funding", help="Funding rate из S3")
    p_fund.add_argument("--symbol", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Тикеры")
    p_fund.add_argument("--years", type=int, default=3, help="Сколько лет")
    p_fund.add_argument("--klines", action="store_true", help="Также загрузить перп klines (15m)")

    # --- symbols ---
    p_sym = sub.add_parser("symbols", help="Символы с Binance (exchangeInfo)")
    p_sym.add_argument("--output-dir", help="Папка для сохранения JSON")
    p_sym.add_argument("--list", action="store_true", help="Вывести список общих пар")

    # --- stream ---
    p_stream = sub.add_parser("stream", help="Реалтайм WebSocket pipeline")

    args = parser.parse_args()

    # --- dispatch ---
    if args.command == "ohlcv":
        if args.source == "ccxt":
            _run_ccxt(args)
        elif args.source == "vision":
            _run_vision(args)
    elif args.command == "agg-trades":
        _run_agg_trades(args)
    elif args.command == "book-depth":
        _run_book_depth(args)
    elif args.command == "funding":
        _run_funding(args)
    elif args.command == "symbols":
        _run_symbols(args)
    elif args.command == "stream":
        _run_stream(args)


def _run_ccxt(args):
    from data_fetcher.ccxt_api.fetcher import discover_common_symbols, run_download, upload_to_huggingface

    common = discover_common_symbols(min_volume=args.min_volume)

    if args.list_only:
        for base, info in common.items():
            print(f"  {base:>8} | {info['spot_symbol']:>16} | {info['perp_symbol']:>16} | ${info['total_volume']:>14,.0f}")
        return

    symbols = []
    for info in common.values():
        symbols.append(info["spot_symbol"])
        symbols.append(info["perp_symbol"])

    print(f"\nЗагрузка {len(symbols)} символов ({len(common)} пар x 2)")
    run_download(symbols, args.timeframe, args.since, args.data_dir, args.workers)

    if args.upload and args.repo:
        upload_to_huggingface(args.data_dir, args.repo)


def _run_vision(args):
    from data_fetcher.binance_vision.fetch_klines import fetch_symbol, summary

    import time as ttime
    t0 = ttime.time()
    for symbol in args.symbol:
        df = fetch_symbol(symbol, args.interval, args.years, args.perp, export_parquet=True)
        src = "perp" if args.perp else "spot"
        print(f"  {symbol} ({src}): {len(df):,} баров")

    summary()
    print(f"\n  Время: {ttime.time()-t0:.1f}s")


def _run_agg_trades(args):
    from data_fetcher.binance_vision.fetch_agg_trades import fetch_range as af

    start, end = _parse_date_args(args)
    print(f"Загрузка aggTrades: {args.symbol} {start.date()} - {end.date()}")
    import time as ttime
    t0 = ttime.time()
    df = af(args.symbol, start, end)
    print(f"  Строк: {len(df):,}")
    print(f"  Время: {ttime.time()-t0:.1f}s")


def _run_book_depth(args):
    from data_fetcher.binance_vision.fetch_book_depth import fetch_range as bf

    start, end = _parse_date_args(args)
    print(f"Загрузка bookDepth: {args.symbol} {start.date()} - {end.date()}")
    import time as ttime
    t0 = ttime.time()
    df = bf(args.symbol, start, end)
    print(f"  Строк: {len(df):,}")
    print(f"  Время: {ttime.time()-t0:.1f}s")


def _run_funding(args):
    from data_fetcher.binance_vision.fetch_funding import fetch_funding, fetch_perp_klines

    import time as ttime
    t0 = ttime.time()
    for symbol in args.symbol:
        print(f"  {symbol}: загрузка funding...", end=" ", flush=True)
        df = fetch_funding(symbol, args.years)
        print(f"{len(df):,} записей")
        if args.klines:
            print(f"  {symbol}: загрузка perp klines...", end=" ", flush=True)
            df_k = fetch_perp_klines(symbol, "15m", args.years)
            print(f"{len(df_k):,} баров")
    print(f"\n  Время: {ttime.time()-t0:.1f}s")


def _run_symbols(args):
    from data_fetcher.binance_api.fetch_symbols import fetch_all as sym_fetch

    result = sym_fetch(args.output_dir)
    if args.list:
        print(f"\nОбщие пары ({len(result['common'])}):")
        for s in result["common"]:
            print(f"  {s['symbol']}")
    print("Готово.")


def _run_stream(args):
    from data_fetcher.websocket.tick_pipeline import main as stream_main

    import asyncio
    asyncio.run(stream_main())


if __name__ == "__main__":
    main()
