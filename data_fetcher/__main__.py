"""CLI для загрузки данных с Binance."""

import argparse
import sys
import io
from pathlib import Path

from data_fetcher.fetcher import run_download, discover_common_symbols, upload_to_huggingface

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Загрузка OHLCV с Binance (спот + перпы)")
    parser.add_argument("--common", action="store_true", help="Пары, общие для спота и перпов")
    parser.add_argument("--list-only", action="store_true", help="Только список символов")
    parser.add_argument("--timeframe", default="1h", help="Таймфрейм (default: 1h)")
    parser.add_argument("--since", default="2022-03-01", help="Дата начала (default: 2022-03-01)")
    parser.add_argument("--min-volume", type=float, default=1_000_000, help="Мин. объём (default: 1M)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/common_1h"), help="Папка данных")
    parser.add_argument("--workers", type=int, default=8, help="Воркеров (default: 8)")
    parser.add_argument("--upload", action="store_true", help="Загрузить на HuggingFace")
    parser.add_argument("--repo", type=str, help="HuggingFace repo ID")
    args = parser.parse_args()

    if not args.common:
        parser.print_help()
        return

    common = discover_common_symbols(min_volume=args.min_volume)

    if args.list_only:
        for base, info in common.items():
            print(f"  {base:>8} | {info['spot_symbol']:>16} | {info['perp_symbol']:>16} | ${info['total_volume']:>14,.0f}")
        return

    symbols = []
    for info in common.values():
        symbols.append(info["spot_symbol"])
        symbols.append(info["perp_symbol"])

    print(f"\nЗагрузка {len(symbols)} символов ({len(common)} пар × 2)")
    run_download(symbols, args.timeframe, args.since, args.data_dir, args.workers)

    if args.upload and args.repo:
        upload_to_huggingface(args.data_dir, args.repo)


if __name__ == "__main__":
    main()
