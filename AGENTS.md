f xnj# AGENTS.md

## Project

Cross-exchange crypto data fetcher and alpha research toolkit. Fetches OHLCV (spot + perpetual) from Binance, aggregates tick data, and runs strategy backtests.

## Package manager

Uses `uv`. Lockfile: `uv.lock`. Python >= 3.12.

```sh
uv sync
```

## CLI

```sh
python -m data_fetcher <subcommand>
```

Subcommands: `ohlcv ccxt`, `ohlcv vision`, `agg-trades`, `book-depth`, `funding`, `symbols`, `stream`.

Example: download common spot+perp OHLCV:
```sh
python -m data_fetcher ohlcv ccxt --common --since 2026-01-01
```

## Data

- Parquet files in `data/`. Naming: `{SYMBOL}_{TF}_{type}.parquet` (e.g. `BTCUSDT_1h_spot.parquet`).
- `data/top5_2026/` is expected by alpha_research scripts but **not in repo** (gitignored `data/`).
- `symbols/` has pre-fetched Binance symbol lists (JSON).

## alpha_research/

Standalone scripts (not importable package). Each runs independently:
```sh
python alpha_research/strategies_basis.py
python alpha_research/signal_search_v4.py
```

Expect data in `data/top5_2026/` with paired spot+perp files per base asset.

## No tests / No CI

No test suite, linting, or CI config exists. Run `uv run python -m data_fetcher ...` to verify functionality.

## Style

- Russian comments, print statements, and variable names throughout.
- Stdout rewrapped to UTF-8 in CLI scripts (`sys.stdout = io.TextIOWrapper(...)`).
- Data config in `data_fetcher/config.py` (MIN_VOLUME_USD, SINCE, TIMEFRAME, WORKERS).
