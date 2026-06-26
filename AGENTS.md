# AGENTS.md

## Project

Cross-exchange crypto data fetcher and alpha research toolkit. Fetches OHLCV (spot + perpetual) from Binance, aggregates tick data, and runs strategy backtests.

## Package manager

Uses `uv`. Lockfile: `uv.lock`. Python >= 3.12.

```sh
uv sync                    # основные зависимости (без ML)
uv sync --extra ml         # + ML (lightgbm, optuna, neuralforecast → torch)
```

## CLI

```sh
python -m data_fetcher <subcommand>
```

Subcommands: `ohlcv ccxt`, `ohlcv vision`, `agg-trades`, `book-depth`, `funding`, `symbols`, `stream`, `exchange-audit`.

Examples:
```sh
# Download common spot+perp OHLCV via ccxt
python -m data_fetcher ohlcv ccxt --common --since 2026-01-01

# Download from Binance S3 archives
python -m data_fetcher ohlcv vision --symbol BTCUSDT --interval 1h

# All common symbols (from symbols/spot_perpetual_common_usdt.json)
python -m data_fetcher ohlcv vision --all --years 2

# + tail via REST API (last 48h, no gaps for current month)
python -m data_fetcher ohlcv vision --symbol BTCUSDT --tail

# Tail only (skip S3, merge with existing parquet)
python -m data_fetcher ohlcv vision --symbol BTCUSDT --tail-only

# Audit 17 exchanges for data availability
python -m data_fetcher exchange-audit --exchanges binance bybit okx

# Upload to HuggingFace Bucket
python -m data_fetcher ohlcv ccxt --common --upload --bucket Kabanchik/mimo
```

## Data

- **Корневая папка:** `fin_data/` (вместо `data/`)
- **Bucket prefix:** `fin_data/` на HF

### Структура

```
fin_data/
├── binance/
│   ├── ohlcv_spot/
│   │   ├── BTCUSDT_1h_2025.parquet     ← годовые партиции
│   │   ├── BTCUSDT_1h_2026.parquet     ← только этот перезаписывается при tail
│   │   └── ...
│   ├── ohlcv_perp/
│   │   └── ...
│   ├── funding/
│   │   └── BTCUSDT_funding.parquet
│   └── metrics/
│       └── BTCUSDT_metrics.parquet
├── bybit/    ← future
└── yfinance/ ← future
```

- DuckDB cache: `fin_data/cache/binance_vision.db`
- `symbols/` has pre-fetched Binance symbol lists (JSON).
- `data/top5_2026/` is expected by alpha_research scripts but **not in repo** (gitignored `data/`).

### Data columns (parquet)

All OHLCV files share these 15 columns:
```
symbol, source, interval, open_time, open, high, low, close, volume,
close_time, quote_volume, count, taker_buy_base, taker_buy_quote, ts
```
**Known issue:** spot `open_time` is in microseconds (16 digits), perp is in milliseconds (13 digits). Both have `ts` column but it's not always consistent with `open_time`.

## data_fetcher/

Package structure:
```
data_fetcher/
  ccxt_api/          — OHLCV via ccxt API (spot + perp), exchange audit
  binance_vision/    — Historical data from data.binance.vision (S3)
  binance_api/       — Binance REST API (exchangeInfo, OHLCV tail)
  websocket/         — Realtime WebSocket pipeline (ccxt.pro)
  config.py          — MIN_VOLUME_USD, SINCE, TIMEFRAME, WORKERS
  audit.py           — Data quality audit for parquet files
  benchmark.py       — Binance API parallel benchmark
```

### Key functions
- `ccxt_api/fetcher.py` — `discover_common_symbols()`, `run_download()`, `upload_to_bucket()`
- `binance_vision/fetch_klines.py` — `fetch_symbol()` handles both spot + perp klines
- `binance_vision/fetch_funding.py` — `fetch_funding()` for funding rates only
- `binance_api/klines.py` — `fetch_tail()` Binance REST API (last 48h, no key needed)
- `ccxt_api/exchange_audit.py` — `run_audit()` checks data availability across 17 exchanges

## alpha_research/

Standalone scripts (not importable package). Each runs independently.

### Active scripts (11)
| Script | What it does |
|---|---|
| `strategies_basis.py` | Walk-forward basis z-score strategy search |
| `strategies_xgboost.py` | XGBoost classifier for 12h direction prediction |
| `strategies_xgb_variations.py` | Batch XGBoost across horizons/features |
| `strategies_all.py` | Combined: basis + reversion + momentum + vol-squeeze |
| `strategies_cost_check.py` | Re-evaluate strategies with realistic costs |
| `strategies_lgbm_optuna.py` | LightGBM + Optuna hyperparameter search |
| `basis_walkforward.py` | Basis walk-forward with parameter sweep |
| `basis_realistic.py` | Basis backtest with realistic cost model |
| `signal_search_v4.py` | Signal search with strict IC stability |
| `volatility_regime_v3.py` | Volatility regime with dynamic exits |
| `strategies_nf_nhits.py` | NHITS (NeuralForecast) 12h direction classifier with spot+perp+metrics |
| `strategies.json` | Auto-generated: saved winning strategies |

### Archive (old versions)
`archive/` — signal_search (v1-v3), volatility_regime (v1-v2). Superseded by latest versions.

### Data format for alpha_research
All scripts expect `data/top5_2026/` with:
- `{BASE}_USDT_1h.parquet` (spot)
- `{BASE}_USDT_USDT_1h.parquet` (perp)

### Running alpha_research
```sh
python alpha_research/strategies_lgbm_optuna.py
python alpha_research/strategies_basis.py
```

## Critical lessons (from old project)

1. **Lookahead bias** — all TA indicators (RSI, BB, ATR, etc.) MUST use `.shift(1)`. Using close[t] to predict at time t is a leak.
2. **Quality > quantity** — 800+ features failed; top-50 focused features worked. Simple models (num_leaves=5) often beat complex ones.
3. **Target horizon matters** — long-term targets (window=72) are more stable than short (window=24).
4. **Composite metric** — DA alone is insufficient. Use DA + Corr (or DA + F1) for model selection.
5. **Cross-asset features** — BTC vs ETH spreads are strong predictors. Not yet implemented in current codebase.
6. **Funding rate** — short lags are useless. Only long lags (24h+) have signal.

## Data quality

### Timestamp normalization
- `open_time`/`close_time` ALWAYS in milliseconds. Per-row check: `if x > 1e14 → // 1000`.
- `ts` всегда равен `open_time` (после нормализации оба в ms).
- Parquet files имеют колонку `timestamp = ts` для совместимости с alpha_research.

### Validation (автоматически при загрузке)
- Zero-volume строки (биржевое ТО) удаляются.
- Дубликаты по `ts` удаляются.
- OHLC consistency: high >= low >= open/close.
- Гэпы > 1.5x интервал детектятся и выводятся.

### Data Contracts
- `data_fetcher/contracts.py` — валидация по контракту для OHLCV, funding, metrics
- Каждый тип данных имеет: схему (типы, nullability), правила качества (high >= low, volume >= 0), проверку монотонности timestamp
- Вызывается автоматически в каждом fetcher-е после загрузки
- Контракт гарантирует одинаковую схему независимо от источника (Binance, Bybit, yfinance)

### DuckDB caveats
- INSERT в `klines` использует **явные имена колонок**, не `SELECT *` — порядок колонок в DataFrame не совпадает с таблицей.
- `_db_insert_batch` фильтрует `df[df["ts"] > max_ts]` — дедупликация при повторном запуске.

### Tail via REST API
- `binance_api/klines.py:fetch_tail()` — 48 последних часов через Binance REST API.
- Без API-ключа, бесплатно, ~1200 req/min.
- `--tail`: S3 + API. `--tail-only`: только API (мерж с существующим parquet).

## No tests / No CI

No test suite, linting, or CI config exists. Run `uv run python -m data_fetcher ...` to verify functionality.

## Style

- Russian comments, print statements, and variable names throughout.
- Stdout rewrapped to UTF-8 in CLI scripts (`sys.stdout = io.TextIOWrapper(...)`).
- Data config in `data_fetcher/config.py` (MIN_VOLUME_USD, SINCE, TIMEFRAME, WORKERS).
- Always ask before `git push`.

## External storage

- HuggingFace Bucket: `hf://buckets/Kabanchik/mimo/fin_data/`
- Upload: `upload_to_bucket(Path("data"), "Kabanchik/mimo")`

### Установленные инструменты
- `hf` CLI v1.21.0 (`C:\Trad_proj\mimo_code\.venv\Scripts\hf.exe`)
- `hf skills add --global` — skill установлен в `~/.agents/skills/hf-cli/`
  (виден после перезапуска OpenCode)

### Что уже поправлено

**Коммит 3123fa4:**
- `pyproject.toml` — форматирование зависимостей
- `alpha_research/strategies_lgbm_optuna.py` — пути данных совместимы с fetcher-ом (data/ вместо data/top5_2026/, имена *_1h_spot.parquet / *_1h_perp.parquet)
- `data_fetcher/config.py` — комментарий про env для BUCKET_ID
- `.gitignore` — добавлен `!old_trash/`

**Текущая сессия:**
- `data_fetcher/binance_vision/fetch_klines.py` — добавлены `_sync_from_bucket`, `_sync_to_bucket`, проверка HF Bucket первой очередью, авто-sync после загрузки
- `data_fetcher/binance_vision/fetch_funding.py` — то же самое (HF Bucket → Parquet → DuckDB → S3 → sync)
- Установлен `hf` CLI v1.21.0, login через `hf auth login`
- `hf skills add --global` — skill для OpenCode

**Коммит e5bb611:**
- `data_fetcher/binance_vision/fetch_klines.py` — пути `fin_data/binance/ohlcv_{spot|perp}/`, годовой суффикс, `validate_ohlcv` из `contracts.py`
- `data_fetcher/binance_vision/fetch_funding.py` — пути `fin_data/binance/funding/`
- `data_fetcher/binance_vision/fetch_metrics.py` — пути `fin_data/binance/metrics/`
- `data_fetcher/config.py` — `DATA_DIR = Path("fin_data")`, helpers `ohlcv_path()`, `funding_path()`, `metrics_path()`
- `data_fetcher/contracts.py` — **новый**: Data Contracts для OHLCV/funding/metrics
- `AGENTS.md` — обновлена структура данных

### Запуск на других ПК
После первого клонирования репо на новом ПК:
1. `uv sync` — установить зависимости
2. `hf auth login` — войти в HuggingFace
3. `hf skills add --global` — установить skill (опционально)
4. Запускать fetcher-ы как обычно — они сами подтянут данные из bucket
