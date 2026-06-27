# data_fetcher — документация

## CLI

```sh
python -m data_fetcher <subcommand>
```

Subcommands: `ohlcv ccxt`, `ohlcv vision`, `agg-trades`, `book-depth`, `funding`, `symbols`, `stream`, `exchange-audit`.

Основные флаги:
- `--common` — только общие для спота и перпа символы
- `--since YYYY-MM-DD` — начальная дата
- `--all` — все символы из `symbols/spot_perpetual_common_usdt.json`
- `--tail` — S3 + REST (последние 48ч)
- `--tail-only` — только REST, мёрж с существующим parquet
- `--upload --bucket Kabanchik/mimo` — загрузка в HF Bucket
- `--exchanges binance bybit okx` — для `exchange-audit`

Примеры:

```sh
python -m data_fetcher ohlcv ccxt --common --since 2026-01-01
python -m data_fetcher ohlcv vision --symbol BTCUSDT --interval 1h
python -m data_fetcher ohlcv vision --all --years 2
python -m data_fetcher ohlcv vision --symbol BTCUSDT --tail
python -m data_fetcher exchange-audit --exchanges binance bybit okx
```

## Структура пакета

```
data_fetcher/
  ccxt_api/          — OHLCV via ccxt API (spot + perp), exchange audit
  binance_vision/    — Historical data from data.binance.vision (S3)
  binance_api/       — Binance REST API (exchangeInfo, OHLCV tail)
  websocket/         — Realtime WebSocket pipeline (ccxt.pro)
  config.py          — MIN_VOLUME_USD, SINCE, TIMEFRAME, WORKERS
  audit.py           — Data quality audit for parquet files
  benchmark.py       — Binance API parallel benchmark
  contracts.py       — Data contracts (схемы, правила качества)
```

## Ключевые функции

| Файл | Функция | Что делает |
|---|---|---|
| `ccxt_api/fetcher.py` | `discover_common_symbols()` | Находит символы общие для spot+perp |
| `ccxt_api/fetcher.py` | `run_download()` | Скачивает OHLCV через ccxt |
| `ccxt_api/fetcher.py` | `upload_to_bucket()` | Загружает в HF Bucket |
| `binance_vision/fetch_klines.py` | `fetch_symbol()` | Скачивает spot+perp klines из S3 |
| `binance_vision/fetch_funding.py` | `fetch_funding()` | Скачивает funding rate из S3 |
| `binance_api/klines.py` | `fetch_tail()` | Binance REST API (последние 48ч) |
| `ccxt_api/exchange_audit.py` | `run_audit()` | Проверка данных на 17+ биржах |

## Данные

**Корневая папка:** `fin_data/` (не `data/`)

```
fin_data/
├── binance/
│   ├── ohlcv_spot/
│   │   ├── BTCUSDT_1h_2025.parquet     ← годовые партиции
│   │   ├── BTCUSDT_1h_2026.parquet     ← перезаписывается при tail
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
- `symbols/` — предзагруженные списки символов Binance (JSON)
- `data/top5_2026/` — ожидается alpha_research (gitignored)

### Колонки parquet (OHLCV)

```
symbol, source, interval, open_time, open, high, low, close, volume,
close_time, quote_volume, count, taker_buy_base, taker_buy_quote, ts
```

**Известный баг:** spot `open_time` в микросекундах (16 digits), perp в миллисекундах (13 digits). `ts` всегда равен `open_time` после нормализации (оба в ms). Parquet имеет колонку `timestamp = ts`.

## Data Contracts (`contracts.py`)

- Валидация по контракту для OHLCV, funding, metrics
- Схема (типы, nullability), правила качества (high >= low, volume >= 0), монотонность timestamp
- Вызывается автоматически в каждом fetcher-е после загрузки
- Гарантирует одинаковую схему независимо от источника (Binance, Bybit, yfinance)

## Валидация (автоматически при загрузке)

- Zero-volume строки удаляются
- Дубликаты по `ts` удаляются
- OHLC consistency: high >= low >= open/close
- Гэпы > 1.5x интервал детектятся и выводятся

### Timestamp normalization

- `open_time` / `close_time` всегда в миллисекундах
- Per-row check: `if x > 1e14 → // 1000`
- `ts` всегда равен `open_time` (после нормализации оба в ms)

## DuckDB caveats

- INSERT в `klines` использует **явные имена колонок**, не `SELECT *` — порядок колонок в DataFrame не совпадает с таблицей
- `_db_insert_batch` фильтрует `df[df["ts"] > max_ts]` — дедупликация при повторном запуске

## Tail via REST API

- `binance_api/klines.py:fetch_tail()` — 48 последних часов через Binance REST API
- Без API-ключа, бесплатно, ~1200 req/min
- `--tail`: S3 + API; `--tail-only`: только API (мерж с существующим parquet)

## External storage

- HuggingFace Bucket: `hf://buckets/Kabanchik/mimo/fin_data/`
- Upload: `upload_to_bucket(Path("data"), "Kabanchik/mimo")`

## WebSocket status ⚠️

`data_fetcher/websocket/` — **экспериментальный** функционал.

Сбор данных в реальном времени через WebSocket требует **выделенного сервера, работающего 24/7**, поэтому пока это не приоритет. Весь текущий фокус — на исторических данных через REST + Binance Vision S3.

Не тратьте время на websocket, пока не решены базовые задачи:
- загрузка и качество исторических данных
- генерация альфа-сигналов
- бэктестинг стратегий
