# Progress

## Setup для нового ПК

После первого клонирования:
1. `uv sync` — установить зависимости
2. `hf auth login` — войти в HuggingFace
3. `hf skills add --global` — установить skill для OpenCode (опционально)
4. Запускать fetcher-ы как обычно — они сами подтянут данные из bucket

## Коммит e5bb611

- `data_fetcher/binance_vision/fetch_klines.py` — пути `fin_data/binance/ohlcv_{spot|perp}/`, годовой суффикс, `validate_ohlcv` из `contracts.py`
- `data_fetcher/binance_vision/fetch_funding.py` — пути `fin_data/binance/funding/`
- `data_fetcher/binance_vision/fetch_metrics.py` — пути `fin_data/binance/metrics/`
- `data_fetcher/config.py` — `DATA_DIR = Path("fin_data")`, helpers `ohlcv_path()`, `funding_path()`, `metrics_path()`
- `data_fetcher/contracts.py` — Data Contracts для OHLCV/funding/metrics
- `AGENTS.md` — обновлена структура данных

## Коммит 3123fa4

- `pyproject.toml` — форматирование зависимостей
- `alpha_research/strategies_lgbm_optuna.py` — пути данных совместимы с fetcher-ом (`data/` вместо `data/top5_2026/`, имена `*_1h_spot.parquet` / `*_1h_perp.parquet`)
- `data_fetcher/config.py` — комментарий про env для `BUCKET_ID`
- `.gitignore` — добавлен `!old_trash/`

## Идеи: микроструктурные фичи из трейдов

Из старого проекта (`freebuff_test_google_cloud_shellproject/src/features/aggregate.py`):
- **buy_ratio** — объём buy / sell внутри 30s окна (перекос агрессии)
- **large_trade_count / volume** — количество/объём сделок > $100 (whale detection)
- **inter_trade_mean / std** — частота и неравномерность трейдов (признак активности)
- **price_impact** — (max-min)/vwap за 30s (проскальзывание)

Идеи валидны, но реализация сырая (SQL injection, groupby в pandas вместо DuckDB SQL).
Когда дойдут руки до фичей на основе aggTrades — стоит сделать нормально: DuckDB `date_bin`, `ARG_MIN/MAX`, оконные функции.

## Идеи: target-ы и фильтры (из experiment_5m.py, experiment.py, experiment_funding.py)

### Triple Barrier (ATR-based TP/SL)
Вместо бинарного "цена вырастет" — метка +1 при пробое ATR×1.5 вверх, -1 при пробое ATR×1.0 вниз, 0 если таймаут (12 баров). Приближает target к реальной торговле.
Файл: `experiment_5m.py:70-88`, функция `triple_barrier_atr`

### Whale vs Retail divergence
Трейды ≥1000 qty = whale, меньше = retail. `whale_vs_retail = whale_buy_ratio - retail_buy_ratio`. Если киты покупают, розница продаёт — разворот.
Файл: `experiment.py:48-81`, SQL в `load_data()`

### Hurst regime filter + OFI acceleration
Торговать только при Hurst > 0.5 (трендовый режим). OFI_accel = diff(taker_pressure, 5) — вторая производная потока заявок.
Файл: `experiment.py:104-110, 163-165`, функции `rolling_hurst`, `add_features`

### Funding extreme filter
Торговать только при |funding| > 0.0001. Гипотеза: экстремальное фондирование → mean reversion.
Файл: `experiment_funding.py:273-280`

### Permutation test baseline
Перемешать target, переобучить, сравнить PF с реальным. Если разницы нет — модель ловит шум.
Файл: `experiment_5m.py:302-313`

### Composite score: Sortino × min(PF, 2) × Acc
Лучше, чем оптимизация по одной метрике. PF > 2 не улучшает score — фокус на стабильность.
Файл: `optuna_indicators.py:230`, функция `objective`

## Текущая сессия

- `data_fetcher/binance_vision/fetch_klines.py` — добавлены `_sync_from_bucket`, `_sync_to_bucket`, проверка HF Bucket первой очередью, авто-sync после загрузки
- `data_fetcher/binance_vision/fetch_funding.py` — то же самое (HF Bucket > Parquet > DuckDB > S3 > sync)
- Установлен `hf` CLI v1.21.0
- `hf skills add --global` — skill для OpenCode
- Документация распределена: `progress.md`, `data_fetcher/README.md`, `alpha_research/README.md`
