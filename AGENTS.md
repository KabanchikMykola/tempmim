# AGENTS.md

Cross-exchange crypto data fetcher and alpha research toolkit. Python 3.12+, `uv` package manager.

## Commands

```sh
uv sync                           # установка
python -m data_fetcher list       # список данных в bucket + локально
python -m data_fetcher menu       # интерактивное меню (выбор пар, типов, загрузка)
python -m data_fetcher menu --run # быстрый запуск с сохранённым конфигом

# Прямые CLI команды (без меню):
python -m data_fetcher ohlcv vision --symbol BTCUSDT --years 1 --tail
python -m data_fetcher funding --symbol BTCUSDT --years 1
python -m data_fetcher metrics --symbol BTCUSDT --years 1
python -m data_fetcher hyperliquid --symbol BTC --years 1
```

## Структура проекта (упрощённая)

```
data_fetcher/
  __main__.py       — CLI: list, menu, ohlcv vision, funding, metrics, hyperliquid
  config.py         — переменные (DATA_DIR, BUCKET_ID, WORKERS, HF_TOKEN)
  contracts.py      — валидация данных
  menu.py           — интерактивное меню (сохраняет настройки в menu_config.py)
  hyperliquid/      — HyperLiquid fetcher (native API + опционально Chainticks)
  binance_vision/   — исторические данные из data.binance.vision S3
    fetch_klines.py — OHLCV
    fetch_funding.py
    fetch_metrics.py
  binance_api/
    tail.py         — REST API для свежих данных
    klines.py       — fetch_tail прокси
```

Хранилище: **только HuggingFace Bucket** (`hf://buckets/Kabanchik/mimo/fin_data/binance/`). DuckDB удалён. Локального кеша нет.

## 🛡 Guardrails

### 1. Think Before Coding
Перед любым изменением кода опиши план текстом:
- Какие данные ожидаются (multi-symbol? single-symbol?)
- Какие предположения о структуре DataFrame
- Почему выбран этот подход

### 2. Simplicity First
- Минимум классов, максимум ясности
- Используй стандартные библиотеки (pandas, numpy)
- Избегай абстракций «на будущее» (YAGNI)
- Функции небольшие (20–50 строк), с единой ответственностью

### 3. Surgical Changes
- Изменяй **только** строки, относящиеся к задаче
- Не трогай форматирование, импорты, комментарии в других частях файла
- Если нужен рефакторинг — отдельная задача/коммит

### 4. Goal-Driven Execution
Задача не считается выполненной, пока:
- ✅ Код написан и проверен через `uv run python -m data_fetcher ...`
- ✅ Данные в Parquet/bucket корректны (нет дубликатов, валидация пройдена)
- ✅ AGENTS.md или подпапка README обновлены (если изменилась архитектура)

## Critical rules (MUST read)

1. **Lookahead bias** — все TA-индикаторы (RSI, BB, ATR) обязательно через `.shift(1)`. `close[t]` для предсказания t — утечка.
2. **Quality > quantity** — 800+ фич провалились; топ-50 focused работают. `num_leaves=5` лучше сложных моделей.
3. **Target horizon** — долгосрочные (window=72) стабильнее коротких (window=24).
4. **Composite metric** — DA + Corr (или DA + F1), а не DA в одиночку.
5. **Cross-asset features** — BTC vs ETH spreads — сильные предикторы (ещё не реализовано).
6. **Funding rate** — короткие лаги бесполезны, сигнал только от 24h+.

## Menu config

`menu_config.py` создаётся меню, можно править руками:

```python
symbols = ["BTCUSDT", "ETHUSDT"]
years = 1
types = {"spot": True, "perp": True, "funding": False, "metrics": False}
upload = False
```

## Style

- Русские комментарии, print, имена переменных
- `sys.stdout = io.TextIOWrapper(...)` в CLI
- Всегда спрашивать перед `git push`
- Нет тестов / CI — проверка через `uv run python -m data_fetcher ...`

## Data and external storage

- HuggingFace Bucket: `hf://buckets/Kabanchik/mimo/fin_data/`
- Parquet в `fin_data/binance/{ohlcv_spot,ohlcv_perp,funding,metrics}/`
- Чтение из bucket работает без токена. Запись требует `HF_TOKEN`
- DuckDB **удалён** — данные хранятся только в bucket, локально не кешируются

## Environment: Google Cloud Shell (persistent disk экономия)

**Проблема:** `/home` — 4.8G, был забит (4.6G). Решение: тяжёлые папки в `/tmp` (эфемерный, 23G).

**Схема (`.startup.sh` + `.bashrc`):**
- `UV_CACHE_DIR=/tmp/uv-cache` — uv кэш
- `~/.venv` → `/tmp/tempmim-venv` (uv venv + uv sync на старте)
- `~/.npm` → `/tmp/npm-cache`
- `~/.local/lib` → `/tmp/session-local-lib`
- `~/.config/manicode` → `/tmp/manicode-config`

**При старте сессии:**
1. `.startup.sh` ставит symlink-и + `uv venv` + `uv sync` (~15-30 сек)
2. `npm install -g opencode-ai` если opencode не найден

**НИЧЕГО НЕ ДЕЛАТЬ на локальных машинах (Windows/Mac/Linux)** — там persistent диск нормального размера, `/tmp` может быть мал или cleared при перезагрузке. skip все переносы.

## Data Sources: HF Datasets

### HyperLiquid
- **native API** `candleSnapshot` — 5000 свечей (~7 мес 1h), основа
- **Chainticks/perp-data** — 1-min snapshots 2023-2024, `--chain` (медленно)

### Binance
- **data.binance.vision S3** — исторические OHLCV, funding, metrics
- **REST API** — свежие данные (tail)

## References

- `data_fetcher/README.md` — CLI examples, data structure, columns, contracts, validation
- `alpha_research/README.md` — list of scripts, data format, how to run
- `progress.md` — changelog / session history
