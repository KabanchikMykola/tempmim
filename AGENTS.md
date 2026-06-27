# AGENTS.md

Cross-exchange crypto data fetcher and alpha research toolkit. Python 3.12+, `uv` package manager.

## Commands

```sh
uv sync                    # установка (без ML)
uv sync --extra ml         # + ML (lightgbm, optuna, neuralforecast)
uv add <package>           # добавить зависимость
python -m data_fetcher ... # запуск (см. data_fetcher/README.md)
```

## 🛡 Guardrails

### 1. Think Before Coding
Перед любым изменением кода опиши план текстом:
- Какие данные ожидаются (multi-symbol? single-symbol?)
- Какие предположения о структуре DataFrame
- Почему выбран этот подход

### 2. Simplicity First
- Минимум классов, максимум ясности
- Используй стандартные библиотеки (pandas, numpy, duckdb)
- Избегай абстракций «на будущее» (YAGNI)
- Функции небольшие (20–50 строк), с единой ответственностью

### 3. Surgical Changes
- Изменяй **только** строки, относящиеся к задаче
- Не трогай форматирование, импорты, комментарии в других частях файла
- Если нужен рефакторинг — отдельная задача/коммит

### 4. Goal-Driven Execution
Задача не считается выполненной, пока:
- ✅ Код написан и проверен через `uv run python -m data_fetcher ...`
- ✅ Данные в Parquet/БД корректны (нет дубликатов, валидация пройдена)
- ✅ AGENTS.md или подпапка README обновлены (если изменилась архитектура)

## Critical rules (MUST read)

1. **Lookahead bias** — все TA-индикаторы (RSI, BB, ATR) обязательно через `.shift(1)`. `close[t]` для предсказания t — утечка.
2. **Quality > quantity** — 800+ фич провалились; топ-50 focused работают. `num_leaves=5` лучше сложных моделей.
3. **Target horizon** — долгосрочные (window=72) стабильнее коротких (window=24).
4. **Composite metric** — DA + Corr (или DA + F1), а не DA в одиночку.
5. **Cross-asset features** — BTC vs ETH spreads — сильные предикторы (ещё не реализовано).
6. **Funding rate** — короткие лаги бесполезны, сигнал только от 24h+.

## Style

- Русские комментарии, print, имена переменных
- `sys.stdout = io.TextIOWrapper(...)` в CLI
- Всегда спрашивать перед `git push`
- Нет тестов / CI — проверка через `uv run python -m data_fetcher ...`

## Data and external storage

- HuggingFace Bucket: `hf://buckets/Kabanchik/mimo/fin_data/`
- Parquet в `fin_data/binance/{ohlcv_spot,ohlcv_perp,funding,metrics}/`
- После `hf auth login` fetcher-ы сами подтягивают данные из bucket

## References

- `data_fetcher/README.md` — CLI examples, data structure, columns, contracts, validation, DuckDB caveats, tail, storage
- `alpha_research/README.md` — list of scripts, data format, how to run
- `progress.md` — changelog / session history
