# Cross-Exchange Crypto Data Fetcher

Загрузка и анализ криптовалютных данных с бирж Binance, Bybit, OKX и других.

## Возможности

- **OHLCV** — спот + перпетуалы через ccxt API или S3-архивы Binance
- **Funding Rate** — история ставок финансирования
- **WebSocket** — реалтайм стриминг сделок и стакана (Binance, OKX, Bitget)
- **Аудит бирж** — проверка доступности данных на 17+ биржах
- **Alpha Research** — стратегии: basis trading, XGBoost, LightGBM + Optuna

## Быстрый старт

```sh
# Установка
uv sync

# Загрузка данных (BTC, ETH, SOL + другие)
python -m data_fetcher ohlcv ccxt --common --since 2026-01-01

# Загрузка из S3-архивов
python -m data_fetcher ohlcv vision --symbol BTCUSDT --interval 1h

# Фандинг rate
python -m data_fetcher funding --symbol BTCUSDT ETHUSDT

# Аудит бирж
python -m data_fetcher exchange-audit --exchanges binance bybit okx

# Загрузка в HuggingFace Bucket
python -m data_fetcher ohlcv ccxt --common --upload --bucket Kabanchik/mimo
```

## Структура

```
data_fetcher/           # Основной пакет
  ccxt_api/             # OHLCV через ccxt, аудит бирж
  binance_vision/       # S3-архивы data.binance.vision
  binance_api/          # REST API Binance (exchangeInfo)


alpha_research/         # Скрипты исследования
  strategies_basis.py           # Basis z-score стратегия
  strategies_xgboost.py         # XGBoost для предсказания направления
  strategies_lgbm_optuna.py     # LightGBM + Optuna оптимизация
  signal_search_v4.py           # Поиск стабильных сигналов
  volatility_regime_v3.py       # Стратегия на волатильностных режимах
  strategies.json               # Сохранённые лучшие стратегии
```

## Данные

Файлы parquet в `data/`:
- `{SYMBOL}_1h_spot.parquet` — спот OHLCV
- `{SYMBOL}_1h_perp.parquet` — перпетуалы OHLCV
- `{SYMBOL}_funding.parquet` — funding rate

## Требования

- Python >= 3.12
- `uv` для управления зависимостями
