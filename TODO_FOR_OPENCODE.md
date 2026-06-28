# Оставшиеся замечания

Два пункта из IMPROVEMENTS.md ещё не выполнены. Ниже — подробное описание каждого.

---

## 1. Удалить `binance_api/klines.py`

### Что это за файл

`data_fetcher/binance_api/klines.py` — 26 строк. Это тривиальный wrapper-мёртвый груз:

```python
from data_fetcher.binance_api.tail import tail_klines

def fetch_tail(symbol, interval="1h", perp=False, limit=1000, end_time=None):
    return tail_klines(symbol=symbol, interval=interval, perp=perp, limit=limit, end_time=end_time)
```

Функция `fetch_tail` просто делегирует вызов `tail_klines` из `tail.py`, не добавляя никакой логики.

### Где используется

Единственный импорт — `fetch_klines.py:63`:

```python
from data_fetcher.binance_api.klines import fetch_tail
```

И далее `fetch_tail(...)` вызывается 3 раза в `fetch_klines.py` (строки 185, 205, 287).

### Что сделать

1. В `fetch_klines.py:63` заменить импорт:
   ```python
   # Было:
   from data_fetcher.binance_api.klines import fetch_tail

   # Стало:
   from data_fetcher.binance_api.tail import tail_klines as fetch_tail
   ```

   Или переименовать вызовы `fetch_tail(...)` → `tail_klines(...)` и импортировать напрямую:
   ```python
   from data_fetcher.binance_api.tail import tail_klines
   ```

2. Удалить файл `data_fetcher/binance_api/klines.py`

### Почему

- 26 строк кода, которые не делают ничего полезного
- Единственная функция — обёртка без добавленной логики
- При наличии `tail.py` эта прослойка не нужна
- Меньше файлов = проще навигация по проекту

---

## 2. Убрать дублирующие обёртки bucket-хелперов

### Текущая ситуация

В `config.py` уже есть generic-функции:

```python
def bucket_load(uri: str):
    """Прочитать parquet из HF bucket."""
    ...

def bucket_upload(uri: str, df):
    """Записать parquet в HF bucket."""
    ...
```

Все три fetcher-файла используют их. Но в каждом файле остались **тонкие обёртки**, которые строят URI и вызывают generic-функции:

### fetch_klines.py (строки 69-83)

```python
def _bucket_parquet_uri(symbol, interval, source):
    market_dir = "ohlcv_perp" if source == "perp" else "ohlcv_spot"
    return f"{config.BUCKET_URI}/{market_dir}/{symbol}_{interval}.parquet"

def _bucket_uri(symbol, interval, source):
    return _bucket_parquet_uri(symbol, interval, source)

def _load_from_bucket(symbol, interval, source):
    return config.bucket_load(_bucket_uri(symbol, interval, source))

def _upload_to_bucket(symbol, interval, source, df):
    config.bucket_upload(_bucket_uri(symbol, interval, source), df)
```

### fetch_funding.py (строки 29-38)

```python
def _bucket_parquet_uri(symbol):
    return f"{config.BUCKET_URI}/funding/{symbol}_funding.parquet"

def _load_from_bucket(symbol):
    return config.bucket_load(_bucket_parquet_uri(symbol))

def _upload_to_bucket(symbol, df):
    config.bucket_upload(_bucket_parquet_uri(symbol), df)
```

### fetch_metrics.py (строки 53, 108-113)

```python
def _bucket_parquet_uri(symbol):
    return f"{config.BUCKET_URI}/metrics/{symbol}_metrics.parquet"

def _load_from_bucket(symbol):
    return config.bucket_load(_bucket_parquet_uri(symbol))

def _upload_to_bucket(symbol, df):
    config.bucket_upload(_bucket_parquet_uri(symbol), df)
```

### Что сделать

Убрать `_load_from_bucket` и `_upload_to_bucket` из каждого файла. Везде, где они вызываются, заменить на прямой вызов `config.bucket_load` / `config.bucket_upload` с URI.

**Пример для fetch_funding.py:**

Было:
```python
existing = _load_from_bucket(symbol)
...
_upload_to_bucket(symbol, merged)
```

Стало:
```python
uri = f"{config.BUCKET_URI}/funding/{symbol}_funding.parquet"
existing = config.bucket_load(uri)
...
config.bucket_upload(uri, merged)
```

То же самое для `fetch_klines.py` и `fetch_metrics.py`.

Обёртку `_bucket_parquet_uri` тоже можно убрать — она становится не нужной, когда вызов идёт напрямую.

### Почему

- Дублированный код (3 файла × 3 обёртки = 9 функций, которые делают одно и то же)
- Сейчас обёртки настолько тонкие (1-2 строки), что они не добавляют ясности, а только увеличивают количество функций для перехода при навигации
- Прямой вызов `config.bucket_load(uri)` / `config.bucket_upload(uri, df)` читается так же понятно

### Альтернатива (оставить как есть)

Если не хочется менять — можно оставить. Обёртки тривиальны (2-3 строки каждая) и не ломают ничего. Это вопрос стиля, а не бага.
