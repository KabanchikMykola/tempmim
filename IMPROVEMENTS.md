# Улучшения data_fetcher

## 1. Удалить мёртвый код

### fetch_book_depth.py и fetch_agg_trades.py
- Используют удалённый DuckDB
- Нет CLI-роута в `__main__.py`
- **Удалить оба файла** (~340 строк)

### binance_api/klines.py
- 26 строк, тривиальный wrapper: `return tail_klines(...)`
- Инлайнить импорт в `fetch_klines.py:66` и удалить файл

### config.py:23
- `CACHE_DIR = DATA_DIR / "cache"` — не используется после удаления DuckDB
- Удалить строку

## 2. Объединить bucket-хелперы (самое большое упрощение)

**Проблема:** `_load_from_bucket` / `_upload_to_bucket` скопипащены 3 раза:
- `fetch_klines.py:77-96`
- `fetch_funding.py:29-50`
- `fetch_metrics.py:108-125`

Отличаются только шаблоном URI.

**Решение:** Одна generic-функция в `config.py`:

```python
def bucket_load(uri: str):
    """Прочитать parquet из HF bucket."""
    if not BUCKET_ID:
        return None
    try:
        df = pd.read_parquet(uri)
        return df if not df.empty else None
    except Exception:
        return None

def bucket_upload(uri: str, df):
    """Записать parquet в HF bucket."""
    if df is None or df.empty or not BUCKET_ID:
        return
    df.to_parquet(uri, index=False)
```

**Экономия:** ~50 строк дублированного кода.

## 3. Объединить download_monthly и download_daily

**Проблема:** `fetch_klines.py:99-164` — две функции с 90% одинакового кода (CSV parsing, column names, type conversion). Разница — только URL.

**Решение:** Одна `_download_archive(url, ...)`:

```python
def _download_archive(url):
    """Скачать CSV-архив klines (monthly или daily)."""
    try:
        resp = requests.get(url, timeout=config.VISION_TIMEOUT)
        if resp.status_code != 200:
            return pd.DataFrame()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                content = f.read().decode("utf-8")
                lines = content.strip().split("\n")
                if not lines:
                    return pd.DataFrame()
                if lines[0].startswith("open_time"):
                    lines = lines[1:]
                data = [line.split(",") for line in lines]
                df = pd.DataFrame(data, columns=COLUMNS)
                for col in NUMERIC_COLS:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return _normalize_timestamps(df)
    except Exception:
        return pd.DataFrame()
```

**Экономия:** ~30 строк.

## 4. Вынести tail-логику metrics в хелпер

**Проблема:** `fetch_metrics.py` — блоки строк 172-208 и 265-296 почти идентичны (~30 строк каждый).

**Решение:**

```python
def _append_metrics_tail(df, symbol, start_time):
    """Докачать OI + taker vol через REST API."""
    max_ts = df["ts"].max()

    oi_df = tail_open_interest_hist(symbol, start_time=max_ts + 1)
    if not oi_df.empty:
        oi_new = oi_df[oi_df["ts"] > max_ts]
        if not oi_new.empty:
            oi_new["symbol"] = symbol
            oi_new["create_time"] = oi_new["ts"]
            for col in OI_NULL_COLS:
                oi_new[col] = None
            df = pd.concat([df, oi_new], ignore_index=True)

    tv_df = tail_taker_vol_ratio(symbol, start_time=max_ts + 1)
    if not tv_df.empty:
        tv_new = tv_df[tv_df["ts"] > max_ts]
        if not tv_new.empty:
            tv_new["symbol"] = symbol
            tv_new["create_time"] = tv_new["ts"]
            for col in TV_NULL_COLS:
                tv_new[col] = None
            df = pd.concat([df, tv_new], ignore_index=True)

    return df.drop_duplicates(subset=["symbol", "ts"]).sort_values("ts").reset_index(drop=True)
```

**Экономия:** ~30 строк.

## 5. Удалить дублирующую валидацию metrics

**Проблема:** `fetch_metrics.py:131-144` определяет `_validate_metrics()`, которая делает то же, что `contracts.validate_metrics()`.

**Решение:** Удалить локальную версию, использовать `contracts.validate_metrics()`.

## 6. Удалить неиспользуемый параметр export_parquet

**Проблема:** `fetch_klines.py:190` — `export_parquet=True` не влияет на поведение.

**Решение:** Удалить параметр.

## 7. Исправить документацию

**Проблема:** `fetch_klines.py:3-6` упоминает "DuckDB кеш" как уровень 2.

**Решение:** Обновить docstring.

## Итого

| Что | Экономия строк |
|-----|---------------|
| Удалить мёртвые файлы | ~340 |
| Generic bucket-хелперы | ~50 |
| merge download_monthly/daily | ~30 |
| tail-хелпер metrics | ~30 |
| Прочее | ~10 |
| **Итого** | **~460** |
