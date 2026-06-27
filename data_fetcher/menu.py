"""Интерактивное меню для загрузки данных с Binance.

1. Выбор даты начала (YYYY-MM-DD)
2. Получение пар, проверка даты листинга (кеш 1ч)
3. Фильтрация по ликвидности, выбор пар
4. Выбор типов данных (OHLCV spot/perp, funding, metrics)
5. Загрузка
6. (Опционально) Upload в HuggingFace Bucket

Использование:
    python -m data_fetcher menu
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import ccxt

from data_fetcher import config


BINANCE_SPOT_API = "https://api.binance.com/api/v3"
BINANCE_PERP_API = "https://fapi.binance.com/fapi/v1"

STABLECOINS = frozenset({
    "USDC", "FDUSD", "USDP", "TUSD", "DAI", "USDD", "BUSD",
    "USDY", "USDM", "FRAX", "LUSD", "GHO", "CRVUSD",
})

CACHE_PATH = config.CACHE_DIR / "menu_cache.json"
CACHE_TTL = 3600  # 1 час


# ── Кеш ───────────────────────────────────────────────────────────


def _load_cache():
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(data):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ── Ввод даты ─────────────────────────────────────────────────────


def _input(prompt, default=None):
    if default is not None:
        p = f"{prompt} [{default}]: "
    else:
        p = f"{prompt}: "
    val = input(p).strip()
    return val if val else default


def ask_date() -> datetime:
    now = datetime.now(timezone.utc)
    print()
    print("  Дата начала загрузки")
    print("  ─" * 12)

    raw = _input("  Дата (YYYY-MM-DD)", now.strftime("%Y-%m-%d"))
    try:
        start = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        parts = raw.split("-")
        y, m, d = int(parts[0]), int(parts[1]) if len(parts) > 1 else 1, int(parts[2]) if len(parts) > 2 else 1
        start = datetime(y, m, d, tzinfo=timezone.utc)

    if start >= now:
        print("  ❌ Дата начала должна быть раньше сегодня")
        sys.exit(1)
    print(f"  Период: {start.date()} → {now.date()}")
    return start


# ── Получение пар и проверка листинга ────────────────────────────


def _fetch_exchange_info(url):
    for _ in range(3):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json().get("symbols", [])
        except Exception:
            time.sleep(1)
    return []


def _fetch_first_kline(symbol) -> int | None:
    for _ in range(3):
        try:
            resp = requests.get(
                f"{BINANCE_SPOT_API}/klines",
                params={"symbol": symbol, "interval": "1M", "startTime": 0, "limit": 1},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return int(data[0][0]) if data else None
            return None
        except Exception:
            time.sleep(1)
    return None


def get_common_symbols(start: datetime, force_refresh=False) -> list[dict]:
    cache = _load_cache()
    now_ts = datetime.now(timezone.utc).isoformat()

    # exchangeInfo кеш
    if not force_refresh and "exchange_fetched" in cache:
        fetched = datetime.fromisoformat(cache["exchange_fetched"])
        if datetime.now(timezone.utc) - fetched < timedelta(seconds=CACHE_TTL):
            print("\n  Параметры (кеш)")
            common_symbols = cache["common_symbols"]
            excluded = cache.get("excluded_stablecoins", [])
            if excluded:
                print(f"    Исключено стейблкоинов: {len(excluded)} ({', '.join(excluded)})")
            print(f"    Общих: {len(common_symbols)}")
            return [{"symbol": s["symbol"], "baseAsset": s["baseAsset"]} for s in common_symbols]

    print("\n  Получение пар с Binance...")
    spot_data = _fetch_exchange_info(f"{BINANCE_SPOT_API}/exchangeInfo")
    spot = {
        s["symbol"]: s for s in spot_data
        if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
    }
    print(f"    Спот USDT: {len(spot)}")

    perp_data = _fetch_exchange_info(f"{BINANCE_PERP_API}/exchangeInfo")
    perp = {
        s["symbol"]: s for s in perp_data
        if s.get("quoteAsset") == "USDT"
        and s.get("contractType") == "PERPETUAL"
        and s.get("status") == "TRADING"
    }
    print(f"    Перп USDT: {len(perp)}")

    common_symbols = sorted(set(spot.keys()) & set(perp.keys()))
    excluded_stable = sorted({spot[s]["baseAsset"] for s in common_symbols if spot[s]["baseAsset"] in STABLECOINS})
    common_symbols = [s for s in common_symbols if spot[s]["baseAsset"] not in STABLECOINS]
    print(f"    Общих: {len(common_symbols)}")
    if excluded_stable:
        print(f"    Исключено стейблкоинов: {len(excluded_stable)} ({', '.join(excluded_stable)})")

    data_list = [{"symbol": s, "baseAsset": spot[s]["baseAsset"]} for s in common_symbols]
    _save_cache({
        "exchange_fetched": now_ts,
        "common_symbols": data_list,
        "excluded_stablecoins": excluded_stable,
    })
    return data_list


def check_listing_dates(symbols: list[dict], start: datetime, force_refresh=False) -> list[dict]:
    cache = _load_cache()
    listing_cache = cache.get("listing_dates", {})
    start_ms = int(start.timestamp() * 1000)
    total = len(symbols)
    checked = []
    missing = []

    print(f"\n  Проверка даты листинга {total} пар...")
    print(f"  Отсеиваем пары, появившиеся после {start.date()}")

    for sym in symbols:
        cached_ts = listing_cache.get(sym["symbol"])
        if cached_ts is not None and not force_refresh:
            if cached_ts == 0:
                checked.append({**sym, "first_kline": 0})
            elif cached_ts <= start_ms:
                checked.append({**sym, "first_kline": cached_ts})
            else:
                print(f"    {sym['symbol']:>12} — пропускаем (кеш, листинг {datetime.fromtimestamp(cached_ts / 1000, tz=timezone.utc).date()})")
        else:
            missing.append(sym)

    if missing:
        print(f"  Запрос к API для {len(missing)} пар...")
        def _check(sym):
            ts = _fetch_first_kline(sym["symbol"])
            return sym, ts

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(_check, s): s for s in missing}
            done = 0
            for f in as_completed(futures):
                done += 1
                sym, ts = f.result()
                listed = ts if ts else 0
                listing_cache[sym["symbol"]] = listed
                if listed > 0 and listed > start_ms:
                    print(f"    [{done}/{len(missing)}] {sym['symbol']:>12} — пропускаем (листинг {datetime.fromtimestamp(listed / 1000, tz=timezone.utc).date()})")
                else:
                    checked.append({**sym, "first_kline": listed})

    # сохраняем кеш листингов
    cache["listing_dates"] = listing_cache
    _save_cache(cache)

    print(f"  После фильтрации: {len(checked)}/{total} пар")
    return checked


# ── Объём и сортировка ──────────────────────────────────────────


def get_volumes(symbols: list[dict]) -> list[dict]:
    print("\n  Получение 24h объёмов...")

    volumes = {}

    def _batch_fetch(batch):
        symbols_json = json.dumps([s["symbol"] for s in batch], separators=(",", ":"))
        for _ in range(3):
            try:
                resp = requests.get(
                    f"{BINANCE_SPOT_API}/ticker/24hr",
                    params={"symbols": symbols_json},
                    timeout=15,
                )
                if resp.status_code == 200:
                    for t in resp.json():
                        volumes[t["symbol"]] = float(t.get("quoteVolume", 0) or 0)
                    return
            except Exception:
                time.sleep(1)

    for i in range(0, len(symbols), 100):
        _batch_fetch(symbols[i:i + 100])

    total_vol = sum(volumes.values())
    if total_vol == 0 and len(symbols) > 0:
        print("  ⚠ REST не вернул объёмы, пробуем ccxt...")
        try:
            ex = ccxt.binance({"enableRateLimit": True})
            ex.load_markets()
            tickers = ex.fetch_tickers()
            for s in symbols:
                t = tickers.get(s["symbol"])
                if t:
                    volumes[s["symbol"]] = float(t.get("quoteVolume", 0) or 0)
        except Exception as e:
            print(f"  ⚠ {e}")

    result = sorted(
        [{"symbol": s["symbol"], "baseAsset": s["baseAsset"],
          "volume_usd": volumes.get(s["symbol"], 0)} for s in symbols],
        key=lambda x: x["volume_usd"], reverse=True,
    )
    n_nonzero = sum(1 for r in result if r["volume_usd"] > 0)
    print(f"  Всего: {len(result)} пар (с объёмом: {n_nonzero})")
    return result


# ── Выбор пар ─────────────────────────────────────────────────────


def select_symbols(symbols: list[dict]) -> list[str]:
    print()
    print("  Топ пар по 24h объёму:")
    print(f"  {'':>4} {'Symbol':<12} {'Base':<8} {'Volume 24h':>14}")
    print(f"  {'':>4} {'──────':<12} {'────':<8} {'──────────':>14}")

    n = min(50, len(symbols))
    for i, s in enumerate(symbols[:n], 1):
        vol_str = f"${s['volume_usd'] / 1e6:.1f}M" if s['volume_usd'] >= 1e6 else f"${s['volume_usd']:,.0f}"
        print(f"  {i:>3}. {s['symbol']:<12} {s['baseAsset']:<8} {vol_str:>14}")
    print()

    choice = _input(
        "  Что загружаем (all / N / BTCUSDT,ETHUSDT,...)",
        "all"
    )

    if choice == "all":
        return [s["symbol"] for s in symbols]
    if choice.isdigit():
        n_top = int(choice)
        if n_top == 0:
            return []
        return [s["symbol"] for s in symbols[:n_top]]
    manual = [s.strip().upper() for s in choice.replace(",", " ").split() if s.strip()]
    known = {s["symbol"] for s in symbols}
    matched = [sym for sym in manual if sym in known]
    if not matched:
        print("  ❌ Ни один символ не найден в списке.")
        return []
    return matched


# ── Выбор типов данных ───────────────────────────────────────────


def select_data_types() -> dict:
    print()
    print("  Типы данных для загрузки:")
    print("  ─" * 12)

    types = {}
    for key, label in [
        ("spot", "OHLCV спот (1h)"),
        ("perp", "OHLCV перп (1h)"),
        ("funding", "Funding rate"),
        ("metrics", "Metrics (OI, long/short)"),
    ]:
        default = "y" if key != "metrics" else "n"
        val = _input(f"  {label}?", default)
        types[key] = val.lower() in ("y", "yes", "д", "да")

    return types


# ── Upload ────────────────────────────────────────────────────────


def ask_upload() -> str | None:
    print()
    if not config.BUCKET_ID:
        print("  Upload: BUCKET_ID не задан в config.py")
        return None

    val = _input(f"  Загрузить в HuggingFace Bucket ({config.BUCKET_ID})?", "y")
    if val.lower() in ("y", "yes", "д", "да"):
        return config.BUCKET_ID
    return None


# ── Загрузка ──────────────────────────────────────────────────────


def _years_since(start: datetime) -> int:
    now = datetime.now(timezone.utc)
    return max(1, (now.year - start.year) + 1)


def run_download(symbols: list[str], data_types: dict, start: datetime,
                 bucket_id: str | None):
    from data_fetcher.binance_vision.fetch_klines import fetch_symbol as fetch_klines
    from data_fetcher.binance_vision.fetch_funding import fetch_funding
    from data_fetcher.binance_vision.fetch_metrics import fetch_metrics
    from data_fetcher.ccxt_api.fetcher import upload_to_bucket

    years = _years_since(start)
    total_ops = 0
    for dt in ("spot", "perp", "funding", "metrics"):
        if data_types.get(dt):
            total_ops += len(symbols)
    ops_done = 0
    t0 = time.time()

    for dt, label in [
        ("spot", "OHLCV спот"),
        ("perp", "OHLCV перп"),
        ("funding", "Funding rate"),
        ("metrics", "Metrics"),
    ]:
        if not data_types.get(dt):
            continue

        for symbol in symbols:
            ops_done += 1
            pct = ops_done / total_ops * 100
            print(f"\n  [{ops_done}/{total_ops} {pct:.0f}%] {label}: {symbol}")
            print(f"  {'─' * 40}")

            try:
                if dt in ("spot", "perp"):
                    perp = dt == "perp"
                    df, warns = fetch_klines(
                        symbol, interval="1h", years=years, perp=perp,
                        export_parquet=True, tail=True,
                    )
                    print(f"    строк: {len(df):,}")
                    for w in warns:
                        print(f"    ⚠ {w}")

                elif dt == "funding":
                    df = fetch_funding(symbol, years=years, export_parquet=True)
                    print(f"    строк: {len(df):,}")

                elif dt == "metrics":
                    df, warns = fetch_metrics(symbol, years=years, export_parquet=True)
                    print(f"    строк: {len(df):,}")
                    for w in warns:
                        print(f"    ⚠ {w}")

            except Exception as e:
                print(f"    ❌ {e}")

    print(f"\n  {'=' * 50}")
    print(f"  Загрузка завершена за {time.time() - t0:.1f}s")

    if bucket_id:
        print(f"\n  Upload в bucket...")
        upload_to_bucket(config.DATA_DIR, bucket_id)
        print(f"  Готово: hf://buckets/{bucket_id}/{config.DATA_DIR.name}/")


# ── Основная функция ─────────────────────────────────────────────


def main():
    print()
    print("  ╔═══════════════════════════════════════╗")
    print("  ║  Binance Data Downloader — Меню      ║")
    print("  ╚═══════════════════════════════════════╝")

    start = ask_date()

    common = get_common_symbols(start)
    if not common:
        print("  ❌ Нет общих пар.")
        return

    filtered = check_listing_dates(common, start)
    if not filtered:
        print("  ❌ Все пары отсеяны по дате листинга.")
        return

    with_volumes = get_volumes(filtered)
    symbols = select_symbols(with_volumes)
    if not symbols:
        print("  Пары не выбраны.")
        return

    data_types = select_data_types()
    active = [k for k, v in data_types.items() if v]
    if not active:
        print("  Ничего не выбрано.")
        return

    print(f"\n  Типы: {', '.join(active)}")

    bucket_id = ask_upload()

    print()
    print("  ┌── Сводка ───────────────────────────┐")
    print(f"  │  Период    : {start.date()} → {datetime.now(timezone.utc).date()}")
    print(f"  │  Пар       : {len(symbols)}")
    print(f"  │  Типы      : {', '.join(active)}")
    print(f"  │  Bucket    : {bucket_id or 'нет'}")
    print(f"  │  Папка     : {config.DATA_DIR}")
    print(f"  └──────────────────────────────────────┘")

    val = _input("\n  Начать загрузку?", "y")
    if val.lower() not in ("y", "yes", "д", "да"):
        print("  Отменено.")
        return

    run_download(symbols, data_types, start, bucket_id)
    print("  ✅ Готово")


if __name__ == "__main__":
    main()
