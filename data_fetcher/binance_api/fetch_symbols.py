"""Получение USDT пар с Binance Spot и Perpetual.
Возвращает только те пары, что есть на ОБОИХ рынках."""

import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone

import requests

from data_fetcher import config

BINANCE_SPOT_API = "https://api.binance.com"
BINANCE_PERPETUAL_API = "https://fapi.binance.com"

QUOTE_ASSET = "USDT"
STATUS_TRADING = "TRADING"
CONTRACT_TYPE_PERPETUAL = "PERPETUAL"

TIMEOUT_SECONDS = 10
MAX_RETRIES = 3
RETRY_BACKOFF = 2


def http_get_with_retry(url, retries=MAX_RETRIES):
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            last_error = f"Timeout after {TIMEOUT_SECONDS}s"
        except requests.exceptions.HTTPError as e:
            last_error = f"HTTP {e.response.status_code}: {e}"
            if e.response.status_code < 500:
                return None
        except requests.exceptions.RequestException as e:
            last_error = str(e)
        if attempt < retries - 1:
            time.sleep(RETRY_BACKOFF ** attempt)
    return None


def fetch_spot_symbols():
    """Получить все USDT спот-пары в статусе TRADING."""
    data = http_get_with_retry(f"{BINANCE_SPOT_API}/api/v3/exchangeInfo")
    if data is None:
        return []
    symbols = []
    for item in data.get("symbols", []):
        if item.get("quoteAsset") == QUOTE_ASSET and item.get("status") == STATUS_TRADING:
            symbols.append({
                "symbol": item["symbol"],
                "baseAsset": item["baseAsset"],
                "quoteAsset": item["quoteAsset"],
                "status": item["status"],
            })
    return symbols


def fetch_perpetual_symbols():
    """Получить все USDT-M PERPETUAL пары в статусе TRADING."""
    data = http_get_with_retry(f"{BINANCE_PERPETUAL_API}/fapi/v1/exchangeInfo")
    if data is None:
        return []
    symbols = []
    for item in data.get("symbols", []):
        if (item.get("quoteAsset") == QUOTE_ASSET
                and item.get("contractType") == CONTRACT_TYPE_PERPETUAL
                and item.get("status") == STATUS_TRADING):
            symbols.append({
                "symbol": item["symbol"],
                "baseAsset": item["baseAsset"],
                "quoteAsset": item["quoteAsset"],
                "status": item["status"],
            })
    return symbols


def find_common_symbols(spot, perpetual):
    """Найти пары, которые есть и на споте, и на перпах."""
    perp_set = {s["symbol"] for s in perpetual}
    return [s for s in spot if s["symbol"] in perp_set]


def get_server_time():
    try:
        response = requests.get(f"{BINANCE_SPOT_API}/api/v3/time", timeout=TIMEOUT_SECONDS)
        return response.json().get("serverTime", int(time.time() * 1000))
    except Exception:
        return int(time.time() * 1000)


def save_json(filepath, data):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def fetch_all(output_dir=None):
    """Основная функция: получить все пары со всех рынков, сохранить в JSON.

    Args:
        output_dir: Путь к папке для JSON. По умолчанию config.DATA_DIR/symbols.

    Returns:
        dict со списками spot, perpetual, common и server_time.
    """
    if output_dir is None:
        output_dir = Path(config.DATA_DIR).parent / "symbols"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Получение спотовых пар...")
    spot = fetch_spot_symbols()
    print(f"  Спот USDT: {len(spot)}")

    print("Получение перп-пар...")
    perpetual = fetch_perpetual_symbols()
    print(f"  Перп USDT: {len(perpetual)}")

    server_time = get_server_time()
    common = find_common_symbols(spot, perpetual)
    print(f"  Общих (spot + perp): {len(common)}")

    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {
        "timestamp": timestamp,
        "serverTime": server_time,
        "source": "binance_api",
        "quoteAsset": QUOTE_ASSET,
        "filters": {
            "quoteAsset": [QUOTE_ASSET],
            "market": ["spot", "perpetual"],
            "status": STATUS_TRADING,
            "contractType": CONTRACT_TYPE_PERPETUAL,
        },
    }

    def with_meta(symbols, label):
        return {"metadata": {**metadata, "label": label, "count": len(symbols)}, "pairs": symbols}

    save_json(output_dir / "spot_usdt_symbols.json", with_meta(spot, "spot"))
    save_json(output_dir / "perpetual_usdt_symbols.json", with_meta(perpetual, "perpetual"))
    save_json(output_dir / "spot_perpetual_common_usdt.json", with_meta(common, "common"))

    save_json(output_dir / "fetch_metadata.json", {
        "timestamp": timestamp,
        "serverTime": server_time,
        "source": "binance_api",
        "counts": {"spot": len(spot), "perpetual": len(perpetual), "common": len(common)},
    })

    print(f"\nСохранено в {output_dir}/")
    return {"spot": spot, "perpetual": perpetual, "common": common, "server_time": server_time}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Получение USDT пар с Binance (спот + перпы)")
    parser.add_argument("--output-dir", help="Папка для сохранения JSON")
    parser.add_argument("--list", action="store_true", help="Вывести список общих пар")
    args = parser.parse_args()

    result = fetch_all(args.output_dir)

    if args.list:
        print(f"\nОбщие пары ({len(result['common'])}):")
        for s in result["common"]:
            print(f"  {s['symbol']}")

    print(f"Готово.")


if __name__ == "__main__":
    main()
