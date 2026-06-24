"""Аудит криптобирж: доступность данных для поиска альфы.

Проверяет биржи через ccxt на:
- Историю OHLCV, Funding Rate, Open Interest
- Long/Short Ratio (unified + прямые REST запросы)
- Ликвидации
- Топ пар по объёму

Использование:
    python -m data_fetcher exchange-audit
    python -m data_fetcher exchange-audit --exchanges binance bybit okx
"""

import ccxt
import pandas as pd
import time
import requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

EXCHANGES = [
    "binance", "bybit", "okx",
    "bitget", "gate", "htx",
    "mexc", "kucoin", "bitfinex",
    "kraken", "bingx", "cryptocom",
    "deribit", "coinbase",
    "woo", "poloniex", "gemini",
]

SINCE_MS = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000)
TIMEOUT = 15000


def init_exchange(name: str, market_type: str = "swap"):
    config = {"enableRateLimit": True, "timeout": TIMEOUT, "options": {}}
    if market_type == "swap":
        if name == "binance":
            config["options"]["defaultType"] = "future"
        elif name == "bybit":
            config["options"]["defaultType"] = "linear"
        elif name == "kucoin":
            config["options"]["defaultType"] = "futures"
        else:
            config["options"]["defaultType"] = "swap"
    try:
        ex_class = getattr(ccxt, name)
        return ex_class(config)
    except AttributeError:
        return None


def find_swap_symbol(exchange):
    """Найти BTC/USDT swap символ на бирже (linear, USDT-settle)."""
    for s in exchange.markets:
        m = exchange.markets[s]
        if (m.get("swap") and m.get("linear")
                and m.get("base") == "BTC" and m.get("quote") == "USDT"
                and m.get("settle") == "USDT"):
            return m["symbol"]
    fallback = None
    for s in exchange.markets:
        m = exchange.markets[s]
        if (m.get("swap") and m.get("linear")
                and m.get("base") == "BTC" and m.get("quote") == "USDT"):
            if fallback is None:
                fallback = m["symbol"]
    return fallback


def try_longshort_via_rest(exchange_name: str, symbol: str) -> dict:
    """Проверить L/S Ratio через прямые REST запросы (не unified)."""
    result = {"available": False, "data_since": None, "method": None}
    try:
        if exchange_name == "bybit" and symbol:
            clean = symbol.replace("/", "").replace(":", "")
            url = "https://api.bybit.com/v5/market/account-ratio"
            params = {"category": "linear", "symbol": clean, "period": "1d", "limit": 5}
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                    result["available"] = True
                    result["method"] = "bybit_v5_public"
                    first = data["result"]["list"][-1]
                    ts = first.get("timestamp", "")
                    result["data_since"] = str(ts)[:10] if ts else None

        elif exchange_name == "okx":
            url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio"
            resp = requests.get(url, params={"ccy": "BTC", "period": "1D", "limit": 5}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == "0" and data.get("data"):
                    result["available"] = True
                    result["method"] = "okx_ls_ratio_public"

        elif exchange_name == "binance":
            url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
            resp = requests.get(url, params={"symbol": "BTCUSDT", "period": "1d", "limit": 5}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    result["available"] = True
                    result["method"] = "binance_fapi_public"
                    ts = data[0].get("timestamp", "")
                    result["data_since"] = str(ts)[:10] if ts else None

        elif exchange_name == "bitget":
            url = "https://api.bitget.com/api/v2/mix/market/account-long-short-ratio"
            params = {"symbol": "BTCUSDT", "marginCoin": "USDT", "productType": "umcbl", "limit": 5}
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == "00000" and data.get("data"):
                    result["available"] = True
                    result["method"] = "bitget_public"
                    result["data_since"] = str(data["data"][-1].get("timestamp", ""))[:10]

        elif exchange_name == "gate":
            url = "https://api.gateio.ws/api/v4/delivery/btc/usdt/long_short_ratio"
            resp = requests.get(url, params={"limit": 5}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    result["available"] = True
                    result["method"] = "gate_public"
                    result["data_since"] = str(data[-1].get("time", ""))[:10]

        elif exchange_name == "bingx":
            url = "https://open-api.bingx.com/openApi/swap/v2/account/longShortRatio"
            params = {"symbol": "BTC-USDT", "period": "1d", "limit": 5}
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == 0 and data.get("data"):
                    result["available"] = True
                    result["method"] = "bingx_public"
    except Exception:
        pass
    return result


def _check_funding_depth(exchange, symbol: str) -> str | None:
    """Проверить глубину истории funding rate, идя от текущей даты назад."""
    if not exchange.has.get("fetchFundingRateHistory"):
        return None
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    steps = [now_ms, now_ms - 365 * 86400000, now_ms - 2 * 365 * 86400000]
    for step in steps:
        for attempt in range(3):
            try:
                data = exchange.fetch_funding_rate_history(symbol, since=step, limit=5)
                if data and len(data) > 0:
                    first_ts = data[0]["timestamp"]
                    return datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc).strftime("%Y-%m")
            except Exception:
                time.sleep(1)
    return None


def check_exchange(exchange_name: str) -> dict:
    result = {
        "exchange": exchange_name,
        "status": "ok",
        "swap_symbol": None,
        "swap_pairs": 0,
        "has_ohlcv": False,
        "ohlcv_years": 0,
        "has_funding": False,
        "funding_since": None,
        "has_open_interest": False,
        "has_long_short_ratio_unified": False,
        "has_long_short_ratio_rest": False,
        "ls_ratio_method": None,
        "ls_data_since": None,
        "has_liquidations": False,
        "volume_btc_24h": 0,
        "methods": [],
        "error": None,
    }

    try:
        ex = init_exchange(exchange_name, "swap")
        if not ex:
            result["status"] = "no_module"
            return result

        ex.load_markets()
        has = ex.has

        unified_map = {
            "fetchFundingRateHistory": "has_funding",
            "fetchOpenInterest": "has_open_interest",
            "fetchLiquidations": "has_liquidations",
            "fetchLongShortRatio": "has_long_short_ratio_unified",
        }
        for method, key in unified_map.items():
            if has.get(method):
                result[key] = True
                result["methods"].append(method)

        swap_symbol = find_swap_symbol(ex)
        result["swap_symbol"] = swap_symbol
        result["swap_pairs"] = sum(1 for s in ex.markets if ex.markets[s].get("swap"))

        if swap_symbol and has.get("fetchOHLCV"):
            try:
                tf = "1D" if exchange_name == "bitfinex" else "1d"
                data = ex.fetch_ohlcv(swap_symbol, tf, limit=365 * 3)
                if data and len(data) >= 2:
                    oldest = datetime.fromtimestamp(data[0][0] / 1000, tz=timezone.utc)
                    result["ohlcv_years"] = round((datetime.now(timezone.utc) - oldest).days / 365, 1)
                    result["has_ohlcv"] = True
            except Exception:
                pass

        if swap_symbol:
            result["funding_since"] = _check_funding_depth(ex, swap_symbol)
            result["has_funding"] = result["funding_since"] is not None

        if swap_symbol:
            try:
                ticker = ex.fetch_ticker(swap_symbol)
                if ticker and ticker.get("quoteVolume"):
                    result["volume_btc_24h"] = round(ticker["quoteVolume"] / max(ticker.get("last", 1), 0.01), 1)
            except Exception:
                pass

        ls_rest = try_longshort_via_rest(exchange_name, swap_symbol)
        if ls_rest["available"]:
            result["has_long_short_ratio_rest"] = True
            result["ls_ratio_method"] = ls_rest["method"]
            result["ls_data_since"] = ls_rest["data_since"]

        time.sleep(0.3)

    except ccxt.AuthenticationError:
        result["status"] = "no_auth"
    except ccxt.NotSupported:
        result["status"] = "unsupported"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:80]

    return result


def print_results(results: list[dict]):
    try:
        from tabulate import tabulate
        tablefmt = "simple"
    except ImportError:
        tabulate = None
        tablefmt = None

    print("\n" + "=" * 120)
    print("   АУДИТ БИРЖ: ДОСТУПНОСТЬ ДАННЫХ ДЛЯ ПОИСКА АЛЬФЫ")
    print("=" * 120)

    rows = []
    for r in results:
        if r["status"] == "no_module":
            continue
        ls = "Y" if r["has_long_short_ratio_unified"] else ("~" if r["has_long_short_ratio_rest"] else "N")
        ls_info = r["ls_ratio_method"] or ""
        rows.append([
            r["exchange"],
            r["swap_pairs"],
            f"{r['ohlcv_years']}y" if r["ohlcv_years"] else "~",
            "Y" if r["has_funding"] else "N",
            r["funding_since"] or "-",
            "Y" if r["has_open_interest"] else "N",
            ls,
            ls_info,
            "Y" if r["has_liquidations"] else "N",
            f"{r['volume_btc_24h']:>8,.0f}" if r["volume_btc_24h"] else "-",
        ])

    headers = ["Биржа", "Swap", "OHLCV", "Funding", "FR с", "OI", "L/S", "L/S метод", "Ликв.", "Объём(BTC)"]
    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt=tablefmt))
    else:
        print("  ".join(headers))
        for row in rows:
            print("  ".join(str(x) for x in row))

    print(f"\nВсего: {sum(1 for r in results if r['status'] != 'no_module')} / {len(EXCHANGES)}")

    ls_all = [r for r in results if r["has_long_short_ratio_unified"] or r["has_long_short_ratio_rest"]]
    if ls_all:
        print(f"\nL/S Ratio доступен на: {', '.join(r['exchange'] for r in ls_all)}")

    deep_fr = [r["exchange"] for r in results if r["funding_since"] and r["funding_since"] <= "2022-06"]
    if deep_fr:
        print(f"Фандинг >2.5 лет: {', '.join(deep_fr)}")

    vol_sorted = sorted([r for r in results if r["volume_btc_24h"] > 0], key=lambda x: x["volume_btc_24h"], reverse=True)
    if vol_sorted:
        print(f"\nТоп по объёму (BTC swap, 24h):")
        for r in vol_sorted[:5]:
            print(f"  {r['exchange']:<12} {r['volume_btc_24h']:>10,.0f} BTC")


def run_audit(exchanges: list[str] | None = None):
    """Запустить аудит бирж."""
    ex_list = exchanges or EXCHANGES
    t0 = time.time()
    print(f"Аудит {len(ex_list)} бирж через ccxt v{ccxt.__version__}...")

    results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(check_exchange, name): name for name in ex_list}
        for i, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            r = future.result()
            results.append(r)
            status = "+" if r["status"] == "ok" else "!"
            vol = f"{r['volume_btc_24h']:>8,.0f} BTC" if r["volume_btc_24h"] else "   -"
            ls = " L/S!" if (r["has_long_short_ratio_unified"] or r["has_long_short_ratio_rest"]) else ""
            print(f"  [{i:>2}/{len(ex_list)}] {status} {name:<12} | пар: {r['swap_pairs']:<3} | {vol}{ls}", flush=True)

    print_results(results)
    print(f"\nЗавершено за {time.time() - t0:.0f}s")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Аудит криптобирж")
    parser.add_argument("--exchanges", nargs="+", help="Список бирж (по умолчанию все)")
    args = parser.parse_args()
    run_audit(args.exchanges)


if __name__ == "__main__":
    main()
