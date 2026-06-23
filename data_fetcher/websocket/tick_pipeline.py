"""Реалтайм стриминг тиковых данных (trades + orderbook).

Собирает сделки и снапшоты стакана с:
  - Binance Spot, Binance Perpetual
  - OKX Spot, OKX Perpetual
  - Bitget Spot, Bitget Perpetual

Агрегирует трейды в 10-секундные свечи с микроструктурными фичами.
Собирает глубину стакана на уровнях 10k/50k/100k/250k/500k USD.

Использование:
    python -m data_fetcher stream
"""

import asyncio
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import ccxt.pro as ccxt
import duckdb

from data_fetcher import config

SYMBOL = "SOL/USDT"
AGGREGATION_INTERVAL = 10
BOOK_SNAPSHOT_INTERVAL = 3

EXCHANGES = {
    "binance_spot": {"id": "binance", "type": "spot", "book_limit": 100},
    "binance_perp": {"id": "binance", "type": "swap", "book_limit": 100},
    "okx_spot": {"id": "okx", "type": "spot", "book_limit": 100},
    "okx_perp": {"id": "okx", "type": "swap", "book_limit": 100},
    "bitget_spot": {"id": "bitget", "type": "spot", "book_limit": 100},
    "bitget_perp": {"id": "bitget", "type": "swap", "book_limit": 100},
}


def _db_path():
    return Path(config.DATA_DIR) / "realtime.db"


def get_exchange(exchange_id, exchange_type):
    ex_cfg = {"enableRateLimit": True, "options": {"defaultType": exchange_type}}
    if exchange_id == "binance":
        return ccxt.binance(ex_cfg)
    elif exchange_id == "okx":
        return ccxt.okx(ex_cfg)
    elif exchange_id == "bitget":
        return ccxt.bitget(ex_cfg)
    raise ValueError(f"Unknown exchange: {exchange_id}")


def now_ms():
    return int(time.time() * 1000)


def format_size(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    elif b < 1024 ** 3:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024**3:.2f} GB"


class TradeAggregator:
    """Агрегирует сырые трейды в N-секундные свечи с микрострутурными фичами."""

    def __init__(self, source):
        self.source = source
        self.buckets = defaultdict(list)

    def _bucket_key(self, ts_ms):
        return (ts_ms // (AGGREGATION_INTERVAL * 1000)) * AGGREGATION_INTERVAL * 1000

    def add_trade(self, trade):
        self.buckets[self._bucket_key(trade["timestamp"])].append(trade)

    def flush_ready(self, before_ms):
        ready_keys = [k for k in self.buckets if k + AGGREGATION_INTERVAL * 1000 <= before_ms]
        candles = []
        for key in sorted(ready_keys):
            trades = self.buckets.pop(key)
            candles.append(self._aggregate(key, trades))
        return candles

    def _aggregate(self, bucket_ts, trades):
        prices = [t["price"] for t in trades]
        amounts = [t["amount"] for t in trades]
        sides = [t.get("side", "buy") for t in trades]
        usd_volumes = [t["price"] * t["amount"] for t in trades]

        buy_vol = sum(v for v, s in zip(usd_volumes, sides) if s == "buy")
        sell_vol = sum(v for v, s in zip(usd_volumes, sides) if s == "sell")
        total_vol = buy_vol + sell_vol

        large_count = sum(1 for v in usd_volumes if v >= 10000)
        vwap = sum(p * a for p, a in zip(prices, amounts)) / sum(amounts) if amounts else 0

        sorted_trades = sorted(trades, key=lambda t: t["timestamp"])
        inter_trade_times = []
        for i in range(1, len(sorted_trades)):
            inter_trade_times.append(sorted_trades[i]["timestamp"] - sorted_trades[i - 1]["timestamp"])

        return {
            "ts": bucket_ts,
            "open": prices[0] if prices else 0,
            "high": max(prices) if prices else 0,
            "low": min(prices) if prices else 0,
            "close": prices[-1] if prices else 0,
            "volume_usd": total_vol,
            "vwap": vwap,
            "buy_ratio": buy_vol / total_vol if total_vol > 0 else 0.5,
            "trade_count": len(trades),
            "mean_trade_size_usd": sum(usd_volumes) / len(usd_volumes) if usd_volumes else 0,
            "large_trade_count": large_count,
            "mean_inter_trade_ms": sum(inter_trade_times) / len(inter_trade_times) if inter_trade_times else 0,
        }


class OrderBookCollector:
    """Собирает снапшоты стакана на фиксированных уровнях USD-глубины."""

    DEPTH_LEVELS_USD = [10_000, 50_000, 100_000, 250_000, 500_000]

    def __init__(self):
        self.last_snapshot_ms = {}
        self.snapshots = defaultdict(list)

    def _should_snapshot(self, source, ts_ms):
        last = self.last_snapshot_ms.get(source, 0)
        if ts_ms - last >= BOOK_SNAPSHOT_INTERVAL * 1000:
            self.last_snapshot_ms[source] = ts_ms
            return True
        return False

    def add_book(self, source, orderbook, ts_ms):
        if not self._should_snapshot(source, ts_ms):
            return

        bids = orderbook.get("bids", [])[:5]
        asks = orderbook.get("asks", [])[:5]
        if not bids or not asks:
            return

        mid = (bids[0][0] + asks[0][0]) / 2
        spread = asks[0][0] - bids[0][0]
        bid_depth = self._calc_depth(bids)
        ask_depth = self._calc_depth(asks)
        total_bid = sum(v for v in bid_depth.values() if v is not None) or 0
        total_ask = sum(v for v in ask_depth.values() if v is not None) or 0

        self.snapshots[source].append({
            "ts": ts_ms,
            "mid": mid,
            "spread": spread,
            "bid_depth_usd": bid_depth,
            "ask_depth_usd": ask_depth,
            "imbalance": total_bid / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 0.5,
        })

    def _calc_depth(self, levels):
        result = {}
        cum_usd = 0.0
        for lvl in levels:
            price, qty = lvl[0], lvl[1]
            cum_usd += price * qty
            for depth_usd in self.DEPTH_LEVELS_USD:
                key = f"{depth_usd // 1000}k"
                if key not in result and cum_usd >= depth_usd:
                    result[key] = round(cum_usd, 2)
        for depth_usd in self.DEPTH_LEVELS_USD:
            key = f"{depth_usd // 1000}k"
            if key not in result:
                result[key] = None
        return result

    def flush(self, source):
        return self.snapshots.pop(source, [])


def init_db():
    db = _db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db))

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ticks_raw (
            source VARCHAR NOT NULL,
            trade_id VARCHAR NOT NULL,
            timestamp BIGINT NOT NULL,
            received_at BIGINT,
            price DOUBLE,
            amount DOUBLE,
            side VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles_10s (
            source VARCHAR NOT NULL,
            ts BIGINT NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume_usd DOUBLE, vwap DOUBLE, buy_ratio DOUBLE,
            trade_count INTEGER, mean_trade_size_usd DOUBLE,
            large_trade_count INTEGER, mean_inter_trade_ms DOUBLE,
            PRIMARY KEY (source, ts)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS book_snapshots (
            source VARCHAR NOT NULL,
            ts BIGINT NOT NULL,
            mid DOUBLE, spread DOUBLE, imbalance DOUBLE,
            bid_10k DOUBLE, bid_50k DOUBLE, bid_100k DOUBLE, bid_250k DOUBLE, bid_500k DOUBLE,
            ask_10k DOUBLE, ask_50k DOUBLE, ask_100k DOUBLE, ask_250k DOUBLE, ask_500k DOUBLE,
            PRIMARY KEY (source, ts)
        )
    """)
    return conn


def write_ticks_db(conn, ticks):
    if not ticks:
        return
    conn.executemany(
        "INSERT INTO ticks_raw VALUES (?,?,?,?,?,?,?)",
        [
            (t["exchange"], t.get("trade_id", ""), t["timestamp"],
             t.get("received_at"), t["price"], t["amount"], t.get("side", "unknown"))
            for t in ticks
        ],
    )


def write_candles_db(conn, candles, source):
    if not candles:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO candles_10s VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (source, c["ts"], c["open"], c["high"], c["low"], c["close"],
             c["volume_usd"], c["vwap"], c["buy_ratio"], c["trade_count"],
             c["mean_trade_size_usd"], c["large_trade_count"], c["mean_inter_trade_ms"])
            for c in candles
        ],
    )


def write_book_db(conn, snaps, source):
    if not snaps:
        return
    rows = []
    for s in snaps:
        bid = s["bid_depth_usd"]
        ask = s["ask_depth_usd"]
        rows.append((
            source, s["ts"], s["mid"], s["spread"], s["imbalance"],
            bid.get("10k", 0), bid.get("50k", 0), bid.get("100k", 0), bid.get("250k", 0), bid.get("500k", 0),
            ask.get("10k", 0), ask.get("50k", 0), ask.get("100k", 0), ask.get("250k", 0), ask.get("500k", 0),
        ))
    conn.executemany("INSERT OR REPLACE INTO book_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)


def export_to_parquet(conn):
    """Экспортировать все таблицы realtime.db в Parquet."""
    for table in ["ticks_raw", "candles_10s", "book_snapshots"]:
        try:
            df = conn.execute(f"SELECT * FROM {table}").fetchdf()
            if not df.empty:
                pq = Path(config.DATA_DIR) / f"{table}.parquet"
                df.to_parquet(pq, index=False)
        except Exception:
            pass


class Pipeline:
    def __init__(self):
        self.exchanges = {}
        self.aggregators = {}
        self.book_collectors = {}
        self.tick_buffer = defaultdict(list)
        self.db = init_db()
        self.running = True
        self.start_time = time.time()
        self.total_ticks = defaultdict(int)
        self.total_candles = defaultdict(int)
        self.total_books = defaultdict(int)
        self.flush_count = 0

    async def start(self):
        db_path = _db_path()
        print("=" * 60, flush=True)
        print("  TICK DATA PIPELINE", flush=True)
        print(f"  Symbol: {SYMBOL}", flush=True)
        print(f"  Aggregation: {AGGREGATION_INTERVAL}s candles", flush=True)
        print(f"  Book: every {BOOK_SNAPSHOT_INTERVAL}s", flush=True)
        print(f"  Storage: DuckDB ({db_path})", flush=True)
        print("=" * 60, flush=True)

        for name, cfg in EXCHANGES.items():
            try:
                ex = get_exchange(cfg["id"], cfg["type"])
                self.exchanges[name] = ex
                self.aggregators[name] = TradeAggregator(name)
                self.book_collectors[name] = OrderBookCollector()
                print(f"  + {name}", flush=True)
            except Exception as e:
                print(f"  ! {name}: {e}", flush=True)

        tasks = []
        for name in self.exchanges:
            tasks.append(asyncio.create_task(self._watch_trades(name)))
            tasks.append(asyncio.create_task(self._watch_orderbook(name)))
        tasks.append(asyncio.create_task(self._periodic_flush()))

        print(f"\n  Listening on {len(self.exchanges)} streams...\n", flush=True)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _watch_trades(self, source):
        ex = self.exchanges[source]
        while self.running:
            try:
                trades = await ex.watch_trades(SYMBOL)
                for t in trades:
                    trade = {
                        "trade_id": str(t["id"]) if t.get("id") else f"{t['timestamp']}_{uuid.uuid4().hex[:8]}",
                        "timestamp": t["timestamp"],
                        "received_at": now_ms(),
                        "price": t["price"],
                        "amount": t["amount"],
                        "side": t.get("side", "unknown"),
                        "exchange": source,
                    }
                    self.aggregators[source].add_trade(trade)
                    self.tick_buffer[source].append(trade)
            except ccxt.NetworkError as e:
                print(f"  ~ {source}: network error, retry in 5s...", flush=True)
                await asyncio.sleep(5)
            except ccxt.ExchangeError as e:
                print(f"  ! {source}: exchange error - {e}", flush=True)
                await asyncio.sleep(10)
            except Exception as e:
                print(f"  ! {source}: {e}", flush=True)
                await asyncio.sleep(5)

    async def _watch_orderbook(self, source):
        ex = self.exchanges[source]
        limit = EXCHANGES[source]["book_limit"]
        while self.running:
            try:
                book = await ex.watch_order_book(SYMBOL, limit=limit)
                ts = book.get("timestamp") or now_ms()
                self.book_collectors[source].add_book(source, book, ts)
            except ccxt.NetworkError as e:
                print(f"  ~ {source} book: network error, retry in 5s...", flush=True)
                await asyncio.sleep(5)
            except Exception as e:
                print(f"  ! {source} book: {e}", flush=True)
                await asyncio.sleep(5)

    async def _periodic_flush(self):
        while self.running:
            await asyncio.sleep(AGGREGATION_INTERVAL)
            ts = now_ms()
            self.flush_count += 1

            for source in list(self.aggregators.keys()):
                candles = self.aggregators[source].flush_ready(ts)
                if candles:
                    await asyncio.to_thread(write_candles_db, self.db, candles, source)
                    self.total_candles[source] += len(candles)

                ticks = self.tick_buffer.pop(source, [])
                if ticks:
                    await asyncio.to_thread(write_ticks_db, self.db, ticks)
                    self.total_ticks[source] += len(ticks)

                snaps = self.book_collectors[source].flush(source)
                if snaps:
                    await asyncio.to_thread(write_book_db, self.db, snaps, source)
                    self.total_books[source] += len(snaps)

            if self.flush_count % 6 == 0:
                self._print_status()
                export_to_parquet(self.db)

    def _print_status(self):
        uptime_s = time.time() - self.start_time
        h, m, s = int(uptime_s // 3600), int((uptime_s % 3600) // 60), int(uptime_s % 60)
        db_path = _db_path()
        db_size = format_size(db_path.stat().st_size) if db_path.exists() else "0 B"
        total_t = sum(self.total_ticks.values())
        total_c = sum(self.total_candles.values())
        total_b = sum(self.total_books.values())
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")

        print(f"\n--- [{now_str}] uptime {h}h{m}m{s}s | DB: {db_size} ---", flush=True)
        print(f"  {'Source':<16} {'Ticks':>8} {'Candles':>8} {'Books':>8}", flush=True)
        print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8}", flush=True)
        for src in EXCHANGES:
            print(f"  {src:<16} {self.total_ticks[src]:>8} {self.total_candles[src]:>8} {self.total_books[src]:>8}", flush=True)
        print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8}", flush=True)
        print(f"  {'TOTAL':<16} {total_t:>8} {total_c:>8} {total_b:>8}", flush=True)

    def _print_final_report(self):
        uptime_s = time.time() - self.start_time
        h, m, s = int(uptime_s // 3600), int((uptime_s % 3600) // 60), int(uptime_s % 60)
        db_path = _db_path()
        db_size = format_size(db_path.stat().st_size) if db_path.exists() else "0 B"
        total_t = sum(self.total_ticks.values())
        total_c = sum(self.total_candles.values())
        total_b = sum(self.total_books.values())

        print(f"\n{'='*50}", flush=True)
        print(f"  FINAL REPORT", flush=True)
        print(f"{'='*50}", flush=True)
        print(f"  Uptime:    {h}h {m}m {s}s", flush=True)
        print(f"  Ticks:     {total_t:,}", flush=True)
        print(f"  Candles:   {total_c:,}", flush=True)
        print(f"  Books:     {total_b:,}", flush=True)
        print(f"  DuckDB:    {db_size}", flush=True)
        print(f"  Path:      {db_path}", flush=True)
        print(f"{'='*50}", flush=True)


async def main():
    pipeline = Pipeline()
    try:
        await pipeline.start()
    except KeyboardInterrupt:
        pipeline.running = False
        print("\n\n  Stopping...", flush=True)
        for ex in pipeline.exchanges.values():
            await ex.close()

        ts = now_ms()
        for source in list(pipeline.aggregators.keys()):
            candles = pipeline.aggregators[source].flush_ready(ts + AGGREGATION_INTERVAL * 1000)
            if candles:
                await asyncio.to_thread(write_candles_db, pipeline.db, candles, source)
                pipeline.total_candles[source] += len(candles)
            ticks = pipeline.tick_buffer.pop(source, [])
            if ticks:
                await asyncio.to_thread(write_ticks_db, pipeline.db, ticks)
                pipeline.total_ticks[source] += len(ticks)
            snaps = pipeline.book_collectors[source].flush(source)
            if snaps:
                await asyncio.to_thread(write_book_db, pipeline.db, snaps, source)
                pipeline.total_books[source] += len(snaps)

        export_to_parquet(pipeline.db)
        pipeline._print_final_report()
        pipeline.db.close()
        print("  Done", flush=True)


if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    asyncio.run(main())
