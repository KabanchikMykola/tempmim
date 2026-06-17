"""Walk-forward проверка Basis стратегии."""

import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path("data/top5_2026")
TF = "1h"


def load_pair(base: str) -> pd.DataFrame | None:
    spot_file = DATA_DIR / f"{base}_USDT_{TF}.parquet"
    perp_file = DATA_DIR / f"{base}_USDT_USDT_{TF}.parquet"
    if not spot_file.exists() or not perp_file.exists():
        return None
    spot = pd.read_parquet(spot_file)[["timestamp", "close"]].rename(columns={"close": "spot"})
    perp = pd.read_parquet(perp_file)[["timestamp", "close"]].rename(columns={"close": "perp"})
    df = spot.merge(perp, on="timestamp", how="inner")
    df["basis_pct"] = ((df["perp"] - df["spot"]) / df["spot"]) * 100
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def walk_forward(df: pd.DataFrame, train_days: int = 30, test_days: int = 7,
                 entry_z: float = 2.0, exit_z: float = 0.5) -> pd.DataFrame:
    """Walk-forward: train на train_days, trade на test_days."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    train_ms = train_days * 24 * 3600000
    test_ms = test_days * 24 * 3600000

    results = []
    start = df["timestamp"].iloc[0]
    end = df["timestamp"].iloc[-1]

    cursor = start
    while cursor + train_ms + test_ms <= end:
        train_start = cursor
        train_end = cursor + train_ms
        test_start = train_end
        test_end = min(train_end + test_ms, end)

        train = df[(df["timestamp"] >= train_start) & (df["timestamp"] < train_end)]
        test = df[(df["timestamp"] >= test_start) & (df["timestamp"] < test_end)]

        if len(train) < 100 or len(test) < 10:
            cursor += test_ms
            continue

        mean = train["basis_pct"].mean()
        std = train["basis_pct"].std()
        if std == 0:
            cursor += test_ms
            continue

        position = 0
        entry_price = 0
        for _, row in test.iterrows():
            z = (row["basis_pct"] - mean) / std
            if position == 0:
                if z > entry_z:
                    position = -1
                    entry_price = row["basis_pct"]
                elif z < -entry_z:
                    position = 1
                    entry_price = row["basis_pct"]
            elif position == 1 and z > -exit_z:
                results.append({
                    "exit_date": row["datetime"],
                    "side": "long",
                    "entry": entry_price,
                    "exit": row["basis_pct"],
                    "pnl_bps": (row["basis_pct"] - entry_price) * 100,
                })
                position = 0
            elif position == -1 and z < exit_z:
                results.append({
                    "exit_date": row["datetime"],
                    "side": "short",
                    "entry": entry_price,
                    "exit": row["basis_pct"],
                    "pnl_bps": (entry_price - row["basis_pct"]) * 100,
                })
                position = 0

        cursor += test_ms

    return pd.DataFrame(results)


def parameter_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Перебор параметров entry_z и exit_z."""
    results = []
    for entry_z in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for exit_z in [0.0, 0.25, 0.5, 0.75, 1.0]:
            wf = walk_forward(df, entry_z=entry_z, exit_z=exit_z, train_days=30, test_days=7)
            if wf.empty:
                continue
            wins = (wf["pnl_bps"] > 0).sum()
            results.append({
                "entry_z": entry_z,
                "exit_z": exit_z,
                "trades": len(wf),
                "win_rate": round(wins / len(wf) * 100, 1),
                "total_pnl_bps": round(wf["pnl_bps"].sum(), 2),
                "avg_pnl_bps": round(wf["pnl_bps"].mean(), 2),
                "max_loss_bps": round(wf["pnl_bps"].min(), 2),
            })
    return pd.DataFrame(results)


def main():
    files = list(DATA_DIR.glob(f"*_USDT_{TF}.parquet"))
    bases = sorted(set(f.stem.replace(f"_{TF}", "").replace("_USDT", "") for f in files))

    print("WALK-FORWARD BASIS ANALYSIS")
    print("=" * 80)

    # 1. Walk-forward для всех пар
    print("\n1) WALK-FORWARD (30d train, 7d test, entry_z=2.0, exit_z=0.5)")
    print("-" * 80)
    print(f"{'Symbol':>8} {'Trades':>7} {'Win%':>6} {'PnL(bps)':>10} {'Avg(bps)':>9} {'MaxLoss':>9}")

    all_summary = []
    for base in bases:
        df = load_pair(base)
        if df is None or df.empty:
            continue
        wf = walk_forward(df)
        if wf.empty:
            print(f"{base:>8}  нет сделок")
            continue
        wins = (wf["pnl_bps"] > 0).sum()
        row = {
            "symbol": base,
            "trades": len(wf),
            "win_rate": round(wins / len(wf) * 100, 1),
            "total_pnl_bps": round(wf["pnl_bps"].sum(), 2),
            "avg_pnl_bps": round(wf["pnl_bps"].mean(), 2),
            "max_loss_bps": round(wf["pnl_bps"].min(), 2),
        }
        all_summary.append(row)
        print(f"{base:>8} {row['trades']:>7} {row['win_rate']:>6.1f} {row['total_pnl_bps']:>10.2f} {row['avg_pnl_bps']:>9.2f} {row['max_loss_bps']:>9.2f}")

    profitable = [r for r in all_summary if r["total_pnl_bps"] > 0]
    print(f"\nПрибыльных: {len(profitable)}/{len(all_summary)}")

    # 2. Parameter sweep для лучшей пары
    if all_summary:
        best = max(all_summary, key=lambda x: x["total_pnl_bps"])
        print(f"\n2) PARAMETER SWEEP: {best['symbol']}")
        print("-" * 80)
        df_best = load_pair(best["symbol"])
        if df_best is not None:
            sweep = parameter_sweep(df_best)
            print(f"{'entry_z':>8} {'exit_z':>7} {'Trades':>7} {'Win%':>6} {'PnL(bps)':>10} {'Avg':>8} {'MaxLoss':>9}")
            for _, r in sweep.iterrows():
                print(f"{r['entry_z']:>8.1f} {r['exit_z']:>7.2f} {r['trades']:>7} {r['win_rate']:>6.1f} {r['total_pnl_bps']:>10.2f} {r['avg_pnl_bps']:>8.2f} {r['max_loss_bps']:>9.2f}")

    # 3. Текущие сигналы
    print(f"\n3) CURRENT SIGNALS (z-score)")
    print("-" * 80)
    print(f"{'Symbol':>8} {'Basis%':>8} {'Z-score':>8} {'Signal':>12}")
    for base in bases:
        df = load_pair(base)
        if df is None or len(df) < 100:
            continue
        mean = df["basis_pct"].mean()
        std = df["basis_pct"].std()
        if std == 0:
            continue
        current_basis = df["basis_pct"].iloc[-1]
        z = (current_basis - mean) / std
        if z > 2:
            signal = "SHORT"
        elif z < -2:
            signal = "LONG"
        else:
            signal = "NEUTRAL"
        print(f"{base:>8} {current_basis:>8.4f} {z:>8.2f} {signal:>12}")


if __name__ == "__main__":
    main()
