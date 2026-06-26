"""Исследование Spot-Perp Basis — спред между спот и перп ценой."""

import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path("data")
TF = "1h"


def load_pair(base: str) -> pd.DataFrame | None:
    """Загрузить спот и перп для пары, вернуть merged DataFrame."""
    spot_file = DATA_DIR / f"{base}USDT_{TF}_spot.parquet"
    perp_file = DATA_DIR / f"{base}USDT_{TF}_perp.parquet"

    if not spot_file.exists() or not perp_file.exists():
        return None

    spot = pd.read_parquet(spot_file)[["timestamp", "close", "volume"]].rename(
        columns={"close": "spot_close", "volume": "spot_volume"}
    )
    perp = pd.read_parquet(perp_file)[["timestamp", "close", "volume"]].rename(
        columns={"close": "perp_close", "volume": "perp_volume"}
    )

    merged = spot.merge(perp, on="timestamp", how="inner")
    merged["basis"] = merged["perp_close"] - merged["spot_close"]
    merged["basis_pct"] = (merged["basis"] / merged["spot_close"]) * 100
    merged["datetime"] = pd.to_datetime(merged["timestamp"], unit="ms", utc=True)
    return merged


def analyze_basis(df: pd.DataFrame, base: str) -> dict:
    """Анализ базиса для одной пары."""
    basis = df["basis_pct"].dropna()

    return {
        "symbol": base,
        "bars": len(df),
        "mean": round(basis.mean(), 4),
        "std": round(basis.std(), 4),
        "min": round(basis.min(), 4),
        "max": round(basis.max(), 4),
        "median": round(basis.median(), 4),
        "skew": round(basis.skew(), 4),
        "positive_pct": round((basis > 0).mean() * 100, 1),
        "zscore_current": round((basis.iloc[-1] - basis.mean()) / basis.std(), 2) if basis.std() > 0 else 0,
    }


def backtest_basis(df: pd.DataFrame, entry_z: float = 2.0, exit_z: float = 0.5) -> dict:
    """Простой бэктест: вход при z-score > entry, выход при < exit."""
    basis = df["basis_pct"]
    zscore = (basis - basis.mean()) / basis.std()

    position = 0
    entry_price = 0
    trades = []
    equity = 10000
    equity_curve = [equity]

    for i in range(1, len(zscore)):
        z = zscore.iloc[i]

        if position == 0:
            if z > entry_z:
                position = -1
                entry_price = basis.iloc[i]
            elif z < -entry_z:
                position = 1
                entry_price = basis.iloc[i]
        elif position == 1 and z > -exit_z:
            pnl = (basis.iloc[i] - entry_price) * equity / 100
            equity += pnl
            trades.append({"side": "long", "entry": entry_price, "exit": basis.iloc[i], "pnl": round(pnl, 2)})
            position = 0
        elif position == -1 and z < exit_z:
            pnl = (entry_price - basis.iloc[i]) * equity / 100
            equity += pnl
            trades.append({"side": "short", "entry": entry_price, "exit": basis.iloc[i], "pnl": round(pnl, 2)})
            position = 0

        equity_curve.append(equity)

    if not trades:
        return {"trades": 0, "win_rate": 0, "total_pnl": 0, "max_drawdown": 0}

    wins = sum(1 for t in trades if t["pnl"] > 0)
    peak = max(equity_curve)
    dd = min((e - peak) / peak * 100 for e in equity_curve)

    return {
        "trades": len(trades),
        "win_rate": round(wins / len(trades) * 100, 1),
        "total_pnl": round(equity - 10000, 2),
        "max_drawdown": round(dd, 2),
        "final_equity": round(equity, 2),
    }


def main():
    files = list(DATA_DIR.glob(f"*_{TF}_spot.parquet"))
    bases = sorted(set(f.stem.replace(f"_{TF}_spot", "") for f in files))

    print(f"Basis Research | {len(bases)} пар | {TF}")
    print("=" * 80)

    all_stats = []
    all_bt = []

    for base in bases:
        df = load_pair(base)
        if df is None or df.empty:
            continue

        stats = analyze_basis(df, base)
        bt = backtest_basis(df)
        bt["symbol"] = base

        all_stats.append(stats)
        all_bt.append(bt)

    print("\n1) BASIS STATISTICS (% premium/discount)")
    print("-" * 80)
    print(f"{'Symbol':>8} {'Mean%':>7} {'Std%':>7} {'Min%':>8} {'Max%':>8} {'Pos%':>6} {'Z(last)':>8}")
    for s in sorted(all_stats, key=lambda x: abs(x["mean"]), reverse=True):
        print(f"{s['symbol']:>8} {s['mean']:>7.4f} {s['std']:>7.4f} {s['min']:>8.4f} {s['max']:>8.4f} {s['positive_pct']:>6.1f} {s['zscore_current']:>8.2f}")

    print("\n2) BACKTEST (z-entry=2.0, z-exit=0.5)")
    print("-" * 80)
    print(f"{'Symbol':>8} {'Trades':>7} {'Win%':>6} {'PnL$':>10} {'MaxDD%':>8} {'Equity$':>10}")
    for b in sorted(all_bt, key=lambda x: x["total_pnl"], reverse=True):
        print(f"{b['symbol']:>8} {b['trades']:>7} {b['win_rate']:>6.1f} {b['total_pnl']:>10.2f} {b['max_drawdown']:>8.2f} {b['final_equity']:>10.2f}")

    winners = [b for b in all_bt if b["total_pnl"] > 0]
    print(f"\nИтого: {len(winners)}/{len(all_bt)} прибыльных пар")


if __name__ == "__main__":
    main()
