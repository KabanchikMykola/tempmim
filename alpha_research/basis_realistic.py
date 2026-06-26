"""Basis стратегия с реальными издержками."""

import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path("data")
TF = "1h"

# Реальные издержки Binance
COMMISSION_PCT = 0.04        # 0.04% maker/taker (спот + перп = 0.08%Round trip)
SLIPPAGE_BPS = 2             # 2 bps slippage на каждый вход/выход
FUNDING_RATE_AVG = 0.0001    # ~0.01% каждые 8h (средний)


def load_pair(base: str) -> pd.DataFrame | None:
    spot_file = DATA_DIR / f"{base}USDT_{TF}_spot.parquet"
    perp_file = DATA_DIR / f"{base}USDT_{TF}_perp.parquet"
    if not spot_file.exists() or not perp_file.exists():
        return None
    spot = pd.read_parquet(spot_file)[["timestamp", "close"]].rename(columns={"close": "spot"})
    perp = pd.read_parquet(perp_file)[["timestamp", "close"]].rename(columns={"close": "perp"})
    df = spot.merge(perp, on="timestamp", how="inner")
    df["basis_pct"] = ((df["perp"] - df["spot"]) / df["spot"]) * 100
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def backtest_realistic(df: pd.DataFrame, entry_z: float = 2.0, exit_z: float = 0.5) -> dict:
    """Бэктест с реальными издержками."""
    basis = df["basis_pct"]
    mean = basis.mean()
    std = basis.std()
    if std == 0:
        return {}

    zscore = (basis - mean) / std

    position = 0
    entry_price = 0
    entry_idx = 0
    equity = 10000
    equity_curve = [equity]
    trades = []

    for i in range(1, len(zscore)):
        z = zscore.iloc[i]

        if position == 0:
            if z > entry_z:
                position = -1
                entry_price = basis.iloc[i]
                entry_idx = i
                # Комиссия на вход (спот + перп)
                equity *= (1 - COMMISSION_PCT * 2 / 100)
                equity *= (1 - SLIPPAGE_BPS / 10000)
            elif z < -entry_z:
                position = 1
                entry_price = basis.iloc[i]
                entry_idx = i
                equity *= (1 - COMMISSION_PCT * 2 / 100)
                equity *= (1 - SLIPPAGE_BPS / 10000)

        elif position == 1 and z > -exit_z:
            # Funding rate: 8h = 3 бара. Платим funding пока держим позицию
            bars_held = i - entry_idx
            funding_cost = FUNDING_RATE_AVG * (bars_held / 3) * 100  # в %
            pnl_gross = basis.iloc[i] - entry_price
            pnl_net = pnl_gross - funding_cost
            pnl_dollar = pnl_net * equity / 100
            # Комиссия на выход
            equity += pnl_dollar
            equity *= (1 - COMMISSION_PCT * 2 / 100)
            equity *= (1 - SLIPPAGE_BPS / 10000)
            trades.append({
                "side": "long", "bars": bars_held,
                "pnl_gross": round(pnl_gross, 4),
                "pnl_net": round(pnl_net, 4),
                "pnl_dollar": round(pnl_dollar, 2),
            })
            position = 0

        elif position == -1 and z < exit_z:
            bars_held = i - entry_idx
            funding_cost = FUNDING_RATE_AVG * (bars_held / 3) * 100
            pnl_gross = entry_price - basis.iloc[i]
            pnl_net = pnl_gross - funding_cost
            pnl_dollar = pnl_net * equity / 100
            equity += pnl_dollar
            equity *= (1 - COMMISSION_PCT * 2 / 100)
            equity *= (1 - SLIPPAGE_BPS / 10000)
            trades.append({
                "side": "short", "bars": bars_held,
                "pnl_gross": round(pnl_gross, 4),
                "pnl_net": round(pnl_net, 4),
                "pnl_dollar": round(pnl_dollar, 2),
            })
            position = 0

        equity_curve.append(equity)

    # Открытые позиции считаем как unrealized loss
    if position != 0:
        unrealized = basis.iloc[-1] - entry_price if position == 1 else entry_price - basis.iloc[-1]
        trades.append({
            "side": f"{'long' if position == 1 else 'short'}_OPEN",
            "bars": len(zscore) - entry_idx,
            "pnl_gross": round(unrealized, 4),
            "pnl_net": round(unrealized, 4),
            "pnl_dollar": 0,
        })

    if not trades:
        return {"trades": 0}

    closed = [t for t in trades if "_OPEN" not in t["side"]]
    wins = sum(1 for t in closed if t["pnl_net"] > 0)
    peak = max(equity_curve)
    dd = min((e - peak) / peak * 100 for e in equity_curve)

    return {
        "trades": len(closed),
        "open": len(trades) - len(closed),
        "win_rate": round(wins / len(closed) * 100, 1) if closed else 0,
        "total_pnl_gross": round(sum(t["pnl_gross"] for t in closed), 2),
        "total_pnl_net": round(sum(t["pnl_net"] for t in closed), 2),
        "total_cost": round(sum(t["pnl_gross"] - t["pnl_net"] for t in closed), 2),
        "max_drawdown": round(dd, 2),
        "final_equity": round(equity, 2),
        "avg_bars": round(np.mean([t["bars"] for t in closed]), 1),
    }


def main():
    files = list(DATA_DIR.glob(f"*_{TF}_spot.parquet"))
    bases = sorted(set(f.stem.replace(f"_{TF}_spot", "") for f in files))

    print("BASIS STRATEGY — REALISTIC COSTS")
    print(f"Комиссия: {COMMISSION_PCT}% × 2 (вход+выход) × 2 (спот+перп)")
    print(f"Slippage: {SLIPPAGE_BPS} bps")
    print(f"Funding: ~{FUNDING_RATE_AVG*100:.2f}% каждые 8h")
    print("=" * 90)

    for entry_z in [1.5, 2.0, 2.5]:
        print(f"\n--- entry_z={entry_z}, exit_z=0.5 ---")
        print(f"{'Symbol':>8} {'Trades':>7} {'Open':>5} {'Win%':>6} {'Gross(bps)':>11} {'Net(bps)':>9} {'Cost(bps)':>10} {'MaxDD%':>7} {'Equity$':>9} {'AvgBars':>8}")

        for base in bases:
            df = load_pair(base)
            if df is None or df.empty:
                continue
            r = backtest_realistic(df, entry_z=entry_z)
            if not r or r["trades"] == 0:
                continue
            print(f"{base:>8} {r['trades']:>7} {r['open']:>5} {r['win_rate']:>6.1f} {r['total_pnl_gross']:>11.2f} {r['total_pnl_net']:>9.2f} {r['total_cost']:>10.2f} {r['max_drawdown']:>7.2f} {r['final_equity']:>9.2f} {r['avg_bars']:>8.1f}")

    # Текущие экстремальные сигналы
    print(f"\n{'=' * 90}")
    print("EXTREME SIGNALS (|z| > 2):")
    for base in bases:
        df = load_pair(base)
        if df is None or len(df) < 100:
            continue
        mean = df["basis_pct"].mean()
        std = df["basis_pct"].std()
        if std == 0:
            continue
        z = (df["basis_pct"].iloc[-1] - mean) / std
        if abs(z) > 2:
            direction = "LONG (перп дешевле)" if z < 0 else "SHORT (перп дороже)"
            print(f"  {base:>8}: z={z:>6.2f} | basis={df['basis_pct'].iloc[-1]:.4f}% | → {direction}")


if __name__ == "__main__":
    main()
