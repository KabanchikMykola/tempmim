"""Volatility Regime v2 — только High Vol + Mean Reversion + Basis фильтр."""

import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path("data/top5_2026")
TF = "1h"

EXCLUDE = {"USDC"}  # stablecoins


def load_pair(base: str) -> pd.DataFrame | None:
    spot_file = DATA_DIR / f"{base}_USDT_{TF}.parquet"
    perp_file = DATA_DIR / f"{base}_USDT_USDT_{TF}.parquet"
    if not spot_file.exists() or not perp_file.exists():
        return None
    spot = pd.read_parquet(spot_file)[["timestamp", "close"]].rename(columns={"close": "spot"})
    perp = pd.read_parquet(perp_file)[["timestamp", "close"]].rename(columns={"close": "perp"})
    df = spot.merge(perp, on="timestamp", how="inner")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["returns"] = df["spot"].pct_change()
    df["basis_pct"] = ((df["perp"] - df["spot"]) / df["spot"]) * 100
    return df


def calc_vol(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    df = df.copy()
    df["vol"] = df["returns"].rolling(window).std() * np.sqrt(8760)
    df["vol_rank"] = df["vol"].rolling(168).rank(pct=True)
    return df


def calc_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["spot"].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def calc_basis_z(df: pd.DataFrame, window: int = 168) -> pd.DataFrame:
    """Z-score basis: насколько текущий спред экстремален."""
    df = df.copy()
    df["basis_mean"] = df["basis_pct"].rolling(window).mean()
    df["basis_std"] = df["basis_pct"].rolling(window).std()
    df["basis_z"] = (df["basis_pct"] - df["basis_mean"]) / df["basis_std"]
    return df


def backtest_v2(
    df: pd.DataFrame,
    vol_threshold: float = 0.75,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    basis_filter: bool = True,
    stop_loss_pct: float = 3.0,
    take_profit_pct: float = 5.0,
) -> dict:
    """
    Mean Reversion ТОЛЬКО в High Vol с Basis фильтром.

    LONG: high vol + RSI < oversold + basis_z < -1 (перп дешевле — паника)
    SHORT: high vol + RSI > overbought + basis_z > 1 (перп дороже — эйфория)
    """
    df = df.dropna(subset=["vol_rank", "rsi", "basis_z"]).copy()
    if len(df) < 100:
        return {}

    equity = 10000
    equity_curve = [equity]
    trades = []
    position = 0
    entry_price = 0
    entry_idx = 0

    for i in range(1, len(df)):
        vol_rank = df["vol_rank"].iloc[i]
        rsi = df["rsi"].iloc[i]
        basis_z = df["basis_z"].iloc[i]
        price = df["spot"].iloc[i]
        high_vol = vol_rank > vol_threshold

        if position == 0:
            if high_vol and rsi < rsi_oversold:
                if not basis_filter or basis_z < -1:
                    position = 1
                    entry_price = price
                    entry_idx = i
            elif high_vol and rsi > rsi_overbought:
                if not basis_filter or basis_z > 1:
                    position = -1
                    entry_price = price
                    entry_idx = i

        elif position == 1:
            pnl_pct = (price / entry_price - 1) * 100
            exit_cond = (
                rsi > 50 or
                pnl_pct <= -stop_loss_pct or
                pnl_pct >= take_profit_pct or
                i - entry_idx > 48
            )
            if exit_cond:
                equity *= (1 + pnl_pct / 100 * 0.998)
                trades.append({"side": "long", "pnl": round(pnl_pct, 4), "bars": i - entry_idx, "exit": "rsi" if rsi > 50 else "sl" if pnl_pct <= -stop_loss_pct else "tp" if pnl_pct >= take_profit_pct else "time"})
                position = 0

        elif position == -1:
            pnl_pct = (1 - price / entry_price) * 100
            exit_cond = (
                rsi < 50 or
                pnl_pct <= -stop_loss_pct or
                pnl_pct >= take_profit_pct or
                i - entry_idx > 48
            )
            if exit_cond:
                equity *= (1 + pnl_pct / 100 * 0.998)
                trades.append({"side": "short", "pnl": round(pnl_pct, 4), "bars": i - entry_idx, "exit": "rsi" if rsi < 50 else "sl" if pnl_pct <= -stop_loss_pct else "tp" if pnl_pct >= take_profit_pct else "time"})
                position = 0

        equity_curve.append(equity)

    if not trades:
        return {"trades": 0}

    wins = sum(1 for t in trades if t["pnl"] > 0)
    peak = max(equity_curve)
    dd = min((e - peak) / peak * 100 for e in equity_curve)

    exits = {}
    for t in trades:
        e = t["exit"]
        exits.setdefault(e, {"count": 0, "wins": 0, "pnl": 0})
        exits[e]["count"] += 1
        if t["pnl"] > 0:
            exits[e]["wins"] += 1
        exits[e]["pnl"] += t["pnl"]

    return {
        "trades": len(trades),
        "win_rate": round(wins / len(trades) * 100, 1),
        "total_pnl": round(equity - 10000, 2),
        "max_drawdown": round(dd, 2),
        "final_equity": round(equity, 2),
        "avg_pnl": round(np.mean([t["pnl"] for t in trades]), 4),
        "exits": {k: {"count": v["count"], "win_rate": round(v["wins"]/v["count"]*100, 1) if v["count"] > 0 else 0} for k, v in exits.items()},
    }


def main():
    files = list(DATA_DIR.glob(f"*_USDT_{TF}.parquet"))
    bases = sorted(set(f.stem.replace(f"_{TF}", "").replace("_USDT", "") for f in files) - EXCLUDE)

    print("VOLATILITY REGIME v2 — High Vol Mean Reversion + Basis Filter")
    print("=" * 95)

    all_data = {}
    for base in bases:
        df = load_pair(base)
        if df is None or df.empty:
            continue
        df = calc_vol(df)
        df = calc_rsi(df)
        df = calc_basis_z(df)
        all_data[base] = df

    configs = [
        {"label": "v1 (no filter, no SL)", "basis_filter": False, "stop_loss_pct": 999, "take_profit_pct": 999},
        {"label": "v2 (basis + SL 3% + TP 5%)", "basis_filter": True, "stop_loss_pct": 3.0, "take_profit_pct": 5.0},
        {"label": "v2 (basis + SL 2% + TP 3%)", "basis_filter": True, "stop_loss_pct": 2.0, "take_profit_pct": 3.0},
    ]

    for cfg in configs:
        print(f"\n--- {cfg['label']} ---")
        print(f"{'Symbol':>8} {'Trades':>7} {'Win%':>6} {'PnL$':>9} {'Avg%':>7} {'MaxDD%':>8} {'Exits':>30}")

        for base in bases:
            if base not in all_data:
                continue
            r = backtest_v2(
                all_data[base],
                basis_filter=cfg["basis_filter"],
                stop_loss_pct=cfg["stop_loss_pct"],
                take_profit_pct=cfg["take_profit_pct"],
            )
            if not r or r["trades"] == 0:
                continue
            exits_str = " ".join(f"{k}:{v['count']}({v['win_rate']}%)" for k, v in r["exits"].items())
            print(f"{base:>8} {r['trades']:>7} {r['win_rate']:>6.1f} {r['total_pnl']:>9.2f} {r['avg_pnl']:>7.4f} {r['max_drawdown']:>8.2f} {exits_str:>30}")

    print(f"\n{'=' * 95}")
    print("CURRENT SIGNALS (high vol + RSI extremes + basis filter):")
    for base in bases:
        if base not in all_data:
            continue
        df = all_data[base]
        if len(df) < 100:
            continue
        last = df.iloc[-1]
        vol_rank = last["vol_rank"]
        rsi = last["rsi"]
        basis_z = last["basis_z"]
        if pd.isna(vol_rank) or pd.isna(rsi) or pd.isna(basis_z):
            continue
        if vol_rank > 0.75:
            if rsi < 30 and basis_z < -1:
                print(f"  {base:>8}: LONG | vol_rank={vol_rank:.2f} rsi={rsi:.1f} basis_z={basis_z:.2f}")
            elif rsi > 70 and basis_z > 1:
                print(f"  {base:>8}: SHORT | vol_rank={vol_rank:.2f} rsi={rsi:.1f} basis_z={basis_z:.2f}")


if __name__ == "__main__":
    main()
