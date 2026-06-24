"""Volatility Regime Research — low vol → breakout, high vol → mean reversion."""

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
    spot = pd.read_parquet(spot_file)[["timestamp", "close", "high", "low"]].rename(
        columns={"close": "spot_close", "high": "spot_high", "low": "spot_low"}
    )
    perp = pd.read_parquet(perp_file)[["timestamp", "close"]].rename(columns={"close": "perp_close"})
    df = spot.merge(perp, on="timestamp", how="inner")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["returns"] = df["spot_close"].pct_change()
    df["basis_pct"] = ((df["perp_close"] - df["spot_close"]) / df["spot_close"]) * 100
    return df


def calc_volatility(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """Rolling volatility (annualized)."""
    df = df.copy()
    df["vol"] = df["returns"].rolling(window).std() * np.sqrt(8760)
    df["vol_pct"] = df["vol"] * 100
    df["vol_rank"] = df["vol"].rolling(168).rank(pct=True)
    return df


def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Классификация regimes по vol."""
    df = df.copy()
    df["regime"] = pd.cut(
        df["vol_rank"],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["low", "mid_low", "mid_high", "high"],
        include_lowest=True,
    )
    return df


def backtest_regime(df: pd.DataFrame, strategy: str = "both") -> dict:
    """
    Стратегия по regime:
    - Low vol: momentum (buy if up, sell if down)
    - High vol: mean reversion (buy if oversold, sell if overbought)
    """
    df = df.dropna(subset=["vol_rank", "regime"]).copy()
    if len(df) < 100:
        return {}

    rsi_window = 14
    delta = df["spot_close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_window).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_window).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["momentum"] = df["spot_close"].pct_change(6)

    equity = 10000
    equity_curve = [equity]
    trades = []
    position = 0
    entry_price = 0
    entry_idx = 0

    for i in range(max(rsi_window, 24) + 1, len(df)):
        regime = df["regime"].iloc[i]
        rsi = df["rsi"].iloc[i]
        mom = df["momentum"].iloc[i]
        price = df["spot_close"].iloc[i]

        if position == 0:
            if regime == "low" and strategy in ("momentum", "both"):
                if mom > 0.005:
                    position = 1
                    entry_price = price
                    entry_idx = i
                elif mom < -0.005:
                    position = -1
                    entry_price = price
                    entry_idx = i
            elif regime == "high" and strategy in ("reversion", "both"):
                if rsi < 30:
                    position = 1
                    entry_price = price
                    entry_idx = i
                elif rsi > 70:
                    position = -1
                    entry_price = price
                    entry_idx = i

        elif position == 1:
            exit_cond = False
            if regime == "low" and mom < -0.005:
                exit_cond = True
            elif regime == "high" and rsi > 60:
                exit_cond = True
            elif i - entry_idx > 48:
                exit_cond = True
            if exit_cond:
                pnl_pct = (price / entry_price - 1) * 100
                equity *= (1 + pnl_pct / 100 * 0.998)  # costs
                trades.append({"side": "long", "regime": regime, "pnl": round(pnl_pct, 4), "bars": i - entry_idx})
                position = 0

        elif position == -1:
            exit_cond = False
            if regime == "low" and mom > 0.005:
                exit_cond = True
            elif regime == "high" and rsi < 40:
                exit_cond = True
            elif i - entry_idx > 48:
                exit_cond = True
            if exit_cond:
                pnl_pct = (1 - price / entry_price) * 100
                equity *= (1 + pnl_pct / 100 * 0.998)
                trades.append({"side": "short", "regime": regime, "pnl": round(pnl_pct, 4), "bars": i - entry_idx})
                position = 0

        equity_curve.append(equity)

    if not trades:
        return {"trades": 0}

    wins = sum(1 for t in trades if t["pnl"] > 0)
    peak = max(equity_curve)
    dd = min((e - peak) / peak * 100 for e in equity_curve)

    by_regime = {}
    for t in trades:
        r = t["regime"]
        by_regime.setdefault(r, []).append(t["pnl"])

    regime_summary = {}
    for r, pnls in by_regime.items():
        regime_summary[r] = {
            "trades": len(pnls),
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
            "avg_pnl": round(np.mean(pnls), 4),
        }

    return {
        "trades": len(trades),
        "win_rate": round(wins / len(trades) * 100, 1),
        "total_pnl": round(equity - 10000, 2),
        "max_drawdown": round(dd, 2),
        "final_equity": round(equity, 2),
        "regime_summary": regime_summary,
    }


def vol_analysis(df: pd.DataFrame, base: str) -> dict:
    """Анализ волатильности."""
    vol = df["vol_pct"].dropna()
    return {
        "symbol": base,
        "mean_vol": round(vol.mean(), 2),
        "std_vol": round(vol.std(), 2),
        "current_vol": round(vol.iloc[-1], 2),
        "vol_rank": round(df["vol_rank"].iloc[-1], 2) if not pd.isna(df["vol_rank"].iloc[-1]) else None,
        "current_regime": str(df["regime"].iloc[-1]) if not pd.isna(df["regime"].iloc[-1]) else None,
    }


def main():
    files = list(DATA_DIR.glob(f"*_USDT_{TF}.parquet"))
    bases = sorted(set(f.stem.replace(f"_{TF}", "").replace("_USDT", "") for f in files))

    print("VOLATILITY REGIME RESEARCH")
    print("=" * 90)

    # 1. Volatility stats
    print("\n1) VOLATILITY STATS")
    print("-" * 90)
    print(f"{'Symbol':>8} {'MeanVol%':>9} {'CurVol%':>8} {'VolRank':>8} {'Regime':>10}")

    all_data = {}
    for base in bases:
        df = load_pair(base)
        if df is None or df.empty:
            continue
        df = calc_volatility(df)
        df = classify_regime(df)
        all_data[base] = df
        stats = vol_analysis(df, base)
        print(f"{base:>8} {stats['mean_vol']:>9.2f} {stats['current_vol']:>8.2f} {stats['vol_rank']:>8.2f} {stats['current_regime']:>10}")

    # 2. Backtest: momentum in low vol, reversion in high vol
    print(f"\n2) BACKTEST: Momentum(low vol) + MeanReversion(high vol)")
    print("-" * 90)
    print(f"{'Symbol':>8} {'Trades':>7} {'Win%':>6} {'PnL$':>9} {'MaxDD%':>8} {'Equity$':>9}")
    for base in bases:
        if base not in all_data:
            continue
        r = backtest_regime(all_data[base], strategy="both")
        if not r or r["trades"] == 0:
            continue
        print(f"{base:>8} {r['trades']:>7} {r['win_rate']:>6.1f} {r['total_pnl']:>9.2f} {r['max_drawdown']:>8.2f} {r['final_equity']:>9.2f}")

    # 3. Regime breakdown
    print(f"\n3) REGIME BREAKDOWN (best performer)")
    print("-" * 90)
    best = None
    best_pnl = -999
    for base in bases:
        if base not in all_data:
            continue
        r = backtest_regime(all_data[base], strategy="both")
        if r and r.get("total_pnl", 0) > best_pnl:
            best_pnl = r["total_pnl"]
            best = (base, r)

    if best:
        base, r = best
        print(f"Best: {base}")
        print(f"{'Regime':>10} {'Trades':>7} {'Win%':>6} {'Avg PnL%':>9}")
        for regime, stats in r.get("regime_summary", {}).items():
            print(f"{regime:>10} {stats['trades']:>7} {stats['win_rate']:>6.1f} {stats['avg_pnl']:>9.4f}")

    # 4. Pure momentum vs pure reversion comparison
    print(f"\n4) PURE STRATEGIES COMPARISON")
    print("-" * 90)
    print(f"{'Symbol':>8} {'Mom PnL$':>9} {'Rev PnL$':>9} {'Both PnL$':>10}")
    for base in bases:
        if base not in all_data:
            continue
        mom = backtest_regime(all_data[base], strategy="momentum")
        rev = backtest_regime(all_data[base], strategy="reversion")
        both = backtest_regime(all_data[base], strategy="both")
        mom_pnl = mom.get("total_pnl", 0) if mom else 0
        rev_pnl = rev.get("total_pnl", 0) if rev else 0
        both_pnl = both.get("total_pnl", 0) if both else 0
        print(f"{base:>8} {mom_pnl:>9.2f} {rev_pnl:>9.2f} {both_pnl:>10.2f}")


if __name__ == "__main__":
    main()
