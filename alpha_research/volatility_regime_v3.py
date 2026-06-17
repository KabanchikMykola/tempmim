"""Volatility Regime v3 — Walk-Forward + Holdout + Dynamic Exits."""

import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path("data/top5_2026")
TF = "1h"
EXCLUDE = {"USDC"}
HOLDOUT_DAYS = 14


def load_pair(base: str) -> pd.DataFrame | None:
    spot_file = DATA_DIR / f"{base}_USDT_{TF}.parquet"
    perp_file = DATA_DIR / f"{base}_USDT_USDT_{TF}.parquet"
    if not spot_file.exists() or not perp_file.exists():
        return None
    spot = pd.read_parquet(spot_file)[["timestamp", "open", "close", "high", "low"]].rename(
        columns={"close": "spot", "high": "spot_high", "low": "spot_low", "open": "spot_open"}
    )
    perp = pd.read_parquet(perp_file)[["timestamp", "close"]].rename(columns={"close": "perp"})
    df = spot.merge(perp, on="timestamp", how="inner")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["returns"] = df["spot"].pct_change()
    df["basis_pct"] = ((df["perp"] - df["spot"]) / df["spot"]) * 100
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["vol"] = df["returns"].rolling(24).std() * np.sqrt(8760)
    df["vol_rank"] = df["vol"].rolling(168).rank(pct=True)

    delta = df["spot"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    df["basis_mean"] = df["basis_pct"].rolling(168).mean()
    df["basis_std"] = df["basis_pct"].rolling(168).std()
    df["basis_z"] = (df["basis_pct"] - df["basis_mean"]) / df["basis_std"]

    df["ma24"] = df["spot"].rolling(24).mean()
    df["std24"] = df["spot"].rolling(24).std()
    df["bb_upper"] = df["ma24"] + 2 * df["std24"]
    df["bb_lower"] = df["ma24"] - 2 * df["std24"]

    tr = pd.concat([
        df["spot_high"] - df["spot_low"],
        (df["spot_high"] - df["spot"].shift(1)).abs(),
        (df["spot_low"] - df["spot"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    return df


def backtest_v3(
    df: pd.DataFrame,
    vol_threshold: float = 0.75,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    basis_filter: bool = True,
    atr_stop_mult: float = 2.0,
    use_bb_exit: bool = True,
) -> dict:
    """Walk-forward backtest: signal on bar i, execute on bar i+1 open."""
    df = df.dropna(subset=["vol_rank", "rsi", "basis_z", "atr14", "bb_upper"]).copy()
    if len(df) < 200:
        return {}

    equity = 10000
    equity_curve = [equity]
    trades = []
    position = 0
    entry_price = 0
    stop_loss = 0
    entry_idx = 0

    for i in range(1, len(df) - 1):
        vol_rank = df["vol_rank"].iloc[i]
        rsi = df["rsi"].iloc[i]
        basis_z = df["basis_z"].iloc[i]
        atr = df["atr14"].iloc[i]
        high_vol = vol_rank > vol_threshold

        signal = 0
        if high_vol and rsi < rsi_oversold:
            if not basis_filter or basis_z < -1:
                signal = 1
        elif high_vol and rsi > rsi_overbought:
            if not basis_filter or basis_z > 1:
                signal = -1

        if position == 0 and signal != 0:
            entry_price = df["spot_open"].iloc[i + 1]
            entry_idx = i + 1
            position = signal
            if signal == 1:
                stop_loss = entry_price - atr_stop_mult * atr
            else:
                stop_loss = entry_price + atr_stop_mult * atr

        elif position != 0:
            low = df["spot_low"].iloc[i]
            high = df["spot_high"].iloc[i]
            price = df["spot"].iloc[i]

            hit_sl = (position == 1 and low <= stop_loss) or (position == -1 and high >= stop_loss)

            bb_exit = False
            if use_bb_exit:
                if position == 1 and high >= df["bb_upper"].iloc[i]:
                    bb_exit = True
                elif position == -1 and low <= df["bb_lower"].iloc[i]:
                    bb_exit = True

            ma_exit = False
            if position == 1 and price >= df["ma24"].iloc[i]:
                ma_exit = True
            elif position == -1 and price <= df["ma24"].iloc[i]:
                ma_exit = True

            timeout = i - entry_idx > 48

            if hit_sl:
                exit_price = stop_loss
                pnl_pct = (exit_price / entry_price - 1) * 100 * position
                equity *= (1 + pnl_pct / 100 * 0.998)
                trades.append({"side": "long" if position == 1 else "short", "pnl": round(pnl_pct, 4), "exit": "sl"})
                position = 0
            elif bb_exit or ma_exit or timeout:
                exit_price = price
                pnl_pct = (exit_price / entry_price - 1) * 100 * position
                equity *= (1 + pnl_pct / 100 * 0.998)
                exit_type = "bb" if bb_exit else ("ma" if ma_exit else "time")
                trades.append({"side": "long" if position == 1 else "short", "pnl": round(pnl_pct, 4), "exit": exit_type})
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
        exits.setdefault(e, {"count": 0, "wins": 0})
        exits[e]["count"] += 1
        if t["pnl"] > 0:
            exits[e]["wins"] += 1

    return {
        "trades": len(trades),
        "win_rate": round(wins / len(trades) * 100, 1),
        "total_pnl": round(equity - 10000, 2),
        "max_drawdown": round(dd, 2),
        "final_equity": round(equity, 2),
        "avg_pnl": round(np.mean([t["pnl"] for t in trades]), 4),
        "exits": {k: v["count"] for k, v in exits.items()},
    }


def walk_forward(df: pd.DataFrame, train_days: int = 30, test_days: int = 7) -> list[dict]:
    """Walk-forward: train на train_days, trade на test_days."""
    train_ms = train_days * 24 * 3600000
    test_ms = test_days * 24 * 3600000
    start = df["timestamp"].iloc[0]
    end = df["timestamp"].iloc[-1]
    results = []
    cursor = start

    while cursor + train_ms + test_ms <= end:
        test_start = cursor + train_ms
        test_end = min(test_start + test_ms, end)
        test_df = df[(df["timestamp"] >= test_start) & (df["timestamp"] < test_end)].copy()

        if len(test_df) < 24:
            cursor += test_ms
            continue

        r = backtest_v3(test_df)
        if r and r["trades"] > 0:
            results.append(r)
        cursor += test_ms

    return results


def holdout_test(df: pd.DataFrame, holdout_days: int = 14) -> dict:
    """Тест на последних N днях (out-of-sample)."""
    cutoff = df["timestamp"].iloc[-1] - holdout_days * 24 * 3600000
    test_df = df[df["timestamp"] >= cutoff].copy()
    return backtest_v3(test_df)


def main():
    files = list(DATA_DIR.glob(f"*_USDT_{TF}.parquet"))
    bases = sorted(set(f.stem.replace(f"_{TF}", "").replace("_USDT", "") for f in files) - EXCLUDE)

    print("VOLATILITY REGIME v3 — Walk-Forward + Holdout + Dynamic Exits")
    print("=" * 95)

    all_data = {}
    for base in bases:
        df = load_pair(base)
        if df is None or df.empty:
            continue
        df = add_indicators(df)
        all_data[base] = df

    # 1. Walk-forward
    print("\n1) WALK-FORWARD (30d train, 7d test)")
    print("-" * 95)
    print(f"{'Symbol':>8} {'Periods':>8} {'AvgTrades':>10} {'AvgWin%':>8} {'AvgPnL$':>9} {'AvgDD%':>8}")

    for base in bases:
        if base not in all_data:
            continue
        wf_results = walk_forward(all_data[base])
        if not wf_results:
            continue
        avg_trades = np.mean([r["trades"] for r in wf_results])
        avg_wr = np.mean([r["win_rate"] for r in wf_results])
        avg_pnl = np.mean([r["total_pnl"] for r in wf_results])
        avg_dd = np.mean([r["max_drawdown"] for r in wf_results])
        print(f"{base:>8} {len(wf_results):>8} {avg_trades:>10.1f} {avg_wr:>8.1f} {avg_pnl:>9.2f} {avg_dd:>8.2f}")

    # 2. Holdout (last 14 days)
    print(f"\n2) HOLDOUT TEST (last {HOLDOUT_DAYS} days — out-of-sample)")
    print("-" * 95)
    print(f"{'Symbol':>8} {'Trades':>7} {'Win%':>6} {'PnL$':>9} {'Avg%':>7} {'MaxDD%':>8} {'Exits':>20}")

    holdout_results = []
    for base in bases:
        if base not in all_data:
            continue
        r = holdout_test(all_data[base], HOLDOUT_DAYS)
        if not r or r["trades"] == 0:
            continue
        holdout_results.append((base, r))
        exits_str = " ".join(f"{k}:{v}" for k, v in r["exits"].items())
        print(f"{base:>8} {r['trades']:>7} {r['win_rate']:>6.1f} {r['total_pnl']:>9.2f} {r['avg_pnl']:>7.4f} {r['max_drawdown']:>8.2f} {exits_str:>20}")

    profitable_ho = sum(1 for _, r in holdout_results if r["total_pnl"] > 0)
    print(f"\nПрибыльных на holdout: {profitable_ho}/{len(holdout_results)}")

    # 3. Full sample comparison
    print(f"\n3) FULL SAMPLE (для сравнения — НЕ для оценки!)")
    print("-" * 95)
    print(f"{'Symbol':>8} {'Trades':>7} {'Win%':>6} {'PnL$':>9} {'Avg%':>7} {'MaxDD%':>8}")

    for base in bases:
        if base not in all_data:
            continue
        r = backtest_v3(all_data[base])
        if not r or r["trades"] == 0:
            continue
        print(f"{base:>8} {r['trades']:>7} {r['win_rate']:>6.1f} {r['total_pnl']:>9.2f} {r['avg_pnl']:>7.4f} {r['max_drawdown']:>8.2f}")

    # 4. Current signals
    print(f"\n4) CURRENT SIGNALS (holdout period)")
    print("-" * 95)
    for base in bases:
        if base not in all_data:
            continue
        df = all_data[base]
        last = df.iloc[-1]
        vol_rank = last["vol_rank"]
        rsi = last["rsi"]
        basis_z = last["basis_z"]
        if pd.isna(vol_rank) or pd.isna(rsi) or pd.isna(basis_z):
            continue
        if vol_rank > 0.75:
            if rsi < 30 and basis_z < -1:
                print(f"  {base:>8}: LONG | vol={vol_rank:.2f} rsi={rsi:.1f} basis_z={basis_z:.2f} atr={last['atr14']:.4f}")
            elif rsi > 70 and basis_z > 1:
                print(f"  {base:>8}: SHORT | vol={vol_rank:.2f} rsi={rsi:.1f} basis_z={basis_z:.2f} atr={last['atr14']:.4f}")


if __name__ == "__main__":
    main()
