"""Re-evaluate top 3 strategies with real costs."""

import sys, io, json, time
import pandas as pd, numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path("data")
WF_TRAIN = 30 * 24 * 3600000
WF_TEST = 7 * 24 * 3600000
HOLDOUT_DAYS = 14
COST_PCT = 0.08 + 0.02  # 0.08% commission + 0.02% slippage = 0.10% round trip


def load_pair(base):
    sf = DATA_DIR / f"{base}USDT_1h_spot.parquet"
    pf = DATA_DIR / f"{base}USDT_1h_perp.parquet"
    if not sf.exists() or not pf.exists(): return None
    s = pd.read_parquet(sf)[["timestamp","close"]].rename(columns={"close":"spot"})
    p = pd.read_parquet(pf)[["timestamp","close"]].rename(columns={"close":"perp"})
    df = s.merge(p, on="timestamp", how="inner")
    df["basis"] = (df["perp"] - df["spot"]) / df["spot"]
    df["basis_z"] = (df["basis"] - df["basis"].rolling(168).mean()) / df["basis"].rolling(168).std()
    return df.dropna(subset=["basis_z"])


def bt_basis_cost(df, entry_z, exit_z, max_hold):
    pos = 0; entry = 0; entry_idx = 0; trades = []
    for i in range(1, len(df)):
        z = df["basis_z"].iloc[i]
        if pos == 0:
            if z > entry_z:
                pos = -1; entry = df["spot"].iloc[i]; entry_idx = i
            elif z < -entry_z:
                pos = 1; entry = df["spot"].iloc[i]; entry_idx = i
        elif pos == 1 and (z > -exit_z or i - entry_idx > max_hold):
            pnl = (df["spot"].iloc[i] / entry - 1) * 100 - COST_PCT
            trades.append(pnl); pos = 0
        elif pos == -1 and (z < exit_z or i - entry_idx > max_hold):
            pnl = (1 - df["spot"].iloc[i] / entry) * 100 - COST_PCT
            trades.append(pnl); pos = 0
    return trades


def evaluate_cost(df, params):
    start = df["timestamp"].iloc[0]; end = df["timestamp"].iloc[-1]
    wf_results = []; cursor = start
    while cursor + WF_TRAIN + WF_TEST <= end:
        t_s = cursor + WF_TRAIN; t_e = t_s + WF_TEST
        chunk = df[(df["timestamp"] >= t_s) & (df["timestamp"] < t_e)]
        if len(chunk) >= 100:
            trades = bt_basis_cost(chunk, **params)
            if trades:
                wins = sum(1 for t in trades if t > 0)
                gross = sum(t for t in trades if t > 0)
                loss = abs(sum(t for t in trades if t <= 0))
                wf_results.append({"trades": len(trades), "wr": wins/len(trades), "pnl": sum(trades), "pf": gross/loss if loss > 0 else 999})
        cursor += WF_TEST

    cutoff = df["timestamp"].iloc[-1] - HOLDOUT_DAYS * 24 * 3600000
    ho_df = df[df["timestamp"] >= cutoff]
    ho_trades = bt_basis_cost(ho_df, **params)
    if not ho_trades or not wf_results: return None
    ho_wins = sum(1 for t in ho_trades if t > 0)
    ho_gross = sum(t for t in ho_trades if t > 0)
    ho_loss = abs(sum(t for t in ho_trades if t <= 0))
    return {
        "wf_pf": np.mean([r["pf"] for r in wf_results]),
        "wf_wr": np.mean([r["wr"] for r in wf_results]),
        "wf_trades": sum(r["trades"] for r in wf_results),
        "wf_pnl": sum(r["pnl"] for r in wf_results),
        "ho_pnl": sum(ho_trades), "ho_wr": ho_wins/len(ho_trades),
        "ho_trades": len(ho_trades), "ho_pf": ho_gross/ho_loss if ho_loss > 0 else 999
    }


def main():
    t0 = time.time()
    configs = [
        ("Basis_SOL", "SOL", {"entry_z": 2.0, "exit_z": 0.5, "max_hold": 48}),
        ("Basis_ADA", "ADA", {"entry_z": 2.0, "exit_z": 0.5, "max_hold": 48}),
        ("Basis_tight_BNB", "BNB", {"entry_z": 1.5, "exit_z": 0.25, "max_hold": 36}),
    ]

    print(f"Re-evaluating with costs: {COST_PCT}% round-trip")
    print("=" * 80)
    print(f"{'Strategy':>20} {'WF_PF':>7} {'WF_WR':>7} {'WF_PnL':>9} {'HO_Ret':>8} {'HO_WR':>7} {'HO_T':>5} {'HO_PF':>7}")

    for name, base, params in configs:
        df = load_pair(base)
        if df is None: continue
        r = evaluate_cost(df, params)
        if r is None: continue
        print(f"{name:>20} {r['wf_pf']:>7.2f} {r['wf_wr']:>6.0%} {r['wf_pnl']:>9.1f} {r['ho_pnl']:>7.1f}% {r['ho_wr']:>6.0%} {r['ho_trades']:>5} {r['ho_pf']:>7.2f}")

    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
