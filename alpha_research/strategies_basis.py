"""Goal Mode: Strategy Search — Class 1: Basis Trading."""

import sys, io, json, time
import pandas as pd, numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path("data/top5_2026")
WF_TRAIN = 30 * 24 * 3600000
WF_TEST = 7 * 24 * 3600000
HOLDOUT_DAYS = 14
RESULTS_FILE = Path("alpha_research/strategies.json")


def load_pair(base):
    sf = DATA_DIR / f"{base}_USDT_1h.parquet"
    pf = DATA_DIR / f"{base}_USDT_USDT_1h.parquet"
    if not sf.exists() or not pf.exists(): return None
    s = pd.read_parquet(sf)[["timestamp","close"]].rename(columns={"close":"spot"})
    p = pd.read_parquet(pf)[["timestamp","close"]].rename(columns={"close":"perp"})
    df = s.merge(p, on="timestamp", how="inner")
    df["basis"] = (df["perp"] - df["spot"]) / df["spot"]
    df["basis_z"] = (df["basis"] - df["basis"].rolling(168).mean()) / df["basis"].rolling(168).std()
    df["fwd_24h"] = df["spot"].shift(-24) / df["spot"] - 1
    return df.dropna(subset=["basis_z","fwd_24h"])


def backtest_basis(df, entry_z=2.0, exit_z=0.5, hold_bars=48):
    df = df.copy()
    trades = []
    pos = 0; entry = 0; entry_idx = 0
    for i in range(1, len(df)):
        z = df["basis_z"].iloc[i]
        if pos == 0:
            if z > entry_z:
                pos = -1; entry = df["spot"].iloc[i]; entry_idx = i
            elif z < -entry_z:
                pos = 1; entry = df["spot"].iloc[i]; entry_idx = i
        elif pos == 1 and (z > -exit_z or i - entry_idx > hold_bars):
            pnl = (df["spot"].iloc[i] / entry - 1) * 100
            trades.append(pnl); pos = 0
        elif pos == -1 and (z < exit_z or i - entry_idx > hold_bars):
            pnl = (1 - df["spot"].iloc[i] / entry) * 100
            trades.append(pnl); pos = 0
    return trades


def walk_forward(df, entry_z, exit_z, hold_bars):
    results = []
    start = df["timestamp"].iloc[0]; end = df["timestamp"].iloc[-1]
    cursor = start
    while cursor + WF_TRAIN + WF_TEST <= end:
        t_s = cursor + WF_TRAIN; t_e = t_s + WF_TEST
        chunk = df[(df["timestamp"] >= t_s) & (df["timestamp"] < t_e)]
        if len(chunk) >= 100:
            trades = backtest_basis(chunk, entry_z, exit_z, hold_bars)
            if trades:
                wins = sum(1 for t in trades if t > 0)
                results.append({"trades": len(trades), "win_rate": wins/len(trades), "total_pnl": sum(trades)})
        cursor += WF_TEST
    return results


def holdout_test(df, entry_z, exit_z, hold_bars):
    cutoff = df["timestamp"].iloc[-1] - HOLDOUT_DAYS * 24 * 3600000
    test = df[df["timestamp"] >= cutoff]
    trades = backtest_basis(test, entry_z, exit_z, hold_bars)
    if not trades: return None
    wins = sum(1 for t in trades if t > 0)
    gross = sum(t for t in trades if t > 0)
    loss = abs(sum(t for t in trades if t <= 0))
    pf = gross / loss if loss > 0 else 999
    return {"trades": len(trades), "win_rate": wins/len(trades), "total_pnl": sum(trades), "profit_factor": pf, "max_dd": min(trades) if trades else 0}


def main():
    t0 = time.time()
    files = list(DATA_DIR.glob("*_1h.parquet"))
    bases = sorted(set(f.stem.replace("_1h","").replace("_USDT","") for f in files) - {"USDC"})
    data = {}
    for b in bases:
        d = load_pair(b)
        if d is not None and len(d) > 500: data[b] = d

    params = [(1.5, 0.5, 48), (2.0, 0.5, 48), (2.5, 0.5, 48), (2.0, 0.25, 36), (2.0, 0.75, 60)]
    found = []

    for entry_z, exit_z, hold in params:
        for base, df in data.items():
            wf = walk_forward(df, entry_z, exit_z, hold)
            ho = holdout_test(df, entry_z, exit_z, hold)
            if not wf or not ho: continue
            avg_pf = np.mean([1 + r["total_pnl"]/100 for r in wf]) if wf else 0
            avg_wr = np.mean([r["win_rate"] for r in wf])
            if avg_pf > 1.2 and ho["total_pnl"] > 0 and ho["trades"] >= 15 and ho["win_rate"] > 0.48:
                found.append({"pair": base, "params": {"entry_z": entry_z, "exit_z": exit_z, "hold": hold},
                    "wf_pf": round(avg_pf, 3), "wf_wr": round(avg_wr, 3), "wf_trades": sum(r["trades"] for r in wf),
                    "ho_pnl": round(ho["total_pnl"], 2), "ho_wr": round(ho["win_rate"], 3), "ho_trades": ho["trades"], "ho_pf": round(ho["profit_factor"], 3)})

    found.sort(key=lambda x: x["wf_pf"], reverse=True)
    print(f"BASIS TRADING: {len(found)} strategies found ({time.time()-t0:.0f}s)")
    for f in found[:10]:
        print(f"  {f['pair']:>8} entry_z={f['params']['entry_z']} exit_z={f['params']['exit_z']}: WF_PF={f['wf_pf']:.2f} WR={f['wf_wr']:.0%} | HO_PnL={f['ho_pnl']:+.1f} WR={f['ho_wr']:.0%} trades={f['ho_trades']}")

    strategies = []
    for f in found[:3]:
        strategies.append({"name": f"Basis_{f['pair']}", "class": "basis", "pairs": [f["pair"]], "params": f["params"],
            "walk_forward": {"profit_factor": f["wf_pf"], "win_rate": f["wf_wr"], "trades": f["wf_trades"]},
            "holdout": {"return": f["ho_pnl"], "win_rate": f["ho_wr"], "trades": f["ho_trades"], "profit_factor": f["ho_pf"]}})

    if strategies:
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w") as fp: json.dump(strategies, fp, indent=2)
        print(f"Saved {len(strategies)} strategies to {RESULTS_FILE}")

    return strategies


if __name__ == "__main__":
    main()
