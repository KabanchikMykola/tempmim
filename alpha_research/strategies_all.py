"""Goal Mode: All strategy classes in one script."""

import sys, io, json, time
import pandas as pd, numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path("data")
WF_TRAIN = 30 * 24 * 3600000
WF_TEST = 7 * 24 * 3600000
HOLDOUT_DAYS = 14
RESULTS_FILE = Path("alpha_research/strategies.json")


def load_pair(base):
    sf = DATA_DIR / f"{base}USDT_1h_spot.parquet"
    pf = DATA_DIR / f"{base}USDT_1h_perp.parquet"
    if not sf.exists() or not pf.exists(): return None
    s = pd.read_parquet(sf)[["timestamp","close","high","low","volume"]].rename(columns={"close":"spot","high":"high_","low":"low_","volume":"vol_"})
    p = pd.read_parquet(pf)[["timestamp","close"]].rename(columns={"close":"perp"})
    df = s.merge(p, on="timestamp", how="inner")
    df["returns"] = df["spot"].pct_change()
    df["basis"] = (df["perp"] - df["spot"]) / df["spot"]
    df["basis_z"] = (df["basis"] - df["basis"].rolling(168).mean()) / df["basis"].rolling(168).std()
    d = df["spot"].diff(); g = d.clip(lower=0).rolling(14).mean(); l = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + g / l))
    df["vol"] = df["returns"].rolling(24).std() * np.sqrt(8760)
    df["vol_rank"] = df["vol"].rolling(168).rank(pct=True)
    df["vol_ratio"] = df["vol_"] / df["vol_"].rolling(24).mean()
    df["ma24"] = df["spot"].rolling(24).mean()
    df["mom_6h"] = df["spot"].pct_change(6)
    for h in [6, 12, 24, 48]:
        df[f"fwd_{h}h"] = df["spot"].shift(-h) / df["spot"] - 1
    return df.dropna()


def bt_reversion(df, rsi_entry=30, rsi_exit=50, vol_filter=True, max_hold=48):
    pos = 0; entry = 0; entry_idx = 0; trades = []
    for i in range(1, len(df)):
        rsi = df["rsi"].iloc[i]; high_vol = df["vol_rank"].iloc[i] > 0.75 if vol_filter else True
        if pos == 0:
            if high_vol and rsi < rsi_entry: pos = 1; entry = df["spot"].iloc[i]; entry_idx = i
            elif high_vol and rsi > (100 - rsi_entry): pos = -1; entry = df["spot"].iloc[i]; entry_idx = i
        elif pos == 1 and (rsi > rsi_exit or i - entry_idx > max_hold):
            trades.append((df["spot"].iloc[i] / entry - 1) * 100); pos = 0
        elif pos == -1 and (rsi < (100 - rsi_exit) or i - entry_idx > max_hold):
            trades.append((1 - df["spot"].iloc[i] / entry) * 100); pos = 0
    return trades


def bt_basis(df, entry_z=2.0, exit_z=0.5, max_hold=48):
    pos = 0; entry = 0; entry_idx = 0; trades = []
    for i in range(1, len(df)):
        z = df["basis_z"].iloc[i]
        if pos == 0:
            if z > entry_z: pos = -1; entry = df["spot"].iloc[i]; entry_idx = i
            elif z < -entry_z: pos = 1; entry = df["spot"].iloc[i]; entry_idx = i
        elif pos == 1 and (z > -exit_z or i - entry_idx > max_hold):
            trades.append((df["spot"].iloc[i] / entry - 1) * 100); pos = 0
        elif pos == -1 and (z < exit_z or i - entry_idx > max_hold):
            trades.append((1 - df["spot"].iloc[i] / entry) * 100); pos = 0
    return trades


def bt_momentum(df, mom_threshold=0.005, vol_filter=False, max_hold=24):
    pos = 0; entry = 0; entry_idx = 0; trades = []
    for i in range(1, len(df)):
        mom = df["mom_6h"].iloc[i]; high_vol = df["vol_rank"].iloc[i] < 0.25 if vol_filter else True
        if pos == 0:
            if high_vol and mom > mom_threshold: pos = 1; entry = df["spot"].iloc[i]; entry_idx = i
            elif high_vol and mom < -mom_threshold: pos = -1; entry = df["spot"].iloc[i]; entry_idx = i
        elif pos == 1 and (mom < 0 or i - entry_idx > max_hold):
            trades.append((df["spot"].iloc[i] / entry - 1) * 100); pos = 0
        elif pos == -1 and (mom > 0 or i - entry_idx > max_hold):
            trades.append((1 - df["spot"].iloc[i] / entry) * 100); pos = 0
    return trades


def bt_vol_squeeze(df, range_threshold=0.03, vol_filter=True, max_hold=48):
    pos = 0; entry = 0; entry_idx = 0; trades = []
    for i in range(1, len(df)):
        rsi = df["rsi"].iloc[i]; high_vol = df["vol_rank"].iloc[i] > 0.75 if vol_filter else True
        if pos == 0 and high_vol:
            if rsi < 30: pos = 1; entry = df["spot"].iloc[i]; entry_idx = i
            elif rsi > 70: pos = -1; entry = df["spot"].iloc[i]; entry_idx = i
        elif pos == 1 and (rsi > 55 or i - entry_idx > max_hold):
            trades.append((df["spot"].iloc[i] / entry - 1) * 100); pos = 0
        elif pos == -1 and (rsi < 45 or i - entry_idx > max_hold):
            trades.append((1 - df["spot"].iloc[i] / entry) * 100); pos = 0
    return trades


def evaluate(df, bt_func, bt_params):
    start = df["timestamp"].iloc[0]; end = df["timestamp"].iloc[-1]
    wf_results = []; cursor = start
    while cursor + WF_TRAIN + WF_TEST <= end:
        t_s = cursor + WF_TRAIN; t_e = t_s + WF_TEST
        chunk = df[(df["timestamp"] >= t_s) & (df["timestamp"] < t_e)]
        if len(chunk) >= 100:
            trades = bt_func(chunk, **bt_params)
            if trades:
                wins = sum(1 for t in trades if t > 0)
                gross = sum(t for t in trades if t > 0)
                loss = abs(sum(t for t in trades if t <= 0))
                wf_results.append({"trades": len(trades), "win_rate": wins/len(trades), "pnl": sum(trades), "pf": gross/loss if loss > 0 else 999})
        cursor += WF_TEST

    cutoff = df["timestamp"].iloc[-1] - HOLDOUT_DAYS * 24 * 3600000
    ho_df = df[df["timestamp"] >= cutoff]
    ho_trades = bt_func(ho_df, **bt_params)
    if not ho_trades or not wf_results: return None
    ho_wins = sum(1 for t in ho_trades if t > 0)
    ho_gross = sum(t for t in ho_trades if t > 0)
    ho_loss = abs(sum(t for t in ho_trades if t <= 0))
    ho_pf = ho_gross / ho_loss if ho_loss > 0 else 999
    avg_wf_pf = np.mean([r["pf"] for r in wf_results])
    avg_wf_wr = np.mean([r["win_rate"] for r in wf_results])
    return {"wf_pf": avg_wf_pf, "wf_wr": avg_wf_wr, "wf_trades": sum(r["trades"] for r in wf_results),
        "ho_pnl": sum(ho_trades), "ho_wr": ho_wins/len(ho_trades), "ho_trades": len(ho_trades), "ho_pf": ho_pf}


def main():
    t0 = time.time()
    files = list(DATA_DIR.glob("*_1h_spot.parquet"))
    bases = sorted(set(f.stem.replace("_1h_spot","").removesuffix("USDT") for f in files) - {"USDC"})
    data = {}
    for b in bases:
        d = load_pair(b)
        if d is not None and len(d) > 500: data[b] = d
    print(f"Loaded {len(data)} pairs")

    strategies = []
    configs = [
        ("Basis", bt_basis, {"entry_z": 2.0, "exit_z": 0.5, "max_hold": 48}),
        ("Basis_tight", bt_basis, {"entry_z": 1.5, "exit_z": 0.25, "max_hold": 36}),
        ("Reversion", bt_reversion, {"rsi_entry": 30, "rsi_exit": 50, "vol_filter": True, "max_hold": 48}),
        ("Reversion_loose", bt_reversion, {"rsi_entry": 35, "rsi_exit": 55, "vol_filter": True, "max_hold": 60}),
        ("Momentum", bt_momentum, {"mom_threshold": 0.005, "vol_filter": False, "max_hold": 24}),
        ("VolSqueeze", bt_vol_squeeze, {"range_threshold": 0.03, "vol_filter": True, "max_hold": 48}),
    ]

    for name, bt_func, params in configs:
        for base, df in data.items():
            r = evaluate(df, bt_func, params)
            if r and r["wf_pf"] > 1.2 and r["ho_pnl"] > 0 and r["ho_trades"] >= 15 and r["ho_wr"] > 0.48:
                strategies.append({"name": f"{name}_{base}", "class": name.lower(), "pairs": [base], "params": params,
                    "walk_forward": {"profit_factor": round(r["wf_pf"], 3), "win_rate": round(r["wf_wr"], 3), "trades": r["wf_trades"]},
                    "holdout": {"return": round(r["ho_pnl"], 2), "win_rate": round(r["ho_wr"], 3), "trades": r["ho_trades"], "profit_factor": round(r["ho_pf"], 3)}})

    strategies.sort(key=lambda x: x["walk_forward"]["profit_factor"], reverse=True)
    print(f"\nFOUND: {len(strategies)} strategies ({time.time()-t0:.0f}s)")
    for s in strategies[:15]:
        wf = s["walk_forward"]; ho = s["holdout"]
        print(f"  {s['name']:>25} WF:PF={wf['profit_factor']:.2f} WR={wf['win_rate']:.0%} T={wf['trades']} | HO:ret={ho['return']:+.1f} WR={ho['win_rate']:.0%} T={ho['trades']}")

    if strategies:
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w") as fp: json.dump(strategies, fp, indent=2)
        print(f"\nSaved to {RESULTS_FILE}")
    else:
        print("\nNo strategies met criteria.")


if __name__ == "__main__":
    main()
