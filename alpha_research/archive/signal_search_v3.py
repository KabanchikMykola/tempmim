"""Signal Search v3 — per-pair analysis + multiple horizons."""

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
WF_TRAIN = 30 * 24 * 3600000
WF_TEST = 7 * 24 * 3600000


def load_all():
    files = list(DATA_DIR.glob(f"*_{TF}.parquet"))
    bases = sorted(set(f.stem.replace(f"_{TF}", "").replace("_USDT", "") for f in files) - EXCLUDE)
    data = {}
    for base in bases:
        spot_f = DATA_DIR / f"{base}_USDT_{TF}.parquet"
        perp_f = DATA_DIR / f"{base}_USDT_USDT_{TF}.parquet"
        if not spot_f.exists() or not perp_f.exists():
            continue
        spot = pd.read_parquet(spot_f)
        perp = pd.read_parquet(perp_f)
        df = spot[["timestamp", "close", "high", "low", "volume"]].rename(
            columns={"close": "spot", "high": "high_", "low": "low_", "volume": "vol_"}
        ).merge(perp[["timestamp", "close"]].rename(columns={"close": "perp"}), on="timestamp", how="inner")
        df["returns"] = df["spot"].pct_change()
        df["basis_pct"] = ((df["perp"] - df["spot"]) / df["spot"]) * 100
        data[base] = df.sort_values("timestamp").reset_index(drop=True)
    return data


def prepare(df):
    df = df.copy()
    df["vol"] = df["returns"].rolling(24).std() * np.sqrt(8760)
    df["vol_rank"] = df["vol"].rolling(168).rank(pct=True)
    d = df["spot"].diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + g / l))
    df["basis_mean"] = df["basis_pct"].rolling(168).mean()
    df["basis_std"] = df["basis_pct"].rolling(168).std()
    df["basis_z"] = (df["basis_pct"] - df["basis_mean"]) / df["basis_std"]
    df["vol_ratio"] = df["vol_"] / df["vol_"].rolling(24).mean()
    df["mom_6h"] = df["spot"].pct_change(6)

    for h in [6, 12, 24, 48]:
        df[f"fwd_{h}h"] = df["spot"].shift(-h) / df["spot"] - 1
    return df


def calc_ic_series(df, sig, fwd):
    df_v = df.dropna(subset=[sig, fwd]).copy()
    if len(df_v) < 500:
        return []
    ic_series = []
    cursor = df_v["timestamp"].iloc[0]
    end = df_v["timestamp"].iloc[-1]
    while cursor + WF_TRAIN + WF_TEST <= end:
        t_s = cursor + WF_TRAIN
        t_e = t_s + WF_TEST
        chunk = df_v[(df_v["timestamp"] >= t_s) & (df_v["timestamp"] < t_e)]
        if len(chunk) >= 12:
            ic = chunk[sig].corr(chunk[fwd])
            if not np.isnan(ic):
                ic_series.append(ic)
        cursor += WF_TEST
    return ic_series


def calc_holdout_acc(df, sig, fwd):
    df_v = df.dropna(subset=[sig, fwd]).copy()
    df_v = df_v[df_v[sig] != 0]
    cutoff = df_v["timestamp"].iloc[-1] - HOLDOUT_DAYS * 24 * 3600000
    ho = df_v[df_v["timestamp"] >= cutoff]
    if len(ho) < 10:
        return 0, 0
    preds = np.sign(ho[sig])
    actuals = np.sign(ho[fwd])
    valid = ~(preds.isna() | actuals.isna())
    if valid.sum() < 5:
        return 0, int(valid.sum())
    return (preds[valid] == actuals[valid]).mean(), int(valid.sum())


def main():
    print("GOAL MODE: Per-Pair Signal Analysis")
    print("=" * 90)

    data = load_all()
    prepared = {b: prepare(d) for b, d in data.items()}

    signals = [
        ("basis_z", "fwd_6h"), ("basis_z", "fwd_12h"), ("basis_z", "fwd_24h"), ("basis_z", "fwd_48h"),
        ("rsi", "fwd_6h"), ("rsi", "fwd_12h"), ("rsi", "fwd_24h"),
        ("vol_ratio", "fwd_6h"), ("vol_ratio", "fwd_12h"),
        ("mom_6h", "fwd_6h"), ("mom_6h", "fwd_12h"),
        ("vol_rank", "fwd_24h"), ("vol_rank", "fwd_48h"),
    ]

    candidates = []
    for sig, fwd in signals:
        for base, df in prepared.items():
            ic_series = calc_ic_series(df, sig, fwd)
            ho_acc, ho_trades = calc_holdout_acc(df, sig, fwd)
            if len(ic_series) < 3 or ho_trades < 5:
                continue
            avg_ic = np.mean(ic_series)
            stable_pct = sum(1 for ic in ic_series if ic > 0) / len(ic_series)
            if avg_ic > 0.01 and stable_pct > 0.4 and ho_acc > 0.50:
                candidates.append({
                    "pair": base, "signal": sig, "fwd": fwd,
                    "ic": round(avg_ic, 4), "stable": round(stable_pct, 2),
                    "ho_acc": round(ho_acc, 4), "ho_trades": ho_trades,
                    "windows": len(ic_series),
                })

    candidates.sort(key=lambda x: x["ic"], reverse=True)

    print(f"\nCANDIDATES (IC > 0.01, stable > 40%, HO > 50%):")
    print(f"{'Pair':>8} {'Signal':>10} {'Fwd':>6} {'IC':>7} {'Stable':>7} {'HO_Acc':>7} {'HO#':>5} {'Win':>7}")
    for c in candidates[:30]:
        print(f"{c['pair']:>8} {c['signal']:>10} {c['fwd']:>6} {c['ic']:>7.4f} {c['stable']:>7.0%} {c['ho_acc']:>7.1%} {c['ho_trades']:>5} {'WIN' if c['ho_acc'] > 0.52 else ''}")

    if not candidates:
        print("\nNo candidates found. Checking raw stats...")
        for sig, fwd in signals[:3]:
            ics = []
            for base, df in prepared.items():
                ic_s = calc_ic_series(df, sig, fwd)
                if ic_s:
                    ics.extend(ic_s)
            if ics:
                print(f"  {sig}→{fwd}: mean_ic={np.mean(ics):.4f} std={np.std(ics):.4f} n={len(ics)} positive={sum(1 for i in ics if i>0)/len(ics):.0%}")


if __name__ == "__main__":
    main()
