"""Signal Search v4 — strict stability (IC > 0 in ALL windows)."""

import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path("data")
TF = "1h"
EXCLUDE = {"USDC"}
HOLDOUT_DAYS = 14
WF_TRAIN = 30 * 24 * 3600000
WF_TEST = 7 * 24 * 3600000


def load_all():
    files = list(DATA_DIR.glob(f"*_{TF}_spot.parquet"))
    bases = sorted(set(f.stem.replace(f"_{TF}_spot", "").removesuffix("USDT") for f in files) - EXCLUDE)
    data = {}
    for base in bases:
        spot_f = DATA_DIR / f"{base}USDT_{TF}_spot.parquet"
        perp_f = DATA_DIR / f"{base}USDT_{TF}_perp.parquet"
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


def strict_eval(df, sig, fwd):
    df_v = df.dropna(subset=[sig, fwd]).copy()
    if len(df_v) < 500:
        return None

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

    if len(ic_series) < 3:
        return None

    all_positive = all(ic > 0 for ic in ic_series)
    avg_ic = np.mean(ic_series)
    min_ic = min(ic_series)

    df_v2 = df_v[df_v[sig] != 0]
    cutoff = df_v2["timestamp"].iloc[-1] - HOLDOUT_DAYS * 24 * 3600000
    ho = df_v2[df_v2["timestamp"] >= cutoff]
    if len(ho) < 10:
        return None
    preds = np.sign(ho[sig])
    actuals = np.sign(ho[fwd])
    valid = ~(preds.isna() | actuals.isna())
    ho_acc = (preds[valid] == actuals[valid]).mean() if valid.sum() > 5 else 0
    ho_trades = int(valid.sum())

    return {
        "avg_ic": avg_ic, "min_ic": min_ic, "all_positive": all_positive,
        "windows": len(ic_series), "ic_series": ic_series,
        "ho_acc": ho_acc, "ho_trades": ho_trades,
    }


def main():
    print("GOAL MODE: Strict Stability Search")
    print("=" * 90)

    data = load_all()
    prepared = {b: prepare(d) for b, d in data.items()}

    signals = []
    for sig in ["basis_z", "rsi", "vol_rank", "vol_ratio", "mom_6h"]:
        for fwd in ["fwd_6h", "fwd_12h", "fwd_24h", "fwd_48h"]:
            signals.append((sig, fwd))

    strict_found = []
    near_found = []

    for sig, fwd in signals:
        for base, df in prepared.items():
            r = strict_eval(df, sig, fwd)
            if r is None:
                continue

            meets_all = (
                r["avg_ic"] > 0.03 and
                r["ho_acc"] > 0.52 and
                r["all_positive"] and
                r["ho_trades"] >= 10
            )
            near = (
                r["avg_ic"] > 0.02 and
                r["ho_acc"] > 0.50 and
                r["all_positive"]
            )

            if meets_all:
                strict_found.append({"pair": base, "signal": sig, "fwd": fwd, **r})
            elif near:
                near_found.append({"pair": base, "signal": sig, "fwd": fwd, **r})

    print(f"\nSTRICT MATCHES (IC>0.03, HO>52%, IC>0 ALL windows, trades>=10):")
    if strict_found:
        for s in strict_found:
            print(f"  {s['pair']:>8} {s['signal']:>10} {s['fwd']:>6} IC={s['avg_ic']:.4f} minIC={s['min_ic']:.4f} HO={s['ho_acc']:.1%} trades={s['ho_trades']} windows={s['windows']}")
    else:
        print("  None found.")

    print(f"\nNEAR MATCHES (IC>0.02, HO>50%, IC>0 ALL windows):")
    near_found.sort(key=lambda x: x["avg_ic"], reverse=True)
    for s in near_found[:15]:
        print(f"  {s['pair']:>8} {s['signal']:>10} {s['fwd']:>6} IC={s['avg_ic']:.4f} minIC={s['min_ic']:.4f} HO={s['ho_acc']:.1%} trades={s['ho_trades']} win={'Y' if s['ho_acc']>0.52 else 'N'}")

    if strict_found:
        print(f"\nGOAL ACHIEVED: {len(strict_found)} strict signals found")
    else:
        print(f"\nGOAL NOT MET: 0 strict signals. {len(near_found)} near-misses.")
        if near_found:
            best = near_found[0]
            print(f"Best candidate: {best['pair']} {best['signal']}→{best['fwd']} IC={best['avg_ic']:.4f} HO={best['ho_acc']:.1%}")
            print(f"Gap: IC needs +{0.03-best['avg_ic']:.4f}, HO needs +{0.52-best['ho_acc']:.1%}")


if __name__ == "__main__":
    main()
