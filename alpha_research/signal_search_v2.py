"""Signal Search v2 — composite signals."""

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


def load_all() -> dict[str, pd.DataFrame]:
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
            columns={"close": "spot", "high": "spot_high", "low": "spot_low", "volume": "spot_vol"}
        ).merge(perp[["timestamp", "close"]].rename(columns={"close": "perp"}), on="timestamp", how="inner")
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["returns"] = df["spot"].pct_change()
        df["basis_pct"] = ((df["perp"] - df["spot"]) / df["spot"]) * 100
        data[base] = df.sort_values("timestamp").reset_index(drop=True)
    return data


def prepare(df: pd.DataFrame) -> pd.DataFrame:
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

    df["vol_ma"] = df["spot_vol"].rolling(24).mean()
    df["vol_ratio"] = df["spot_vol"] / df["vol_ma"]

    df["ma24"] = df["spot"].rolling(24).mean()
    df["mom_6h"] = df["spot"].pct_change(6)

    tr = pd.concat([
        df["spot_high"] - df["spot_low"],
        (df["spot_high"] - df["spot"].shift(1)).abs(),
        (df["spot_low"] - df["spot"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    df["fwd_6h"] = df["spot"].shift(-6) / df["spot"] - 1
    df["fwd_12h"] = df["spot"].shift(-12) / df["spot"] - 1
    df["fwd_24h"] = df["spot"].shift(-24) / df["spot"] - 1

    return df


def composite_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sig_basis_vol"] = df["basis_z"] * df["vol_rank"]
    df["sig_rsi_vol"] = (50 - df["rsi"]) / 50 * df["vol_rank"]
    df["sig_reversion"] = np.where(
        df["vol_rank"] > 0.75,
        np.where(df["rsi"] < 30, 1, np.where(df["rsi"] > 70, -1, 0)),
        0
    )
    df["sig_basis_reversion"] = np.where(
        (df["vol_rank"] > 0.75) & (df["basis_z"].abs() > 1),
        -np.sign(df["basis_z"]),
        0
    )
    df["sig_combined"] = (
        0.3 * (df["basis_z"] / df["basis_z"].rolling(168).std()) +
        0.3 * ((50 - df["rsi"]) / 50) +
        0.2 * (df["vol_rank"] - 0.5) * 2 +
        0.2 * df["mom_6h"] * 100
    )
    df["sig_mean_rev_high_vol"] = np.where(
        df["vol_rank"] > 0.7,
        -(df["basis_z"].fillna(0) * 0.5 + (df["rsi"] - 50) / 50 * 0.5),
        0
    )
    df["sig_basis_rsi_contra"] = np.where(
        (df["basis_z"] < -1.5) & (df["rsi"] < 35), 1,
        np.where(
            (df["basis_z"] > 1.5) & (df["rsi"] > 65), -1, 0
        )
    )

    return df


def evaluate(df: pd.DataFrame, sig_col: str, fwd_col: str) -> dict:
    df_v = df.dropna(subset=[sig_col, fwd_col]).copy()
    df_v = df_v[df_v[sig_col] != 0]
    if len(df_v) < 500:
        return {"ic": 0, "hit": 0, "windows": 0, "stable": False, "ho_acc": 0, "ho_trades": 0}

    ic_series = []
    hits = []
    start = df_v["timestamp"].iloc[0]
    end = df_v["timestamp"].iloc[-1]
    cursor = start

    while cursor + WF_TRAIN + WF_TEST <= end:
        t_s = cursor + WF_TRAIN
        t_e = t_s + WF_TEST
        chunk = df_v[(df_v["timestamp"] >= t_s) & (df_v["timestamp"] < t_e)]
        if len(chunk) >= 12:
            ic = chunk[sig_col].corr(chunk[fwd_col])
            if not np.isnan(ic):
                ic_series.append(ic)
            preds = np.sign(chunk[sig_col])
            actuals = np.sign(chunk[fwd_col])
            valid = ~(preds.isna() | actuals.isna())
            if valid.sum() > 3:
                hits.append((preds[valid] == actuals[valid]).mean())
        cursor += WF_TEST

    if not ic_series:
        return {"ic": 0, "hit": 0, "windows": 0, "stable": False, "ho_acc": 0, "ho_trades": 0}

    cutoff = df_v["timestamp"].iloc[-1] - HOLDOUT_DAYS * 24 * 3600000
    ho = df_v[df_v["timestamp"] >= cutoff]
    if len(ho) >= 10:
        preds = np.sign(ho[sig_col])
        actuals = np.sign(ho[fwd_col])
        valid = ~(preds.isna() | actuals.isna())
        ho_acc = (preds[valid] == actuals[valid]).mean() if valid.sum() > 5 else 0
        ho_trades = int(valid.sum())
    else:
        ho_acc = 0
        ho_trades = 0

    return {
        "ic": np.mean(ic_series),
        "hit": np.mean(hits) if hits else 0,
        "windows": len(ic_series),
        "stable": all(ic > 0 for ic in ic_series),
        "ho_acc": ho_acc,
        "ho_trades": ho_trades,
    }


def main():
    print("GOAL MODE: Composite Signal Search v2")
    print("=" * 80)

    data = load_all()
    print(f"Loaded {len(data)} pairs")

    prepared = {}
    for base, df in data.items():
        prepared[base] = composite_signals(prepare(df))

    hypotheses = [
        ("basis_z * vol_rank", "sig_basis_vol", "fwd_12h"),
        ("rsi_vol_signal", "sig_rsi_vol", "fwd_12h"),
        ("high_vol_reversion", "sig_reversion", "fwd_12h"),
        ("basis_reversion", "sig_basis_reversion", "fwd_12h"),
        ("combined_4factor", "sig_combined", "fwd_12h"),
        ("mean_rev_high_vol", "sig_mean_rev_high_vol", "fwd_12h"),
        ("basis_rsi_contra", "sig_basis_rsi_contra", "fwd_12h"),
        ("basis_z * vol_rank → 6h", "sig_basis_vol", "fwd_6h"),
        ("combined_4factor → 6h", "sig_combined", "fwd_6h"),
        ("high_vol_reversion → 6h", "sig_reversion", "fwd_6h"),
        ("basis_rsi_contra → 6h", "sig_basis_rsi_contra", "fwd_6h"),
    ]

    found = []
    for name, sig, fwd in hypotheses:
        pair_results = []
        for base, df in prepared.items():
            r = evaluate(df, sig, fwd)
            if r["windows"] >= 3:
                pair_results.append({"base": base, **r})

        if not pair_results:
            print(f"{name}: no data")
            continue

        avg_ic = np.mean([r["ic"] for r in pair_results])
        avg_ho = np.mean([r["ho_acc"] for r in pair_results if r["ho_trades"] >= 5])
        stable = sum(1 for r in pair_results if r["stable"])
        total = len(pair_results)

        passed = avg_ic > 0.02 and avg_ho > 0.52 and stable >= total * 0.5
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status} | IC={avg_ic:.4f} | HO={avg_ho:.1%} | Stable={stable}/{total}")

        if passed:
            found.append({"name": name, "ic": avg_ic, "ho": avg_ho, "stable": f"{stable}/{total}"})
            print(f"  >>> SIGNAL #{len(found)} FOUND")

        if len(found) >= 3:
            break

    print(f"\n{'=' * 80}")
    print(f"RESULT: {len(found)}/3 signals")
    for i, s in enumerate(found, 1):
        print(f"  {i}. {s['name']} | IC={s['ic']:.4f} | HO={s['ho']:.1%} | {s['stable']}")


if __name__ == "__main__":
    main()
