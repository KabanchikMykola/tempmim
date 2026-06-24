"""Goal Mode: Search for stable alpha signals."""

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
WF_TRAIN_DAYS = 30
WF_TEST_DAYS = 7


def load_all() -> dict[str, pd.DataFrame]:
    """Load all pairs with indicators."""
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
        df = spot[["timestamp", "open", "close", "high", "low", "volume"]].rename(
            columns={"close": "spot", "high": "spot_high", "low": "spot_low", "open": "spot_open", "volume": "spot_vol"}
        ).merge(
            perp[["timestamp", "close", "volume"]].rename(columns={"close": "perp", "volume": "perp_vol"}),
            on="timestamp", how="inner"
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["returns"] = df["spot"].pct_change()
        df["basis_pct"] = ((df["perp"] - df["spot"]) / df["spot"]) * 100
        data[base] = df.sort_values("timestamp").reset_index(drop=True)
    return data


def add_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
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

    df["vol_ma"] = df["spot_vol"].rolling(24).mean()
    df["vol_ratio"] = df["spot_vol"] / df["vol_ma"]

    df["mom_6h"] = df["spot"].pct_change(6)
    df["mom_24h"] = df["spot"].pct_change(24)

    df["high_24h"] = df["spot_high"].rolling(24).max()
    df["low_24h"] = df["spot_low"].rolling(24).min()
    df["range_pct"] = (df["high_24h"] - df["low_24h"]) / df["spot"]

    df["fwd_6h"] = df["spot"].shift(-6) / df["spot"] - 1
    df["fwd_12h"] = df["spot"].shift(-12) / df["spot"] - 1
    df["fwd_24h"] = df["spot"].shift(-24) / df["spot"] - 1
    df["fwd_6h_dir"] = np.sign(df["fwd_6h"])
    df["fwd_12h_dir"] = np.sign(df["fwd_12h"])
    df["fwd_24h_dir"] = np.sign(df["fwd_24h"])

    return df


def evaluate_signal_on_windows(df: pd.DataFrame, signal_col: str, fwd_col: str) -> dict:
    """Walk-forward evaluation of a signal."""
    df_valid = df.dropna(subset=[signal_col, fwd_col]).copy()
    if len(df_valid) < 500:
        return {"ic": 0, "hit_rate": 0, "windows": 0, "ic_all_positive": False, "ic_series": []}

    train_ms = WF_TRAIN_DAYS * 24 * 3600000
    test_ms = WF_TEST_DAYS * 24 * 3600000
    start = df_valid["timestamp"].iloc[0]
    end = df_valid["timestamp"].iloc[-1]

    ic_series = []
    hit_rates = []
    cursor = start

    while cursor + train_ms + test_ms <= end:
        test_start = cursor + train_ms
        test_end = test_start + test_ms
        test_df = df_valid[(df_valid["timestamp"] >= test_start) & (df_valid["timestamp"] < test_end)]

        if len(test_df) >= 24:
            ic = test_df[signal_col].corr(test_df[fwd_col])
            if not np.isnan(ic):
                ic_series.append(ic)
            preds = np.sign(test_df[signal_col])
            actuals = np.sign(test_df[fwd_col])
            valid = ~(preds.isna() | actuals.isna() | (preds == 0))
            if valid.sum() > 5:
                hr = (preds[valid] == actuals[valid]).mean()
                hit_rates.append(hr)

        cursor += test_ms

    if not ic_series:
        return {"ic": 0, "hit_rate": 0, "windows": 0, "ic_all_positive": False, "ic_series": []}

    return {
        "ic": np.mean(ic_series),
        "hit_rate": np.mean(hit_rates) if hit_rates else 0,
        "windows": len(ic_series),
        "ic_all_positive": all(ic > 0 for ic in ic_series),
        "ic_series": ic_series,
    }


def evaluate_holdout(df: pd.DataFrame, signal_col: str, fwd_col: str) -> dict:
    """Evaluate signal on last 14 days."""
    cutoff = df["timestamp"].iloc[-1] - HOLDOUT_DAYS * 24 * 3600000
    test = df[df["timestamp"] >= cutoff].dropna(subset=[signal_col, fwd_col])

    if len(test) < 10:
        return {"accuracy": 0, "trades": 0}

    preds = np.sign(test[signal_col])
    actuals = np.sign(test[fwd_col])
    valid = ~(preds.isna() | actuals.isna() | (preds == 0))

    if valid.sum() < 5:
        return {"accuracy": 0, "trades": int(valid.sum())}

    accuracy = (preds[valid] == actuals[valid]).mean()
    return {"accuracy": accuracy, "trades": int(valid.sum())}


def test_signal(name: str, signal_col: str, fwd_col: str, data: dict) -> dict:
    """Test one signal across all pairs."""
    results = []
    for base, df in data.items():
        df = add_base_indicators(df)
        if signal_col not in df.columns:
            continue
        wf = evaluate_signal_on_windows(df, signal_col, fwd_col)
        ho = evaluate_holdout(df, signal_col, fwd_col)
        results.append({
            "symbol": base,
            "ic": wf["ic"],
            "hit_rate": wf["hit_rate"],
            "windows": wf["windows"],
            "ic_positive": wf["ic_all_positive"],
            "ho_accuracy": ho["accuracy"],
            "ho_trades": ho["trades"],
        })

    if not results:
        return {"name": name, "passed": False, "reason": "no data"}

    avg_ic = np.mean([r["ic"] for r in results])
    avg_hit = np.mean([r["hit_rate"] for r in results if r["hit_rate"] > 0])
    avg_ho_acc = np.mean([r["ho_accuracy"] for r in results if r["ho_trades"] >= 5])
    stable_pairs = sum(1 for r in results if r["ic_positive"])
    min_ho_trades = min(r["ho_trades"] for r in results) if results else 0

    passed = avg_ic > 0.02 and avg_ho_acc > 0.52 and stable_pairs >= len(results) * 0.5

    return {
        "name": name,
        "passed": passed,
        "avg_ic": round(avg_ic, 4),
        "avg_hit_rate": round(avg_hit, 4),
        "avg_ho_accuracy": round(avg_ho_acc, 4),
        "stable_pairs": f"{stable_pairs}/{len(results)}",
        "details": results,
    }


def main():
    print("GOAL MODE: Signal Search")
    print("=" * 80)

    data = load_all()
    print(f"Loaded {len(data)} pairs")

    hypotheses = [
        ("Basis Z → 24h return", "basis_z", "fwd_24h"),
        ("Basis Z → 12h return", "basis_z", "fwd_12h"),
        ("Basis Z → 6h return", "basis_z", "fwd_6h"),
        ("Volume spike → 12h reversal", "vol_ratio", "fwd_12h"),
        ("Volume spike → 6h reversal", "vol_ratio", "fwd_6h"),
        ("RSI extreme → 12h return", "rsi", "fwd_12h"),
        ("RSI extreme → 6h return", "rsi", "fwd_6h"),
        ("Vol regime → 24h return", "vol_rank", "fwd_24h"),
        ("Momentum 6h → 6h return", "mom_6h", "fwd_6h"),
        ("Momentum 24h → 24h return", "mom_24h", "fwd_24h"),
        ("Vol compression → 12h return", "range_pct", "fwd_12h"),
        ("BTC mom → alt return (lagged)", None, None),  # special
    ]

    found_signals = []

    for name, sig_col, fwd_col in hypotheses:
        print(f"\nTesting: {name}")
        if name == "BTC mom → alt return (lagged)":
            result = test_btc_lead(data)
        else:
            result = test_signal(name, sig_col, fwd_col, data)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {status} | IC={result.get('avg_ic', 0):.4f} | Hit={result.get('avg_hit_rate', 0):.1%} | HO_Acc={result.get('avg_ho_accuracy', 0):.1%} | Stable={result.get('stable_pairs', '0/0')}")

        if result["passed"]:
            found_signals.append(result)
            print(f"  >>> SIGNAL #{len(found_signals)} FOUND: {name}")

        if len(found_signals) >= 3:
            break

    print(f"\n{'=' * 80}")
    print(f"GOAL STATUS: {len(found_signals)}/3 signals found")
    if found_signals:
        for i, s in enumerate(found_signals, 1):
            print(f"  {i}. {s['name']} | IC={s['avg_ic']:.4f} | HO={s['avg_ho_accuracy']:.1%}")


def test_btc_lead(data: dict) -> dict:
    """BTC return at t → altcoin return at t+6h."""
    if "BTC" not in data:
        return {"name": "BTC lead", "passed": False, "reason": "no BTC"}

    btc = add_base_indicators(data["BTC"])[["timestamp", "returns"]].rename(columns={"returns": "btc_ret"})
    results = []

    for base, df in data.items():
        if base == "BTC":
            continue
        df = add_base_indicators(df)
        merged = df.merge(btc, on="timestamp", how="inner")
        merged["btc_ret_lag6"] = merged["btc_ret"].shift(6)
        merged["fwd_6h"] = merged["spot"].shift(-6) / merged["spot"] - 1
        merged = merged.dropna(subset=["btc_ret_lag6", "fwd_6h"])

        if len(merged) < 500:
            continue

        train_ms = WF_TRAIN_DAYS * 24 * 3600000
        test_ms = WF_TEST_DAYS * 24 * 3600000
        start = merged["timestamp"].iloc[0]
        end = merged["timestamp"].iloc[-1]
        ic_series = []
        cursor = start

        while cursor + train_ms + test_ms <= end:
            t_start = cursor + train_ms
            t_end = t_start + test_ms
            chunk = merged[(merged["timestamp"] >= t_start) & (merged["timestamp"] < t_end)]
            if len(chunk) >= 24:
                ic = chunk["btc_ret_lag6"].corr(chunk["fwd_6h"])
                if not np.isnan(ic):
                    ic_series.append(ic)
            cursor += test_ms

        if ic_series:
            ho_cutoff = merged["timestamp"].iloc[-1] - HOLDOUT_DAYS * 24 * 3600000
            ho = merged[merged["timestamp"] >= ho_cutoff]
            if len(ho) >= 10:
                preds = np.sign(ho["btc_ret_lag6"])
                actuals = np.sign(ho["fwd_6h"])
                valid = ~(preds.isna() | actuals.isna() | (preds == 0))
                ho_acc = (preds[valid] == actuals[valid]).mean() if valid.sum() > 5 else 0
            else:
                ho_acc = 0

            results.append({
                "symbol": base,
                "ic": np.mean(ic_series),
                "hit_rate": 0,
                "windows": len(ic_series),
                "ic_positive": all(ic > 0 for ic in ic_series),
                "ho_accuracy": ho_acc,
                "ho_trades": len(ho) if len(ho) >= 10 else 0,
            })

    if not results:
        return {"name": "BTC lead", "passed": False, "reason": "no results"}

    avg_ic = np.mean([r["ic"] for r in results])
    avg_ho = np.mean([r["ho_accuracy"] for r in results if r["ho_trades"] >= 5])
    stable = sum(1 for r in results if r["ic_positive"])
    passed = avg_ic > 0.02 and avg_ho > 0.52 and stable >= len(results) * 0.5

    return {
        "name": "BTC lead",
        "passed": passed,
        "avg_ic": round(avg_ic, 4),
        "avg_hit_rate": 0,
        "avg_ho_accuracy": round(avg_ho, 4),
        "stable_pairs": f"{stable}/{len(results)}",
        "details": results,
    }


if __name__ == "__main__":
    main()
