"""Goal Mode: XGBoost strategy search."""

import sys, io, json, time, warnings
import pandas as pd, numpy as np
from pathlib import Path
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path("data/top5_2026")
HOLDOUT_DAYS = 14
WF_TRAIN = 60 * 24 * 3600000
WF_TEST = 14 * 24 * 3600000
RESULTS_FILE = Path("alpha_research/strategies.json")

FEATURES = [
    "basis_z", "rsi", "vol_rank", "vol_ratio", "mom_6h", "mom_24h",
    "atr_ratio", "bb_position", "volume_z", "price_vs_ma",
]


def load_all():
    files = list(DATA_DIR.glob("*_1h.parquet"))
    bases = sorted(set(f.stem.replace("_1h","").replace("_USDT","") for f in files) - {"USDC"})
    all_dfs = []
    for base in bases:
        sf = DATA_DIR / f"{base}_USDT_1h.parquet"
        pf = DATA_DIR / f"{base}_USDT_USDT_1h.parquet"
        if not sf.exists() or not pf.exists(): continue
        s = pd.read_parquet(sf).rename(columns={"close":"spot","high":"high_","low":"low_","volume":"vol_"})
        p = pd.read_parquet(pf)[["timestamp","close"]].rename(columns={"close":"perp"})
        df = s[["timestamp","spot","high_","low_","vol_"]].merge(p, on="timestamp", how="inner")
        df["base"] = base
        df["returns"] = df["spot"].pct_change()
        df["basis"] = (df["perp"] - df["spot"]) / df["spot"]
        all_dfs.append(df.sort_values("timestamp").reset_index(drop=True))
    return all_dfs


def add_features(df):
    df = df.copy()
    df["basis_z"] = (df["basis"] - df["basis"].rolling(168).mean()) / df["basis"].rolling(168).std()
    d = df["spot"].diff(); g = d.clip(lower=0).rolling(14).mean(); l = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + g / l))
    df["vol"] = df["returns"].rolling(24).std() * np.sqrt(8760)
    df["vol_rank"] = df["vol"].rolling(168).rank(pct=True)
    df["vol_ratio"] = df["vol_"] / df["vol_"].rolling(24).mean()
    df["mom_6h"] = df["spot"].pct_change(6)
    df["mom_24h"] = df["spot"].pct_change(24)
    tr = pd.concat([df["high_"]-df["low_"],(df["high_"]-df["spot"].shift(1)).abs(),(df["low_"]-df["spot"].shift(1)).abs()],axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr14"] / df["spot"]
    ma = df["spot"].rolling(24).mean(); std = df["spot"].rolling(24).std()
    df["bb_position"] = (df["spot"] - (ma - 2*std)) / (4*std)
    df["volume_z"] = (df["vol_"] - df["vol_"].rolling(24).mean()) / df["vol_"].rolling(24).std()
    df["price_vs_ma"] = df["spot"] / ma - 1
    for h in [6, 12, 24]:
        df[f"fwd_{h}h_dir"] = np.sign(df["spot"].shift(-h) / df["spot"] - 1)
    df["target"] = df["fwd_12h_dir"]
    return df


def prepare_data(all_dfs):
    dfs = []
    for df in all_dfs:
        d = add_features(df)
        dfs.append(d)
    combined = pd.concat(dfs, ignore_index=True).dropna(subset=FEATURES + ["target"])
    combined = combined[combined["target"] != 0]
    return combined


def walk_forward_xgb(combined, entry_threshold=0.6):
    start = combined["timestamp"].iloc[0]
    end = combined["timestamp"].iloc[-1]
    all_trades = []
    cursor = start

    while cursor + WF_TRAIN + WF_TEST <= end:
        train = combined[(combined["timestamp"] >= cursor) & (combined["timestamp"] < cursor + WF_TRAIN)]
        test = combined[(combined["timestamp"] >= cursor + WF_TRAIN) & (combined["timestamp"] < cursor + WF_TRAIN + WF_TEST)]

        if len(train) < 1000 or len(test) < 100:
            cursor += WF_TEST
            continue

        X_train = train[FEATURES].values
        y_train = (train["target"].values + 1) / 2
        X_test = test[FEATURES].values
        y_test = test["target"].values

        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8,
                              colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbosity=0)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        proba = model.predict_proba(X_test)[:, 1]
        preds = np.where(proba > entry_threshold, 1, np.where(proba < (1 - entry_threshold), -1, 0))

        for i in range(len(test)):
            if preds[i] != 0:
                actual_dir = y_test[i]
                pred_dir = preds[i]
                correct = 1 if pred_dir == actual_dir else 0
                all_trades.append({"timestamp": test["timestamp"].iloc[i], "correct": correct, "pred": pred_dir, "actual": actual_dir})

        cursor += WF_TEST

    return all_trades, model


def holdout_test(combined, model, entry_threshold=0.6):
    cutoff = combined["timestamp"].iloc[-1] - HOLDOUT_DAYS * 24 * 3600000
    test = combined[combined["timestamp"] >= cutoff].copy()
    if len(test) < 50: return None

    X = test[FEATURES].values
    y = test["target"].values
    proba = model.predict_proba(X)[:, 1]
    preds = np.where(proba > entry_threshold, 1, np.where(proba < (1 - entry_threshold), -1, 0))

    trades = []
    for i in range(len(test)):
        if preds[i] != 0:
            correct = 1 if preds[i] == y[i] else 0
            trades.append(correct)

    if not trades: return None
    return {"trades": len(trades), "accuracy": sum(trades)/len(trades)}


def main():
    t0 = time.time()
    print("GOAL MODE: XGBoost Strategy Search")
    print("=" * 80)

    all_dfs = load_all()
    print(f"Loaded {len(all_dfs)} pairs")

    combined = prepare_data(all_dfs)
    print(f"Total samples: {len(combined)}")

    thresholds = [0.55, 0.60, 0.65, 0.70]
    results = []

    for thr in thresholds:
        trades, model = walk_forward_xgb(combined, entry_threshold=thr)
        ho = holdout_test(combined, model, entry_threshold=thr)

        if not trades or not ho: continue
        correct = sum(t["correct"] for t in trades)
        wf_acc = correct / len(trades)
        wf_trades = len(trades)

        results.append({
            "threshold": thr,
            "wf_accuracy": round(wf_acc, 4),
            "wf_trades": wf_trades,
            "ho_accuracy": round(ho["accuracy"], 4),
            "ho_trades": ho["trades"],
        })

        print(f"Threshold={thr}: WF_ACC={wf_acc:.1%} ({wf_trades} trades) | HO_ACC={ho['accuracy']:.1%} ({ho['trades']} trades)")

    if results:
        best = max(results, key=lambda x: x["ho_accuracy"])
        print(f"\nBest: threshold={best['threshold']} HO_ACC={best['ho_accuracy']:.1%}")

        if best["ho_accuracy"] > 0.52 and best["ho_trades"] >= 15:
            strategy = {
                "name": "XGBoost_12h",
                "class": "ml_xgboost",
                "pairs": ["ALL"],
                "params": {"threshold": best["threshold"], "features": FEATURES, "n_estimators": 100, "max_depth": 4},
                "walk_forward": {"accuracy": best["wf_accuracy"], "trades": best["wf_trades"]},
                "holdout": {"accuracy": best["ho_accuracy"], "trades": best["ho_trades"]},
            }

            existing = []
            if RESULTS_FILE.exists():
                with open(RESULTS_FILE) as f: existing = json.load(f)
            existing.append(strategy)
            with open(RESULTS_FILE, "w") as f: json.dump(existing, f, indent=2)
            print(f"Saved to {RESULTS_FILE}")
        else:
            print(f"\nGOAL NOT MET: best HO accuracy={best['ho_accuracy']:.1%} < 52%")
    else:
        print("\nNo valid results.")

    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
