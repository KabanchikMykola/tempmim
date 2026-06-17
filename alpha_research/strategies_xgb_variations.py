"""XGBoost variations: different horizons, features, params."""

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

FEATURES_CORE = ["basis_z", "rsi", "vol_rank", "vol_ratio", "mom_6h", "mom_24h", "atr_ratio", "bb_position", "volume_z", "price_vs_ma"]


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
        df["base"] = base; df["returns"] = df["spot"].pct_change()
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
    return df


def run_xgb(combined, features, target_col, threshold=0.7):
    start = combined["timestamp"].iloc[0]; end = combined["timestamp"].iloc[-1]
    trades = []; cursor = start; last_model = None
    while cursor + WF_TRAIN + WF_TEST <= end:
        train = combined[(combined["timestamp"] >= cursor) & (combined["timestamp"] < cursor + WF_TRAIN)]
        test = combined[(combined["timestamp"] >= cursor + WF_TRAIN) & (combined["timestamp"] < cursor + WF_TRAIN + WF_TEST)]
        if len(train) < 1000 or len(test) < 100:
            cursor += WF_TEST; continue
        X_tr = train[features].values; y_tr = ((train[target_col].values + 1) / 2).astype(int)
        X_te = test[features].values; y_te = test[target_col].values
        model = XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        preds = np.where(proba > threshold, 1, np.where(proba < (1-threshold), -1, 0))
        for i in range(len(test)):
            if preds[i] != 0:
                trades.append(1 if preds[i] == y_te[i] else 0)
        last_model = model
        cursor += WF_TEST

    cutoff = combined["timestamp"].iloc[-1] - HOLDOUT_DAYS * 24 * 3600000
    ho = combined[combined["timestamp"] >= cutoff]
    if len(ho) < 50 or not last_model: return None
    X_ho = ho[features].values; y_ho = ho[target_col].values
    proba = last_model.predict_proba(X_ho)[:, 1]
    preds = np.where(proba > threshold, 1, np.where(proba < (1-threshold), -1, 0))
    ho_trades = [1 if preds[i] == y_ho[i] else 0 for i in range(len(ho)) if preds[i] != 0]
    if not ho_trades: return None
    return {"wf_acc": sum(trades)/len(trades) if trades else 0, "wf_trades": len(trades),
            "ho_acc": sum(ho_trades)/len(ho_trades), "ho_trades": len(ho_trades)}


def main():
    t0 = time.time()
    all_dfs = load_all()
    combined = pd.concat([add_features(d) for d in all_dfs], ignore_index=True).dropna(subset=FEATURES_CORE)
    combined = combined[combined["fwd_12h_dir"] != 0]
    print(f"Samples: {len(combined)}")

    configs = [
        ("XGB_6h", FEATURES_CORE, "fwd_6h_dir", 0.70),
        ("XGB_24h", FEATURES_CORE, "fwd_24h_dir", 0.70),
        ("XGB_12h_thr0.65", FEATURES_CORE, "fwd_12h_dir", 0.65),
        ("XGB_no_basis", [f for f in FEATURES_CORE if f != "basis_z"], "fwd_12h_dir", 0.70),
        ("XGB_momentum_only", ["mom_6h", "mom_24h", "vol_rank"], "fwd_12h_dir", 0.70),
    ]

    found = []
    for name, feats, target, thr in configs:
        r = run_xgb(combined, feats, target, thr)
        if r:
            status = "PASS" if r["ho_acc"] > 0.52 and r["ho_trades"] >= 15 else "FAIL"
            print(f"{name}: {status} | WF={r['wf_acc']:.1%}({r['wf_trades']}) HO={r['ho_acc']:.1%}({r['ho_trades']})")
            if status == "PASS":
                found.append({"name": name, "features": feats, "target": target, "threshold": thr, **r})

    print(f"\n({time.time()-t0:.0f}s) Found: {len(found)}")
    for f in found:
        print(f"  {f['name']}: WF={f['wf_acc']:.1%} HO={f['ho_acc']:.1%} trades={f['ho_trades']}")


if __name__ == "__main__":
    main()
