"""Optuna + LightGBM стратегия с оптимизацией гиперпараметров.

Идеи взяты из run_hybrid_optimizer.py и backtest_model.py:
- Optuna для подбора гиперпараметров
- Walk-forward CV + Holdout
- Feature importance (на лучшей модели)
- Метрики: DA, F1
- LGBMClassifier (не регрессор)
- Holdout: ретрейн на всех данных
"""

import sys, io, json, time, warnings
import pandas as pd, numpy as np
from pathlib import Path
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path("data/top5_2026")
RESULTS_FILE = Path("alpha_research/strategies.json")

WF_TRAIN = 30 * 24 * 3600000  # 30 дней
WF_TEST = 7 * 24 * 3600000    # 7 дней
HOLDOUT_DAYS = 14
N_TRIALS = 30

FEATURES = [
    "basis_z", "rsi", "vol_rank", "vol_ratio", "mom_6h", "mom_24h",
    "atr_ratio", "bb_position", "volume_z", "price_vs_ma",
]


def load_all():
    files = list(DATA_DIR.glob("*_1h.parquet"))
    bases = sorted(set(f.stem.replace("_1h", "").replace("_USDT", "") for f in files) - {"USDC"})
    all_dfs = []
    for base in bases:
        sf = DATA_DIR / f"{base}_USDT_1h.parquet"
        pf = DATA_DIR / f"{base}_USDT_USDT_1h.parquet"
        if not sf.exists() or not pf.exists():
            continue
        s = pd.read_parquet(sf).rename(columns={"close": "spot", "high": "high_", "low": "low_", "volume": "vol_"})
        p = pd.read_parquet(pf)[["timestamp", "close"]].rename(columns={"close": "perp"})
        df = s[["timestamp", "spot", "high_", "low_", "vol_"]].merge(p, on="timestamp", how="inner")
        df["base"] = base
        df["returns"] = df["spot"].pct_change()
        df["basis"] = (df["perp"] - df["spot"]) / df["spot"]
        all_dfs.append(df.sort_values("timestamp").reset_index(drop=True))
    print(f"Загружено {len(all_dfs)} пар")
    return all_dfs


def add_features(df):
    df = df.copy()
    df["basis_z"] = (df["basis"] - df["basis"].rolling(168).mean()) / df["basis"].rolling(168).std()
    d = df["spot"].diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean().replace(0, 1)
    df["rsi"] = 100 - (100 / (1 + g / l))
    df["vol"] = df["returns"].rolling(24).std() * np.sqrt(8760)
    df["vol_rank"] = df["vol"].rolling(168).rank(pct=True)
    df["vol_ratio"] = df["vol_"] / df["vol_"].rolling(24).mean()
    df["mom_6h"] = df["spot"].pct_change(6)
    df["mom_24h"] = df["spot"].pct_change(24)
    tr = pd.concat([
        df["high_"] - df["low_"],
        (df["high_"] - df["spot"].shift(1)).abs(),
        (df["low_"] - df["spot"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr14"] / df["spot"]
    ma = df["spot"].rolling(24).mean()
    std = df["spot"].rolling(24).std()
    df["bb_position"] = (df["spot"] - (ma - 2 * std)) / (4 * std)
    df["volume_z"] = (df["vol_"] - df["vol_"].rolling(24).mean()) / df["vol_"].rolling(24).std()
    df["price_vs_ma"] = df["spot"] / ma - 1
    df["fwd_12h_dir"] = np.sign(df["spot"].shift(-12) / df["spot"] - 1)
    df["target"] = df["fwd_12h_dir"]
    return df


def prepare_data(all_dfs):
    dfs = [add_features(df) for df in all_dfs]
    combined = pd.concat(dfs, ignore_index=True).dropna(subset=FEATURES + ["target"])
    combined = combined[combined["target"] != 0]
    print(f"Всего сэмплов: {len(combined)}")
    return combined


def walk_forward_evaluate(combined, params, entry_threshold=0.6):
    """Walk-forward оценка с заданными параметрами."""
    start = combined["timestamp"].iloc[0]
    end = combined["timestamp"].iloc[-1]
    all_trades = []
    cursor = start

    while cursor + WF_TRAIN + WF_TEST <= end:
        train = combined[(combined["timestamp"] >= cursor) & (combined["timestamp"] < cursor + WF_TRAIN)]
        test = combined[(combined["timestamp"] >= cursor + WF_TRAIN) & (combined["timestamp"] < cursor + WF_TRAIN + WF_TEST)]

        if len(train) < 500 or len(test) < 50:
            cursor += WF_TEST
            continue

        X_train = train[FEATURES].values
        y_train = train["target"].values.astype(int)
        X_test = test[FEATURES].values
        y_test = test["target"].values.astype(int)

        model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])

        proba = model.predict_proba(X_test, num_iteration=model.best_iteration_)
        # proba[:, 1] = P(class=1), proba[:, -1] = P(class=-1)
        pred_class = model.predict(X_test, num_iteration=model.best_iteration_)

        # entry_threshold: насколько уверенно предсказание должно быть
        max_proba = proba.max(axis=1)
        confident = max_proba > entry_threshold
        pred_dir = np.where(confident, pred_class, 0)

        for i in range(len(test)):
            if pred_dir[i] != 0:
                all_trades.append({
                    "correct": 1 if pred_dir[i] == y_test[i] else 0,
                    "pred": pred_dir[i],
                    "actual": y_test[i],
                    "confidence": max_proba[i],
                })
        cursor += WF_TEST

    if not all_trades:
        return None, None

    y_true = np.array([t["actual"] for t in all_trades])
    y_pred = np.array([t["pred"] for t in all_trades])
    da = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    trades_count = len(all_trades)
    accuracy = sum(t["correct"] for t in all_trades) / trades_count

    metrics = {"DA": da, "F1": f1, "trades": trades_count, "accuracy": accuracy}
    return metrics, model


def holdout_test(combined, params, entry_threshold=0.6):
    """Holdout тест: ретрейн на всех данных кроме holdout, потом тест."""
    cutoff = combined["timestamp"].iloc[-1] - HOLDOUT_DAYS * 24 * 3600000
    train = combined[combined["timestamp"] < cutoff]
    test = combined[combined["timestamp"] >= cutoff].copy()
    if len(train) < 500 or len(test) < 50:
        return None, None

    X_train = train[FEATURES].values
    y_train = train["target"].values.astype(int)
    X_test = test[FEATURES].values
    y_test = test["target"].values.astype(int)

    model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])

    proba = model.predict_proba(X_test, num_iteration=model.best_iteration_)
    pred_class = model.predict(X_test, num_iteration=model.best_iteration_)
    max_proba = proba.max(axis=1)
    confident = max_proba > entry_threshold
    pred_dir = np.where(confident, pred_class, 0)

    trades = []
    for i in range(len(test)):
        if pred_dir[i] != 0:
            trades.append({"correct": 1 if pred_dir[i] == y_test[i] else 0, "pred": pred_dir[i], "actual": y_test[i]})

    if not trades:
        return None, model

    y_true = np.array([t["actual"] for t in trades])
    y_pred = np.array([t["pred"] for t in trades])
    da = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    metrics = {"DA": da, "F1": f1, "trades": len(trades), "accuracy": sum(t["correct"] for t in trades) / len(trades)}
    return metrics, model


def objective(trial, combined):
    """Optuna objective — максимизируем DA на walk-forward."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    threshold = trial.suggest_float("threshold", 0.55, 0.75)

    metrics, _ = walk_forward_evaluate(combined, params, entry_threshold=threshold)
    if metrics is None:
        return -999

    return metrics["DA"]


def main():
    t0 = time.time()
    print("OPTUNA + LIGHTGBM Strategy Search")
    print("=" * 80)

    all_dfs = load_all()
    combined = prepare_data(all_dfs)

    # Optuna оптимизация
    print(f"\nЗапуск Optuna ({N_TRIALS} trials)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name="lgbm_optuna"
    )
    study.optimize(lambda trial: objective(trial, combined), n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nЛучший DA: {study.best_value:.4f}")
    print(f"Лучшие параметры: {study.best_params}")

    # Финальная оценка
    best_params = {k: v for k, v in study.best_params.items() if k != "threshold"}
    best_threshold = study.best_params["threshold"]

    wf_metrics, wf_model = walk_forward_evaluate(combined, best_params, entry_threshold=best_threshold)
    ho_metrics, ho_model = holdout_test(combined, best_params, entry_threshold=best_threshold)

    if wf_metrics:
        print(f"\nWalk-Forward: DA={wf_metrics['DA']:.1%} F1={wf_metrics['F1']:.3f} trades={wf_metrics['trades']}")
    if ho_metrics:
        print(f"Holdout:      DA={ho_metrics['DA']:.1%} F1={ho_metrics['F1']:.3f} trades={ho_metrics['trades']}")

    # Feature importance на лучшей модели (обученной на всех данных)
    if ho_model:
        fi = pd.DataFrame({
            "feature": FEATURES,
            "importance": ho_model.feature_importances_
        }).sort_values("importance", ascending=False)
        print("\nFeature Importance (holdout model):")
        print(fi.to_string(index=False))

    # Сохранение
    if wf_metrics and ho_metrics and ho_metrics["accuracy"] > 0.52:
        strategy = {
            "name": "LGBM_Optuna_12h",
            "class": "ml_lgbm_optuna",
            "pairs": ["ALL"],
            "params": {**best_params, "threshold": best_threshold, "features": FEATURES},
            "walk_forward": {k: round(v, 4) if isinstance(v, float) else v for k, v in wf_metrics.items()},
            "holdout": {k: round(v, 4) if isinstance(v, float) else v for k, v in ho_metrics.items()},
            "optuna_best_da": round(study.best_value, 4),
        }

        existing = []
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                existing = json.load(f)
        existing.append(strategy)
        with open(RESULTS_FILE, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"\nСохранено в {RESULTS_FILE}")
    else:
        print(f"\nGOAL NOT MET: best HO accuracy={ho_metrics['accuracy']:.1%}" if ho_metrics else "\nGOAL NOT MET: no holdout data")

    print(f"\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
