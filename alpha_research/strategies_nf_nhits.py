"""NHITS (NeuralForecast) стратегия для предсказания направления на 12ч.

Входные данные: OHLCV (spot + perp) + derivatives metrics (OI, long/short).
Модель: NHITS — Neural Hierarchical Interpolation for Time Series.
Walk-forward: 60d train / 14d test (для DL нужно больше данных).

Критические lessons:
  - Все фичи со shift(1) — без lookahead bias
  - Metrics только для BTCUSDT — для остальных NaN → drop
  - Confidence threshold для фильтрации сигналов
  - Per-asset predict: train+test concat для корректного ds

Использование:
    python alpha_research/strategies_nf_nhits.py
    python alpha_research/strategies_nf_nhits.py --symbols BTCUSDT
"""

import sys, io, json, time, warnings, argparse
import pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path("data")
RESULTS_FILE = Path("alpha_research/strategies.json")

WF_TRAIN_MS = 90 * 24 * 3600000   # 90 дней (больше данных для DL)
WF_TEST_MS = 30 * 24 * 3600000    # 30 дней (меньше folds = быстрее)
HOLDOUT_DAYS = 14

BASE_FEATURES = [
    "basis_z", "rsi", "vol_rank", "vol_ratio", "mom_6h", "mom_24h",
    "atr_ratio", "bb_position", "volume_z", "price_vs_ma",
]

METRIC_FEATURES = [
    "oi_change_24h", "oi_z", "ls_ratio_z", "taker_ratio_z",
]

# Все возможные фичи (base + metrics)
ALL_FEATURES = BASE_FEATURES + METRIC_FEATURES


# ── Загрузка данных ──────────────────────────────────────────────


def load_all(symbols=None):
    """Загрузить spot + perp + metrics для всех символов."""
    spot_files = sorted(DATA_DIR.glob("*_1h_spot.parquet"))
    bases = sorted(set(f.stem.replace("_1h_spot", "") for f in spot_files))
    if symbols:
        bases = [b for b in bases if b in symbols]

    all_dfs = []
    for base in bases:
        sf = DATA_DIR / f"{base}_1h_spot.parquet"
        pf = DATA_DIR / f"{base}_1h_perp.parquet"
        mf = DATA_DIR / f"{base}_metrics.parquet"
        if not sf.exists() or not pf.exists():
            continue

        s = pd.read_parquet(sf)
        p = pd.read_parquet(pf)

        if "timestamp" not in s.columns and "ts" in s.columns:
            s = s.rename(columns={"ts": "timestamp"})
        if "timestamp" not in p.columns and "ts" in p.columns:
            p = p.rename(columns={"ts": "timestamp"})

        spot_cols = ["timestamp", "close", "high", "low", "volume"]
        df = s[spot_cols].merge(
            p[["timestamp", "close"]].rename(columns={"close": "perp"}),
            on="timestamp", how="inner",
        )
        df = df.rename(columns={
            "close": "spot", "high": "high_", "low": "low_", "volume": "vol_",
        })
        df["base"] = base
        df["returns"] = df["spot"].pct_change()
        df["basis"] = (df["perp"] - df["spot"]) / df["spot"]

        # Metrics (resample 5m → 1h)
        if mf.exists():
            mdf = pd.read_parquet(mf)
            if "ts" in mdf.columns:
                mdf["timestamp"] = mdf["ts"]
            mdf["timestamp"] = pd.to_datetime(mdf["timestamp"], unit="ms", utc=True)
            mdf = mdf.set_index("timestamp").resample("1h").last().dropna(how="all").reset_index()
            mdf["timestamp"] = mdf["timestamp"].astype("int64") // 10**6
            mcols = [
                "sum_open_interest", "sum_open_interest_value",
                "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
                "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
            ]
            avail = [c for c in mcols if c in mdf.columns]
            df = df.merge(mdf[["timestamp"] + avail], on="timestamp", how="left")
            for c in avail:
                df[c] = df[c].ffill()

        all_dfs.append(df.sort_values("timestamp").reset_index(drop=True))

    print(f"Загружено {len(all_dfs)} пар")
    return all_dfs


# ── Feature engineering ──────────────────────────────────────────


def add_features(df):
    """Добавить TA фичи + metrics фичи. Все со shift(1)."""
    df = df.copy()

    # Basis z-score (basis = perp/spot, оба известны в t — OK без shift)
    df["basis_z"] = (df["basis"] - df["basis"].rolling(168).mean()) / df["basis"].rolling(168).std()

    # RSI — shift(1)
    d = df["spot"].diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean().replace(0, 1)
    df["rsi"] = (100 - (100 / (1 + g / l))).shift(1)

    # Volatility rank — shift(1) на vol, rank на сдвинутых
    df["vol"] = (df["returns"].rolling(24).std() * np.sqrt(8760)).shift(1)
    df["vol_rank"] = df["vol"].rolling(168).rank(pct=True)

    # Volume ratio — shift(1)
    df["vol_ratio"] = (df["vol_"] / df["vol_"].rolling(24).mean()).shift(1)

    # Momentum — shift(1)
    df["mom_6h"] = df["spot"].pct_change(6).shift(1)
    df["mom_24h"] = df["spot"].pct_change(24).shift(1)

    # ATR ratio — shift(1)
    tr = pd.concat([
        df["high_"] - df["low_"],
        (df["high_"] - df["spot"].shift(1)).abs(),
        (df["low_"] - df["spot"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_ratio"] = (tr.rolling(14).mean() / df["spot"]).shift(1)

    # Bollinger — shift(1)
    ma = df["spot"].rolling(24).mean()
    std = df["spot"].rolling(24).std()
    df["bb_position"] = ((df["spot"] - (ma - 2 * std)) / (4 * std)).shift(1)

    # Volume z-score — shift(1)
    df["volume_z"] = ((df["vol_"] - df["vol_"].rolling(24).mean())
                      / df["vol_"].rolling(24).std()).shift(1)

    # Price vs MA — shift(1)
    df["price_vs_ma"] = (df["spot"] / ma - 1).shift(1)

    # Metrics фичи (если есть данные)
    if "sum_open_interest" in df.columns:
        df["oi_change_24h"] = df["sum_open_interest"].pct_change(24).shift(1)
        oi_mean = df["sum_open_interest"].rolling(168).mean()
        oi_std = df["sum_open_interest"].rolling(168).std()
        df["oi_z"] = ((df["sum_open_interest"] - oi_mean) / oi_std).shift(1)
        ls_mean = df["count_long_short_ratio"].rolling(168).mean()
        ls_std = df["count_long_short_ratio"].rolling(168).std()
        df["ls_ratio_z"] = ((df["count_long_short_ratio"] - ls_mean) / ls_std).shift(1)
        tk_mean = df["sum_taker_long_short_vol_ratio"].rolling(168).mean()
        tk_std = df["sum_taker_long_short_vol_ratio"].rolling(168).std()
        df["taker_ratio_z"] = ((df["sum_taker_long_short_vol_ratio"] - tk_mean) / tk_std).shift(1)
    else:
        for c in METRIC_FEATURES:
            df[c] = np.nan

    # Таргет: 12h forward return
    df["fwd_ret_12h"] = df["spot"].shift(-12) / df["spot"] - 1
    df["fwd_dir"] = np.sign(df["fwd_ret_12h"])

    return df


# ── NHITS Walk-Forward ───────────────────────────────────────────


def _prepare_nf_df(df, feature_cols):
    """Подготовить DataFrame для NeuralForecast (unique_id, ds, y + exog).

    ds — per-unique_id монотонный индекс (cumcount), НЕ глобальный range.
    При multi-asset данные конкатенируются, поэтому каждой серии нужна
    своя ось времени [0..N_i]. Иначе NF трактует все активы как одну серию.
    """
    nf = df[["base", "timestamp", "fwd_ret_12h"] + feature_cols].copy()
    nf = nf.rename(columns={"base": "unique_id", "timestamp": "ds", "fwd_ret_12h": "y"})
    nf["ds"] = nf.groupby("unique_id").cumcount()
    return nf


def walk_forward_nhits(all_dfs, h=12, threshold=0.001):
    """Walk-forward оценка NHITS на всех парах.

    Args:
        all_dfs: список DataFrames с фичами
        h: горизонт прогноза (12 часов)
        threshold: мин. |pred_ret| для входа (confidence filter)
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    combined = pd.concat(all_dfs, ignore_index=True)

    # Определяем доступные фичи: только те, где нет сплошных NaN
    feature_cols = BASE_FEATURES.copy()
    for c in METRIC_FEATURES:
        if c in combined.columns and combined[c].notna().sum() > 1000:
            feature_cols.append(c)

    print(f"Фичи ({len(feature_cols)}): {feature_cols}")

    # Удаляем строки с NaN в фичах или таргете
    combined = combined.dropna(subset=feature_cols + ["fwd_ret_12h"])
    combined = combined[combined["fwd_dir"] != 0].copy()
    print(f"Сэмплов после очистки: {len(combined):,}")

    start = combined["timestamp"].iloc[0]
    end = combined["timestamp"].iloc[-1]
    all_preds = []
    cursor = start
    fold = 0

    while cursor + WF_TRAIN_MS + WF_TEST_MS <= end:
        fold += 1
        train_end = cursor + WF_TRAIN_MS
        test_end = train_end + WF_TEST_MS

        train_df = combined[(combined["timestamp"] >= cursor) & (combined["timestamp"] < train_end)]
        test_df = combined[(combined["timestamp"] >= train_end) & (combined["timestamp"] < test_end)]

        if len(train_df) < 500 or len(test_df) < 50:
            cursor += WF_TEST_MS
            continue

        # Подготовка train в NF-формате
        nf_train = _prepare_nf_df(train_df, feature_cols)

        # Модель
        model = NHITS(
            h=h,
            input_size=48,
            max_steps=50,
            learning_rate=1e-3,
        )
        nf = NeuralForecast(models=[model], freq=1)

        try:
            nf.fit(nf_train)
        except Exception as e:
            print(f"  Fold {fold}: fit error: {e}")
            cursor += WF_TEST_MS
            continue

        # Per-asset predict: NeuralForecast сам генерирует h-step forecast.
        # НЕ передаём test-данные в predict — это misuse API.
        # predict(df=asset_train) вернёт h=12 прогнозов для этого актива.
        fold_preds = 0
        for asset in test_df["base"].unique():
            asset_train = train_df[train_df["base"] == asset]
            asset_test = test_df[test_df["base"] == asset]

            if len(asset_test) < h:
                continue

            nf_asset_train = _prepare_nf_df(asset_train, feature_cols)

            try:
                forecast = nf.predict(df=nf_asset_train)
            except Exception as e:
                print(f"  Fold {fold}/{asset} predict error: {e}")
                continue

            if forecast is None or forecast.empty:
                continue

            pred_col = [c for c in forecast.columns if c not in ("unique_id", "ds")]
            if not pred_col:
                continue

            # Используем ВСЕ прогнозы (не только последний)
            pred_vals = forecast[pred_col[0]].values
            actual_vals = asset_test["fwd_ret_12h"].values[:len(pred_vals)]

            for i in range(min(len(pred_vals), len(actual_vals))):
                pred_val = pred_vals[i]
                actual_ret = actual_vals[i]
                actual_dir = np.sign(actual_ret)

                if actual_dir == 0:
                    continue

                # Confidence threshold
                if abs(pred_val) < threshold:
                    continue

                pred_dir = 1 if pred_val > 0 else -1
                all_preds.append({
                    "fold": fold,
                    "asset": asset,
                    "pred_dir": pred_dir,
                    "actual_dir": int(actual_dir),
                    "correct": 1 if pred_dir == actual_dir else 0,
                    "pred_ret": float(pred_val),
                    "actual_ret": float(actual_ret),
                })
                fold_preds += 1

        if fold % 5 == 0:
            print(f"  Fold {fold}: {fold_preds} preds this fold", flush=True)
        cursor += WF_TEST_MS

    if not all_preds:
        return None, None

    y_true = np.array([p["actual_dir"] for p in all_preds])
    y_pred = np.array([p["pred_dir"] for p in all_preds])
    da = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    correct = sum(p["correct"] for p in all_preds)
    total = len(all_preds)

    metrics = {
        "DA": round(da, 4),
        "F1": round(f1, 4),
        "trades": total,
        "accuracy": round(correct / total, 4),
        "folds": fold,
    }
    return metrics, model


# ── Holdout ──────────────────────────────────────────────────────


def holdout_nhits(all_dfs, h=12, threshold=0.001):
    """Holdout тест на последних HOLDOUT_DAYS днях."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    combined = pd.concat(all_dfs, ignore_index=True)

    feature_cols = BASE_FEATURES.copy()
    for c in METRIC_FEATURES:
        if c in combined.columns and combined[c].notna().sum() > 1000:
            feature_cols.append(c)

    combined = combined.dropna(subset=feature_cols + ["fwd_ret_12h"])
    combined = combined[combined["fwd_dir"] != 0].copy()

    cutoff = combined["timestamp"].iloc[-1] - HOLDOUT_DAYS * 24 * 3600000
    train_all = combined[combined["timestamp"] < cutoff]
    test_all = combined[combined["timestamp"] >= cutoff]

    if len(train_all) < 500 or len(test_all) < 50:
        return None

    # Обучаем на ВСЕХ данных до cutoff
    nf_train = _prepare_nf_df(train_all, feature_cols)

    ho_model = NHITS(
        h=h, input_size=48, max_steps=50, learning_rate=1e-3,
    )
    nf = NeuralForecast(models=[ho_model], freq=1)

    try:
        nf.fit(nf_train)
    except Exception as e:
        print(f"Holdout fit error: {e}")
        return None

    all_preds = []
    for asset in test_all["base"].unique():
        asset_train = train_all[train_all["base"] == asset]
        asset_test = test_all[test_all["base"] == asset]

        if len(asset_test) < h:
            continue

        nf_asset_train = _prepare_nf_df(asset_train, feature_cols)

        try:
            forecast = nf.predict(df=nf_asset_train)
        except Exception as e:
            print(f"  Holdout/{asset} predict error: {e}")
            continue

        if forecast is None or forecast.empty:
            continue

        pred_col = [c for c in forecast.columns if c not in ("unique_id", "ds")]
        if not pred_col:
            continue

        pred_vals = forecast[pred_col[0]].values
        actual_vals = asset_test["fwd_ret_12h"].values[:len(pred_vals)]

        for i in range(min(len(pred_vals), len(actual_vals))):
            pred_val = pred_vals[i]
            actual_ret = actual_vals[i]
            actual_dir = np.sign(actual_ret)

            if actual_dir == 0 or abs(pred_val) < threshold:
                continue

            pred_dir = 1 if pred_val > 0 else -1
            all_preds.append({"correct": 1 if pred_dir == actual_dir else 0})

    if not all_preds:
        return None

    correct = sum(p["correct"] for p in all_preds)
    return {"trades": len(all_preds), "accuracy": round(correct / len(all_preds), 4)}


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="NHITS strategy search")
    parser.add_argument("--symbols", nargs="+", help="Символы (по умолчанию все)")
    parser.add_argument("--threshold", type=float, default=0.001,
                        help="Мин. |pred_ret| для входа (default: 0.001)")
    args = parser.parse_args()

    t0 = time.time()
    print("NHITS Strategy Search (NeuralForecast)")
    print("=" * 80)

    # Загрузка
    all_dfs = load_all(args.symbols)
    if not all_dfs:
        print("Нет данных!")
        return

    # Feature engineering
    featured = [add_features(df) for df in all_dfs]

    # Walk-forward
    print(f"\nWalk-forward (train={WF_TRAIN_MS // 86400000}d, test={WF_TEST_MS // 86400000}d, "
          f"threshold={args.threshold})...")
    wf_metrics, wf_model = walk_forward_nhits(featured, h=12, threshold=args.threshold)

    if not wf_metrics:
        print("\nWalk-forward: нет данных")
        print(f"\n({time.time() - t0:.0f}s)")
        return

    print(f"\nWalk-Forward: DA={wf_metrics['DA']:.1%} F1={wf_metrics['F1']:.3f} "
          f"trades={wf_metrics['trades']} folds={wf_metrics['folds']}")

    # Holdout
    print(f"\nHoldout (последние {HOLDOUT_DAYS}д)...")
    ho_metrics = holdout_nhits(featured, h=12, threshold=args.threshold)

    if not ho_metrics:
        print("Holdout: нет данных")
        print(f"\n({time.time() - t0:.0f}s)")
        return

    print(f"Holdout:      DA={ho_metrics['accuracy']:.1%} trades={ho_metrics['trades']}")

    # Сохранение
    if ho_metrics["accuracy"] > 0.52 and ho_metrics["trades"] >= 15:
        strategy = {
            "name": "NHITS_12h",
            "class": "nf_nhits",
            "pairs": ["ALL"],
            "params": {
                "h": 12, "input_size": 168, "max_steps": 300,
                "learning_rate": 1e-3, "threshold": args.threshold,
                "features": BASE_FEATURES + METRIC_FEATURES,
            },
            "walk_forward": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in wf_metrics.items()},
            "holdout": {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in ho_metrics.items()},
        }
        existing = []
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                existing = json.load(f)
        existing.append(strategy)
        with open(RESULTS_FILE, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"\n✅ GOAL MET! Сохранено в {RESULTS_FILE}")
    else:
        print(f"\nGOAL NOT MET: HO accuracy={ho_metrics['accuracy']:.1%} < 52% "
              f"или trades={ho_metrics['trades']} < 15")

    print(f"\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
