"""WFO + ML Backtester v3: Walk-Forward Optimization with LightGBM Meta-Labeling.

Architecture:
1. Feature Engineering: Multi-TF momentum, vol, regime indicators
2. Walk-Forward: rolling windows with IS/OOS splits
3. ML Filter: LightGBM meta-labeling on momentum signals
4. Honest Metrics: stitched OOS performance
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from time import time
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("data/binance_4h")
CAPITAL = 10000.0
FEE_PCT = 0.0008  # 8 bps per side
EXCLUDE_SYMBOLS = ["ARB_USDT", "OP_USDT"]


# =============================================================================
# DATA
# =============================================================================

def load_data():
    frames = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        symbol = f.stem.replace("_4h", "")
        if symbol in EXCLUDE_SYMBOLS:
            continue
        frames.append(pd.read_parquet(f))
    df = pd.concat(frames, ignore_index=True)
    close = df.pivot_table(index="datetime", columns="symbol", values="close").sort_index().ffill().bfill()
    vol = df.pivot_table(index="datetime", columns="symbol", values="volume").sort_index().ffill().bfill()
    miss = close.isna().mean()
    close = close[miss[miss < 0.3].index].dropna()
    vol = vol[close.columns].reindex(close.index).ffill().dropna()
    ret = np.log(close / close.shift(1)).iloc[1:]
    vol = vol.iloc[1:]
    return close, ret, vol


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def compute_features(close, ret, vol, lookbacks=[16, 42, 99]):
    """Compute cross-sectional features for each asset at each bar."""
    n_assets = ret.shape[1]
    
    # Build MultiIndex (datetime, symbol) matching stacked returns
    cr = ret.cumsum()
    stacked_idx = cr.stack().index
    
    features = pd.DataFrame(index=stacked_idx, dtype=float)
    
    # Multi-TF momentum signals
    for lb in lookbacks:
        features[f"mom_{lb}"] = cr.rolling(lb).sum().stack()
    
    # Cross-sectional ranks of momentum
    for lb in lookbacks:
        mom = cr.rolling(lb).sum()
        features[f"rank_{lb}"] = mom.rank(axis=1, pct=True).stack()
    
    # Volatility features
    for lb in [20, 60]:
        features[f"vol_{lb}"] = ret.rolling(lb).std().stack()
        features[f"vol_ratio_{lb}"] = (ret.rolling(lb).std() / ret.rolling(lb*3).std()).stack()
    
    # Volume features
    vol_ma = vol.rolling(20).mean()
    features["vol_spike"] = (vol / vol_ma).stack()
    
    # Regime: market-wide momentum
    mkt_ret = ret.mean(axis=1)
    mkt_mom_20 = mkt_ret.rolling(20).sum()
    mkt_mom_60 = mkt_ret.rolling(60).sum()
    # Tile for each asset
    features["mkt_mom_20"] = np.tile(mkt_mom_20.values, n_assets)[:len(features)]
    features["mkt_mom_60"] = np.tile(mkt_mom_60.values, n_assets)[:len(features)]
    
    # Multi-TF composite signal
    w_short, w_long = 0.68, 0.32
    features["multi_tf"] = (w_short * cr.rolling(16).sum() + w_long * cr.rolling(99).sum()).stack()
    
    # Target: next-bar return sign (for ML)
    features["target"] = (ret.shift(-1) > 0).astype(float).stack()
    
    # Forward return for backtest evaluation
    features["fwd_ret"] = ret.shift(-1).stack()
    
    features = features.dropna()
    return features


# =============================================================================
# WALK-FORWARD ENGINE
# =============================================================================

def walk_forward_backtest(close, ret, vol, train_months=6, test_months=2,
                          top_n=7, holding_bars=36, use_ml=True):
    """Walk-Forward Optimization with optional ML filter."""
    
    dates = ret.index
    n = len(dates)
    
    # Window sizes in bars (4h bars: ~6 per day, ~126 per month)
    bars_per_month = 6 * 30
    train_bars = train_months * bars_per_month
    test_bars = test_months * bars_per_month
    
    all_oos_positions = []
    all_oos_returns = []
    window_results = []
    
    window_start = 0
    window_id = 0
    
    while window_start + train_bars + test_bars <= n:
        train_end = window_start + train_bars
        test_end = min(train_end + test_bars, n)
        
        # IS / OOS masks
        is_mask = np.zeros(n, dtype=bool)
        is_mask[window_start:train_end] = True
        oos_mask = np.zeros(n, dtype=bool)
        oos_mask[train_end:test_end] = True
        
        is_dates = dates[is_mask]
        oos_dates = dates[oos_mask]
        
        # --- Feature engineering ---
        features = compute_features(close, ret, vol)
        
        # Get feature columns (exclude targets)
        feat_cols = [c for c in features.columns if c not in ["target", "fwd_ret"]]
        
        # IS data
        is_features = features.loc[features.index.get_level_values(0).isin(is_dates)]
        # OOS data
        oos_features = features.loc[features.index.get_level_values(0).isin(oos_dates)]
        
        if len(is_features) < 100 or len(oos_features) < 20:
            window_start += test_bars
            continue
        
        # --- ML Training (if enabled) ---
        if use_ml:
            X_is = is_features[feat_cols].fillna(0)
            y_is = is_features["target"]
            
            # Train LightGBM
            dtrain = lgb.Dataset(X_is, label=y_is)
            params = {
                "objective": "binary",
                "metric": "auc",
                "num_leaves": 8,
                "max_depth": 3,
                "learning_rate": 0.05,
                "n_estimators": 100,
                "min_child_samples": 20,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "verbose": -1,
            }
            model = lgb.train(params, dtrain, num_boost_round=100,
                             valid_sets=[dtrain],
                             callbacks=[lgb.early_stopping(20, verbose=False)])
            
            # Predict on OOS — reshape to DataFrame for proper alignment
            X_oos = oos_features[feat_cols].fillna(0)
            preds = model.predict(X_oos)
            ml_preds_df = (
                pd.Series(preds, index=X_oos.index)
                .unstack(level=1)
                .reindex(index=oos_dates, columns=ret.columns)
                .fillna(0.0)
            )
        else:
            ml_preds_df = None
        
        # --- Momentum Signal on OOS ---
        oos_ret_slice = ret.loc[oos_dates]
        cr = ret.cumsum().loc[oos_dates]
        
        # Multi-TF momentum
        mom_s = cr.rolling(16).sum()
        mom_l = cr.rolling(99).sum()
        combined = 0.68 * mom_s + 0.32 * mom_l
        
        # Generate positions
        positions = pd.DataFrame(0.0, index=oos_dates, columns=ret.columns)
        
        for t_idx in range(len(oos_dates)):
            t = oos_dates[t_idx]
            
            if t_idx % holding_bars != 0 and t_idx > 0:
                positions.iloc[t_idx] = positions.iloc[t_idx - 1]
                continue
            
            mom_row = combined.loc[t].values
            valid = ~np.isnan(mom_row)
            
            if valid.sum() < top_n * 2:
                if t_idx > 0:
                    positions.iloc[t_idx] = positions.iloc[t_idx - 1]
                continue
            
            # ML filter: only trade assets where ML probability > threshold
            if use_ml and ml_preds_df is not None:
                ml_row = ml_preds_df.loc[t].values
                valid = valid & (ml_row > 0.48)
            
            if valid.sum() < top_n * 2:
                if t_idx > 0:
                    positions.iloc[t_idx] = positions.iloc[t_idx - 1]
                continue
            
            ranks = np.argsort(np.where(valid, mom_row, -np.inf))
            positions.iloc[t_idx, ranks[-top_n:]] = 1.0 / top_n
            positions.iloc[t_idx, ranks[:top_n]] = -1.0 / top_n
        
        # Shift to avoid look-ahead
        positions = positions.shift(1).fillna(0)
        
        # --- Compute returns ---
        oos_ret_values = ret.loc[oos_dates].values
        pos_values = positions.values
        
        gross_ret = (pos_values * oos_ret_values).sum(axis=1)
        delta = np.abs(np.diff(pos_values, axis=0, prepend=pos_values[:1]))
        costs = FEE_PCT * delta.sum(axis=1) * (CAPITAL / 2)
        net_dollars = gross_ret * CAPITAL - costs
        
        # Window metrics
        sharpe = (net_dollars / CAPITAL).mean() / (net_dollars / CAPITAL).std() * np.sqrt(6 * 365) if (net_dollars / CAPITAL).std() > 0 else 0
        cum_pnl = np.cumsum(net_dollars)
        max_dd = abs((cum_pnl - np.maximum.accumulate(cum_pnl)).min())
        gains = net_dollars[net_dollars > 0].sum()
        losses = abs(net_dollars[net_dollars < 0].sum())
        pf = gains / losses if losses > 0 else 0
        
        window_results.append({
            "window": window_id,
            "train": f"{is_dates[0].strftime('%Y-%m')} → {is_dates[-1].strftime('%Y-%m')}",
            "test": f"{oos_dates[0].strftime('%Y-%m')} → {oos_dates[-1].strftime('%Y-%m')}",
            "sharpe": sharpe,
            "pf": pf,
            "max_dd_pct": max_dd / CAPITAL,
            "pnl": net_dollars.sum(),
            "bars": len(oos_dates),
        })
        
        all_oos_returns.append(pd.Series(net_dollars, index=oos_dates))
        
        print(f"  W{window_id}: {oos_dates[0].strftime('%Y-%m')} → {oos_dates[-1].strftime('%Y-%m')} | "
              f"Sharpe={sharpe:.2f} PF={pf:.2f} MaxDD={max_dd/CAPITAL:.1%} PnL=${net_dollars.sum():.0f}")
        
        window_id += 1
        window_start += test_bars  # sliding window
    
    # --- Stitched OOS results ---
    if all_oos_returns:
        stitched = pd.concat(all_oos_returns)
        stitched = stitched[~stitched.index.duplicated(keep='first')]
        stitched = stitched.sort_index()
        
        net_ret = stitched / CAPITAL
        sharpe = net_ret.mean() / net_ret.std() * np.sqrt(6 * 365) if net_ret.std() > 0 else 0
        cum_pnl = np.cumsum(stitched)
        max_dd = abs((cum_pnl - np.maximum.accumulate(cum_pnl)).min())
        gains = stitched[stitched > 0].sum()
        losses = abs(stitched[stitched < 0].sum())
        pf = gains / losses if losses > 0 else 0
        total_pnl = stitched.sum()
        
        return {
            "sharpe": sharpe,
            "pf": pf,
            "max_dd_pct": max_dd / CAPITAL,
            "pnl": total_pnl,
            "n_windows": window_id,
            "windows": window_results,
            "stitched_returns": stitched,
        }
    
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time()
    print("=" * 70)
    print("WFO + ML BACKTESTER v3")
    print("=" * 70)
    
    print("\n[1] Loading data...")
    close, ret, vol = load_data()
    print(f"  Assets: {ret.shape[1]}, Bars: {len(ret)}")
    print(f"  Period: {ret.index[0]} — {ret.index[-1]}")
    
    # =========================================================================
    # Experiment 1: Pure Momentum (no ML) — Baseline
    # =========================================================================
    print(f"\n{'='*70}")
    print("[2] EXPERIMENT 1: Pure Multi-TF Momentum (no ML)")
    print(f"{'='*70}")
    
    result_pure = walk_forward_backtest(
        close, ret, vol,
        train_months=6, test_months=2,
        top_n=7, holding_bars=36, use_ml=False
    )
    
    if result_pure:
        print(f"\n  Stitched OOS Results:")
        print(f"  Sharpe:  {result_pure['sharpe']:.2f}")
        print(f"  PF:      {result_pure['pf']:.2f}")
        print(f"  MaxDD:   {result_pure['max_dd_pct']:.1%}")
        print(f"  PnL:     ${result_pure['pnl']:.0f}")
        print(f"  Windows: {result_pure['n_windows']}")
    
    # =========================================================================
    # Experiment 2: Momentum + ML Filter
    # =========================================================================
    print(f"\n{'='*70}")
    print("[3] EXPERIMENT 2: Multi-TF Momentum + LightGBM Filter")
    print(f"{'='*70}")
    
    result_ml = walk_forward_backtest(
        close, ret, vol,
        train_months=6, test_months=2,
        top_n=7, holding_bars=36, use_ml=True
    )
    
    if result_ml:
        print(f"\n  Stitched OOS Results:")
        print(f"  Sharpe:  {result_ml['sharpe']:.2f}")
        print(f"  PF:      {result_ml['pf']:.2f}")
        print(f"  MaxDD:   {result_ml['max_dd_pct']:.1%}")
        print(f"  PnL:     ${result_ml['pnl']:.0f}")
        print(f"  Windows: {result_ml['n_windows']}")
    
    # =========================================================================
    # Experiment 3: WFO with expanding window
    # =========================================================================
    print(f"\n{'='*70}")
    print("[4] EXPERIMENT 3: Expanding Window (train grows, test=2mo)")
    print(f"{'='*70}")
    
    # Custom expanding window
    dates = ret.index
    n = len(dates)
    bars_per_month = 6 * 30
    test_bars = 2 * bars_per_month
    min_train = 6 * bars_per_month
    
    all_returns = []
    window_id = 0
    train_end = min_train
    
    features = compute_features(close, ret, vol)
    feat_cols = [c for c in features.columns if c not in ["target", "fwd_ret"]]
    
    while train_end + test_bars <= n:
        test_start = train_end
        test_end = min(test_start + test_bars, n)
        
        is_dates = dates[:train_end]
        oos_dates = dates[test_start:test_end]
        
        is_feat = features.loc[features.index.get_level_values(0).isin(is_dates)]
        oos_feat = features.loc[features.index.get_level_values(0).isin(oos_dates)]
        
        if len(is_feat) < 100 or len(oos_feat) < 20:
            train_end += test_bars
            continue
        
        # Train ML
        X_is = is_feat[feat_cols].fillna(0)
        y_is = is_feat["target"]
        dtrain = lgb.Dataset(X_is, label=y_is)
        params = {"objective": "binary", "metric": "auc", "num_leaves": 8,
                  "max_depth": 3, "learning_rate": 0.05, "n_estimators": 100,
                  "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.1,
                  "verbose": -1}
        model = lgb.train(params, dtrain, num_boost_round=100,
                         valid_sets=[dtrain],
                         callbacks=[lgb.early_stopping(20, verbose=False)])
        
        # Predict — reshape to DataFrame for proper alignment
        X_oos = oos_feat[feat_cols].fillna(0)
        preds = model.predict(X_oos)
        ml_preds_df = (
            pd.Series(preds, index=X_oos.index)
            .unstack(level=1)
            .reindex(index=oos_dates, columns=ret.columns)
            .fillna(0.0)
        )
        
        # Momentum on OOS
        cr = ret.cumsum().loc[oos_dates]
        mom_s = cr.rolling(16).sum()
        mom_l = cr.rolling(99).sum()
        combined = 0.68 * mom_s + 0.32 * mom_l
        
        positions = pd.DataFrame(0.0, index=oos_dates, columns=ret.columns)
        top_n = 7
        holding = 36
        
        for t_idx in range(len(oos_dates)):
            if t_idx % holding != 0 and t_idx > 0:
                positions.iloc[t_idx] = positions.iloc[t_idx - 1]
                continue
            
            mom_row = combined.iloc[t_idx].values
            valid = ~np.isnan(mom_row)
            
            # ML filter: proper alignment via DataFrame
            ml_row = ml_preds_df.iloc[t_idx].values
            valid = valid & (ml_row > 0.48)
            
            if valid.sum() < top_n * 2:
                positions.iloc[t_idx] = positions.iloc[t_idx - 1] if t_idx > 0 else 0
                continue
            
            ranks = np.argsort(np.where(valid, mom_row, -np.inf))
            positions.iloc[t_idx, ranks[-top_n:]] = 1.0 / top_n
            positions.iloc[t_idx, ranks[:top_n]] = -1.0 / top_n
        
        positions = positions.shift(1).fillna(0)
        
        # Returns
        oos_ret_vals = ret.loc[oos_dates].values
        pos_vals = positions.values
        gross = (pos_vals * oos_ret_vals).sum(axis=1)
        delta = np.abs(np.diff(pos_vals, axis=0, prepend=pos_vals[:1]))
        costs = FEE_PCT * delta.sum(axis=1) * (CAPITAL / 2)
        net = gross * CAPITAL - costs
        
        all_returns.append(pd.Series(net, index=oos_dates))
        
        sharpe = (net / CAPITAL).mean() / (net / CAPITAL).std() * np.sqrt(6 * 365) if (net / CAPITAL).std() > 0 else 0
        cum = np.cumsum(net)
        dd = abs((cum - np.maximum.accumulate(cum)).min())
        g = net[net > 0].sum(); l = abs(net[net < 0].sum())
        pf = g / l if l > 0 else 0
        
        print(f"  W{window_id}: {oos_dates[0].strftime('%Y-%m')} → {oos_dates[-1].strftime('%Y-%m')} | "
              f"Sharpe={sharpe:.2f} PF={pf:.2f} MaxDD={dd/CAPITAL:.1%} PnL=${net.sum():.0f}")
        
        window_id += 1
        train_end += test_bars
    
    if all_returns:
        stitched = pd.concat(all_returns).sort_index()
        stitched = stitched[~stitched.index.duplicated(keep='first')]
        net_ret = stitched / CAPITAL
        shp = net_ret.mean() / net_ret.std() * np.sqrt(6 * 365) if net_ret.std() > 0 else 0
        cum = np.cumsum(stitched)
        dd = abs((cum - np.maximum.accumulate(cum)).min())
        g = stitched[stitched > 0].sum(); l = abs(stitched[stitched < 0].sum())
        pf = g / l if l > 0 else 0
        
        print(f"\n  Expanding Window Stitched Results:")
        print(f"  Sharpe:  {shp:.2f}")
        print(f"  PF:      {pf:.2f}")
        print(f"  MaxDD:   {dd/CAPITAL:.1%}")
        print(f"  PnL:     ${stitched.sum():.0f}")
    
    # =========================================================================
    # FINAL COMPARISON
    # =========================================================================
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Method':<35} {'Sharpe':>8} {'PF':>8} {'MaxDD':>8} {'PnL':>10}")
    print("-" * 70)
    
    if result_pure:
        print(f"{'Multi-TF (no ML, rolling)':<35} {result_pure['sharpe']:>8.2f} "
              f"{result_pure['pf']:>8.2f} {result_pure['max_dd_pct']:>7.1%} ${result_pure['pnl']:>9.0f}")
    if result_ml:
        print(f"{'Multi-TF + LightGBM (rolling)':<35} {result_ml['sharpe']:>8.2f} "
              f"{result_ml['pf']:>8.2f} {result_ml['max_dd_pct']:>7.1%} ${result_ml['pnl']:>9.0f}")
    if all_returns:
        print(f"{'Multi-TF + LightGBM (expanding)':<35} {shp:>8.2f} "
              f"{pf:>8.2f} {dd/CAPITAL:>7.1%} ${stitched.sum():>9.0f}")
    
    # Degradation check
    if result_pure and result_ml:
        deg_sharpe = (result_pure['sharpe'] - result_ml['sharpe']) / result_pure['sharpe'] * 100
        deg_pf = (result_pure['pf'] - result_ml['pf']) / result_pure['pf'] * 100
        print(f"\n  ML Impact on Sharpe: {deg_sharpe:+.0f}%")
        print(f"  ML Impact on PF:     {deg_pf:+.0f}%")
    
    print(f"\nTotal time: {time()-t0:.1f}s")


if __name__ == "__main__":
    main()
