"""4H Alpha Research v2: Fast version with vectorized operations."""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("data/binance_4h")
SPLIT_DATE = "2026-02-01"
CAPITAL = 10000.0


def load_all_4h() -> pd.DataFrame:
    frames = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        frames.append(pd.read_parquet(f))
    return pd.concat(frames, ignore_index=True)


def build_close_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(index="datetime", columns="symbol", values="close")
    pivot = pivot.sort_index().ffill()
    missing_pct = pivot.isna().mean()
    valid = missing_pct[missing_pct < 0.3].index
    pivot = pivot[valid].dropna()
    return pivot


def build_returns(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close / close.shift(1)).iloc[1:]


# =============================================================================
# METRICS
# =============================================================================

def metrics(ret: pd.Series, capital: float = CAPITAL, factor: int = 6 * 365) -> dict:
    if len(ret) == 0 or ret.std() == 0:
        return {"pnl": 0, "sharpe": 0, "max_dd": 0, "win_rate": 0, "sortino": 0}
    
    dollars = ret * capital
    cum = np.cumsum(dollars)
    total = cum.iloc[-1]
    sharpe = ret.mean() / ret.std() * np.sqrt(factor)
    max_dd = np.max(np.maximum.accumulate(cum) - cum)
    down = ret[ret < 0]
    sortino = ret.mean() / down.std() * np.sqrt(factor) if len(down) > 0 and down.std() > 0 else 0
    active = ret[ret != 0]
    win_rate = (active > 0).mean() if len(active) > 0 else 0
    return {"pnl": total, "sharpe": sharpe, "max_dd": max_dd, "win_rate": win_rate, "sortino": sortino}


# =============================================================================
# STRATEGY 1: CROSS-SECTIONAL MOMENTUM (vectorized)
# =============================================================================

def cs_momentum(returns: pd.DataFrame, lookback: int = 30,
                top_n: int = 5, holding: int = 6) -> pd.Series:
    """Vectorized cross-sectional momentum."""
    n = len(returns)
    n_assets = returns.shape[1]
    top_n = min(top_n, n_assets // 2)
    
    # Cumulative returns for ranking
    cumret = returns.cumsum()
    
    positions = np.zeros((n, returns.shape[1]))
    
    for t in range(lookback, n):
        if t % holding != 0:
            positions[t] = positions[t - 1]
            continue
        
        mom = cumret.iloc[t].values - cumret.iloc[t - lookback].values
        valid = ~np.isnan(mom)
        if valid.sum() < top_n * 2:
            continue
        
        ranks = np.argsort(np.where(valid, mom, -np.inf))
        long_idx = ranks[-top_n:]
        short_idx = ranks[:top_n]
        
        positions[t, long_idx] = 1.0 / top_n
        positions[t, short_idx] = -1.0 / top_n
    
    pos_df = pd.DataFrame(positions, index=returns.index, columns=returns.columns)
    pos_df = pos_df.shift(1).fillna(0)
    return (pos_df * returns).sum(axis=1)


# =============================================================================
# STRATEGY 2: VOL-TARGETED MOMENTUM (vectorized)
# =============================================================================

def vol_momentum(returns: pd.DataFrame, lookback: int = 30,
                 target_vol: float = 0.15, top_n: int = 5,
                 rebalance: int = 6) -> pd.Series:
    n = len(returns)
    n_assets = returns.shape[1]
    top_n = min(top_n, n_assets // 2)
    
    positions = np.zeros((n, n_assets))
    
    for t in range(lookback, n):
        if t % rebalance != 0:
            positions[t] = positions[t - 1]
            continue
        
        mom = returns.iloc[t - lookback:t].sum().values
        vol = returns.iloc[t - lookback:t].std().values * np.sqrt(6)
        vol = np.where(vol == 0, np.nan, vol)
        
        valid = ~np.isnan(mom) & ~np.isnan(vol)
        if valid.sum() < top_n * 2:
            continue
        
        scores = np.where(valid, mom, -np.inf)
        top_idx = np.argsort(scores)[-top_n:]
        
        inv_vol = 1.0 / vol[top_idx]
        inv_vol = inv_vol / inv_vol.sum()
        positions[t, top_idx] = inv_vol * (target_vol / (np.nanmean(vol[top_idx]) + 1e-8))
    
    pos_df = pd.DataFrame(np.clip(positions, -1, 1), index=returns.index, columns=returns.columns)
    pos_df = pos_df.shift(1).fillna(0)
    return (pos_df * returns).sum(axis=1)


# =============================================================================
# STRATEGY 3: MEAN REVERSION (z-score based)
# =============================================================================

def cs_mean_reversion(returns: pd.DataFrame, lookback: int = 30,
                      z_thresh: float = 1.5, holding: int = 6) -> pd.Series:
    n = len(returns)
    n_assets = returns.shape[1]
    
    positions = np.zeros((n, n_assets))
    
    for t in range(lookback, n):
        if t % holding != 0:
            positions[t] = positions[t - 1]
            continue
        
        cumret = returns.iloc[t - lookback:t].sum().values
        std = returns.iloc[t - lookback:t].std().values
        
        valid = ~np.isnan(cumret) & (std > 0)
        if valid.sum() < 4:
            continue
        
        mean = np.nanmean(cumret[valid])
        sd = np.nanstd(cumret[valid])
        if sd == 0:
            continue
        
        z = np.where(valid, (cumret - mean) / sd, 0)
        
        longs = np.where(z < -z_thresh)[0]
        shorts = np.where(z > z_thresh)[0]
        
        if len(longs) > 0:
            positions[t, longs] = 1.0 / len(longs)
        if len(shorts) > 0:
            positions[t, shorts] = -1.0 / len(shorts)
    
    pos_df = pd.DataFrame(positions, index=returns.index, columns=returns.columns)
    pos_df = pos_df.shift(1).fillna(0)
    return (pos_df * returns).sum(axis=1)


# =============================================================================
# STRATEGY 4: HURST-BASED DYNAMIC SELECTOR
# =============================================================================

def hurst_fast(series: np.ndarray, max_lag: int = 10) -> float:
    """Fast Hurst via variance ratio."""
    if len(series) < max_lag * 2:
        return 0.5
    
    lags = np.arange(2, max_lag + 1)
    variances = np.array([np.var(np.diff(series, n=n)) for n in lags])
    variances = np.where(variances == 0, 1e-10, variances)
    
    log_lags = np.log(lags)
    log_var = np.log(variances + 1e-10)
    
    slope = np.polyfit(log_lags, log_var, 1)[0]
    return np.clip(1 - slope / 2, 0, 1)


def dynamic_selector(returns: pd.DataFrame, lookback: int = 60,
                     top_n: int = 5, holding: int = 6) -> pd.Series:
    n = len(returns)
    n_assets = returns.shape[1]
    top_n = min(top_n, n_assets // 2)
    
    positions = np.zeros((n, n_assets))
    
    for t in range(lookback, n):
        if t % holding != 0:
            positions[t] = positions[t - 1]
            continue
        
        hurst = {}
        mom = {}
        for j, col in enumerate(returns.columns):
            w = returns.iloc[t - lookback:t, j].values
            if len(w) >= lookback and not np.any(np.isnan(w)):
                hurst[j] = hurst_fast(w)
                mom[j] = np.sum(w)
        
        if not hurst:
            continue
        
        mom_assets = [a for a, h in hurst.items() if h > 0.55]
        mr_assets = [a for a, h in hurst.items() if h < 0.45]
        
        if mom_assets:
            scores = [(a, mom[a]) for a in mom_assets]
            scores.sort(key=lambda x: x[1], reverse=True)
            n_l = min(top_n, len(scores))
            for a, _ in scores[:n_l]:
                positions[t, a] = 1.0 / n_l
        
        if mr_assets:
            scores = [(a, mom[a]) for a in mr_assets]
            scores.sort(key=lambda x: x[1])
            n_l = min(top_n, len(scores))
            for a, _ in scores[:n_l]:
                positions[t, a] = 1.0 / n_l
    
    pos_df = pd.DataFrame(positions, index=returns.index, columns=returns.columns)
    pos_df = pos_df.shift(1).fillna(0)
    return (pos_df * returns).sum(axis=1)


# =============================================================================
# GRID SEARCH
# =============================================================================

def grid_momentum(returns: pd.DataFrame) -> dict:
    best = {"sharpe": -np.inf, "params": None}
    for lb in [18, 30, 42]:
        for tn in [3, 5, 7]:
            for h in [3, 6, 12]:
                ret = cs_momentum(returns, lookback=lb, top_n=tn, holding=h)
                is_r = ret.loc[ret.index < SPLIT_DATE]
                m = metrics(is_r)
                if m["sharpe"] > best["sharpe"]:
                    best = {"sharpe": m["sharpe"], "params": {"lookback": lb, "top_n": tn, "holding": h}}
    return best


def grid_vol_mom(returns: pd.DataFrame) -> dict:
    best = {"sharpe": -np.inf, "params": None}
    for lb in [18, 30, 42]:
        for tv in [0.10, 0.15, 0.20]:
            for tn in [3, 5]:
                ret = vol_momentum(returns, lookback=lb, target_vol=tv, top_n=tn)
                is_r = ret.loc[ret.index < SPLIT_DATE]
                m = metrics(is_r)
                if m["sharpe"] > best["sharpe"]:
                    best = {"sharpe": m["sharpe"], "params": {"lookback": lb, "target_vol": tv, "top_n": tn}}
    return best


def grid_mr(returns: pd.DataFrame) -> dict:
    best = {"sharpe": -np.inf, "params": None}
    for lb in [18, 30, 42]:
        for zt in [1.0, 1.5, 2.0]:
            for h in [3, 6, 12]:
                ret = cs_mean_reversion(returns, lookback=lb, z_thresh=zt, holding=h)
                is_r = ret.loc[ret.index < SPLIT_DATE]
                m = metrics(is_r)
                if m["sharpe"] > best["sharpe"]:
                    best = {"sharpe": m["sharpe"], "params": {"lookback": lb, "z_thresh": zt, "holding": h}}
    return best


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("4H ALPHA RESEARCH v2")
    print("=" * 70)
    
    print("\n[1] Data loading...")
    df = load_all_4h()
    close = build_close_matrix(df)
    returns = build_returns(close)
    
    is_bars = len(returns.loc[returns.index < SPLIT_DATE])
    oos_bars = len(returns.loc[returns.index >= SPLIT_DATE])
    print(f"    Assets: {close.shape[1]}, Bars: {len(close)}")
    print(f"    Period: {close.index[0]} — {close.index[-1]}")
    print(f"    IS: {is_bars} | OOS: {oos_bars}")
    
    results = {}
    
    # =========================================================================
    # S1: Cross-Sectional Momentum
    # =========================================================================
    print(f"\n{'='*70}")
    print("[2] S1: Cross-Sectional Momentum (grid search)")
    print(f"{'='*70}")
    
    best = grid_momentum(returns)
    if best["params"]:
        p = best["params"]
        print(f"  Best: lb={p['lookback']}, top={p['top_n']}, hold={p['holding']}")
        
        full = cs_momentum(returns, **p)
        is_r = full.loc[full.index < SPLIT_DATE]
        oos_r = full.loc[full.index >= SPLIT_DATE]
        is_m = metrics(is_r)
        oos_m = metrics(oos_r)
        
        print(f"  IS:  Sharpe={is_m['sharpe']:.2f} PnL=${is_m['pnl']:.0f} MaxDD=${is_m['max_dd']:.0f} WR={is_m['win_rate']:.1%}")
        print(f"  OOS: Sharpe={oos_m['sharpe']:.2f} PnL=${oos_m['pnl']:.0f} MaxDD=${oos_m['max_dd']:.0f} WR={oos_m['win_rate']:.1%}")
        results["momentum"] = {"is": is_m, "oos": oos_m, "params": p}
    
    # =========================================================================
    # S2: Vol-Targeted Momentum
    # =========================================================================
    print(f"\n{'='*70}")
    print("[3] S2: Vol-Targeted Momentum (grid search)")
    print(f"{'='*70}")
    
    best = grid_vol_mom(returns)
    if best["params"]:
        p = best["params"]
        print(f"  Best: lb={p['lookback']}, tv={p['target_vol']}, top={p['top_n']}")
        
        full = vol_momentum(returns, **p)
        is_r = full.loc[full.index < SPLIT_DATE]
        oos_r = full.loc[full.index >= SPLIT_DATE]
        is_m = metrics(is_r)
        oos_m = metrics(oos_r)
        
        print(f"  IS:  Sharpe={is_m['sharpe']:.2f} PnL=${is_m['pnl']:.0f} MaxDD=${is_m['max_dd']:.0f} WR={is_m['win_rate']:.1%}")
        print(f"  OOS: Sharpe={oos_m['sharpe']:.2f} PnL=${oos_m['pnl']:.0f} MaxDD=${oos_m['max_dd']:.0f} WR={oos_m['win_rate']:.1%}")
        results["vol_mom"] = {"is": is_m, "oos": oos_m, "params": p}
    
    # =========================================================================
    # S3: Mean Reversion
    # =========================================================================
    print(f"\n{'='*70}")
    print("[4] S3: Cross-Sectional Mean Reversion (grid search)")
    print(f"{'='*70}")
    
    best = grid_mr(returns)
    if best["params"]:
        p = best["params"]
        print(f"  Best: lb={p['lookback']}, z={p['z_thresh']}, hold={p['holding']}")
        
        full = cs_mean_reversion(returns, **p)
        is_r = full.loc[full.index < SPLIT_DATE]
        oos_r = full.loc[full.index >= SPLIT_DATE]
        is_m = metrics(is_r)
        oos_m = metrics(oos_r)
        
        print(f"  IS:  Sharpe={is_m['sharpe']:.2f} PnL=${is_m['pnl']:.0f} MaxDD=${is_m['max_dd']:.0f} WR={is_m['win_rate']:.1%}")
        print(f"  OOS: Sharpe={oos_m['sharpe']:.2f} PnL=${oos_m['pnl']:.0f} MaxDD=${oos_m['max_dd']:.0f} WR={oos_m['win_rate']:.1%}")
        results["mean_rev"] = {"is": is_m, "oos": oos_m, "params": p}
    
    # =========================================================================
    # S4: Dynamic Selector
    # =========================================================================
    print(f"\n{'='*70}")
    print("[5] S4: Dynamic Selector (Hurst-based)")
    print(f"{'='*70}")
    
    for lb in [42, 60]:
        full = dynamic_selector(returns, lookback=lb, top_n=5, holding=6)
        is_r = full.loc[full.index < SPLIT_DATE]
        oos_r = full.loc[full.index >= SPLIT_DATE]
        is_m = metrics(is_r)
        oos_m = metrics(oos_r)
        
        print(f"  lb={lb} ({lb//6}d):")
        print(f"    IS:  Sharpe={is_m['sharpe']:.2f} PnL=${is_m['pnl']:.0f} MaxDD=${is_m['max_dd']:.0f} WR={is_m['win_rate']:.1%}")
        print(f"    OOS: Sharpe={oos_m['sharpe']:.2f} PnL=${oos_m['pnl']:.0f} MaxDD=${oos_m['max_dd']:.0f} WR={oos_m['win_rate']:.1%}")
        results[f"dynamic_{lb}"] = {"is": is_m, "oos": oos_m, "params": {"lookback": lb}}
    
    # =========================================================================
    # ENSEMBLE
    # =========================================================================
    print(f"\n{'='*70}")
    print("[6] ENSEMBLE (equal weight)")
    print(f"{'='*70}")
    
    strat_rets = {}
    if "momentum" in results:
        p = results["momentum"]["params"]
        strat_rets["mom"] = cs_momentum(returns, **p)
    if "vol_mom" in results:
        p = results["vol_mom"]["params"]
        strat_rets["vol"] = vol_momentum(returns, **p)
    if "mean_rev" in results:
        p = results["mean_rev"]["params"]
        strat_rets["mr"] = cs_mean_reversion(returns, **p)
    if "dynamic_60" in results:
        strat_rets["dyn"] = dynamic_selector(returns, lookback=60, top_n=5, holding=6)
    
    if len(strat_rets) > 1:
        ens = pd.DataFrame(strat_rets).fillna(0).mean(axis=1)
        is_r = ens.loc[ens.index < SPLIT_DATE]
        oos_r = ens.loc[ens.index >= SPLIT_DATE]
        is_m = metrics(is_r)
        oos_m = metrics(oos_r)
        
        print(f"  Components: {list(strat_rets.keys())}")
        print(f"  IS:  Sharpe={is_m['sharpe']:.2f} PnL=${is_m['pnl']:.0f} MaxDD=${is_m['max_dd']:.0f} WR={is_m['win_rate']:.1%}")
        print(f"  OOS: Sharpe={oos_m['sharpe']:.2f} PnL=${oos_m['pnl']:.0f} MaxDD=${oos_m['max_dd']:.0f} WR={oos_m['win_rate']:.1%}")
        results["ensemble"] = {"is": is_m, "oos": oos_m}
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("OOS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Strategy':<20} {'IS Sharpe':>10} {'OOS Sharpe':>10} {'OOS PnL':>10} {'OOS MaxDD':>10}")
    print("-" * 60)
    
    for name, r in results.items():
        print(f"{name:<20} {r['is']['sharpe']:>10.2f} {r['oos']['sharpe']:>10.2f} "
              f"${r['oos']['pnl']:>9.0f} ${r['oos']['max_dd']:>9.0f}")
    
    # Best OOS
    best_oos = max(results.items(), key=lambda x: x[1]["oos"]["sharpe"])
    print(f"\n  Best OOS: {best_oos[0]} (Sharpe={best_oos[1]['oos']['sharpe']:.2f})")
    
    if best_oos[1]["oos"]["sharpe"] > 0.8:
        print(f"  >>> TARGET ACHIEVED <<<")
    else:
        print(f"  Target: Sharpe > 0.8 (current: {best_oos[1]['oos']['sharpe']:.2f})")


if __name__ == "__main__":
    main()
