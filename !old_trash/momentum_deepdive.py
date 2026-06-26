"""Momentum deep-dive: walk-forward, monthly returns, parameter stability."""

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


def cs_momentum(returns: pd.DataFrame, lookback: int = 30,
                top_n: int = 5, holding: int = 6) -> pd.DataFrame:
    """Returns positions DataFrame for analysis."""
    n = len(returns)
    n_assets = returns.shape[1]
    top_n = min(top_n, n_assets // 2)
    
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
    
    return pd.DataFrame(positions, index=returns.index, columns=returns.columns)


def metrics(ret: pd.Series, capital: float = CAPITAL, factor: int = 6 * 365) -> dict:
    if len(ret) == 0 or ret.std() == 0:
        return {"pnl": 0, "sharpe": 0, "max_dd": 0, "win_rate": 0, "trades": 0}
    
    dollars = ret * capital
    cum = np.cumsum(dollars)
    total = cum.iloc[-1]
    sharpe = ret.mean() / ret.std() * np.sqrt(factor)
    max_dd = np.max(np.maximum.accumulate(cum) - cum)
    active = ret[ret != 0]
    win_rate = (active > 0).mean() if len(active) > 0 else 0
    
    # Count trades (position changes)
    pos_changes = (positions_filled != positions_filled.shift(1)).any(axis=1).sum() if 'positions_filled' in dir() else 0
    
    return {"pnl": total, "sharpe": sharpe, "max_dd": max_dd, "win_rate": win_rate}


def main():
    print("=" * 70)
    print("MOMENTUM DEEP DIVE")
    print("=" * 70)
    
    df = load_all_4h()
    close = build_close_matrix(df)
    returns = build_returns(close)
    
    print(f"Assets: {close.shape[1]}, Bars: {len(close)}")
    print(f"Period: {close.index[0]} — {close.index[-1]}")
    
    # =========================================================================
    # 1. PARAMETER STABILITY
    # =========================================================================
    print(f"\n{'='*70}")
    print("[1] PARAMETER STABILITY (IS vs OOS)")
    print(f"{'='*70}")
    
    param_grid = []
    for lb in [12, 18, 24, 30, 36, 42, 48]:
        for tn in [3, 4, 5, 6, 7]:
            for h in [3, 6, 9, 12]:
                positions = cs_momentum(returns, lookback=lb, top_n=tn, holding=h)
                pos_ret = (positions.shift(1).fillna(0) * returns).sum(axis=1)
                
                is_r = pos_ret.loc[pos_ret.index < SPLIT_DATE]
                oos_r = pos_ret.loc[pos_ret.index >= SPLIT_DATE]
                
                is_m = metrics(is_r)
                oos_m = metrics(oos_r)
                
                param_grid.append({
                    "lookback": lb, "top_n": tn, "holding": h,
                    "is_sharpe": is_m["sharpe"],
                    "oos_sharpe": oos_m["sharpe"],
                    "is_pnl": is_m["pnl"],
                    "oos_pnl": oos_m["pnl"],
                })
    
    df_grid = pd.DataFrame(param_grid)
    
    # Show top 10 by OOS Sharpe
    print("\nTop 10 by OOS Sharpe:")
    print(df_grid.nlargest(10, "oos_sharpe")[["lookback", "top_n", "holding", 
                                               "is_sharpe", "oos_sharpe", "oos_pnl"]].to_string(index=False))
    
    # Consistency: how many param combos have positive OOS Sharpe?
    pos_oos = (df_grid["oos_sharpe"] > 0).sum()
    total = len(df_grid)
    print(f"\nPositive OOS Sharpe: {pos_oos}/{total} ({pos_oos/total*100:.0f}%)")
    
    # Correlation between IS and OOS Sharpe
    corr = df_grid["is_sharpe"].corr(df_grid["oos_sharpe"])
    print(f"IS-OOS Sharpe correlation: {corr:.3f}")
    
    # =========================================================================
    # 2. WALK-FORWARD ANALYSIS
    # =========================================================================
    print(f"\n{'='*70}")
    print("[2] WALK-FORWARD ANALYSIS (6-month rolling)")
    print(f"{'='*70}")
    
    # Split into 6-month windows
    dates = returns.index
    start = dates[0]
    windows = []
    
    window_size = 6 * 30 * 6  # ~6 months in 4h bars (30 days * 6 bars/day * 6)
    test_size = 2 * 30 * 6    # ~2 months test
    
    t = start
    while True:
        train_end = t + pd.Timedelta(days=180)
        test_end = train_end + pd.Timedelta(days=60)
        
        if test_end > dates[-1]:
            break
        
        windows.append((t, train_end, test_end))
        t = train_end
    
    print(f"Windows: {len(windows)}")
    
    wf_results = []
    for i, (train_start, train_end, test_end) in enumerate(windows):
        # Train: find best params
        train_returns = returns.loc[(returns.index >= train_start) & (returns.index < train_end)]
        
        best_sharpe = -np.inf
        best_params = {"lookback": 30, "top_n": 5, "holding": 6}
        
        for lb in [18, 30, 42]:
            for tn in [3, 5]:
                for h in [3, 6]:
                    pos = cs_momentum(train_returns, lookback=lb, top_n=tn, holding=h)
                    ret = (pos.shift(1).fillna(0) * train_returns).sum(axis=1)
                    m = metrics(ret)
                    if m["sharpe"] > best_sharpe:
                        best_sharpe = m["sharpe"]
                        best_params = {"lookback": lb, "top_n": tn, "holding": h}
        
        # Test: apply best params to test period
        test_returns = returns.loc[(returns.index >= train_end) & (returns.index < test_end)]
        
        # Need some history before test for momentum lookback
        history_start = train_end - pd.Timedelta(days=best_params["lookback"] * 4 / 24)
        extended_returns = returns.loc[(returns.index >= history_start) & (returns.index < test_end)]
        
        pos = cs_momentum(extended_returns, **best_params)
        # Only evaluate on test period
        test_mask = extended_returns.index >= train_end
        test_ret = (pos.shift(1).fillna(0) * extended_returns).sum(axis=1)
        test_ret = test_ret[test_mask]
        
        m = metrics(test_ret)
        
        wf_results.append({
            "window": i + 1,
            "train": f"{train_start.strftime('%Y-%m')} → {train_end.strftime('%Y-%m')}",
            "test": f"{train_end.strftime('%Y-%m')} → {test_end.strftime('%Y-%m')}",
            "params": best_params,
            "train_sharpe": best_sharpe,
            "test_sharpe": m["sharpe"],
            "test_pnl": m["pnl"],
        })
        
        print(f"  W{i+1}: {train_end.strftime('%Y-%m')} → {test_end.strftime('%Y-%m')} | "
              f"lb={best_params['lookback']} top={best_params['top_n']} | "
              f"Train={best_sharpe:.2f} Test={m['sharpe']:.2f} PnL=${m['pnl']:.0f}")
    
    wf_df = pd.DataFrame(wf_results)
    print(f"\n  Walk-Forward OOS Sharpe: mean={wf_df['test_sharpe'].mean():.2f}, "
          f"median={wf_df['test_sharpe'].median():.2f}, "
          f"std={wf_df['test_sharpe'].std():.2f}")
    print(f"  Positive windows: {(wf_df['test_sharpe'] > 0).sum()}/{len(wf_df)}")
    
    # =========================================================================
    # 3. MONTHLY RETURNS (OOS only)
    # =========================================================================
    print(f"\n{'='*70}")
    print("[3] MONTHLY RETURNS (OOS period)")
    print(f"{'='*70}")
    
    best_params = {"lookback": 42, "top_n": 5, "holding": 3}  # From grid search
    positions = cs_momentum(returns, **best_params)
    pos_ret = (positions.shift(1).fillna(0) * returns).sum(axis=1)
    
    oos_ret = pos_ret.loc[pos_ret.index >= SPLIT_DATE]
    oos_dollars = oos_ret * CAPITAL
    
    monthly = oos_dollars.resample("ME").sum()
    monthly_cum = oos_dollars.cumsum().resample("ME").last()
    
    print(f"\n{'Month':<12} {'PnL':>10} {'Cumulative':>12}")
    print("-" * 35)
    for date, pnl in monthly.items():
        cum = monthly_cum.loc[date]
        print(f"  {date.strftime('%Y-%m'):<10} ${pnl:>9.0f} ${cum:>11.0f}")
    
    # =========================================================================
    # 4. DRAWDOWN ANALYSIS
    # =========================================================================
    print(f"\n{'='*70}")
    print("[4] DRAWDOWN ANALYSIS (OOS)")
    print(f"{'='*70}")
    
    cum_pnl = np.cumsum(oos_dollars)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    
    print(f"  Max Drawdown: ${drawdown.min():.0f}")
    print(f"  Max DD Duration: {len(drawdown[drawdown < 0])} bars")
    
    # Recovery
    max_dd_idx = drawdown.idxmin()
    after_dd = drawdown.loc[max_dd_idx:]
    recovered = after_dd[after_dd >= 0]
    if len(recovered) > 0:
        recovery_bars = (recovered.index[0] - max_dd_idx).total_seconds() / 3600 / 4
        print(f"  Recovery time: ~{recovery_bars:.0f} bars ({recovery_bars/6:.0f} days)")
    
    # =========================================================================
    # 5. RISK METRICS
    # =========================================================================
    print(f"\n{'='*70}")
    print("[5] RISK METRICS (OOS)")
    print(f"{'='*70}")
    
    # Value at Risk (95%)
    var_95 = np.percentile(oos_dollars, 5)
    cvar_95 = oos_dollars[oos_dollars <= var_95].mean()
    
    print(f"  VaR (95%): ${var_95:.0f}")
    print(f"  CVaR (95%): ${cvar_95:.0f}")
    print(f"  Daily Return Mean: ${oos_dollars.mean():.2f}")
    print(f"  Daily Return Std: ${oos_dollars.std():.2f}")
    print(f"  Skew: {oos_dollars.skew():.3f}")
    print(f"  Kurtosis: {oos_dollars.kurtosis():.3f}")
    
    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print(f"{'='*70}")
    
    final_oos = metrics(oos_ret)
    print(f"  Strategy: Cross-Sectional Momentum")
    print(f"  Params: lookback=42, top_n=5, holding=3")
    print(f"  OOS Sharpe: {final_oos['sharpe']:.2f}")
    print(f"  OOS PnL: ${final_oos['pnl']:.0f}")
    print(f"  OOS MaxDD: ${final_oos['max_dd']:.0f}")
    print(f"  OOS WinRate: {final_oos['win_rate']:.1%}")
    
    if final_oos["sharpe"] > 0.8:
        print(f"\n  >>> ROBUST ALPHA CONFIRMED: Sharpe {final_oos['sharpe']:.2f} > 0.8 <<<")
    else:
        print(f"\n  Sharpe {final_oos['sharpe']:.2f} — below 0.8 target")


if __name__ == "__main__":
    main()
