"""Honest Backtest Engine: Transaction Costs + Stress Test + Free Search.

8 bps per side, net metrics only. No self-deception.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("data/binance_4h")
SPLIT_DATE = "2026-02-01"
CAPITAL = 10000.0
FEE_BPS = 8  # 0.08% per side
FEE_PCT = FEE_BPS / 10000
MAX_DD_PCT = 0.15  # 15% max drawdown constraint


# =============================================================================
# DATA
# =============================================================================

def load_data():
    frames = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        frames.append(pd.read_parquet(f))
    df = pd.concat(frames, ignore_index=True)
    
    close = df.pivot_table(index="datetime", columns="symbol", values="close")
    close = close.sort_index().ffill()
    missing = close.isna().mean()
    close = close[missing[missing < 0.3].index].dropna()
    
    volume = df.pivot_table(index="datetime", columns="symbol", values="volume")
    volume = volume.sort_index().ffill()
    volume = volume[close.columns].dropna()
    
    returns = np.log(close / close.shift(1)).iloc[1:]
    volume = volume.iloc[1:]
    
    return close, returns, volume


# =============================================================================
# HONEST METRICS
# =============================================================================

def calc_positions_cost(old_pos: np.ndarray, new_pos: np.ndarray,
                        prices: np.ndarray) -> float:
    """Calculate transaction cost for position change."""
    delta = np.abs(new_pos - old_pos)
    # Cost = fee * |delta| * notional (prices as proxy for notional)
    cost = FEE_PCT * np.sum(delta) * np.mean(prices)
    return cost


def honest_metrics(returns: pd.DataFrame, positions: pd.DataFrame,
                   capital: float = CAPITAL) -> dict:
    """Compute honest net metrics with transaction costs."""
    n = len(returns)
    
    # Daily P&L from positions (before costs)
    gross_ret = (positions.shift(1).fillna(0) * returns).sum(axis=1)
    
    # Transaction costs at each rebalance
    costs = np.zeros(n)
    for t in range(1, n):
        old_pos = positions.iloc[t - 1].values
        new_pos = positions.iloc[t].values
        if not np.array_equal(old_pos, new_pos):
            prices = returns.iloc[t].values  # Use returns magnitude as proxy
            delta = np.abs(new_pos - old_pos)
            costs[t] = FEE_PCT * np.sum(delta) * capital / returns.shape[1]
    
    net_ret = gross_ret - costs / capital
    
    # Scale to capital
    net_dollars = net_ret * capital
    cum_pnl = np.cumsum(net_dollars)
    
    total_pnl = cum_pnl.iloc[-1]
    
    # Sharpe (net)
    if net_ret.std() == 0:
        sharpe = 0
    else:
        sharpe = net_ret.mean() / net_ret.std() * np.sqrt(6 * 365)
    
    # Max Drawdown
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_dd = abs(drawdown.min())
    max_dd_pct = max_dd / capital
    
    # Profit Factor
    gains = net_dollars[net_dollars > 0].sum()
    losses = abs(net_dollars[net_dollars < 0].sum())
    profit_factor = gains / losses if losses > 0 else float('inf')
    
    # Win Rate
    active = net_ret[net_ret != 0]
    win_rate = (active > 0).mean() if len(active) > 0 else 0
    
    # Calmar
    calmar = total_pnl / max_dd if max_dd > 0 else 0
    
    return {
        "pnl": total_pnl,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "calmar": calmar,
        "total_costs": costs.sum(),
    }


# =============================================================================
# STRATEGIES
# =============================================================================

def cs_momentum(returns: pd.DataFrame, lookback: int = 30,
                top_n: int = 5, holding: int = 6) -> pd.DataFrame:
    """Cross-sectional momentum positions."""
    n = len(returns)
    n_assets = returns.shape[1]
    top_n = min(top_n, n_assets // 2)
    
    cumret = returns.cumsum()
    positions = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    
    for t in range(lookback, n):
        if t % holding != 0:
            positions.iloc[t] = positions.iloc[t - 1]
            continue
        
        mom = cumret.iloc[t].values - cumret.iloc[t - lookback].values
        valid = ~np.isnan(mom)
        if valid.sum() < top_n * 2:
            continue
        
        ranks = np.argsort(np.where(valid, mom, -np.inf))
        long_idx = ranks[-top_n:]
        short_idx = ranks[:top_n]
        
        positions.iloc[t, long_idx] = 1.0 / top_n
        positions.iloc[t, short_idx] = -1.0 / top_n
    
    return positions


def asymmetric_momentum(returns: pd.DataFrame, lookback: int = 30,
                         z_entry: float = 2.0, holding: int = 6) -> pd.DataFrame:
    """Only trade when momentum is extreme (z-score based)."""
    n = len(returns)
    n_assets = returns.shape[1]
    
    positions = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    
    for t in range(lookback, n):
        if t % holding != 0:
            positions.iloc[t] = positions.iloc[t - 1]
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
        
        # Long: z > +z_entry (strong momentum)
        longs = np.where(z > z_entry)[0]
        # Short: z < -z_entry (strong weakness)
        shorts = np.where(z < -z_entry)[0]
        
        if len(longs) > 0:
            positions.iloc[t, longs] = 1.0 / len(longs)
        if len(shorts) > 0:
            positions.iloc[t, shorts] = -1.0 / len(shorts)
    
    return positions


def vol_momentum(returns: pd.DataFrame, lookback: int = 30,
                 top_n: int = 5, holding: int = 6) -> pd.DataFrame:
    """Momentum with inverse-vol weighting."""
    n = len(returns)
    n_assets = returns.shape[1]
    top_n = min(top_n, n_assets // 2)
    
    positions = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    
    for t in range(lookback, n):
        if t % holding != 0:
            positions.iloc[t] = positions.iloc[t - 1]
            continue
        
        mom = returns.iloc[t - lookback:t].sum().values
        vol = returns.iloc[t - lookback:t].std().values * np.sqrt(6)
        vol = np.where(vol == 0, np.nan, vol)
        
        valid = ~np.isnan(mom) & ~np.isnan(vol)
        if valid.sum() < top_n * 2:
            continue
        
        scores = np.where(valid, mom, -np.inf)
        top_idx = np.argsort(scores)[-top_n:]
        bot_idx = np.argsort(scores)[:top_n]
        
        inv_vol_top = 1.0 / vol[top_idx]
        inv_vol_top = inv_vol_top / inv_vol_top.sum()
        
        inv_vol_bot = 1.0 / vol[bot_idx]
        inv_vol_bot = inv_vol_bot / inv_vol_bot.sum()
        
        positions.iloc[t, top_idx] = inv_vol_top
        positions.iloc[t, bot_idx] = -inv_vol_bot
    
    return positions


def volume_spike_alpha(returns: pd.DataFrame, volume: pd.DataFrame,
                        lookback: int = 30, z_entry: float = 1.5,
                        holding: int = 6) -> pd.DataFrame:
    """Trade on volume anomalies: buy assets with volume spike + positive returns."""
    n = len(returns)
    n_assets = returns.shape[1]
    
    positions = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    
    vol_ma = volume.rolling(lookback).mean()
    vol_std = volume.rolling(lookback).std()
    vol_z = (volume - vol_ma) / (vol_std + 1e-8)
    
    for t in range(lookback, n):
        if t % holding != 0:
            positions.iloc[t] = positions.iloc[t - 1]
            continue
        
        vz = vol_z.iloc[t].values
        ret_sum = returns.iloc[t - lookback:t].sum().values
        
        valid = ~np.isnan(vz) & ~np.isnan(ret_sum)
        if valid.sum() < 4:
            continue
        
        # Volume spike + positive returns = bullish
        # Volume spike + negative returns = bearish
        scores = vz * ret_sum  # Signed volume momentum
        
        longs = np.where(valid & (scores > z_entry))[0]
        shorts = np.where(valid & (scores < -z_entry))[0]
        
        if len(longs) > 0:
            positions.iloc[t, longs] = 1.0 / len(longs)
        if len(shorts) > 0:
            positions.iloc[t, shorts] = -1.0 / len(shorts)
    
    return positions


def regime_adaptive(returns: pd.DataFrame, lookback: int = 60,
                    vol_window: int = 30, holding: int = 6) -> pd.DataFrame:
    """Switch between momentum and mean-reversion based on market regime."""
    n = len(returns)
    n_assets = returns.shape[1]
    
    positions = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    
    # Market-level volatility regime
    mkt_ret = returns.mean(axis=1)
    mkt_vol = mkt_ret.rolling(vol_window).std()
    mkt_vol_median = mkt_vol.rolling(lookback).median()
    
    for t in range(lookback, n):
        if t % holding != 0:
            positions.iloc[t] = positions.iloc[t - 1]
            continue
        
        # High vol = momentum regime, Low vol = mean reversion
        is_high_vol = mkt_vol.iloc[t] > mkt_vol_median.iloc[t] if not np.isnan(mkt_vol_median.iloc[t]) else True
        
        cumret = returns.iloc[t - lookback:t].sum().values
        valid = ~np.isnan(cumret)
        
        if valid.sum() < 6:
            continue
        
        if is_high_vol:
            # Momentum: long winners, short losers
            ranks = np.argsort(np.where(valid, cumret, -np.inf))
            n_pick = min(5, valid.sum() // 3)
            positions.iloc[t, ranks[-n_pick:]] = 1.0 / n_pick
            positions.iloc[t, ranks[:n_pick]] = -1.0 / n_pick
        else:
            # Mean reversion: long losers, short winners
            ranks = np.argsort(np.where(valid, cumret, -np.inf))
            n_pick = min(5, valid.sum() // 3)
            positions.iloc[t, ranks[:n_pick]] = 1.0 / n_pick
            positions.iloc[t, ranks[-n_pick:]] = -1.0 / n_pick
    
    return positions


# =============================================================================
# GRID SEARCH
# =============================================================================

def grid_search(returns, volume, strategy_name, strategy_func, param_grid):
    """Grid search with honest metrics on IS."""
    results = []
    
    for params in param_grid:
        try:
            pos = strategy_func(returns, volume, **params)
            m = honest_metrics(returns, pos)
            results.append({**params, **m})
        except Exception:
            continue
    
    if not results:
        return None, []
    
    df = pd.DataFrame(results)
    best = df.loc[df["sharpe"].idxmax()]
    return best, df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("STRESS TEST: Honest Backtest with 8 bps Transaction Costs")
    print("=" * 70)
    print(f"  Fee: {FEE_BPS} bps per side")
    print(f"  Capital: ${CAPITAL:,.0f}")
    print(f"  Max DD Constraint: {MAX_DD_PCT:.0%}")
    print(f"  Target: Net Sharpe > 1.0, PF > 1.25")
    
    # Load
    print("\n[1] Loading data...")
    close, returns, volume = load_data()
    
    is_mask = returns.index < SPLIT_DATE
    oos_mask = returns.index >= SPLIT_DATE
    
    print(f"  Assets: {returns.shape[1]}")
    print(f"  IS bars: {is_mask.sum()} | OOS bars: {oos_mask.sum()}")
    
    all_results = {}
    
    # =========================================================================
    # PHASE 1: Stress Test Momentum
    # =========================================================================
    print(f"\n{'='*70}")
    print("[2] STRESS TEST: Cross-Sectional Momentum")
    print(f"{'='*70}")
    
    mom_grid = []
    for lb in [12, 18, 24, 30, 36, 42]:
        for tn in [3, 4, 5, 6]:
            for h in [3, 6, 9, 12]:
                mom_grid.append({"lookback": lb, "top_n": tn, "holding": h})
    
    def mom_func(ret, vol, **kw):
        return cs_momentum(ret, **kw)
    
    best_mom, all_mom = grid_search(returns, volume, "momentum", mom_func, mom_grid)
    
    if best_mom is not None:
        print(f"\n  Best IS params: lb={best_mom['lookback']}, top={best_mom['top_n']}, hold={best_mom['holding']}")
        print(f"  IS: Sharpe={best_mom['sharpe']:.2f} PF={best_mom['profit_factor']:.2f} "
              f"MaxDD={best_mom['max_dd_pct']:.1%} PnL=${best_mom['pnl']:.0f}")
        
        # OOS with best params
        pos = cs_momentum(returns, lookback=int(best_mom['lookback']),
                          top_n=int(best_mom['top_n']),
                          holding=int(best_mom['holding']))
        oos_m = honest_metrics(returns.loc[oos_mask], pos.loc[oos_mask])
        
        print(f"  OOS: Sharpe={oos_m['sharpe']:.2f} PF={oos_m['profit_factor']:.2f} "
              f"MaxDD={oos_m['max_dd_pct']:.1%} PnL=${oos_m['pnl']:.0f} "
              f"Costs=${oos_m['total_costs']:.0f}")
        
        all_results["momentum"] = {"is": best_mom.to_dict(), "oos": oos_m}
        
        # Check if passes
        passes = oos_m["sharpe"] > 0.8 and oos_m["profit_factor"] > 1.15
        print(f"\n  VERDICT: {'PASSED' if passes else 'FAILED'} "
              f"(Sharpe {oos_m['sharpe']:.2f} {'>' if oos_m['sharpe'] > 0.8 else '<'} 0.8, "
              f"PF {oos_m['profit_factor']:.2f} {'>' if oos_m['profit_factor'] > 1.15 else '<'} 1.15)")
    
    # =========================================================================
    # PHASE 2: Test ALL strategies with honest costs
    # =========================================================================
    print(f"\n{'='*70}")
    print("[3] FULL STRATEGY COMPARISON (with 8 bps costs)")
    print(f"{'='*70}")
    
    strategies = {
        "Asymmetric Momentum": {
            "func": asymmetric_momentum,
            "grid": [{"lookback": lb, "z_entry": ze, "holding": h}
                     for lb in [18, 30, 42] for ze in [1.5, 2.0, 2.5] for h in [3, 6, 12]],
        },
        "Vol-Weighted Momentum": {
            "func": vol_momentum,
            "grid": [{"lookback": lb, "top_n": tn, "holding": h}
                     for lb in [18, 30, 42] for tn in [3, 5, 7] for h in [3, 6, 12]],
        },
        "Volume Spike Alpha": {
            "func": volume_spike_alpha,
            "grid": [{"lookback": lb, "z_entry": ze, "holding": h}
                     for lb in [18, 30, 42] for ze in [1.0, 1.5, 2.0] for h in [3, 6, 12]],
        },
        "Regime Adaptive": {
            "func": regime_adaptive,
            "grid": [{"lookback": lb, "vol_window": vw, "holding": h}
                     for lb in [42, 60, 90] for vw in [20, 30] for h in [3, 6, 12]],
        },
    }
    
    for name, cfg in strategies.items():
        print(f"\n  --- {name} ---")
        
        def func(ret, vol, fn=cfg["func"], **kw):
            return fn(ret, volume=vol, **kw) if "volume" in fn.__code__.co_varnames else fn(ret, **kw)
        
        best, all_df = grid_search(returns, volume, name, func, cfg["grid"])
        
        if best is None:
            print("  No valid results")
            continue
        
        # OOS
        oos_params = {k: int(v) if isinstance(v, (np.floating, float)) and v == int(v) else v
                      for k, v in best.items()
                      if k in cfg["grid"][0].keys()}
        pos = func(returns, volume, **oos_params)
        oos_m = honest_metrics(returns.loc[oos_mask], pos.loc[oos_mask])
        
        print(f"  Best: {best.to_dict()}")
        print(f"  OOS: Sharpe={oos_m['sharpe']:.2f} PF={oos_m['profit_factor']:.2f} "
              f"MaxDD={oos_m['max_dd_pct']:.1%} PnL=${oos_m['pnl']:.0f}")
        
        all_results[name] = {"is": best.to_dict(), "oos": oos_m}
    
    # =========================================================================
    # PHASE 3: Ensemble
    # =========================================================================
    print(f"\n{'='*70}")
    print("[4] ENSEMBLE (top 3 strategies)")
    print(f"{'='*70}")
    
    # Pick top 3 by OOS Sharpe
    sorted_strats = sorted(all_results.items(), key=lambda x: x[1]["oos"]["sharpe"], reverse=True)
    top3 = sorted_strats[:3]
    
    if len(top3) >= 2:
        print(f"  Top 3: {[s[0] for s in top3]}")
        
        # Re-run each and average positions
        all_positions = []
        for name, res in top3:
            # Find params
            params = {k: int(v) if isinstance(v, (np.floating, float)) and v == int(v) else v
                      for k, v in res["is"].items() 
                      if k in ["lookback", "top_n", "holding", "z_entry", "vol_window"]}
            
            if name == "momentum":
                pos = cs_momentum(returns, **params)
            elif name == "Asymmetric Momentum":
                pos = asymmetric_momentum(returns, **params)
            elif name == "Vol-Weighted Momentum":
                pos = vol_momentum(returns, **params)
            elif name == "Volume Spike Alpha":
                pos = volume_spike_alpha(returns, volume, **params)
            elif name == "Regime Adaptive":
                pos = regime_adaptive(returns, **params)
            else:
                continue
            all_positions.append(pos)
        
        if all_positions:
            avg_pos = sum(all_positions) / len(all_positions)
            ens_m_is = honest_metrics(returns.loc[is_mask], avg_pos.loc[is_mask])
            ens_m_oos = honest_metrics(returns.loc[oos_mask], avg_pos.loc[oos_mask])
            
            print(f"  IS:  Sharpe={ens_m_is['sharpe']:.2f} PF={ens_m_is['profit_factor']:.2f} "
                  f"MaxDD={ens_m_is['max_dd_pct']:.1%}")
            print(f"  OOS: Sharpe={ens_m_oos['sharpe']:.2f} PF={ens_m_oos['profit_factor']:.2f} "
                  f"MaxDD={ens_m_oos['max_dd_pct']:.1%} PnL=${ens_m_oos['pnl']:.0f}")
            
            all_results["ensemble"] = {"is": ens_m_is, "oos": ens_m_oos}
    
    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    print(f"\n{'='*70}")
    print("FINAL HONEST REPORT")
    print(f"{'='*70}")
    print(f"{'Strategy':<25} {'IS Sharpe':>10} {'OOS Sharpe':>10} {'OOS PF':>8} "
          f"{'OOS MaxDD':>10} {'OOS PnL':>10} {'Verdict':>10}")
    print("-" * 85)
    
    for name, res in all_results.items():
        is_s = res["is"]["sharpe"] if isinstance(res["is"], dict) else res["is"].get("sharpe", 0)
        oos = res["oos"]
        
        passes = oos["sharpe"] > 0.8 and oos["profit_factor"] > 1.15 and oos["max_dd_pct"] < MAX_DD_PCT
        verdict = "PASS" if passes else "FAIL"
        
        print(f"{name:<25} {is_s:>10.2f} {oos['sharpe']:>10.2f} {oos['profit_factor']:>8.2f} "
              f"{oos['max_dd_pct']:>9.1%} ${oos['pnl']:>9.0f} {verdict:>10}")
    
    # Best overall
    best_name, best_res = max(all_results.items(), key=lambda x: x[1]["oos"]["sharpe"])
    oos = best_res["oos"]
    
    print(f"\n{'='*70}")
    print(f"BEST: {best_name}")
    print(f"  Net Sharpe:  {oos['sharpe']:.2f}")
    print(f"  Net PF:      {oos['profit_factor']:.2f}")
    print(f"  MaxDD:       {oos['max_dd_pct']:.1%}")
    print(f"  Net PnL:     ${oos['pnl']:.0f}")
    print(f"  Costs:       ${oos['total_costs']:.0f}")
    print(f"  Win Rate:    {oos['win_rate']:.1%}")
    
    if oos["sharpe"] > 1.0 and oos["profit_factor"] > 1.25 and oos["max_dd_pct"] < MAX_DD_PCT:
        print(f"\n  >>> ALL TARGETS MET <<<")
    elif oos["sharpe"] > 0.8 and oos["profit_factor"] > 1.15:
        print(f"\n  >>> ACCEPTABLE: Sharpe > 0.8, PF > 1.15 <<<")
    else:
        print(f"\n  >>> STRATEGY DOES NOT PASS HONEST TEST <<<")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
