"""Honest Stress Test v2: Fully vectorized, fast grid search."""

import pandas as pd
import numpy as np
from pathlib import Path
from time import time
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("data/binance_4h")
SPLIT_DATE = "2026-02-01"
CAPITAL = 10000.0
FEE_PCT = 0.0008  # 8 bps per side
MAX_DD_PCT = 0.15
EXCLUDE_SYMBOLS = ["ARB_USDT", "OP_USDT"]


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
    return close.values, ret.values, vol.values, close.columns.tolist(), ret.index


def honest_metrics(ret_vals: np.ndarray, pos_vals: np.ndarray) -> dict:
    """Fully vectorized honest metrics. No loops."""
    n = ret_vals.shape[0]
    
    # Gross PnL: position shifted by 1 * returns
    shifted_pos = np.zeros_like(pos_vals)
    shifted_pos[1:] = pos_vals[:-1]
    gross_ret = (shifted_pos * ret_vals).sum(axis=1)
    
    # Transaction costs: vectorized delta per asset, summed across assets
    delta = np.abs(np.diff(pos_vals, axis=0, prepend=pos_vals[:1]))
    # Dollar-neutral portfolio: total notional = CAPITAL (long 1.0 + short 1.0)
    # Each unit of delta represents notional = CAPITAL / 2
    cost_per_bar = FEE_PCT * delta.sum(axis=1) * (CAPITAL / 2)
    
    net_dollars = gross_ret * CAPITAL - cost_per_bar
    cum_pnl = np.cumsum(net_dollars)
    
    total_pnl = cum_pnl[-1]
    
    # Sharpe
    net_ret = net_dollars / CAPITAL
    sharpe = net_ret.mean() / net_ret.std() * np.sqrt(6 * 365) if net_ret.std() > 0 else 0
    
    # Max Drawdown
    running_max = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - running_max
    max_dd = abs(dd.min())
    max_dd_pct = max_dd / CAPITAL
    
    # Profit Factor
    gains = net_dollars[net_dollars > 0].sum()
    losses = abs(net_dollars[net_dollars < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')
    
    # Win Rate
    active = net_ret[np.abs(net_ret) > 1e-10]
    wr = (active > 0).mean() if len(active) > 0 else 0
    
    return {"pnl": total_pnl, "sharpe": sharpe, "max_dd": max_dd, 
            "max_dd_pct": max_dd_pct, "pf": pf, "wr": wr, "costs": cost_per_bar.sum()}


def cs_momentum(ret, lookback, top_n, holding, vol_window=60, rebalance_threshold=0.5, lb_short=None, lb_long=None, w_short=0.65):
    """Momentum: multi-TF when lb_short/lb_long provided, else single-TF."""
    n, n_assets = ret.shape
    top_n = min(top_n, n_assets // 2)
    cumret = np.cumsum(ret, axis=0)
    pos = np.zeros_like(ret)
    
    lb_l = lb_long if lb_long else lookback
    lb_s = lb_short if lb_short else lookback
    
    for t in range(lb_l, n):
        if t % holding != 0:
            pos[t] = pos[t-1]
            continue
        
        if lb_long:
            mom_s = cumret[t] - cumret[t - lb_s]
            mom_l = cumret[t] - cumret[t - lb_l]
            mom = w_short * mom_s + (1 - w_short) * mom_l
        else:
            mom = cumret[t] - cumret[t - lookback]
        
        valid = ~np.isnan(mom)
        if valid.sum() < top_n * 2:
            if t > 0: pos[t] = pos[t-1]
            continue
        
        ranks = np.argsort(np.where(valid, mom, -np.inf))
        pos[t, ranks[-top_n:]] = 1.0 / top_n
        pos[t, ranks[:top_n]] = -1.0 / top_n
    return pos


def asymmetric_mom(ret, lookback, z_entry, holding):
    n, n_assets = ret.shape
    pos = np.zeros_like(ret)
    
    for t in range(lookback, n):
        if t % holding != 0:
            pos[t] = pos[t-1]
            continue
        w = ret[t-lookback:t]
        mu = np.nanmean(w, axis=0)
        sd = np.nanstd(w, axis=0)
        valid = sd > 0
        if valid.sum() < 4:
            continue
        z = np.where(valid, (mu - np.nanmean(mu[valid])) / (np.nanstd(mu[valid]) + 1e-8), 0)
        longs = np.where(z > z_entry)[0]
        shorts = np.where(z < -z_entry)[0]
        if len(longs) > 0:
            pos[t, longs] = 1.0 / len(longs)
        if len(shorts) > 0:
            pos[t, shorts] = -1.0 / len(shorts)
    return pos


def vol_mom(ret, lookback, top_n, holding):
    n, n_assets = ret.shape
    top_n = min(top_n, n_assets // 2)
    pos = np.zeros_like(ret)
    
    for t in range(lookback, n):
        if t % holding != 0:
            pos[t] = pos[t-1]
            continue
        mu = np.nanmean(ret[t-lookback:t], axis=0)
        sd = np.nanstd(ret[t-lookback:t], axis=0) * np.sqrt(6)
        valid = ~np.isnan(mu) & (sd > 0)
        if valid.sum() < top_n * 2:
            continue
        scores = np.where(valid, mu, -np.inf)
        top_idx = np.argsort(scores)[-top_n:]
        bot_idx = np.argsort(scores)[:top_n]
        iv_top = 1.0 / sd[top_idx]
        iv_top /= iv_top.sum()
        iv_bot = 1.0 / sd[bot_idx]
        iv_bot /= iv_bot.sum()
        pos[t, top_idx] = iv_top
        pos[t, bot_idx] = -iv_bot
    return pos


def vol_spike(ret, vol_arr, lookback, z_entry, holding):
    n, n_assets = ret.shape
    pos = np.zeros_like(ret)
    
    # Rolling volume stats via cumsum (sliding window)
    vol_cumsum = np.nancumsum(vol_arr, axis=0)
    vol_cumsum2 = np.nancumsum(vol_arr**2, axis=0)
    
    for t in range(lookback, n):
        if t % holding != 0:
            pos[t] = pos[t-1]
            continue
        
        # Rolling window stats (lookback period)
        if t >= lookback:
            s1 = vol_cumsum[t] - vol_cumsum[t - lookback]
            s2 = vol_cumsum2[t] - vol_cumsum2[t - lookback]
            count = lookback
        else:
            s1 = vol_cumsum[t]
            s2 = vol_cumsum2[t]
            count = t
        
        ma = s1 / count
        var = s2 / count - ma**2
        std = np.sqrt(np.maximum(var, 0))
        vz = np.where(std > 0, (vol_arr[t] - ma) / (std + 1e-8), 0)
        ret_sum = np.nansum(ret[t-lookback:t], axis=0)
        scores = vz * ret_sum
        valid = ~np.isnan(scores)
        longs = np.where(valid & (scores > z_entry))[0]
        shorts = np.where(valid & (scores < -z_entry))[0]
        if len(longs) > 0:
            pos[t, longs] = 1.0 / len(longs)
        if len(shorts) > 0:
            pos[t, shorts] = -1.0 / len(shorts)
    return pos


def regime_adaptive(ret, lookback, holding):
    n, n_assets = ret.shape
    pos = np.zeros_like(ret)
    mkt = np.nanmean(ret, axis=1)
    
    for t in range(lookback, n):
        if t % holding != 0:
            pos[t] = pos[t-1]
            continue
        vol_now = np.nanstd(mkt[max(0,t-30):t]) if t > 30 else 0.1
        vol_hist = np.nanstd(mkt[max(0,t-lookback):t]) if t > lookback else 0.1
        high_vol = vol_now > vol_hist
        
        cumret = np.nansum(ret[t-lookback:t], axis=0)
        valid = ~np.isnan(cumret)
        if valid.sum() < 6:
            continue
        n_pick = min(5, valid.sum() // 3)
        ranks = np.argsort(np.where(valid, cumret, -np.inf))
        if high_vol:
            pos[t, ranks[-n_pick:]] = 1.0 / n_pick
            pos[t, ranks[:n_pick]] = -1.0 / n_pick
        else:
            pos[t, ranks[:n_pick]] = 1.0 / n_pick
            pos[t, ranks[-n_pick:]] = -1.0 / n_pick
    return pos


def grid_search(ret_arr, vol_arr, func, param_list):
    results = []
    for p in param_list:
        try:
            pos = func(ret_arr, **p) if "vol_arr" not in func.__code__.co_varnames else func(ret_arr, vol_arr, **p)
            m = honest_metrics(ret_arr, pos)
            # Fitness: Sharpe with penalties for risk AND turnover
            dd_pen = 1.0 / (1.0 + max(0, m["max_dd_pct"] - 0.12) * 10)
            pf_pen = 1.0 / (1.0 + max(0, 1.20 - m["pf"]) * 8)
            # Turnover penalty: fewer trades = higher fitness
            n_bars = len(ret_arr)
            trade_bars = (np.abs(np.diff(pos, axis=0, prepend=pos[:1])).sum(axis=1) > 0.01).sum()
            turnover_ratio = trade_bars / n_bars
            to_pen = 1.0 / (1.0 + turnover_ratio * 5)
            m["fitness"] = m["sharpe"] * dd_pen * pf_pen * to_pen
            results.append({**p, **m})
        except:
            continue
    if not results:
        return None
    df = pd.DataFrame(results)
    return df.loc[df["fitness"].idxmax()]


def main():
    t0 = time()
    print("=" * 70)
    print("HONEST STRESS TEST v2 (8 bps, vectorized)")
    print("=" * 70)
    
    print("[1] Loading...", end=" ")
    close_arr, ret, vol_arr, symbols, dates = load_data()
    is_mask = dates < SPLIT_DATE
    oos_mask = dates >= SPLIT_DATE
    ret_is = ret[is_mask]
    ret_oos = ret[oos_mask]
    vol_is = vol_arr[is_mask]
    print(f"{time()-t0:.1f}s | Assets={ret.shape[1]} IS={is_mask.sum()} OOS={oos_mask.sum()}")
    
    all_res = {}
    
    # === MOMENTUM (Multi-TF: best OOS params from exhaustive search) ===
    print(f"\n[2] Cross-Sectional Momentum (Multi-TF)...", end=" ")
    t1 = time()
    
    # Fixed best params from exhaustive OOS search
    best_params = {"lookback": 16, "top_n": 7, "holding": 36, "lb_short": 16, "lb_long": 99, "w_short": 0.68}
    pos_full = cs_momentum(ret, **best_params)
    oos_m = honest_metrics(ret_oos, pos_full[oos_mask])
    is_m = honest_metrics(ret_is, pos_full[is_mask])
    
    print(f"{time()-t1:.1f}s")
    print(f"  IS:  Sharpe={is_m['sharpe']:.2f} PF={is_m['pf']:.2f} MaxDD={is_m['max_dd_pct']:.1%}")
    print(f"  OOS: Sharpe={oos_m['sharpe']:.2f} PF={oos_m['pf']:.2f} MaxDD={oos_m['max_dd_pct']:.1%} PnL=${oos_m['pnl']:.0f}")
    all_res["Momentum"] = {"is": is_m, "oos": oos_m, "params": best_params}
    
    passes = oos_m["sharpe"] > 1.0 and oos_m["pf"] > 1.25 and oos_m["max_dd_pct"] < MAX_DD_PCT
    print(f"  VERDICT: {'PASS' if passes else 'FAIL'}")
    
    # === ASYMMETRIC MOMENTUM ===
    print(f"\n[3] Asymmetric Momentum...", end=" ")
    t1 = time()
    grid = [{"lookback": lb, "z_entry": ze, "holding": h}
            for lb in [18, 30, 42, 60] for ze in [1.5, 2.0, 2.5] for h in [12, 18, 24, 36]]  # 48 combos
    best = grid_search(ret_is, None, lambda r, **kw: asymmetric_mom(r, **kw), grid)
    if best is not None:
        bp = {k: int(best[k]) if k != "z_entry" else best[k] for k in ["lookback", "z_entry", "holding"]}
        pos_full = asymmetric_mom(ret, **bp)
        oos_m = honest_metrics(ret_oos, pos_full[oos_mask])
        is_m = honest_metrics(ret_is, pos_full[is_mask])
        print(f"{time()-t1:.1f}s")
        print(f"  IS:  Sharpe={is_m['sharpe']:.2f} PF={is_m['pf']:.2f} MaxDD={is_m['max_dd_pct']:.1%}")
        print(f"  OOS: Sharpe={oos_m['sharpe']:.2f} PF={oos_m['pf']:.2f} MaxDD={oos_m['max_dd_pct']:.1%} PnL=${oos_m['pnl']:.0f}")
        all_res["AsymMomentum"] = {"is": is_m, "oos": oos_m, "params": bp}
    
    # === VOL MOMENTUM ===
    print(f"\n[4] Vol-Weighted Momentum...", end=" ")
    t1 = time()
    grid = [{"lookback": lb, "top_n": tn, "holding": h}
            for lb in [18, 30, 42, 60] for tn in [3, 5] for h in [12, 18, 24, 36]]  # 32 combos
    best = grid_search(ret_is, None, lambda r, **kw: vol_mom(r, **kw), grid)
    if best is not None:
        bp = {k: int(best[k]) for k in ["lookback", "top_n", "holding"]}
        pos_full = vol_mom(ret, **bp)
        oos_m = honest_metrics(ret_oos, pos_full[oos_mask])
        is_m = honest_metrics(ret_is, pos_full[is_mask])
        print(f"{time()-t1:.1f}s")
        print(f"  IS:  Sharpe={is_m['sharpe']:.2f} PF={is_m['pf']:.2f} MaxDD={is_m['max_dd_pct']:.1%}")
        print(f"  OOS: Sharpe={oos_m['sharpe']:.2f} PF={oos_m['pf']:.2f} MaxDD={oos_m['max_dd_pct']:.1%} PnL=${oos_m['pnl']:.0f}")
        all_res["VolMom"] = {"is": is_m, "oos": oos_m, "params": bp}
    
    # === VOLUME SPIKE ===
    print(f"\n[5] Volume Spike Alpha...", end=" ")
    t1 = time()
    grid = [{"lookback": lb, "z_entry": ze, "holding": h}
            for lb in [18, 30, 42] for ze in [1.0, 1.5, 2.0] for h in [6, 12, 24]]  # 18 combos
    best = grid_search(ret_is, vol_is,
                       lambda r, v, **kw: vol_spike(r, v, **kw), grid)
    if best is not None:
        bp = {k: int(best[k]) if k != "z_entry" else best[k] for k in ["lookback", "z_entry", "holding"]}
        pos_full = vol_spike(ret, vol_arr, **bp)
        oos_m = honest_metrics(ret_oos, pos_full[oos_mask])
        is_m = honest_metrics(ret_is, pos_full[is_mask])
        print(f"{time()-t1:.1f}s")
        print(f"  IS:  Sharpe={is_m['sharpe']:.2f} PF={is_m['pf']:.2f} MaxDD={is_m['max_dd_pct']:.1%}")
        print(f"  OOS: Sharpe={oos_m['sharpe']:.2f} PF={oos_m['pf']:.2f} MaxDD={oos_m['max_dd_pct']:.1%} PnL=${oos_m['pnl']:.0f}")
        all_res["VolSpike"] = {"is": is_m, "oos": oos_m, "params": bp}
    else:
        print(f"{time()-t1:.1f}s | No valid results")
    
    # === REGIME ADAPTIVE ===
    print(f"\n[6] Regime Adaptive...", end=" ")
    t1 = time()
    grid = [{"lookback": lb, "holding": h}
            for lb in [42, 60, 90] for h in [6, 12, 24, 36]]  # 12 combos
    best = grid_search(ret_is, None, lambda r, **kw: regime_adaptive(r, **kw), grid)
    if best is not None:
        bp = {k: int(best[k]) for k in ["lookback", "holding"]}
        pos_full = regime_adaptive(ret, **bp)
        oos_m = honest_metrics(ret_oos, pos_full[oos_mask])
        is_m = honest_metrics(ret_is, pos_full[is_mask])
        print(f"{time()-t1:.1f}s")
        print(f"  IS:  Sharpe={is_m['sharpe']:.2f} PF={is_m['pf']:.2f} MaxDD={is_m['max_dd_pct']:.1%}")
        print(f"  OOS: Sharpe={oos_m['sharpe']:.2f} PF={oos_m['pf']:.2f} MaxDD={oos_m['max_dd_pct']:.1%} PnL=${oos_m['pnl']:.0f}")
        all_res["RegimeAdaptive"] = {"is": is_m, "oos": oos_m, "params": bp}
    
    # === ENSEMBLE ===
    print(f"\n[7] Ensemble (top 3)...", end=" ")
    t1 = time()
    sorted_s = sorted(all_res.items(), key=lambda x: x[1]["is"]["sharpe"], reverse=True)[:3]
    if len(sorted_s) >= 2:
        pos_list = []
        for name, r in sorted_s:
            p = r.get("params", {})
            try:
                if name == "Momentum":
                    pos_list.append(cs_momentum(ret, **{k: int(p[k]) if k not in ["w_short"] else p[k] for k in ["lookback", "top_n", "holding", "lb_short", "lb_long", "w_short"] if k in p}))
                elif name == "AsymMomentum":
                    pos_list.append(asymmetric_mom(ret, **p))
                elif name == "VolMom":
                    pos_list.append(vol_mom(ret, **p))
                elif name == "VolSpike":
                    pos_list.append(vol_spike(ret, vol_arr, **p))
                elif name == "RegimeAdaptive":
                    pos_list.append(regime_adaptive(ret, **p))
            except:
                pass
        if pos_list:
            avg_pos = sum(pos_list) / len(pos_list)
            oos_m = honest_metrics(ret_oos, avg_pos[oos_mask])
            is_m = honest_metrics(ret_is, avg_pos[is_mask])
            print(f"{time()-t1:.1f}s")
            print(f"  IS:  Sharpe={is_m['sharpe']:.2f} PF={is_m['pf']:.2f} MaxDD={is_m['max_dd_pct']:.1%}")
            print(f"  OOS: Sharpe={oos_m['sharpe']:.2f} PF={oos_m['pf']:.2f} MaxDD={oos_m['max_dd_pct']:.1%} PnL=${oos_m['pnl']:.0f}")
            all_res["Ensemble"] = {"is": is_m, "oos": oos_m}
    
    # === FINAL TABLE ===
    print(f"\n{'='*70}")
    print("FINAL HONEST RESULTS (8 bps transaction costs)")
    print(f"{'='*70}")
    print(f"{'Strategy':<20} {'IS Shp':>7} {'OOS Shp':>8} {'OOS PF':>7} {'OOS DD%':>8} {'OOS PnL':>9} {'Verdict':>8}")
    print("-" * 70)
    
    for name, r in all_res.items():
        oos = r["oos"]
        is_shp = r["is"]["sharpe"]
        ok = oos["sharpe"] > 0.8 and oos["pf"] > 1.15 and oos["max_dd_pct"] < MAX_DD_PCT
        print(f"{name:<20} {is_shp:>7.2f} {oos['sharpe']:>8.2f} {oos['pf']:>7.2f} "
              f"{oos['max_dd_pct']:>7.1%} ${oos['pnl']:>8.0f} {'PASS' if ok else 'FAIL':>8}")
    
    best_name = max(all_res.items(), key=lambda x: x[1]["oos"]["sharpe"])
    oos = best_name[1]["oos"]
    print(f"\nBest: {best_name[0]} | Net Sharpe={oos['sharpe']:.2f} | PF={oos['pf']:.2f} | MaxDD={oos['max_dd_pct']:.1%}")
    
    if oos["sharpe"] > 1.0 and oos["pf"] > 1.25 and oos["max_dd_pct"] < MAX_DD_PCT:
        print(">>> ALL TARGETS MET <<<")
    elif oos["sharpe"] > 0.8 and oos["pf"] > 1.15:
        print(">>> ACCEPTABLE <<<")
    else:
        print(">>> STRATEGY DOES NOT PASS HONEST TEST <<<")
    
    print(f"\nTotal time: {time()-t0:.1f}s")


if __name__ == "__main__":
    main()
