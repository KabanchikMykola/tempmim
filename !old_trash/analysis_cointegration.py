"""Анализ коинтеграции + Калман (логарифмы) + бэктест.

Версия 3: непрерывный Калман, IS/OOS, оптимизация, честный Sharpe.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations, product
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("data/binance")
SPLIT_DATE = "2025-01-01"
CAPITAL = 10000.0
MIN_TRADES = 10


def load_all() -> pd.DataFrame:
    frames = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        frames.append(pd.read_parquet(f))
    return pd.concat(frames, ignore_index=True)


def get_close_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(index="datetime", columns="symbol", values="close")
    return pivot.sort_index()


def kalman_pair_regression(y, x, delta=1e-4, R=1e-2):
    """Динамическая регрессия Калмана: y = beta * x + alpha."""
    n = len(y)
    state_means = np.zeros((n, 2))
    P = np.zeros((2, 2))
    Q = np.eye(2) * delta
    spread_res = np.zeros(n)

    for t in range(n):
        if t > 0:
            P = P + Q

        H = np.array([1.0, x[t]])
        y_pred = np.dot(H, state_means[t - 1 if t > 0 else 0])
        v = y[t] - y_pred

        S = np.dot(H, np.dot(P, H.T)) + R
        K = np.dot(P, H.T) / S

        state_means[t] = state_means[t - 1 if t > 0 else 0] + K * v
        P = P - np.outer(K, np.dot(H, P))

        spread_res[t] = v

    return state_means[:, 1], spread_res


def compute_zscore(spread, span=10):
    """Z-score через экспоненциальное окно."""
    s = pd.Series(spread)
    mean = s.ewm(span=span, adjust=False).mean()
    std = s.ewm(span=span, adjust=False).std()
    std[std == 0] = 1e-8
    return ((s - mean) / std).values


def backtest_pair(pair_df, s1, s2, inertia=0.05, span=10, z_entry=1.5, eval_start_date=None):
    """Бэктест: Калман на всей истории, метрики на eval-интервале."""
    y = np.log(pair_df[s1].values)
    x = np.log(pair_df[s2].values)

    betas, spread = kalman_pair_regression(y, x)
    z = compute_zscore(spread, span=span)

    signal = pd.Series(0, index=pair_df.index)
    signal[z > z_entry] = -1
    signal[z < -z_entry] = 1
    signal[np.abs(z) < 0.5] = 0

    pos = signal.shift(1).fillna(0)

    betas_adj = np.copy(betas)
    last_beta = betas[0]
    for t in range(1, len(betas)):
        if abs(betas[t] - last_beta) > inertia:
            last_beta = betas[t]
        betas_adj[t] = last_beta

    log_ret_s1 = np.diff(y, prepend=y[0])
    log_ret_s2 = np.diff(x, prepend=x[0])

    strat_ret_pct = pos.values * (log_ret_s1 - betas_adj * log_ret_s2)
    strat_ret_dollars = strat_ret_pct * CAPITAL

    prices_s1 = pair_df[s1].values
    prices_s2 = pair_df[s2].values
    trades_s1 = np.abs(np.diff(pos.values, prepend=0))
    trades_s2 = np.abs(np.diff(pos.values * betas_adj, prepend=0))
    notional = trades_s1 * prices_s1 + trades_s2 * prices_s2
    commissions = notional * 0.001

    strat_ret_net = strat_ret_dollars - commissions

    # Срез для оценки
    if eval_start_date is not None:
        mask = pair_df.index >= eval_start_date
        strat_ret_net = strat_ret_net[mask]
        pos_eval = pos.values[mask]
    else:
        pos_eval = pos.values

    cum_pnl = np.cumsum(strat_ret_net)
    total = cum_pnl[-1] if len(cum_pnl) > 0 else 0
    n_trades = int(np.abs(np.diff(pos_eval, prepend=0)).sum() / 2)

    ret_std = np.std(strat_ret_net) if len(strat_ret_net) > 0 else 1
    if ret_std < 1e-4:
        ret_std = 1.0
    sharpe = np.mean(strat_ret_net) / ret_std * np.sqrt(365) if len(strat_ret_net) > 0 else 0

    max_dd = np.max(np.maximum.accumulate(cum_pnl) - cum_pnl) if len(cum_pnl) > 0 else 0
    win_rate = (strat_ret_net[strat_ret_net != 0] > 0).mean() if (strat_ret_net != 0).any() else 0

    return {
        "pair": f"{s1}/{s2}", "s1": s1, "s2": s2,
        "trades": n_trades, "pnl": total, "sharpe": sharpe,
        "max_dd": max_dd, "win_rate": win_rate,
    }


def optimize_pair(pair_df, s1, s2):
    """Grid Search по параметрам на In-Sample данных."""
    spans = [10, 20, 30]
    inertias = [0.01, 0.05, 0.10]
    z_entries = [1.5, 2.0]

    best_sharpe = -np.inf
    best_params = {"span": 10, "inertia": 0.05, "z_entry": 1.5}

    for span, inertia, z_entry in product(spans, inertias, z_entries):
        res = backtest_pair(pair_df, s1, s2, inertia=inertia, span=span, z_entry=z_entry)
        if res["sharpe"] > best_sharpe:
            best_sharpe = res["sharpe"]
            best_params = {"span": span, "inertia": inertia, "z_entry": z_entry}

    return best_params, best_sharpe


def main():
    print("Загрузка данных...")
    df = load_all()
    close = get_close_matrix(df)
    symbols = close.columns.tolist()
    print(f"Активов: {len(symbols)}, баров: {len(close)}")

    close_is = close.loc[close.index < SPLIT_DATE]
    close_oos = close.loc[close.index >= SPLIT_DATE]
    print(f"In-Sample: {close_is.index[0].date()} — {close_is.index[-1].date()} ({len(close_is)} баров)")
    print(f"Out-of-Sample: {close_oos.index[0].date()} — {close_oos.index[-1].date()} ({len(close_oos)} баров)")

    # === ЭТАП 1: Скрининг + Коинтеграция (In-Sample) ===
    print(f"\n{'='*70}")
    print("ЭТАП 1: Скрининг + Коинтеграция (In-Sample)")
    print(f"{'='*70}")

    corr_matrix = close_is.corr()
    pairs_corr = []
    for s1, s2 in combinations(symbols, 2):
        corr = corr_matrix.loc[s1, s2]
        if abs(corr) > 0.7:
            pairs_corr.append((s1, s2, corr))
    pairs_corr.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"Пар с |corr| > 0.7: {len(pairs_corr)}")

    print(f"\nТест коинтеграции на топ-{min(20, len(pairs_corr))} пар...")
    coint_results = []
    for s1, s2, corr in pairs_corr[:20]:
        pair = close_is[[s1, s2]].dropna()
        if len(pair) < 100:
            continue
        try:
            log_pair = np.log(pair)
            score, pvalue, _ = coint(log_pair[s1], log_pair[s2])
            coint_results.append({
                "pair": f"{s1}/{s2}", "s1": s1, "s2": s2,
                "eg_pvalue": pvalue, "corr": corr,
            })
        except Exception:
            pass

    coint_df = pd.DataFrame(coint_results).sort_values("eg_pvalue")
    top = coint_df.head(10)

    print(f"\nТоп-10 пар (In-Sample):")
    print(top[["pair", "eg_pvalue", "corr"]].to_string(index=False))

    # === ЭТАП 2: Оптимизация (Grid Search, In-Sample) ===
    print(f"\n{'='*70}")
    print("ЭТАП 2: Оптимизация параметров (Grid Search, In-Sample)")
    print(f"{'='*70}")

    optimized = []
    for _, row in top.iterrows():
        s1, s2 = row["s1"], row["s2"]
        pair_is = close_is[[s1, s2]].dropna()

        best_params, best_sharpe = optimize_pair(pair_is, s1, s2)
        optimized.append({
            "pair": f"{s1}/{s2}", "s1": s1, "s2": s2,
            "eg_pvalue": row["eg_pvalue"], "corr": row["corr"],
            "span": best_params["span"],
            "inertia": best_params["inertia"],
            "z_entry": best_params["z_entry"],
            "is_sharpe": best_sharpe,
        })

        print(f"  {s1}/{s2}: span={best_params['span']}, inertia={best_params['inertia']}, "
              f"z_entry={best_params['z_entry']}, IS_Sharpe={best_sharpe:.2f}")

    # === ЭТАП 3: Бэктест на In-Sample ===
    print(f"\n{'='*70}")
    print("ЭТАП 3: Бэктест на In-Sample (с оптимальными параметрами)")
    print(f"{'='*70}")

    is_results = []
    for opt in optimized:
        s1, s2 = opt["s1"], opt["s2"]
        pair_is = close_is[[s1, s2]].dropna()

        res = backtest_pair(pair_is, s1, s2,
                            inertia=opt["inertia"],
                            span=opt["span"],
                            z_entry=opt["z_entry"])
        is_results.append(res)

        print(f"  {res['pair']} | IS_Sharpe={res['sharpe']:.2f} | "
              f"Trades={res['trades']} | WR={res['win_rate']:.1%} | "
              f"PnL=${res['pnl']:.2f} | MaxDD=${res['max_dd']:.2f}")

    # === ЭТАП 4: Out-of-Sample (Калман на ВСЕЙ истории, метрики только OOS) ===
    print(f"\n{'='*70}")
    print("ЭТАП 4: Out-of-Sample (Калман на полной истории, метрики с 2025-01-01)")
    print(f"{'='*70}")

    oos_results = []
    for opt in optimized:
        s1, s2 = opt["s1"], opt["s2"]

        # Полная история для непрерывного Калмана
        pair_all = close[[s1, s2]].dropna()

        if len(pair_all.loc[pair_all.index >= SPLIT_DATE]) < 30:
            print(f"  {s1}/{s2}: мало данных OOS, пропуск")
            continue

        res = backtest_pair(pair_all, s1, s2,
                            inertia=opt["inertia"],
                            span=opt["span"],
                            z_entry=opt["z_entry"],
                            eval_start_date=SPLIT_DATE)
        res["is_sharpe"] = opt["is_sharpe"]
        oos_results.append(res)

        print(f"\n  {res['pair']} | IS_Sharpe={opt['is_sharpe']:.2f} → OOS_Sharpe={res['sharpe']:.2f}")
        print(f"    Trades={res['trades']} | WR={res['win_rate']:.1%} | "
              f"PnL=${res['pnl']:.2f} | MaxDD=${res['max_dd']:.2f}")

    # === ИТОГИ ===
    print(f"\n{'='*70}")
    print("ИТОГИ (Out-of-Sample)")
    print(f"{'='*70}")

    if oos_results:
        # Фильтр по минимальному числу сделок
        valid = [r for r in oos_results if r["trades"] >= MIN_TRADES]
        skipped = [r for r in oos_results if r["trades"] < MIN_TRADES]

        if skipped:
            print(f"Пропущены (< {MIN_TRADES} сделок): {', '.join(r['pair'] for r in skipped)}")

        winners = [r for r in valid if r["sharpe"] > 0]
        losers = [r for r in valid if r["sharpe"] <= 0]

        print(f"Прибыльных: {len(winners)}/{len(valid)}")
        print(f"Убыточных:  {len(losers)}/{len(valid)}")

        if winners:
            print("\nПрибыльные пары:")
            for r in sorted(winners, key=lambda x: x["sharpe"], reverse=True):
                print(f"  {r['pair']:>20} OOS_Sharpe={r['sharpe']:.2f} "
                      f"IS_Sharpe={r['is_sharpe']:.2f} PnL=${r['pnl']:.2f}")

        if losers:
            print("\nУбыточные пары:")
            for r in sorted(losers, key=lambda x: x["sharpe"]):
                print(f"  {r['pair']:>20} OOS_Sharpe={r['sharpe']:.2f} "
                      f"IS_Sharpe={r['is_sharpe']:.2f} PnL=${r['pnl']:.2f}")

        if valid:
            avg_oos = np.mean([r["sharpe"] for r in valid])
            avg_is = np.mean([r["is_sharpe"] for r in valid])
            print(f"\nСредний IS Sharpe:  {avg_is:.2f}")
            print(f"Средний OOS Sharpe: {avg_oos:.2f}")
            if avg_is != 0:
                deg = (avg_is - avg_oos) / abs(avg_is) * 100
                print(f"Деградация:         {deg:.0f}%")
    else:
        print("Нет результатов для OOS.")

    pd.DataFrame(oos_results).to_csv("data/cointegration_results_oos.csv", index=False)
    print(f"\nРезультаты: data/cointegration_results_oos.csv")


if __name__ == "__main__":
    main()
