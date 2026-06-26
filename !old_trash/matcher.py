import pandas as pd
import numpy as np


def filter_spot_and_swap(df: pd.DataFrame) -> pd.DataFrame:
    """Оставить только spot и perpetual (swap) рынки."""
    return df[df["market_type"].isin(["spot", "swap"])].copy()


def find_common_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Найти инструменты на ВСЕХ 3 биржах в ОБОИХ типах (spot + swap).
    """
    filtered = filter_spot_and_swap(df)
    filtered = filtered[filtered["active"] == True].copy()
    filtered["present"] = True

    pivot = filtered.pivot_table(
        index="base",
        columns=["exchange", "market_type"],
        values="present",
        aggfunc="first",
    ).fillna(False)

    required_cols = [
        ("binance", "spot"), ("binance", "swap"),
        ("bybit", "spot"), ("bybit", "swap"),
        ("okx", "spot"), ("okx", "swap"),
    ]
    available = [c for c in required_cols if c in pivot.columns]

    if len(available) < 6:
        print(f"Отсутствуют: {set(required_cols) - set(available)}")

    mask = pivot[available].all(axis=1)
    result = pivot[mask].reset_index()
    result.columns = ["base"] + [f"{ex}_{mt}" for ex, mt in pivot[mask].columns]

    return result.sort_values("base").reset_index(drop=True)


def add_volume(common: pd.DataFrame, tickers: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить 24h объём к общим инструментам.
    Суммирует quoteVolume по всем биржам и типам (spot+swap).
    """
    # Фильтруем только нужные base
    common_bases = set(common["base"])
    tickers = tickers[tickers["base"].isin(common_bases)].copy()

    # Суммарный объём по base (quote volume в USDT)
    volume_by_base = tickers.groupby("base")["quote_volume_24h"].sum().reset_index()
    volume_by_base.columns = ["base", "total_volume_24h"]

    # Объём по каждой бирже отдельно
    vol_by_exchange = tickers.groupby(["base", "exchange"])["quote_volume_24h"].sum().unstack(fill_value=0)
    vol_by_exchange.columns = [f"vol_{ex}" for ex in vol_by_exchange.columns]

    result = common.merge(volume_by_base, on="base", how="left")
    result = result.merge(vol_by_exchange, on="base", how="left")
    result["total_volume_24h"] = result["total_volume_24h"].fillna(0)

    return result.sort_values("total_volume_24h", ascending=False).reset_index(drop=True)


def get_stats(df_all: pd.DataFrame, common: pd.DataFrame, since_year: int = 2022) -> pd.DataFrame:
    """Собрать символы и market ID с каждой биржи."""
    filtered = filter_spot_and_swap(df_all)
    filtered = filtered[filtered["active"] == True].copy()

    stats_rows = []
    for _, row in common.iterrows():
        base = row["base"]
        sub = filtered[filtered["base"] == base]

        stat = {"base": base}
        for ex in ["binance", "bybit", "okx"]:
            for mt in ["spot", "swap"]:
                col = f"{ex}_{mt}"
                mt_data = sub[(sub["exchange"] == ex) & (sub["market_type"] == mt)]
                if len(mt_data) > 0:
                    stat[f"{col}_symbol"] = mt_data["symbol"].iloc[0]
                    stat[f"{col}_id"] = mt_data["id"].iloc[0]
                else:
                    stat[f"{col}_symbol"] = ""
                    stat[f"{col}_id"] = ""
        stats_rows.append(stat)

    return pd.DataFrame(stats_rows)
