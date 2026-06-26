"""Data quality audit for mimo_code crypto data."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/binance_4h")
EXCLUDE = ["ARB_USDT", "OP_USDT"]

# ── 1. Структура данных ──
print("=" * 60)
print("DATA QUALITY AUDIT")
print("=" * 60)

frames = []
for f in sorted(DATA_DIR.glob("*.parquet")):
    symbol = f.stem.replace("_4h", "")
    if symbol in EXCLUDE:
        continue
    df = pd.read_parquet(f)
    df["symbol"] = symbol
    # Check data types
    print(f"  {symbol:>12}: types={dict(df.dtypes)}, memory={f.stat().st_size/1024:.0f}KB")
    # Check for NaNs per column
    nans = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().sum() > 0}
    if nans:
        print(f"               NaNs: {nans}")
    frames.append(df)

merged = pd.concat(frames, ignore_index=True)

print(f"\nMerged: {len(merged)} rows, {merged['symbol'].nunique()} symbols")
print(f"Period: {merged['datetime'].min()} — {merged['datetime'].max()}")
print(f"Timezone: {merged['datetime'].dt.tz}")

# ── 2. Проверка pivot — дубликаты в индексе ──
close = merged.pivot_table(index="datetime", columns="symbol", values="close").sort_index()
idx_dupes = close.index.duplicated().sum()
if idx_dupes:
    print(f"\n⚠️  DUPLICATE INDEX: {idx_dupes} duplicated timestamps!")
else:
    print(f"\n✅ Index: {len(close)} unique timestamps, no duplicates")

# ── 3. NaN в каждом символе ДО fill ──
print("\n── NaN per symbol (before fill) ──")
nans_before = close.isna().sum()
bad_symbols = nans_before[nans_before > 0]
for s, n in bad_symbols.sort_values(ascending=False).items():
    print(f"  ⚠️  {s:>12}: {n:>5} NaN ({n/len(close)*100:.2f}%)")
if len(bad_symbols) == 0:
    print("  ✅ None found — all symbols have complete data")

# ── 4. Имитация load_data() ──
print("\n── Simulating load_data() ──")
miss = close.isna().mean()
close_f = close[miss[miss < 0.3].index].dropna()
print(f"  Excluded symbols (>{30}% NaN): {set(close.columns) - set(close_f.columns)}")
print(f"  After dropna: {close_f.shape[1]} symbols, {len(close_f)} bars")

# ── 5. Проверка на свечи с H/L в которых close вне диапазона ──
print("\n── Sanity: price integrity ──")
issues = 0
for symbol in merged["symbol"].unique():
    sdf = merged[merged["symbol"] == symbol]
    outside = (sdf["close"] > sdf["high"]) | (sdf["close"] < sdf["low"]) | \
              (sdf["high"] < sdf["low"]) | (sdf["open"] > sdf["high"]) | \
              (sdf["open"] < sdf["low"])
    n = outside.sum()
    if n > 0:
        issues += 1
        print(f"  ⚠️  {symbol}: {n} bars with invalid OHLC (close outside high/low)")
if issues == 0:
    print("  ✅ No data integrity issues found")

# ── 6. Проверка stale pricing ──
print("\n── Stale pricing (>=4 bars consecutive zero change) ──")
for symbol in close.columns:
    diffs = close[symbol].diff()
    zero_run = (diffs == 0).astype(int).groupby((diffs != 0).cumsum()).transform('sum')
    stale = zero_run[zero_run >= 4].sum()
    if stale > 0:
        print(f"  📌 {symbol}: {int(stale)} bars in stale runs")

print("\n── Volume sanity ──")
for symbol in merged["symbol"].unique():
    sdf = merged[merged["symbol"] == symbol]
    vol = sdf["volume"]
    zero_pct = (vol == 0).mean() * 100
    if zero_pct > 5:
        print(f"  ⚠️  {symbol}: {zero_pct:.1f}% bars with zero volume!")
    spike = (vol / vol.rolling(100).mean()).max()
    if spike > 20:
        print(f"  📌 {symbol}: max volume spike = {spike:.0f}x median (check for data errors)")

print("\n── Cross-asset alignment ──")
# Are all symbols aligned on the same timestamps?
align_check = close.notna().all(axis=1)
if align_check.mean() < 0.99:
    print(f"  ⚠️  {align_check.mean()*100:.1f}% of bars have ALL symbols present")
    missing_dates = close.index[~align_check][:10]
    print(f"  Partial bars: {len(close) - align_check.sum()}")
else:
    print(f"  ✅ {align_check.mean()*100:.1f}% of bars have ALL {close.shape[1]} symbols present")

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)