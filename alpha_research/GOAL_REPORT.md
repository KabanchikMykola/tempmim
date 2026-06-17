# Goal Mode: Signal Search Report

## Goal
Find 3 stable signals for crypto trading with:
- Walk-forward IC > 0.03 (30d train / 7d test)
- Holdout accuracy > 52% (last 14 days)
- IC > 0 in EVERY walk-forward window (100% stability)

## Hypotheses Tested

### 1. Basis Z-score → forward return
- **Pairs tested:** 21
- **Best:** JTO basis_z → 24h: IC=0.072, stable=61%, HO=55.5%
- **Status:** FAIL (stability 61% < 100%)

### 2. Volume spike → reversal
- **Pairs tested:** 21
- **Best:** BTC vol_ratio → 6h: IC=0.053, stable=68%, HO=50.4%
- **Status:** FAIL (stability 68% < 100%)

### 3. RSI extreme + vol regime → directional
- **Pairs tested:** 21
- **Best:** No pair exceeds IC>0.02
- **Status:** FAIL (IC too low)

### 4. Cross-asset: BTC return → altcoin (lagged)
- **Pairs tested:** 20
- **Best:** Mean IC negative
- **Status:** FAIL (no predictive power)

### 5. Volatility compression → breakout
- **Pairs tested:** 21
- **Best:** Mean IC negative
- **Status:** FAIL (no predictive power)

### Composite signals (v2)
- basis_z × vol_rank, rsi_vol, combined_4factor, etc.
- All FAIL (IC near 0, stability 0%)

### Per-pair analysis (v3)
- Top candidates: JTO, BTC, ENA, ASTER, WLD
- All have IC > 0.02 and HO > 50%
- But max stability = 78% (ENA basis_z 48h)

## Conclusion

**100% stability is unachievable** with:
- 22 pairs × 5.5 months of hourly data
- ~20 walk-forward windows per signal
- Market noise at hourly frequency

The best achievable is ~78% stability (ENA basis_z 48h).

**Recommendation:** If the goal is to find actionable signals, relax stability to ≥70%.
If the goal is strictly 100% stability, this dataset cannot produce it.
