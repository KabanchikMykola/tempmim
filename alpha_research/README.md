# alpha_research — документация

Standalone скрипты (не импортируемый пакет). Каждый запускается независимо.

## Скрипты

| Скрипт | Что делает |
|---|---|
| `strategies_basis.py` | Walk-forward basis z-score strategy search |
| `strategies_xgboost.py` | XGBoost classifier for 12h direction prediction |
| `strategies_xgb_variations.py` | Batch XGBoost across horizons/features |
| `strategies_all.py` | Combined: basis + reversion + momentum + vol-squeeze |
| `strategies_cost_check.py` | Re-evaluate strategies with realistic costs |
| `strategies_lgbm_optuna.py` | LightGBM + Optuna hyperparameter search |
| `basis_walkforward.py` | Basis walk-forward with parameter sweep |
| `basis_realistic.py` | Basis backtest with realistic cost model |
| `signal_search_v4.py` | Signal search with strict IC stability |
| `volatility_regime_v3.py` | Volatility regime with dynamic exits |
| `strategies_nf_nhits.py` | NHITS (NeuralForecast) 12h direction classifier with spot+perp+metrics |
| `strategies.json` | Auto-generated: saved winning strategies |

### Archive

`archive/` — signal_search (v1-v3), volatility_regime (v1-v2). Superseded by latest versions.

## Формат данных

Все скрипты ожидают `data/top5_2026/` с:
- `{BASE}_USDT_1h.parquet` (spot)
- `{BASE}_USDT_USDT_1h.parquet` (perp)

## Запуск

```sh
python alpha_research/strategies_lgbm_optuna.py
python alpha_research/strategies_basis.py
```
