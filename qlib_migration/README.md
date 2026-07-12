# Walk-Forward Ablation: IC-Weighted Linear vs LightGBM

Ablation study comparing three scoring models on Russell 1000 (~495 stocks after liquidity filter), using the same 5 factors (momentum, B/P, log market cap, low volatility, reversal) and the same backtest setup (Top-20 equal-weight, monthly rebalance, 10bps/side cost).

## Setup

- **Universe:** Russell 1000, $5M+ average daily volume filter
- **Period:** 2016-01 to 2024-12 (factor panel), backtest 2022-04 to 2024-12
- **Walk-forward:** 24-month training / 3-month test, 8 rolling folds
- **PCA:** Applied per fold (fit on train, transform test) to orthogonalize factors
- **Backtest engine:** Microsoft Qlib (`qlib_backtest_lib.py`)

## Models

| Model | Description |
|---|---|
| **IC-weighted linear** | Rolling 6M Spearman IC per factor, normalize to weights, composite z-score |
| **LightGBM Regression** | Predict next-month cross-sectional return, standard MSE loss |
| **LightGBM LambdaRank** | Learning-to-rank objective, optimizes for correct ordering rather than point prediction |

## Results

| Model | Sharpe (net) | Ann. Return (net) | Max Drawdown | Mean IC (backtest) | ICIR |
|---|---|---|---|---|---|
| **IC-weighted linear** | **0.56** | 12.1% | **-21.5%** | +0.016 | +0.082 |
| LightGBM LambdaRank | 0.44 | 13.7% | -26.9% | -0.010 | -0.062 |
| LightGBM Regression | 0.17 | 5.2% | -38.0% | -0.018 | -0.152 |

## Interpretation

On this universe and horizon, the linear IC-weighted ensemble achieves the best risk-adjusted return (Sharpe 0.56) with the shallowest drawdown (-21.5%). LightGBM LambdaRank has higher raw annual return but worse Sharpe and 5% deeper max drawdown.

The ML models show negative IC during the backtest sub-period, suggesting they overfit to the training window. With only 5 features and limited cross-sectional signal on monthly factors, LightGBM picks up noise rather than durable patterns. The linear model avoids this by design: it simply weights factors proportional to their recent realized IC, making no parametric assumptions.

This is consistent with the broader quant finding that simple, robust models often dominate complex ones on low-signal problems, especially when the feature space is small and the train/test regime shifts are frequent.

## Cross-Validation Against Qlib

The locked SP500 baseline (Sharpe 1.77, 30.0% gross annual) was independently reproduced using Microsoft Qlib's backtest engine. The difference between the custom pipeline and Qlib was <0.02% in net annual return, confirming that the portfolio construction logic is correctly implemented.

## Scripts

| File | Purpose |
|---|---|
| `build_lgbm_data_v2.py` | Build factor panel parquet (2016-2024) using agents pipeline |
| `walk_forward_lgbm_v2.py` | Walk-forward LightGBM training (Regression + LambdaRank) |
| `compare_lgbm_v2.py` | Compare all three models, generate charts and result JSON |
| `run_qlib_backtest.py` | Reproduce locked baseline via Qlib engine |
| `qlib_backtest_lib.py` | Reusable Qlib backtest wrapper |
| `dump_to_qlib_v2.py` | Dump price data to Qlib binary format |
