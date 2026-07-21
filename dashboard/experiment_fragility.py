"""
Ch.3 Experiment: Fragility Target (1990+ extended data)

Core hypothesis: predicting continuous fragility (rather than binary crash)
produces more stable walk-forward results, because fragility is an observable
market structure property while crash triggers are random.

Three candidate targets:
  A. Vol Surprise = realized_vol_20d / current_VIX (market underpricing risk)
  B. Continuous Max DD = max_drawdown_next_20d (no binarization)
  C. Structural Fragility = absorption_ratio_percentile × VIX_percentile

Evaluation:
  - Regression (not classification) — Ridge / GBDT
  - Walk-forward with Spearman rank correlation as primary metric
  - R² and MSE as secondary
  - Compare stability (std across folds) vs Ch.2 binary models
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from predict_model import build_features_slim
from experiment_extended_history import load_extended_data
from experiment_phase3 import percentile_clip

DATA_DIR = Path(__file__).parent / 'data'
EMBARGO = 20


def compute_target_vol_surprise(df, horizon=20):
    """Target A: realized vol / implied vol (VIX).
    High = market underpriced risk, volatility surprise."""
    sp500 = df['sp500'].astype(float)
    returns = sp500.pct_change()

    realized_vol = returns.rolling(horizon).std() * np.sqrt(252)
    realized_vol_fwd = realized_vol.shift(-horizon)

    vix_normalized = df['vix'].astype(float) / 100.0

    surprise = realized_vol_fwd / vix_normalized.clip(lower=0.05)
    surprise = surprise.clip(0, 5)
    return surprise


def compute_target_max_dd(df, horizon=20):
    """Target B: continuous max drawdown over next horizon days.
    Values between 0 (no drawdown) and -1 (total loss)."""
    sp500 = df['sp500'].astype(float)
    max_dd = pd.Series(index=sp500.index, dtype=float)
    for i in range(len(sp500) - horizon):
        current = sp500.iloc[i]
        future = sp500.iloc[i+1:i+horizon+1]
        dd = (future.min() / current) - 1
        max_dd.iloc[i] = dd
    return -max_dd.clip(upper=0)


def compute_target_structural_fragility(df, window=60):
    """Target C: absorption_ratio_percentile × VIX_percentile.
    Both normalized to [0,1] using rolling percentile rank."""
    ar = df['absorption_ratio'].astype(float)
    vix = df['vix'].astype(float)

    ar_pctile = ar.rolling(window * 4, min_periods=window).rank(pct=True)
    vix_pctile = vix.rolling(window * 4, min_periods=window).rank(pct=True)

    fragility = ar_pctile * vix_pctile
    return fragility


def prep_regression_data(features, target):
    """Combine features + continuous target, drop NaN, clip outliers."""
    combined = features.copy()
    combined['target'] = target
    core_cols = [c for c in combined.columns if c.startswith(('vix_', 'sp500_'))]
    core_cols.append('target')
    combined = combined.dropna(subset=core_cols)
    combined = combined.fillna(0)
    combined.index = pd.to_datetime(combined.index)
    feat_cols = [c for c in combined.columns if c != 'target']
    for c in feat_cols:
        lo, hi = combined[c].quantile(0.01), combined[c].quantile(0.99)
        combined[c] = combined[c].clip(lo, hi)
    X = combined.drop('target', axis=1)
    y = combined['target']
    return X, y


def generate_wf_folds(n_samples, min_train_years=10, step_years=3):
    tpy = 252
    min_train = min_train_years * tpy
    step = step_years * tpy
    folds = []
    fold_num = 1
    train_end = min_train
    while train_end + EMBARGO + step <= n_samples:
        test_start = train_end + EMBARGO
        test_end = min(test_start + step, n_samples)
        folds.append({'fold': fold_num, 'train_end': train_end, 'test_start': test_start, 'test_end': test_end})
        fold_num += 1
        train_end += step
    return folds


def train_ridge(X_train, y_train, X_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = Ridge(alpha=1.0)
    model.fit(Xtr, y_train)
    return model.predict(Xte)


def train_gbdt_reg(X_train, y_train, X_test):
    model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=4, learning_rate=0.03,
        min_samples_leaf=30, l2_regularization=2.0, random_state=42,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def run_walk_forward_regression(X, y, model_name, target_name, train_fn, folds):
    results = []
    for fold in folds:
        X_train = X.iloc[:fold['train_end']]
        X_test = X.iloc[fold['test_start']:fold['test_end']]
        y_train = y.iloc[:fold['train_end']]
        y_test = y.iloc[fold['test_start']:fold['test_end']]

        if len(y_test) < 50:
            continue

        y_pred = train_fn(X_train, y_train, X_test)

        rank_corr, _ = spearmanr(y_test, y_pred)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        mse = float(np.mean((y_test - y_pred) ** 2))

        results.append({
            'fold': fold['fold'],
            'rank_corr': round(rank_corr, 4) if np.isfinite(rank_corr) else 0,
            'r2': round(r2, 4),
            'mse': round(mse, 6),
            'test_start': str(X_test.index[0].date()),
            'test_end': str(X_test.index[-1].date()),
        })

    if not results:
        return None

    return {
        'model_name': model_name,
        'target_name': target_name,
        'n_folds': len(results),
        'rank_corr_mean': round(np.mean([r['rank_corr'] for r in results]), 4),
        'rank_corr_std': round(np.std([r['rank_corr'] for r in results]), 4),
        'r2_mean': round(np.mean([r['r2'] for r in results]), 4),
        'r2_std': round(np.std([r['r2'] for r in results]), 4),
        'mse_mean': round(np.mean([r['mse'] for r in results]), 6),
        'folds': results,
    }


def main():
    print("=" * 60)
    print("Ch.3: FRAGILITY TARGET EXPERIMENT (1990+)")
    print("=" * 60)

    print("\n[1/4] Loading extended data...")
    df = load_extended_data()
    print(f"  {len(df)} trading days")

    print("\n[2/4] Computing fragility targets...")
    targets = {
        'Vol Surprise': compute_target_vol_surprise(df),
        'Max DD (continuous)': compute_target_max_dd(df),
        'Structural Fragility': compute_target_structural_fragility(df),
    }
    target_stats = []
    for name, t in targets.items():
        valid = t.dropna()
        stats = {'mean': round(float(valid.mean()), 4), 'std': round(float(valid.std()), 4),
                 'min': round(float(valid.min()), 4), 'max': round(float(valid.max()), 4)}
        target_stats.append({'name': name, 'description': '', 'stats': stats})
        print(f"  {name}: mean={stats['mean']:.4f} std={stats['std']:.4f} "
              f"range=[{stats['min']:.4f}, {stats['max']:.4f}] ({len(valid)} valid)")

    print("\n[3/4] Building features & walk-forward...")
    features = build_features_slim(df)
    folds = generate_wf_folds(len(features.dropna()))
    print(f"  {folds[-1]['fold'] if folds else 0} folds")

    all_results = []
    model_fns = {
        'Ridge': train_ridge,
        'GBDT': train_gbdt_reg,
    }

    for target_name, target_series in targets.items():
        X, y = prep_regression_data(features, target_series)
        target_folds = generate_wf_folds(len(X))
        print(f"\n  --- Target: {target_name} ({len(X)} samples, {len(target_folds)} folds) ---")

        for model_name, fn in model_fns.items():
            result = run_walk_forward_regression(X, y, model_name, target_name, fn, target_folds)
            if result:
                all_results.append(result)
                print(f"    {model_name}: RankCorr={result['rank_corr_mean']:.4f}±{result['rank_corr_std']:.4f} "
                      f"R²={result['r2_mean']:.4f} MSE={result['mse_mean']:.6f}")

    print("\n[4/4] Saving results...")
    best = max(all_results, key=lambda x: x['rank_corr_mean']) if all_results else None

    verdict = []
    if best:
        verdict.append(f"Best: {best['model_name']} × {best['target_name']} "
                       f"(Rank Corr = {best['rank_corr_mean']:.4f})")
        if best['rank_corr_mean'] > 0.3:
            verdict.append("Rank Corr > 0.3: 脆弱性信号有显著排序能力，可用于仓位缩放")
        elif best['rank_corr_mean'] > 0.15:
            verdict.append("Rank Corr 0.15-0.30: 弱信号，作为复合因子的一部分有价值")
        else:
            verdict.append("Rank Corr < 0.15: 信号微弱，需要换特征或换 target 定义")

        ch2_f1 = 0.31
        verdict.append(f"对比 Ch.2 binary (WF F1={ch2_f1}): "
                       f"{'脆弱性 target 跨 fold 更稳定' if best['rank_corr_std'] < 0.1 else '稳定性未明显改善'}")

    output = {
        'title': 'Ch.3 Fragility Target Experiment',
        'hypothesis': '预测连续脆弱性（而非 binary crash）能获得更稳定的 walk-forward 信号',
        'data_range': f"{df.index[0]} ~ {df.index[-1]}",
        'n_samples': len(df),
        'targets': target_stats,
        'walk_forward_results': all_results,
        'verdict': verdict,
    }

    out_path = DATA_DIR / 'fragility_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved {out_path.name}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  {'Model × Target':<35} {'Rank Corr':>15} {'R²':>12}")
    print(f"  {'-'*65}")
    for r in sorted(all_results, key=lambda x: -x['rank_corr_mean']):
        print(f"  {r['model_name']+' × '+r['target_name']:<35} "
              f"{r['rank_corr_mean']:.4f}±{r['rank_corr_std']:.4f}  "
              f"{r['r2_mean']:.4f}±{r['r2_std']:.4f}")
    if verdict:
        print(f"\n  Verdict: {verdict[0]}")
    print("=" * 60)


if __name__ == '__main__':
    main()
