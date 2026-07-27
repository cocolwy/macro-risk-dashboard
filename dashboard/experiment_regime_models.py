"""
Experiment: Regime-Conditional & Non-Linear Models on Extended History (1990+)

Goal: Address feature–crash relationship flipping across regimes.
  - LR + regime interaction terms (baseline, extended data)
  - GBDT (captures nonlinear regime-dependent splits natively)
  - LR per-regime (separate models for 'tight' vs 'normal')
  - Walk-forward evaluation on 1990+ data (more episodes, fairer comparison)

Key insight from prior work:
  - Single-split F1=0.73 on short data collapsed to WF F1=0.19
  - Extended AUC dropped from 0.89 to 0.636 with LR
  - Hypothesis: GBDT and regime-conditioning can better generalize

Outputs: regime_model_metrics.json → merged into phase3_metrics.json
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, brier_score_loss

from predict_model import (
    build_features_slim, compute_target, fetch_regime_data,
)
from experiment_extended_history import load_extended_data, EXTENDED_KEY_EVENTS
from experiment_phase3 import (
    percentile_clip, compute_practical_metrics, detect_regime,
)

DATA_DIR = Path(__file__).parent / 'data'
EMBARGO = 20


def build_regime_features_extended(df, regime_df):
    """Slim features + regime context + interaction terms for extended data."""
    features = build_features_slim(df)

    if 'term_spread' in df.columns:
        ts = df['term_spread']
        features['curve_inverted'] = (ts < 0).astype(float)
        features['curve_steep'] = (ts > 2.0).astype(float)

    if regime_df is not None and 'fed_funds' in regime_df.columns:
        ff_idx = pd.to_datetime(regime_df.index)
        ff_series = regime_df['fed_funds'].copy()
        ff_series.index = ff_idx
        df_idx = pd.to_datetime(df.index)
        ff = ff_series.reindex(df_idx, method='ffill').fillna(method='bfill').fillna(0)
        ff.index = df.index
        features['fed_rate_level'] = ff.values
        chg63 = ff.diff(63).fillna(0)
        features['fed_hiking'] = (chg63 > 0.25).astype(float).values
        features['fed_cutting'] = (chg63 < -0.25).astype(float).values

    regime_labels = _detect_regime_extended(df, regime_df)
    regime_binary = (regime_labels == 'tight').astype(float)
    features['regime_tight'] = regime_binary.values

    for f in ['vix_level', 'credit_spread_10d_chg', 'sp500_vs_50ma', 'term_spread_level']:
        if f in features.columns:
            features[f'tight_x_{f}'] = regime_binary.values * features[f].values

    return features, regime_labels


def _detect_regime_extended(df, regime_df):
    """Regime detection that works with string-indexed extended data."""
    regime_labels = pd.Series('normal', index=df.index)

    inverted = pd.Series(False, index=df.index)
    if 'term_spread' in df.columns:
        inverted = df['term_spread'] < 0

    hiking = pd.Series(False, index=df.index)
    if regime_df is not None and 'fed_funds' in regime_df.columns:
        ff_idx = pd.to_datetime(regime_df.index)
        ff_series = regime_df['fed_funds'].copy()
        ff_series.index = ff_idx
        df_idx = pd.to_datetime(df.index)
        ff = ff_series.reindex(df_idx, method='ffill').fillna(method='bfill').fillna(0)
        ff.index = df.index
        hiking = ff.diff(63).fillna(0) > 0.25

    regime_labels[inverted | hiking] = 'tight'
    return regime_labels


def prep_data(features, target):
    """Standard prep: combine, drop NaN, percentile clip."""
    combined = features.copy()
    combined['target'] = target
    core_cols = [c for c in combined.columns if c.startswith(('vix_', 'sp500_'))]
    core_cols.append('target')
    combined = combined.dropna(subset=core_cols)
    combined = combined.fillna(0)
    combined.index = pd.to_datetime(combined.index)
    combined = percentile_clip(combined)
    X = combined.drop('target', axis=1)
    y = combined['target']
    return X, y


def generate_walk_forward_folds(n_samples, min_train_years=10, step_years=3, embargo=EMBARGO):
    """Generate expanding-window WF folds for extended data (~36 years)."""
    trading_days_per_year = 252
    min_train = min_train_years * trading_days_per_year
    step = step_years * trading_days_per_year
    test_size = step

    folds = []
    fold_num = 1
    train_end = min_train
    while train_end + embargo + test_size <= n_samples:
        test_start = train_end + embargo
        test_end = min(test_start + test_size, n_samples)
        folds.append({
            'fold': fold_num,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
        })
        fold_num += 1
        train_end += step

    return folds


def train_lr_unbalanced(X_train, y_train, X_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = LogisticRegression(C=0.1, max_iter=1000)
    model.fit(Xtr, y_train)
    return model, scaler, model.predict_proba(Xte)[:, 1]


def train_gbdt(X_train, y_train, X_test):
    pos = max((y_train == 1).sum(), 1)
    neg = (y_train == 0).sum()
    sample_weight = np.where(y_train == 1, neg / pos, 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.03,
        min_samples_leaf=30, l2_regularization=2.0, random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model, None, model.predict_proba(X_test)[:, 1]


def train_gbdt_unbalanced(X_train, y_train, X_test):
    """GBDT without sample reweighting — better calibrated probs."""
    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.03,
        min_samples_leaf=30, l2_regularization=2.0, random_state=42,
    )
    model.fit(X_train, y_train)
    return model, None, model.predict_proba(X_test)[:, 1]


def run_walk_forward(X, y, model_name, train_fn, folds):
    """Run WF and collect per-fold metrics."""
    results = []
    for fold in folds:
        X_train = X.iloc[:fold['train_end']]
        X_test = X.iloc[fold['test_start']:fold['test_end']]
        y_train = y.iloc[:fold['train_end']]
        y_test = y.iloc[fold['test_start']:fold['test_end']]

        if y_train.sum() < 5 or y_test.sum() < 3:
            continue

        model, scaler, probs_test = train_fn(X_train, y_train, X_test)
        pm = compute_practical_metrics(y_test, probs_test)
        roc_auc = auc(*roc_curve(y_test, probs_test)[:2])

        results.append({
            'fold': fold['fold'],
            'train_end': str(X_train.index[-1].date()),
            'test_start': str(X_test.index[0].date()),
            'test_end': str(X_test.index[-1].date()),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_pos_rate': round(float(y_test.mean()), 4),
            'auc': round(roc_auc, 4),
            'best_f1': pm['best_f1'],
            'brier': pm['brier_score'],
            'best_f1_threshold': pm['best_f1_threshold'],
        })

    summary = {
        'model_name': model_name,
        'n_folds': len(results),
        'f1_mean': round(np.mean([r['best_f1'] for r in results]), 4) if results else 0,
        'f1_std': round(np.std([r['best_f1'] for r in results]), 4) if results else 0,
        'auc_mean': round(np.mean([r['auc'] for r in results]), 4) if results else 0,
        'auc_std': round(np.std([r['auc'] for r in results]), 4) if results else 0,
        'brier_mean': round(np.mean([r['brier'] for r in results]), 4) if results else 0,
        'folds': results,
    }
    return summary


def run_single_split(X, y, model_name, train_fn):
    """Run single 70/30 split with embargo for comparison."""
    split = int(len(X) * 0.7)
    train_end = max(split - EMBARGO, 1)
    test_start = min(split + EMBARGO, len(X))
    X_train, X_test = X.iloc[:train_end], X.iloc[test_start:]
    y_train, y_test = y.iloc[:train_end], y.iloc[test_start:]

    if y_train.sum() < 5 or y_test.sum() < 3:
        return None

    model, scaler, probs_test = train_fn(X_train, y_train, X_test)
    pm = compute_practical_metrics(y_test, probs_test)
    roc_auc = auc(*roc_curve(y_test, probs_test)[:2])

    return {
        'model_name': model_name,
        'auc': round(roc_auc, 4),
        'best_f1': pm['best_f1'],
        'brier': pm['brier_score'],
        'train_period': f"{X_train.index[0].date()} ~ {X_train.index[-1].date()}",
        'test_period': f"{X_test.index[0].date()} ~ {X_test.index[-1].date()}",
        'practical_metrics': pm,
    }


def main():
    print("=" * 60)
    print("REGIME-CONDITIONAL & NON-LINEAR MODELS (1990+)")
    print("=" * 60)

    print("\n[1/5] Loading extended data...")
    df = load_extended_data()
    sp500 = df['sp500']
    target = compute_target(sp500)
    print(f"  {len(df)} trading days")

    print("\n[2/5] Fetching regime data (Fed Funds + CPI)...")
    regime_df = fetch_regime_data(start='1990-01-01')

    print("\n[3/5] Building feature sets...")
    features_slim = build_features_slim(df)
    features_regime, regime_labels = build_regime_features_extended(df, regime_df)

    X_slim, y_slim = prep_data(features_slim, target)
    X_regime, y_regime = prep_data(features_regime, target)

    tight_pct = (regime_labels == 'tight').mean() * 100
    print(f"  Slim: {X_slim.shape[1]} features, {len(X_slim)} samples")
    print(f"  Regime: {X_regime.shape[1]} features, {len(X_regime)} samples")
    print(f"  Tight regime days: {tight_pct:.1f}% ({(regime_labels == 'tight').sum()} days)")
    print(f"  Positive rate: {y_slim.mean()*100:.1f}%")

    print("\n[4/5] Walk-Forward evaluation (10yr min train, 3yr step)...")
    folds = generate_walk_forward_folds(len(X_slim))
    print(f"  {len(folds)} folds generated")

    models = {
        'LR Slim': (X_slim, y_slim, train_lr_unbalanced),
        'LR Regime+Interact': (X_regime, y_regime, train_lr_unbalanced),
        'GBDT Slim': (X_slim, y_slim, train_gbdt),
        'GBDT Regime+Interact': (X_regime, y_regime, train_gbdt),
        'GBDT Slim (unbal)': (X_slim, y_slim, train_gbdt_unbalanced),
        'GBDT Regime (unbal)': (X_regime, y_regime, train_gbdt_unbalanced),
    }

    wf_results = {}
    ss_results = {}
    for model_name, (X, y, fn) in models.items():
        print(f"\n  --- {model_name} ---")
        wf = run_walk_forward(X, y, model_name, fn, folds)
        wf_results[model_name] = wf
        print(f"    WF: AUC={wf['auc_mean']:.4f}±{wf['auc_std']:.4f}  "
              f"F1={wf['f1_mean']:.4f}±{wf['f1_std']:.4f}  "
              f"Brier={wf['brier_mean']:.4f}")

        ss = run_single_split(X, y, model_name, fn)
        ss_results[model_name] = ss
        if ss:
            print(f"    SS: AUC={ss['auc']}  F1={ss['best_f1']}  Brier={ss['brier']}")

    print("\n[5/5] Saving results...")

    output = {
        'title': 'Regime-Conditional & Non-Linear Models (1990+)',
        'data_range': f"{X_slim.index[0].date()} ~ {X_slim.index[-1].date()}",
        'n_samples': len(X_slim),
        'n_positive': int(y_slim.sum()),
        'positive_rate': round(float(y_slim.mean()), 4),
        'tight_regime_pct': round(tight_pct, 1),
        'walk_forward_config': {
            'min_train_years': 10,
            'step_years': 3,
            'embargo_days': EMBARGO,
            'n_folds': len(folds),
        },
        'walk_forward_summary': {
            name: {
                'auc_mean': r['auc_mean'],
                'auc_std': r['auc_std'],
                'f1_mean': r['f1_mean'],
                'f1_std': r['f1_std'],
                'brier_mean': r['brier_mean'],
            }
            for name, r in wf_results.items()
        },
        'single_split_summary': {
            name: {
                'auc': r['auc'],
                'best_f1': r['best_f1'],
                'brier': r['brier'],
                'train_period': r['train_period'],
                'test_period': r['test_period'],
            }
            for name, r in ss_results.items() if r
        },
        'walk_forward_detail': {name: r['folds'] for name, r in wf_results.items()},
        'verdict': [],
    }

    best_wf_f1 = max(wf_results.items(), key=lambda x: x[1]['f1_mean'])
    best_wf_auc = max(wf_results.items(), key=lambda x: x[1]['auc_mean'])
    best_wf_brier = min(wf_results.items(), key=lambda x: x[1]['brier_mean'])

    output['verdict'].append(f"WF Best F1: {best_wf_f1[0]} ({best_wf_f1[1]['f1_mean']:.4f})")
    output['verdict'].append(f"WF Best AUC: {best_wf_auc[0]} ({best_wf_auc[1]['auc_mean']:.4f})")
    output['verdict'].append(f"WF Best Brier: {best_wf_brier[0]} ({best_wf_brier[1]['brier_mean']:.4f})")

    lr_slim_f1 = wf_results['LR Slim']['f1_mean']
    gbdt_slim_f1 = wf_results['GBDT Slim']['f1_mean']
    gbdt_regime_f1 = wf_results['GBDT Regime+Interact']['f1_mean']
    if gbdt_slim_f1 > lr_slim_f1 * 1.1:
        output['verdict'].append(f"GBDT shows {(gbdt_slim_f1/lr_slim_f1-1)*100:.0f}% F1 improvement over LR — nonlinearity helps")
    else:
        output['verdict'].append("GBDT vs LR: marginal difference — regime interactions may matter more than nonlinearity alone")
    if gbdt_regime_f1 > gbdt_slim_f1 * 1.05:
        output['verdict'].append("Regime+Interact features boost GBDT — confirms regime-dependent crash relationships")
    else:
        output['verdict'].append("Regime features don't clearly help GBDT — tree already captures conditional splits")

    out_path = DATA_DIR / 'regime_model_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved {out_path.name}")

    phase3_path = DATA_DIR / 'phase3_metrics.json'
    if phase3_path.exists():
        with open(phase3_path) as f:
            phase3 = json.load(f)
        phase3['regime_models'] = output
        with open(phase3_path, 'w') as f:
            json.dump(phase3, f, indent=2, default=str)
        print(f"  Merged into phase3_metrics.json")

    print("\n" + "=" * 60)
    print("RESULTS COMPARISON (Walk-Forward)")
    print("=" * 60)
    print(f"  {'Model':<28} {'AUC':>12} {'F1':>12} {'Brier':>10}")
    print(f"  {'-'*66}")
    for name, r in sorted(wf_results.items(), key=lambda x: -x[1]['f1_mean']):
        print(f"  {name:<28} {r['auc_mean']:.4f}±{r['auc_std']:.4f}  "
              f"{r['f1_mean']:.4f}±{r['f1_std']:.4f}  {r['brier_mean']:.4f}")
    print(f"\n  Verdict: {' | '.join(output['verdict'][:2])}")
    print("=" * 60)


if __name__ == '__main__':
    main()
