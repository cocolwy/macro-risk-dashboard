"""
Autoresearch evaluation harness — DO NOT MODIFY.

This file provides the fixed evaluation pipeline that experiment_auto.py
runs against. It is analogous to Karpathy's prepare.py: the ground truth
metric lives here and must not be changed by the research agent.

Pluggable design:
  The harness loads a ResearchConfig that defines data loading, target
  construction, walk-forward params, and scoring weights. Swap the config
  to run different research tracks (macro crash, micro-structure, etc.)
  without changing this file.

Usage:
  python -m autoresearch.run            # single experiment
  python -m autoresearch.evaluate_harness --dry-run  # validate setup
"""

import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import auc, brier_score_loss, roc_curve

warnings.filterwarnings('ignore')

AUTORESEARCH_DIR = Path(__file__).parent
RESULTS_TSV = AUTORESEARCH_DIR / 'results.tsv'

# ---------------------------------------------------------------------------
# Research config: swap this to run different research tracks
# ---------------------------------------------------------------------------

@dataclass
class ResearchConfig:
    """Defines a research track (data, target, evaluation, scoring)."""
    name: str
    load_data: Callable[[], pd.DataFrame]
    build_target: Callable[[pd.DataFrame], pd.Series]
    # walk-forward params
    min_train_days: int = 378
    test_days: int = 126
    step_days: int = 126
    embargo_days: int = 20
    # composite score weights (higher = better)
    f1_weight: float = 0.6
    brier_weight: float = 0.4
    # guardrails
    max_features: int = 30
    f1_ceiling: float = 0.80       # F1 above this is suspicious overfitting
    max_corr_between_features: float = 0.85
    forbid_balanced_training: bool = True
    # optional: extra data (e.g. regime data)
    load_extra: Optional[Callable] = None


def get_macro_crash_config() -> ResearchConfig:
    """Config for macro crash prediction (current primary research track)."""
    sys.path.insert(0, str(Path(__file__).parent.parent / 'dashboard'))
    from predict_model import load_indicators, compute_target

    def _load():
        df = load_indicators()
        df.index = pd.to_datetime(df.index)
        return df

    def _target(df):
        return compute_target(df['sp500'], horizon=20, threshold=-0.05)

    def _extra():
        from predict_model import fetch_regime_data
        return fetch_regime_data()

    return ResearchConfig(
        name='macro_crash',
        load_data=_load,
        build_target=_target,
        load_extra=_extra,
        min_train_days=378,
        test_days=126,
        step_days=126,
        embargo_days=20,
        f1_weight=0.6,
        brier_weight=0.4,
        max_features=30,
        f1_ceiling=0.80,
        forbid_balanced_training=True,
    )


# ---------------------------------------------------------------------------
# Walk-forward evaluation (fixed, do not modify)
# ---------------------------------------------------------------------------

def generate_folds(n: int, index: pd.DatetimeIndex, config: ResearchConfig):
    """Generate expanding-window walk-forward folds."""
    folds = []
    test_start = config.min_train_days + config.embargo_days
    fold_id = 1
    while test_start < n:
        test_end = min(test_start + config.test_days, n)
        if test_end <= test_start:
            break
        train_end = test_start - config.embargo_days
        if train_end < config.min_train_days:
            test_start += config.step_days
            continue
        folds.append({
            'fold': fold_id,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'train_start_date': str(index[0].date()),
            'train_end_date': str(index[train_end - 1].date()),
            'test_start_date': str(index[test_start].date()),
            'test_end_date': str(index[test_end - 1].date()),
        })
        fold_id += 1
        test_start += config.step_days
        if test_end >= n:
            break
    return folds


def clip_per_fold(X_train: pd.DataFrame, X_test: pd.DataFrame,
                  lo_q=0.01, hi_q=0.99):
    """Percentile clip fit on train, applied to both. Returns copies."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    for c in X_train.columns:
        lo, hi = X_train[c].quantile(lo_q), X_train[c].quantile(hi_q)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            continue
        X_train[c] = X_train[c].clip(lo, hi)
        X_test[c] = X_test[c].clip(lo, hi)
    return X_train, X_test


def compute_practical_metrics(y_test: pd.Series, probs_test: np.ndarray) -> dict:
    """Compute Best F1, Brier score, and other practical metrics.

    This is the ground truth evaluation — identical to experiment_phase3.py.
    """
    brier = brier_score_loss(y_test, probs_test)
    base_rate = float(y_test.mean())

    best_f1, best_f1_thresh = 0.0, 0.5
    for thresh_pct in range(10, 95, 5):
        thresh = thresh_pct / 100
        preds = (probs_test > thresh).astype(int)
        tp = ((preds == 1) & (y_test.values == 1)).sum()
        fp = ((preds == 1) & (y_test.values == 0)).sum()
        fn = ((preds == 0) & (y_test.values == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_f1_thresh = f1, thresh

    return {
        'brier_score': round(brier, 4),
        'base_rate': round(base_rate, 4),
        'best_f1': round(best_f1, 3),
        'best_f1_threshold': best_f1_thresh,
        'mean_prob': round(float(probs_test.mean()), 4),
    }


def compute_composite_score(best_f1: float, brier: float,
                            config: ResearchConfig) -> float:
    """Single scalar score: higher = better.

    composite = f1_weight * best_f1 + brier_weight * (1 - brier)
    """
    return round(
        config.f1_weight * best_f1
        + config.brier_weight * (1.0 - brier),
        6,
    )


# ---------------------------------------------------------------------------
# Guardrails (fixed, do not modify)
# ---------------------------------------------------------------------------

def check_guardrails(X: pd.DataFrame, config: ResearchConfig) -> list[str]:
    """Return list of warning strings. Empty = all clear."""
    warnings_list = []

    if X.shape[1] > config.max_features:
        warnings_list.append(
            f'FEATURE_COUNT: {X.shape[1]} > max {config.max_features}')

    corr = X.corr(method='spearman').abs()
    np.fill_diagonal(corr.values, 0)
    high = corr.max().max()
    if high > config.max_corr_between_features:
        pair = corr.stack().idxmax()
        warnings_list.append(
            f'HIGH_CORR: {pair[0]} × {pair[1]} = {high:.3f} '
            f'> {config.max_corr_between_features}')

    return warnings_list


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(build_features_fn, train_model_fn,
             config: ResearchConfig,
             extra_data=None) -> dict:
    """Run full walk-forward evaluation. Returns result dict.

    Parameters
    ----------
    build_features_fn : callable(df, extra_data=None) -> pd.DataFrame
        Feature engineering function (from experiment_auto.py).
    train_model_fn : callable(X_train, y_train, X_test) -> (model, scaler, probs_test)
        Model training function (from experiment_auto.py).
    config : ResearchConfig
        Research track configuration.
    extra_data : any, optional
        Extra data (e.g. regime DataFrame) passed to build_features_fn.
    """
    t0 = time.time()

    # Load data
    df = config.load_data()
    target = config.build_target(df)

    # Build features
    features = build_features_fn(df, extra_data=extra_data)
    combined = features.copy()
    combined['target'] = target
    combined = combined.dropna()
    combined.index = pd.to_datetime(combined.index)
    X = combined.drop('target', axis=1)
    y = combined['target']

    n_features = X.shape[1]
    n_samples = len(X)
    n_positive = int(y.sum())
    positive_rate = round(float(y.mean()), 4)

    # Guardrails
    guardrail_warnings = check_guardrails(X, config)

    # Walk-forward folds
    folds = generate_folds(len(X), X.index, config)
    if not folds:
        return {
            'status': 'error',
            'error': 'No valid walk-forward folds generated',
            'n_samples': n_samples,
            'n_features': n_features,
        }

    fold_results = []
    all_f1s, all_briers = [], []

    for fold in folds:
        te = fold['train_end']
        ts = fold['test_start']
        tt = fold['test_end']

        X_train_raw = X.iloc[:te]
        X_test_raw = X.iloc[ts:tt]
        y_train = y.iloc[:te]
        y_test = y.iloc[ts:tt]

        if y_train.nunique() < 2 or y_test.nunique() < 2 or len(y_test) < 20:
            fold_results.append({
                'fold': fold['fold'],
                'status': 'skipped',
                'reason': 'insufficient classes or samples',
            })
            continue

        X_train, X_test = clip_per_fold(X_train_raw, X_test_raw)

        try:
            _, _, probs_test = train_model_fn(X_train, y_train, X_test)
            probs_test = np.asarray(probs_test, dtype=float)
        except Exception as e:
            fold_results.append({
                'fold': fold['fold'],
                'status': 'crash',
                'error': str(e),
            })
            continue

        if np.any(np.isnan(probs_test)) or np.any(np.isinf(probs_test)):
            fold_results.append({
                'fold': fold['fold'],
                'status': 'crash',
                'error': 'NaN/Inf in predictions',
            })
            continue

        pm = compute_practical_metrics(y_test, probs_test)
        try:
            fold_auc = float(auc(*roc_curve(y_test, probs_test)[:2]))
        except ValueError:
            fold_auc = 0.5

        fold_results.append({
            'fold': fold['fold'],
            'status': 'ok',
            'train_period': f"{fold['train_start_date']} ~ {fold['train_end_date']}",
            'test_period': f"{fold['test_start_date']} ~ {fold['test_end_date']}",
            'train_n': te,
            'test_n': tt - ts,
            'test_pos': int(y_test.sum()),
            'auc': round(fold_auc, 3),
            **pm,
        })
        all_f1s.append(pm['best_f1'])
        all_briers.append(pm['brier_score'])

    elapsed = time.time() - t0
    ok_folds = [f for f in fold_results if f.get('status') == 'ok']

    if not ok_folds:
        return {
            'status': 'error',
            'error': 'All folds crashed or skipped',
            'fold_results': fold_results,
            'elapsed_seconds': round(elapsed, 1),
        }

    # Aggregate metrics
    mean_f1 = float(np.mean(all_f1s))
    std_f1 = float(np.std(all_f1s))
    mean_brier = float(np.mean(all_briers))
    std_brier = float(np.std(all_briers))
    composite = compute_composite_score(mean_f1, mean_brier, config)

    # Overfitting check
    overfit_flag = mean_f1 > config.f1_ceiling

    result = {
        'status': 'ok',
        'composite_score': composite,
        'mean_f1': round(mean_f1, 3),
        'std_f1': round(std_f1, 3),
        'mean_brier': round(mean_brier, 4),
        'std_brier': round(std_brier, 4),
        'n_folds': len(ok_folds),
        'n_folds_total': len(folds),
        'n_features': n_features,
        'n_samples': n_samples,
        'n_positive': n_positive,
        'positive_rate': positive_rate,
        'feature_names': list(X.columns),
        'overfit_flag': overfit_flag,
        'guardrail_warnings': guardrail_warnings,
        'fold_results': fold_results,
        'elapsed_seconds': round(elapsed, 1),
    }

    return result


# ---------------------------------------------------------------------------
# Results TSV management
# ---------------------------------------------------------------------------

TSV_HEADER = 'commit\tcomposite\tmean_f1\tmean_brier\tn_features\tstatus\tdescription\n'


def init_results_tsv():
    """Create results.tsv with header if it doesn't exist."""
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(TSV_HEADER)


def append_result(commit: str, result: dict, description: str):
    """Append one row to results.tsv."""
    init_results_tsv()
    status = result.get('status', 'error')
    if status == 'ok':
        row = (
            f"{commit}\t"
            f"{result['composite_score']:.6f}\t"
            f"{result['mean_f1']:.3f}\t"
            f"{result['mean_brier']:.4f}\t"
            f"{result['n_features']}\t"
            f"{'keep' if not result.get('overfit_flag') else 'suspect'}\t"
            f"{description}\n"
        )
    else:
        error = result.get('error', 'unknown error')
        row = f"{commit}\t0.000000\t0.000\t1.0000\t0\tcrash\t{description} ({error})\n"
    with open(RESULTS_TSV, 'a') as f:
        f.write(row)


def read_best_composite() -> float:
    """Read the best composite score from results.tsv. Returns 0 if empty."""
    if not RESULTS_TSV.exists():
        return 0.0
    lines = RESULTS_TSV.read_text().strip().split('\n')[1:]  # skip header
    best = 0.0
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 6 and parts[5] in ('keep', 'suspect'):
            try:
                best = max(best, float(parts[1]))
            except ValueError:
                pass
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_summary(result: dict):
    """Pretty-print evaluation results (analogous to autoresearch's --- block)."""
    print('---')
    print(f"composite_score:  {result.get('composite_score', 0):.6f}")
    print(f"mean_f1:          {result.get('mean_f1', 0):.3f} ± {result.get('std_f1', 0):.3f}")
    print(f"mean_brier:       {result.get('mean_brier', 0):.4f} ± {result.get('std_brier', 0):.4f}")
    print(f"n_features:       {result.get('n_features', 0)}")
    print(f"n_samples:        {result.get('n_samples', 0)}")
    print(f"n_folds:          {result.get('n_folds', 0)}/{result.get('n_folds_total', 0)}")
    print(f"overfit_flag:     {result.get('overfit_flag', False)}")
    print(f"elapsed_seconds:  {result.get('elapsed_seconds', 0):.1f}")
    if result.get('guardrail_warnings'):
        for w in result['guardrail_warnings']:
            print(f"WARNING:          {w}")
    print('---')


if __name__ == '__main__':
    if '--dry-run' in sys.argv:
        config = get_macro_crash_config()
        df = config.load_data()
        target = config.build_target(df)
        print(f'Config: {config.name}')
        print(f'Data: {len(df)} rows, {df.index[0]} ~ {df.index[-1]}')
        print(f'Target: {int(target.sum())} positive / {len(target)} total ({target.mean()*100:.1f}%)')
        folds = generate_folds(len(df), df.index, config)
        print(f'Walk-forward folds: {len(folds)}')
        for f in folds:
            print(f"  Fold {f['fold']}: train ~{f['train_start_date']}~{f['train_end_date']} "
                  f"| test {f['test_start_date']}~{f['test_end_date']}")
        print('Setup OK.')
    else:
        print('Usage: python -m autoresearch.evaluate_harness --dry-run')
