"""
Walk-forward validation (expanding window) for recommended production models.

Models:
  - LR Slim+Events      (Best F1)
  - LR Events+Interact  (Best Brier)

Design:
  - Expanding train window from data start
  - Test window: ~6 months (~126 trading days)
  - Step: ~6 months
  - Min first train: ~1.5 years (~378 trading days)
  - Embargo: 20 days between train end and test start
  - Percentile clip (1st-99th) fit on TRAIN only per fold
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, brier_score_loss, roc_curve

warnings.filterwarnings('ignore')

from predict_model import (
    load_indicators,
    build_features_with_events,
    compute_target,
    fetch_regime_data,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from experiment_phase3 import (
    EMBARGO,
    detect_regime,
    train_lr_no_balance,
    compute_practical_metrics,
)

DATA_DIR = Path(__file__).parent / 'data'

MIN_TRAIN_DAYS = 378   # ~1.5 years
TEST_DAYS = 126        # ~6 months
STEP_DAYS = 126        # ~6 months
DECAY_HALF_LIFE = 252  # ~1 year half-life for sample weights


def train_lr_decay(X_train, y_train, X_test, half_life=DECAY_HALF_LIFE):
    """LR with exponential time decay — recent samples weighted higher."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    ages = np.arange(len(X_train))[::-1].astype(float)
    weights = np.exp(-ages / half_life)
    model = LogisticRegression(C=0.1, max_iter=1000)
    model.fit(X_train_s, y_train, sample_weight=weights)
    probs = model.predict_proba(X_test_s)[:, 1]
    return model, scaler, probs


def build_events_interact_features(df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    features = build_features_with_events(df)
    regime_labels = detect_regime(df, regime_df)
    regime_aligned = regime_labels.reindex(features.index).fillna('normal')
    regime_binary = (regime_aligned == 'tight').astype(float)
    for f in ['vix_level', 'credit_spread_10d_chg', 'sp500_vs_50ma']:
        if f in features.columns:
            features[f'tight_x_{f}'] = regime_binary * features[f]
    return features


def prep_xy(features: pd.DataFrame, target: pd.Series):
    combined = features.copy()
    combined['target'] = target
    combined = combined.dropna()
    combined.index = pd.to_datetime(combined.index)
    X = combined.drop('target', axis=1)
    y = combined['target']
    return X, y


def clip_fit_transform(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col='target'):
    train = train_df.copy()
    test = test_df.copy()
    feat_cols = [c for c in train.columns if c != target_col]
    for c in feat_cols:
        lo, hi = train[c].quantile(0.01), train[c].quantile(0.99)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            continue
        train[c] = train[c].clip(lo, hi)
        test[c] = test[c].clip(lo, hi)
    return train, test


def generate_folds(n: int, index: pd.DatetimeIndex):
    folds = []
    test_start = MIN_TRAIN_DAYS + EMBARGO
    fold_id = 1
    while test_start < n:
        test_end = min(test_start + TEST_DAYS, n)
        if test_end <= test_start:
            break
        train_end = test_start - EMBARGO
        if train_end < MIN_TRAIN_DAYS:
            test_start += STEP_DAYS
            continue
        folds.append({
            'fold': fold_id,
            'train_start': str(index[0].date()),
            'train_end': str(index[train_end - 1].date()),
            'embargo_start': str(index[train_end].date()) if train_end < n else None,
            'embargo_end': str(index[test_start - 1].date()) if test_start > train_end else None,
            'test_start': str(index[test_start].date()),
            'test_end': str(index[test_end - 1].date()),
            'train_n': train_end,
            'test_n': test_end - test_start,
            'train_pos': None,
            'test_pos': None,
        })
        fold_id += 1
        test_start += STEP_DAYS
        if test_end >= n:
            break
    return folds


def run_fold(model_name: str, X: pd.DataFrame, y: pd.Series, fold: dict, train_fn=train_lr_no_balance):
    te = fold['train_n']
    ts = te + EMBARGO
    tt = ts + fold['test_n']

    train_df = X.iloc[:te].copy()
    train_df['target'] = y.iloc[:te]
    test_df = X.iloc[ts:tt].copy()
    test_df['target'] = y.iloc[ts:tt]

    train_df, test_df = clip_fit_transform(train_df, test_df)
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    fold['train_pos'] = int(y_train.sum())
    fold['test_pos'] = int(y_test.sum())

    if y_train.nunique() < 2 or y_test.nunique() < 2 or len(y_test) < 20:
        return None

    _, _, probs_test = train_fn(X_train, y_train, X_test)
    probs_test = np.asarray(probs_test)
    practical = compute_practical_metrics(y_test, probs_test)
    try:
        test_auc = float(auc(*roc_curve(y_test, probs_test)[:2]))
    except ValueError:
        test_auc = 0.5

    return {
        'model': model_name,
        'fold': fold['fold'],
        'train_period': f"{fold['train_start']} ~ {fold['train_end']}",
        'test_period': f"{fold['test_start']} ~ {fold['test_end']}",
        'train_n': fold['train_n'],
        'test_n': fold['test_n'],
        'train_pos': fold['train_pos'],
        'test_pos': fold['test_pos'],
        'auc': round(test_auc, 3),
        'practical_metrics': practical,
    }


def summarize_model(results: list):
    f1s = [r['practical_metrics']['best_f1'] for r in results]
    briers = [r['practical_metrics']['brier_score'] for r in results]
    aucs = [r['auc'] for r in results]
    return {
        'n_folds': len(results),
        'f1_mean': round(float(np.mean(f1s)), 3),
        'f1_std': round(float(np.std(f1s)), 3),
        'f1_min': round(float(np.min(f1s)), 3),
        'f1_max': round(float(np.max(f1s)), 3),
        'brier_mean': round(float(np.mean(briers)), 4),
        'brier_std': round(float(np.std(briers)), 4),
        'brier_min': round(float(np.min(briers)), 4),
        'brier_max': round(float(np.max(briers)), 4),
        'auc_mean': round(float(np.mean(aucs)), 3),
        'auc_std': round(float(np.std(aucs)), 3),
    }


def load_single_split_baselines():
    p = DATA_DIR / 'phase3_metrics.json'
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    out = {}
    for name in ['LR Slim+Events', 'LR Events+Interact']:
        exp = next((e for e in data.get('experiments', []) if e.get('name') == name), None)
        if exp and exp.get('practical_metrics'):
            pm = exp['practical_metrics']
            out[name] = {
                'method': 'single 70/30 split + embargo 20d',
                'train_end': '2025-04-22',
                'test_start': '2025-06-20',
                'best_f1': pm['best_f1'],
                'brier_score': pm['brier_score'],
                'auc': exp.get('auc'),
            }
    return out


def run_walkforward(models: dict, folds: list, train_fn=train_lr_no_balance, label_suffix=''):
    all_results = []
    model_summaries = {}
    for model_name, (X, y) in models.items():
        full_name = f"{model_name}{label_suffix}"
        print(f"\n--- {full_name} ---")
        results = []
        for fold in folds:
            row = run_fold(full_name, X, y, fold, train_fn=train_fn)
            if row is None:
                print(f"  Fold {fold['fold']}: skipped (insufficient classes/samples)")
                continue
            results.append(row)
            pm = row['practical_metrics']
            print(f"  Fold {row['fold']}: F1={pm['best_f1']:.3f} Brier={pm['brier_score']:.4f} "
                  f"AUC={row['auc']:.3f} test_pos={row['test_pos']}/{row['test_n']}")
        all_results.extend(results)
        if results:
            model_summaries[full_name] = summarize_model(results)
    return all_results, model_summaries


def main():
    print('=' * 60)
    print('WALK-FORWARD VALIDATION (Expanding Window)')
    print('=' * 60)

    df = load_indicators()
    df.index = pd.to_datetime(df.index)
    target = compute_target(df['sp500'])
    regime_df = fetch_regime_data()

    X_events, y_events = prep_xy(build_features_with_events(df), target)
    X_combo, y_combo = prep_xy(build_events_interact_features(df, regime_df), target)

    print(f"\nData: {X_events.index[0].date()} ~ {X_events.index[-1].date()} ({len(X_events)} samples)")
    print(f"  LR Slim+Events: {X_events.shape[1]} features")
    print(f"  LR Events+Interact: {X_combo.shape[1]} features")

    folds = generate_folds(len(X_events), X_events.index)
    print(f"\nFolds: {len(folds)} (min train {MIN_TRAIN_DAYS}d, test {TEST_DAYS}d, step {STEP_DAYS}d, embargo {EMBARGO}d)")
    for f in folds:
        print(f"  Fold {f['fold']}: train {f['train_start']}~{f['train_end']} ({f['train_n']}d) | "
              f"test {f['test_start']}~{f['test_end']} ({f['test_n']}d)")

    models = {
        'LR Slim+Events': (X_events, y_events),
        'LR Events+Interact': (X_combo, y_combo),
    }

    all_results, model_summaries = run_walkforward(models, folds)

    print(f"\n=== TIME DECAY (half-life={DECAY_HALF_LIFE}d) ===")
    decay_results, decay_summaries = run_walkforward(
        models, folds, train_fn=train_lr_decay, label_suffix=' +Decay',
    )

    baselines = load_single_split_baselines()

    # Verdict heuristics
    verdict_lines = []
    for name, summary in model_summaries.items():
        base = baselines.get(name.replace(' +Decay', ''), {})
        verdict_lines.append(
            f"{name}: WF mean F1={summary['f1_mean']}±{summary['f1_std']} "
            f"(single-split {base.get('best_f1', '—')}); "
            f"Brier={summary['brier_mean']}±{summary['brier_std']}"
        )
    for name, summary in decay_summaries.items():
        base_name = name.replace(' +Decay', '')
        base = model_summaries.get(base_name, {})
        delta = summary['f1_mean'] - base.get('f1_mean', 0) if base else 0
        verdict_lines.append(
            f"{name}: WF F1={summary['f1_mean']}±{summary['f1_std']} "
            f"(vs no-decay Δ={delta:+.3f})"
        )

    output = {
        'title': 'Walk-Forward Validation — Expanding Window',
        'design': {
            'train_window': 'expanding from data start',
            'min_train_days': MIN_TRAIN_DAYS,
            'test_days': TEST_DAYS,
            'step_days': STEP_DAYS,
            'embargo_days': EMBARGO,
            'clip': 'per-fold 1st-99th percentile on train only',
            'models': list(models.keys()),
        },
        'data_range': f"{X_events.index[0].date()} ~ {X_events.index[-1].date()}",
        'total_samples': len(X_events),
        'folds': folds,
        'results': all_results,
        'summary_by_model': model_summaries,
        'single_split_baseline': baselines,
        'verdict': verdict_lines,
        'decay': {
            'half_life_days': DECAY_HALF_LIFE,
            'results': decay_results,
            'summary_by_model': decay_summaries,
        },
    }

    out_path = DATA_DIR / 'walkforward_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")

    # Merge into phase3_metrics.json for dashboard
    p3 = DATA_DIR / 'phase3_metrics.json'
    if p3.exists():
        phase3 = json.loads(p3.read_text())
        phase3['walk_forward'] = output
        with open(p3, 'w') as f:
            json.dump(phase3, f)
        print(f"Merged walk_forward into {p3}")

    print('\n=== SUMMARY ===')
    for line in verdict_lines:
        print(f'  {line}')


if __name__ == '__main__':
    main()
