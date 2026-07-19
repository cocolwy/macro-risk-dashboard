"""
Target sensitivity grid + walk-forward validation.

Grid: horizon (10/20/40d) × drawdown threshold (3%/5%/7%)
Model: LR Slim+Events (fixed feature set)
Eval: expanding-window walk-forward (same as experiment_walkforward.py)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from predict_model import load_indicators, build_features_with_events, compute_target
from experiment_walkforward import (
    prep_xy,
    generate_folds,
    run_walkforward,
    MIN_TRAIN_DAYS,
    TEST_DAYS,
    STEP_DAYS,
    EMBARGO,
)

DATA_DIR = Path(__file__).parent / 'data'

HORIZONS = [10, 20, 40]
THRESHOLDS = [-0.03, -0.05, -0.07]


def main():
    print('=' * 60)
    print('TARGET SENSITIVITY GRID + WALK-FORWARD')
    print('=' * 60)

    df = load_indicators()
    df.index = pd.to_datetime(df.index)
    sp500 = df['sp500']

    grid_rows = []
    all_wf_results = []

    for horizon in HORIZONS:
        for thresh in THRESHOLDS:
            label = f'h{horizon}_dd{abs(thresh * 100):.0f}pct'
            print(f"\n--- {label} (horizon={horizon}d, threshold={thresh:.0%}) ---")
            target = compute_target(sp500, horizon=horizon, threshold=thresh)
            X, y = prep_xy(build_features_with_events(df), target)
            pos_rate = float(y.mean())
            print(f"  Samples: {len(X)}, positive: {int(y.sum())} ({pos_rate*100:.1f}%)")

            if len(X) < MIN_TRAIN_DAYS + TEST_DAYS + EMBARGO or y.sum() < 10:
                print('  Skipped: insufficient samples or positives')
                continue

            folds = generate_folds(len(X), X.index)
            models = {'LR Slim+Events': (X, y)}
            wf_results, wf_summary = run_walkforward(models, folds)
            summary = wf_summary.get('LR Slim+Events', {})

            row = {
                'id': label,
                'horizon_days': horizon,
                'drawdown_threshold': thresh,
                'drawdown_pct': abs(thresh * 100),
                'n_samples': len(X),
                'positive_rate': round(pos_rate, 4),
                'n_positives': int(y.sum()),
                'wf_f1_mean': summary.get('f1_mean'),
                'wf_f1_std': summary.get('f1_std'),
                'wf_brier_mean': summary.get('brier_mean'),
                'wf_auc_mean': summary.get('auc_mean'),
                'n_folds': summary.get('n_folds', 0),
                'is_default': horizon == 20 and abs(thresh + 0.05) < 1e-6,
            }
            grid_rows.append(row)
            all_wf_results.extend(wf_results)

            if summary:
                print(f"  WF F1={summary['f1_mean']:.3f}±{summary['f1_std']:.3f} "
                      f"Brier={summary['brier_mean']:.4f}")

    grid_rows.sort(key=lambda r: (r['wf_f1_mean'] or 0), reverse=True)
    best = grid_rows[0] if grid_rows else None
    default = next((r for r in grid_rows if r['is_default']), None)

    output = {
        'title': 'Target Sensitivity — Horizon × Drawdown Threshold',
        'design': {
            'horizons_days': HORIZONS,
            'thresholds': THRESHOLDS,
            'model': 'LR Slim+Events',
            'walk_forward': 'expanding window, same as Ch.2 WF',
            'min_train_days': MIN_TRAIN_DAYS,
            'test_days': TEST_DAYS,
            'step_days': STEP_DAYS,
            'embargo_days': EMBARGO,
        },
        'grid': grid_rows,
        'default_config': default,
        'best_wf_config': best,
        'verdict': [],
    }

    if default and best:
        output['verdict'].append(
            f"默认 target (20d/5%): WF F1={default.get('wf_f1_mean')} — "
            f"pos rate {default.get('positive_rate', 0)*100:.1f}%"
        )
        if best['id'] != default['id']:
            output['verdict'].append(
                f"WF 最佳: {best['id']} F1={best['wf_f1_mean']}±{best['wf_f1_std']} "
                f"(pos rate {best['positive_rate']*100:.1f}%)"
            )
        else:
            output['verdict'].append('默认 20d/5% 在 WF 下仍为最佳配置。')

    out_path = DATA_DIR / 'target_grid_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")

    p3 = DATA_DIR / 'phase3_metrics.json'
    if p3.exists():
        phase3 = json.loads(p3.read_text())
        phase3['target_sensitivity'] = output
        with open(p3, 'w') as f:
            json.dump(phase3, f)
        print(f"Merged target_sensitivity into {p3}")

    print('\n=== TOP 3 BY WF F1 ===')
    for r in grid_rows[:3]:
        print(f"  {r['id']}: F1={r['wf_f1_mean']} pos={r['positive_rate']*100:.1f}%")


if __name__ == '__main__':
    main()
