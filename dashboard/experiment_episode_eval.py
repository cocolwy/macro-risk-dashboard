"""
Episode-based supplementary evaluation (Walk-Forward).

Primary model metrics remain Best F1 + Brier — this script does NOT replace them.

Design (user-approved):
  - Alert threshold: best_f1_threshold fit on TRAIN only (same sweep as F1)
  - Hit windows: Hit@10 / Hit@15 / Hit@20 (default read Hit@20)
  - False alarms: elevated days with target=0, consecutive days deduped
  - Episodes: contiguous target=1 clusters in test window; peak = worst forward DD day
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from predict_model import (
    load_indicators,
    build_features_with_events,
    compute_target,
    fetch_regime_data,
)
from experiment_phase3 import detect_regime, train_lr_no_balance, compute_practical_metrics
from experiment_walkforward import (
    prep_xy,
    generate_folds,
    clip_fit_transform,
    build_events_interact_features,
    EMBARGO,
    MIN_TRAIN_DAYS,
)

DATA_DIR = Path(__file__).parent / 'data'
HIT_WINDOWS = [10, 15, 20]
MODELS = ['LR Slim+Events', 'LR Events+Interact']


def forward_max_dd(sp500: pd.Series, date: pd.Timestamp, horizon: int = 20) -> float:
    idx = sp500.index.get_indexer([date], method='pad')[0]
    if idx < 0 or idx + horizon >= len(sp500):
        return np.nan
    current = sp500.iloc[idx]
    if not np.isfinite(current) or current <= 0:
        return np.nan
    future = sp500.iloc[idx + 1: idx + horizon + 1]
    return float(future.min() / current - 1)


def find_episodes(dates: pd.DatetimeIndex, y: pd.Series, sp500: pd.Series, horizon: int = 20):
    """Cluster consecutive target=1 days; peak = day with most negative forward DD."""
    episodes = []
    y_vals = y.values
    i = 0
    n = len(dates)
    while i < n:
        if y_vals[i] != 1:
            i += 1
            continue
        j = i
        while j < n and y_vals[j] == 1:
            j += 1
        cluster_dates = dates[i:j]
        peak_date = cluster_dates[0]
        worst_dd = np.inf
        for d in cluster_dates:
            dd = forward_max_dd(sp500, d, horizon)
            if np.isfinite(dd) and dd < worst_dd:
                worst_dd = dd
                peak_date = d
        episodes.append({
            'peak_date': str(peak_date.date()),
            'cluster_start': str(cluster_dates[0].date()),
            'cluster_end': str(cluster_dates[-1].date()),
            'n_positive_days': int(j - i),
            'max_forward_dd_pct': round(float(worst_dd * 100), 2) if np.isfinite(worst_dd) else None,
        })
        i = j
    return episodes


def train_threshold(X_train, y_train, X_test, y_test, train_fn=train_lr_no_balance):
    """Threshold from TRAIN best-F1 sweep; also return test probs for episode eval."""
    model, scaler, probs_test = train_fn(X_train, y_train, X_test)
    probs_train = model.predict_proba(scaler.transform(X_train))[:, 1]
    train_pm = compute_practical_metrics(y_train, probs_train)
    return float(train_pm['best_f1_threshold']), np.asarray(probs_test), train_pm


def episode_hits(episodes, dates: pd.DatetimeIndex, probs: np.ndarray, threshold: float, windows=HIT_WINDOWS):
    """For each episode peak, check if prob exceeded threshold in prior K trading days."""
    date_to_idx = {str(d.date()): i for i, d in enumerate(dates)}
    hits = {w: 0 for w in windows}
    lead_days = []
    details = []

    for ep in episodes:
        peak = ep['peak_date']
        peak_i = date_to_idx.get(peak)
        if peak_i is None:
            continue
        ep_hits = {}
        first_alert_i = None
        for w in windows:
            start = max(0, peak_i - w)
            window_probs = probs[start:peak_i]
            hit = bool(len(window_probs) > 0 and window_probs.max() >= threshold)
            ep_hits[w] = hit
            if hit:
                hits[w] += 1
        for k in range(peak_i - 1, max(-1, peak_i - max(windows) - 1), -1):
            if k >= 0 and probs[k] >= threshold:
                first_alert_i = k
                break
        lead = (peak_i - first_alert_i) if first_alert_i is not None else None
        if first_alert_i is not None:
            lead_days.append(lead)
        details.append({
            **ep,
            'hit_at_10': ep_hits.get(10, False),
            'hit_at_15': ep_hits.get(15, False),
            'hit_at_20': ep_hits.get(20, False),
            'lead_days': lead,
            'peak_prob': round(float(probs[peak_i]), 4),
        })

    n_ep = len(episodes)
    hit_rates = {f'hit_at_{w}': round(hits[w] / n_ep, 3) if n_ep else None for w in windows}
    return {
        'n_episodes': n_ep,
        **hit_rates,
        'mean_lead_days': round(float(np.mean(lead_days)), 1) if lead_days else None,
        'episodes': details,
    }


def false_alarm_stats_with_sp500(dates, y, probs, sp500, threshold, horizon=20):
    y_vals = y.values
    n = len(dates)
    alarms = []
    i = 0
    while i < n:
        if y_vals[i] == 0 and probs[i] >= threshold:
            start_i = i
            start_date = str(dates[i].date())
            max_prob = float(probs[i])
            post_dds = []
            while i < n and y_vals[i] == 0 and probs[i] >= threshold:
                max_prob = max(max_prob, float(probs[i]))
                dd = forward_max_dd(sp500, dates[i], horizon)
                if np.isfinite(dd):
                    post_dds.append(dd)
                i += 1
            end_date = str(dates[i - 1].date())
            alarms.append({
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': i - start_i,
                'max_prob': round(max_prob, 4),
                'forward_20d_dd_pct': round(float(min(post_dds)) * 100, 2) if post_dds else None,
            })
        else:
            i += 1
    test_years = len(dates) / 252.0
    alert_days = int((probs >= threshold).sum())
    return {
        'n_false_alarms': len(alarms),
        'false_alarms_per_year': round(len(alarms) / test_years, 2) if test_years > 0 else None,
        'alert_day_pct': round(alert_days / n * 100, 1) if n else 0,
        'false_alarms': alarms,
    }


def run_fold_episodes(model_name, X, y, sp500, fold, train_fn=train_lr_no_balance):
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

    if y_train.nunique() < 2 or len(y_test) < 20:
        return None

    threshold, probs_test, train_pm = train_threshold(X_train, y_train, X_test, y_test, train_fn)
    test_dates = X_test.index

    episodes = find_episodes(test_dates, y_test, sp500)
    if len(episodes) == 0:
        ep_stats = {f'hit_at_{w}': None for w in HIT_WINDOWS}
        ep_stats.update({'n_episodes': 0, 'mean_lead_days': None, 'episodes': []})
    else:
        ep_stats = episode_hits(episodes, test_dates, probs_test, threshold)

    fa_stats = false_alarm_stats_with_sp500(test_dates, y_test, probs_test, sp500, threshold)
    test_pm = compute_practical_metrics(y_test, probs_test)

    return {
        'model': model_name,
        'fold': fold['fold'],
        'test_period': f"{fold['test_start']} ~ {fold['test_end']}",
        'train_threshold': threshold,
        'train_best_f1': train_pm['best_f1'],
        'test_best_f1': test_pm['best_f1'],
        'test_brier': test_pm['brier_score'],
        'episode_eval': {
            'n_episodes': ep_stats['n_episodes'],
            'hit_at_10': ep_stats.get('hit_at_10'),
            'hit_at_15': ep_stats.get('hit_at_15'),
            'hit_at_20': ep_stats.get('hit_at_20'),
            'mean_lead_days': ep_stats.get('mean_lead_days'),
            'episodes': ep_stats.get('episodes', []),
        },
        'false_alarms': {
            'n_false_alarms': fa_stats['n_false_alarms'],
            'false_alarms_per_year': fa_stats['false_alarms_per_year'],
            'alert_day_pct': fa_stats['alert_day_pct'],
            'events': fa_stats['false_alarms'],
        },
    }


def summarize_episodes(results: list):
    if not results:
        return {}
    out = {'n_folds': len(results)}
    total_ep = 0
    hits = {w: 0 for w in HIT_WINDOWS}
    leads = []
    fa_yr = []
    for r in results:
        total_ep += r['episode_eval']['n_episodes']
        for ep in r['episode_eval'].get('episodes', []):
            for w in HIT_WINDOWS:
                if ep.get(f'hit_at_{w}'):
                    hits[w] += 1
            if ep.get('lead_days') is not None:
                leads.append(ep['lead_days'])
        fa = r['false_alarms']['false_alarms_per_year']
        if fa is not None:
            fa_yr.append(fa)
    for w in HIT_WINDOWS:
        out[f'hit_at_{w}'] = round(hits[w] / total_ep, 3) if total_ep else None
    out['mean_lead_days'] = round(float(np.mean(leads)), 1) if leads else None
    out['total_episodes'] = total_ep
    out['false_alarms_per_year_mean'] = round(float(np.mean(fa_yr)), 2) if fa_yr else None
    return out


def main():
    print('=' * 60)
    print('EPISODE-BASED EVALUATION (Supplementary, WF)')
    print('=' * 60)

    df = load_indicators()
    df.index = pd.to_datetime(df.index)
    sp500 = df['sp500']
    target = compute_target(sp500)
    regime_df = fetch_regime_data()

    X_events, y_events = prep_xy(build_features_with_events(df), target)
    X_combo, y_combo = prep_xy(build_events_interact_features(df, regime_df), target)

    model_xy = {
        'LR Slim+Events': (X_events, y_events),
        'LR Events+Interact': (X_combo, y_combo),
    }

    folds = generate_folds(len(X_events), X_events.index)
    all_results = []
    summary_by_model = {}

    for model_name, (X, y) in model_xy.items():
        print(f"\n--- {model_name} ---")
        fold_results = []
        for fold in folds:
            row = run_fold_episodes(model_name, X, y, sp500, fold)
            if row is None:
                print(f"  Fold {fold['fold']}: skipped")
                continue
            fold_results.append(row)
            ee = row['episode_eval']
            fa = row['false_alarms']
            print(
                f"  Fold {row['fold']}: episodes={ee['n_episodes']} "
                f"Hit@10/15/20={ee['hit_at_10']}/{ee['hit_at_15']}/{ee['hit_at_20']} "
                f"lead={ee['mean_lead_days']}d thresh={row['train_threshold']:.2f} "
                f"FA/yr={fa['false_alarms_per_year']}"
            )
        all_results.extend(fold_results)
        summary_by_model[model_name] = summarize_episodes(fold_results)

    output = {
        'title': 'Episode-Based Evaluation (Supplementary)',
        'role': 'supplementary',
        'primary_metrics_unchanged': 'Best F1 (primary) · Brier (secondary)',
        'design': {
            'threshold': 'best_f1_threshold from TRAIN sweep (same grid as F1)',
            'hit_windows_days': HIT_WINDOWS,
            'default_hit_window': 20,
            'episode_definition': 'contiguous target=1 clusters; peak = worst forward 20d DD in cluster',
            'false_alarm': 'target=0 day with prob≥threshold; consecutive elevated days deduped',
            'models': MODELS,
            'walk_forward': 'expanding window, same folds as experiment_walkforward.py',
        },
        'results': all_results,
        'summary_by_model': summary_by_model,
        'verdict': [
            '辅指标：不参与 primary F1 模型排序。',
            '默认解读 Hit@20（与 20d target horizon 对齐）；Hit@10/15 供控仓提前量敏感性参考。',
            f"LR Slim+Events: Hit@20 均值={summary_by_model.get('LR Slim+Events', {}).get('hit_at_20', '—')}, "
            f"FA/yr={summary_by_model.get('LR Slim+Events', {}).get('false_alarms_per_year_mean', '—')}",
            f"LR Events+Interact: Hit@20 均值={summary_by_model.get('LR Events+Interact', {}).get('hit_at_20', '—')}, "
            f"FA/yr={summary_by_model.get('LR Events+Interact', {}).get('false_alarms_per_year_mean', '—')}",
        ],
    }

    out_path = DATA_DIR / 'episode_eval_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")

    p3 = DATA_DIR / 'phase3_metrics.json'
    if p3.exists():
        phase3 = json.loads(p3.read_text())
        phase3['episode_eval'] = output
        with open(p3, 'w') as f:
            json.dump(phase3, f)
        print(f"Merged episode_eval into {p3}")


if __name__ == '__main__':
    main()
