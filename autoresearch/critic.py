"""
Autoresearch Critic — post-experiment result analysis.

After an experiment finishes, the Critic decides:
  KEEP     — improvement is real and worth the complexity
  DISCARD  — no improvement, or improvement is suspicious
  FLAG     — possible improvement but needs human review

Checks:
  1. Statistical significance (is the delta meaningful?)
  2. Fold stability (does it improve in most folds, or just one?)
  3. Complexity budget (is the improvement worth the added features?)
  4. Regression detection (did it hurt Brier while helping F1, or vice versa?)
  5. Overfitting signals (fold variance, ceiling check)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CriticVerdict:
    """The Critic's decision on an experiment."""
    decision: str          # 'keep', 'discard', 'flag'
    composite_delta: float
    reasons: list[str]     # why this decision
    warnings: list[str]    # observations that don't block
    fold_detail: str       # per-fold breakdown

    def __str__(self):
        icon = {'keep': '✓', 'discard': '✗', 'flag': '⚠'}[self.decision]
        lines = [f'{icon} VERDICT: {self.decision.upper()} (Δ composite = {self.composite_delta:+.6f})']
        for r in self.reasons:
            lines.append(f'  → {r}')
        if self.warnings:
            lines.append('  Warnings:')
            for w in self.warnings:
                lines.append(f'    ⚠ {w}')
        lines.append(f'  Fold detail: {self.fold_detail}')
        return '\n'.join(lines)


# Thresholds
MIN_MEANINGFUL_DELTA = 0.002     # composite improvement must exceed this
MIN_FOLDS_IMPROVED = 0.5        # fraction of folds that must improve
MAX_FOLD_F1_STD = 0.35          # F1 std across folds above this = unstable
MAX_FEATURES_PER_COMPOSITE_POINT = 50  # features per 0.01 composite improvement
BRIER_REGRESSION_THRESHOLD = 0.01  # Brier worsening more than this = penalty


def critique(current_result: dict, baseline_result: dict,
             baseline_features: int = 0) -> CriticVerdict:
    """Compare current experiment against baseline and issue verdict.

    Parameters
    ----------
    current_result : dict from evaluate_harness.evaluate()
    baseline_result : dict from evaluate_harness.evaluate()
    baseline_features : int, number of features in baseline (for complexity check)
    """
    reasons = []
    warnings = []

    # --- Basic checks ---
    if current_result.get('status') != 'ok':
        return CriticVerdict(
            decision='discard',
            composite_delta=0.0,
            reasons=[f"Experiment failed: {current_result.get('error', 'unknown')}"],
            warnings=[],
            fold_detail='N/A (experiment failed)',
        )

    cur_composite = current_result['composite_score']
    base_composite = baseline_result.get('composite_score', 0)
    delta = cur_composite - base_composite

    cur_f1 = current_result['mean_f1']
    base_f1 = baseline_result.get('mean_f1', 0)
    cur_brier = current_result['mean_brier']
    base_brier = baseline_result.get('mean_brier', 1.0)

    cur_features = current_result['n_features']
    cur_folds = current_result.get('fold_results', [])
    ok_folds = [f for f in cur_folds if f.get('status') == 'ok']

    # --- 1. Minimum improvement threshold ---
    if delta < MIN_MEANINGFUL_DELTA:
        reasons.append(
            f'Δ composite = {delta:+.6f} < minimum threshold {MIN_MEANINGFUL_DELTA}')

    # --- 2. Fold stability ---
    fold_f1s = [f['best_f1'] for f in ok_folds]
    fold_briers = [f['brier_score'] for f in ok_folds]

    if len(fold_f1s) >= 2:
        f1_std = float(np.std(fold_f1s))
        if f1_std > MAX_FOLD_F1_STD:
            warnings.append(
                f'High F1 variance across folds: std={f1_std:.3f} '
                f'(threshold {MAX_FOLD_F1_STD})')

    # Check how many folds improved vs baseline
    if baseline_result.get('fold_results'):
        base_ok = [f for f in baseline_result['fold_results'] if f.get('status') == 'ok']
        base_fold_composites = {
            f['fold']: 0.6 * f['best_f1'] + 0.4 * (1 - f['brier_score'])
            for f in base_ok
        }
        cur_fold_composites = {
            f['fold']: 0.6 * f['best_f1'] + 0.4 * (1 - f['brier_score'])
            for f in ok_folds
        }
        common_folds = set(base_fold_composites) & set(cur_fold_composites)
        if common_folds:
            improved = sum(
                1 for fold_id in common_folds
                if cur_fold_composites[fold_id] > base_fold_composites[fold_id]
            )
            frac = improved / len(common_folds)
            fold_detail = (
                f'{improved}/{len(common_folds)} folds improved '
                f'({frac:.0%})')
            if frac < MIN_FOLDS_IMPROVED:
                reasons.append(
                    f'Only {improved}/{len(common_folds)} folds improved '
                    f'(need ≥{MIN_FOLDS_IMPROVED:.0%})')
        else:
            fold_detail = 'No common folds to compare'
    else:
        fold_detail = 'No baseline fold data for comparison'

    # --- 3. Complexity budget ---
    if baseline_features > 0 and delta > 0:
        added_features = cur_features - baseline_features
        if added_features > 0 and delta > 0:
            efficiency = added_features / (delta / 0.01)
            if efficiency > MAX_FEATURES_PER_COMPOSITE_POINT:
                warnings.append(
                    f'Low efficiency: {added_features} features added for '
                    f'Δ={delta:+.4f} ({efficiency:.0f} features per 0.01 composite)')

    # --- 4. Regression detection ---
    f1_delta = cur_f1 - base_f1
    brier_delta = cur_brier - base_brier  # positive = worse

    if brier_delta > BRIER_REGRESSION_THRESHOLD and f1_delta > 0:
        warnings.append(
            f'F1 improved ({f1_delta:+.3f}) but Brier regressed '
            f'({brier_delta:+.4f}) — check calibration')

    if f1_delta < -0.02 and brier_delta < 0:
        warnings.append(
            f'Brier improved ({brier_delta:+.4f}) but F1 regressed '
            f'({f1_delta:+.3f}) — tradeoff')

    # --- 5. Overfitting signals ---
    if current_result.get('overfit_flag'):
        reasons.append(
            f'Overfit flag triggered: mean F1 = {cur_f1:.3f} exceeds ceiling')

    if current_result.get('guardrail_warnings'):
        for gw in current_result['guardrail_warnings']:
            warnings.append(f'Guardrail: {gw}')

    # --- Final decision ---
    if reasons:
        decision = 'discard'
    elif warnings and delta < 0.005:
        decision = 'flag'
    else:
        if delta >= MIN_MEANINGFUL_DELTA:
            reasons.append(f'Improvement: Δ composite = {delta:+.6f}')
        decision = 'keep' if delta >= MIN_MEANINGFUL_DELTA else 'discard'

    return CriticVerdict(
        decision=decision,
        composite_delta=round(delta, 6),
        reasons=reasons if reasons else ['Meets all criteria'],
        warnings=warnings,
        fold_detail=fold_detail,
    )


def format_comparison_table(current: dict, baseline: dict) -> str:
    """Pretty table comparing current vs baseline metrics."""
    lines = [
        '┌─────────────────┬──────────┬──────────┬──────────┐',
        '│ Metric          │ Baseline │ Current  │ Delta    │',
        '├─────────────────┼──────────┼──────────┼──────────┤',
    ]

    metrics = [
        ('composite_score', 'composite', '.6f', 1),
        ('mean_f1', 'mean F1', '.3f', 1),
        ('mean_brier', 'mean Brier', '.4f', -1),   # lower is better
        ('n_features', 'features', 'd', -1),        # fewer is better
    ]

    for key, label, fmt, direction in metrics:
        bv = baseline.get(key, 0)
        cv = current.get(key, 0)
        d = cv - bv
        sign = '+' if d > 0 else ''
        if isinstance(bv, int):
            bstr = f'{bv:>8d}'
            cstr = f'{cv:>8d}'
            dstr = f'{sign}{d:>7d}'
        else:
            bstr = f'{bv:>8{fmt}}'
            cstr = f'{cv:>8{fmt}}'
            dstr = f'{sign}{d:>7{fmt}}'

        good = (d * direction > 0)
        marker = ' ✓' if good and abs(d) > 0.0001 else ('✗' if d * direction < -0.001 else '  ')
        lines.append(f'│ {label:<15} │ {bstr} │ {cstr} │ {dstr}{marker}│')

    lines.append('└─────────────────┴──────────┴──────────┴──────────┘')
    return '\n'.join(lines)
