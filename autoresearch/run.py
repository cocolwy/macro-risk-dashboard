"""
Autoresearch runner — multi-agent experiment pipeline.

Pipeline stages:
  1. REVIEWER  — static analysis of experiment_auto.py (data leakage, overfitting, correctness)
  2. RUNNER    — walk-forward evaluation via the harness
  3. CRITIC    — result analysis (significance, fold stability, complexity budget)

Usage:
  python autoresearch/run.py                    # full pipeline
  python autoresearch/run.py --skip-review      # skip code review (debugging)
  python autoresearch/run.py --baseline-file X  # compare against saved baseline JSON
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_harness import (
    evaluate,
    get_macro_crash_config,
    print_summary,
    append_result,
)
from experiment_auto import build_features, train_model, DESCRIPTION
from review_checklist import review_experiment, has_blockers, format_review
from critic import critique, format_comparison_table, CriticVerdict

AUTORESEARCH_DIR = Path(__file__).parent
BASELINE_FILE = AUTORESEARCH_DIR / 'baseline_result.json'


def save_baseline(result: dict):
    """Save current result as the baseline for future comparisons."""
    with open(BASELINE_FILE, 'w') as f:
        json.dump(result, f, indent=2)


def load_baseline() -> dict:
    """Load saved baseline, or return empty dict."""
    if BASELINE_FILE.exists():
        return json.loads(BASELINE_FILE.read_text())
    return {}


def main():
    parser = argparse.ArgumentParser(description='autoresearch experiment runner')
    parser.add_argument('--skip-review', action='store_true', help='Skip code review stage')
    parser.add_argument('--baseline-file', type=str, help='Path to baseline result JSON')
    parser.add_argument('--save-baseline', action='store_true', help='Save this result as baseline')
    args = parser.parse_args()

    print(f'=== autoresearch: {DESCRIPTION} ===')
    print(f'Pipeline: Reviewer → Runner → Critic\n')
    t0 = time.time()

    # ---------------------------------------------------------------
    # Stage 1: REVIEWER — code review
    # ---------------------------------------------------------------
    print('─── Stage 1: REVIEWER ───')
    if args.skip_review:
        print('  (skipped)\n')
    else:
        issues = review_experiment()
        print(format_review(issues))
        print()
        if has_blockers(issues):
            print('PIPELINE ABORTED: fix blockers before running.\n')
            sys.exit(2)

    # ---------------------------------------------------------------
    # Stage 2: RUNNER — walk-forward evaluation
    # ---------------------------------------------------------------
    print('─── Stage 2: RUNNER ───')
    config = get_macro_crash_config()
    print(f'Research track: {config.name}')

    extra_data = None
    if config.load_extra:
        print('Loading extra data...')
        extra_data = config.load_extra()

    print('Running walk-forward evaluation...')
    result = evaluate(build_features, train_model, config, extra_data=extra_data)
    print_summary(result)

    if result['status'] != 'ok':
        print(f'RUNNER FAILED: {result.get("error", "unknown")}\n')
        sys.exit(1)

    # Save baseline if requested
    if args.save_baseline:
        save_baseline(result)
        print(f'Baseline saved to {BASELINE_FILE}\n')

    # ---------------------------------------------------------------
    # Stage 3: CRITIC — result analysis
    # ---------------------------------------------------------------
    print('─── Stage 3: CRITIC ───')

    baseline_path = Path(args.baseline_file) if args.baseline_file else BASELINE_FILE
    baseline = {}
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        print(f'Comparing against baseline: {baseline_path.name}')
    else:
        print('No baseline found — this is the first run.')
        print('Run with --save-baseline to establish the baseline.\n')
        print_summary(result)
        elapsed = time.time() - t0
        print(f'total_seconds:    {elapsed:.1f}')
        sys.exit(0)

    # Comparison table
    print()
    print(format_comparison_table(result, baseline))
    print()

    # Critic verdict
    verdict = critique(
        result, baseline,
        baseline_features=baseline.get('n_features', 0),
    )
    print(str(verdict))
    print()

    # Output machine-readable verdict line for agent parsing
    print(f'verdict:          {verdict.decision}')
    print(f'composite_delta:  {verdict.composite_delta:+.6f}')

    elapsed = time.time() - t0
    print(f'total_seconds:    {elapsed:.1f}')

    # Exit codes: 0 = keep, 1 = discard, 3 = flag (needs human review)
    code = {'keep': 0, 'discard': 1, 'flag': 3}.get(verdict.decision, 1)
    sys.exit(code)


if __name__ == '__main__':
    main()
