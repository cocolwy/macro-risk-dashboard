"""
Autoresearch runner — executes one experiment cycle.

This is the script the agent calls: `python autoresearch/run.py`
It loads experiment_auto.py, runs the walk-forward evaluation via the
harness, prints the summary, and exits.

The agent then reads the output, decides keep/discard, and loops.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_harness import (
    evaluate,
    get_macro_crash_config,
    print_summary,
)
from experiment_auto import build_features, train_model, DESCRIPTION


def main():
    print(f'=== autoresearch: {DESCRIPTION} ===')
    t0 = time.time()

    config = get_macro_crash_config()
    print(f'Research track: {config.name}')

    extra_data = None
    if config.load_extra:
        print('Loading extra data...')
        extra_data = config.load_extra()

    print('Running walk-forward evaluation...')
    result = evaluate(build_features, train_model, config, extra_data=extra_data)

    print_summary(result)

    elapsed_total = time.time() - t0
    print(f'total_seconds:    {elapsed_total:.1f}')

    if result.get('overfit_flag'):
        print('\n⚠ OVERFIT WARNING: mean F1 exceeds ceiling — treat with suspicion')
    if result.get('guardrail_warnings'):
        print(f"\n⚠ GUARDRAIL WARNINGS: {len(result['guardrail_warnings'])}")

    sys.exit(0 if result['status'] == 'ok' else 1)


if __name__ == '__main__':
    main()
