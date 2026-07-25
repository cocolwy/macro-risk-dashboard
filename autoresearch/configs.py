"""
Research track configs — add new tracks here.

Each config defines a self-contained research problem:
  - How to load data
  - How to build the target
  - Walk-forward parameters
  - Composite score weights
  - Guardrails

To add a new track (e.g. micro-structure):
  1. Define a get_xxx_config() function below
  2. Create a new experiment file (e.g. experiment_micro.py)
  3. Create a new program_micro.md with research instructions
  4. Create a new run_micro.py that imports the right config + experiment

The evaluate_harness.py is shared across all tracks — no changes needed.
"""

from evaluate_harness import ResearchConfig


def get_micro_structure_config() -> ResearchConfig:
    """Template config for micro-structure research.

    Uncomment and implement load_data / build_target when ready.
    Micro-structure typically uses intraday data (order flow, bid-ask spread,
    volume profile, etc.) to predict short-term price moves.
    """
    raise NotImplementedError(
        'Micro-structure config not yet implemented. '
        'Define load_data() and build_target() for your intraday dataset.'
    )

    # Example skeleton:
    # def _load():
    #     import pandas as pd
    #     df = pd.read_parquet('data/intraday_features.parquet')
    #     return df
    #
    # def _target(df):
    #     # e.g. predict 5-min forward return sign
    #     return (df['close'].pct_change(5).shift(-5) > 0).astype(int)
    #
    # return ResearchConfig(
    #     name='micro_structure',
    #     load_data=_load,
    #     build_target=_target,
    #     min_train_days=5000,      # ~20 trading days of 1-min bars
    #     test_days=1000,           # ~4 trading days
    #     step_days=1000,
    #     embargo_days=100,         # ~2 hours of 1-min bars
    #     f1_weight=0.5,
    #     brier_weight=0.5,
    #     max_features=50,
    #     f1_ceiling=0.65,          # lower ceiling for noisier data
    #     forbid_balanced_training=True,
    # )


def get_factor_rotation_config() -> ResearchConfig:
    """Template config for equity factor rotation research.

    Predict which factor (momentum, value, size, etc.) will outperform
    next month, using macro regime indicators.
    """
    raise NotImplementedError(
        'Factor rotation config not yet implemented.'
    )
