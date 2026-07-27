# autoresearch — Research Track: Macro Crash Prediction

Reference document for the macro crash prediction research track.
Read this file to understand the problem, constraints, prior results,
and promising research directions.

**Orchestration is handled by the `/experiment` command in agent-system.**
This file provides context, not execution instructions.

## Problem Definition

**Target**: Binary — 1 if S&P 500 max drawdown in next 20 trading days exceeds -5%

**Features**: Rate-of-change transforms on macro indicators:
- VIX (level, 20d change)
- Turbulence index (20d change)
- Absorption ratio (20d change)
- Term spread (level, 20d change)
- Credit spread (10d change)
- Market breadth (level, 10d change)
- SP500 relative to 50MA
- Event calendar (FOMC/CPI/NFP proximity)

**Evaluation**: Walk-forward expanding window (378d min train, 126d test, 20d embargo)

**Metric**: `composite_score = 0.6 × mean_F1 + 0.4 × (1 − mean_Brier)`

**Verified baseline**: composite=0.4723 (F1=0.216, Brier=0.1431, 19 features)

## File Structure

| File | Role | Who modifies |
|------|------|-------------|
| `experiment_auto.py` | Features + model definition | Coder subagent |
| `evaluate_harness.py` | Walk-forward evaluation | Nobody (ground truth) |
| `run.py` | 3-stage pipeline (Reviewer → Runner → Critic) | Nobody |
| `review_checklist.py` | Static code analysis | Nobody |
| `critic.py` | Result validation | Nobody |
| `configs.py` | Research track configs | Human (new tracks) |
| `baseline_result.json` | Current best for comparison | Auto-updated on keep |
| `results.tsv` | Experiment log | Auto-appended |

## Constraints (hard rules)

1. **Only modify `experiment_auto.py`** — all other files are read-only
2. **No `class_weight='balanced'`** — inflates probabilities (F1 jumps 0.467→0.588 artificially)
3. **No new packages** — only what's in requirements.txt
4. **Walk-forward is the only valid test** — single-split results are unreliable
5. **Overfit ceiling: F1 > 0.80 = suspicious** — the harness auto-flags this
6. **Max 30 features** — each added feature is an overfitting degree of freedom
7. **Max feature correlation < 0.85** — redundant features waste capacity

## Interface Contract

`experiment_auto.py` must define:

```python
DESCRIPTION: str  # logged to results.tsv

def build_features(df: pd.DataFrame, extra_data=None) -> pd.DataFrame:
    """df has columns: vix, turbulence, absorption_ratio, term_spread,
    credit_spread, breadth, sp500. extra_data = regime DataFrame.
    Return feature matrix (may contain NaN from lookbacks)."""

def train_model(X_train, y_train, X_test) -> (model, scaler, probs_test):
    """Train on X_train/y_train, predict probabilities on X_test.
    X_train/X_test are already percentile-clipped by the harness."""
```

## Pipeline Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | keep (Critic approved) | Advance branch, update baseline |
| 1 | discard (no improvement or crash) | `git reset --hard HEAD~1` |
| 2 | review blocked (code issue) | Fix code, recommit |
| 3 | flag (marginal) | Keep commit, log as "flag" |

## Reviewer Blocks These

- `class_weight='balanced'` in code
- `.shift(-N)` (future leakage)
- `X_test` in `fit_transform()` (data leakage)
- `GridSearchCV` (overfits validation)
- `torch` / `tensorflow` imports
- Missing `build_features` or `train_model`
- Importing `evaluate_harness`

## Critic Rejects These

- Δ composite < 0.002 (too small)
- < 50% folds improved (unstable)
- F1 up but Brier regressed > 0.01 (tradeoff, not win)
- > 50 features per 0.01 composite improvement (complexity not worth it)
- mean_f1 > 0.80 (overfit flag)

## Known Results from Prior Experiments

Source: `dashboard/experiment_phase3.py` + `dashboard/experiment_walkforward.py`

| Finding | Implication |
|---------|------------|
| LR Slim (10 feat) ≈ LR Full (23 feat) | Fewer features is often better |
| GBDT single-split F1 high, WF F1 collapses | GBDT overfits on small macro data |
| Event features give marginal LR improvement | Worth keeping, low cost |
| Regime interactions help Brier, may hurt F1 | Tradeoff — try carefully |
| Time decay (half-life=252d) gives Δ≈0 | Probably not worth the complexity |
| `class_weight='balanced'` inflates F1 0.467→0.588 | Forbidden — artificial |

## Research Directions (prioritized)

### Tier 1: Most likely to help
- Feature selection: mutual info ranking, remove low-importance features
- Interaction terms: `vix_level × credit_spread_10d_chg`
- Event encoding: exponential decay `exp(-days_to/5)` instead of binary windows
- Regime conditioning: `tight × vix_level` interaction
- Lookback window search: 10d vs 20d vs 50d per indicator

### Tier 2: Worth trying
- GBDT with heavy regularization (max_depth=3, min_samples_leaf=30)
- Rolling Z-scores instead of raw pct_change
- Non-linear transforms: log(VIX), sqrt(turbulence)
- Rolling rank: `df[col].rolling(252).rank(pct=True)`
- Vol of vol: `vix.rolling(10).std()`

### Tier 3: Speculative
- LR + GBDT probability averaging
- Stacking (LR probs as GBDT feature)
- Feature momentum (acceleration of VIX change)
- Cross-indicator ratios: `credit_spread / term_spread`

## Data Characteristics

- ~950 samples (2022-01 to 2026-07)
- ~60 crash episodes (positive rate ~6%)
- Walk-forward produces 4-5 folds
- Each fold has 10-20 positive samples in test
- **Implication**: Very small dataset. Every feature competes for limited signal.
  Simpler is almost always better.
