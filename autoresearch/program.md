# autoresearch — Macro Crash Prediction

Autonomous experiment loop for improving the macro crash prediction model.
You are an AI research agent. Your job: iterate on `experiment_auto.py` to
lower the composite score's Brier component and raise F1, measured by
walk-forward validation. You run experiments, keep what works, discard what
doesn't, and never stop until the human interrupts you.

## Setup

1. **Agree on a run tag** with the user (e.g. `jul25`). Create branch
   `autoresearch/<tag>` from current HEAD.
2. **Read the in-scope files** for full context:
   - This file (`program.md`) — your instructions
   - `experiment_auto.py` — **the file you modify** (features + model)
   - `evaluate_harness.py` — fixed evaluation (DO NOT MODIFY)
   - `run.py` — fixed runner (DO NOT MODIFY)
   - `../dashboard/predict_model.py` — reference: existing feature builders
   - `../dashboard/experiment_phase3.py` — reference: model variants tested
3. **Initialize `results.tsv`**: header only (the runner auto-creates it)
4. **Run the baseline**: `python autoresearch/run.py > run.log 2>&1`
5. **Log baseline** to `results.tsv` and confirm with user.

Once confirmed, begin the experiment loop.

## What you CAN modify

Only `experiment_auto.py`. Everything is fair game:

- **Feature engineering** (`build_features`):
  - Add/remove/transform features
  - Change lookback windows (5d, 10d, 20d, 50d, ...)
  - Add interaction terms (e.g. `vix * credit_spread`)
  - Add Z-score normalization, rolling percentile ranks
  - Add regime features (use `extra_data` param for regime_df)
  - Add polynomial features, ratios, diffs
  - Remove features to simplify

- **Model definition** (`train_model`):
  - Switch model type (LR, GBDT, RF, Ridge, ElasticNet, ...)
  - Tune hyperparameters (C, max_depth, learning_rate, ...)
  - Add feature selection (mutual information, L1 path, ...)
  - Ensemble multiple models
  - Add calibration (isotonic, Platt)
  - Change scaler (StandardScaler, RobustScaler, QuantileTransformer)

- `DESCRIPTION` string (update every experiment)

## What you CANNOT modify

- `evaluate_harness.py` — the ground truth evaluation
- `run.py` — the experiment runner
- Any file in `../dashboard/` — production code
- Do NOT install new packages
- Do NOT use `class_weight='balanced'` or equivalent sample_weight
  (known to inflate probabilities — see eval-metrics.mdc)

## The goal

**Maximize `composite_score`** (printed in the `---` summary block):

```
composite_score = 0.6 × mean_F1 + 0.4 × (1 − mean_Brier)
```

Both components are walk-forward averages across expanding-window folds.
Higher is better. The verified baseline is **0.4723** (F1=0.216, Brier=0.1431).

## Output format

The runner prints a summary block:

```
---
composite_score:  0.512345
mean_f1:          0.250 ± 0.150
mean_brier:       0.0800 ± 0.0200
n_features:       19
n_samples:        953
n_folds:          4/4
overfit_flag:     False
elapsed_seconds:  12.3
---
```

Extract the key metric:
```bash
grep "^composite_score:" run.log
```

## The experiment loop

LOOP FOREVER:

1. Look at git state and current results.tsv
2. Modify `experiment_auto.py` with an experimental idea
3. Update `DESCRIPTION` to describe what this experiment tries
4. `git add autoresearch/experiment_auto.py && git commit -m "exp: <description>"`
5. Run: `python autoresearch/run.py > run.log 2>&1`
6. Read results: `grep "^composite_score:\|^mean_f1:\|^mean_brier:\|^overfit_flag:" run.log`
7. If grep is empty → crash. Run `tail -n 50 run.log` for the traceback
8. Log to results.tsv (do NOT commit results.tsv — keep it untracked)
9. If composite_score improved → **keep** (advance the branch)
10. If composite_score equal or worse → **discard** (`git reset --hard HEAD~1`)
11. GOTO 1

## Research directions to explore

Ordered roughly by expected impact:

### Tier 1: Most likely to help
- **Feature selection**: Remove low-importance features, test mutual info ranking
- **Interaction terms**: `vix_level × credit_spread_10d_chg`, `term_spread × breadth_level`
- **Better event encoding**: Exponential decay (`exp(-days_to/5)`) instead of binary windows
- **Regime conditioning**: Add `tight × vix_level` interaction (tight = inverted curve or hiking)
- **Lookback window search**: Test 10d vs 20d vs 50d for each indicator

### Tier 2: Worth trying
- **GBDT instead of LR**: HistGradientBoostingClassifier (max_depth=4, lr=0.05, max_iter=200)
- **Feature Z-scores**: Rolling 60d Z-score instead of raw pct_change
- **Non-linear transforms**: Log(VIX), sqrt(turbulence)
- **Rolling rank**: `df[col].rolling(252).rank(pct=True)`
- **Volatility of volatility**: `vix.rolling(10).std()`
- **Cross-indicator ratios**: `credit_spread / term_spread`

### Tier 3: Speculative
- **Ensemble**: Average LR + GBDT probabilities
- **Stacking**: Use LR probs as feature for GBDT
- **Time decay weighting**: Recent samples weighted higher
- **Target engineering**: Different horizon (10d, 30d) or threshold (-3%, -7%)
  — note: harness fixes target at 20d/-5%, but you can add auxiliary targets as features
- **Momentum of features**: 2nd derivative (acceleration of VIX change)

## Known results from previous experiments

(Read `../dashboard/experiment_phase3.py` for details)

- LR Slim (10 features) is competitive with LR Full (23 features) — fewer is often better
- GBDT shows higher single-split F1 but collapses under walk-forward (overfitting)
- Event features (FOMC/CPI proximity) provide marginal improvement to LR
- Regime interaction terms help Brier but may hurt F1
- `class_weight='balanced'` inflates F1 from 0.467 → 0.588 but is artificial

## Important constraints

- **Overfitting is the enemy.** Walk-forward with embargo is the only valid test.
  If mean_f1 > 0.80, it's almost certainly overfit — the harness flags this.
- **Data is small.** ~950 samples, ~60 crash episodes since 2022. Every feature
  added is a degree of freedom that can overfit.
- **Simplicity wins.** All else equal, fewer features is better. A 0.001
  improvement from adding 5 features is not worth it. A 0.001 improvement
  from removing features is a great result.
- **Each experiment takes ~3 seconds** (no GPU needed, small dataset).
  You can run ~1200 experiments/hour, ~10000 overnight.

## NEVER STOP

Once the loop begins, do NOT pause to ask the human if you should continue.
Do NOT ask "should I keep going?" or "is this a good stopping point?".
The human may be asleep. You run until manually stopped.

If you run out of ideas:
1. Re-read `experiment_phase3.py` for angles you haven't tried
2. Combine previous near-misses (e.g. "decay events" + "interaction terms")
3. Try more radical changes (different model entirely, feature selection)
4. Ablate: remove features one at a time to find what's actually helping
5. Try the opposite of your last failed idea
