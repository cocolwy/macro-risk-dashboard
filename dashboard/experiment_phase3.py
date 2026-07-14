"""
Phase 3: Model Evolution — Non-linear Models & Feature Exploration

Step 1: XGBoost vs Logistic Regression on existing features
Step 2: Event calendar features (FOMC/CPI/NFP proximity)
Step 3: Regime-as-context (conditional models, interaction features, post-hoc calibration)
Step 4: Long-term (2005+) retest with non-linear models

Each step is an A/B experiment added to phase3_metrics.json.
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from predict_model import (
    load_indicators, build_features, build_features_slim,
    build_features_with_events, fetch_regime_data,
    compute_target, KEY_EVENTS, build_comparison_metrics,
)
from experiment_extended_history import load_extended_data, EXTENDED_KEY_EVENTS

DATA_DIR = Path(__file__).parent / 'data'
EMBARGO = 20


def train_lr(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
    model.fit(X_train_s, y_train)
    return model, scaler, model.predict_proba(X_test_s)[:, 1]


def train_gbdt(X_train, y_train, X_test):
    """HistGradientBoosting — sklearn's built-in XGBoost equivalent."""
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    sample_weight = np.where(y_train == 1, neg / pos, 1.0)

    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    probs = model.predict_proba(X_test)[:, 1]
    return model, None, probs


def train_rf(X_train, y_train, X_test):
    """Random Forest baseline."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return model, None, probs


def train_gbdt_calibrated(X_train, y_train, X_test):
    """GBDT with isotonic probability calibration via 3-fold CV."""
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    sample_weight = np.where(y_train == 1, neg / pos, 1.0)

    base = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=20, l2_regularization=1.0, random_state=42,
    )
    model = CalibratedClassifierCV(base, cv=3, method='isotonic')
    model.fit(X_train, y_train, sample_weight=sample_weight)
    probs = model.predict_proba(X_test)[:, 1]
    return model, None, probs


def train_lr_no_balance(X_train, y_train, X_test):
    """LR without class_weight='balanced' — outputs calibrated probabilities."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(C=0.1, max_iter=1000)
    model.fit(X_train_s, y_train)
    return model, scaler, model.predict_proba(X_test_s)[:, 1]


def train_gbdt_no_balance(X_train, y_train, X_test):
    """GBDT without sample_weight — natural probability output."""
    model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=20, l2_regularization=1.0, random_state=42,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return model, None, probs


def train_rf_no_balance(X_train, y_train, X_test):
    """Random Forest without class_weight='balanced'."""
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return model, None, probs


def detect_regime(df, regime_df):
    """Classify each day into a regime: 'tight' or 'normal'.

    Tight = yield curve inverted OR fed actively hiking.
    This is a binary context variable, not a model feature.
    """
    regime_labels = pd.Series('normal', index=df.index)

    inverted = pd.Series(False, index=df.index)
    if 'term_spread' in df.columns:
        inverted = df['term_spread'] < 0

    hiking = pd.Series(False, index=df.index)
    if regime_df is not None and 'fed_funds' in regime_df.columns:
        ff_idx = pd.to_datetime(regime_df.index)
        ff_series = regime_df['fed_funds'].copy()
        ff_series.index = ff_idx
        df_idx = pd.to_datetime(df.index)
        ff = ff_series.reindex(df_idx, method='ffill').fillna(method='bfill').fillna(0)
        ff.index = df.index
        hiking = ff.diff(63).fillna(0) > 0.25

    regime_labels[inverted | hiking] = 'tight'
    return regime_labels


def split_with_embargo(X, y, split_ratio=0.7, embargo=EMBARGO):
    split = int(len(X) * split_ratio)
    train_end = max(split - embargo, 1)
    test_start = min(split + embargo, len(X))
    return (X.iloc[:train_end], X.iloc[test_start:],
            y.iloc[:train_end], y.iloc[test_start:],
            train_end, test_start)


def full_probs(model, scaler, X):
    if scaler is not None:
        return model.predict_proba(scaler.transform(X))[:, 1]
    return model.predict_proba(X)[:, 1]


def compute_practical_metrics(y_test, probs_test):
    """Compute metrics that matter for real use: calibration, best F1, P@high thresholds."""
    brier = brier_score_loss(y_test, probs_test)
    base_rate = float(y_test.mean())

    best_f1, best_f1_thresh = 0, 0.5
    practical = {}
    for thresh_pct in range(10, 95, 5):
        thresh = thresh_pct / 100
        preds = (probs_test > thresh).astype(int)
        tp = ((preds == 1) & (y_test.values == 1)).sum()
        fp = ((preds == 1) & (y_test.values == 0)).sum()
        fn = ((preds == 0) & (y_test.values == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        alert_rate = preds.sum() / len(preds)
        practical[thresh_pct] = {"p": prec, "r": rec, "f1": f1, "alert_rate": alert_rate}
        if f1 > best_f1:
            best_f1, best_f1_thresh = f1, thresh

    p80 = practical.get(80, {}).get('p', 0)
    r80 = practical.get(80, {}).get('r', 0)
    lift_at_80 = p80 / base_rate if base_rate > 0 else 0

    return {
        "brier_score": round(brier, 4),
        "base_rate": round(base_rate, 4),
        "best_f1": round(best_f1, 3),
        "best_f1_threshold": best_f1_thresh,
        "p_at_80": round(p80, 3),
        "r_at_80": round(r80, 3),
        "lift_at_80": round(lift_at_80, 1),
        "mean_prob": round(float(probs_test.mean()), 4),
        "median_prob": round(float(np.median(probs_test)), 4),
        "prob_gt50_pct": round(float((probs_test > 0.5).mean() * 100), 1),
    }


def run_experiment(name, X, y, sp500, train_fn, events=KEY_EVENTS):
    X_train, X_test, y_train, y_test, _, test_start = split_with_embargo(X, y)
    model, scaler, probs_test = train_fn(X_train, y_train, X_test)
    probs_all = full_probs(model, scaler, X)
    test_auc = auc(*roc_curve(y_test, probs_test)[:2])

    result = build_comparison_metrics(y_test, probs_test, probs_all, X, sp500, name, events)

    practical = compute_practical_metrics(y_test, probs_test)
    result['practical_metrics'] = practical

    feature_imp = None
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        feature_imp = [{"feature": k, "importance": round(float(v), 4)} for k, v in imp.items()]
    elif hasattr(model, 'coef_'):
        coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
        feature_imp = [{"feature": k, "importance": round(float(abs(v)), 4)} for k, v in coefs.items()]

    return result, test_auc, feature_imp, model, practical


def percentile_clip(df, target_col='target'):
    """Clip features to 1st-99th percentile (per-column), preserving target."""
    feat_cols = [c for c in df.columns if c != target_col]
    for c in feat_cols:
        lo, hi = df[c].quantile(0.01), df[c].quantile(0.99)
        df[c] = df[c].clip(lo, hi)
    return df


def main():
    print("=" * 60)
    print("PHASE 3: Model Evolution")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/6] Loading data...")
    df = load_indicators()
    sp500 = df['sp500']
    print(f"  {len(df)} trading days ({df.index[0]} ~ {df.index[-1]})")

    features_full = build_features(df)
    features_slim = build_features_slim(df)
    target = compute_target(sp500)

    def prep(features):
        combined = features.copy()
        combined['target'] = target
        combined = combined.dropna()
        combined = percentile_clip(combined)
        X = combined.drop('target', axis=1)
        y = combined['target']
        return X, y

    X_full, y_full = prep(features_full)
    X_slim, y_slim = prep(features_slim)
    print(f"  Full features: {len(X_full)} samples, {X_full.shape[1]} features, {int(y_full.sum())} positive ({y_full.mean()*100:.1f}%)")
    print(f"  Slim features: {len(X_slim)} samples, {X_slim.shape[1]} features, {int(y_slim.sum())} positive ({y_slim.mean()*100:.1f}%)")

    features_events = build_features_with_events(df)
    X_events, y_events = prep(features_events)
    print(f"  Event features: {len(X_events)} samples, {X_events.shape[1]} features")
    print(f"    New event cols: {[c for c in X_events.columns if c not in X_slim.columns]}")

    # =========================================================
    # Board A: 非线性模型探索 (Ch.2)
    # =========================================================
    print("\n[2/6] Ch.2 Non-linear model experiments...")
    model_experiments = []
    model_fi = {}
    model_practical = {}

    def run_and_log(name, X, y, train_fn, exps, fi, prac):
        print(f"  Training {name}...")
        result, test_auc, imp, _, practical = run_experiment(name, X, y, sp500, train_fn)
        exps.append(result)
        prac[name] = practical
        if imp:
            fi[name] = imp
        pm = practical
        print(f"    AUC={test_auc:.3f}  F1={pm['best_f1']:.3f}@{pm['best_f1_threshold']:.0%}  Brier={pm['brier_score']:.4f}  Mean_p={pm['mean_prob']:.3f}")
        return test_auc

    lr_slim_auc = run_and_log("LR Slim", X_slim, y_slim, train_lr_no_balance, model_experiments, model_fi, model_practical)
    lr_full_auc = run_and_log("LR Full", X_full, y_full, train_lr_no_balance, model_experiments, model_fi, model_practical)
    gbdt_slim_auc = run_and_log("GBDT Slim", X_slim, y_slim, train_gbdt_no_balance, model_experiments, model_fi, model_practical)
    gbdt_full_auc = run_and_log("GBDT Full", X_full, y_full, train_gbdt_no_balance, model_experiments, model_fi, model_practical)
    rf_slim_auc = run_and_log("RF Slim", X_slim, y_slim, train_rf_no_balance, model_experiments, model_fi, model_practical)
    rf_full_auc = run_and_log("RF Full", X_full, y_full, train_rf_no_balance, model_experiments, model_fi, model_practical)

    model_pairwise = [
        {
            "id": "linear_vs_gbdt_slim",
            "label": "线性 vs 非线性（Slim 10特征）",
            "variable": "LR vs GBDT — 相同 Slim 特征",
            "baseline": "LR Slim",
            "challenger": "GBDT Slim",
            "method_note": "控制变量：相同的 10 个特征 + Embargo 20d，仅改变模型类型。GBDT 可学到非线性交互（如「VIX 高 且 利差走阔 → 危机」）。",
        },
        {
            "id": "linear_vs_gbdt_full",
            "label": "线性 vs 非线性（Full 23特征）",
            "variable": "LR vs GBDT — 相同 Full 特征",
            "baseline": "LR Full",
            "challenger": "GBDT Full",
            "method_note": "LR 受共线性影响，23 个特征中冗余特征反而有害。树模型天然抗共线性，可能从更多特征中受益。",
        },
        {
            "id": "gbdt_vs_rf",
            "label": "GBDT vs Random Forest（Slim）",
            "variable": "Boosting vs Bagging — 相同 Slim 特征",
            "baseline": "GBDT Slim",
            "challenger": "RF Slim",
            "method_note": "GBDT 逐步纠错（boosting），RF 并行投票（bagging）。GBDT 通常在结构化数据上更强。",
        },
        {
            "id": "slim_vs_full_gbdt",
            "label": "特征数量（GBDT）",
            "variable": "10 特征 vs 23 特征 — GBDT",
            "baseline": "GBDT Slim",
            "challenger": "GBDT Full",
            "method_note": "LR 中去冗余有用（Ch.1），但树模型天然处理冗余。测试 GBDT 是否能利用更多特征。",
        },
    ]

    # =========================================================
    # Step 2: Event calendar features
    # =========================================================
    print("\n[3/6] Step 2: Event calendar experiments...")
    step2_exps = []
    step2_fi = {}
    step2_prac = {}

    run_and_log("LR Slim+Events", X_events, y_events, train_lr_no_balance, step2_exps, step2_fi, step2_prac)
    run_and_log("GBDT Slim+Events", X_events, y_events, train_gbdt_no_balance, step2_exps, step2_fi, step2_prac)

    step2_pairwise = [
        {
            "id": "events_lr",
            "label": "Step 2a: 事件日历增益（LR）",
            "variable": "Slim vs Slim+Events — LR",
            "baseline": "LR Slim",
            "challenger": "LR Slim+Events",
            "method_note": "加入 FOMC/CPI/NFP 前后天数和窗口标记。测试事件邻近性是否有预测价值。",
        },
        {
            "id": "events_gbdt",
            "label": "Step 2b: 事件日历增益（GBDT）",
            "variable": "GBDT Slim vs GBDT Slim+Events",
            "baseline": "GBDT Slim",
            "challenger": "GBDT Slim+Events",
            "method_note": "树模型可学到「FOMC 前 3 天 + VIX 高 → 特别危险」类组合。",
        },
    ]

    model_experiments.extend(step2_exps)
    model_fi.update(step2_fi)
    model_practical.update(step2_prac)
    model_pairwise.extend(step2_pairwise)

    # =========================================================
    # Step 3: Regime-as-Context experiments
    # =========================================================
    print("\n[4/6] Step 3: Regime-as-context experiments...")
    regime_ctx_exps = []
    regime_ctx_fi = {}
    regime_ctx_prac = {}

    print("  Fetching regime data...")
    regime_df = fetch_regime_data()
    regime_labels = detect_regime(df, regime_df)
    regime_aligned = regime_labels.reindex(X_slim.index).fillna('normal')
    print(f"  Regime distribution: tight={int((regime_labels=='tight').sum())}, "
          f"normal={int((regime_labels=='normal').sum())}")

    # --- Scheme A: Regime-Conditional Models ---
    print("  Scheme A: Regime-Conditional models...")
    X_cond, y_cond = X_slim.copy(), y_slim.copy()
    X_train_cond, X_test_cond, y_train_cond, y_test_cond, train_end_cond, test_start_cond = \
        split_with_embargo(X_cond, y_cond)
    regime_train = regime_aligned.iloc[:train_end_cond]
    regime_test = regime_aligned.iloc[test_start_cond:]

    probs_cond_test = np.zeros(len(X_test_cond))
    probs_cond_all = np.zeros(len(X_cond))
    fallback_model, fallback_scaler = None, None
    for reg in ['tight', 'normal']:
        mask_train = regime_train == reg
        mask_test = regime_test == reg
        if mask_train.sum() < 10 or mask_test.sum() < 5:
            print(f"    Regime '{reg}': too few samples (train={mask_train.sum()}, test={mask_test.sum()}), using full model")
            if fallback_model is None:
                fallback_model, fallback_scaler, _ = train_lr_no_balance(
                    X_train_cond, y_train_cond, X_test_cond)
            probs_cond_test[mask_test.values] = full_probs(
                fallback_model, fallback_scaler, X_test_cond)[mask_test.values]
            mask_all = regime_aligned == reg
            probs_cond_all[mask_all.values] = full_probs(
                fallback_model, fallback_scaler, X_cond[mask_all])
            continue
        Xtr, ytr = X_train_cond[mask_train], y_train_cond[mask_train]
        model_r, scaler_r, _ = train_lr_no_balance(Xtr, ytr, X_test_cond[mask_test])
        probs_cond_test[mask_test.values] = full_probs(model_r, scaler_r, X_test_cond[mask_test])
        mask_all = regime_aligned == reg
        probs_cond_all[mask_all.values] = full_probs(model_r, scaler_r, X_cond[mask_all])

    test_auc_cond = auc(*roc_curve(y_test_cond, probs_cond_test)[:2])
    result_cond = build_comparison_metrics(
        y_test_cond, probs_cond_test, probs_cond_all, X_cond, sp500,
        "LR Regime-Conditional", KEY_EVENTS)
    prac_cond = compute_practical_metrics(y_test_cond, probs_cond_test)
    result_cond['practical_metrics'] = prac_cond
    regime_ctx_exps.append(result_cond)
    regime_ctx_prac["LR Regime-Conditional"] = prac_cond
    print(f"    AUC={test_auc_cond:.3f}  F1={prac_cond['best_f1']:.3f}  Brier={prac_cond['brier_score']:.4f}")

    # --- Scheme B: Interaction Features ---
    print("  Scheme B: Interaction features...")
    X_interact = X_slim.copy()
    regime_binary = (regime_aligned.reindex(X_interact.index) == 'tight').astype(float)
    key_feats = ['vix_level', 'credit_spread_10d_chg', 'sp500_vs_50ma']
    created = []
    for f in key_feats:
        if f in X_interact.columns:
            col_name = f'tight_x_{f}'
            X_interact[col_name] = regime_binary * X_interact[f]
            created.append(col_name)
    print(f"    Created interaction features: {created} ({len(created)}/{len(key_feats)})")

    y_interact = y_slim.reindex(X_interact.index)
    combined_int = X_interact.copy()
    combined_int['target'] = y_interact
    combined_int = combined_int.dropna()
    combined_int = percentile_clip(combined_int)
    X_int = combined_int.drop('target', axis=1)
    y_int = combined_int['target']

    run_and_log("LR Slim+Interact", X_int, y_int, train_lr_no_balance,
                regime_ctx_exps, regime_ctx_fi, regime_ctx_prac)
    run_and_log("GBDT Slim+Interact", X_int, y_int, train_gbdt_no_balance,
                regime_ctx_exps, regime_ctx_fi, regime_ctx_prac)

    # --- Scheme C: Post-hoc Regime Calibration ---
    print("  Scheme C: Post-hoc regime calibration...")
    X_cal, y_cal = X_slim.copy(), y_slim.copy()
    regime_cal = regime_aligned.reindex(X_cal.index).fillna('normal')
    X_train_cal, X_test_cal, y_train_cal, y_test_cal, train_end_cal, test_start_cal = \
        split_with_embargo(X_cal, y_cal)
    regime_train_cal = regime_cal.iloc[:train_end_cal]
    regime_test_cal = regime_cal.iloc[test_start_cal:]

    model_base, scaler_base, probs_base_test = train_lr_no_balance(
        X_train_cal, y_train_cal, X_test_cal)
    probs_base_all = full_probs(model_base, scaler_base, X_cal)

    adjustment = {}
    for reg in ['tight', 'normal']:
        mask = regime_train_cal == reg
        if mask.sum() < 20:
            adjustment[reg] = 1.0
            continue
        actual_rate = float(y_train_cal[mask].mean())
        X_reg = X_train_cal[mask]
        y_reg = y_train_cal[mask]
        n_cal = int(len(X_reg) * 0.8)
        X_cal_train, X_cal_val = X_reg.iloc[:n_cal], X_reg.iloc[n_cal:]
        y_cal_train = y_reg.iloc[:n_cal]
        if y_cal_train.nunique() < 2:
            adjustment[reg] = 1.0
            continue
        scaler_oof = StandardScaler()
        X_ct = scaler_oof.fit_transform(X_cal_train)
        X_cv = scaler_oof.transform(X_cal_val)
        lr_oof = LogisticRegression(C=0.1, max_iter=1000)
        lr_oof.fit(X_ct, y_cal_train)
        pred_mean = float(lr_oof.predict_proba(X_cv)[:, 1].mean())
        val_actual = float(y_reg.iloc[n_cal:].mean())
        adjustment[reg] = val_actual / pred_mean if pred_mean > 0 else 1.0
    print(f"    Calibration adjustments (OOF): {adjustment}")

    probs_posthoc_test = probs_base_test.copy()
    probs_posthoc_all = probs_base_all.copy()
    for reg in ['tight', 'normal']:
        mask_test_r = (regime_test_cal == reg).values
        mask_all_r = (regime_cal == reg).values
        probs_posthoc_test[mask_test_r] = np.clip(
            probs_base_test[mask_test_r] * adjustment[reg], 0, 1)
        probs_posthoc_all[mask_all_r] = np.clip(
            probs_base_all[mask_all_r] * adjustment[reg], 0, 1)

    test_auc_posthoc = auc(*roc_curve(y_test_cal, probs_posthoc_test)[:2])
    result_posthoc = build_comparison_metrics(
        y_test_cal, probs_posthoc_test, probs_posthoc_all, X_cal, sp500,
        "LR Regime-PostCal", KEY_EVENTS)
    prac_posthoc = compute_practical_metrics(y_test_cal, probs_posthoc_test)
    result_posthoc['practical_metrics'] = prac_posthoc
    regime_ctx_exps.append(result_posthoc)
    regime_ctx_prac["LR Regime-PostCal"] = prac_posthoc
    print(f"    AUC={test_auc_posthoc:.3f}  F1={prac_posthoc['best_f1']:.3f}  Brier={prac_posthoc['brier_score']:.4f}")

    regime_ctx_pairwise = [
        {
            "id": "regime_conditional",
            "label": "Step 3a: Regime 条件建模",
            "variable": "LR Slim（单模型）vs LR Regime-Conditional（分 regime 建模）",
            "baseline": "LR Slim",
            "challenger": "LR Regime-Conditional",
            "method_note": "分别在 tight（加息/倒挂）和 normal 两个 regime 训练独立 LR，预测时根据当前 regime 切换模型。",
        },
        {
            "id": "regime_interact_lr",
            "label": "Step 3b: Regime 交互项（LR）",
            "variable": "LR Slim vs LR Slim+Interact（加 tight×vix_level 等交互）",
            "baseline": "LR Slim",
            "challenger": "LR Slim+Interact",
            "method_note": "不加 regime 作为独立特征，而是加 regime×核心特征 的交互项，让模型学到条件效应。",
        },
        {
            "id": "regime_interact_gbdt",
            "label": "Step 3c: Regime 交互项（GBDT）",
            "variable": "GBDT Slim vs GBDT Slim+Interact",
            "baseline": "GBDT Slim",
            "challenger": "GBDT Slim+Interact",
            "method_note": "GBDT 天然学交互，显式加入 regime 交互项是否仍有增量？",
        },
        {
            "id": "regime_postcal",
            "label": "Step 3d: Regime 后处理校准",
            "variable": "LR Slim vs LR Regime-PostCal",
            "baseline": "LR Slim",
            "challenger": "LR Regime-PostCal",
            "method_note": "训练全局模型，根据各 regime 的实际崩盘率/OOF预测的比值来调整输出概率。",
        },
    ]

    model_experiments.extend(regime_ctx_exps)
    model_fi.update(regime_ctx_fi)
    model_practical.update(regime_ctx_prac)
    model_pairwise.extend(regime_ctx_pairwise)

    # =========================================================
    # Step 4: Long-term retest (2005+)
    # =========================================================
    print("\n[5/6] Step 4: Long-term retest (2005+)...")
    try:
        df_ext = load_extended_data()
        sp500_ext = df_ext['sp500']
        print(f"  Extended data: {len(df_ext)} days ({df_ext.index[0]} ~ {df_ext.index[-1]})")

        features_slim_ext = build_features_slim(df_ext)
        target_ext = compute_target(sp500_ext)
        combined_ext = features_slim_ext.copy()
        combined_ext['target'] = target_ext
        combined_ext = combined_ext.dropna()
        combined_ext = percentile_clip(combined_ext)
        X_ext = combined_ext.drop('target', axis=1)
        y_ext = combined_ext['target']
        print(f"  Extended Slim: {len(X_ext)} samples, {X_ext.shape[1]} features, {int(y_ext.sum())} positive ({y_ext.mean()*100:.1f}%)")

        step4_exps = []
        step4_fi = {}
        step4_prac = {}

        def run_ext(name, X, y, train_fn):
            print(f"  Training {name}...")
            result, test_auc, imp, _, practical = run_experiment(
                name, X, y, sp500_ext, train_fn, events=EXTENDED_KEY_EVENTS,
            )
            step4_exps.append(result)
            step4_prac[name] = practical
            if imp:
                step4_fi[name] = imp
            print(f"    AUC={test_auc:.3f}  F1={practical['best_f1']:.3f}  "
                  f"Brier={practical['brier_score']:.4f}")

        run_ext("LR Ext", X_ext, y_ext, train_lr_no_balance)
        run_ext("GBDT Ext", X_ext, y_ext, train_gbdt_no_balance)
        run_ext("RF Ext", X_ext, y_ext, train_rf_no_balance)

        step4_pairwise = [
            {
                "id": "longterm_lr",
                "label": "Step 4a: LR 短期 vs 长期",
                "variable": "953 样本 vs {0} 样本 — LR Slim".format(len(X_ext)),
                "baseline": "LR Slim",
                "challenger": "LR Ext",
                "method_note": "相同模型和特征，仅扩展数据量。长期数据覆盖 2005+ 多个经济周期（次贷、欧债、COVID）。",
            },
            {
                "id": "longterm_gbdt",
                "label": "Step 4b: GBDT 短期 vs 长期",
                "variable": "953 样本 vs {0} 样本 — GBDT Slim".format(len(X_ext)),
                "baseline": "GBDT Slim",
                "challenger": "GBDT Ext",
                "method_note": "GBDT 能否在多周期数据上学到更好的非线性模式？",
            },
            {
                "id": "longterm_model_compare",
                "label": "Step 4c: 长期数据上 LR vs GBDT",
                "variable": "LR vs GBDT — 长期 {0} 样本".format(len(X_ext)),
                "baseline": "LR Ext",
                "challenger": "GBDT Ext",
                "method_note": "在跨越多个经济周期的长期数据上，非线性模型是否终于超过线性模型？",
            },
        ]

        model_experiments.extend(step4_exps)
        model_fi.update(step4_fi)
        model_practical.update(step4_prac)
        model_pairwise.extend(step4_pairwise)
        print("  Step 4 complete.")
    except Exception as e:
        print(f"  Step 4 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================
    # Board B: 指标探索 (Ch.2.1)
    # =========================================================
    print("\n[6/6] Ch.2.1 Metric exploration experiments...")
    metric_experiments = []
    metric_fi = {}
    metric_practical = {}

    run_and_log("LR Balanced", X_slim, y_slim, train_lr, metric_experiments, metric_fi, metric_practical)
    run_and_log("LR Unbalanced", X_slim, y_slim, train_lr_no_balance, metric_experiments, metric_fi, metric_practical)
    run_and_log("GBDT Balanced", X_full, y_full, train_gbdt, metric_experiments, metric_fi, metric_practical)
    run_and_log("GBDT Unbalanced", X_full, y_full, train_gbdt_no_balance, metric_experiments, metric_fi, metric_practical)
    run_and_log("GBDT Calibrated", X_full, y_full, train_gbdt_calibrated, metric_experiments, metric_fi, metric_practical)
    run_and_log("LR Full Balanced", X_full, y_full, train_lr, metric_experiments, metric_fi, metric_practical)
    run_and_log("LR Full Unbalanced", X_full, y_full, train_lr_no_balance, metric_experiments, metric_fi, metric_practical)

    metric_pairwise = [
        {
            "id": "lr_balanced",
            "label": "Exp A: class_weight 的影响（LR Slim）",
            "variable": "Balanced vs Unbalanced — LR",
            "baseline": "LR Balanced",
            "challenger": "LR Unbalanced",
            "method_note": "class_weight=balanced 上调正样本权重 → 模型认为大跌很常见 → 输出概率整体偏高 → 50% 阈值下大量误报。",
        },
        {
            "id": "gbdt_balanced",
            "label": "Exp B: sample_weight 的影响（GBDT Full）",
            "variable": "Balanced vs Unbalanced — GBDT",
            "baseline": "GBDT Balanced",
            "challenger": "GBDT Unbalanced",
            "method_note": "GBDT 使用 sample_weight 上调正样本（等价于 balanced）。去掉后观察概率校准和 F1 变化。",
        },
        {
            "id": "gbdt_calibration",
            "label": "Exp C: Isotonic 概率校准",
            "variable": "GBDT Balanced vs GBDT Calibrated",
            "baseline": "GBDT Balanced",
            "challenger": "GBDT Calibrated",
            "method_note": "Isotonic calibration 通过 3-fold CV 将 GBDT 的原始概率映射到真实频率。Brier 越低 = 概率越准。",
        },
        {
            "id": "calibration_vs_unbalance",
            "label": "Exp D: 校准 vs 去 balanced",
            "variable": "GBDT Calibrated vs GBDT Unbalanced",
            "baseline": "GBDT Calibrated",
            "challenger": "GBDT Unbalanced",
            "method_note": "后处理校准（isotonic）vs 训练端修正（去掉 sample_weight），哪种更有效？",
        },
    ]

    # =========================================================
    # Save both JSONs
    # =========================================================
    print("\nSaving results...")

    # Correlation analysis
    print("  Computing feature correlation analysis...")
    corr_matrix = X_full.corr(method='spearman')
    high_corr_pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) >= 0.5:
                high_corr_pairs.append({
                    "feat_a": cols[i], "feat_b": cols[j],
                    "spearman": round(float(r), 3),
                })
    high_corr_pairs.sort(key=lambda x: abs(x['spearman']), reverse=True)

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_data = []
        X_vif = X_slim.copy()
        for i, col in enumerate(X_vif.columns):
            vif_val = variance_inflation_factor(X_vif.values, i)
            vif_data.append({"feature": col, "vif": round(float(vif_val), 2)})
        vif_data.sort(key=lambda x: x['vif'], reverse=True)
    except Exception:
        vif_data = []

    redundancy_groups = [
        {"group": "VIX 系", "features": ["vix_level", "vix_z", "vix_roc"], "representative": "vix_z"},
        {"group": "利差系", "features": ["credit_spread_z", "credit_spread_roc", "term_spread_z"], "representative": "credit_spread_z"},
        {"group": "市场内部", "features": ["market_breadth", "sp500_above_200d"], "representative": "market_breadth"},
        {"group": "波动率", "features": ["sp500_vol_20d", "sp500_vol_ratio"], "representative": "sp500_vol_20d"},
    ]

    correlation_analysis = {
        "high_corr_pairs": high_corr_pairs,
        "vif": vif_data,
        "redundancy_groups": redundancy_groups,
        "total_features": len(cols),
        "slim_features": list(X_slim.columns),
        "full_features": cols,
    }

    def build_summary(prac_dict):
        best_f1 = max(prac_dict.items(), key=lambda x: x[1]['best_f1'])
        best_brier = min(prac_dict.items(), key=lambda x: x[1]['brier_score'])
        best_lift = max(prac_dict.items(), key=lambda x: x[1]['lift_at_80'])
        return {
            "best_f1_model": best_f1[0], "best_f1": best_f1[1]['best_f1'],
            "best_brier_model": best_brier[0], "best_brier": best_brier[1]['brier_score'],
            "best_lift_model": best_lift[0], "best_lift": best_lift[1]['lift_at_80'],
        }

    phase3_models = {
        "phase": 3,
        "title": "Non-linear Model Exploration",
        "experiments": model_experiments,
        "pairwise": model_pairwise,
        "feature_importances": model_fi,
        "practical_summary": build_summary(model_practical),
        "correlation_analysis": correlation_analysis,
        "summary": {
            "lr_slim_auc": round(lr_slim_auc, 3),
            "lr_full_auc": round(lr_full_auc, 3),
            "gbdt_slim_auc": round(gbdt_slim_auc, 3),
            "gbdt_full_auc": round(gbdt_full_auc, 3),
            "rf_slim_auc": round(rf_slim_auc, 3),
            "rf_full_auc": round(rf_full_auc, 3),
            "best_model": max(model_practical.items(), key=lambda x: x[1]['best_f1'])[0],
            "data_range": f"{df.index[0]} ~ {df.index[-1]}",
            "total_samples": len(X_slim),
        },
    }
    p1 = DATA_DIR / 'phase3_metrics.json'
    with open(p1, 'w') as f:
        json.dump(phase3_models, f)
    print(f"  Saved {p1}")

    metric_data = {
        "title": "Metric Exploration — 评估指标优化",
        "experiments": metric_experiments,
        "pairwise": metric_pairwise,
        "practical_summary": build_summary(metric_practical),
        "summary": {
            "data_range": f"{df.index[0]} ~ {df.index[-1]}",
            "total_samples": len(X_slim),
            "base_rate": round(float(y_slim.mean()), 4),
        },
    }
    p2 = DATA_DIR / 'metric_exploration.json'
    with open(p2, 'w') as f:
        json.dump(metric_data, f)
    print(f"  Saved {p2}")

    all_practical = {**model_practical, **metric_practical}
    print("\n" + "=" * 60)
    print("COMBINED SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<25} {'BestF1':>7} {'Brier':>7} {'Mean_p':>7}")
    print(f"  {'-'*50}")
    for name, pm in all_practical.items():
        print(f"  {name:<25} {pm['best_f1']:>6.3f} {pm['brier_score']:>7.4f} {pm['mean_prob']:>7.3f}")

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
