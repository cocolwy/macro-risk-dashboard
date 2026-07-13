"""
Phase 3: Model Evolution — Breaking the Regime Barrier

Step 1: XGBoost vs Logistic Regression on existing features
Step 2: Regime features (FOMC rate direction, CPI trend, yield curve state)
Step 3: Event calendar features (FOMC/CPI/NFP proximity)
Step 4: Long-term (1986+) retest with non-linear models

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
    compute_target, KEY_EVENTS, build_comparison_metrics,
)

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


def main():
    print("=" * 60)
    print("PHASE 3: Model Evolution")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/3] Loading data...")
    df = load_indicators()
    sp500 = df['sp500']
    print(f"  {len(df)} trading days ({df.index[0]} ~ {df.index[-1]})")

    features_full = build_features(df)
    features_slim = build_features_slim(df)
    target = compute_target(sp500)

    def prep(features):
        combined = features.copy()
        combined['target'] = target
        combined = combined.dropna().clip(-10, 10)
        X = combined.drop('target', axis=1)
        y = combined['target']
        return X, y

    X_full, y_full = prep(features_full)
    X_slim, y_slim = prep(features_slim)
    print(f"  Full features: {len(X_full)} samples, {X_full.shape[1]} features, {int(y_full.sum())} positive ({y_full.mean()*100:.1f}%)")
    print(f"  Slim features: {len(X_slim)} samples, {X_slim.shape[1]} features, {int(y_slim.sum())} positive ({y_slim.mean()*100:.1f}%)")

    # --- Step 1: Model comparison with practical metrics ---
    print("\n[2/3] Step 1 experiments...")
    experiments = []
    feature_importances = {}
    all_practical = {}

    def run_and_log(name, X, y, train_fn):
        print(f"  Training {name}...")
        result, test_auc, imp, _, practical = run_experiment(name, X, y, sp500, train_fn)
        experiments.append(result)
        all_practical[name] = practical
        if imp:
            feature_importances[name] = imp
        pm = practical
        print(f"    AUC={test_auc:.3f}  P@80%={pm['p_at_80']:.1%}  Lift={pm['lift_at_80']}x  BestF1={pm['best_f1']:.3f}@{pm['best_f1_threshold']:.0%}  Brier={pm['brier_score']:.4f}  Mean_prob={pm['mean_prob']:.3f}")
        return test_auc

    lr_auc = run_and_log("LR Balanced", X_slim, y_slim, train_lr)
    run_and_log("LR Unbalanced", X_slim, y_slim, train_lr_no_balance)
    gbdt_slim_auc = run_and_log("GBDT Slim", X_slim, y_slim, train_gbdt)
    gbdt_full_auc = run_and_log("GBDT Full (23feat)", X_full, y_full, train_gbdt)
    run_and_log("GBDT Calibrated", X_full, y_full, train_gbdt_calibrated)
    rf_slim_auc = run_and_log("RF Slim", X_slim, y_slim, train_rf)
    lr_full_auc = run_and_log("LR Full (23feat)", X_full, y_full, train_lr)
    run_and_log("LR Full Unbalanced", X_full, y_full, train_lr_no_balance)

    # --- Build pairwise comparisons metadata ---
    pairwise = [
        {
            "id": "step1a",
            "label": "Step 1a: class_weight 的影响",
            "variable": "Balanced vs Unbalanced — LR Slim",
            "baseline": "LR Balanced",
            "challenger": "LR Unbalanced",
            "method_note": "class_weight=balanced 会上调正样本权重，导致模型输出概率偏高（mean_prob 远超 base_rate）。不平衡训练产生更接近真实概率的输出。",
        },
        {
            "id": "step1b",
            "label": "Step 1b: GBDT 概率校准",
            "variable": "GBDT Raw vs GBDT Calibrated（isotonic）",
            "baseline": "GBDT Full (23feat)",
            "challenger": "GBDT Calibrated",
            "method_note": "Isotonic calibration 通过 3-fold CV 将 GBDT 的原始概率映射到真实频率。Brier score 越低 = 概率越准。关注 P@80% 和 mean_prob 的变化。",
        },
        {
            "id": "step1c",
            "label": "Step 1c: 线性 vs 非线性 (Full特征)",
            "variable": "LR Full Unbalanced vs GBDT Calibrated",
            "baseline": "LR Full Unbalanced",
            "challenger": "GBDT Calibrated",
            "method_note": "两者都使用校准/不平衡策略输出合理概率。纯粹比较模型能力：线性 vs 非线性。",
        },
    ]

    # --- Save results ---
    print("\n[3/3] Saving results...")

    best_by_f1 = max(all_practical.items(), key=lambda x: x[1]['best_f1'])
    best_by_brier = min(all_practical.items(), key=lambda x: x[1]['brier_score'])
    best_by_lift = max(all_practical.items(), key=lambda x: x[1]['lift_at_80'])

    phase3_data = {
        "phase": 3,
        "title": "Model Evolution — 优化目标重定义",
        "experiments": experiments,
        "pairwise": pairwise,
        "feature_importances": feature_importances,
        "practical_summary": {
            "best_f1_model": best_by_f1[0],
            "best_f1": best_by_f1[1]['best_f1'],
            "best_brier_model": best_by_brier[0],
            "best_brier": best_by_brier[1]['brier_score'],
            "best_lift_model": best_by_lift[0],
            "best_lift": best_by_lift[1]['lift_at_80'],
        },
        "summary": {
            "lr_slim_auc": round(lr_auc, 3),
            "gbdt_slim_auc": round(gbdt_slim_auc, 3),
            "gbdt_full_auc": round(gbdt_full_auc, 3),
            "rf_slim_auc": round(rf_slim_auc, 3),
            "lr_full_auc": round(lr_full_auc, 3),
            "best_model": best_by_f1[0],
            "data_range": f"{df.index[0]} ~ {df.index[-1]}",
            "total_samples": len(X_slim),
        },
    }

    out_path = DATA_DIR / 'phase3_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(phase3_data, f)
    print(f"  Saved {out_path}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("PRACTICAL METRICS SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<25} {'BestF1':>7} {'P@80%':>6} {'Lift':>5} {'Brier':>7} {'Mean_p':>7}")
    print(f"  {'-'*65}")
    for name, pm in all_practical.items():
        print(f"  {name:<25} {pm['best_f1']:>6.3f} {pm['p_at_80']:>5.1%} {pm['lift_at_80']:>5.1f}x {pm['brier_score']:>7.4f} {pm['mean_prob']:>7.3f}")

    print(f"\n  Best F1: {best_by_f1[0]} ({best_by_f1[1]['best_f1']:.3f} @ {best_by_f1[1]['best_f1_threshold']:.0%})")
    print(f"  Best Calibration: {best_by_brier[0]} (Brier={best_by_brier[1]['brier_score']:.4f})")
    print(f"  Best Lift@80%: {best_by_lift[0]} ({best_by_lift[1]['lift_at_80']:.1f}x)")

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
