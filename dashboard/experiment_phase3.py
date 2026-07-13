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

    # =========================================================
    # Board A: 非线性模型探索 (Ch.2)
    #   All models use unbalanced/no-sample-weight for fair comparison
    # =========================================================
    print("\n[2/4] Ch.2 Non-linear model experiments...")
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
    # Board B: 指标探索 (Ch.2.1)
    #   Balanced vs Unbalanced, Calibration experiments
    # =========================================================
    print("\n[3/4] Ch.2.1 Metric exploration experiments...")
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
    print("\n[4/4] Saving results...")

    def build_summary(prac_dict):
        best_f1 = max(prac_dict.items(), key=lambda x: x[1]['best_f1'])
        best_brier = min(prac_dict.items(), key=lambda x: x[1]['brier_score'])
        best_lift = max(prac_dict.items(), key=lambda x: x[1]['lift_at_80'])
        return {
            "best_f1_model": best_f1[0], "best_f1": best_f1[1]['best_f1'],
            "best_brier_model": best_brier[0], "best_brier": best_brier[1]['brier_score'],
            "best_lift_model": best_lift[0], "best_lift": best_lift[1]['lift_at_80'],
        }

    # Ch.2 Non-linear models
    phase3_models = {
        "phase": 3,
        "title": "Non-linear Model Exploration",
        "experiments": model_experiments,
        "pairwise": model_pairwise,
        "feature_importances": model_fi,
        "practical_summary": build_summary(model_practical),
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

    # Ch.2.1 Metric exploration
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

    # --- Print combined summary ---
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
