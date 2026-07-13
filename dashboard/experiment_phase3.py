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
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

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


def run_experiment(name, X, y, sp500, train_fn, events=KEY_EVENTS):
    X_train, X_test, y_train, y_test, _, test_start = split_with_embargo(X, y)
    model, scaler, probs_test = train_fn(X_train, y_train, X_test)
    probs_all = full_probs(model, scaler, X)
    test_auc = auc(*roc_curve(y_test, probs_test)[:2])

    result = build_comparison_metrics(y_test, probs_test, probs_all, X, sp500, name, events)

    feature_imp = None
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        feature_imp = [{"feature": k, "importance": round(float(v), 4)} for k, v in imp.items()]
    elif hasattr(model, 'coef_'):
        coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
        feature_imp = [{"feature": k, "importance": round(float(abs(v)), 4)} for k, v in coefs.items()]

    return result, test_auc, feature_imp, model


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

    # --- Step 1: XGBoost vs LR ---
    print("\n[2/3] Step 1 experiments: XGBoost vs LR...")
    experiments = []
    feature_importances = {}

    # 1a: LR baseline (slim, embargo) — same as D1 from Phase 2
    print("  Training LR Slim (baseline)...")
    lr_result, lr_auc, lr_imp, _ = run_experiment(
        "LR Slim (baseline)", X_slim, y_slim, sp500, train_lr)
    experiments.append(lr_result)
    if lr_imp:
        feature_importances["LR Slim"] = lr_imp
    print(f"    AUC: {lr_auc:.3f}")

    # 1b: GBDT slim — same features, non-linear model
    print("  Training GBDT Slim...")
    gbdt_slim_result, gbdt_slim_auc, gbdt_slim_imp, _ = run_experiment(
        "GBDT Slim", X_slim, y_slim, sp500, train_gbdt)
    experiments.append(gbdt_slim_result)
    if gbdt_slim_imp:
        feature_importances["GBDT Slim"] = gbdt_slim_imp
    print(f"    AUC: {gbdt_slim_auc:.3f}")

    # 1c: GBDT full features — can tree models handle 23 features better than LR?
    print("  Training GBDT Full (23 features)...")
    gbdt_full_result, gbdt_full_auc, gbdt_full_imp, _ = run_experiment(
        "GBDT Full (23feat)", X_full, y_full, sp500, train_gbdt)
    experiments.append(gbdt_full_result)
    if gbdt_full_imp:
        feature_importances["GBDT Full"] = gbdt_full_imp
    print(f"    AUC: {gbdt_full_auc:.3f}")

    # 1d2: Random Forest slim — another non-linear baseline
    print("  Training RF Slim...")
    rf_slim_result, rf_slim_auc, rf_slim_imp, _ = run_experiment(
        "RF Slim", X_slim, y_slim, sp500, train_rf)
    experiments.append(rf_slim_result)
    if rf_slim_imp:
        feature_importances["RF Slim"] = rf_slim_imp
    print(f"    AUC: {rf_slim_auc:.3f}")

    # 1d: LR full for comparison
    print("  Training LR Full (23 features)...")
    lr_full_result, lr_full_auc, lr_full_imp, _ = run_experiment(
        "LR Full (23feat)", X_full, y_full, sp500, train_lr)
    experiments.append(lr_full_result)
    if lr_full_imp:
        feature_importances["LR Full"] = lr_full_imp
    print(f"    AUC: {lr_full_auc:.3f}")

    # --- Build pairwise comparisons metadata ---
    pairwise = [
        {
            "id": "step1a",
            "label": "Step 1a: 模型类型 (Slim特征)",
            "variable": "Logistic Regression vs GBDT — 10个特征",
            "baseline": "LR Slim (baseline)",
            "challenger": "GBDT Slim",
            "method_note": "相同的 Slim 10特征 + Embargo 20d，仅改变模型类型。GBDT（梯度提升树）可隐式学到 regime 条件规律（如「VIX高 且 利差走阔→危机」），LR 只能学线性组合。",
        },
        {
            "id": "step1b",
            "label": "Step 1b: 特征数量 (GBDT)",
            "variable": "10特征 vs 23特征 — GBDT 模型",
            "baseline": "GBDT Slim",
            "challenger": "GBDT Full (23feat)",
            "method_note": "LR 中冗余特征有害（共线性），但树模型天然抗共线性。测试非线性模型是否能利用更多特征。",
        },
        {
            "id": "step1c",
            "label": "Step 1c: 非线性模型对比",
            "variable": "GBDT vs Random Forest — Slim特征",
            "baseline": "GBDT Slim",
            "challenger": "RF Slim",
            "method_note": "两种主流树模型对比。GBDT 逐步纠错（boosting），RF 并行投票（bagging），各有擅长的场景。",
        },
    ]

    # --- Save results ---
    print("\n[3/3] Saving results...")
    phase3_data = {
        "phase": 3,
        "title": "Model Evolution — Breaking the Regime Barrier",
        "experiments": experiments,
        "pairwise": pairwise,
        "feature_importances": feature_importances,
        "summary": {
            "lr_slim_auc": round(lr_auc, 3),
            "gbdt_slim_auc": round(gbdt_slim_auc, 3),
            "gbdt_full_auc": round(gbdt_full_auc, 3),
            "rf_slim_auc": round(rf_slim_auc, 3),
            "lr_full_auc": round(lr_full_auc, 3),
            "best_model": max(
                [("LR Slim", lr_auc), ("GBDT Slim", gbdt_slim_auc),
                 ("GBDT Full", gbdt_full_auc), ("RF Slim", rf_slim_auc),
                 ("LR Full", lr_full_auc)],
                key=lambda x: x[1]
            )[0],
            "data_range": f"{df.index[0]} ~ {df.index[-1]}",
            "total_samples": len(X_slim),
        },
    }

    out_path = DATA_DIR / 'phase3_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(phase3_data, f)
    print(f"  Saved {out_path}")

    # --- Print summary ---
    all_aucs = [lr_auc, gbdt_slim_auc, gbdt_full_auc, rf_slim_auc, lr_full_auc]
    best_auc = max(all_aucs)

    print("\n" + "=" * 60)
    print("STEP 1 RESULTS: Non-linear vs Linear")
    print("=" * 60)
    print(f"  {'Model':<25} {'AUC':>6}  {'Features':>8}")
    print(f"  {'-'*45}")
    for name, a, nf in [
        ("LR Slim", lr_auc, X_slim.shape[1]),
        ("GBDT Slim", gbdt_slim_auc, X_slim.shape[1]),
        ("RF Slim", rf_slim_auc, X_slim.shape[1]),
        ("LR Full", lr_full_auc, X_full.shape[1]),
        ("GBDT Full", gbdt_full_auc, X_full.shape[1]),
    ]:
        marker = " ★" if a == best_auc else ""
        print(f"  {name:<25} {a:>6.3f}  {nf:>8}{marker}")

    print("\n  Feature importance (GBDT Slim, top 5):")
    if gbdt_slim_imp:
        for item in gbdt_slim_imp[:5]:
            print(f"    {item['feature']:<30} {item['importance']:.4f}")

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
