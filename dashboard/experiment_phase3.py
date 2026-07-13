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
    build_features_regime, build_features_regime_minimal, fetch_regime_data,
    build_features_with_events, build_features_kitchen_sink,
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
    print("\n[1/5] Loading data...")
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

    # Regime features
    print("  Fetching regime data (Fed Funds + CPI)...")
    regime_df = fetch_regime_data()
    features_regime = build_features_regime(df, regime_df)
    X_regime, y_regime = prep(features_regime)
    print(f"  Regime features: {len(X_regime)} samples, {X_regime.shape[1]} features, {int(y_regime.sum())} positive ({y_regime.mean()*100:.1f}%)")
    print(f"    New regime cols: {[c for c in X_regime.columns if c not in X_slim.columns]}")

    features_regime_min = build_features_regime_minimal(df, regime_df)
    X_regime_min, y_regime_min = prep(features_regime_min)
    print(f"  Regime-minimal: {len(X_regime_min)} samples, {X_regime_min.shape[1]} features")

    features_events = build_features_with_events(df)
    X_events, y_events = prep(features_events)
    print(f"  Event features: {len(X_events)} samples, {X_events.shape[1]} features")
    print(f"    New event cols: {[c for c in X_events.columns if c not in X_slim.columns]}")

    features_ks = build_features_kitchen_sink(df, regime_df)
    X_ks, y_ks = prep(features_ks)
    print(f"  Kitchen sink: {len(X_ks)} samples, {X_ks.shape[1]} features")

    # =========================================================
    # Board A: 非线性模型探索 (Ch.2)
    #   All models use unbalanced/no-sample-weight for fair comparison
    # =========================================================
    print("\n[2/5] Ch.2 Non-linear model experiments...")
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
    # Step 2: Regime features experiments (added to Ch.2)
    # =========================================================
    print("\n[3/5] Step 2: Regime feature experiments...")
    regime_experiments = []
    regime_fi = {}
    regime_practical = {}

    run_and_log("LR Slim+Regime9", X_regime, y_regime, train_lr_no_balance, regime_experiments, regime_fi, regime_practical)
    run_and_log("LR Slim+Regime2", X_regime_min, y_regime_min, train_lr_no_balance, regime_experiments, regime_fi, regime_practical)
    run_and_log("GBDT Slim+Regime9", X_regime, y_regime, train_gbdt_no_balance, regime_experiments, regime_fi, regime_practical)
    run_and_log("GBDT Slim+Regime2", X_regime_min, y_regime_min, train_gbdt_no_balance, regime_experiments, regime_fi, regime_practical)

    regime_pairwise = [
        {
            "id": "regime_lr_full",
            "label": "Step 2a: Regime 全量 9特征（LR）",
            "variable": "Slim 10特征 vs Slim+Regime 19特征",
            "baseline": "LR Slim",
            "challenger": "LR Slim+Regime9",
            "method_note": "加入 9 个 regime 特征（利率水平/方向、CPI、曲线状态）。如果 F1 下降 = regime 信息对当前短期数据是噪声。",
        },
        {
            "id": "regime_lr_min",
            "label": "Step 2b: Regime 精简 2特征（LR）",
            "variable": "Slim 10特征 vs Slim+2 (curve_inverted + fed_hiking)",
            "baseline": "LR Slim",
            "challenger": "LR Slim+Regime2",
            "method_note": "只加 curve_inverted 和 fed_hiking 两个二值特征。测试精简 regime 信号是否比全量更有效。",
        },
        {
            "id": "regime_gbdt",
            "label": "Step 2c: Regime 精简 2特征（GBDT）",
            "variable": "GBDT Slim vs GBDT Slim+Regime2",
            "baseline": "GBDT Slim",
            "challenger": "GBDT Slim+Regime2",
            "method_note": "树模型可以学到「加息中 + VIX 上涨 → 高危」类交互。测试 GBDT 能否更好利用 regime 信息。",
        },
    ]

    # =========================================================
    # Step 3: Event calendar features
    # =========================================================
    print("\n  Step 3: Event calendar experiments...")
    step3_exps = []
    step3_fi = {}
    step3_prac = {}

    run_and_log("LR Slim+Events", X_events, y_events, train_lr_no_balance, step3_exps, step3_fi, step3_prac)
    run_and_log("GBDT Slim+Events", X_events, y_events, train_gbdt_no_balance, step3_exps, step3_fi, step3_prac)
    run_and_log("LR KitchenSink", X_ks, y_ks, train_lr_no_balance, step3_exps, step3_fi, step3_prac)
    run_and_log("GBDT KitchenSink", X_ks, y_ks, train_gbdt_no_balance, step3_exps, step3_fi, step3_prac)

    step3_pairwise = [
        {
            "id": "events_lr",
            "label": "Step 3a: 事件日历增益（LR）",
            "variable": "Slim vs Slim+Events — LR",
            "baseline": "LR Slim",
            "challenger": "LR Slim+Events",
            "method_note": "加入 FOMC/CPI/NFP 前后天数和窗口标记。测试事件邻近性是否有预测价值。",
        },
        {
            "id": "events_gbdt",
            "label": "Step 3b: 事件日历增益（GBDT）",
            "variable": "GBDT Slim vs GBDT Slim+Events",
            "baseline": "GBDT Slim",
            "challenger": "GBDT Slim+Events",
            "method_note": "树模型可学到「FOMC 前 3 天 + VIX 高 → 特别危险」类组合。",
        },
        {
            "id": "kitchen_sink",
            "label": "Step 3c: Kitchen Sink（全特征）",
            "variable": "LR Slim vs LR KitchenSink",
            "baseline": "LR Slim",
            "challenger": "LR KitchenSink",
            "method_note": "Slim + Regime2 + Events 全部组合，测试特征叠加的综合效果。",
        },
    ]

    # Merge all step 2+3 experiments into Ch.2
    model_experiments.extend(regime_experiments)
    model_experiments.extend(step3_exps)
    model_fi.update(regime_fi)
    model_fi.update(step3_fi)
    model_practical.update(regime_practical)
    model_practical.update(step3_prac)
    model_pairwise.extend(regime_pairwise)
    model_pairwise.extend(step3_pairwise)

    # =========================================================
    # Step 4: Long-term retest (2005+)
    # =========================================================
    print("\n[4/6] Step 4: Long-term retest (2005+)...")
    try:
        df_ext = load_extended_data()
        sp500_ext = df_ext['sp500']
        print(f"  Extended data: {len(df_ext)} days ({df_ext.index[0]} ~ {df_ext.index[-1]})")

        features_slim_ext = build_features_slim(df_ext)
        target_ext = compute_target(sp500_ext)
        combined_ext = features_slim_ext.copy()
        combined_ext['target'] = target_ext
        combined_ext = combined_ext.dropna().clip(-10, 10)
        X_ext = combined_ext.drop('target', axis=1)
        y_ext = combined_ext['target']
        print(f"  Extended Slim: {len(X_ext)} samples, {X_ext.shape[1]} features, {int(y_ext.sum())} positive ({y_ext.mean()*100:.1f}%)")

        step4_exps = []
        step4_fi = {}
        step4_prac = {}

        run_and_log("LR Ext", X_ext, y_ext, train_lr_no_balance, step4_exps, step4_fi, step4_prac)
        run_and_log("GBDT Ext", X_ext, y_ext, train_gbdt_no_balance, step4_exps, step4_fi, step4_prac)
        run_and_log("RF Ext", X_ext, y_ext, train_rf_no_balance, step4_exps, step4_fi, step4_prac)

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
                "method_note": "GBDT 在更多数据上是否能学到跨周期的 regime 切换规律？",
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
    #   Balanced vs Unbalanced, Calibration experiments
    # =========================================================
    print("\n[5/6] Ch.2.1 Metric exploration experiments...")
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
    print("\n[6/6] Saving results...")

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
