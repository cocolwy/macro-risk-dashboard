"""
Step 2d–2f: Regime features on extended history (2005+).

Short-sample Regime9 looked like noise. This run asks whether multi-cycle
data makes curve/Fed regime flags useful.

Merges results into existing phase3_metrics.json (does not re-run Steps 1–4a–c).
"""

import json
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from predict_model import (
    build_features_slim, build_features_regime, build_features_regime_minimal,
    fetch_regime_data, compute_target,
)
from experiment_extended_history import load_extended_data, EXTENDED_KEY_EVENTS
from experiment_phase3 import (
    train_lr_no_balance, train_gbdt_no_balance, run_experiment, DATA_DIR,
)

PUBLIC_JSON = Path(__file__).parent / 'frontend' / 'public' / 'data' / 'phase3_metrics.json'

NEW_NAMES = {'LR Ext+Regime9', 'LR Ext+Regime2', 'GBDT Ext+Regime2'}
NEW_PAIR_IDS = {'regime_ext_r9', 'regime_ext_r2', 'regime_ext_gbdt'}


def main():
    print("=" * 60)
    print("Step 2d–2f: Regime × Extended History (2005+)")
    print("=" * 60)

    print("\n[1/3] Loading extended data + regime series...")
    df_ext = load_extended_data()
    sp500_ext = df_ext['sp500']
    print(f"  Extended: {len(df_ext)} days ({df_ext.index[0]} ~ {df_ext.index[-1]})")

    regime_df = fetch_regime_data(start='2005-01-01')
    target = compute_target(sp500_ext)

    def prep(features):
        combined = features.copy()
        combined['target'] = target
        combined = combined.dropna().clip(-10, 10)
        return combined.drop('target', axis=1), combined['target']

    X_slim, y_slim = prep(build_features_slim(df_ext))
    X_r9, y_r9 = prep(build_features_regime(df_ext, regime_df))
    X_r2, y_r2 = prep(build_features_regime_minimal(df_ext, regime_df))
    print(f"  Slim: {len(X_slim)} samples, {X_slim.shape[1]} feat")
    print(f"  Regime9: {len(X_r9)} samples, {X_r9.shape[1]} feat "
          f"(+{[c for c in X_r9.columns if c not in X_slim.columns]})")
    print(f"  Regime2: {len(X_r2)} samples, {X_r2.shape[1]} feat "
          f"(+{[c for c in X_r2.columns if c not in X_slim.columns]})")

    print("\n[2/3] Training...")
    new_exps, new_fi, new_prac = [], {}, {}

    def run(name, X, y, train_fn):
        result, test_auc, fi, _, practical = run_experiment(
            name, X, y, sp500_ext, train_fn, events=EXTENDED_KEY_EVENTS,
        )
        new_exps.append(result)
        new_prac[name] = practical
        if fi:
            new_fi[name] = fi
        print(f"  {name}: AUC={test_auc:.3f} F1={practical['best_f1']:.3f} "
              f"Brier={practical['brier_score']:.4f}")

    run("LR Ext+Regime9", X_r9, y_r9, train_lr_no_balance)
    run("LR Ext+Regime2", X_r2, y_r2, train_lr_no_balance)
    run("GBDT Ext+Regime2", X_r2, y_r2, train_gbdt_no_balance)

    new_pairwise = [
        {
            "id": "regime_ext_r9",
            "label": "Step 2d: Regime9 × 长期（LR）",
            "variable": "LR Ext Slim vs LR Ext+Regime9 — 2005+",
            "baseline": "LR Ext",
            "challenger": "LR Ext+Regime9",
            "method_note": "短期上 Regime9 是噪声（F1 0.588→0.208）。多周期后若 F1 上升，说明 regime 叙事成立，瓶颈是样本周期数；若仍下降，说明特征本身有害。",
        },
        {
            "id": "regime_ext_r2",
            "label": "Step 2e: Regime2 × 长期（LR）",
            "variable": "LR Ext Slim vs LR Ext+Regime2 — 2005+",
            "baseline": "LR Ext",
            "challenger": "LR Ext+Regime2",
            "method_note": "只加 curve_inverted + fed_hiking。短期已接近 baseline；长期多周期下是否终于带来增量？",
        },
        {
            "id": "regime_ext_gbdt",
            "label": "Step 2f: Regime2 × 长期（GBDT）",
            "variable": "GBDT Ext vs GBDT Ext+Regime2 — 2005+",
            "baseline": "GBDT Ext",
            "challenger": "GBDT Ext+Regime2",
            "method_note": "树模型可学「倒挂 × VIX 上涨 → 高危」交互。长期数据上 GBDT 能否利用 regime 条件？",
        },
    ]

    print("\n[3/3] Merging into phase3_metrics.json...")
    path = DATA_DIR / 'phase3_metrics.json'
    with open(path) as f:
        data = json.load(f)

    data['experiments'] = [e for e in data['experiments'] if e['name'] not in NEW_NAMES]
    data['experiments'].extend(new_exps)

    data['pairwise'] = [p for p in data.get('pairwise', []) if p.get('id') not in NEW_PAIR_IDS]
    data['pairwise'].extend(new_pairwise)

    fi = data.get('feature_importances', {})
    for k in list(fi.keys()):
        if k in NEW_NAMES:
            del fi[k]
    fi.update(new_fi)
    data['feature_importances'] = fi

    # Refresh practical_summary over all models still in the file
    prac_by_name = {
        e['name']: e['practical_metrics']
        for e in data['experiments']
        if e.get('practical_metrics')
    }
    if prac_by_name:
        best_f1 = max(prac_by_name.items(), key=lambda x: x[1]['best_f1'])
        best_brier = min(prac_by_name.items(), key=lambda x: x[1]['brier_score'])
        best_lift = max(prac_by_name.items(), key=lambda x: x[1]['lift_at_80'])
        data['practical_summary'] = {
            "best_f1_model": best_f1[0], "best_f1": best_f1[1]['best_f1'],
            "best_brier_model": best_brier[0], "best_brier": best_brier[1]['brier_score'],
            "best_lift_model": best_lift[0], "best_lift": best_lift[1]['lift_at_80'],
        }

    with open(path, 'w') as f:
        json.dump(data, f)
    shutil.copy2(path, PUBLIC_JSON)
    print(f"  Saved {path}")
    print(f"  Copied → {PUBLIC_JSON}")

    # Compare vs existing Ext baselines if present
    baselines = {e['name']: e.get('practical_metrics', {}) for e in data['experiments']}
    print("\n=== Comparison (F1 / Brier / AUC) ===")
    for name in ['LR Ext', 'LR Ext+Regime9', 'LR Ext+Regime2', 'GBDT Ext', 'GBDT Ext+Regime2']:
        e = next((x for x in data['experiments'] if x['name'] == name), None)
        if not e:
            print(f"  {name}: (missing)")
            continue
        pm = e.get('practical_metrics', {})
        print(f"  {name:<22} F1={pm.get('best_f1', 0):.3f}  "
              f"Brier={pm.get('brier_score', 0):.4f}  AUC={e.get('auc', 0):.3f}")

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
