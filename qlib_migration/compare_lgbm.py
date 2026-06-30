"""
Week4 Step 3 — LightGBM vs 线性, 对比 IC / Sharpe / MDD + 特征重要性。

- IC: 每月 Spearman(score, fwd_ret), 报告全 OOS 和回测子区间 (2022-04+)。
- 回测: Top20 等权月度调仓, 复用已验证 Qlib 引擎 (qlib_backtest_lib)。
- 参照: 锁定 baseline (全样本 PCA 线性, Qlib gross 30.04% / net 28.69%)。
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qlib_backtest_lib import run_backtest

ROOT = Path(__file__).resolve().parent.parent
QM = ROOT / "qlib_migration"
BT_START = pd.Timestamp("2022-03-01")   # 对齐 baseline 回测区间
PCS = [f"PC{i+1}" for i in range(5)]


def monthly_ic(scores, fwd, months=None):
    rows = []
    df = pd.concat([scores.rename("s"), fwd.rename("y")], axis=1).dropna()
    for m, g in df.groupby(level=0):
        if months is not None and m not in months:
            continue
        if len(g) >= 10:
            rows.append(g["s"].corr(g["y"], method="spearman"))
    ic = pd.Series(rows)
    return {"mean_ic": round(float(ic.mean()), 4),
            "ic_std": round(float(ic.std()), 4),
            "icir": round(float(ic.mean() / ic.std()), 3) if ic.std() > 0 else 0.0,
            "n_months": len(ic), "pct_positive": round(float((ic > 0).mean()), 3)}


def top20_holdings(scores, min_date):
    holds = {}
    for m, g in scores.groupby(level=0):
        if m < min_date:
            continue
        top = g.droplevel(0).sort_values(ascending=False).head(20).index.tolist()
        holds[m] = top
    return holds


def main():
    oos = pd.read_parquet(QM / "oos_scores.parquet")
    fwd = oos["fwd_ret"]
    bt_months = [m for m in oos.index.get_level_values(0).unique() if m >= BT_START]

    # ── IC ──
    print("=== IC (full OOS) ===")
    ic_full = {k: monthly_ic(oos[f"{k}_score"], fwd) for k in ("lgbm", "linear")}
    for k, v in ic_full.items():
        print(f"  {k:7} mean_IC={v['mean_ic']:+.4f}  ICIR={v['icir']:+.3f}  +months={v['pct_positive']:.0%}  n={v['n_months']}")
    print("=== IC (backtest sub-period 2022-04+) ===")
    ic_bt = {k: monthly_ic(oos[f"{k}_score"], fwd, months=set(bt_months)) for k in ("lgbm", "linear")}
    for k, v in ic_bt.items():
        print(f"  {k:7} mean_IC={v['mean_ic']:+.4f}  ICIR={v['icir']:+.3f}  +months={v['pct_positive']:.0%}  n={v['n_months']}")

    # ── 回测 (same engine) ──
    print("\n=== Qlib backtest (Top20 EW, 2022-04+) ===")
    res = {}
    for k in ("lgbm", "linear"):
        holds = top20_holdings(oos[f"{k}_score"], BT_START)
        res[k] = run_backtest(holds, cost=0.001)
        g, n = res[k]["gross"], res[k]["net"]
        print(f"  WF-{k:7} gross={g['ann_return']:.4f} (Sharpe {g['sharpe']}, MDD {g['max_drawdown']}) | net={n['ann_return']:.4f}  [{res[k]['period']}]")

    base = json.loads((ROOT / "workspace" / "baseline_sp500" / "baseline.json").read_text())["performance"]
    qres = json.loads((QM / "qlib_result.json").read_text())
    print(f"  REF locked-baseline (full-sample PCA linear): gross {qres['qlib_gross']['ann_return']:.4f} "
          f"(Sharpe {qres['qlib_gross']['sharpe']}), net {qres['qlib_net_ann']:.4f}")

    # ── 特征重要性图 ──
    imp = pd.read_csv(QM / "feature_importance.csv", index_col=0).iloc[:, 0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp.reindex(PCS).plot.bar(ax=ax, color="steelblue", alpha=0.85)
    ax.set_title("LightGBM Feature Importance (avg gain across folds)")
    ax.set_ylabel("gain"); ax.set_xlabel("PCA component")
    plt.xticks(rotation=0); plt.tight_layout()
    fig.savefig(QM / "feature_importance.png", dpi=150)
    plt.close(fig)

    out = {"ic_full": ic_full, "ic_backtest": ic_bt,
           "backtest": {k: res[k] for k in res},
           "ref_locked_baseline": {"gross_ann": qres["qlib_gross"]["ann_return"],
                                   "gross_sharpe": qres["qlib_gross"]["sharpe"],
                                   "net_ann": qres["qlib_net_ann"]}}
    (QM / "compare_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[done] feature_importance.png + compare_result.json written")


if __name__ == "__main__":
    main()
