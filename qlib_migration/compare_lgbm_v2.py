"""
Week4 v2 Step 3 — Regression LightGBM vs LambdaRank vs 线性, 宽 universe 对比。

三种模型 × IC / Sharpe / MDD + 参照锁定 baseline。
回测: Top20 等权月度调仓, Qlib 引擎, 0.1% 单边成本。
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
BT_START = pd.Timestamp("2022-03-01")

MODELS = ["reg", "rank", "linear"]
LABELS = {"reg": "Regression", "rank": "LambdaRank", "linear": "WF-Linear"}
COLORS = {"reg": "#2196F3", "rank": "#FF9800", "linear": "#4CAF50"}


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


def top_n_holdings(scores, min_date, n=20):
    holds = {}
    for m, g in scores.groupby(level=0):
        if m < min_date:
            continue
        top = g.droplevel(0).sort_values(ascending=False).head(n).index.tolist()
        holds[m] = top
    return holds


def main():
    oos = pd.read_parquet(QM / "oos_scores_v2.parquet")
    fwd = oos["fwd_ret"]
    bt_months = [m for m in oos.index.get_level_values(0).unique() if m >= BT_START]

    n_inst = oos.index.get_level_values(1).nunique()
    n_months = oos.index.get_level_values(0).nunique()
    print(f"[v2] OOS: {n_months} months × {n_inst} instruments = {len(oos)} rows")

    # -- IC --
    print("\n=== IC (full OOS) ===")
    ic_full = {}
    for k in MODELS:
        ic_full[k] = monthly_ic(oos[f"{k}_score"], fwd)
        v = ic_full[k]
        print(f"  {LABELS[k]:12} mean_IC={v['mean_ic']:+.4f}  "
              f"ICIR={v['icir']:+.3f}  +months={v['pct_positive']:.0%}  n={v['n_months']}")

    print("\n=== IC (backtest sub-period 2022-04+) ===")
    ic_bt = {}
    for k in MODELS:
        ic_bt[k] = monthly_ic(oos[f"{k}_score"], fwd, months=set(bt_months))
        v = ic_bt[k]
        print(f"  {LABELS[k]:12} mean_IC={v['mean_ic']:+.4f}  "
              f"ICIR={v['icir']:+.3f}  +months={v['pct_positive']:.0%}  n={v['n_months']}")

    # -- Backtest --
    russell_provider = ROOT / "qlib_data" / "russell1000"
    sp500_provider = ROOT / "qlib_data" / "sp500"
    provider = str(russell_provider) if russell_provider.exists() else str(sp500_provider)
    print(f"\n=== Qlib backtest (Top20 EW, 2022-04+, provider={Path(provider).name}) ===")

    res = {}
    for k in MODELS:
        holds = top_n_holdings(oos[f"{k}_score"], BT_START)
        res[k] = run_backtest(holds, cost=0.001, provider_uri=provider)
        g, n = res[k]["gross"], res[k]["net"]
        print(f"  {LABELS[k]:12} gross={g['ann_return']:.4f} "
              f"(Sharpe {g['sharpe']}, MDD {g['max_drawdown']}) | "
              f"net={n['ann_return']:.4f}  [{res[k]['period']}]")

    # -- Reference --
    baseline_path = ROOT / "workspace" / "baseline_sp500" / "baseline.json"
    qlib_ref_path = QM / "qlib_result.json"
    if baseline_path.exists() and qlib_ref_path.exists():
        qres = json.loads(qlib_ref_path.read_text())
        print(f"  {'REF baseline':12} gross {qres['qlib_gross']['ann_return']:.4f} "
              f"(Sharpe {qres['qlib_gross']['sharpe']}), "
              f"net {qres['qlib_net_ann']:.4f}")

    # -- v1 comparison --
    v1_path = QM / "compare_result.json"
    if v1_path.exists():
        v1 = json.loads(v1_path.read_text())
        print("\n  --- v1 (SP500-107, LambdaRank) for reference ---")
        for k in ("lgbm", "linear"):
            bt = v1["backtest"][k]
            g = bt["gross"]
            print(f"  v1-{k:7} gross={g['ann_return']:.4f} (Sharpe {g['sharpe']})")

    # -- Feature importance chart --
    imp_path = QM / "feature_importance_v2.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path, index_col=0).iloc[:, 0]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        imp.plot.bar(ax=ax, color="steelblue", alpha=0.85)
        ax.set_title("LightGBM Regression Feature Importance (v2, avg gain)")
        ax.set_ylabel("gain")
        ax.set_xlabel("feature")
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig.savefig(QM / "feature_importance_v2.png", dpi=150)
        plt.close(fig)

    # -- Cumulative IC chart --
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for k in MODELS:
        scores = oos[f"{k}_score"]
        df = pd.concat([scores.rename("s"), fwd.rename("y")], axis=1).dropna()
        monthly = df.groupby(level=0).apply(
            lambda g: g["s"].corr(g["y"], method="spearman") if len(g) >= 10 else np.nan
        ).dropna()
        axes[0].plot(monthly.index, monthly.cumsum(), label=LABELS[k], color=COLORS[k])
    axes[0].set_title("Cumulative IC")
    axes[0].legend()
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].grid(True, alpha=0.3)

    for k in MODELS:
        holds = top_n_holdings(oos[f"{k}_score"], BT_START)
        months_sorted = sorted(holds.keys())
        if len(months_sorted) < 2:
            continue
        rets = []
        monthly_close_dates = sorted(oos.index.get_level_values(0).unique())
        for j, m in enumerate(months_sorted[:-1]):
            tickers = holds[m]
            next_m = months_sorted[j + 1]
            month_fwd = fwd.loc[next_m] if next_m in fwd.index.get_level_values(0) else None
            if month_fwd is not None:
                avg_ret = month_fwd.reindex(tickers).mean()
                rets.append((next_m, avg_ret))
        if rets:
            ret_series = pd.Series(dict(rets))
            cum = (1 + ret_series).cumprod()
            axes[1].plot(cum.index, cum, label=LABELS[k], color=COLORS[k])
    axes[1].set_title("Cumulative Return (Top20 EW, approx)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(QM / "compare_v2.png", dpi=150)
    plt.close(fig)

    # -- Save JSON --
    out = {
        "version": "v2",
        "universe": "russell1000",
        "ic_full": ic_full,
        "ic_backtest": ic_bt,
        "backtest": {k: res[k] for k in res},
    }
    if baseline_path.exists() and qlib_ref_path.exists():
        qres = json.loads(qlib_ref_path.read_text())
        out["ref_locked_baseline"] = {
            "gross_ann": qres["qlib_gross"]["ann_return"],
            "gross_sharpe": qres["qlib_gross"]["sharpe"],
            "net_ann": qres["qlib_net_ann"],
        }
    (QM / "compare_result_v2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(f"\n[done] compare_result_v2.json + compare_v2.png + feature_importance_v2.png")


if __name__ == "__main__":
    main()
