"""
对比正交化前后的因子表现

运行两遍 pipeline：
  1. 原始因子（无正交化）
  2. PCA 正交化后
对比 IC、Sharpe、权重分布，输出结论
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from agents.base import MessageBus
from agents.orchestrator import Orchestrator, PIPELINE


OUTPUT_DIR = Path("workspace/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline_mode(label, orthogonalize=False):
    """运行一次完整 pipeline，返回关键指标"""
    print(f"\n{'='*60}")
    print(f"  运行模式: {label} (orthogonalize={orthogonalize})")
    print(f"{'='*60}\n")

    workspace = f"workspace/{label}"
    bus = MessageBus(workspace=workspace)

    # 修改 researcher 的参数
    pipeline_with_args = []
    for name, cls, kwargs in PIPELINE:
        if name == "researcher":
            kwargs = {"orthogonalize": orthogonalize}
        pipeline_with_args.append((name, cls, kwargs))

    # 临时替换 PIPELINE
    import agents.orchestrator as orch_mod
    original = orch_mod.PIPELINE
    orch_mod.PIPELINE = pipeline_with_args

    orch = Orchestrator(bus)
    results = orch.run()
    summary = orch.get_summary()

    orch_mod.PIPELINE = original

    # 提取关键指标
    final = bus.get("final_report", {})
    perf = final.get("performance", {})
    backtest = bus.get("backtest_results", {})
    factor_names = bus.get("factor_names", [])

    # IC 汇总
    ic_summary = {}
    for fname, res in backtest.items():
        if "error" not in res:
            ic_vals = res.get("ic_mean", {})
            first_ic = list(ic_vals.values())[0] if ic_vals else 0
            ic_summary[fname] = first_ic

    return {
        "label": label,
        "performance": perf,
        "ic_summary": ic_summary,
        "factor_names": factor_names,
        "corr_before": bus.get("corr_before"),
        "corr_after": bus.get("corr_after"),
        "pca_info": bus.get("pca_info"),
        "backtest_results": backtest,
    }


def plot_comparison(raw, ortho):
    """画对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Factor Orthogonalization: Before vs After",
                 fontsize=16, fontweight="bold")

    # ── [0,0] IC 对比柱状图 ──
    ax = axes[0, 0]
    raw_ic = raw["ic_summary"]
    ortho_ic = ortho["ic_summary"]

    x = np.arange(max(len(raw_ic), len(ortho_ic)))
    width = 0.35

    raw_names = list(raw_ic.keys())
    raw_vals = list(raw_ic.values())
    ortho_names = list(ortho_ic.keys())
    ortho_vals = list(ortho_ic.values())

    bars1 = ax.bar(x[:len(raw_vals)] - width/2, raw_vals, width,
                   label="Original", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x[:len(ortho_vals)] + width/2, ortho_vals, width,
                   label="PCA Orthogonalized", color="coral", alpha=0.8)
    ax.set_xticks(x[:max(len(raw_names), len(ortho_names))])
    ax.set_xticklabels(raw_names if len(raw_names) >= len(ortho_names) else ortho_names,
                       rotation=15, fontsize=9)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_title("Mean IC Comparison")
    ax.set_ylabel("IC")
    ax.legend()

    # ── [0,1] Sharpe + 收益对比 ──
    ax = axes[0, 1]
    metrics = ["组合年化收益(净)", "超额收益(净)", "组合Sharpe", "最大回撤"]
    labels_short = ["Ann.Ret(Net)", "Excess(Net)", "Sharpe", "MaxDD"]
    raw_perf = raw["performance"]
    ortho_perf = ortho["performance"]

    raw_m = [raw_perf.get(m, 0) for m in metrics]
    ortho_m = [ortho_perf.get(m, 0) for m in metrics]

    x2 = np.arange(len(metrics))
    ax.bar(x2 - width/2, raw_m, width, label="Original", color="steelblue", alpha=0.8)
    ax.bar(x2 + width/2, ortho_m, width, label="PCA Orthogonalized", color="coral", alpha=0.8)
    ax.set_xticks(x2)
    ax.set_xticklabels(labels_short, fontsize=10)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_title("Performance Metrics Comparison")
    ax.legend()

    # 在柱子上标数值
    for bars, vals in [(bars1, raw_vals), (bars2, ortho_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    # ── [1,0] 因子权重对比 ──
    ax = axes[1, 0]
    raw_w = raw_perf.get("末月权重", {})
    ortho_w = ortho_perf.get("末月权重", {})

    if raw_w and ortho_w:
        all_names = list(raw_w.keys()) + [k for k in ortho_w.keys() if k not in raw_w]
        x3 = np.arange(len(all_names))
        raw_w_vals = [raw_w.get(n, 0) for n in all_names]

        ortho_names_w = list(ortho_w.keys())
        ortho_w_vals = list(ortho_w.values())
        x4 = np.arange(len(ortho_names_w))

        ax.barh(x3, raw_w_vals, height=0.4, label="Original", color="steelblue", alpha=0.7)
        ax.set_yticks(x3)
        ax.set_yticklabels(all_names, fontsize=9)

        # 画第二组在右边
        ax2 = ax.twiny()
        ax2.barh(x4 + 0.4, ortho_w_vals, height=0.4, label="PCA", color="coral", alpha=0.7)
        ax2.set_yticks(x4)

        ax.axvline(0, color="k", ls="--", lw=0.5)
        ax.set_title("Factor Weights: Original (blue) vs PCA (coral)")
        ax.legend(loc="lower right")

    # ── [1,1] 汇总结论 ──
    ax = axes[1, 1]
    ax.axis("off")

    raw_sharpe = raw_perf.get("组合Sharpe", 0)
    ortho_sharpe = ortho_perf.get("组合Sharpe", 0)
    delta_sharpe = ortho_sharpe - raw_sharpe

    raw_excess = raw_perf.get("超额收益(净)", 0)
    ortho_excess = ortho_perf.get("超额收益(净)", 0)

    raw_dd = raw_perf.get("最大回撤", 0)
    ortho_dd = ortho_perf.get("最大回撤", 0)

    # 判断结论
    if delta_sharpe > 0.1:
        verdict = "SIGNIFICANT IMPROVEMENT"
        verdict_color = "#3fb950"
    elif delta_sharpe > 0:
        verdict = "MARGINAL IMPROVEMENT"
        verdict_color = "#d29922"
    else:
        verdict = "NO IMPROVEMENT"
        verdict_color = "#f85149"

    text = f"""
    ORTHOGONALIZATION COMPARISON SUMMARY
    ─────────────────────────────────────

    Metric              Original    PCA Ortho    Delta
    ──────────────────────────────────────────────────
    Sharpe Ratio        {raw_sharpe:>8.2f}     {ortho_sharpe:>8.2f}     {delta_sharpe:>+7.2f}
    Excess Return       {raw_excess:>7.2%}     {ortho_excess:>7.2%}     {ortho_excess-raw_excess:>+7.2%}
    Max Drawdown        {raw_dd:>7.2%}     {ortho_dd:>7.2%}     {ortho_dd-raw_dd:>+7.2%}

    Verdict: {verdict}

    Notes:
    - PCA 将 5 个相关因子转为 5 个正交主成分
    - PC1~PC5 按方差解释比排序
    - 正交化消除多重共线性，使 IC 加权更稳定
    """
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=10, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22",
                      edgecolor=verdict_color, linewidth=2))

    plt.tight_layout()
    path = str(OUTPUT_DIR / "ortho_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[INFO] 对比图已保存: {path}")

    return {
        "verdict": verdict,
        "delta_sharpe": delta_sharpe,
        "raw_sharpe": raw_sharpe,
        "ortho_sharpe": ortho_sharpe,
    }


def main():
    # 安装 sklearn
    print("[Step 1/3] 运行原始因子 pipeline ...")
    raw = run_pipeline_mode("raw", orthogonalize=False)

    print("\n[Step 2/3] 运行 PCA 正交化 pipeline ...")
    ortho = run_pipeline_mode("ortho", orthogonalize=True)

    print("\n[Step 3/3] 生成对比报告 ...")
    result = plot_comparison(raw, ortho)

    # 打印结论
    print(f"\n{'='*60}")
    print(f"  CONCLUSION: {result['verdict']}")
    print(f"{'='*60}")
    print(f"  原始 Sharpe:   {result['raw_sharpe']:.2f}")
    print(f"  正交 Sharpe:   {result['ortho_sharpe']:.2f}")
    print(f"  Delta Sharpe:  {result['delta_sharpe']:+.2f}")
    print()


if __name__ == "__main__":
    main()
