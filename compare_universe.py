"""
对比股票池扩展前后的效果

SP500 核心 (~107只) vs Russell 1000 (~500只，流动性过滤后)
其他参数不变：PCA 正交化、滚动 6M IC、Top20 等权、交易成本 0.1%/边
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
from agents.universe import SP500_CORE, RUSSELL1000
import agents.orchestrator as orch_mod
from agents.orchestrator import Orchestrator, PIPELINE
from agents.data_engineer import DataEngineer
from agents.researcher import Researcher
from agents.backtester import Backtester
from agents.critic import Critic
from agents.risk_manager import RiskManager
from agents.pm import PortfolioManager


OUTPUT_DIR = Path("workspace/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline_with_universe(label, tickers, liquidity_filter=False):
    """运行一次完整 pipeline，返回关键指标"""
    print(f"\n{'='*60}")
    print(f"  {label}: {len(tickers)} tickers, liquidity_filter={liquidity_filter}")
    print(f"{'='*60}\n")

    workspace = f"workspace/{label}"
    bus = MessageBus(workspace=workspace)

    pipeline = [
        ("data_engineer", DataEngineer,
         {"tickers": tickers, "liquidity_filter": liquidity_filter, "min_adv": 5_000_000}),
        ("researcher", Researcher, {"orthogonalize": True}),
        ("backtester", Backtester, {}),
        ("critic", Critic, {}),
        ("risk_manager", RiskManager, {}),
        ("pm", PortfolioManager, {}),
    ]

    original = orch_mod.PIPELINE
    orch_mod.PIPELINE = pipeline
    orch = Orchestrator(bus)
    results = orch.run()
    orch_mod.PIPELINE = original

    final = bus.get("final_report", {})
    perf = final.get("performance", {})
    quality = bus.get("data_quality", {})
    factor_stats = bus.get("factor_stats", {})
    factors_dict = bus.get("factors_dict", {})

    # 每月平均候选股票数
    if factors_dict:
        avg_candidates = np.mean([len(df.dropna(how="all")) for df in factors_dict.values()])
    else:
        avg_candidates = 0

    return {
        "label": label,
        "performance": perf,
        "quality": quality,
        "factor_stats": factor_stats,
        "avg_candidates": avg_candidates,
        "n_tickers_input": len(tickers),
        "n_tickers_final": quality.get("股票数", 0),
    }


def plot_comparison(sp500, russell):
    """对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Universe Expansion: S&P 500 Core vs Russell 1000",
                 fontsize=16, fontweight="bold")

    sp = sp500["performance"]
    rs = russell["performance"]

    # ── [0,0] 绩效对比柱状图 ──
    ax = axes[0, 0]
    metrics = ["组合年化收益(净)", "超额收益(净)", "组合Sharpe", "最大回撤"]
    labels = ["Ann.Ret(Net)", "Excess(Net)", "Sharpe", "MaxDD"]
    sp_vals = [sp.get(m, 0) for m in metrics]
    rs_vals = [rs.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax.bar(x - w/2, sp_vals, w, label=f"S&P500 ({sp500['n_tickers_final']}只)",
                   color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + w/2, rs_vals, w, label=f"Russell1000 ({russell['n_tickers_final']}只)",
                   color="coral", alpha=0.8)

    # 标数值
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_title("Performance Comparison")
    ax.legend()

    # ── [0,1] 换手率 + 成本 ──
    ax = axes[0, 1]
    cost_metrics = ["平均月换手率", "年化成本拖累"]
    cost_labels = ["Monthly Turnover", "Ann. Cost Drag"]
    sp_c = [sp.get(m, 0) for m in cost_metrics]
    rs_c = [rs.get(m, 0) for m in cost_metrics]

    x2 = np.arange(len(cost_metrics))
    ax.bar(x2 - w/2, sp_c, w, label="S&P500", color="steelblue", alpha=0.8)
    ax.bar(x2 + w/2, rs_c, w, label="Russell1000", color="coral", alpha=0.8)
    ax.set_xticks(x2)
    ax.set_xticklabels(cost_labels)
    ax.set_title("Turnover & Cost")
    ax.legend()

    # 标数值
    for i, (sv, rv) in enumerate(zip(sp_c, rs_c)):
        ax.text(i - w/2, sv, f'{sv:.1%}', ha='center', va='bottom', fontsize=9)
        ax.text(i + w/2, rv, f'{rv:.1%}', ha='center', va='bottom', fontsize=9)

    # ── [1,0] 股票池信息 ──
    ax = axes[1, 0]
    ax.axis("off")

    info_text = f"""
    UNIVERSE COMPARISON
    ─────────────────────────────────────────────

    Metric               S&P 500       Russell 1000
    ─────────────────────────────────────────────
    输入股票数            {sp500['n_tickers_input']:>6}         {russell['n_tickers_input']:>6}
    有效股票数            {sp500['n_tickers_final']:>6}         {russell['n_tickers_final']:>6}
    月均候选数            {sp500['avg_candidates']:>6.0f}         {russell['avg_candidates']:>6.0f}

    年化收益(净)       {sp.get('组合年化收益(净)',0):>8.2%}       {rs.get('组合年化收益(净)',0):>8.2%}
    超额收益(净)       {sp.get('超额收益(净)',0):>8.2%}       {rs.get('超额收益(净)',0):>8.2%}
    Sharpe              {sp.get('组合Sharpe',0):>8.2f}       {rs.get('组合Sharpe',0):>8.2f}
    最大回撤            {sp.get('最大回撤',0):>8.2%}       {rs.get('最大回撤',0):>8.2%}
    月均换手率          {sp.get('平均月换手率',0):>8.1%}       {rs.get('平均月换手率',0):>8.1%}
    年化成本            {sp.get('年化成本拖累',0):>8.2%}       {rs.get('年化成本拖累',0):>8.2%}
    """
    ax.text(0.02, 0.95, info_text, transform=ax.transAxes,
            fontsize=10, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22",
                      edgecolor="#30363d", linewidth=1))

    # ── [1,1] 结论 ──
    ax = axes[1, 1]
    ax.axis("off")

    sp_sharpe = sp.get("组合Sharpe", 0)
    rs_sharpe = rs.get("组合Sharpe", 0)
    delta_sharpe = rs_sharpe - sp_sharpe

    sp_dd = abs(sp.get("最大回撤", 0))
    rs_dd = abs(rs.get("最大回撤", 0))

    sp_turnover = sp.get("平均月换手率", 0)
    rs_turnover = rs.get("平均月换手率", 0)

    conclusions = []
    if delta_sharpe > 0.1:
        conclusions.append(f"Sharpe 提升 {delta_sharpe:+.2f} → 更宽截面带来更好信号")
    elif delta_sharpe > 0:
        conclusions.append(f"Sharpe 微升 {delta_sharpe:+.2f} → 边际改善")
    else:
        conclusions.append(f"Sharpe 下降 {delta_sharpe:+.2f} → 扩池未改善信号质量")

    if rs_dd < sp_dd:
        conclusions.append(f"回撤缩小 ({sp_dd:.1%} → {rs_dd:.1%}) → 分散化有效")
    else:
        conclusions.append(f"回撤增大 ({sp_dd:.1%} → {rs_dd:.1%}) → 中小盘增加波动")

    if rs_turnover < sp_turnover:
        conclusions.append(f"换手率降低 ({sp_turnover:.0%} → {rs_turnover:.0%}) → 选股更稳定")
    else:
        conclusions.append(f"换手率升高 ({sp_turnover:.0%} → {rs_turnover:.0%}) → 候选多→更多轮换")

    verdict = "WIDER UNIVERSE HELPS" if delta_sharpe > 0 and rs_dd <= sp_dd else \
              "MIXED RESULTS" if delta_sharpe > 0 else "NO IMPROVEMENT"

    conclusion_text = f"""
    CONCLUSION: {verdict}
    ─────────────────────────

    """ + "\n    ".join(f"• {c}" for c in conclusions) + f"""

    更宽的截面 ({sp500['n_tickers_final']} → {russell['n_tickers_final']} 只):
    - 更大的选股空间 → 找到 alpha 的机会更多
    - 更好的分散化 → 单股风险降低
    - 但中小盘股流动性差 → 需要流动性过滤
    - 基本面数据覆盖率可能下降
    """

    ax.text(0.02, 0.95, conclusion_text, transform=ax.transAxes,
            fontsize=10, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22",
                      edgecolor="#58a6ff" if "HELPS" in verdict else "#d29922",
                      linewidth=2))

    plt.tight_layout()
    path = str(OUTPUT_DIR / "universe_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[INFO] 对比图已保存: {path}")

    return {"verdict": verdict, "delta_sharpe": delta_sharpe, "conclusions": conclusions}


def main():
    print("[Step 1/3] 运行 S&P 500 核心 pipeline ...")
    sp500 = run_pipeline_with_universe("sp500", SP500_CORE, liquidity_filter=False)

    print("\n[Step 2/3] 运行 Russell 1000 pipeline (流动性过滤) ...")
    russell = run_pipeline_with_universe("russell1000", RUSSELL1000, liquidity_filter=True)

    print("\n[Step 3/3] 生成对比报告 ...")
    result = plot_comparison(sp500, russell)

    print(f"\n{'='*60}")
    print(f"  CONCLUSION: {result['verdict']}")
    print(f"{'='*60}")
    for c in result["conclusions"]:
        print(f"  • {c}")
    print()


if __name__ == "__main__":
    main()
