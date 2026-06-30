"""
对比行业中性化的效果

三组实验：
  A. S&P 500 核心 + PCA（当前最优配置）
  B. Russell 1000 + PCA（无中性化，上次实验已知 Sharpe 下降）
  C. Russell 1000 + 行业中性化 + PCA（本次新增）

目标：验证行业中性化是否能让 Russell 1000 扩池后 Sharpe 回升
"""

import os
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


def run_experiment(label, tickers, liquidity_filter, industry_neutral):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  tickers={len(tickers)}, liquidity={liquidity_filter}, neutral={industry_neutral}")
    print(f"{'='*60}\n")

    workspace = f"workspace/{label}"
    bus = MessageBus(workspace=workspace)

    pipeline = [
        ("data_engineer", DataEngineer,
         {"tickers": tickers, "liquidity_filter": liquidity_filter, "min_adv": 5_000_000}),
        ("researcher", Researcher,
         {"orthogonalize": True, "industry_neutral": industry_neutral}),
        ("backtester", Backtester, {}),
        ("critic", Critic, {}),
        ("risk_manager", RiskManager, {}),
        ("pm", PortfolioManager, {}),
    ]

    original = orch_mod.PIPELINE
    orch_mod.PIPELINE = pipeline
    orch = Orchestrator(bus)
    orch.run()
    orch_mod.PIPELINE = original

    perf = bus.get("final_report", {}).get("performance", {})
    quality = bus.get("data_quality", {})
    factors_dict = bus.get("factors_dict", {})
    avg_cand = np.mean([len(df.dropna(how="all")) for df in factors_dict.values()]) if factors_dict else 0

    return {
        "label": label,
        "perf": perf,
        "n_input": len(tickers),
        "n_final": quality.get("股票数", 0),
        "avg_candidates": avg_cand,
    }


def plot_three_way(results):
    """三组对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Industry Neutralization Experiment", fontsize=16, fontweight="bold")

    labels = [r["label"] for r in results]
    colors = ["steelblue", "coral", "#3fb950"]

    # ── [0,0] Sharpe / 收益 / 回撤 ──
    ax = axes[0, 0]
    metrics = [
        ("组合年化收益(净)", "Ann.Return(Net)"),
        ("超额收益(净)", "Excess(Net)"),
        ("组合Sharpe", "Sharpe"),
        ("最大回撤", "MaxDD"),
    ]
    x = np.arange(len(metrics))
    w = 0.25
    for i, r in enumerate(results):
        vals = [r["perf"].get(m[0], 0) for m in metrics]
        bars = ax.bar(x + i*w - w, vals, w, label=r["label"], color=colors[i], alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics], fontsize=9)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_title("Performance Comparison")
    ax.legend(fontsize=8)

    # ── [0,1] 换手率 + 成本 ──
    ax = axes[0, 1]
    cost_m = [("平均月换手率", "Turnover"), ("年化成本拖累", "Cost Drag")]
    x2 = np.arange(len(cost_m))
    for i, r in enumerate(results):
        vals = [r["perf"].get(m[0], 0) for m in cost_m]
        bars = ax.bar(x2 + i*w - w, vals, w, label=r["label"], color=colors[i], alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.1%}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x2)
    ax.set_xticklabels([m[1] for m in cost_m])
    ax.set_title("Turnover & Cost")
    ax.legend(fontsize=8)

    # ── [1,0] 股票池信息 ──
    ax = axes[1, 0]
    ax.axis("off")
    header = f"{'Metric':<22}" + "".join(f"{r['label']:>18}" for r in results)
    sep = "─" * (22 + 18 * len(results))

    rows = [
        ("输入股票数", [r["n_input"] for r in results], "d"),
        ("有效股票数", [r["n_final"] for r in results], "d"),
        ("月均候选数", [r["avg_candidates"] for r in results], ".0f"),
        ("年化收益(净)", [r["perf"].get("组合年化收益(净)", 0) for r in results], ".2%"),
        ("超额收益(净)", [r["perf"].get("超额收益(净)", 0) for r in results], ".2%"),
        ("Sharpe", [r["perf"].get("组合Sharpe", 0) for r in results], ".2f"),
        ("最大回撤", [r["perf"].get("最大回撤", 0) for r in results], ".2%"),
        ("月均换手率", [r["perf"].get("平均月换手率", 0) for r in results], ".1%"),
        ("年化成本", [r["perf"].get("年化成本拖累", 0) for r in results], ".2%"),
    ]

    text = f"  {header}\n  {sep}\n"
    for name, vals, fmt in rows:
        line = f"  {name:<22}"
        for v in vals:
            line += f"{v:>18{fmt}}"
        text += line + "\n"

    ax.text(0.02, 0.95, text, transform=ax.transAxes,
            fontsize=9, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22",
                      edgecolor="#30363d", linewidth=1))

    # ── [1,1] 结论 ──
    ax = axes[1, 1]
    ax.axis("off")

    sp_sharpe = results[0]["perf"].get("组合Sharpe", 0)
    rs_sharpe = results[1]["perf"].get("组合Sharpe", 0)
    rn_sharpe = results[2]["perf"].get("组合Sharpe", 0)

    sp_dd = abs(results[0]["perf"].get("最大回撤", 0))
    rs_dd = abs(results[1]["perf"].get("最大回撤", 0))
    rn_dd = abs(results[2]["perf"].get("最大回撤", 0))

    conclusions = []
    conclusions.append(f"SP500 Sharpe: {sp_sharpe:.2f}")
    conclusions.append(f"R1000 无中性化 Sharpe: {rs_sharpe:.2f} ({rs_sharpe-sp_sharpe:+.2f})")
    conclusions.append(f"R1000 行业中性化 Sharpe: {rn_sharpe:.2f} ({rn_sharpe-sp_sharpe:+.2f})")
    conclusions.append("")

    if rn_sharpe > rs_sharpe:
        conclusions.append(f"行业中性化提升了 Sharpe {rn_sharpe-rs_sharpe:+.2f}")
    else:
        conclusions.append(f"行业中性化未能提升 Sharpe ({rn_sharpe-rs_sharpe:+.2f})")

    if rn_dd < rs_dd:
        conclusions.append(f"回撤从 {rs_dd:.1%} 缩小到 {rn_dd:.1%}")
    else:
        conclusions.append(f"回撤未改善 ({rs_dd:.1%} → {rn_dd:.1%})")

    if rn_sharpe > sp_sharpe:
        verdict = "INDUSTRY NEUTRAL + R1000 WINS"
        color = "#3fb950"
    elif rn_sharpe > rs_sharpe:
        verdict = "NEUTRAL HELPS, BUT SP500 STILL BETTER"
        color = "#d29922"
    else:
        verdict = "STICK WITH SP500"
        color = "#f85149"

    text = f"  VERDICT: {verdict}\n  {'─'*40}\n\n"
    text += "\n".join(f"  • {c}" for c in conclusions)

    ax.text(0.02, 0.95, text, transform=ax.transAxes,
            fontsize=10, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22",
                      edgecolor=color, linewidth=2))

    plt.tight_layout()
    path = str(OUTPUT_DIR / "neutral_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[INFO] 对比图已保存: {path}")
    return verdict, conclusions


def main():
    print("[1/3] S&P 500 + PCA (baseline) ...")
    a = run_experiment("SP500+PCA", SP500_CORE,
                       liquidity_filter=False, industry_neutral=False)

    print("[2/3] Russell 1000 + PCA (no neutral) ...")
    b = run_experiment("R1000+PCA", RUSSELL1000,
                       liquidity_filter=True, industry_neutral=False)

    print("[3/3] Russell 1000 + Industry Neutral + PCA ...")
    c = run_experiment("R1000+Neutral+PCA", RUSSELL1000,
                       liquidity_filter=True, industry_neutral=True)

    print("\n[生成对比报告] ...")
    verdict, conclusions = plot_three_way([a, b, c])

    print(f"\n{'='*60}")
    print(f"  {verdict}")
    print(f"{'='*60}")
    for line in conclusions:
        print(f"  {line}")
    print()


if __name__ == "__main__":
    main()
