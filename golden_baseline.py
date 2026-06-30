"""
Golden baseline runner — 锁定的基准配置: SP500-107, PCA-only (无行业中性化)

运行完整 pipeline 并把"精确"参考结果落盘，作为后续 Qlib 迁移 <1% 对齐的金标准。
产出 (workspace/baseline_sp500/):
  - baseline.json          : 绩效指标 + 元信息
  - daily_returns.csv      : 组合(净/毛)与基准的日收益序列
  - holdings_by_date.json  : 每个调仓日的 Top20 持仓
  - weights_by_date.json   : 每月动态因子权重
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents.base import MessageBus
from agents.universe import SP500_CORE
import agents.orchestrator as orch_mod
from agents.orchestrator import Orchestrator
from agents.data_engineer import DataEngineer
from agents.researcher import Researcher
from agents.backtester import Backtester
from agents.critic import Critic
from agents.risk_manager import RiskManager
from agents.pm import PortfolioManager

OUT = Path("workspace/baseline_sp500")
OUT.mkdir(parents=True, exist_ok=True)

# 锁定配置：SP500 核心, 无流动性过滤, PCA 正交化, 无行业中性化
PIPELINE = [
    ("data_engineer", DataEngineer,
     {"tickers": SP500_CORE, "liquidity_filter": False}),
    ("researcher", Researcher, {"orthogonalize": True, "industry_neutral": False}),
    ("backtester", Backtester, {}),
    ("critic", Critic, {}),
    ("risk_manager", RiskManager, {}),
    ("pm", PortfolioManager, {}),
]


def main():
    bus = MessageBus(workspace=str(OUT))
    orch_mod.PIPELINE = PIPELINE
    orch = Orchestrator(bus)
    orch.run()

    perf = (bus.get("final_report") or {}).get("performance", {})
    quality = bus.get("data_quality", {})
    pca = bus.get("pca_info", {})

    # ── 日收益序列 ──
    rets = pd.DataFrame({
        "port_net": bus.get("port_ret_net"),
        "port_gross": bus.get("port_ret_gross"),
        "benchmark": bus.get("bench_ret"),
    })
    rets.to_csv(OUT / "daily_returns.csv")

    # ── 持仓 / 权重 ──
    (OUT / "holdings_by_date.json").write_text(
        json.dumps(bus.get("holdings_by_date", {}), ensure_ascii=False, indent=2))
    weights = {str(k.date()): {f: round(float(v), 6) for f, v in w.items()}
               for k, w in (bus.get("weight_history") or {}).items()}
    (OUT / "weights_by_date.json").write_text(
        json.dumps(weights, ensure_ascii=False, indent=2))

    # ── 主参考 JSON ──
    baseline = {
        "config": {
            "universe": "SP500_CORE",
            "n_input_tickers": len(SP500_CORE),
            "liquidity_filter": False,
            "orthogonalize": True,
            "industry_neutral": False,
            "top_n": 20,
            "ic_window": 6,
            "one_way_cost": 0.001,
            "rebalance": "month-end",
        },
        "data_quality": quality,
        "performance": perf,
        "pca_explained_variance_ratio": pca.get("explained_variance_ratio"),
        "n_trading_days": int(rets["port_net"].notna().sum()),
    }
    (OUT / "baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2, default=str))

    print("\n=== GOLDEN BASELINE WRITTEN ===")
    print(json.dumps(perf, ensure_ascii=False, indent=2, default=str))
    print("stocks:", quality.get("股票数"), "| trading days:", baseline["n_trading_days"])


if __name__ == "__main__":
    main()
