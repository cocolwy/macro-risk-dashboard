"""
Orchestrator: 流程编排 — 按顺序调度各 Agent，处理打回逻辑
"""

import time
from .base import BaseAgent, MessageBus, AgentStatus
from .data_engineer import DataEngineer
from .researcher import Researcher
from .backtester import Backtester
from .critic import Critic
from .risk_manager import RiskManager
from .pm import PortfolioManager


PIPELINE = [
    ("data_engineer", DataEngineer,
     {"tickers": None, "liquidity_filter": True, "min_adv": 5_000_000}),
    ("researcher", Researcher, {"orthogonalize": True, "industry_neutral": True}),
    ("backtester", Backtester, {}),
    ("critic", Critic, {}),
    ("risk_manager", RiskManager, {}),
    ("pm", PortfolioManager, {}),
]


class Orchestrator:
    """
    流水线编排器

    DataEngineer → Researcher → Backtester → Critic → RiskManager → PM
                                               ↑ 打回则停止并报告
    """

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.agents = {}
        self.results = {}

        for name, cls, _ in PIPELINE:
            self.agents[name] = cls(bus)

    def run(self, stop_on_reject=False):
        """执行完整流水线"""
        self.bus.log("orchestrator", "info", "流水线启动",
                     f"共 {len(PIPELINE)} 个阶段")

        for name, cls, kwargs in PIPELINE:
            agent = self.agents[name]
            self.bus.log("orchestrator", "info", "调度",
                         f"→ {agent.role} ({name})")

            try:
                result = agent.run(**kwargs)
                self.results[name] = result
            except Exception as e:
                self.results[name] = {"error": str(e)}
                self.bus.log("orchestrator", "error", "阶段失败",
                             f"{name}: {str(e)}")
                if name in ("data_engineer", "researcher"):
                    # 关键阶段失败，终止流水线
                    self.bus.log("orchestrator", "error", "流水线终止",
                                 "关键阶段失败，无法继续")
                    break
                continue

            # Critic 打回逻辑
            if name == "critic" and stop_on_reject:
                review = self.bus.get("review", {})
                if review.get("verdict") == "REJECT":
                    self.bus.log("orchestrator", "error", "Critic 打回",
                                 "存在严重问题，流水线暂停")
                    break

        self.bus.log("orchestrator", "success", "流水线完成",
                     f"完成阶段: {list(self.results.keys())}")

        return self.results

    def get_summary(self) -> dict:
        """生成最终摘要"""
        return {
            "agents": {
                name: self.bus.get_status(name).value
                for name, _, _ in PIPELINE
            },
            "results": self.results,
            "review": self.bus.get("review"),
            "risk_report": self.bus.get("risk_report"),
            "final_report": self.bus.get("final_report"),
        }
