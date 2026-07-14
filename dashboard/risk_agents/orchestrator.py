"""
RiskOrchestrator — 风控模型流水线编排器。

复用 quant multi-agent 基础设施 (BaseAgent + MessageBus + Orchestrator 模式)。

Pipeline:
  MacroDataAgent → FeatureAgent → ModelTrainerAgent → ModelCriticAgent → ScoreAgent
                                                        ↑ 打回则报告但不停止
"""
import time
import sys
from pathlib import Path

from ._base import MessageBus, AgentStatus

from .data_agent import MacroDataAgent
from .feature_agent import FeatureAgent
from .model_agent import ModelTrainerAgent
from .critic_agent import ModelCriticAgent
from .score_agent import ScoreAgent


RISK_PIPELINE = [
    ("macro_data",     MacroDataAgent,    {}),
    ("feature_eng",    FeatureAgent,      {}),
    ("model_trainer",  ModelTrainerAgent,  {}),
    ("model_critic",   ModelCriticAgent,   {}),
    ("score_output",   ScoreAgent,        {}),
]


class RiskOrchestrator:
    """
    风控流水线编排器

    MacroDataAgent → FeatureAgent → ModelTrainerAgent → ModelCriticAgent → ScoreAgent
    """

    def __init__(self, workspace="workspace/risk_pipeline"):
        self.bus = MessageBus(workspace=workspace)
        self.agents = {}
        self.results = {}

        for name, cls, _ in RISK_PIPELINE:
            self.agents[name] = cls(self.bus)

    def run(self, use_cached=False, stop_on_reject=False):
        """
        执行完整风控流水线。

        Parameters
        ----------
        use_cached : bool
            True 时跳过数据下载, 使用已有 JSON (快速模式)
        stop_on_reject : bool
            True 时, ModelCritic 打回后停止流水线
        """
        start = time.time()
        self.bus.log("orchestrator", "info", "风控流水线启动",
                     f"共 {len(RISK_PIPELINE)} 个阶段, cached={use_cached}")

        for i, (name, cls, default_kwargs) in enumerate(RISK_PIPELINE, 1):
            agent = self.agents[name]
            kwargs = dict(default_kwargs)

            if name == "macro_data":
                kwargs["use_cached"] = use_cached

            self.bus.log("orchestrator", "info", f"[{i}/{len(RISK_PIPELINE)}] 调度",
                         f"→ {agent.role} ({name})")

            try:
                result = agent.run(**kwargs)
                self.results[name] = result
            except Exception as e:
                self.results[name] = {"error": str(e)}
                self.bus.log("orchestrator", "error", "阶段失败",
                             f"{name}: {str(e)}")
                if name in ("macro_data", "feature_eng"):
                    self.bus.log("orchestrator", "error", "流水线终止",
                                 "关键阶段失败, 无法继续")
                    break
                continue

            if name == "model_critic" and stop_on_reject:
                review = self.bus.get("review", {})
                if review.get("verdict") == "REJECT":
                    self.bus.log("orchestrator", "error", "Critic 打回",
                                 "存在严重问题, 流水线暂停")
                    break

        elapsed = time.time() - start
        self.bus.log("orchestrator", "success", "风控流水线完成",
                     f"{len(self.results)} 阶段完成, 耗时 {elapsed:.1f}s")

        return self.results

    def get_summary(self) -> dict:
        """生成执行摘要。"""
        review = self.bus.get("review", {})
        metrics = self.bus.get("metrics", {})
        experiments = self.bus.get("experiments", [])
        composite = self.bus.get("composite_scores", [])

        return {
            "pipeline": "RiskOrchestrator",
            "stages": {
                name: self.bus.get_status(name).value
                for name, _, _ in RISK_PIPELINE
            },
            "results": self.results,
            "review": review,
            "current_prediction": metrics.get("current_prediction"),
            "n_experiments": len(experiments),
            "composite_latest": composite[-1] if composite else None,
        }

    def print_summary(self):
        """打印人类可读的执行摘要。"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("Risk Pipeline Summary")
        print("=" * 60)

        print("\n[Stages]")
        for name, status in summary["stages"].items():
            icon = {"success": "✓", "failed": "✗", "running": "⋯"}.get(status, "·")
            print(f"  {icon} {name}: {status}")

        review = summary.get("review", {})
        if review:
            verdict = review.get("verdict", "N/A")
            icon = {"PASS": "✓", "WARN": "⚠", "REJECT": "✗"}.get(verdict, "?")
            print(f"\n[Model Review] {icon} {verdict}")
            for w in review.get("warnings", []):
                print(f"  ⚠ {w}")

        pred = summary.get("current_prediction")
        if pred:
            print(f"\n[Current Prediction]")
            print(f"  Crash probability: {pred['probability']*100:.1f}% ({pred['signal']})")
            print(f"  Date: {pred['date']}")

        comp = summary.get("composite_latest")
        if comp:
            print(f"\n[Composite Risk Score]")
            print(f"  Score: {comp['composite_score']:.0f}/100")
            print(f"  Date: {comp['date']}")

        print(f"\n[Experiments] {summary['n_experiments']} model variants compared")
        print("=" * 60)
