"""
RiskOrchestrator — 风控模型流水线编排器, 配置驱动。

Pipeline:
  MacroDataAgent → FeatureAgent → ModelTrainerAgent → ModelCriticAgent → ScoreAgent

所有配置通过 config dict 传入, 流转到各 Agent:
  config = {
      "indicators": ["vix", "sp500", "gold"],     # 要获取的指标
      "features": ["full", "slim", "my_feats"],    # 要构建的特征集
      "target_col": "sp500",                       # 标签基于哪个列
      "embargo": 20,                               # 训练测试间隔
  }
"""
import time
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
    风控流水线编排器 — 配置驱动, 可适应不同任务。

    MacroDataAgent → FeatureAgent → ModelTrainerAgent → ModelCriticAgent → ScoreAgent
    """

    def __init__(self, workspace="workspace/risk_pipeline", config=None):
        """
        Parameters
        ----------
        config : dict | None
            流水线配置, 支持以下 key:
            - indicators: list[str]   选择哪些指标 (默认全部注册指标)
            - features: list[str]     选择哪些特征集 (默认全部注册特征)
            - target_col: str         标签列 (默认 "sp500")
            - target_fn: callable     自定义标签函数
            - embargo: int            训练测试间隔天数 (默认 20)
            - split_ratio: float      训练集比例 (默认 0.7)
        """
        self.bus = MessageBus(workspace=workspace)
        self.config = config or {}
        self.agents = {}
        self.results = {}

        for name, cls, _ in RISK_PIPELINE:
            self.agents[name] = cls(self.bus)

    def _get_agent_kwargs(self, name, use_cached):
        """根据 config 为每个 Agent 生成 kwargs。"""
        c = self.config

        if name == "macro_data":
            return {
                "use_cached": use_cached,
                "indicators": c.get("indicators"),
            }
        elif name == "feature_eng":
            kw = {}
            if "features" in c:
                kw["feature_variants"] = c["features"]
            if "target_col" in c:
                kw["target_col"] = c["target_col"]
            if "target_fn" in c:
                kw["target_fn"] = c["target_fn"]
            return kw
        elif name == "model_trainer":
            kw = {}
            if "embargo" in c:
                kw["embargo"] = c["embargo"]
            if "split_ratio" in c:
                kw["split_ratio"] = c["split_ratio"]
            return kw
        else:
            return {}

    def run(self, use_cached=False, stop_on_reject=False):
        start = time.time()
        self.bus.log("orchestrator", "info", "风控流水线启动",
                     f"共 {len(RISK_PIPELINE)} 个阶段, cached={use_cached}, config={list(self.config.keys())}")

        for i, (name, cls, _) in enumerate(RISK_PIPELINE, 1):
            agent = self.agents[name]
            kwargs = self._get_agent_kwargs(name, use_cached)

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
        review = self.bus.get("review", {})
        metrics = self.bus.get("metrics", {})
        experiments = self.bus.get("experiments", [])
        composite = self.bus.get("composite_scores", [])

        return {
            "pipeline": "RiskOrchestrator",
            "config": self.config,
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
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("Risk Pipeline Summary")
        if summary["config"]:
            print(f"Config: {summary['config']}")
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
