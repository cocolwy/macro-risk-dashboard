"""
Risk Model Agent Pipeline — 配置驱动, 可插拔的风控流水线。

Pipeline:
  MacroDataAgent → FeatureAgent → ModelTrainerAgent → ModelCriticAgent → ScoreAgent

指标/特征/模型全部通过注册表管理, 可按需增删:

  from dashboard.risk_agents.registry import INDICATOR, FEATURE_BUILDER

  # 添加黄金指标
  @INDICATOR.register("gold", json_field="gold", parser=lambda d: d.get("gold"))
  def fetch_gold():
      ...

  # 添加自定义特征
  @FEATURE_BUILDER.register("my_features")
  def my_features(df):
      ...

  # 只用部分指标运行
  orch = RiskOrchestrator(config={"indicators": ["vix", "sp500", "gold"]})
  orch.run()
"""
from .registry import INDICATOR, FEATURE_BUILDER, MODEL_CONFIG, CRITIC_CHECK
from .data_agent import MacroDataAgent
from .feature_agent import FeatureAgent
from .model_agent import ModelTrainerAgent
from .critic_agent import ModelCriticAgent
from .score_agent import ScoreAgent
from .orchestrator import RiskOrchestrator

__all__ = [
    "INDICATOR", "FEATURE_BUILDER", "MODEL_CONFIG", "CRITIC_CHECK",
    "MacroDataAgent", "FeatureAgent", "ModelTrainerAgent",
    "ModelCriticAgent", "ScoreAgent", "RiskOrchestrator",
]
