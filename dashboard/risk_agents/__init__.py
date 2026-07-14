"""
Risk Model Agent Pipeline — 复用 quant multi-agent 基础设施的风控流水线。

Pipeline:
  MacroDataAgent → FeatureAgent → ModelTrainerAgent → ModelCriticAgent → ScoreAgent

用法:
  python -m dashboard.risk_agents            # 运行完整流水线
  python -m dashboard.risk_agents --quick     # 跳过数据下载, 用现有 JSON
"""
from .data_agent import MacroDataAgent
from .feature_agent import FeatureAgent
from .model_agent import ModelTrainerAgent
from .critic_agent import ModelCriticAgent
from .score_agent import ScoreAgent
from .orchestrator import RiskOrchestrator

__all__ = [
    "MacroDataAgent", "FeatureAgent", "ModelTrainerAgent",
    "ModelCriticAgent", "ScoreAgent", "RiskOrchestrator",
]
