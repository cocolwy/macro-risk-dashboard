"""
风控 Agent 流水线入口。

用法:
  python -m dashboard.risk_agents              # 完整运行 (联网下载数据)
  python -m dashboard.risk_agents --quick      # 快速模式 (用现有 JSON)
  python -m dashboard.risk_agents --strict     # 严格模式 (Critic打回则停止)
"""
import argparse
from .orchestrator import RiskOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Risk Model Agent Pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="使用已有 JSON 数据, 跳过网络下载")
    parser.add_argument("--strict", action="store_true",
                        help="ModelCritic 打回时停止流水线")
    args = parser.parse_args()

    orch = RiskOrchestrator()
    orch.run(use_cached=args.quick, stop_on_reject=args.strict)
    orch.print_summary()


if __name__ == "__main__":
    main()
