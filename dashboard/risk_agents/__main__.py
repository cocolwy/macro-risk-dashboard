"""
风控 Agent 流水线入口。

用法:
  python -m dashboard.risk_agents                            # 完整运行
  python -m dashboard.risk_agents --quick                    # 用缓存数据
  python -m dashboard.risk_agents --indicators vix sp500     # 只跑 2 个指标
  python -m dashboard.risk_agents --features slim            # 只用 slim 特征
  python -m dashboard.risk_agents --list                     # 列出已注册的指标和特征
"""
import argparse
from .orchestrator import RiskOrchestrator
from .registry import INDICATOR, FEATURE_BUILDER


def main():
    parser = argparse.ArgumentParser(description="Risk Model Agent Pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="使用已有 JSON 数据, 跳过网络下载")
    parser.add_argument("--strict", action="store_true",
                        help="ModelCritic 打回时停止流水线")
    parser.add_argument("--indicators", nargs="*", default=None,
                        help="只获取指定指标 (空格分隔)")
    parser.add_argument("--features", nargs="*", default=None,
                        help="只构建指定特征集 (空格分隔)")
    parser.add_argument("--embargo", type=int, default=None,
                        help="训练测试间隔天数")
    parser.add_argument("--list", action="store_true",
                        help="列出已注册的指标和特征构建器")
    args = parser.parse_args()

    if args.list:
        print(f"Registered Indicators ({len(INDICATOR)}):")
        for k in INDICATOR.keys():
            meta = INDICATOR.get_meta(k)
            req = " [REQUIRED]" if meta.get("required") else ""
            print(f"  - {k}{req}")
        print(f"\nRegistered Feature Builders ({len(FEATURE_BUILDER)}):")
        for k in FEATURE_BUILDER.keys():
            print(f"  - {k}")
        return

    config = {}
    if args.indicators:
        config["indicators"] = args.indicators
    if args.features:
        config["features"] = args.features
    if args.embargo is not None:
        config["embargo"] = args.embargo

    orch = RiskOrchestrator(config=config)
    orch.run(use_cached=args.quick, stop_on_reject=args.strict)
    orch.print_summary()


if __name__ == "__main__":
    main()
