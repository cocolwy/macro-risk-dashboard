"""
Quant Multi-Agent System — 主入口

用法:
    python run.py                  # 运行流水线 + 启动前端
    python run.py --no-frontend    # 只运行流水线
    python run.py --frontend-only  # 只启动前端（查看历史结果）
"""

import os
import sys
import json
import time
import argparse
import threading
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent
WORKSPACE = ROOT / "workspace"
STATE_FILE = WORKSPACE / "state.json"

os.environ["QUAN_WORKSPACE"] = str(WORKSPACE)


def save_state(bus):
    """将 MessageBus 状态写入 JSON，供前端读取"""
    snapshot = bus.snapshot()
    # 附加详细结果
    snapshot["review"] = bus.get("review")
    snapshot["risk_report"] = bus.get("risk_report")
    snapshot["final_report"] = bus.get("final_report")
    snapshot["data_quality"] = bus.get("data_quality")
    snapshot["factor_stats"] = bus.get("factor_stats")
    snapshot["backtest_results"] = bus.get("backtest_results")
    snapshot["pca_info"] = bus.get("pca_info")

    # 清理不可序列化的内容
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, float):
            if obj != obj:  # NaN
                return None
            return obj
        return obj

    STATE_FILE.write_text(json.dumps(clean(snapshot), ensure_ascii=False, indent=2, default=str))


def run_pipeline(bus):
    """运行 Agent 流水线"""
    from agents import Orchestrator

    orch = Orchestrator(bus)

    # 每次状态变化时自动保存
    bus.subscribe(lambda: save_state(bus))

    print("\n" + "=" * 60)
    print("  Quant Multi-Agent Pipeline")
    print("=" * 60 + "\n")

    results = orch.run()

    # 最终保存
    save_state(bus)

    # 打印摘要
    summary = orch.get_summary()
    print("\n" + "=" * 60)
    print("  Pipeline Summary")
    print("=" * 60)
    for agent, status in summary["agents"].items():
        icon = "✓" if status == "success" else "✗" if status == "failed" else "○"
        print(f"  {icon} {agent}: {status}")

    final = summary.get("final_report")
    if final and "performance" in final:
        p = final["performance"]
        print(f"\n  年化收益(净): {p['组合年化收益(净)']:.2%}")
        print(f"  年化收益(毛): {p['组合年化收益(毛)']:.2%}")
        print(f"  超额收益(净): {p['超额收益(净)']:.2%}")
        print(f"  Sharpe:       {p['组合Sharpe']}")
        print(f"  最大回撤:     {p['最大回撤']:.2%}")
        print(f"  月均换手率:   {p['平均月换手率']:.1%}")
        print(f"  年化成本:     {p['年化成本拖累']:.2%}")

    review = summary.get("review")
    if review:
        print(f"\n  Critic: {review['verdict']} ({len(review.get('warnings',[]))} warnings)")

    risk = summary.get("risk_report")
    if risk:
        print(f"  Risk:   {risk['verdict']}")

    print()
    return results


def run_frontend():
    """启动前端服务器"""
    from frontend.app import start_server
    print("[Frontend] Dashboard: http://localhost:8765")
    start_server(host="0.0.0.0", port=8765)


def main():
    parser = argparse.ArgumentParser(description="Quant Multi-Agent System")
    parser.add_argument("--no-frontend", action="store_true", help="不启动前端")
    parser.add_argument("--frontend-only", action="store_true", help="只启动前端")
    parser.add_argument("--port", type=int, default=8765, help="前端端口")
    args = parser.parse_args()

    WORKSPACE.mkdir(parents=True, exist_ok=True)

    if args.frontend_only:
        run_frontend()
        return

    from agents.base import MessageBus
    bus = MessageBus(workspace=str(WORKSPACE))

    if args.no_frontend:
        run_pipeline(bus)
    else:
        # 先启动前端（后台线程），再运行流水线
        frontend_thread = threading.Thread(target=run_frontend, daemon=True)
        frontend_thread.start()
        time.sleep(1)  # 等待服务器启动

        print("[INFO] 打开浏览器访问 http://localhost:8765 查看实时进展\n")
        run_pipeline(bus)

        print("[INFO] 流水线完成，前端仍在运行，Ctrl+C 退出")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[INFO] 退出")


if __name__ == "__main__":
    main()
