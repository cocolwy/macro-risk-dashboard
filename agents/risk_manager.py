"""
RiskManager: 风控 — 压力测试、回撤分析、极端情景
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from .base import BaseAgent


class RiskManager(BaseAgent):
    name = "risk_manager"
    role = "风控经理"
    color = "#9C27B0"

    def execute(self):
        close = self.bus.get("close")
        backtest_results = self.bus.get("backtest_results")
        factors_dict = self.bus.get("factors_dict")
        analysis_start = self.bus.get("analysis_start", "2022-01-01")

        if close is None:
            raise ValueError("价格数据未就绪")

        self.log("风控分析", "开始压力测试和风险评估")
        output_dir = str(self.bus.workspace / "output")
        os.makedirs(output_dir, exist_ok=True)

        prices = close[close.index >= analysis_start]
        ret_daily = prices.pct_change().dropna()

        # ── 1. 市场层面风险 ──
        self.log("市场风险", "分析等权组合的风险特征")
        eq_ret = ret_daily.mean(axis=1)
        cum_eq = (1 + eq_ret).cumprod()
        running_max = cum_eq.cummax()
        drawdown = (cum_eq - running_max) / running_max

        market_risk = {
            "年化波动率": round(eq_ret.std() * np.sqrt(252), 4),
            "最大回撤": round(drawdown.min(), 4),
            "最大回撤日期": str(drawdown.idxmin().date()),
            "VaR_95": round(eq_ret.quantile(0.05), 4),
            "CVaR_95": round(eq_ret[eq_ret <= eq_ret.quantile(0.05)].mean(), 4),
            "偏度": round(eq_ret.skew(), 4),
            "峰度": round(eq_ret.kurtosis(), 4),
        }
        self.log("市场风险指标", str(market_risk))

        # ── 2. 压力测试：极端行情时因子表现 ──
        self.log("压力测试", "分析极端行情下因子表现")
        stress_results = {}
        worst_days = eq_ret.nsmallest(10)

        if factors_dict:
            # 找到极端月份
            monthly_ret = eq_ret.resample("ME").sum()
            worst_months = monthly_ret.nsmallest(5)
            best_months = monthly_ret.nlargest(5)

            stress_results["最差5个月"] = {
                str(d.date()): round(v, 4) for d, v in worst_months.items()
            }
            stress_results["最好5个月"] = {
                str(d.date()): round(v, 4) for d, v in best_months.items()
            }

        # ── 3. 尾部风险分析图 ──
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Risk Dashboard", fontsize=16, fontweight="bold")

        # 回撤曲线
        ax = axes[0, 0]
        drawdown.plot(ax=ax, color="crimson", alpha=0.8)
        ax.fill_between(drawdown.index, drawdown.values, alpha=0.3, color="crimson")
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown")

        # 收益分布
        ax = axes[0, 1]
        eq_ret.hist(bins=80, ax=ax, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(eq_ret.quantile(0.05), color="red", ls="--", label="VaR 5%")
        ax.set_title("Daily Return Distribution")
        ax.legend()

        # 滚动波动率
        ax = axes[1, 0]
        rolling_vol = eq_ret.rolling(21).std() * np.sqrt(252)
        rolling_vol.plot(ax=ax, color="darkorange")
        ax.set_title("Rolling 21D Annualized Volatility")

        # 累计收益
        ax = axes[1, 1]
        cum_eq.plot(ax=ax, color="steelblue")
        ax.set_title("Equal-Weight Portfolio Cumulative Return")

        plt.tight_layout()
        risk_chart = os.path.join(output_dir, "risk_dashboard.png")
        fig.savefig(risk_chart, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.log("风控图表", f"已保存: {risk_chart}", status="success")

        # ── 4. 风险判定 ──
        risk_alerts = []
        if abs(market_risk["最大回撤"]) > 0.3:
            risk_alerts.append(f"最大回撤 {market_risk['最大回撤']:.1%} 超过 30%，风险较高")
        if market_risk["峰度"] > 5:
            risk_alerts.append(f"收益分布峰度 {market_risk['峰度']:.1f}，存在显著尾部风险")

        risk_report = {
            "market_risk": market_risk,
            "stress": stress_results,
            "alerts": risk_alerts,
            "chart": risk_chart,
            "verdict": "ALERT" if risk_alerts else "OK",
        }

        self.bus.put("risk_report", risk_report)

        if risk_alerts:
            for alert in risk_alerts:
                self.log("风险警报", alert, status="warning")
        else:
            self.log("风险评估", "各项指标正常", status="success")

        return risk_report
