"""
Backtester: 回测工程师 — alphalens 因子分析 + 绩效指标
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import alphalens
from .base import BaseAgent, MessageBus


class Backtester(BaseAgent):
    name = "backtester"
    role = "回测工程师"
    color = "#FF9800"

    def execute(self, output_dir=None):
        close = self.bus.get("close")
        factors_dict = self.bus.get("factors_dict")
        factor_names = self.bus.get("factor_names")
        analysis_start = self.bus.get("analysis_start", "2022-01-01")

        if factors_dict is None:
            raise ValueError("因子数据未就绪，请先运行 Researcher")

        output_dir = output_dir or str(self.bus.workspace / "output")
        os.makedirs(output_dir, exist_ok=True)

        prices = close[close.index >= analysis_start]
        results = {}

        for fname in factor_names:
            self.log("回测因子", f"分析 {fname}")
            try:
                factor_series = self._format_factor(factors_dict, fname)
                factor_data = alphalens.utils.get_clean_factor_and_forward_returns(
                    factor_series, prices, quantiles=5, periods=(5, 10, 21),
                )

                # IC
                ic = alphalens.performance.factor_information_coefficient(factor_data)
                ic_mean = ic.mean()

                # 分层收益
                mean_ret_by_q = alphalens.performance.mean_return_by_quantile(
                    factor_data, by_group=False
                )[0]

                # 多空收益
                factor_returns = alphalens.performance.factor_returns(factor_data)
                cum_ret = (1 + factor_returns).cumprod()

                # 绘图
                fig_path = os.path.join(output_dir, f"{fname}_analysis.png")
                self._plot(fname, ic, mean_ret_by_q, cum_ret, fig_path)

                # Sharpe (年化，用 21D 列)
                col = factor_returns.columns[-1]
                sharpe = (factor_returns[col].mean() / factor_returns[col].std()) * np.sqrt(252 / 21)

                results[fname] = {
                    "ic_mean": {str(k): round(v, 4) for k, v in ic_mean.items()},
                    "sharpe_21d": round(sharpe, 2),
                    "cum_return": round(cum_ret.iloc[-1].iloc[-1] - 1, 4),
                    "chart": fig_path,
                }
                self.log(f"{fname} 结果",
                         f"IC={ic_mean.iloc[0]:.4f}, Sharpe={sharpe:.2f}",
                         status="success")

            except Exception as e:
                results[fname] = {"error": str(e)}
                self.log(f"{fname} 失败", str(e), status="error")

        self.bus.put("backtest_results", results)
        self.bus.put("output_dir", output_dir)
        return results

    def _format_factor(self, factors_dict, factor_name):
        pieces = []
        for date, df in factors_dict.items():
            s = df[factor_name].dropna()
            s.index = pd.MultiIndex.from_product(
                [[date], s.index], names=["date", "asset"]
            )
            pieces.append(s)
        factor = pd.concat(pieces)
        factor.name = factor_name
        return factor

    def _plot(self, fname, ic, mean_ret_by_q, cum_ret, path):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Factor: {fname}", fontsize=16, fontweight="bold")

        # IC 时序
        ax = axes[0, 0]
        ic.plot(ax=ax, alpha=0.7)
        ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.set_title("IC Time Series")
        ax.legend(fontsize=8)

        # 月度 IC
        ax = axes[0, 1]
        monthly_ic = ic.resample("ME").mean()
        monthly_ic.plot(kind="bar", ax=ax, width=0.8, alpha=0.7)
        ax.set_title("Monthly Mean IC")
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.legend(fontsize=8)

        # 分层收益
        ax = axes[1, 0]
        col = mean_ret_by_q.columns[0]
        mean_ret_by_q[col].plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"Mean Return by Quantile ({col})")
        ax.set_xlabel("Quantile")

        # 累计收益
        ax = axes[1, 1]
        cum_ret.plot(ax=ax)
        ax.set_title("Cumulative Factor Returns (Long-Short)")
        ax.legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
