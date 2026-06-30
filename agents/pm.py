"""
PortfolioManager: 组合经理 — 多因子加权、组合构建
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from .base import BaseAgent


class PortfolioManager(BaseAgent):
    name = "pm"
    role = "组合经理"
    color = "#E91E63"

    def execute(self, ic_window=6):
        """
        ic_window: 滚动 IC 窗口（月数），用过去 N 个月的 IC 均值作为当月权重
        """
        close = self.bus.get("close")
        factors_dict = self.bus.get("factors_dict")
        factor_names = self.bus.get("factor_names")
        review = self.bus.get("review")
        risk_report = self.bus.get("risk_report")
        analysis_start = self.bus.get("analysis_start", "2022-01-01")

        if factors_dict is None:
            raise ValueError("因子数据未就绪")

        self.log("组合构建",
                 f"滚动 {ic_window} 个月 IC 加权 → 动态权重 → 多因子组合")

        output_dir = str(self.bus.workspace / "output")
        os.makedirs(output_dir, exist_ok=True)

        # ── 1. 计算每个截面的 rank IC ──
        #   IC_t = spearman_corr(因子值_t, 下期收益_t→t+1)
        self.log("滚动IC", "逐月计算每个因子的截面 Rank IC")

        dates = sorted(factors_dict.keys())
        monthly_close = close.resample("ME").last()
        fwd_ret = monthly_close.pct_change().shift(-1)  # 下个月收益

        # ic_history[factor_name] = Series(index=date, value=IC)
        ic_history = {fn: {} for fn in factor_names}

        for date in dates:
            if date not in fwd_ret.index:
                continue
            df = factors_dict[date]
            ret_next = fwd_ret.loc[date]

            for fn in factor_names:
                factor_vals = df[fn].dropna()
                common = factor_vals.index.intersection(ret_next.dropna().index)
                if len(common) < 10:
                    continue
                # Spearman rank correlation
                ic_val = factor_vals[common].corr(ret_next[common], method="spearman")
                if not np.isnan(ic_val):
                    ic_history[fn][date] = ic_val

        # 转为 DataFrame: columns=因子, index=日期, values=IC
        ic_df = pd.DataFrame(ic_history)
        ic_df = ic_df.sort_index()
        self.log("IC计算完成", f"共 {len(ic_df)} 期, 因子: {list(ic_df.columns)}")

        # ── 2. 滚动窗口 IC → 动态权重 ──
        self.log("动态权重", f"每月用过去 {ic_window} 个月 IC 均值确定权重")

        rolling_ic = ic_df.rolling(window=ic_window, min_periods=3).mean()
        weight_history = {}   # date → {factor: weight}

        portfolio_scores = {}
        for date in dates:
            if date not in rolling_ic.index:
                continue
            row = rolling_ic.loc[date].dropna()
            if len(row) == 0:
                continue

            # 归一化：按绝对值归一，保留符号
            total = row.abs().sum()
            if total == 0:
                continue
            weights = (row / total).to_dict()
            weight_history[date] = weights

            # 综合打分
            df = factors_dict[date]
            z = df.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
            composite = sum(weights.get(fn, 0) * z[fn] for fn in factor_names)
            portfolio_scores[date] = composite.sort_values(ascending=False)

        self.log("权重示例",
                 f"首月: {self._fmt_weights(weight_history, 0)} | "
                 f"末月: {self._fmt_weights(weight_history, -1)}")

        # ── 3. 构建 Top/Bottom 组合（含交易成本）──
        ONE_WAY_COST = 0.001   # 单边交易成本 0.1%
        self.log("组合回测",
                 f"Top20 多头组合，单边交易成本 {ONE_WAY_COST:.1%}")

        prices = close[close.index >= analysis_start]
        ret_daily = prices.pct_change()

        top_n = 20
        portfolio_returns = []
        portfolio_returns_gross = []
        benchmark_returns = []
        turnover_list = []
        prev_holdings = set()

        dates_scored = sorted(portfolio_scores.keys())
        for i, date in enumerate(dates_scored[:-1]):
            next_date = dates_scored[i + 1]
            top_stocks = portfolio_scores[date].head(top_n).index.tolist()
            valid = [s for s in top_stocks if s in ret_daily.columns]
            curr_holdings = set(valid)

            # 换手率 = 新进 + 退出的股票数 / (2 × 持仓数)
            if prev_holdings:
                changed = len(curr_holdings - prev_holdings) + len(prev_holdings - curr_holdings)
                turnover = changed / (2 * max(len(curr_holdings), 1))
            else:
                turnover = 1.0  # 首期全部建仓
            turnover_list.append(turnover)
            prev_holdings = curr_holdings

            # 交易成本：换手部分双边扣费
            trade_cost = turnover * 2 * ONE_WAY_COST

            # 期间日收益
            mask = (ret_daily.index > date) & (ret_daily.index <= next_date)
            period_ret = ret_daily.loc[mask]

            if len(valid) > 0 and len(period_ret) > 0:
                port_ret_gross = period_ret[valid].mean(axis=1)
                bench_ret = period_ret.mean(axis=1)

                # 在调仓日（期间第一天）扣除交易成本
                port_ret_net = port_ret_gross.copy()
                port_ret_net.iloc[0] -= trade_cost

                portfolio_returns.append(port_ret_net)
                portfolio_returns_gross.append(port_ret_gross)
                benchmark_returns.append(bench_ret)

        if not portfolio_returns:
            self.log("组合构建失败", "无有效收益数据", status="error")
            return {"error": "无有效收益数据"}

        port_ret_series = pd.concat(portfolio_returns)
        port_ret_gross_series = pd.concat(portfolio_returns_gross)
        bench_ret_series = pd.concat(benchmark_returns)

        cum_port = (1 + port_ret_series).cumprod()
        cum_port_gross = (1 + port_ret_gross_series).cumprod()
        cum_bench = (1 + bench_ret_series).cumprod()

        avg_turnover = np.mean(turnover_list)

        # ── 4. 绩效指标 ──
        ann_ret_port = port_ret_series.mean() * 252
        ann_ret_gross = port_ret_gross_series.mean() * 252
        ann_vol_port = port_ret_series.std() * np.sqrt(252)
        sharpe_port = ann_ret_port / ann_vol_port if ann_vol_port > 0 else 0

        ann_ret_bench = bench_ret_series.mean() * 252
        excess_ret = ann_ret_port - ann_ret_bench

        # 年化成本拖累 = 毛收益 - 净收益（同口径）
        ann_cost_drag = ann_ret_gross - ann_ret_port
        self.log("交易成本",
                 f"平均月换手率 {avg_turnover:.1%}, "
                 f"年化成本拖累 {ann_cost_drag:.2%}",
                 status="info")

        # 最大回撤
        running_max = cum_port.cummax()
        dd = (cum_port - running_max) / running_max
        max_dd = dd.min()

        # 最终权重（末月）
        final_weights = weight_history[dates_scored[-1]] if dates_scored else {}
        performance = {
            "组合年化收益(净)": round(ann_ret_port, 4),
            "组合年化收益(毛)": round(ann_ret_gross, 4),
            "基准年化收益": round(ann_ret_bench, 4),
            "超额收益(净)": round(excess_ret, 4),
            "组合Sharpe": round(sharpe_port, 2),
            "最大回撤": round(max_dd, 4),
            "平均月换手率": round(avg_turnover, 4),
            "年化成本拖累": round(ann_cost_drag, 4),
            "持仓数": top_n,
            "IC窗口": ic_window,
            "末月权重": {k: round(v, 3) for k, v in final_weights.items()},
        }
        self.log("绩效汇总", str(performance), status="success")

        # ── 5. 绘图（3×2 = 6 子图）──
        fig, axes = plt.subplots(3, 2, figsize=(18, 20))
        fig.suptitle(f"Portfolio Manager Report (Rolling {ic_window}M IC Weighting)",
                     fontsize=16, fontweight="bold")

        # [0,0] 累计收益对比（含 gross vs net）
        ax = axes[0, 0]
        cum_port.plot(ax=ax, label=f"Net (cost {ONE_WAY_COST:.1%}/side)", color="steelblue", lw=2)
        cum_port_gross.plot(ax=ax, label="Gross (no cost)", color="steelblue", lw=1, ls=":", alpha=0.6)
        cum_bench.plot(ax=ax, label="Equal-Weight Benchmark", color="gray", lw=1.5, ls="--")
        ax.set_title("Cumulative Returns (Gross vs Net vs Benchmark)")
        ax.legend(fontsize=9)

        # [0,1] 滚动 IC 时序（每个因子一条线）
        ax = axes[0, 1]
        rolling_ic.plot(ax=ax, alpha=0.8, lw=1.2)
        ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.set_title(f"Rolling {ic_window}M Mean IC per Factor")
        ax.legend(fontsize=8)

        # [1,0] 动态权重面积图
        ax = axes[1, 0]
        weight_df = pd.DataFrame(weight_history).T.sort_index()
        weight_df.plot.area(ax=ax, alpha=0.7, stacked=False)
        ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.set_title("Dynamic Factor Weights Over Time")
        ax.legend(fontsize=8)

        # [1,1] 末月权重饼图
        ax = axes[1, 1]
        if final_weights:
            abs_w = {k: abs(v) for k, v in final_weights.items() if abs(v) > 0.01}
            colors_pie = plt.cm.Set3.colors[:len(abs_w)]
            wedges, texts, autotexts = ax.pie(
                abs_w.values(), labels=abs_w.keys(), autopct='%1.1f%%',
                colors=colors_pie)
            # 标注负权重
            for k, v in final_weights.items():
                if v < 0 and k in abs_w:
                    ax.set_title(f"Final Month Weights (neg = short signal)")
                    break
            else:
                ax.set_title("Final Month Weight Allocation")
        else:
            ax.set_title("No weights available")

        # [2,0] 超额收益
        ax = axes[2, 0]
        excess = port_ret_series - bench_ret_series
        cum_excess = (1 + excess).cumprod()
        cum_excess.plot(ax=ax, color="green", lw=1.5)
        ax.axhline(1, color="k", ls="--", lw=0.5)
        ax.set_title("Cumulative Excess Return")

        # [2,1] 回撤
        ax = axes[2, 1]
        dd.plot(ax=ax, color="crimson", alpha=0.8)
        ax.fill_between(dd.index, dd.values, alpha=0.3, color="crimson")
        ax.set_title("Portfolio Drawdown")

        plt.tight_layout()
        chart_path = os.path.join(output_dir, "portfolio_report.png")
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.log("组合报告图表", f"已保存: {chart_path}", status="success")

        # ── 6. 汇总 ──
        # 审核意见和风险提示
        if review:
            self.log("审核反馈", f"Critic 判定: {review['verdict']}, "
                     f"{len(review.get('warnings', []))} 个警告")
        if risk_report:
            self.log("风控反馈", f"RiskManager 判定: {risk_report['verdict']}")

        final_report = {
            "performance": performance,
            "chart": chart_path,
            "review_verdict": review.get("verdict") if review else "N/A",
            "risk_verdict": risk_report.get("verdict") if risk_report else "N/A",
        }

        # ── 暴露中间序列（供 golden baseline / Qlib 对比使用，非侵入式）──
        self.bus.put("port_ret_net", port_ret_series)
        self.bus.put("port_ret_gross", port_ret_gross_series)
        self.bus.put("bench_ret", bench_ret_series)
        self.bus.put("weight_history", weight_history)
        self.bus.put("holdings_by_date", {
            str(d.date()): portfolio_scores[d].head(top_n).index.tolist()
            for d in dates_scored
        })

        self.bus.put("final_report", final_report)
        return final_report

    @staticmethod
    def _fmt_weights(weight_history, idx):
        """格式化权重字典用于日志"""
        keys = sorted(weight_history.keys())
        if not keys:
            return "{}"
        w = weight_history[keys[idx]]
        return "{" + ", ".join(f"{k}:{v:+.2f}" for k, v in w.items()) + "}"
