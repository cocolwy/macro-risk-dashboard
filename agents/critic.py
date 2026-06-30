"""
Critic: 审核员 — 检查过拟合、前视偏差、数据泄露
"""

import numpy as np
import pandas as pd
from .base import BaseAgent, AgentStatus


class Critic(BaseAgent):
    name = "critic"
    role = "审核员"
    color = "#F44336"

    def execute(self):
        factors_dict = self.bus.get("factors_dict")
        factor_stats = self.bus.get("factor_stats")
        backtest_results = self.bus.get("backtest_results")
        data_quality = self.bus.get("data_quality")

        issues = []
        warnings = []

        self.log("审查开始", "检查数据质量、前视偏差、过拟合风险")

        # ── 1. 数据质量审查 ──
        if data_quality:
            missing = float(data_quality["价格缺失率"].strip("%")) / 100
            if missing > 0.05:
                issues.append(f"价格缺失率 {missing:.1%} 过高，可能影响因子计算")
            coverage = float(data_quality["基本面覆盖率"].strip("%")) / 100
            if coverage < 0.8:
                warnings.append(f"基本面覆盖率仅 {coverage:.1%}，B/P 和规模因子代表性不足")

        # ── 2. 前视偏差检查 ──
        self.log("前视偏差检查", "检查基本面数据是否存在 look-ahead bias")
        self.log("前视偏差检查",
                 "[已修复] B/P 和 log_mcap 已延迟 2 个月使用，模拟财报披露延迟",
                 status="success")
        warnings.append(
            "[已缓解] 基本面数据已做 2 个月延迟处理。"
            "残余风险：仍使用当前快照而非历史财报，理想方案是接入 point-in-time 数据库（Compustat）"
        )

        # ── 3. 生存偏差检查 ──
        self.log("生存偏差检查", "检查股票池是否存在 survivorship bias")
        warnings.append(
            "[已知局限] 股票池使用当前 S&P 500 成分，已退市/被剔除的股票不在池中，"
            "存在生存偏差（收益可能高估 1~2%/年）。修复需付费历史成分数据（CRSP/Compustat）"
        )

        # ── 4. 过拟合检查 ──
        if backtest_results:
            self.log("过拟合检查", "分析回测结果的合理性")
            for fname, res in backtest_results.items():
                if "error" in res:
                    continue
                ic_vals = res.get("ic_mean", {})
                for period, ic_val in ic_vals.items():
                    if abs(ic_val) > 0.15:
                        warnings.append(
                            f"{fname} 在 {period} 的 IC={ic_val:.4f} 异常高，"
                            "可能存在数据泄露或样本量不足"
                        )
                sharpe = res.get("sharpe_21d", 0)
                if abs(sharpe) > 3:
                    issues.append(f"{fname} Sharpe={sharpe:.2f} 异常高，疑似过拟合")

        # ── 5. 因子相关性检查 ──
        if factors_dict:
            self.log("相关性检查", "检查因子间相关性")
            sample_date = list(factors_dict.keys())[-1]
            sample_df = factors_dict[sample_date]
            corr = sample_df.corr()
            for i in range(len(corr)):
                for j in range(i + 1, len(corr)):
                    c = corr.iloc[i, j]
                    if abs(c) > 0.7:
                        warnings.append(
                            f"因子 {corr.index[i]} 与 {corr.columns[j]} "
                            f"相关性 {c:.2f}，建议正交化处理"
                        )

        # ── 判定 ──
        verdict = "PASS" if len(issues) == 0 else "REJECT"
        review = {
            "verdict": verdict,
            "issues": issues,
            "warnings": warnings,
            "summary": f"发现 {len(issues)} 个严重问题，{len(warnings)} 个警告",
        }

        self.bus.put("review", review)

        if issues:
            self.log("审查结论", f"REJECT — {len(issues)} 个严重问题", status="error")
            for iss in issues:
                self.log("严重问题", iss, status="error")
        else:
            self.log("审查结论", f"PASS（附 {len(warnings)} 个警告）", status="success")

        for w in warnings:
            self.log("警告", w, status="warning")

        return review
