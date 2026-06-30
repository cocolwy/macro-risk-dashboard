"""
Researcher: 因子研究员 — 计算五因子截面值，可选 PCA 正交化
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from .base import BaseAgent, MessageBus


class Researcher(BaseAgent):
    name = "researcher"
    role = "因子研究员"
    color = "#2196F3"

    def execute(self, analysis_start="2022-01-01", orthogonalize=False,
                industry_neutral=False):
        close = self.bus.get("close")
        fundamentals = self.bus.get("fundamentals")
        if close is None or fundamentals is None:
            raise ValueError("数据未就绪，请先运行 DataEngineer")

        self.log("计算因子", "开始计算五因子截面值")

        ret_daily = close.pct_change()
        monthly_close = close.resample("ME").last()

        # ── 动量 ret_2_12 ──
        self.log("动量因子", "计算 2~12 月累计收益")
        ret_12m = monthly_close.pct_change(12)
        ret_1m = monthly_close.pct_change(1)
        momentum = ret_12m - ret_1m

        # ── 反转 ──
        self.log("反转因子", "计算 -ret_1m")
        reversal = -ret_1m

        # ── 低波动 ──
        self.log("低波动因子", "计算 20 日滚动 std（取负）")
        vol_20d = ret_daily.rolling(20).std()
        vol_monthly = vol_20d.resample("ME").last()
        low_vol = -vol_monthly

        # ── 价值 & 规模（延迟 2 个月，修复前视偏差）──
        self.log("价值/规模因子",
                 "基本面数据延迟 2 个月使用（模拟财报披露延迟 ~60 天）")
        bp_static = fundamentals["bp_ratio"]
        mcap_static = fundamentals["market_cap"]

        # 将静态快照展开为月度 DataFrame，然后 shift(2)
        # 这样 T 月只能看到 T-2 月的基本面
        bp_monthly = pd.DataFrame(
            {t: bp_static.get(t, np.nan) for t in monthly_close.columns},
            index=monthly_close.index,
        ).shift(2)
        log_mcap_monthly = pd.DataFrame(
            {t: np.log(mcap_static.get(t, np.nan)) if mcap_static.get(t, 0) > 0 else np.nan
             for t in monthly_close.columns},
            index=monthly_close.index,
        ).shift(2)

        # ── 组装截面 ──
        dates = monthly_close.index[monthly_close.index >= analysis_start]
        factors_dict = {}

        for date in dates:
            if date not in momentum.index:
                continue
            rows = {}
            for ticker in monthly_close.columns:
                rows[ticker] = {
                    "momentum": momentum.at[date, ticker] if date in momentum.index else np.nan,
                    "reversal": reversal.at[date, ticker] if date in reversal.index else np.nan,
                    "low_vol": low_vol.at[date, ticker] if date in low_vol.index else np.nan,
                    "bp": bp_monthly.at[date, ticker] if date in bp_monthly.index else np.nan,
                    "log_mcap": log_mcap_monthly.at[date, ticker] if date in log_mcap_monthly.index else np.nan,
                }
            factors_dict[date] = pd.DataFrame(rows).T

        factor_names = ["momentum", "bp", "log_mcap", "low_vol", "reversal"]

        # ── 行业中性化 ──
        if industry_neutral:
            sector_map = self.bus.get("sector_map", {})
            if sector_map:
                self.log("行业中性化",
                         f"在 {len(set(sector_map.values()))} 个行业内做 Z-score 标准化")
                factors_dict = self._industry_neutralize(
                    factors_dict, factor_names, sector_map)
                self.log("行业中性化完成",
                         "因子值已转化为行业内相对排名",
                         status="success")
            else:
                self.log("行业中性化跳过", "无行业分类数据", status="warning")

        self.bus.put("industry_neutral", industry_neutral)

        # ── 相关矩阵（正交化前）──
        self.log("相关性分析", "计算因子间 Pearson 相关矩阵")
        corr_before = self._compute_avg_correlation(factors_dict, factor_names)
        self.bus.put("corr_before", corr_before)
        self._plot_corr_heatmap(corr_before, "Factor Correlation (Before Orthogonalization)",
                                "corr_before.png")
        self.log("相关矩阵", f"最大非对角相关系数: {self._max_offdiag(corr_before):.3f}")

        # ── PCA 正交化 ──
        if orthogonalize:
            self.log("PCA正交化", "对五因子做截面 PCA 正交化")
            factors_dict, pca_info = self._orthogonalize_pca(factors_dict, factor_names)
            factor_names = [f"PC{i+1}" for i in range(len(factor_names))]

            corr_after = self._compute_avg_correlation(factors_dict, factor_names)
            self.bus.put("corr_after", corr_after)
            self._plot_corr_heatmap(corr_after, "Factor Correlation (After Orthogonalization)",
                                    "corr_after.png")
            self.log("正交化完成",
                     f"最大非对角相关系数: {self._max_offdiag(corr_after):.3f}, "
                     f"方差解释比: {pca_info['explained_variance_ratio']}",
                     status="success")
            self.bus.put("pca_info", pca_info)
            self._plot_pca_loadings(pca_info)

        # ── 因子统计 ──
        factor_stats = {}
        for fn in factor_names:
            all_vals = pd.concat([df[fn] for df in factors_dict.values()])
            factor_stats[fn] = {
                "mean": round(all_vals.mean(), 4),
                "std": round(all_vals.std(), 4),
                "coverage": f"{all_vals.notna().mean():.1%}",
            }

        self.log("因子统计", str(factor_stats), status="success")

        self.bus.put("factors_dict", factors_dict)
        self.bus.put("factor_names", factor_names)
        self.bus.put("factor_stats", factor_stats)
        self.bus.put("analysis_start", analysis_start)
        self.bus.put("orthogonalized", orthogonalize)

        return {"截面月数": len(factors_dict), "因子": factor_names,
                "统计": factor_stats, "正交化": orthogonalize}

    # ── 辅助方法 ──

    def _compute_avg_correlation(self, factors_dict, factor_names):
        """计算所有截面的平均相关矩阵"""
        corrs = []
        for date, df in factors_dict.items():
            sub = df[factor_names].dropna()
            if len(sub) > 10:
                corrs.append(sub.corr())
        if corrs:
            return sum(corrs) / len(corrs)
        return pd.DataFrame()

    @staticmethod
    def _max_offdiag(corr):
        """相关矩阵中最大的非对角线绝对值"""
        if corr.empty:
            return 0.0
        mask = ~np.eye(len(corr), dtype=bool)
        return float(np.abs(corr.values[mask]).max())

    def _plot_corr_heatmap(self, corr, title, filename):
        """画相关系数热力图"""
        output_dir = str(self.bus.workspace / "output")
        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        plt.tight_layout()
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.log("热力图", f"已保存: {path}")

    def _orthogonalize_pca(self, factors_dict, factor_names):
        """
        截面 PCA 正交化：
        1. 用全样本拟合 PCA（学习旋转矩阵）
        2. 逐截面 transform → 得到正交主成分
        """
        # 收集全样本
        all_data = []
        for date, df in factors_dict.items():
            # fillna(0) 而非 dropna：中性化后 NaN 填 0 = 行业中等水平
            sub = df[factor_names].fillna(0)
            all_data.append(sub)
        full = pd.concat(all_data)

        # 标准化 + PCA
        mean = full.mean()
        std = full.std()
        std[std == 0] = 1

        pca = PCA(n_components=len(factor_names))
        pca.fit((full - mean) / std)

        # 逐截面 transform
        new_factors_dict = {}
        pc_names = [f"PC{i+1}" for i in range(len(factor_names))]
        for date, df in factors_dict.items():
            sub = df[factor_names].fillna(0)
            if len(sub) < 5:
                continue
            z = (sub - mean) / std
            transformed = pd.DataFrame(
                pca.transform(z),
                index=sub.index,
                columns=pc_names,
            )
            new_factors_dict[date] = transformed

        pca_info = {
            "explained_variance_ratio": [round(x, 3) for x in pca.explained_variance_ratio_],
            "cumulative_variance": [round(x, 3) for x in np.cumsum(pca.explained_variance_ratio_)],
            "components": pd.DataFrame(
                pca.components_,
                columns=factor_names,
                index=pc_names,
            ).round(3).to_dict(),
            "original_factors": factor_names,
        }

        return new_factors_dict, pca_info

    def _plot_pca_loadings(self, pca_info):
        """画 PCA 载荷图"""
        output_dir = str(self.bus.workspace / "output")
        components = pd.DataFrame(pca_info["components"])
        evr = pca_info["explained_variance_ratio"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 载荷热力图
        ax = axes[0]
        sns.heatmap(components, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, square=True, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.8})
        ax.set_title("PCA Loadings (Component × Original Factor)", fontsize=12, fontweight="bold")

        # 方差解释比
        ax = axes[1]
        cum = pca_info["cumulative_variance"]
        x = range(1, len(evr) + 1)
        ax.bar(x, evr, color="steelblue", alpha=0.7, label="Individual")
        ax.plot(x, cum, "o-", color="crimson", lw=2, label="Cumulative")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("PCA Explained Variance", fontsize=12, fontweight="bold")
        ax.legend()
        ax.set_xticks(list(x))

        plt.tight_layout()
        path = os.path.join(output_dir, "pca_loadings.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.log("PCA载荷图", f"已保存: {path}")

    def _industry_neutralize(self, factors_dict, factor_names, sector_map):
        """
        行业中性化：在每个截面日期，对每个因子按行业分组做 Z-score。

        效果：因子值从"全截面绝对水平"变成"行业内相对水平"。
        比如某银行股 B/P=0.8，在全截面排前列，但在金融行业内只是中等水平。
        中性化后它的 B/P 得分就是中等，而不是偏高。
        """
        new_dict = {}
        for date, df in factors_dict.items():
            # 给每只股票打上行业标签
            sectors = pd.Series(
                {t: sector_map.get(t, "Unknown") for t in df.index},
                name="sector"
            )
            df_with_sector = df[factor_names].copy()
            df_with_sector["sector"] = sectors

            # 按行业分组 Z-score
            def _zscore(x):
                s = x.std()
                if pd.isna(s) or s == 0 or len(x) < 3:
                    return pd.Series(np.nan, index=x.index)
                return (x - x.mean()) / s

            neutralized = df_with_sector.groupby("sector")[factor_names].transform(_zscore)

            new_dict[date] = neutralized

        return new_dict
