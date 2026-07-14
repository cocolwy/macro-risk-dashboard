"""
FeatureAgent — 特征工程 + 标签构建, 特征构建器可插拔。

从 Bus 读取: macro_df
输出到 Bus:
  feature_sets   dict {name: pd.DataFrame}
  target         pd.Series
  sp500          pd.Series
  prep_data      dict {name: (X, y)}

默认使用 "full" 和 "slim" 两套特征, 但你可以:
  - 通过 feature_variants= 参数选择子集
  - 通过 registry.FEATURE_BUILDER.add() 注册自定义特征构建器
  - 通过 target_fn= 传入自定义标签函数
  - 通过 target_col= 指定用哪个列做标签基础 (默认 "sp500")
"""
import sys
from pathlib import Path

import pandas as pd

from ._base import BaseAgent
from .registry import FEATURE_BUILDER

_DASHBOARD_DIR = str(Path(__file__).resolve().parent.parent)
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)
from predict_model import compute_target


class FeatureAgent(BaseAgent):
    name = "feature_eng"
    role = "特征工程师"
    color = "#2196F3"

    def execute(self, feature_variants=None, target_fn=None, target_col="sp500"):
        """
        Parameters
        ----------
        feature_variants : list[str] | None
            要构建的特征集, None = 注册表中的全部
        target_fn : callable | None
            自定义标签函数 (接受 pd.Series, 返回 pd.Series), None = 默认 compute_target
        target_col : str
            用哪个列做标签计算基础, 默认 "sp500"
        """
        macro_df = self.bus.get("macro_df")
        if macro_df is None:
            raise ValueError("macro_df 未就绪, 请先运行 MacroDataAgent")

        variants = feature_variants or FEATURE_BUILDER.keys()
        self.log("特征构建", f"共 {len(variants)} 套特征集: {list(variants)}")

        feature_sets = {}
        for name in variants:
            builder = FEATURE_BUILDER.get(name)
            if builder is None:
                self.log(f"跳过 {name}", "未注册的特征构建器", status="warning")
                continue
            try:
                feats = builder(macro_df)
                feature_sets[name] = feats
                self.log(f"特征集 {name}", f"{len(feats.columns)} 个特征, {len(feats)} 行")
            except Exception as e:
                self.log(f"特征集 {name}", f"构建失败: {e}", status="error")

        self.bus.put("feature_sets", feature_sets)

        if target_col not in macro_df.columns:
            raise ValueError(f"target_col='{target_col}' 不在 macro_df 中, 可选: {list(macro_df.columns)}")

        target_func = target_fn or compute_target
        target = target_func(macro_df[target_col])
        self.bus.put("target", target)
        self.bus.put("sp500", macro_df[target_col])

        prep_data = {}
        for name, feats in feature_sets.items():
            combined = feats.copy()
            combined["target"] = target
            combined = combined.dropna()
            X = combined.drop("target", axis=1).clip(-10, 10)
            y = combined["target"]
            prep_data[name] = (X, y)
            pos_rate = y.mean() * 100
            self.log(f"数据集 {name}",
                     f"{len(X)} 样本, {int(y.sum())} 正样本 ({pos_rate:.1f}%)",
                     status="success")

        self.bus.put("prep_data", prep_data)
        return {
            "feature_sets": list(feature_sets.keys()),
            "target_positive_rate": round(float(target.mean()), 3),
        }
