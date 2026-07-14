"""
FeatureAgent — 特征工程 + 标签构建。

从 Bus 读取: macro_df
输出到 Bus:
  feature_sets   dict {name: pd.DataFrame}  (多种特征集)
  target         pd.Series (binary: 1=未来20日>5%回撤)
  sp500          pd.Series
  prep_data      dict {name: (X, y)}  已对齐、dropna 的训练就绪数据
"""
import sys
from pathlib import Path

from ._base import BaseAgent

_DASHBOARD_DIR = str(Path(__file__).resolve().parent.parent)
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)
from predict_model import (
    build_features, build_features_slim,
    compute_target,
)

import pandas as pd


class FeatureAgent(BaseAgent):
    name = "feature_eng"
    role = "特征工程师"
    color = "#2196F3"

    def execute(self, feature_variants=None):
        """
        Parameters
        ----------
        feature_variants : list[str]
            要构建的特征集, 默认 ["full", "slim"]
            可选: "full", "slim", "regime", "events"
        """
        macro_df = self.bus.get("macro_df")
        if macro_df is None:
            raise ValueError("macro_df 未就绪, 请先运行 MacroDataAgent")

        variants = feature_variants or ["full", "slim"]

        self.log("特征构建", f"共 {len(variants)} 套特征集: {variants}")

        builders = {
            "full": lambda df: build_features(df),
            "slim": lambda df: build_features_slim(df),
        }

        try:
            from predict_model import build_features_regime, fetch_regime_data
            builders["regime"] = lambda df: build_features_regime(df, fetch_regime_data())
        except ImportError:
            pass

        try:
            from predict_model import build_features_with_events
            builders["events"] = lambda df: build_features_with_events(df)
        except ImportError:
            pass

        feature_sets = {}
        for name in variants:
            if name not in builders:
                self.log(f"跳过 {name}", "未定义的特征集", status="warning")
                continue
            feats = builders[name](macro_df)
            feature_sets[name] = feats
            self.log(f"特征集 {name}", f"{len(feats.columns)} 个特征, {len(feats)} 行")

        self.bus.put("feature_sets", feature_sets)

        target = compute_target(macro_df["sp500"])
        self.bus.put("target", target)
        self.bus.put("sp500", macro_df["sp500"])

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
