"""
ModelTrainerAgent — 训练多个模型变体 + A/B 对比。

从 Bus 读取: prep_data, sp500
输出到 Bus:
  models       dict {name: (model, scaler)}
  experiments  list[dict]  每个模型的完整指标 (AUC, threshold, events, timeline)
  metrics      dict  主模型的完整指标 (model_info, current_prediction, etc.)
"""
import sys
from pathlib import Path

from ._base import BaseAgent

_DASHBOARD_DIR = str(Path(__file__).resolve().parent.parent)
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)
from predict_model import (
    train_and_evaluate, human_model_probs, build_metrics,
    build_comparison_metrics, HUMAN_WEIGHTS, KEY_EVENTS,
)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


class ModelTrainerAgent(BaseAgent):
    name = "model_trainer"
    role = "模型训练师"
    color = "#FF9800"

    def execute(self, embargo=20, split_ratio=0.7):
        prep_data = self.bus.get("prep_data")
        sp500 = self.bus.get("sp500")
        if prep_data is None:
            raise ValueError("prep_data 未就绪, 请先运行 FeatureAgent")

        models = {}
        experiments = []

        # --- 1. 主模型 (full features) ---
        if "full" in prep_data:
            self.log("训练 ML 主模型", "Logistic Regression, full features")
            X, y = prep_data["full"]
            model, scaler, X_train, X_test, y_train, y_test, y_prob = \
                train_and_evaluate(X, y, split_ratio=split_ratio)

            models["ml_full"] = (model, scaler)
            metrics, _ = build_metrics(model, scaler, X, y, X_train, X_test, y_test, y_prob, sp500)
            self.bus.put("metrics", metrics)

            ml_exp = {
                "name": "ML (Logistic Regression)",
                "auc": metrics["model_info"]["roc_auc"],
                "current_probability": metrics["current_prediction"]["probability"],
                "current_signal": metrics["current_prediction"]["signal"],
                "threshold_analysis": metrics["threshold_analysis"],
                "events_backtest": metrics["events_backtest"],
                "probability_timeline": metrics["probability_timeline"],
            }
            experiments.append(ml_exp)
            self.log("ML 主模型", f"AUC={metrics['model_info']['roc_auc']}", status="success")

            # --- Human Logic model ---
            self.log("训练 Human Logic", "手工权重模型")
            split = int(len(X) * split_ratio)
            human_probs_all = human_model_probs(X, scaler, model, y_train)
            human_probs_test = human_probs_all[split:]
            human_exp = build_comparison_metrics(
                y_test, human_probs_test, human_probs_all,
                X, sp500, "Human Logic v1", KEY_EVENTS,
            )
            experiments.append(human_exp)
            self.log("Human Logic", f"AUC={human_exp['auc']}", status="success")

            # --- Weight comparison ---
            ml_coefs = pd.Series(model.coef_[0], index=X.columns)
            weight_comparison = []
            for col in X.columns:
                mc = float(ml_coefs[col])
                hc = HUMAN_WEIGHTS.get(col, 0.0)
                agree = "same" if (mc > 0.01 and hc > 0) or (mc < -0.01 and hc < 0) else ("zero" if abs(mc) < 0.01 else "diff")
                weight_comparison.append({"feature": col, "ml_weight": round(mc, 4), "human_weight": hc, "agree": agree})
            self.bus.put("weight_comparison", weight_comparison)

        # --- 2. Slim model (deduplicated features) ---
        if "slim" in prep_data:
            self.log("训练 Slim 模型", "去冗余特征")
            X_slim, y_slim = prep_data["slim"]
            model_s, scaler_s, X_tr_s, X_te_s, y_tr_s, y_te_s, y_prob_s = \
                train_and_evaluate(X_slim, y_slim, split_ratio=split_ratio)
            models["ml_slim"] = (model_s, scaler_s)

            slim_probs_all = model_s.predict_proba(scaler_s.transform(X_slim))[:, 1]
            slim_exp = build_comparison_metrics(
                y_te_s, y_prob_s, slim_probs_all,
                X_slim, sp500, f"ML Slim ({len(X_slim.columns)}feat)", KEY_EVENTS,
            )
            experiments.append(slim_exp)
            self.log("Slim 模型", f"AUC={slim_exp['auc']}", status="success")

            # --- 3. D1: Slim + Embargo ---
            self.log("训练 D1", f"Slim + Embargo({embargo}d)")
            model_d1, scaler_d1, X_tr_d1, X_te_d1, y_tr_d1, y_te_d1, y_prob_d1 = \
                train_and_evaluate(X_slim, y_slim, split_ratio=split_ratio, embargo=embargo)
            models["d1_embargo"] = (model_d1, scaler_d1)

            d1_probs_all = model_d1.predict_proba(scaler_d1.transform(X_slim))[:, 1]
            d1_exp = build_comparison_metrics(
                y_te_d1, y_prob_d1, d1_probs_all,
                X_slim, sp500, f"D1 Slim+Embargo({embargo}d)", KEY_EVENTS,
            )
            experiments.append(d1_exp)
            self.log("D1 Embargo", f"AUC={d1_exp['auc']}", status="success")

            # --- 4. AND Ensemble (D1 x Human) ---
            if "full" in prep_data:
                self.log("构建 AND 集成", "D1 x Human Logic")
                X_full, _ = prep_data["full"]
                full_model, full_scaler = models["ml_full"]
                y_full = prep_data["full"][1]

                human_all = human_model_probs(X_full, full_scaler, full_model,
                                              y_full.iloc[:int(len(X_full) * split_ratio)])

                d1_series = pd.Series(d1_probs_all, index=X_slim.index)
                human_series = pd.Series(human_all, index=X_full.index)
                common = d1_series.index.intersection(human_series.index)

                d1_test_start = min(int(len(X_slim) * split_ratio) + embargo, len(X_slim))
                test_dates = X_slim.index[d1_test_start:]
                test_mask = common.isin(test_dates)
                y_and_test = y_slim.reindex(common)[test_mask].values
                ref_X = pd.DataFrame(index=common)

                min_probs = np.minimum(d1_series[common].values, human_series[common].values)
                min_exp = build_comparison_metrics(
                    pd.Series(y_and_test), min_probs[test_mask], min_probs,
                    ref_X, sp500, "MIN (D1, Human)", KEY_EVENTS,
                )
                experiments.append(min_exp)
                self.log("MIN 集成", f"AUC={min_exp['auc']}", status="success")

        self.bus.put("models", models)
        self.bus.put("experiments", experiments)

        self.log("训练完成", f"共 {len(experiments)} 个模型变体", status="success")
        return {"n_models": len(models), "n_experiments": len(experiments)}
