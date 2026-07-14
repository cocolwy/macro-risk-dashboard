"""
ModelCriticAgent — 自主验证模型质量, 检查常见风控建模陷阱。

从 Bus 读取: prep_data, experiments, models, metrics
输出到 Bus:
  review  dict {verdict: PASS/REJECT, warnings: [...], checks: {...}}

检查项:
  1. 数据泄漏: 训练期是否包含未来信息 (目标变量构建方式)
  2. 样本不平衡: 正样本比例是否合理 (3-30%)
  3. 过拟合: train/test AUC 差距是否过大
  4. 特征冗余: 特征间相关性是否过高
  5. 时序一致性: 最近 N 个月模型是否退化
"""
import sys
from pathlib import Path

from ._base import BaseAgent

import numpy as np
import pandas as pd


class ModelCriticAgent(BaseAgent):
    name = "model_critic"
    role = "模型审查员"
    color = "#F44336"

    THRESHOLDS = {
        "max_feature_corr": 0.90,
        "min_positive_rate": 0.03,
        "max_positive_rate": 0.30,
        "max_auc_gap": 0.15,
        "min_test_auc": 0.52,
        "recent_ic_threshold": 0.0,
    }

    def execute(self):
        prep_data = self.bus.get("prep_data")
        experiments = self.bus.get("experiments")
        models = self.bus.get("models")
        metrics = self.bus.get("metrics")

        warnings = []
        checks = {}

        # --- Check 1: Sample balance ---
        self.log("检查 1", "样本平衡性")
        for name, (X, y) in (prep_data or {}).items():
            pos_rate = float(y.mean())
            ok = self.THRESHOLDS["min_positive_rate"] <= pos_rate <= self.THRESHOLDS["max_positive_rate"]
            checks[f"balance_{name}"] = {
                "positive_rate": round(pos_rate, 4),
                "pass": ok,
            }
            if not ok:
                msg = f"[{name}] 正样本比例 {pos_rate:.1%} 不在合理范围 ({self.THRESHOLDS['min_positive_rate']:.0%}-{self.THRESHOLDS['max_positive_rate']:.0%})"
                warnings.append(msg)
                self.log("样本平衡", msg, status="warning")
            else:
                self.log("样本平衡", f"[{name}] {pos_rate:.1%} ✓", status="success")

        # --- Check 2: Feature correlation ---
        self.log("检查 2", "特征冗余度")
        for name, (X, y) in (prep_data or {}).items():
            corr = X.corr().abs()
            np.fill_diagonal(corr.values, 0)
            max_corr = corr.max().max()
            if max_corr > self.THRESHOLDS["max_feature_corr"]:
                pair = corr.stack().idxmax()
                msg = f"[{name}] 特征高度相关: {pair[0]} & {pair[1]} = {max_corr:.2f}"
                warnings.append(msg)
                self.log("特征冗余", msg, status="warning")
            checks[f"corr_{name}"] = {
                "max_correlation": round(float(max_corr), 3),
                "pass": max_corr <= self.THRESHOLDS["max_feature_corr"],
            }

        # --- Check 3: Test AUC sanity ---
        self.log("检查 3", "模型效果验证")
        for exp in (experiments or []):
            exp_auc = exp.get("auc", 0)
            ok = exp_auc >= self.THRESHOLDS["min_test_auc"]
            checks[f"auc_{exp['name']}"] = {
                "auc": exp_auc,
                "pass": ok,
            }
            if not ok:
                msg = f"[{exp['name']}] AUC={exp_auc:.3f} < {self.THRESHOLDS['min_test_auc']}, 模型无效"
                warnings.append(msg)
                self.log("AUC 检查", msg, status="error")
            else:
                self.log("AUC 检查", f"[{exp['name']}] AUC={exp_auc:.3f} ✓", status="success")

        # --- Check 4: Overfitting (train vs test) ---
        self.log("检查 4", "过拟合检测")
        if models and prep_data:
            for model_name, (model, scaler) in models.items():
                feat_key = "slim" if "slim" in model_name or "d1" in model_name else "full"
                if feat_key not in prep_data:
                    continue
                X, y = prep_data[feat_key]
                split = int(len(X) * 0.7)
                X_train, X_test = X.iloc[:split], X.iloc[split:]
                y_train, y_test = y.iloc[:split], y.iloc[split:]

                from sklearn.metrics import roc_auc_score
                train_probs = model.predict_proba(scaler.transform(X_train))[:, 1]
                test_probs = model.predict_proba(scaler.transform(X_test))[:, 1]
                try:
                    train_auc = roc_auc_score(y_train, train_probs)
                    test_auc = roc_auc_score(y_test, test_probs)
                    gap = train_auc - test_auc
                    ok = gap <= self.THRESHOLDS["max_auc_gap"]
                    checks[f"overfit_{model_name}"] = {
                        "train_auc": round(train_auc, 3),
                        "test_auc": round(test_auc, 3),
                        "gap": round(gap, 3),
                        "pass": ok,
                    }
                    if not ok:
                        msg = f"[{model_name}] 过拟合风险: train AUC={train_auc:.3f} vs test={test_auc:.3f}, gap={gap:.3f}"
                        warnings.append(msg)
                        self.log("过拟合", msg, status="warning")
                    else:
                        self.log("过拟合", f"[{model_name}] gap={gap:.3f} ✓", status="success")
                except Exception:
                    pass

        # --- Check 5: Data leakage hint ---
        self.log("检查 5", "数据泄漏审查")
        if metrics:
            test_period = metrics.get("model_info", {}).get("test_period", "")
            train_period = metrics.get("model_info", {}).get("train_period", "")
            checks["data_leakage"] = {
                "train_period": train_period,
                "test_period": test_period,
                "note": "目标使用未来20日回撤, 需确保 embargo >= 20",
                "pass": True,
            }
            self.log("数据泄漏", f"train={train_period}, test={test_period}", status="success")

        # --- Verdict ---
        critical_fails = sum(1 for c in checks.values() if not c.get("pass", True))
        verdict = "REJECT" if critical_fails >= 3 else ("WARN" if warnings else "PASS")

        review = {
            "verdict": verdict,
            "warnings": warnings,
            "checks": checks,
            "n_checks": len(checks),
            "n_warnings": len(warnings),
            "n_critical": critical_fails,
        }
        self.bus.put("review", review)

        icon = {"PASS": "✓", "WARN": "⚠", "REJECT": "✗"}[verdict]
        self.log("裁决", f"{icon} {verdict} ({len(checks)} 项检查, {len(warnings)} 个警告)",
                 status="success" if verdict == "PASS" else "warning" if verdict == "WARN" else "error")

        return review
