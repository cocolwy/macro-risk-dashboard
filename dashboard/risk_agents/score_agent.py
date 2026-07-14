"""
ScoreAgent — 计算复合风险分数 + 输出最终 JSON, 指标列表从注册表读取。

从 Bus 读取: raw_data, metrics, experiments, weight_comparison, review, alerts
输出到 Bus:
  composite_scores  list[dict]
  summary           dict
"""
import json
import shutil
import datetime
import sys
from pathlib import Path

from ._base import BaseAgent
from .registry import INDICATOR

_DASHBOARD_DIR = str(Path(__file__).resolve().parent.parent)
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)
from composite_score import compute_composite_score, get_score_label


class ScoreAgent(BaseAgent):
    name = "score_output"
    role = "分数输出员"
    color = "#9C27B0"

    def execute(self, data_dir=None, sync_frontend=True):
        data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data"
        data_dir.mkdir(exist_ok=True)

        raw_data = self.bus.get("raw_data") or {}
        metrics = self.bus.get("metrics")
        experiments = self.bus.get("experiments") or []
        weight_comparison = self.bus.get("weight_comparison")
        review = self.bus.get("review")
        alerts = self.bus.get("alerts") or {}

        # --- Composite score ---
        self.log("复合分数", "计算中...")
        scores = compute_composite_score(raw_data)
        self.bus.put("composite_scores", scores)

        composite_summary = None
        if scores:
            latest = scores[-1]
            label = get_score_label(latest["composite_score"])
            composite_summary = {
                "score": latest["composite_score"],
                "label": label["label"],
                "level": label["level"],
                "color": label["color"],
                "action": label["action"],
                "components": latest["components"],
                "date": latest["date"],
            }
            self.log("复合分数", f"{latest['composite_score']:.0f}/100 [{label['label']}]", status="success")

        # --- Save individual JSON files (dynamic from registry) ---
        for key in INDICATOR.keys():
            if key in raw_data and raw_data[key]:
                with open(data_dir / f"{key}.json", "w") as f:
                    json.dump(raw_data[key], f)

        with open(data_dir / "composite_score.json", "w") as f:
            json.dump(scores, f)

        # --- Model metrics ---
        if metrics:
            metrics["experiments"] = experiments
            if weight_comparison:
                metrics["weight_comparison"] = weight_comparison

            metrics_path = data_dir / "model_metrics.json"
            if metrics_path.exists():
                try:
                    with open(metrics_path) as f:
                        old = json.load(f)
                    ext = [e for e in old.get("experiments", []) if "Ext" in e.get("name", "")]
                    if ext:
                        metrics["experiments"].extend(ext)
                        self.log("保留", f"{len(ext)} 个扩展实验")
                    if "experiment_a_info" in old:
                        metrics["experiment_a_info"] = old["experiment_a_info"]
                except (json.JSONDecodeError, KeyError):
                    pass

            with open(metrics_path, "w") as f:
                json.dump(metrics, f)
            self.log("输出", "model_metrics.json", status="success")

            if "probability_timeline" in metrics:
                with open(data_dir / "crash_prediction.json", "w") as f:
                    json.dump(metrics["probability_timeline"], f)
                self.log("输出", "crash_prediction.json", status="success")

        # --- Summary ---
        summary = {
            "alerts": alerts,
            "last_updated": datetime.datetime.now().isoformat(),
            "indicators_available": [k for k in raw_data if k not in ("alerts",) and raw_data[k]],
        }
        if composite_summary:
            summary["composite_score"] = composite_summary
        if review:
            summary["model_review"] = {
                "verdict": review["verdict"],
                "n_warnings": review["n_warnings"],
            }

        with open(data_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        self.bus.put("summary", summary)
        self.log("输出", "summary.json", status="success")

        if sync_frontend:
            public_dir = Path(__file__).resolve().parent.parent / "frontend" / "public" / "data"
            if public_dir.parent.exists():
                if public_dir.exists():
                    shutil.rmtree(public_dir)
                shutil.copytree(data_dir, public_dir)
                self.log("同步", "frontend/public/data", status="success")

        self.log("完成", f"{len(scores)} 日复合分数已生成", status="success")
        return {
            "composite_latest": composite_summary,
            "files_written": ["model_metrics.json", "crash_prediction.json",
                              "composite_score.json", "summary.json"],
        }
