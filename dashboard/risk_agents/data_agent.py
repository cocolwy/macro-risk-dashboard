"""
MacroDataAgent — 从注册表获取指标, 数量和类型完全可配置。

输出到 Bus:
  macro_df     pd.DataFrame (index=date_str, columns=指标名)
  raw_data     dict {indicator_name: [{date, value}, ...]}
  alerts       dict {indicator: {level, message}}

默认获取 7 个宏观指标, 但你可以:
  - 通过 indicators= 参数只跑部分指标
  - 通过 registry.INDICATOR.add() 注册新指标
  - 通过 registry.INDICATOR.remove() 去掉不需要的
"""
import json
import sys
from pathlib import Path

import pandas as pd

from ._base import BaseAgent
from .registry import INDICATOR

_DASHBOARD_DIR = str(Path(__file__).resolve().parent.parent)
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)
from fetch_macro_data import compute_alert_status


class MacroDataAgent(BaseAgent):
    name = "macro_data"
    role = "宏观数据获取"
    color = "#4CAF50"

    def execute(self, use_cached=False, data_dir=None, indicators=None):
        """
        Parameters
        ----------
        use_cached : bool
            True = 从已有 JSON 文件加载 (快速模式, 不联网)
        data_dir : str | Path
            JSON 数据目录, 默认 dashboard/data/
        indicators : list[str] | None
            要获取的指标列表, None = 注册表中的全部指标
            例: ["vix", "sp500", "gold"]  只跑这 3 个
        """
        data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data"
        active = indicators or INDICATOR.keys()

        self.log("指标列表", f"{len(active)} 个: {list(active)}")

        if use_cached:
            return self._load_cached(data_dir, active)

        all_data = {}
        for name in active:
            fetcher = INDICATOR.get(name)
            if fetcher is None:
                self.log(name, f"未注册, 跳过", status="warning")
                continue
            try:
                all_data[name] = fetcher()
                self.log(name, f"{len(all_data[name])} data points", status="success")
            except Exception as e:
                all_data[name] = []
                self.log(name, f"FAILED: {e}", status="error")

        self._publish(all_data)
        return {"days": len(self.bus.get("macro_df")), "indicators": list(all_data.keys())}

    def _load_cached(self, data_dir, active):
        all_data = {}
        for name in active:
            fp = data_dir / f"{name}.json"
            if fp.exists():
                with open(fp) as f:
                    all_data[name] = json.load(f)
                self.log(name, f"loaded {len(all_data[name])} points (cached)")
            else:
                all_data[name] = []

        self._publish(all_data)
        self.log("汇总", f"{len(self.bus.get('macro_df'))} 交易日 (cached)", status="success")
        return {"days": len(self.bus.get("macro_df")), "indicators": list(all_data.keys())}

    def _publish(self, all_data):
        """Build DataFrame + alerts from raw data, put into Bus."""
        alerts = compute_alert_status(all_data)
        all_data["alerts"] = alerts

        self.bus.put("raw_data", all_data)
        self.bus.put("alerts", alerts)

        macro_df = self._build_dataframe(all_data)
        self.bus.put("macro_df", macro_df)
        self.log("汇总", f"{len(macro_df)} 交易日, {sum(1 for k, v in all_data.items() if v and k != 'alerts')} 个指标")

    def _build_dataframe(self, all_data):
        """Convert raw JSON dicts → aligned DataFrame, using registry metadata."""
        data = {}
        for name in INDICATOR.keys():
            raw = all_data.get(name, [])
            if not raw:
                continue
            meta = INDICATOR.get_meta(name)
            parser = meta.get("parser")
            if parser:
                data[name] = {d["date"]: parser(d) for d in raw if "date" in d}
            else:
                field = meta.get("json_field", name)
                data[name] = {d["date"]: d.get(field) for d in raw if "date" in d}

        required = [k for k in INDICATOR.keys()
                    if INDICATOR.get_meta(k).get("required") and k in data]
        if len(required) < 2:
            required = list(data.keys())[:2]

        date_sets = [set(data[k].keys()) for k in required if k in data]
        all_dates = sorted(set.intersection(*date_sets)) if date_sets else []

        df = pd.DataFrame(index=all_dates)
        for name, series in data.items():
            df[name] = df.index.map(lambda d, s=series: s.get(d))
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.ffill()
        if required:
            df = df.dropna(subset=[r for r in required if r in df.columns])
        return df
