"""
MacroDataAgent — 获取 7 个宏观风险指标。

输出到 Bus:
  macro_df     pd.DataFrame (index=date_str, columns=指标名)
  raw_data     dict {indicator_name: [{date, value}, ...]}  (JSON 格式)
  alerts       dict {indicator: {level, message}}
"""
import sys
from pathlib import Path

from ._base import BaseAgent

_DASHBOARD_DIR = str(Path(__file__).resolve().parent.parent)
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)
from fetch_macro_data import (
    fetch_fred_series, fetch_yfinance_history,
    compute_market_breadth, compute_sector_vs_ma,
    compute_absorption_ratio, compute_turbulence_index,
    series_to_json, df_to_json, compute_alert_status,
)

import pandas as pd


class MacroDataAgent(BaseAgent):
    name = "macro_data"
    role = "宏观数据获取"
    color = "#4CAF50"

    def execute(self, use_cached=False, data_dir=None):
        """
        Parameters
        ----------
        use_cached : bool
            True = 从已有 JSON 文件加载 (快速模式, 不联网)
        data_dir : str
            JSON 数据目录, 默认 dashboard/data/
        """
        import json
        data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data"

        if use_cached:
            return self._load_cached(data_dir)

        all_data = {}
        indicators = [
            ("term_spread", self._fetch_term_spread),
            ("credit_spread", self._fetch_credit_spread),
            ("vix", self._fetch_vix),
            ("sp500", self._fetch_sp500),
            ("breadth", self._fetch_breadth),
            ("absorption_ratio", self._fetch_ar),
            ("turbulence", self._fetch_turbulence),
        ]

        for name, fetcher in indicators:
            try:
                all_data[name] = fetcher()
                self.log(name, f"{len(all_data[name])} data points", status="success")
            except Exception as e:
                all_data[name] = []
                self.log(name, f"FAILED: {e}", status="error")

        alerts = compute_alert_status(all_data)
        all_data["alerts"] = alerts

        self.bus.put("raw_data", all_data)
        self.bus.put("alerts", alerts)

        macro_df = self._build_dataframe(all_data)
        self.bus.put("macro_df", macro_df)

        self.log("汇总", f"{len(macro_df)} 交易日, {sum(1 for v in all_data.values() if v)} 个指标")
        return {"days": len(macro_df), "indicators": list(all_data.keys())}

    def _load_cached(self, data_dir):
        import json
        all_data = {}
        for name in ["term_spread", "credit_spread", "vix", "sp500",
                      "breadth", "absorption_ratio", "turbulence"]:
            fp = data_dir / f"{name}.json"
            if fp.exists():
                with open(fp) as f:
                    all_data[name] = json.load(f)
                self.log(name, f"loaded {len(all_data[name])} points (cached)")
            else:
                all_data[name] = []

        alerts = compute_alert_status(all_data)
        all_data["alerts"] = alerts
        self.bus.put("raw_data", all_data)
        self.bus.put("alerts", alerts)

        macro_df = self._build_dataframe(all_data)
        self.bus.put("macro_df", macro_df)

        self.log("汇总", f"{len(macro_df)} 交易日 (cached)", status="success")
        return {"days": len(macro_df), "indicators": list(all_data.keys())}

    def _build_dataframe(self, all_data):
        from predict_model import INDICATOR_PARSERS
        data = {}
        for name, parser in INDICATOR_PARSERS.items():
            raw = all_data.get(name, [])
            data[name] = {d["date"]: parser(d) for d in raw if "date" in d}

        all_dates = sorted(set(data.get("sp500", {}).keys()) &
                           set(data.get("vix", {}).keys()))
        df = pd.DataFrame(index=all_dates)
        for name, series in data.items():
            df[name] = df.index.map(lambda d, s=series: s.get(d))
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.ffill().dropna(subset=["sp500", "vix"])
        return df

    @staticmethod
    def _fetch_term_spread():
        return series_to_json(fetch_fred_series("T10Y2Y"), "term_spread_10y2y")

    @staticmethod
    def _fetch_credit_spread():
        baa = fetch_fred_series("BAA10Y").to_frame("high_yield_spread")
        aaa = fetch_fred_series("AAA10Y").to_frame("investment_grade_spread")
        return df_to_json(baa.join(aaa, how="outer").ffill())

    @staticmethod
    def _fetch_vix():
        vix = fetch_yfinance_history("^VIX")["Close"]
        vix.name = "vix"
        return series_to_json(vix, "vix")

    @staticmethod
    def _fetch_sp500():
        spx = fetch_yfinance_history("^GSPC")["Close"]
        spx.name = "sp500"
        return series_to_json(spx, "sp500")

    @staticmethod
    def _fetch_breadth():
        return df_to_json(compute_market_breadth())

    @staticmethod
    def _fetch_ar():
        return df_to_json(compute_absorption_ratio())

    @staticmethod
    def _fetch_turbulence():
        return df_to_json(compute_turbulence_index())
