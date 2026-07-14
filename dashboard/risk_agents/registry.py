"""
注册表: 指标、特征构建器、模型、审查规则 — 全部可插拔。

用法:
    from dashboard.risk_agents.registry import INDICATOR, FEATURE_BUILDER, MODEL_CONFIG

    # 注册一个新指标
    @INDICATOR.register("gold")
    def fetch_gold():
        import yfinance as yf
        s = yf.Ticker("GC=F").history(period="5y")["Close"]
        return [{"date": d.strftime("%Y-%m-%d"), "gold": round(float(v), 2)}
                for d, v in s.items() if pd.notna(v)]

    # 注册一个新特征构建器
    @FEATURE_BUILDER.register("my_features")
    def my_features(df):
        feats = pd.DataFrame(index=df.index)
        feats["gold_momentum"] = df["gold"].pct_change(20)
        return feats
"""
from typing import Callable, Any, Optional


class Registry:
    """通用注册表 — 用装饰器或 .add() 注册可调用对象。"""

    def __init__(self, name: str):
        self.name = name
        self._items = {}  # type: dict[str, dict]

    def register(self, key: str, **meta):
        """装饰器: @registry.register("name")"""
        def decorator(fn: Callable):
            self._items[key] = {"fn": fn, **meta}
            return fn
        return decorator

    def add(self, key: str, fn: Callable, **meta):
        """函数式注册: registry.add("name", fn)"""
        self._items[key] = {"fn": fn, **meta}

    def remove(self, key: str):
        self._items.pop(key, None)

    def get(self, key: str) -> Optional[Callable]:
        item = self._items.get(key)
        return item["fn"] if item else None

    def get_meta(self, key: str) -> dict:
        return {k: v for k, v in self._items.get(key, {}).items() if k != "fn"}

    def keys(self) -> list[str]:
        return list(self._items.keys())

    def items(self):
        return [(k, v["fn"]) for k, v in self._items.items()]

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"<Registry '{self.name}' [{len(self._items)} items: {', '.join(self._items)}]>"


# ═══════════════════════════════════════════════════════════
# 全局注册表实例
# ═══════════════════════════════════════════════════════════

INDICATOR = Registry("indicator")
FEATURE_BUILDER = Registry("feature_builder")
MODEL_CONFIG = Registry("model_config")
CRITIC_CHECK = Registry("critic_check")


# ═══════════════════════════════════════════════════════════
# 默认指标 (现有的 7 个, 可增删)
# ═══════════════════════════════════════════════════════════

def _register_default_indicators():
    import sys
    from pathlib import Path
    _dir = str(Path(__file__).resolve().parent.parent)
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from fetch_macro_data import (
        fetch_fred_series, fetch_yfinance_history,
        compute_market_breadth, compute_absorption_ratio,
        compute_turbulence_index, series_to_json, df_to_json,
    )

    @INDICATOR.register("term_spread", json_field="term_spread_10y2y",
                         parser=lambda d: d.get("term_spread_10y2y"))
    def _():
        return series_to_json(fetch_fred_series("T10Y2Y"), "term_spread_10y2y")

    @INDICATOR.register("credit_spread", json_field="high_yield_spread",
                         parser=lambda d: d.get("high_yield_spread"))
    def _():
        baa = fetch_fred_series("BAA10Y").to_frame("high_yield_spread")
        aaa = fetch_fred_series("AAA10Y").to_frame("investment_grade_spread")
        return df_to_json(baa.join(aaa, how="outer").ffill())

    @INDICATOR.register("vix", json_field="vix",
                         parser=lambda d: d.get("vix"),
                         required=True)
    def _():
        s = fetch_yfinance_history("^VIX")["Close"]
        s.name = "vix"
        return series_to_json(s, "vix")

    @INDICATOR.register("sp500", json_field="sp500",
                         parser=lambda d: d.get("sp500"),
                         required=True)
    def _():
        s = fetch_yfinance_history("^GSPC")["Close"]
        s.name = "sp500"
        return series_to_json(s, "sp500")

    @INDICATOR.register("breadth", json_field="pct_above_200ma",
                         parser=lambda d: d.get("pct_above_200ma"))
    def _():
        return df_to_json(compute_market_breadth())

    @INDICATOR.register("absorption_ratio", json_field="absorption_ratio",
                         parser=lambda d: d.get("absorption_ratio"))
    def _():
        return df_to_json(compute_absorption_ratio())

    @INDICATOR.register("turbulence", json_field="turbulence",
                         parser=lambda d: d.get("turbulence"))
    def _():
        return df_to_json(compute_turbulence_index())


def _register_default_feature_builders():
    import sys
    from pathlib import Path
    _dir = str(Path(__file__).resolve().parent.parent)
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from predict_model import build_features, build_features_slim

    FEATURE_BUILDER.add("full", build_features)
    FEATURE_BUILDER.add("slim", build_features_slim)


_register_default_indicators()
_register_default_feature_builders()
