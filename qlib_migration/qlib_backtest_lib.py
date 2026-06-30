"""可复用的 Qlib 回测封装 (从已验证的 run_qlib_backtest.py 抽出)。

run_backtest(holdings_map, cost) — 给定 {月末日: [tickers]}, 用 Qlib 引擎跑
Top-N 等权、月度调仓回测, 返回 gross/net 绩效。与已锁定 baseline 同一套引擎/口径。
"""
from pathlib import Path
import numpy as np
import pandas as pd

import qlib
from qlib.data import D
from qlib.backtest import backtest
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROVIDER = ROOT / "qlib_data" / "sp500"
_INIT = {"done": False, "uri": None}


def ensure_qlib(provider_uri=None):
    uri = str(provider_uri or DEFAULT_PROVIDER)
    if not _INIT["done"] or _INIT["uri"] != uri:
        qlib.init(provider_uri=uri, region="us",
                  expression_cache=None, dataset_cache=None)
        _INIT["done"] = True
        _INIT["uri"] = uri


class MonthlyEqualWeight(WeightStrategyBase):
    def __init__(self, holdings_map, rebal_dates, **kwargs):
        super().__init__(**kwargs)
        self._holdings = holdings_map
        self._rebal = sorted(rebal_dates)

    def _active(self, t):
        active = None
        for d in self._rebal:
            if d < t:
                active = d
            else:
                break
        return self._holdings.get(active, [])

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        names = self._active(trade_start_time)
        if not names:
            return {}
        w = 1.0 / len(names)
        return {tk: w for tk in names}


def _membership_signal(holdings_map, rebal_dates, cal):
    rebal = sorted(rebal_dates)
    rows = []
    for d in cal:
        active = None
        for r in rebal:
            if r < d:
                active = r
            else:
                break
        if active is None:
            continue
        for tk in holdings_map.get(active, []):
            rows.append((d, tk, 1.0))
    return pd.DataFrame(rows, columns=["datetime", "instrument", "score"]).set_index(["datetime", "instrument"])["score"]


def _perf(r):
    r = r.dropna()
    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"ann_return": round(float(ann), 4),
            "sharpe": round(float(ann / vol), 2) if vol > 0 else 0.0,
            "max_drawdown": round(float(mdd), 4), "n_days": int(r.notna().sum())}


def run_backtest(holdings_map, cost=0.001, start=None, end=None, provider_uri=None):
    """holdings_map: {pd.Timestamp(month_end): [TICKERS]}. 返回 gross/net 绩效 dict。"""
    ensure_qlib(provider_uri)
    holdings_map = {pd.Timestamp(k): [t.upper() for t in v] for k, v in holdings_map.items()}
    rebal = sorted(holdings_map.keys())
    cal = [pd.Timestamp(d) for d in D.calendar(freq="day")]
    start = start or min(d for d in cal if d > rebal[0])
    end = end or cal[-2]
    cal_bt = [d for d in cal if start <= d <= end]
    signal = _membership_signal(holdings_map, rebal, cal_bt)

    strat = MonthlyEqualWeight(holdings_map=holdings_map, rebal_dates=rebal,
                               signal=signal, risk_degree=1.0)
    executor = {"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True, "verbose": False}}
    exch = {"freq": "day", "limit_threshold": None, "deal_price": "close",
            "open_cost": cost, "close_cost": cost, "min_cost": 0.0, "trade_unit": None}
    pm_dict, _ = backtest(start_time=start, end_time=end, strategy=strat, executor=executor,
                          benchmark="AAPL", account=1e10, exchange_kwargs=exch)
    report, _ = pm_dict["1day"]
    gross = _perf(report["return"])
    net = _perf(report["return"] - report["cost"]) if "cost" in report.columns else None
    return {"gross": gross, "net": net,
            "total_cost": round(float(report["cost"].sum()), 4) if "cost" in report.columns else None,
            "period": f"{start.date()}~{end.date()}"}
