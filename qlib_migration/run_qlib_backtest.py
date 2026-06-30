"""
Step 3/3 — 用 Qlib 回测引擎复现锁定基准 (SP500-107)。

校验目标: GROSS 收益对齐 (cost=0, 每日等权再平衡), 与 baseline 毛收益 29.8% 误差 <1%。
信号 = 每个调仓月 Top20 等权 (来自 golden baseline 的 holdings_by_date.json)。
策略 = 自定义 WeightStrategyBase: 每个交易日把目标权重设为"当前生效月" Top20 的等权,
       从而复现 baseline 的 mean(axis=1) 日度等权口径。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import qlib
from qlib.data import D
from qlib.backtest import backtest
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase

ROOT = Path(__file__).resolve().parent.parent
PROVIDER = ROOT / "qlib_data" / "sp500"
HOLDINGS = ROOT / "workspace" / "baseline_sp500" / "holdings_by_date.json"
BASELINE = ROOT / "workspace" / "baseline_sp500" / "baseline.json"


class MonthlyEqualWeight(WeightStrategyBase):
    """每日把权重设为"最近一次月末调仓"选出的 Top20 的等权组合。"""

    def __init__(self, holdings_map, rebal_dates, **kwargs):
        super().__init__(**kwargs)
        # holdings_map: {Timestamp(month_end): [tickers]}, rebal_dates 升序
        self._holdings = holdings_map
        self._rebal = sorted(rebal_dates)

    def _active(self, trade_start_time):
        # 取严格早于交易日的最近一个调仓月末
        active = None
        for d in self._rebal:
            if d < trade_start_time:
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


def build_membership_signal(holdings_map, rebal_dates, cal):
    """构造逐日信号 (每个交易日, 生效月 Top20 给分 1.0), 让策略每日触发再平衡。"""
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
    sig = pd.DataFrame(rows, columns=["datetime", "instrument", "score"])
    sig = sig.set_index(["datetime", "instrument"])["score"]
    return sig


def perf(daily_ret: pd.Series):
    r = daily_ret.dropna()
    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0.0
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"ann_return": round(float(ann), 4), "sharpe": round(float(sharpe), 2),
            "max_drawdown": round(float(mdd), 4), "n_days": int(r.notna().sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.0, help="单边成本 (gross=0)")
    args = ap.parse_args()

    qlib.init(provider_uri=str(PROVIDER), region="us",
              expression_cache=None, dataset_cache=None)

    holdings_raw = json.loads(HOLDINGS.read_text())
    holdings_map = {pd.Timestamp(k): [t.upper() for t in v] for k, v in holdings_raw.items()}
    rebal_dates = sorted(holdings_map.keys())

    cal = [pd.Timestamp(d) for d in D.calendar(freq="day")]
    start = min(d for d in cal if d > rebal_dates[0])   # 首个调仓后的第一个交易日
    end = cal[-2]   # qlib step 需要"下一天", 末日会越界, 去掉最后一天
    cal_bt = [d for d in cal if start <= d <= end]

    signal = build_membership_signal(holdings_map, rebal_dates, cal_bt)

    strategy = MonthlyEqualWeight(
        holdings_map=holdings_map, rebal_dates=rebal_dates,
        signal=signal, risk_degree=1.0,
    )
    executor = {
        "class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True, "verbose": False},
    }
    exchange_kwargs = {
        "freq": "day", "limit_threshold": None, "deal_price": "close",
        "open_cost": args.cost, "close_cost": args.cost, "min_cost": 0.0,
        "trade_unit": None,  # 允许小数股, 匹配 baseline 连续权重
    }

    print(f"[backtest] {start.date()} ~ {end.date()}  cost={args.cost}")
    pm_dict, _ = backtest(
        start_time=start, end_time=end, strategy=strategy, executor=executor,
        benchmark="AAPL", account=1e10, exchange_kwargs=exchange_kwargs,
    )
    report, _positions = pm_dict["1day"]
    print("[report] columns:", list(report.columns))
    for c in ("cost", "turnover"):
        if c in report.columns:
            print(f"[report] sum({c}) = {report[c].sum():.4f}  mean={report[c].mean():.6f}")
    qlib_ret = report["return"]   # 账户日收益 (qlib 中为毛收益, cost 单列记录)
    if "cost" in report.columns:
        qlib_ret_net = report["return"] - report["cost"]
        npf = perf(qlib_ret_net)
        print(f"[net] qlib net ann_return = {npf['ann_return']:.4f}  (return - cost)")

    qp = perf(qlib_ret)
    base = json.loads(BASELINE.read_text())["performance"]
    base_gross_ann = base["组合年化收益(毛)"]

    print("\n=== QLIB vs BASELINE (GROSS) ===")
    print(f"  qlib  ann_return = {qp['ann_return']:.4f}   sharpe={qp['sharpe']}   mdd={qp['max_drawdown']}  days={qp['n_days']}")
    print(f"  base  ann(gross) = {base_gross_ann:.4f}   sharpe(net)={base['组合Sharpe']}  mdd={base['最大回撤']}")
    diff = abs(qp["ann_return"] - base_gross_ann)
    print(f"  |Δ ann_return| = {diff:.4f}  ({'PASS <1pt' if diff < 0.01 else 'OUTSIDE 1pt — investigate'})")

    net_ann = None
    if "cost" in report.columns and args.cost > 0:
        net_ann = perf(report["return"] - report["cost"])["ann_return"]
    out = ROOT / "qlib_migration" / "qlib_result.json"
    out.write_text(json.dumps({
        "qlib_gross": qp,
        "qlib_net_ann": net_ann,
        "baseline_gross_ann": base_gross_ann,
        "baseline_net_ann": base["组合年化收益(净)"],
        "abs_diff_gross": round(diff, 4),
        "abs_diff_net": round(abs(net_ann - base["组合年化收益(净)"]), 4) if net_ann else None,
        "cost": args.cost,
        "total_cost": round(float(report["cost"].sum()), 4) if "cost" in report.columns else None,
    }, ensure_ascii=False, indent=2))
    qlib_ret.to_frame("qlib_return").to_csv(ROOT / "qlib_migration" / "qlib_daily_returns.csv")
    print(f"\n  written: {out.name}, qlib_daily_returns.csv")


if __name__ == "__main__":
    main()
