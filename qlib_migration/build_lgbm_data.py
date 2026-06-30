"""
Week4 Step 1 — 构建 LightGBM 训练数据 (扩展历史 2016-2024)。

复用 Researcher 的因子公式 (orthogonalize=False → 原始 5 因子), 保证与 baseline 一致。
PCA 故意不在这里做 —— 放到 walk-forward 里逐折拟合, 避免前视偏差。

输出 qlib_migration/lgbm_factor_panel.parquet:
  index = (date[月末], instrument), 列 = [momentum, bp, log_mcap, low_vol, reversal, fwd_ret]
  fwd_ret = 下个月收益 (LambdaRank 的排序标签来源)

在 .venv 里跑 (需要 agents 全栈: seaborn/alphalens)。
"""
from pathlib import Path
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.base import MessageBus
from agents.universe import SP500_CORE
from agents.data_engineer import DataEngineer
from agents.researcher import Researcher

START, END = "2016-06-01", "2024-12-31"
ANALYSIS_START = "2017-06-01"      # 保证 12 个月动量回看可用
FACTORS = ["momentum", "bp", "log_mcap", "low_vol", "reversal"]
OUT = ROOT / "qlib_migration" / "lgbm_factor_panel.parquet"


def main():
    bus = MessageBus(workspace=str(ROOT / "workspace" / "lgbm_data"))

    DataEngineer(bus).execute(start=START, end=END, tickers=SP500_CORE,
                              liquidity_filter=False)
    Researcher(bus).execute(analysis_start=ANALYSIS_START,
                            orthogonalize=False, industry_neutral=False)

    factors_dict = bus.get("factors_dict")     # {date: df[FACTORS]}
    close = bus.get("close")

    # 下个月收益标签
    monthly_close = close.resample("ME").last()
    fwd_ret = monthly_close.pct_change().shift(-1)

    rows = []
    for date in sorted(factors_dict.keys()):
        df = factors_dict[date][FACTORS].copy()
        if date in fwd_ret.index:
            df["fwd_ret"] = fwd_ret.loc[date].reindex(df.index)
        else:
            df["fwd_ret"] = float("nan")
        df.index = [t.upper() for t in df.index]
        df.index.name = "instrument"
        df["date"] = date
        rows.append(df.reset_index())

    panel = pd.concat(rows, ignore_index=True).set_index(["date", "instrument"])
    panel = panel.sort_index()
    panel.to_parquet(OUT)

    n_months = panel.index.get_level_values(0).nunique()
    cov = panel[FACTORS].notna().all(axis=1).mean()
    print(f"[done] panel → {OUT}")
    print(f"  rows={len(panel)}  months={n_months}  "
          f"span={panel.index.get_level_values(0).min().date()}~{panel.index.get_level_values(0).max().date()}")
    print(f"  full-factor coverage={cov:.1%}  fwd_ret coverage={panel['fwd_ret'].notna().mean():.1%}")


if __name__ == "__main__":
    main()
