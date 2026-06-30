"""
Week4 v2 Step 1 — 构建 LightGBM 训练数据 (Russell-1000 宽 universe, 2016-2024)。

相比 v1: 股票池从 SP500_CORE(107) 扩大到 RUSSELL1000(~459),
更多截面股票 → walk-forward 每折训练样本更多, LightGBM query group 更大。

PCA 仍不在此步做 — 逐折拟合避免前视偏差。

输出 qlib_migration/lgbm_factor_panel_v2.parquet:
  index = (date[月末], instrument), 列 = [momentum, bp, log_mcap, low_vol, reversal, fwd_ret]

在 .venv 里跑 (需要 agents 全栈: seaborn/alphalens)。
"""
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.base import MessageBus
from agents.universe import RUSSELL1000
from agents.data_engineer import DataEngineer
from agents.researcher import Researcher

START, END = "2016-06-01", "2024-12-31"
ANALYSIS_START = "2017-06-01"
FACTORS = ["momentum", "bp", "log_mcap", "low_vol", "reversal"]
OUT = ROOT / "qlib_migration" / "lgbm_factor_panel_v2.parquet"


def main():
    print(f"[v2] Building factor panel with RUSSELL1000 ({len(RUSSELL1000)} tickers)")
    bus = MessageBus(workspace=str(ROOT / "workspace" / "lgbm_data_v2"))

    DataEngineer(bus).execute(start=START, end=END, tickers=RUSSELL1000,
                              liquidity_filter=False)
    Researcher(bus).execute(analysis_start=ANALYSIS_START,
                            orthogonalize=False, industry_neutral=False)

    factors_dict = bus.get("factors_dict")
    close = bus.get("close")

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
    n_inst = panel.index.get_level_values(1).nunique()
    cov = panel[FACTORS].notna().all(axis=1).mean()
    print(f"[done] panel → {OUT}")
    print(f"  rows={len(panel)}  months={n_months}  instruments={n_inst}")
    print(f"  span={panel.index.get_level_values(0).min().date()}~"
          f"{panel.index.get_level_values(0).max().date()}")
    print(f"  full-factor coverage={cov:.1%}  "
          f"fwd_ret coverage={panel['fwd_ret'].notna().mean():.1%}")


if __name__ == "__main__":
    main()
