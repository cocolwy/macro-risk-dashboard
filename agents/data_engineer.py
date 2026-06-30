"""
DataEngineer: 数据获取、清洗、流动性过滤、质量检查
"""

import numpy as np
import pandas as pd
import yfinance as yf
from .base import BaseAgent, MessageBus
from .universe import SP500_CORE, RUSSELL1000


class DataEngineer(BaseAgent):
    name = "data_engineer"
    role = "数据工程师"
    color = "#4CAF50"

    def execute(self, start="2021-06-01", end="2024-12-31",
                tickers=None, liquidity_filter=False, min_adv=5_000_000):
        """
        Parameters
        ----------
        tickers : list or None
            股票池，默认 SP500_CORE
        liquidity_filter : bool
            是否做流动性过滤（20日平均成交额 > min_adv）
        min_adv : float
            最低 20 日平均成交额（美元），默认 500 万
        """
        tickers = tickers or RUSSELL1000

        # ── 1. 下载价格 + 成交量 ──
        self.log("下载价格", f"拉取 {len(tickers)} 只股票 {start} ~ {end}")
        data = yf.download(tickers, start=start, end=end,
                           group_by="ticker", auto_adjust=True, threads=True)

        close = pd.DataFrame({
            t: data[t]["Close"]
            for t in tickers
            if t in data.columns.get_level_values(0)
        })
        close = close.dropna(axis=1, thresh=252)
        self.log("价格清洗", f"有效股票: {close.shape[1]}, 交易日: {close.shape[0]}")

        # ── 2. 流动性过滤 ──
        if liquidity_filter:
            self.log("流动性过滤", f"20 日平均成交额 > ${min_adv/1e6:.0f}M")

            # 提取成交量
            volume = pd.DataFrame({
                t: data[t]["Volume"]
                for t in close.columns
                if t in data.columns.get_level_values(0)
            })

            # 成交额 = 收盘价 × 成交量
            dollar_volume = close * volume
            adv_20d = dollar_volume.rolling(20).mean()

            # 在每个月末检查流动性，只保留在大部分月末满足条件的股票
            monthly_adv = adv_20d.resample("ME").last()
            # 至少 70% 的月份满足流动性要求
            pass_ratio = (monthly_adv > min_adv).mean()
            liquid_tickers = pass_ratio[pass_ratio > 0.7].index.tolist()

            dropped = close.shape[1] - len(liquid_tickers)
            close = close[liquid_tickers]
            self.log("流动性结果",
                     f"通过: {len(liquid_tickers)}, 剔除: {dropped} "
                     f"(< ${min_adv/1e6:.0f}M 日均成交额)",
                     status="success")

        # ── 3. 下载基本面 + 行业分类 ──
        self.log("下载基本面", f"获取 {close.shape[1]} 只股票的 B/P、市值、行业数据")
        valid_tickers = close.columns.tolist()
        records = []
        sector_map = {}  # ticker → sector
        for i, t in enumerate(valid_tickers):
            try:
                info = yf.Ticker(t).info
                bp = info.get("bookValue", None)
                price = info.get("currentPrice", None)
                mcap = info.get("marketCap", None)
                sector = info.get("sector", "Unknown")
                sector_map[t] = sector
                if bp and price and price > 0:
                    records.append({
                        "ticker": t, "bp_ratio": bp / price,
                        "market_cap": mcap, "sector": sector,
                    })
            except Exception:
                pass
            if (i + 1) % 50 == 0:
                self.log("基本面进度", f"{i+1}/{len(valid_tickers)}")

        fundamentals = pd.DataFrame(records).set_index("ticker")
        self.log("基本面完成", f"获取到 {len(fundamentals)} 只股票的基本面")

        # 行业分布统计
        if sector_map:
            sector_series = pd.Series(sector_map)
            sector_counts = sector_series.value_counts()
            self.log("行业分布",
                     f"{len(sector_counts)} 个行业, "
                     f"前3: {dict(sector_counts.head(3))}",
                     status="success")
            self.bus.put("sector_map", sector_map)

        # ── 4. 质量检查 ──
        quality = {
            "股票数": close.shape[1],
            "交易日数": close.shape[0],
            "价格缺失率": f"{close.isnull().mean().mean():.2%}",
            "基本面覆盖率": f"{len(fundamentals) / close.shape[1]:.2%}",
            "起止日期": f"{close.index[0].date()} ~ {close.index[-1].date()}",
            "流动性过滤": "是" if liquidity_filter else "否",
        }
        self.log("质量报告", str(quality), status="success")

        # ── 5. 存入共享区 ──
        self.bus.put("close", close)
        self.bus.put("fundamentals", fundamentals)
        self.bus.put("data_quality", quality)

        return {"close_shape": close.shape, "fund_count": len(fundamentals), "quality": quality}
