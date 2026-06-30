"""
五因子研究：动量、价值、规模、低波动、反转
数据源: yfinance | 股票池: S&P 500 | 分析: alphalens-reloaded
"""

import warnings
warnings.filterwarnings("ignore")

import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import alphalens

# ─── 配置 ───────────────────────────────────────────────
START = "2021-06-01"   # 多拉半年，给动量因子留窗口
END = "2024-12-31"
ANALYSIS_START = "2022-01-01"
OUTPUT_DIR = "/mnt/data/x2robot_v2/coco/code/quan/output"

# ─── 1. 获取 S&P 500 成分股 ─────────────────────────────
# 硬编码主要 S&P 500 成分股（约 100 只大盘股代替全量 500，加速下载）
SP500_CORE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "BRK-B", "UNH",
    "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO", "ACN",
    "ABT", "DHR", "NEE", "LIN", "TXN", "PM", "CMCSA", "VZ", "RTX", "HON",
    "AMGN", "UNP", "LOW", "NKE", "UPS", "INTC", "COP", "BMY", "SBUX", "BA",
    "CAT", "DE", "MS", "GS", "BLK", "AXP", "MDLZ", "ADI", "ISRG", "GILD",
    "PLD", "MMC", "REGN", "SYK", "CB", "BKNG", "VRTX", "AMT", "TMUS", "CI",
    "MO", "DUK", "SO", "CL", "ZTS", "BDX", "CME", "TGT", "PNC", "ICE",
    "USB", "TFC", "SLB", "APD", "EOG", "WM", "EMR", "FDX", "ORLY", "NSC",
    "GD", "PSA", "AEP", "SRE", "MCK", "ADSK", "D", "ADP", "CCI", "KLAC",
    "MSCI", "FTNT", "AFL", "AIG", "SPG", "F", "GM", "HUM", "WBA", "DOW",
]


def get_sp500_tickers():
    """返回 S&P 500 核心成分股列表"""
    return SP500_CORE.copy()


# ─── 2. 下载价格数据 ─────────────────────────────────────
def download_prices(tickers):
    """批量下载日线收盘价"""
    print(f"[INFO] 下载 {len(tickers)} 只股票的日线数据 ...")
    data = yf.download(tickers, start=START, end=END, group_by="ticker",
                       auto_adjust=True, threads=True)
    # 提取收盘价
    close = pd.DataFrame({
        t: data[t]["Close"] for t in tickers if t in data.columns.get_level_values(0)
    })
    # 去掉数据太少的股票（至少要有 252 天）
    close = close.dropna(axis=1, thresh=252)
    print(f"[INFO] 有效股票数: {close.shape[1]}")
    return close


# ─── 3. 下载基本面数据（B/P） ────────────────────────────
def download_fundamentals(tickers):
    """获取 Book Value / Price（市净率倒数）和市值"""
    print("[INFO] 下载基本面数据 ...")
    records = []
    for i, t in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(tickers)}")
        try:
            info = yf.Ticker(t).info
            bp = info.get("bookValue", None)
            price = info.get("currentPrice", None)
            mcap = info.get("marketCap", None)
            if bp and price and price > 0:
                records.append({
                    "ticker": t,
                    "book_value": bp,
                    "price": price,
                    "bp_ratio": bp / price,
                    "market_cap": mcap,
                })
        except Exception:
            pass
    df = pd.DataFrame(records).set_index("ticker")
    print(f"[INFO] 获取到 {len(df)} 只股票的基本面")
    return df


# ─── 4. 计算五因子 ──────────────────────────────────────
def compute_factors(close, fundamentals):
    """
    在每个月末截面上计算五因子：
      1. 动量 ret_2_12  : 过去 2~12 月累计收益
      2. 价值 bp        : Book / Price
      3. 规模 log_mcap  : log(市值)
      4. 低波动 vol_20d : 过去 20 日收益率标准差（取负，低波动为正）
      5. 反转 reversal  : -过去 1 个月收益率
    """
    ret_daily = close.pct_change()

    # 月末重采样
    monthly_close = close.resample("ME").last()
    monthly_ret = monthly_close.pct_change()

    # ---- 动量: ret_2_12 = 过去12个月收益 - 过去1个月收益
    ret_12m = monthly_close.pct_change(12)
    ret_1m = monthly_close.pct_change(1)
    momentum = ret_12m - ret_1m  # 跳过最近1个月

    # ---- 反转: -ret_1m
    reversal = -ret_1m

    # ---- 低波动: 20日滚动标准差（取负号，低波动 → 高因子值）
    vol_20d = ret_daily.rolling(20).std()
    vol_monthly = vol_20d.resample("ME").last()
    low_vol = -vol_monthly

    # ---- 价值 & 规模: 静态截面（用最新基本面近似回填每个月）
    bp_series = fundamentals["bp_ratio"]
    log_mcap_series = np.log(fundamentals["market_cap"].replace(0, np.nan))

    # 构建因子 DataFrame: index=日期, columns=股票
    dates = monthly_close.index
    dates = dates[dates >= ANALYSIS_START]

    factors = {}
    for date in dates:
        if date not in momentum.index:
            continue
        row = {}
        for ticker in monthly_close.columns:
            vals = {}
            # 动量
            m = momentum.loc[date, ticker] if date in momentum.index else np.nan
            vals["momentum"] = m
            # 反转
            r = reversal.loc[date, ticker] if date in reversal.index else np.nan
            vals["reversal"] = r
            # 低波动
            v = low_vol.loc[date, ticker] if date in low_vol.index else np.nan
            vals["low_vol"] = v
            # 价值（静态）
            vals["bp"] = bp_series.get(ticker, np.nan)
            # 规模（静态）
            vals["log_mcap"] = log_mcap_series.get(ticker, np.nan)

            row[ticker] = vals
        factors[date] = pd.DataFrame(row).T

    print(f"[INFO] 因子计算完成，共 {len(factors)} 个截面月份")
    return factors


# ─── 5. 格式化为 alphalens 输入 ─────────────────────────
def format_for_alphalens(factors_dict, factor_name):
    """
    alphalens 要求:
      factor: MultiIndex (date, asset) -> float
      prices: DataFrame index=date, columns=asset
    """
    pieces = []
    for date, df in factors_dict.items():
        s = df[factor_name].dropna()
        s.index = pd.MultiIndex.from_product([[date], s.index],
                                              names=["date", "asset"])
        pieces.append(s)
    factor = pd.concat(pieces)
    factor.name = factor_name
    return factor


# ─── 6. 运行 alphalens 分析 ──────────────────────────────
def run_alphalens(factor_series, prices, factor_name):
    """对单个因子做 IC + 分层收益分析"""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  分析因子: {factor_name}")
    print(f"{'='*60}")

    try:
        factor_data = alphalens.utils.get_clean_factor_and_forward_returns(
            factor_series,
            prices,
            quantiles=5,
            periods=(5, 10, 21),  # 1周、2周、1月
        )
    except Exception as e:
        print(f"[WARN] {factor_name} get_clean_factor 失败: {e}")
        return

    # IC 分析
    ic = alphalens.performance.factor_information_coefficient(factor_data)
    print(f"\n--- {factor_name} IC 均值 ---")
    print(ic.mean().round(4))

    # 绘图并保存
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Factor: {factor_name}", fontsize=16)

    # IC 时序
    ax = axes[0, 0]
    ic.plot(ax=ax, alpha=0.7)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_title("IC Time Series")
    ax.legend(fontsize=8)

    # IC 柱状图 (月均)
    ax = axes[0, 1]
    monthly_ic = ic.resample("ME").mean()
    monthly_ic.plot(kind="bar", ax=ax, width=0.8, alpha=0.7)
    ax.set_title("Monthly Mean IC")
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.legend(fontsize=8)

    # 分层累计收益
    ax = axes[1, 0]
    mean_ret_by_q = alphalens.performance.mean_return_by_quantile(
        factor_data, by_group=False
    )[0]
    # 用 period=5D 的收益
    col = mean_ret_by_q.columns[0]
    mean_ret_by_q[col].plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(f"Mean Return by Quantile ({col})")
    ax.set_xlabel("Quantile")

    # 累计收益 (多空)
    ax = axes[1, 1]
    factor_returns = alphalens.performance.factor_returns(factor_data)
    cum_ret = (1 + factor_returns).cumprod()
    cum_ret.plot(ax=ax)
    ax.set_title("Cumulative Factor Returns (Long-Short)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{factor_name}_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] 图表已保存: {path}")


# ─── main ───────────────────────────────────────────────
def main():
    # 获取股票池
    tickers = get_sp500_tickers()

    # 下载数据
    close = download_prices(tickers)
    fundamentals = download_fundamentals(close.columns.tolist())

    # 计算因子
    factors_dict = compute_factors(close, fundamentals)

    # 准备 alphalens 用的价格（日频，需要覆盖 forward return 窗口）
    prices = close[close.index >= ANALYSIS_START]

    # 逐因子分析
    for fname in ["momentum", "bp", "log_mcap", "low_vol", "reversal"]:
        factor_series = format_for_alphalens(factors_dict, fname)
        run_alphalens(factor_series, prices, fname)

    print("\n[DONE] 所有因子分析完成，结果保存在:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
