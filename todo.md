# Research Directions — 美股因子与策略待研清单

> 每个方向附带一段 **Prompt**。在 Cursor 里新开一个 Chat 窗口，粘贴对应 Prompt 即可启动该研究。
> 完成后在 `autoresearch/` 下用 `/experiment` 跑验证。

---

## 1. Quality / Profitability Factor — 质量因子

**是什么：** 买入高毛利率、高 ROE、低应计项的"好公司"，做多这些公司、做空低质量公司。学术名称 RMW（Robust Minus Weak）。Fama-French 五因子模型中最稳定的因子之一，历史 Sharpe ~0.81。核心逻辑是高质量公司盈利能力强、财务健康，长期产生更稳定的超额收益。

**适用场景：** 长期持有（年度再平衡），适合搭配 Momentum 使用。

**状态：** 2024–2025 依然有效，但 AI 龙头股（NVDA/MSFT）天然是 Quality 股，可能存在拥挤风险。

```
Prompt:

在 autoresearch/ 框架下研究 Quality/Profitability 因子。

目标：
1. 用 yfinance 获取 S&P 500 成分股的毛利率（Gross Profit / Total Assets，Novy-Marx 定义）、ROE、Accruals
2. 每月按 Quality 分位排序，构建 Long-Short 组合（Top 20% vs Bottom 20%）
3. 计算月度收益序列，回测 2010-2025
4. 输出：Sharpe、年化收益、最大回撤、Turnover
5. 做 Fama-French 5-factor regression（用 Kenneth French 网站数据），看 alpha 是否显著
6. 与 Momentum 因子做相关性检查，评估组合价值

结果存入 autoresearch/results.tsv，实验文件为 autoresearch/experiment_auto.py。
工作目录：/Users/licoco/Developer/vibecoding/quant
```

---

## 2. Risk-managed Momentum — 风险管理动量

**是什么：** 买入过去 2-12 个月涨幅最大的股票（赢家），卖出跌幅最大的股票（输家）。Momentum 是学术研究中最持久的异常之一（Jegadeesh & Titman 1993），但存在"动量崩溃"风险（如 2009 年 3 月市场反转时动量策略单月亏损 -40%）。现代做法是加入 regime gate：当市场波动率过高或处于 bear market 时自动降仓，避开崩溃。历史 Sharpe 0.61–0.94。

**适用场景：** 月度再平衡，中期持有。

**状态：** 2024–2025 年 Momentum 是表现最好的因子，但需要 vol-scaling 和 regime 过滤。

```
Prompt:

在 autoresearch/ 框架下研究 Risk-managed Momentum 策略。

目标：
1. 用 yfinance 获取 S&P 500 成分股过去 12 个月收益（跳过最近 1 个月，避免短期反转）
2. 按 12-1 Momentum 分位排序，构建 Top/Bottom 20% Long-Short
3. 加入 Regime Gate：当 VIX > 28 或 SPY 200 日均线以下时，组合仓位降至 50%
4. 对比三个版本的表现：(a) 原始动量 (b) vol-scaled 动量 (c) regime-gated 动量
5. 回测 2005-2025，重点标注 2009.3 和 2020.3 的表现
6. 输出：Sharpe、年化收益、最大回撤、月度胜率
7. 对比文献中动量崩溃的时间节点，验证 regime gate 是否成功规避

结果存入 autoresearch/results.tsv，实验文件为 autoresearch/experiment_auto.py。
工作目录：/Users/licoco/Developer/vibecoding/quant
```

---

## 3. PEAD — 盈余公告后漂移

**是什么：** Post-Earnings Announcement Drift 的缩写。当一家公司公布的季度盈余"超预期"（比分析师一致预期高）时，股价通常不会一步到位反映好消息，而是在接下来 60-90 天持续缓慢上涨（反之亦然，miss 预期的股票持续下跌）。这是行为金融学中最著名的异常之一（Ball & Brown 1968 首次发现），被认为是投资者对新信息反应不足（under-reaction）导致的。

**为什么还能赚钱：** 虽然被研究了 50+ 年，但因为需要精确的盈余预期数据（Analyst Estimates）和较快的执行速度，散户难以系统化利用，机构之间也因持仓约束（不能频繁交易）而未完全套利掉。2020 年后 PEAD 反而有复苏趋势，年化 ~8%。

**适用场景：** 事件驱动，每个 earnings season 批量建仓，持有 60 天。

```
Prompt:

在 autoresearch/ 框架下研究 PEAD（盈余公告后漂移）策略。

目标：
1. 用 yfinance 获取 S&P 500 成分股的历史 earnings dates 和 analyst estimates
2. 计算 Standardized Unexpected Earnings (SUE) = (Actual EPS - Consensus EPS) / Std(EPS)
3. 在 earnings 公布后 T+1 建仓，持有 60 个交易日
4. 按 SUE 分位构建多空组合：Top 20%（正面惊喜最大）vs Bottom 20%（负面惊喜最大）
5. 回测 2015-2025，按年拆解收益
6. 控制变量：公司市值、行业、momentum（排除 momentum 解释）
7. 输出：年化 alpha、Sharpe、每季收益分布、最大回撤

结果存入 autoresearch/results.tsv，实验文件为 autoresearch/experiment_auto.py。
工作目录：/Users/licoco/Developer/vibecoding/quant
```

---

## 4. Value Factor — 价值因子

**是什么：** 买入"便宜"的股票（低 P/E、低 P/B、高 Earnings Yield），卖出"贵"的股票。Graham & Dodd 1930s 首创，Fama-French 1992 系统化。核心逻辑是市场对陷入困境的公司过度悲观，导致估值过低，长期均值回归时这些股票跑赢。

**风险：** Value 在 2010-2020 被 Growth 碾压了整整十年（"Value is dead" 论文），2021-2023 有过短暂复苏但不持续。单用 Value 风险极高。

**适用场景：** 与 Quality + Momentum 组合使用效果最好，年度再平衡。

```
Prompt:

在 autoresearch/ 框架下研究 Value 因子。

目标：
1. 用 yfinance 获取 S&P 500 成分股的 P/E、P/B、EV/EBITDA、Earnings Yield
2. 构建 Composite Value Score（多指标等权/IC 加权）
3. 按 Value Score 分位排序，构建 Top/Bottom 20% Long-Short
4. 回测 2005-2025，特别标注 2010-2020 "Value 寒冬" 和 2021-2023 "Value 复苏"
5. 与 Momentum、Quality 因子做相关性矩阵
6. 测试 Value + Quality + Momentum 三因子等权组合 vs 各单因子的表现差异
7. 输出：Sharpe、年化收益、最大回撤、Fama-French alpha

结果存入 autoresearch/results.tsv，实验文件为 autoresearch/experiment_auto.py。
工作目录：/Users/licoco/Developer/vibecoding/quant
```

---

## 5. Multi-Factor Combo — 多因子组合

**是什么：** 不押注单一因子，而是把 Momentum + Value + Quality（+ 可选 Low Vol）组合起来。核心逻辑是各因子之间低相关甚至负相关（比如 Value 和 Momentum 相关系数 ~-0.3），组合后能显著提升 Sharpe、降低单因子崩溃风险。AQR、Dimensional 等头部量化基金的核心方法论。历史 Sharpe 0.8–1.2。

**适用场景：** 这是最稳健的长期策略，月度再平衡。

```
Prompt:

在 autoresearch/ 框架下研究多因子组合策略。

目标：
1. 复用前面 Quality、Momentum、Value 的单因子信号
2. 构建三种组合方式：(a) 等权 (b) IC 加权 (c) 风险平价（Risk Parity）
3. 每月再平衡，回测 2010-2025
4. 做因子间相关性矩阵（月度收益的 Pearson & Spearman）
5. 对比单因子 vs 组合的：Sharpe、Max DD、Calmar Ratio、月度胜率
6. 测试加入 Low Volatility 因子（买低 beta 卖高 beta）后的边际贡献
7. 做 rolling 3-year Sharpe 图，看组合的稳定性
8. 输出：最优组合方式、权重、年化收益、最大回撤

结果存入 autoresearch/results.tsv，实验文件为 autoresearch/experiment_auto.py。
工作目录：/Users/licoco/Developer/vibecoding/quant
```

---

## 6. Exotic Signals — 高阶矩信号（Skew / Kurtosis）

**是什么：** 传统因子（Value、Momentum）只看收益的均值和标准差（前两个矩），而 Exotic Signals 关注第三和第四个统计矩。**Skewness**（偏度）衡量收益分布是否对称——负偏度说明股票有"左尾风险"（突然暴跌），投资者要求更高回报作为补偿。**Kurtosis**（峰度）衡量极端事件发生的频率——高峰度 = 更多黑天鹅。这些信号能捕捉到传统因子遗漏的 alpha，DelphicAlpha 团队验证了 30 个因子中 skew/kurtosis 在特定行业的 Sharpe 可达 0.7–1.5。

**适用场景：** 这就是你说的"微观指标"研究方向，和宏观 crash 模型互补。

```
Prompt:

在 autoresearch/ 框架下研究高阶矩因子（Skewness & Kurtosis）。

目标：
1. 用 yfinance 获取 S&P 500 成分股日度收益数据
2. 计算 rolling 60 日的：Skewness、Kurtosis、Downside Deviation、Max Drawdown
3. 构建两个因子：(a) Skew Factor（买正偏度卖负偏度 or 反向？需实证）(b) Tail Risk Factor（买低峰度卖高峰度）
4. 按行业分组（GICS Sector）检验因子在不同行业的有效性
5. 回测 2010-2025，构建 Top/Bottom 20% Long-Short
6. 与 Momentum、Value、Quality 做相关性检查（应该低相关才有组合价值）
7. 测试与现有 macro crash 模型的信号相关性
8. 输出：分行业 Sharpe、全市场 Sharpe、与传统因子的相关矩阵

结果存入 autoresearch/results.tsv，实验文件为 autoresearch/experiment_auto.py。
工作目录：/Users/licoco/Developer/vibecoding/quant
```

---

## 7. Factor Timing / Rotation — 因子择时

**是什么：** 不是选哪些股票，而是选什么时候用哪个因子。核心思路是不同宏观环境（expansion / recession / recovery / overheating）下不同因子表现差异很大。比如经济复苏期 Value 表现好（便宜的周期股反弹），衰退期 Quality 表现好（防御性强）。你的 macro crash 模型天然可以扩展为 factor rotation 的 regime 信号。

**适用场景：** 月度 / 季度调整因子权重，和你现有的宏观风控模型高度互补。

```
Prompt:

在 autoresearch/ 框架下研究 Factor Timing / Rotation 策略。

目标：
1. 定义 4 个宏观 regime：Expansion、Slowdown、Recession、Recovery
   - 使用 ISM PMI、yield curve slope (10Y-2Y)、unemployment claims 作为 regime 指标
2. 获取 Quality、Momentum、Value、Low Vol 四个因子的月度收益序列（可用 Kenneth French 数据库）
3. 分析每个 regime 下各因子的平均月度收益和 Sharpe
4. 构建 Rotation 策略：根据当前 regime 调整因子权重
5. 对比三种方案：(a) 静态等权 (b) Regime-based rotation (c) 你的 macro crash probability 作为 rotation signal
6. 回测 2005-2025，输出：Sharpe、年化收益、最大回撤
7. 特别验证 2008、2020、2022 年衰退期的因子切换是否及时

结果存入 autoresearch/results.tsv，实验文件为 autoresearch/experiment_auto.py。
工作目录：/Users/licoco/Developer/vibecoding/quant
```

---

## 8. VRP Harvesting — 波动率风险溢价

**是什么：** Volatility Risk Premium 的缩写。市场上的期权（options）有一个长期存在的"保险溢价"——买保险的人（买 Put 保护持仓的基金经理）愿意多付钱，导致隐含波动率（IV，期权价格反映的预期波动率）长期高于实际波动率（RV，实际发生的波动率）。卖出这个溢价就是 VRP Harvesting。最简单的形式是卖 SPX Put Spread 或做空 VIX 期货。1990 年以来 86% 的月份 IV > RV，结构性溢价存在。

**风险：** 尾部风险极大——2018 年 "Volmageddon" XIV ETN 一天归零，2020 年 3 月 VIX 飙到 82。需要严格的 regime gate。

```
Prompt:

在 autoresearch/ 框架下研究 VRP（波动率风险溢价）策略。

目标：
1. 用 yfinance 获取 VIX 指数历史数据和 SPY 日度收益
2. 计算 Variance Risk Premium = VIX² - RV²（RV 用 SPY 过去 21 日实现波动率）
3. 分析 VRP 的统计特征：均值、分布、持续性、regime 依赖
4. 构建信号：当 VRP > 阈值（溢价充足）时做空波动率（模拟卖 Put）
5. 加入 Regime Gate：当 VIX > 30 或 macro crash probability > 0.5 时停止做空
6. 回测 2005-2025，重点标注 2008.10、2018.2（Volmageddon）、2020.3
7. 与你现有的 macro crash 模型联动：crash probability 升高时自动关闭 VRP 头寸
8. 输出：Sharpe、年化收益、最大回撤、月度胜率、尾部最大亏损

结果存入 autoresearch/results.tsv，实验文件为 autoresearch/experiment_auto.py。
工作目录：/Users/licoco/Developer/vibecoding/quant
```

---

## 优先级建议

| 优先级 | 方向 | 理由 |
|---|---|---|
| **P0 立即做** | #3 PEAD | 你已有 yfinance earnings 数据，和 autoresearch 框架直接对接 |
| **P0 立即做** | #7 Factor Timing | 和你现有 macro crash 模型天然互补 |
| **P1 第二批** | #1 Quality + #2 Momentum | 经典因子，是多因子组合的基石 |
| **P1 第二批** | #5 Multi-Factor Combo | 需要先完成 #1 #2 #4 |
| **P2 探索** | #6 Exotic Signals | "微观指标"研究方向，独立于宏观模型 |
| **P2 探索** | #8 VRP | 需要期权数据或 VIX 期货数据，复杂度高 |
| **P3 备选** | #4 Value | 近年表现差，优先级最低 |
