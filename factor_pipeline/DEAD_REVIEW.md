# Dead 因子复盘 · F001–F003

> 2026-07-19 · 三条正交因子线均走完 S0–S6，预注册协议下判定 **dead**。

## 结果一览

| ID | 信号 | 标的 | 窗口 | n | 信号/事件均值 | 稀疏基准 | 超额 | 门禁 |
|----|------|------|------|---|--------------|---------|------|------|
| F001 | FOMC/CPI/NFP | ^VIX | T-5~T-1 | 81 FOMC | — | — | Bonferroni 不显著 | weak |
| F001b | FOMC 等 | **UVIX** | T-5~T-1 | 34 | **-4.8%** | +13.8% | -18.5pp | FAIL |
| F002 | BAA10Y Δ5≥10bps | **UVIX** | hold 5d | 25 | **-3.8%** | +14.5% | -18.3pp | FAIL |
| F003 | FOMC | **SPY** | T-5~T-1 | 268 | +0.13% | +0.24% | -0.11pp | FAIL |

## 死因（三类）

1. **FOMC 日线窗口已穷尽** — 同 T-5~T-1：UVIX 负收益，SPY 无 alpha（n=268 样本充足）。
2. **UVIX 做多 vol 路径结构性失败** — 日历与信用两条独立信号均跑输基准；VIX 命中率 50% vs UVIX 21%（衰减/路径差）。
3. **F003 文献未复现** — Lucca & Moench (2015) 在预注册 SPY 窗口上效应≈0，Post-2015 未衰减但全样本不显著。

## 流程上做对了什么

- S2 预注册：未事后改窗口/基准/α
- F001 研究层（^VIX）→ 产品层（F001b UVIX）分层
- F002/F003 与 F001 正交，独立否定
- 否定结果完整归档（case JSON + Alpha Deck + 分析 JSON）

## 数据坑

Yahoo `range=max` 对 SPY/UVIX 返回**周线**；须 `period1/period2` 拉日线（F003 已修）。

## 明确不追（防 p-hacking）

- F001 低 VIX×FOMC（探索 p=0.015，未 OOS）
- F003 窗口 sweep T-3/T-2（预注册禁止作主结论）
- F002 阈值/持有期事后扫描
- 在 dead 母线上加条件（须新建 case + 预注册）

## F004 立项方向（未开工）

与 dead 路径正交：vol 均值回归/做空 vol、term spread/MOVE、FOMC 日内、非 vol 的 risk-off rotation。

## 产出脚本与数据

| 因子 | 脚本 | 输出 |
|------|------|------|
| F001b | `dashboard/experiment_uvix_event.py` | `event_uvix_analysis.json`, `event_uvix_vix_mapping.json` |
| F002 | `dashboard/experiment_credit_vol_uvix.py` | `credit_vol_uvix_analysis.json` |
| F003 | `dashboard/experiment_spy_fomc_drift.py` | `spy_fomc_drift_analysis.json` |
