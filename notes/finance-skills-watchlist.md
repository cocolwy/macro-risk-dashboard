# Finance Agent Skills — Watchlist

调研日期：2026-07-20  
Star 抓取：2026-07-20（GitHub API）  
用途：有空再玩的开源金融 Skill / 插件清单。

---

## PENDING — 需要 Claude（Cowork / Claude Code / Managed Agents）

| 仓库 | ★ | Fork | 本地路径 | 状态 | 备注 |
|------|---|------|----------|------|------|
| [anthropics/financial-services](https://github.com/anthropics/financial-services) | **33,624** | 4,963 | `/Users/licoco/Developer/vibecoding/financial-services` | **PENDING** | 已 shallow clone；完整功能不在 Cursor。等有 Claude 环境再玩 |

**为何 pending：** 官方只支持 Cowork 插件 / Claude Code marketplace / Managed Agents。Cursor 只能手工抄 `SKILL.md`，拿不到 slash、命名 Agent、M365 Excel、插件级 MCP 装配。

**以后怎么玩（备忘）：**
1. 装 core：`financial-analysis`
2. 再装 vertical：`investment-banking` / `equity-research` / …
3. slash：`/comps` `/dcf` `/lbo` `/earnings` `/one-pager` `/ic-memo`
4. https://www.anthropic.com/news/finance-agents

---

## 已拉取 · Cursor 可玩

| 仓库 | ★ | Fork | 本地 clone | 已装进 quant |
|------|---|------|------------|--------------|
| [himself65/finance-skills](https://github.com/himself65/finance-skills) | **3,031** | 348 | `/Users/licoco/Developer/vibecoding/finance-skills` | `.agents/skills/`（**全部 25 个**） |

**安装状态：** 全量已装（`npx skills add … -s '*' -a cursor --copy`）。  
开箱即用主要是 market-analysis（yfinance）；social / Funda / TradingView 等需额外工具或 API key。

---

## 待看（还没拉）

| # | 仓库 | ★ | Fork | 定位 | Cursor | 适合先玩如果… |
|---|------|---|------|------|--------|----------------|
| 2 | [tradermonty/claude-trading-skills](https://github.com/tradermonty/claude-trading-skills) | **2,449** | 580 | regime / screener / 仓位 / 交易日记 | 中高 | 想要**交易纪律 + 日复盘流程** |
| 3 | [agiprolabs/claude-trading-skills](https://github.com/agiprolabs/claude-trading-skills) | **232** | 52 | 67 个 trading/DeFi/quant skills | 高 | 想扫一大目录、含 DeFi/on-chain |
| 4 | [zubair-trabzada/ai-trading-claude](https://github.com/zubair-trabzada/ai-trading-claude) | **200** | 106 | 多 agent 个股打分 | 中 | 想玩**单票综合研报**交互 |
| 5 | [mphinance/alpha-skills](https://github.com/mphinance/alpha-skills) | **15** | 3 | quant + 市场情报大杂烩 | 中 | 后看 |
| 6 | [GAJETOso/financeskills](https://github.com/GAJETOso/financeskills) | **4** | 1 | 会计 / 审计 / IFRS·GAAP | 中高 | 偏财务岗 |
| 7 | [Viprasol-Tech/viprasol-agent-skills](https://github.com/Viprasol-Tech/viprasol-agent-skills) | **0** | 0 | 财报、earnings call、回测过拟合 | 中高 | ★ 尚无 |

---

## 和本仓库（quant）的关系（备忘）

- Anthropic（33.6k★）：机构办公流 → **PENDING**，有 Claude 再开
- himself65：已装 11 个 market-analysis skill 进 Cursor
- tradermonty：下一候选（交易纪律 / regime）
- 因子 / Event×VIX：仍以本仓库实验脚本为主
