# Factor Pipeline · 挖因子流水线

把「灵机一动」变成可复现、可否定、可归档的研究案例。

## 你要读的两份文档

| 文件 | 用途 |
|------|------|
| [PIPELINE.md](./PIPELINE.md) | **公共协议**：S0–S7 每一步的关键注意点与门禁 |
| [DEAD_REVIEW.md](./DEAD_REVIEW.md) | **Dead 复盘**：F001–F003 否定结果摘要 |
| 本 README | **使用方法**：如何新建因子、怎么填、怎么上看板 |

## 目录结构

```
factor_pipeline/
├── README.md                 ← 你在这里
├── PIPELINE.md               ← 公共注意点（必读）
├── schema/
│   └── factor_case_template.json
└── cases/
    └── F001_uvix_event_calendar.json   ← 第一个因子完整记录
```

看板同步数据：

- `dashboard/data/factor_research_log.json` → Alpha Deck（`#factorlab`）
- 实验细节页（若有）如 Event × VIX（`#ch2-2`）

## 新挖一个因子：五步用法

### 1. 复制模板

```bash
cp factor_pipeline/schema/factor_case_template.json \
   factor_pipeline/cases/F00N_short_slug.json
```

把 `id` 改成 `F00N`，填 `name` / `name_zh` / `created`。

### 2. 先填 S0 → S2，再写代码

按 [PIPELINE.md](./PIPELINE.md) 顺序：

1. **S0** 写清：交易标的（如 UVIX）、动作、持有期、成功指标  
2. **S1** 写可证伪假设 + 失效条件  
3. **S2** 预注册窗口 / 基准 / α — **写完再跑主检验**

未完成 S0–S2 就扫窗口，只允许记入 S4（探索），不能当主结论。

### 3. 实现与检验（S3–S5）

- 脚本建议放在 `dashboard/experiment_*.py`
- 结果 JSON 放在 `dashboard/data/`
- 探索结果在 case 里标 `"exploratory": true`
- 主结论只引用 S2 预注册设定

### 4. 可交易性（S6）— 有实盘目标时必做

若目标是 ETF/ETN（如 UVIX），必须单独回答：

- 研究序列（^VIX）结论如何映射到产品收益？
- 成本、衰减、入场时刻是否定义？
- 产品层是否用**同一预注册协议**重跑？

未过 S6，Alpha Deck 状态不要标 `confirmed`。

### 5. 归档到看板（S7）

1. 更新 `factor_pipeline/cases/F00N_*.json` 的 `stage_checklist`  
2. 同步摘要到 `dashboard/data/factor_research_log.json`  
3. 复制到前端：

```bash
cp dashboard/data/factor_research_log.json \
   dashboard/frontend/public/data/factor_research_log.json
```

4. 打开看板 `#factorlab` 检查卡片与流水线进度  
5. （可选）为实验页加路由，链到 `experiment_hash`

## 状态怎么选

| status | 含义 |
|--------|------|
| `pending` | 还在 S0–S2 |
| `weak` | 主检验弱，或目标与检验错位 / 不可交易 |
| `conditional` | 仅某 regime 有信号，待独立确认 + S6 |
| `confirmed` | S5 确认且 S6 产品层通过 |
| `dead` | 明确否定，保留档案防重复劳动 |

## 第一个因子示例（F001）

案例文件：[`cases/F001_uvix_event_calendar.json`](./cases/F001_uvix_event_calendar.json)

要点（2026-07-19 重审）：

- **真实目标：** 交易 UVIX  
- **实际检验：** 几乎全在 ^VIX 上 → S0/S6 标的错位  
- **VIX 主检验：** 预注册后不显著 → 研究层 weak/conditional  
- **交易结论：** 不建议部署 UVIX 策略；残留线索是「低 VIX × FOMC」，须用 UVIX 数据重新预注册  

看板入口：`#factorlab` · 实验页：`#ch2-2`

## 与 AI / 多 Agent 协作时

- 改 case 或看板 JSON 前，先声明文件清单（见 `.cursor/rules/multi-agent-file-safety.mdc`）  
- 每完成一个逻辑单元就 commit，避免被其他会话覆盖  
- 公共注意点以 `PIPELINE.md` 为准，不要在聊天里另起一套标准
