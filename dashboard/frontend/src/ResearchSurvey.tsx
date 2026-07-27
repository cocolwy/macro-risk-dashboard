import { motion } from 'framer-motion';

const staggerContainer = {
  animate: { transition: { staggerChildren: 0.04 } },
};
const staggerItem = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.3, ease: [0.25, 0.1, 0.25, 1] } },
};

interface Direction {
  id: string;
  title: string;
  titleZh: string;
  priority: 'P0' | 'P1' | 'P2' | 'P3';
  sharpe: string;
  status: string;
  statusColor: string;
  what: string;
  why: string;
  risk: string;
  linkTo: string;
  tags: string[];
}

const DIRECTIONS: Direction[] = [
  {
    id: 'PEAD',
    title: 'Post-Earnings Announcement Drift',
    titleZh: '盈余公告后漂移',
    priority: 'P0',
    sharpe: '0.6–0.8',
    status: '2020 后复苏',
    statusColor: 'var(--accent-green)',
    what: '当公司公布的季度利润"超出预期"时（比如分析师预测 EPS $1.50，实际是 $1.80），股价不会一步到位涨完。它会在接下来 60-90 天持续缓慢上涨。反过来，业绩不及预期的公司会持续下跌。这种现象叫"漂移"（drift）。',
    why: '行为金融学解释：投资者对新信息反应不足（under-reaction）。基金经理不能一听到消息就全仓买入，有审批、仓位限制、风控流程——所以价格调整很慢。虽然被学术界研究了 50 多年，但因为需要精确的盈余预期数据和较快的执行速度，一直没被完全套利掉。',
    risk: '小盘股 PEAD 更强但流动性差，大盘股 PEAD 弱但能执行。需要实时的 Analyst Consensus 数据。',
    linkTo: '你的 yfinance earnings 数据 + autoresearch 框架直接对接',
    tags: ['事件驱动', '60 天持仓', '季度频率'],
  },
  {
    id: 'Factor Timing',
    title: 'Factor Timing / Rotation',
    titleZh: '因子择时 / 轮动',
    priority: 'P0',
    sharpe: '+0.3–0.6 增量',
    status: '你的 crash 模型天然适配',
    statusColor: 'var(--accent-green)',
    what: '不是"选哪只股票"，而是"什么时候用哪个策略"。不同宏观环境下，因子表现天壤之别：经济复苏期 → Value 因子强（便宜的周期股反弹）；衰退期 → Quality 因子强（防御型公司抗跌）；过热期 → Momentum 因子强（趋势延续）。用宏观指标判断当前处于哪个 regime，然后动态调整因子权重。',
    why: '你已有的 macro crash probability 模型就是一个天然的 regime signal。crash 概率高 → 切到 Quality/Low Vol；crash 概率低 → 开 Value/Momentum。等于你不需要从零开始——只需要把现有模型的输出接到因子权重上。',
    risk: 'Regime 判断滞后（通常 1-2 个月），如果切换太慢会两边挨打。需要足够多的因子收益序列来验证。',
    linkTo: '和你现有 macro crash 模型天然互补',
    tags: ['宏观 regime', '月度/季度调仓', '与 crash 模型联动'],
  },
  {
    id: 'Quality',
    title: 'Quality / Profitability Factor',
    titleZh: '质量因子',
    priority: 'P1',
    sharpe: '0.81',
    status: '2024–2025 仍有效',
    statusColor: 'var(--accent-green)',
    what: '买入"好公司"的股票——高毛利率、高 ROE（净资产收益率）、低应计项（accruals，即利润中有多少是"纸面利润"）。学术名称 RMW = Robust Minus Weak。核心逻辑：赚真金白银的公司，长期股价表现更好。Novy-Marx 2013 发现，用 Gross Profit / Total Assets 就能有效区分好公司和差公司。',
    why: 'Fama-French 五因子模型中最稳定的因子，几乎在所有市场和时间段都有正溢价。可以理解为"买好东西很少亏"。',
    risk: '当前 AI 龙头（NVDA/MSFT/AAPL）天然就是 Quality 股票，可能存在"拥挤"风险——所有人都在买同一批股票。',
    linkTo: '多因子组合的基石之一',
    tags: ['基本面', '年度再平衡', '搭配 Momentum'],
  },
  {
    id: 'Momentum',
    title: 'Risk-managed Momentum',
    titleZh: '风险管理动量',
    priority: 'P1',
    sharpe: '0.61–0.94',
    status: '2024–2025 最强因子',
    statusColor: 'var(--accent-green)',
    what: '买入过去 2-12 个月涨得最多的股票（赢家），卖出跌得最多的股票（输家）。简单说就是"强者恒强"——上涨趋势的股票倾向于继续上涨。学术发现于 1993 年（Jegadeesh & Titman），是最持久的市场异常之一。但需要"风险管理"：2009 年 3 月市场反转时，纯动量策略单月亏损 -40%（赢家突然变输家）。现代做法是在市场波动率过高或进入熊市时自动降仓。',
    why: '行为学解释：投资者对好消息反应缓慢（under-reaction），加上从众效应（herding），导致价格趋势持续。加了 regime gate 和 vol-scaling 之后，Sharpe 从 0.4 提升到 0.9+。',
    risk: '"动量崩溃"——市场从熊转牛的瞬间（如 2009.3、2020.3），之前的输家暴涨，赢家暴跌。必须有 regime gate。',
    linkTo: 'crash 模型提供 regime 信号',
    tags: ['趋势跟踪', '月度再平衡', '需要 regime gate'],
  },
  {
    id: 'Multi-Factor',
    title: 'Multi-Factor Combo',
    titleZh: '多因子组合',
    priority: 'P1',
    sharpe: '0.8–1.2',
    status: '机构标准方法',
    statusColor: 'var(--blue-400)',
    what: '不押注单一因子，而是把 Momentum + Value + Quality 组合在一起。核心逻辑：各因子之间低相关甚至负相关（Value 和 Momentum 的相关系数约 -0.3），组合后相当于"分散风险"。就像投资不能只买一只股票，做因子也不能只用一个因子。AQR、Dimensional 等头部量化基金的核心方法论。',
    why: '单因子有"寒冬"（Value 2010-2020 跑输十年），但组合后几乎没有连续亏损超过 2 年的情况。这是从"赌一把"升级为"系统化投资"的关键。',
    risk: '需要先完成 Quality、Momentum、Value 的单因子研究。组合方式选择（等权 vs IC 加权 vs 风险平价）本身也需要验证。',
    linkTo: '需要先完成 #1 #2 #4 单因子',
    tags: ['组合构建', '月度再平衡', '分散化'],
  },
  {
    id: 'Exotic',
    title: 'Exotic Signals (Skew / Kurtosis)',
    titleZh: '高阶矩信号',
    priority: 'P2',
    sharpe: '0.7–1.5（分行业）',
    status: '前沿研究',
    statusColor: 'var(--accent-yellow)',
    what: '传统因子只看"平均回报"和"波动率"（统计学的前两个矩）。Exotic Signals 看第三个矩（Skewness 偏度）和第四个矩（Kurtosis 峰度）。偏度衡量"收益是否对称"——负偏度意味着股票有"突然暴跌"的倾向（左尾风险），投资者要求更高回报作为补偿。峰度衡量"极端事件多不多"——高峰度 = 更多黑天鹅事件。',
    why: '这些信号能捕捉到 Value/Momentum/Quality 都遗漏的 alpha。就像 X 光看到了 MRI 看不到的东西。DelphicAlpha 验证了 30 个因子，skew/kurtosis 在科技和金融行业的 Sharpe 可达 1.5。这正是你想研究的"微观指标"方向。',
    risk: '学术前沿，回测容易过拟合。需要在不同时间段和市场上验证稳健性。',
    linkTo: '和宏观 crash 模型互补，微观维度',
    tags: ['微观指标', '行业差异大', '前沿 alpha'],
  },
  {
    id: 'VRP',
    title: 'VRP Harvesting',
    titleZh: '波动率风险溢价收割',
    priority: 'P2',
    sharpe: '0.5–0.8',
    status: '结构性溢价存在',
    statusColor: 'var(--accent-yellow)',
    what: '期权市场有一个长期存在的"保险溢价"。基金经理买 Put（看跌期权）保护持仓，就像买保险一样要多付钱。这导致"隐含波动率"（期权价格反映的预期波动）长期高于"实际波动率"（真正发生的波动）。这个差价就是 VRP。卖出期权 = 当保险公司收保费。1990 年以来 86% 的月份 IV > RV。',
    why: '结构性溢价（不是暂时的），类似保险公司的商业模式——大部分时间稳定收保费，但偶尔要赔大钱。你的 macro crash 模型可以告诉你什么时候"不要当保险公司"。',
    risk: '尾部风险极大。2018 年 "Volmageddon" XIV ETN 一天归零；2020.3 VIX 飙到 82。必须有严格的 regime gate，crash 概率高时完全停止。',
    linkTo: '和 crash 模型联动做风控',
    tags: ['期权 / VIX', 'regime gate', '尾部风险大'],
  },
  {
    id: 'Value',
    title: 'Value Factor',
    titleZh: '价值因子',
    priority: 'P3',
    sharpe: '0.4–0.5',
    status: '2010–2020 深度衰退',
    statusColor: 'var(--accent-red)',
    what: '买入"便宜"的股票（低 P/E 市盈率、低 P/B 市净率、高 Earnings Yield 盈利收益率），卖出"贵"的股票。核心逻辑：市场对陷入困境的公司过度悲观，估值过低，长期均值回归时这些股票跑赢。Graham & Dodd 在 1930 年代首创，Fama-French 在 1992 年系统化。',
    why: '历史最悠久的因子，但在 2010-2020 被 Growth（成长股）碾压了整整十年。学术界一度发表论文 "Value is Dead"。2021-2023 短暂复苏后又走弱。单独使用风险极高，但与 Quality + Momentum 组合后效果不错。',
    risk: '可以连续跑输 10 年。如果利率长期走低 + 科技股持续主导，Value 可能继续低迷。',
    linkTo: '作为组合因子的一部分而非独立策略',
    tags: ['估值', '年度再平衡', '当前弱势'],
  },
];

const PRIORITY_CONFIG: Record<string, { label: string; bg: string; color: string }> = {
  P0: { label: 'P0 立即做', bg: 'var(--accent-green-bg)', color: 'var(--accent-green)' },
  P1: { label: 'P1 第二批', bg: 'var(--blue-100)', color: 'var(--blue-500)' },
  P2: { label: 'P2 探索', bg: 'var(--accent-yellow-bg)', color: 'var(--accent-yellow)' },
  P3: { label: 'P3 备选', bg: 'var(--accent-red-bg)', color: 'var(--accent-red)' },
};

function DirectionCard({ d }: { d: Direction }) {
  const prio = PRIORITY_CONFIG[d.priority];
  return (
    <motion.div className="survey-card" variants={staggerItem}>
      <div className="survey-card-header">
        <div className="survey-card-title-row">
          <span className="survey-card-id">{d.id}</span>
          <span className="survey-card-prio" style={{ background: prio.bg, color: prio.color }}>
            {prio.label}
          </span>
        </div>
        <h3 className="survey-card-title">{d.titleZh}</h3>
        <p className="survey-card-en">{d.title}</p>
        <div className="survey-card-meta-row">
          <span className="survey-card-sharpe">Sharpe {d.sharpe}</span>
          <span className="survey-card-status" style={{ color: d.statusColor }}>{d.status}</span>
        </div>
      </div>

      <div className="survey-card-body">
        <div className="survey-card-section">
          <div className="survey-card-label">这是什么？</div>
          <p>{d.what}</p>
        </div>
        <div className="survey-card-section">
          <div className="survey-card-label">为什么能赚钱？</div>
          <p>{d.why}</p>
        </div>
        <div className="survey-card-section">
          <div className="survey-card-label">风险</div>
          <p>{d.risk}</p>
        </div>
        <div className="survey-card-section">
          <div className="survey-card-label">和项目的关系</div>
          <p className="survey-card-link-text">{d.linkTo}</p>
        </div>
      </div>

      <div className="survey-card-tags">
        {d.tags.map(t => (
          <span key={t} className="survey-tag">{t}</span>
        ))}
      </div>
    </motion.div>
  );
}

export function ResearchSurvey() {
  return (
    <div className="survey-container">
      <header className="survey-header">
        <h1>Research Survey</h1>
        <p className="survey-subtitle">
          美股公开因子与策略调研 — 8 个可研究方向，每个附带一键启动 Prompt
        </p>
        <p className="survey-hint">
          详细 Prompt 在 <code>todo.md</code>，复制到 Cursor 新窗口即可启动研究
        </p>
      </header>

      <section className="survey-overview">
        <div className="survey-overview-grid">
          <div className="survey-stat-card">
            <div className="survey-stat-value">8</div>
            <div className="survey-stat-label">研究方向</div>
          </div>
          <div className="survey-stat-card">
            <div className="survey-stat-value">2</div>
            <div className="survey-stat-label">P0 立即做</div>
          </div>
          <div className="survey-stat-card">
            <div className="survey-stat-value">3</div>
            <div className="survey-stat-label">P1 第二批</div>
          </div>
          <div className="survey-stat-card">
            <div className="survey-stat-value">0.8–1.2</div>
            <div className="survey-stat-label">组合 Sharpe 目标</div>
          </div>
        </div>
      </section>

      <section className="survey-key-insight">
        <h2>核心结论</h2>
        <p>
          单因子时代基本结束了。2005 年后大盘股中位异常收益仅 ~1%/年。
          但经过<strong>现代化改造</strong>的策略仍然可行：regime 过滤 + 波动率缩放 + 多因子组合。
          最终目标是 Multi-Factor Combo（Sharpe 0.8–1.2），单因子研究是为组合服务的基石。
        </p>
      </section>

      <motion.div
        className="survey-grid"
        variants={staggerContainer}
        initial="initial"
        animate="animate"
      >
        {DIRECTIONS.map(d => (
          <DirectionCard key={d.id} d={d} />
        ))}
      </motion.div>

      <section className="survey-graveyard">
        <h2>策略坟场 — 已经不能赚钱的策略</h2>
        <div className="survey-graveyard-grid">
          {[
            { name: '经典配对交易', cause: 'ETF 套利 + HFT 更快的参与者抢走了利润' },
            { name: '低 P/E 简单选股', cause: '2010–2020 被 Growth 碾压，Value 因子失效' },
            { name: '日历效应 (January Effect)', cause: '被学术出版后迅速套利殆尽' },
            { name: '简单均值回归', cause: 'HFT 将反转周期压缩到 microseconds，散户无法参与' },
            { name: '小盘效应 (Size Factor)', cause: '1980 年后持续衰减，流动性成本抵消超额收益' },
          ].map(s => (
            <div key={s.name} className="survey-grave-item">
              <span className="survey-grave-name">{s.name}</span>
              <span className="survey-grave-cause">{s.cause}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
