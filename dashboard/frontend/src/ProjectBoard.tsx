const PHASES = [
  {
    id: 'foundation',
    title: 'Multi-Agent Foundation',
    weeks: 'Week 1-2',
    status: 'complete' as const,
    color: '#3a82d6',
    tasks: [
      { name: '6-Agent Pipeline Architecture', done: true, detail: 'DataEngineer → Researcher → Backtester → Critic → RiskManager → PM' },
      { name: 'MessageBus Communication', done: true, detail: 'Shared state store with pub/sub pattern' },
      { name: '5-Factor Model Implementation', done: true, detail: 'Momentum, Value (B/P), Size, Low-Vol, Reversal' },
      { name: 'SP500 Core Universe (107 stocks)', done: true, detail: 'yfinance OHLCV + fundamentals download' },
      { name: 'Alphalens Factor Analysis', done: true, detail: 'IC, quantile returns, factor charts' },
      { name: 'Flask Dashboard', done: true, detail: 'Real-time pipeline monitoring' },
    ],
  },
  {
    id: 'optimization',
    title: 'Factor Optimization & A/B Tests',
    weeks: 'Week 3',
    status: 'complete' as const,
    color: '#8b5cf6',
    tasks: [
      { name: 'PCA Orthogonalization A/B', done: true, detail: 'Decorrelate factors before IC-weighting' },
      { name: 'Industry Neutralization A/B', done: true, detail: 'Cross-sectional z-score within sectors' },
      { name: 'Universe Size Comparison', done: true, detail: 'SP500-107 vs Russell-1000' },
      { name: 'Golden Baseline Locked', done: true, detail: '29.8% gross, Sharpe 1.68, MDD -15.8%' },
      { name: 'Lookahead Bias Identification', done: true, detail: 'Full-sample PCA inflates results' },
    ],
  },
  {
    id: 'qlib',
    title: 'Qlib Migration & Validation',
    weeks: 'Week 3-4',
    status: 'complete' as const,
    color: '#16a34a',
    tasks: [
      { name: 'OHLCV → Qlib Binary Dump', done: true, detail: 'calendars/instruments/features/*.day.bin' },
      { name: 'Price Match Verification', done: true, detail: 'Qlib reads match yfinance prices exactly' },
      { name: 'Baseline Backtest Replay', done: true, detail: 'Qlib reproduces baseline within 0.24%' },
      { name: 'Reusable Backtest Library', done: true, detail: 'qlib_backtest_lib.py — shared by all experiments' },
    ],
  },
  {
    id: 'lgbm-v1',
    title: 'LightGBM Walk-Forward (v1)',
    weeks: 'Week 4',
    status: 'complete' as const,
    color: '#ea580c',
    tasks: [
      { name: 'Extended Factor Panel (2016-2024)', done: true, detail: '9,737 rows × 91 months × 107 instruments' },
      { name: 'Walk-Forward Protocol', done: true, detail: '24M train / 3M test / 3M step, 23 folds' },
      { name: 'Per-Fold PCA (No Lookahead)', done: true, detail: 'Honest feature engineering per fold' },
      { name: 'LambdaRank LightGBM', done: true, detail: 'Decile labels + group, 120 rounds' },
      { name: 'Honest Linear Baseline', done: true, detail: '6-month trailing IC-weighted PCs' },
      { name: 'Key Finding: ML < Linear', done: true, detail: 'LightGBM 11.3% vs Linear 15.3% (gross)' },
    ],
  },
  {
    id: 'lgbm-v2',
    title: 'Regression + Wide Universe (v2)',
    weeks: 'Week 4+',
    status: 'complete' as const,
    color: '#d6457a',
    tasks: [
      { name: 'Russell-1000 Universe (~459 stocks)', done: true, detail: '41,769 rows, ~11k samples/fold (4.3× v1)' },
      { name: 'Regression Objective (raw returns)', done: true, detail: 'Direct prediction of forward returns' },
      { name: 'LambdaRank on Wide Universe', done: true, detail: 'Same ranking approach, more data' },
      { name: 'Raw Features vs PCA', done: true, detail: 'Trees benefit from non-orthogonalized inputs' },
      { name: 'Best LightGBM: Raw+Rank', done: true, detail: '16.0% gross — competitive with linear' },
      { name: 'Qlib Backtest on Russell 1000', done: true, detail: '459 instruments in Qlib binary format' },
    ],
  },
];

interface BacktestResult {
  model: string;
  universe: string;
  features: string;
  grossAnn: number;
  sharpe: number;
  mdd: number;
  netAnn: number;
  highlight?: boolean;
}

const RESULTS: BacktestResult[] = [
  { model: 'IC-Weighted Linear', universe: 'Russell-459', features: 'PCA', grossAnn: 17.0, sharpe: 0.86, mdd: -25.7, netAnn: 15.4, highlight: true },
  { model: 'LambdaRank LightGBM', universe: 'Russell-459', features: 'Raw 5-Factor', grossAnn: 16.0, sharpe: 0.51, mdd: -26.2, netAnn: 13.7 },
  { model: 'IC-Weighted Linear', universe: 'Russell-459', features: 'Raw 5-Factor', grossAnn: 13.8, sharpe: 0.63, mdd: -21.0, netAnn: 12.1 },
  { model: 'LambdaRank LightGBM', universe: 'Russell-459', features: 'PCA', grossAnn: 11.2, sharpe: 0.37, mdd: -31.3, netAnn: 8.9 },
  { model: 'LambdaRank LightGBM', universe: 'SP500-107', features: 'PCA', grossAnn: 11.3, sharpe: 0.61, mdd: -23.9, netAnn: 9.5 },
  { model: 'IC-Weighted Linear', universe: 'SP500-107', features: 'PCA', grossAnn: 15.3, sharpe: 0.95, mdd: -23.0, netAnn: 13.9 },
  { model: 'Regression LightGBM', universe: 'Russell-459', features: 'Raw 5-Factor', grossAnn: 7.6, sharpe: 0.25, mdd: -37.0, netAnn: 5.2 },
  { model: 'Regression LightGBM', universe: 'Russell-459', features: 'PCA', grossAnn: 6.4, sharpe: 0.20, mdd: -38.4, netAnn: 3.9 },
];

const AGENTS = [
  { name: 'DataEngineer', role: '数据获取', icon: '📊', color: '#4CAF50', desc: 'yfinance OHLCV + fundamentals' },
  { name: 'Researcher', role: '因子研究', icon: '🔬', color: '#2196F3', desc: '5 factors, PCA, industry neutral' },
  { name: 'Backtester', role: '因子检验', icon: '📈', color: '#FF9800', desc: 'Alphalens IC, quantile returns' },
  { name: 'Critic', role: '质量审查', icon: '🔍', color: '#9C27B0', desc: 'Lookahead, survivorship, overfit' },
  { name: 'RiskManager', role: '风险管理', icon: '🛡️', color: '#F44336', desc: 'VaR, drawdown, stress tests' },
  { name: 'PortfolioManager', role: '组合构建', icon: '💼', color: '#607D8B', desc: 'IC-weighted Top-20 monthly' },
];

const FINDINGS = [
  {
    title: 'ML 不是免费午餐',
    detail: 'Walk-forward 评估下，LightGBM 始终未能在风险调整基础上超越简单 IC 加权线性模型。选择正确的工具比默认使用复杂模型更重要。',
    icon: '⚖️',
    color: '#3a82d6',
  },
  {
    title: 'PCA 帮助线性但伤害树',
    detail: 'PCA 正交化让线性模型从 13.8% 提升到 17.0%，但让 LightGBM 从 16.0% 下降到 11.2%。正交化移除了树可以利用的因子交互信息。',
    icon: '🔄',
    color: '#8b5cf6',
  },
  {
    title: '更宽宇宙改善 ML',
    detail: 'Russell-459 让 LightGBM 每折训练从 ~2.6k 增加到 ~11k 样本，年化从 11.3% 提升到 16.0%。更多截面数据是关键。',
    icon: '📐',
    color: '#16a34a',
  },
  {
    title: '全样本 PCA 存在严重前瞻偏差',
    detail: '锁定 baseline (30%) 与诚实 walk-forward (17%) 之间的巨大差距确认了全样本特征工程会悄然膨胀回测结果。',
    icon: '⚠️',
    color: '#ea580c',
  },
];

const FACTORS = [
  { name: 'Momentum', def: 'ret_2_12', desc: '2-12月累计收益（跳过近1月）', rationale: '价格趋势延续' },
  { name: 'Value (B/P)', def: 'book/price', desc: '账面价值/市价，延迟2月', rationale: '均值回归到基本面' },
  { name: 'Size', def: 'log(mcap)', desc: 'log市值，延迟2月', rationale: '小盘股溢价' },
  { name: 'Low Vol', def: '-20D σ', desc: '-20日滚动标准差', rationale: '低风险异象' },
  { name: 'Reversal', def: '-ret_1M', desc: '-1月收益率', rationale: '短期均值回归' },
];

function statusBadge(status: string) {
  const styles: Record<string, { bg: string; color: string; label: string }> = {
    complete: { bg: '#dcfce7', color: '#15803d', label: 'COMPLETE' },
    active: { bg: '#dceeff', color: '#3a82d6', label: 'ACTIVE' },
    planned: { bg: '#f3f4f6', color: '#6b7280', label: 'PLANNED' },
  };
  const s = styles[status] || styles.planned;
  return (
    <span style={{
      background: s.bg, color: s.color,
      padding: '2px 10px', borderRadius: 12,
      fontSize: 10, fontWeight: 700, letterSpacing: '0.5px',
    }}>
      {s.label}
    </span>
  );
}

function pctColor(val: number, thresholds: [number, number]) {
  if (val >= thresholds[1]) return '#15803d';
  if (val >= thresholds[0]) return '#b45309';
  return '#dc2626';
}

export function ProjectBoard() {
  const completedTasks = PHASES.reduce((sum, p) => sum + p.tasks.filter(t => t.done).length, 0);
  const totalTasks = PHASES.reduce((sum, p) => sum + p.tasks.length, 0);

  return (
    <div className="board-container">
      <header className="board-header">
        <div>
          <h1>Multi-Agent Quant Platform</h1>
          <p className="board-subtitle">项目进度看板 — Factor Research & Portfolio Construction</p>
        </div>
        <div className="board-stats">
          <div className="board-stat">
            <span className="board-stat-value">{completedTasks}/{totalTasks}</span>
            <span className="board-stat-label">Tasks</span>
          </div>
          <div className="board-stat">
            <span className="board-stat-value">{PHASES.length}</span>
            <span className="board-stat-label">Phases</span>
          </div>
          <div className="board-stat">
            <span className="board-stat-value">459</span>
            <span className="board-stat-label">Stocks</span>
          </div>
        </div>
      </header>

      {/* Agent Pipeline */}
      <section className="board-card">
        <div className="ab-header">
          <h2>Agent Pipeline Architecture</h2>
          <span className="ab-badge">6 AGENTS</span>
          <span className="ab-badge" style={{ background: '#dcfce7', color: '#15803d', border: '1px solid #86efac' }}>OPERATIONAL</span>
        </div>
        <p className="lab-card-desc">六个自治 Agent 通过 MessageBus 协作，逐步完成数据获取、因子计算、回测验证、风险管理和组合构建</p>
        <div className="agent-pipeline">
          {AGENTS.map((agent, i) => (
            <div key={agent.name} className="agent-node" style={{ '--agent-color': agent.color } as React.CSSProperties}>
              <div className="agent-icon">{agent.icon}</div>
              <div className="agent-name">{agent.name}</div>
              <div className="agent-role">{agent.role}</div>
              <div className="agent-desc">{agent.desc}</div>
              {i < AGENTS.length - 1 && <div className="agent-arrow">→</div>}
            </div>
          ))}
        </div>
      </section>

      {/* Factor Table */}
      <section className="board-card">
        <h2>Five-Factor Model</h2>
        <p className="lab-card-desc">基本面因子延迟2个月使用，模拟~60天财报披露延迟，避免前视偏差</p>
        <div className="lab-table-wrap">
          <table className="lab-table">
            <thead>
              <tr>
                <th>Factor</th>
                <th>Definition</th>
                <th>Description</th>
                <th>Rationale</th>
              </tr>
            </thead>
            <tbody>
              {FACTORS.map(f => (
                <tr key={f.name}>
                  <td style={{ fontWeight: 600 }}>{f.name}</td>
                  <td className="lab-td-mono" style={{ fontSize: 12 }}>{f.def}</td>
                  <td style={{ fontSize: 12, color: '#5a4452' }}>{f.desc}</td>
                  <td style={{ fontSize: 12, color: '#8a7882' }}>{f.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Phase Timeline */}
      <section className="board-card">
        <div className="ab-header">
          <h2>Project Timeline</h2>
          <span className="ab-badge" style={{ background: '#dcfce7', color: '#15803d', border: '1px solid #86efac' }}>
            {Math.round(completedTasks / totalTasks * 100)}% COMPLETE
          </span>
        </div>
        <p className="lab-card-desc">每个阶段的工作内容与完成状态</p>

        <div className="phase-timeline">
          {PHASES.map((phase) => (
            <div key={phase.id} className="phase-card" style={{ '--phase-color': phase.color } as React.CSSProperties}>
              <div className="phase-header">
                <div className="phase-dot" style={{ background: phase.color }} />
                <div className="phase-title-row">
                  <h3>{phase.title}</h3>
                  <div className="phase-meta">
                    <span className="phase-weeks">{phase.weeks}</span>
                    {statusBadge(phase.status)}
                  </div>
                </div>
              </div>
              <div className="phase-tasks">
                {phase.tasks.map((task, i) => (
                  <div key={i} className={`phase-task ${task.done ? 'done' : ''}`}>
                    <span className="task-check">{task.done ? '✓' : '○'}</span>
                    <div className="task-info">
                      <span className="task-name">{task.name}</span>
                      <span className="task-detail">{task.detail}</span>
                    </div>
                  </div>
                ))}
              </div>
              <div className="phase-progress-bar">
                <div
                  className="phase-progress-fill"
                  style={{
                    width: `${phase.tasks.filter(t => t.done).length / phase.tasks.length * 100}%`,
                    background: phase.color,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Results Comparison */}
      <section className="board-card" style={{ borderColor: '#d6e6f7', background: 'linear-gradient(135deg, #fff7fa 0%, #fff 50%, #f2f8ff 100%)' }}>
        <div className="ab-header">
          <h2>Walk-Forward Backtest Results</h2>
          <span className="ab-badge">2022-04 ~ 2024-12</span>
          <span className="ab-badge" style={{ background: '#fef3c7', color: '#b45309', border: '1px solid #fcd34d' }}>TOP20 EW MONTHLY</span>
        </div>
        <p className="lab-card-desc">所有模型在相同 Walk-Forward 协议下的诚实样本外表现。参照: 锁定基线 (全样本 PCA) 30.0% gross, Sharpe 1.77</p>

        <div className="lab-table-wrap">
          <table className="lab-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Universe</th>
                <th>Features</th>
                <th>Gross Ann.</th>
                <th>Sharpe</th>
                <th>Max DD</th>
                <th>Net Ann.</th>
              </tr>
            </thead>
            <tbody>
              {RESULTS.map((r, i) => (
                <tr key={i} style={r.highlight ? { background: 'rgba(34, 197, 94, 0.06)' } : undefined}>
                  <td style={{ fontWeight: 600 }}>
                    {r.highlight && <span className="ab-best-tag">BEST</span>}
                    {r.model}
                  </td>
                  <td style={{ fontSize: 12 }}>{r.universe}</td>
                  <td style={{ fontSize: 12 }}>{r.features}</td>
                  <td className="lab-td-mono" style={{ color: pctColor(r.grossAnn, [10, 15]) }}>
                    {r.grossAnn.toFixed(1)}%
                  </td>
                  <td className="lab-td-mono" style={{ color: pctColor(r.sharpe, [0.5, 0.8]) }}>
                    {r.sharpe.toFixed(2)}
                  </td>
                  <td className="lab-td-mono" style={{ color: r.mdd > -30 ? '#b45309' : '#dc2626' }}>
                    {r.mdd.toFixed(1)}%
                  </td>
                  <td className="lab-td-mono">
                    {r.netAnn.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="ref-baseline">
          <span className="ref-label">REF</span>
          <span>Golden Baseline (全样本 PCA Linear, 含前瞻偏差): </span>
          <span className="lab-td-mono" style={{ fontWeight: 700, color: '#dc2626' }}>30.0% gross</span>
          <span style={{ color: '#8a7882' }}> / Sharpe 1.77 / Net 28.7%</span>
        </div>
      </section>

      {/* Key Findings */}
      <section className="board-card">
        <h2>Key Findings</h2>
        <p className="lab-card-desc">Walk-forward 诚实评估的核心结论</p>
        <div className="findings-grid">
          {FINDINGS.map((f, i) => (
            <div key={i} className="finding-card" style={{ borderTopColor: f.color }}>
              <div className="finding-icon">{f.icon}</div>
              <h3 style={{ color: f.color }}>{f.title}</h3>
              <p>{f.detail}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Tech Stack */}
      <section className="board-card">
        <h2>Tech Stack</h2>
        <div className="tech-tags">
          {['Python', 'yfinance', 'alphalens', 'scikit-learn', 'pandas', 'numpy', 'matplotlib',
            'LightGBM', 'Microsoft Qlib', 'Flask', 'React', 'Vite', 'Recharts', 'GitHub Actions'].map(t => (
            <span key={t} className="tech-tag">{t}</span>
          ))}
        </div>
      </section>

      <footer className="board-footer">
        <div>Multi-Agent Quant Factor Platform — Walk-forward factor research on US equities</div>
        <div style={{ marginTop: '4px', fontSize: '10px', opacity: 0.6, letterSpacing: '0.3px' }}>
          Built with <strong>auto-dashboard</strong> · design © Coco
        </div>
      </footer>
    </div>
  );
}
