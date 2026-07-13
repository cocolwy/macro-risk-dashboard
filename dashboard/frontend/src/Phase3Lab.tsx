import { useState, useEffect, useMemo, Component, type ReactNode } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, Cell, CartesianGrid,
} from 'recharts';

class ErrorBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null };
  static getDerivedStateFromError(e: Error) { return { error: e.message }; }
  render() {
    if (this.state.error) return (
      <div className="lab-container"><div className="lab-card">
        <h2 style={{ color: '#dc2626' }}>Render Error</h2>
        <pre style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>{this.state.error}</pre>
      </div></div>
    );
    return this.props.children;
  }
}

interface ThresholdRow { threshold: number; precision: number; recall: number; f1: number; alert_days: number; total_days: number; alert_pct: number; }
interface EventBacktest { name: string; event_date: string; max_probability: number; lead_days: number | null; first_alert_date: string | null; }
interface ExperimentData {
  name: string; auc: number;
  threshold_analysis: ThresholdRow[];
  events_backtest: EventBacktest[];
  probability_timeline: { date: string; probability: number }[];
}
interface PairwiseConfig { id: string; label: string; variable: string; baseline: string; challenger: string; method_note: string; }
interface FeatureImp { feature: string; importance: number; }
interface Phase3Data {
  phase: number; title: string;
  experiments: ExperimentData[];
  pairwise: PairwiseConfig[];
  feature_importances: Record<string, FeatureImp[]>;
  summary: {
    lr_slim_auc: number; gbdt_slim_auc: number; gbdt_full_auc: number;
    rf_slim_auc: number; lr_full_auc: number;
    best_model: string; data_range: string; total_samples: number;
  };
}

const COLORS = ['#d6457a', '#3a82d6', '#16a34a', '#ea580c', '#8b5cf6', '#0d9488'];

function DeltaBadge({ base, value }: { base: number; value: number }) {
  const delta = value - base;
  const pct = base > 0 ? (delta / base) * 100 : 0;
  const dir = delta > 0.005 ? 'up' : delta < -0.005 ? 'down' : 'neutral';
  return (
    <span className={`ab-delta-badge ${dir}`}>
      {delta > 0 ? '+' : ''}{delta.toFixed(3)} ({pct > 0 ? '+' : ''}{pct.toFixed(1)}%)
    </span>
  );
}

function OverviewTable({ experiments }: { experiments: ExperimentData[] }) {
  const bestAuc = Math.max(...experiments.map(e => e.auc));
  return (
    <div className="ab-overview-table">
      <div className="lab-table-wrap">
        <table className="lab-table">
          <thead>
            <tr><th>模型</th><th>AUC</th><th>P@50%</th><th>R@50%</th><th>F1@50%</th><th>事件命中</th></tr>
          </thead>
          <tbody>
            {experiments.map((exp, i) => {
              const t50 = exp.threshold_analysis.find(t => t.threshold === 0.5);
              const detected = exp.events_backtest.filter(e => e.lead_days != null).length;
              const isBest = Math.abs(exp.auc - bestAuc) < 0.001;
              return (
                <tr key={i} style={isBest ? { background: 'rgba(34, 197, 94, 0.06)' } : undefined}>
                  <td style={{ color: COLORS[i % COLORS.length], fontWeight: 600 }}>
                    {isBest && <span className="ab-best-tag">BEST</span>}
                    {exp.name}
                  </td>
                  <td className="lab-td-mono" style={{ fontWeight: 700 }}>{exp.auc.toFixed(3)}</td>
                  <td>{t50 ? `${(t50.precision * 100).toFixed(1)}%` : '—'}</td>
                  <td>{t50 ? `${(t50.recall * 100).toFixed(1)}%` : '—'}</td>
                  <td>{t50 ? `${(t50.f1 * 100).toFixed(1)}%` : '—'}</td>
                  <td>{detected}/{exp.events_backtest.length}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ProbTimeline({ baseline, challenger, baseColor, challColor }: {
  baseline: ExperimentData; challenger: ExperimentData; baseColor: string; challColor: string;
}) {
  const data = useMemo(() => {
    const challMap = new Map(challenger.probability_timeline.map(d => [d.date, d.probability]));
    return baseline.probability_timeline
      .filter((_, i) => i % 3 === 0)
      .map(d => ({
        date: d.date,
        [baseline.name]: d.probability,
        [challenger.name]: challMap.get(d.date),
      }));
  }, [baseline, challenger]);

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(241, 216, 226, 0.6)" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 10, fill: '#8a7882' }}
          tickFormatter={d => d?.slice(5, 10)}
          interval={Math.floor(data.length / 8)}
          minTickGap={50}
        />
        <YAxis
          domain={[0, 1]}
          tick={{ fontSize: 10, fill: '#8a7882' }}
          tickFormatter={v => `${(v * 100).toFixed(0)}%`}
        />
        <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line type="monotone" dataKey={baseline.name} stroke={baseColor} dot={false} strokeWidth={1.5} isAnimationActive={false} />
        <Line type="monotone" dataKey={challenger.name} stroke={challColor} dot={false} strokeWidth={1.5} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function FeatureImportanceChart({ data, modelName }: { data: FeatureImp[]; modelName: string }) {
  if (!data || data.length === 0) return null;
  const top10 = data.slice(0, 10);
  return (
    <div className="phase3-fi-card">
      <h4 className="phase3-fi-title">{modelName}</h4>
      <ResponsiveContainer width="100%" height={Math.max(220, top10.length * 30)}>
        <BarChart data={top10} layout="vertical" margin={{ left: 8, right: 16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(241, 216, 226, 0.6)" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 10, fill: '#8a7882' }} />
          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fontSize: 10, fill: '#5c4f56' }}
            width={130}
          />
          <Tooltip />
          <Bar dataKey="importance" radius={[0, 4, 4, 0]} isAnimationActive={false}>
            {top10.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function PairwiseSection({ pairwise, experiments }: { pairwise: PairwiseConfig[]; experiments: ExperimentData[] }) {
  const find = (name: string) => experiments.find(e => e.name.includes(name));

  return (
    <section className="lab-card ab-section">
      <div className="ab-header">
        <h2>成对对比实验</h2>
        <span className="ab-badge">{pairwise.length} PAIRS</span>
      </div>
      <p className="lab-card-desc">
        逐一对比 baseline 与 challenger，观察非线性模型在不同特征集上的增益。
      </p>

      {pairwise.map((pair) => {
        const base = find(pair.baseline);
        const chall = find(pair.challenger);
        if (!base || !chall) return null;
        const baseIdx = experiments.indexOf(base);
        const challIdx = experiments.indexOf(chall);
        const winner = chall.auc > base.auc ? 'challenger' : 'baseline';

        return (
          <div key={pair.id} className="ab-pair">
            <div className="ab-pair-header">
              <h3>{pair.label}</h3>
              <span className="ab-pair-variable">测试变量: {pair.variable}</span>
            </div>

            {pair.method_note && (
              <div className="ab-method-note">
                <span className="ab-method-note-icon">&#9432;</span>
                {pair.method_note}
              </div>
            )}

            <div className="ab-pair-cards">
              <div
                className={`ab-pair-card ${winner === 'baseline' ? 'winner' : ''}`}
                style={{ borderTopColor: COLORS[baseIdx % COLORS.length] }}
              >
                <div className="ab-pair-role">
                  BASELINE {winner === 'baseline' && <span className="ab-winner-tag">WIN</span>}
                </div>
                <div className="ab-model-name" style={{ color: COLORS[baseIdx % COLORS.length] }}>
                  {base.name}
                </div>
                <div className="ab-metric-row">
                  <span className="ab-metric-label">AUC</span>
                  <span className="ab-metric-value">{base.auc.toFixed(3)}</span>
                </div>
              </div>
              <div className="ab-pair-vs">VS</div>
              <div
                className={`ab-pair-card ${winner === 'challenger' ? 'winner' : ''}`}
                style={{ borderTopColor: COLORS[challIdx % COLORS.length] }}
              >
                <div className="ab-pair-role">
                  CHALLENGER {winner === 'challenger' && <span className="ab-winner-tag">WIN</span>}
                </div>
                <div className="ab-model-name" style={{ color: COLORS[challIdx % COLORS.length] }}>
                  {chall.name}
                </div>
                <div className="ab-metric-row">
                  <span className="ab-metric-label">AUC</span>
                  <span className="ab-metric-value">
                    {chall.auc.toFixed(3)}
                    <DeltaBadge base={base.auc} value={chall.auc} />
                  </span>
                </div>
              </div>
            </div>

            <h4 className="lab-subsection-title">概率时间线</h4>
            <ProbTimeline
              baseline={base}
              challenger={chall}
              baseColor={COLORS[baseIdx % COLORS.length]}
              challColor={COLORS[challIdx % COLORS.length]}
            />

            <h4 className="lab-subsection-title">阈值分析</h4>
            <div className="lab-table-wrap">
              <table className="lab-table lab-table-compact">
                <thead>
                  <tr>
                    <th>Threshold</th>
                    <th>{base.name} P</th>
                    <th>{base.name} R</th>
                    <th>{chall.name} P</th>
                    <th>{chall.name} R</th>
                  </tr>
                </thead>
                <tbody>
                  {[0.3, 0.5, 0.7].map(t => {
                    const bRow = base.threshold_analysis.find(r => r.threshold === t);
                    const cRow = chall.threshold_analysis.find(r => r.threshold === t);
                    return (
                      <tr key={t}>
                        <td className="lab-td-mono">{(t * 100).toFixed(0)}%</td>
                        <td>{bRow ? `${(bRow.precision * 100).toFixed(1)}%` : '—'}</td>
                        <td>{bRow ? `${(bRow.recall * 100).toFixed(1)}%` : '—'}</td>
                        <td>{cRow ? `${(cRow.precision * 100).toFixed(1)}%` : '—'}</td>
                        <td>{cRow ? `${(cRow.recall * 100).toFixed(1)}%` : '—'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </section>
  );
}

function Phase3LabInner() {
  const [data, setData] = useState<Phase3Data | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const base = import.meta.env.BASE_URL || '/';
    fetch(`${base}data/phase3_metrics.json?t=${Date.now()}`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(d => setData(d))
      .catch(e => setError(e.message));
  }, []);

  if (error) return <div className="lab-container"><div className="lab-card"><p>Phase 3 data not available yet: {error}</p></div></div>;
  if (!data) return <div className="loading">Loading Phase 3 data...</div>;

  const { experiments, pairwise, feature_importances, summary } = data;
  const bestAuc = Math.max(
    summary.gbdt_slim_auc, summary.gbdt_full_auc,
    summary.rf_slim_auc, summary.lr_slim_auc, summary.lr_full_auc,
  );

  return (
    <div className="lab-container">
      <header className="lab-header">
        <div>
          <h1>Ch.2 Model Evolution</h1>
          <p className="lab-subtitle">Phase 3 · 突破 Regime 瓶颈 — GBDT / RandomForest 非线性探索</p>
        </div>
        <div className="lab-model-badge">
          <span className="lab-badge-version">Phase {data.phase}</span>
          <span className="lab-badge-auc">Best AUC: {bestAuc.toFixed(3)}</span>
        </div>
      </header>

      {/* Summary */}
      <section className="lab-card">
        <div className="best-config-card" style={{ marginTop: 0 }}>
          <div className="best-config-header">
            <span className="best-config-badge">CURRENT STATUS</span>
            <div className="best-config-title">
              <span className="best-config-name">{summary.best_model}</span>
              <span className="best-config-auc">AUC {bestAuc.toFixed(3)}</span>
            </div>
          </div>
          <p className="best-config-desc">
            用梯度提升树与随机森林替换 Logistic Regression，验证非线性模型能否突破 Ch.1 发现的 regime 瓶颈。
          </p>
          <div className="roadmap-status-grid">
            <div className="roadmap-stat">
              <span className="roadmap-stat-value">{summary.total_samples.toLocaleString()}</span>
              <span className="roadmap-stat-label">样本量</span>
            </div>
            <div className="roadmap-stat">
              <span className="roadmap-stat-value text-sm">{summary.data_range}</span>
              <span className="roadmap-stat-label">数据范围</span>
            </div>
            <div className="roadmap-stat">
              <span className="roadmap-stat-value" style={{ color: '#3a82d6' }}>{summary.gbdt_full_auc.toFixed(3)}</span>
              <span className="roadmap-stat-label">GBDT Full</span>
            </div>
            <div className="roadmap-stat">
              <span className="roadmap-stat-value" style={{ color: '#d6457a' }}>{summary.lr_slim_auc.toFixed(3)}</span>
              <span className="roadmap-stat-label">LR Slim (baseline)</span>
            </div>
          </div>
        </div>
      </section>

      {/* Overall comparison */}
      <section className="lab-card ab-section">
        <div className="ab-header">
          <h2>Step 1: 非线性 vs 线性</h2>
          <span className="ab-badge" style={{ background: '#f0fdf4', color: '#166534', border: '1px solid #bbf7d0' }}>A/B TEST</span>
          <span className="ab-badge">{experiments.length} MODELS</span>
        </div>
        <p className="lab-card-desc">
          树模型可隐式学到 regime 条件规律（如「VIX 高 且 利差走阔 → 危机」），不受共线性影响。
        </p>
        <OverviewTable experiments={experiments} />
      </section>

      <PairwiseSection pairwise={pairwise} experiments={experiments} />

      {/* Feature importance */}
      {Object.keys(feature_importances).length > 0 && (
        <section className="lab-card">
          <div className="ab-header">
            <h2>特征重要性对比</h2>
            <span className="ab-badge">FEATURE IMP</span>
          </div>
          <p className="lab-card-desc">
            LR 用系数绝对值衡量重要性，树模型用信息增益。排序差异揭示非线性交互效应。
          </p>
          <div className="phase3-fi-grid">
            {Object.entries(feature_importances).map(([modelName, imps]) => (
              <FeatureImportanceChart key={modelName} data={imps} modelName={modelName} />
            ))}
          </div>
        </section>
      )}

      {/* Roadmap */}
      <section className="lab-card roadmap-section">
        <div className="ab-header">
          <h2>Phase 3 路线图</h2>
          <span className="ab-badge" style={{ background: '#eff6ff', color: '#1e40af', border: '1px solid #bfdbfe' }}>ROADMAP</span>
        </div>
        <div className="roadmap-items">
          <div className="roadmap-item done">
            <span className="roadmap-check">✓</span>
            <div className="roadmap-step-body">
              <div className="roadmap-step-title">Step 1 · 非线性模型</div>
              <div className="roadmap-step-detail">
                <span className="roadmap-metric-chip">GBDT Full {summary.gbdt_full_auc.toFixed(3)}</span>
                <span className="roadmap-metric-chip muted">vs</span>
                <span className="roadmap-metric-chip">LR Slim {summary.lr_slim_auc.toFixed(3)}</span>
              </div>
            </div>
          </div>
          <div className="roadmap-item pending">
            <span className="roadmap-dot" />
            <div className="roadmap-step-body">
              <div className="roadmap-step-title">Step 2 · Regime 特征</div>
              <div className="roadmap-step-desc">FOMC 利率方向 · CPI YoY 趋势 · 收益率曲线状态</div>
            </div>
          </div>
          <div className="roadmap-item pending">
            <span className="roadmap-dot" />
            <div className="roadmap-step-body">
              <div className="roadmap-step-title">Step 3 · 事件日历</div>
              <div className="roadmap-step-desc">FOMC / CPI / 非农前后 N 天标记，捕捉事件窗口模式</div>
            </div>
          </div>
          <div className="roadmap-item pending">
            <span className="roadmap-dot" />
            <div className="roadmap-step-body">
              <div className="roadmap-step-title">Step 4 · 长期重测</div>
              <div className="roadmap-step-desc">1986+ 数据 + 非线性模型，验证 regime 瓶颈是否突破</div>
            </div>
          </div>
        </div>
      </section>

      <footer className="footer">
        <div>Ch.2 Model Evolution · sklearn HistGradientBoosting + RandomForest</div>
      </footer>
    </div>
  );
}

export function Phase3Lab() {
  return <ErrorBoundary><Phase3LabInner /></ErrorBoundary>;
}
