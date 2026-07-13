import { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, Cell,
} from 'recharts';

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
  const color = delta > 0.005 ? '#16a34a' : delta < -0.005 ? '#dc2626' : '#6b7280';
  return (
    <span style={{ color, fontWeight: 600, fontSize: 13 }}>
      {delta > 0 ? '+' : ''}{delta.toFixed(3)} ({pct > 0 ? '+' : ''}{pct.toFixed(1)}%)
    </span>
  );
}

function OverviewTable({ experiments }: { experiments: ExperimentData[] }) {
  const bestAuc = Math.max(...experiments.map(e => e.auc));
  return (
    <div className="ab-table-wrap">
      <table className="ab-table">
        <thead>
          <tr><th>Model</th><th>AUC</th><th>P@50%</th><th>R@50%</th><th>F1@50%</th><th>Events</th></tr>
        </thead>
        <tbody>
          {experiments.map((exp, i) => {
            const t50 = exp.threshold_analysis.find(t => t.threshold === 0.5);
            const detected = exp.events_backtest.filter(e => e.lead_days != null).length;
            const isBest = Math.abs(exp.auc - bestAuc) < 0.001;
            return (
              <tr key={i} style={isBest ? { background: '#f0fdf4' } : undefined}>
                <td style={{ color: COLORS[i % COLORS.length], fontWeight: 600 }}>
                  {isBest && <span className="ab-best-tag">BEST</span>}
                  {exp.name}
                </td>
                <td style={{ fontWeight: 700 }}>{exp.auc.toFixed(3)}</td>
                <td>{t50 ? (t50.precision * 100).toFixed(1) + '%' : '—'}</td>
                <td>{t50 ? (t50.recall * 100).toFixed(1) + '%' : '—'}</td>
                <td>{t50 ? (t50.f1 * 100).toFixed(1) + '%' : '—'}</td>
                <td>{detected}/{exp.events_backtest.length}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ProbTimeline({ baseline, challenger, baseColor, challColor }: {
  baseline: ExperimentData; challenger: ExperimentData; baseColor: string; challColor: string;
}) {
  const challMap = new Map(challenger.probability_timeline.map(d => [d.date, d.probability]));

  const data = baseline.probability_timeline
    .filter((_, i) => i % 3 === 0)
    .map(d => ({
      date: d.date,
      [baseline.name]: d.probability,
      [challenger.name]: challMap.get(d.date),
    }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data}>
        <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={d => d?.slice(5, 10)} interval={Math.floor(data.length / 8)} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
        <Tooltip formatter={(v: number) => (v * 100).toFixed(1) + '%'} />
        <Legend />
        <Line type="monotone" dataKey={baseline.name} stroke={baseColor} dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey={challenger.name} stroke={challColor} dot={false} strokeWidth={1.5} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function FeatureImportanceChart({ data, modelName }: { data: FeatureImp[]; modelName: string }) {
  if (!data || data.length === 0) return null;
  const top10 = data.slice(0, 10);
  return (
    <div style={{ marginTop: 16 }}>
      <h4 style={{ margin: '0 0 8px', fontSize: 14 }}>Feature Importance: {modelName}</h4>
      <ResponsiveContainer width="100%" height={Math.max(200, top10.length * 28)}>
        <BarChart data={top10} layout="vertical" margin={{ left: 120 }}>
          <XAxis type="number" tick={{ fontSize: 10 }} />
          <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={120} />
          <Tooltip />
          <Bar dataKey="importance" fill="#3a82d6">
            {top10.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Phase3Lab() {
  const [data, setData] = useState<Phase3Data | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const base = import.meta.env.BASE_URL || '/';
    fetch(`${base}data/phase3_metrics.json?t=${Date.now()}`)
      .then(r => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then(setData)
      .catch(e => setError(e.message));
  }, []);

  if (error) return <div className="lab-container"><div className="lab-card"><p>Phase 3 data not available yet: {error}</p></div></div>;
  if (!data) return <div className="loading">Loading Phase 3 data...</div>;

  const { experiments, pairwise, feature_importances, summary } = data;
  const find = (name: string) => experiments.find(e => e.name.includes(name));

  return (
    <div className="lab-container">
      <header className="lab-header">
        <h1>Phase 3: Model Evolution</h1>
        <p className="lab-subtitle">突破 Regime 瓶颈 — 非线性模型 + Regime 特征 + 宏观事件</p>
      </header>

      {/* Summary stats */}
      <section className="lab-card">
        <div className="ab-header">
          <h2>当前进度</h2>
          <span className="ab-badge" style={{ background: '#eff6ff', color: '#1e40af', border: '1px solid #bfdbfe' }}>PHASE 3</span>
        </div>
        <div className="roadmap-status-grid">
          <div className="roadmap-stat">
            <span className="roadmap-stat-value">{summary.best_model}</span>
            <span className="roadmap-stat-label">最佳模型</span>
          </div>
          <div className="roadmap-stat">
            <span className="roadmap-stat-value" style={{ color: '#16a34a' }}>
              {Math.max(summary.gbdt_slim_auc, summary.gbdt_full_auc, summary.rf_slim_auc, summary.lr_slim_auc, summary.lr_full_auc).toFixed(3)}
            </span>
            <span className="roadmap-stat-label">最佳 AUC</span>
          </div>
          <div className="roadmap-stat">
            <span className="roadmap-stat-value">{summary.total_samples}</span>
            <span className="roadmap-stat-label">样本量</span>
          </div>
          <div className="roadmap-stat">
            <span className="roadmap-stat-value">{summary.data_range}</span>
            <span className="roadmap-stat-label">数据范围</span>
          </div>
        </div>
      </section>

      {/* Overall comparison */}
      <section className="lab-card">
        <div className="ab-header">
          <h2>Step 1: 非线性模型 vs 线性模型</h2>
          <span className="ab-badge" style={{ background: '#f0fdf4', color: '#166534', border: '1px solid #bbf7d0' }}>A/B TEST</span>
        </div>
        <p className="lab-card-desc">
          用 GBDT（梯度提升树）和 Random Forest 替换 Logistic Regression。
          树模型可隐式学到 regime 条件规律（如「VIX 高 且 利差走阔 → 危机」），不受共线性影响。
        </p>
        <OverviewTable experiments={experiments} />
      </section>

      {/* Pairwise comparisons */}
      {pairwise.map((pair) => {
        const base = find(pair.baseline);
        const chall = find(pair.challenger);
        if (!base || !chall) return null;
        const baseIdx = experiments.indexOf(base);
        const challIdx = experiments.indexOf(chall);

        return (
          <section key={pair.id} className="lab-card">
            <div className="ab-header">
              <h2>{pair.label}</h2>
            </div>
            <p className="lab-card-desc">{pair.variable}</p>

            {pair.method_note && (
              <div className="method-note">
                <span className="method-note-tag">METHOD</span>
                {pair.method_note}
              </div>
            )}

            {/* Summary cards */}
            <div className="pair-summary">
              <div className="pair-card" style={{ borderColor: COLORS[baseIdx % COLORS.length] }}>
                <div className="pair-card-name" style={{ color: COLORS[baseIdx % COLORS.length] }}>
                  {base.name} (baseline)
                </div>
                <div className="pair-card-auc">AUC {base.auc.toFixed(3)}</div>
              </div>
              <div className="pair-vs">vs</div>
              <div className="pair-card" style={{ borderColor: COLORS[challIdx % COLORS.length] }}>
                <div className="pair-card-name" style={{ color: COLORS[challIdx % COLORS.length] }}>
                  {chall.name}
                </div>
                <div className="pair-card-auc">
                  AUC {chall.auc.toFixed(3)}{' '}
                  <DeltaBadge base={base.auc} value={chall.auc} />
                </div>
              </div>
            </div>

            {/* Probability timeline */}
            <h4 style={{ margin: '16px 0 8px', fontSize: 14 }}>概率时间线</h4>
            <ProbTimeline
              baseline={base} challenger={chall}
              baseColor={COLORS[baseIdx % COLORS.length]}
              challColor={COLORS[challIdx % COLORS.length]}
            />

            {/* Threshold comparison */}
            <h4 style={{ margin: '16px 0 8px', fontSize: 14 }}>阈值分析</h4>
            <div className="ab-table-wrap">
              <table className="ab-table" style={{ fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>Threshold</th>
                    <th>{base.name} P</th><th>{base.name} R</th>
                    <th>{chall.name} P</th><th>{chall.name} R</th>
                  </tr>
                </thead>
                <tbody>
                  {[0.3, 0.5, 0.7].map(t => {
                    const bRow = base.threshold_analysis.find(r => r.threshold === t);
                    const cRow = chall.threshold_analysis.find(r => r.threshold === t);
                    return (
                      <tr key={t}>
                        <td style={{ fontWeight: 600 }}>{(t * 100).toFixed(0)}%</td>
                        <td>{bRow ? (bRow.precision * 100).toFixed(1) + '%' : '—'}</td>
                        <td>{bRow ? (bRow.recall * 100).toFixed(1) + '%' : '—'}</td>
                        <td>{cRow ? (cRow.precision * 100).toFixed(1) + '%' : '—'}</td>
                        <td>{cRow ? (cRow.recall * 100).toFixed(1) + '%' : '—'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        );
      })}

      {/* Feature importance */}
      {Object.keys(feature_importances).length > 0 && (
        <section className="lab-card">
          <div className="ab-header">
            <h2>特征重要性对比</h2>
          </div>
          <p className="lab-card-desc">
            LR 用系数绝对值衡量重要性，树模型用信息增益。两者的排序差异揭示了非线性交互效应。
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: 16 }}>
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
        </div>
        <div className="roadmap-items">
          <div className="roadmap-item done"><span className="roadmap-check">✓</span>
            <strong>Step 1 非线性模型:</strong> GBDT Full AUC={summary.gbdt_full_auc} vs LR Slim AUC={summary.lr_slim_auc}
          </div>
          <div className="roadmap-item pending"><span className="roadmap-dot" />
            <strong>Step 2 Regime 特征:</strong> FOMC 利率方向 + CPI YoY 趋势 + 收益率曲线状态
          </div>
          <div className="roadmap-item pending"><span className="roadmap-dot" />
            <strong>Step 3 事件日历:</strong> FOMC/CPI/非农前后 N 天标记，捕捉事件窗口模式
          </div>
          <div className="roadmap-item pending"><span className="roadmap-dot" />
            <strong>Step 4 长期重测:</strong> 1986+ 数据 + 非线性模型，验证 regime 瓶颈是否突破
          </div>
        </div>
      </section>

      <footer className="footer">
        <div>Phase 3: Model Evolution | sklearn HistGradientBoosting + RandomForest</div>
      </footer>
    </div>
  );
}
