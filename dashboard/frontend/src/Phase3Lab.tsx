import { useState, useEffect, useMemo, Component, type ReactNode } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, Cell, CartesianGrid, ReferenceLine,
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

interface PracticalMetrics {
  brier_score: number; base_rate: number; best_f1: number; best_f1_threshold: number;
  p_at_80: number; r_at_80: number; lift_at_80: number;
  mean_prob: number; median_prob: number; prob_gt50_pct: number;
}
interface ThresholdRow { threshold: number; precision: number; recall: number; f1: number; alert_days: number; total_days: number; alert_pct: number; }
interface EventBacktest { name: string; event_date: string; max_probability: number; lead_days: number | null; first_alert_date: string | null; }
interface ExperimentData {
  name: string; auc: number;
  threshold_analysis: ThresholdRow[];
  events_backtest: EventBacktest[];
  probability_timeline: { date: string; probability: number }[];
  practical_metrics?: PracticalMetrics;
}
interface PairwiseConfig { id: string; label: string; variable: string; baseline: string; challenger: string; method_note: string; }
interface FeatureImp { feature: string; importance: number; }
interface PracticalSummary { best_f1_model: string; best_f1: number; best_brier_model: string; best_brier: number; best_lift_model: string; best_lift: number; }
interface Phase3Data {
  phase: number; title: string;
  experiments: ExperimentData[];
  pairwise: PairwiseConfig[];
  feature_importances: Record<string, FeatureImp[]>;
  practical_summary?: PracticalSummary;
  summary: {
    lr_slim_auc: number; lr_full_auc: number;
    gbdt_slim_auc: number; gbdt_full_auc: number;
    rf_slim_auc: number; rf_full_auc?: number;
    best_model: string; data_range: string; total_samples: number;
  };
}

const COLORS = ['#d6457a', '#3a82d6', '#16a34a', '#ea580c', '#8b5cf6', '#0d9488'];

function OverviewTable({ experiments }: { experiments: ExperimentData[] }) {
  const withPM = experiments.filter(e => e.practical_metrics);
  if (withPM.length === 0) return null;

  const bestF1 = Math.max(...withPM.map(e => e.practical_metrics!.best_f1));
  const bestBrier = Math.min(...withPM.map(e => e.practical_metrics!.brier_score));

  return (
    <div className="ab-overview-table">
      <div className="lab-table-wrap">
        <table className="lab-table">
          <thead>
            <tr><th>模型</th><th>Best F1</th><th>Brier ↓</th><th style={{ color: '#8a7882' }}>AUC</th></tr>
          </thead>
          <tbody>
            {withPM.map((exp, i) => {
              const pm = exp.practical_metrics!;
              const isBestF1 = Math.abs(pm.best_f1 - bestF1) < 0.001;
              const isBestBrier = Math.abs(pm.brier_score - bestBrier) < 0.001;
              return (
                <tr key={i}>
                  <td style={{ color: COLORS[i % COLORS.length], fontWeight: 600 }}>
                    {isBestF1 && <span className="ab-best-tag">BEST</span>}
                    {exp.name}
                  </td>
                  <td className="lab-td-mono" style={isBestF1 ? { fontWeight: 700, color: '#16a34a' } : undefined}>
                    {pm.best_f1.toFixed(3)}
                    <span style={{ fontSize: 10, opacity: 0.5, marginLeft: 3 }}>@{(pm.best_f1_threshold * 100).toFixed(0)}%</span>
                  </td>
                  <td className="lab-td-mono" style={isBestBrier ? { fontWeight: 700, color: '#16a34a' } : undefined}>{pm.brier_score.toFixed(3)}</td>
                  <td className="lab-td-mono" style={{ color: '#8a7882' }}>{exp.auc.toFixed(3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p style={{ fontSize: 11, color: '#8a7882', marginTop: 8 }}>
        Unbalanced 训练 · Embargo 20d · 绿色 = 最佳 · 详细指标见 Ch.2.1
      </p>
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
      .map(d => ({ date: d.date, [baseline.name]: d.probability, [challenger.name]: challMap.get(d.date) }));
  }, [baseline, challenger]);

  const baseRate = baseline.practical_metrics?.base_rate ?? 0.12;

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(241,216,226,0.6)" />
        <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#8a7882' }} tickFormatter={d => d?.slice(5, 10)} interval={Math.floor(data.length / 8)} minTickGap={50} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#8a7882' }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
        <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <ReferenceLine y={baseRate} stroke="#999" strokeDasharray="5 5" label={{ value: `base rate ${(baseRate * 100).toFixed(0)}%`, fontSize: 10, fill: '#999' }} />
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
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(241,216,226,0.6)" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 10, fill: '#8a7882' }} />
          <YAxis type="category" dataKey="feature" tick={{ fontSize: 10, fill: '#5c4f56' }} width={130} />
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
        所有模型均使用 Unbalanced 训练，控制变量仅为模型类型或特征数量。
      </p>

      {pairwise.map(pair => {
        const base = find(pair.baseline);
        const chall = find(pair.challenger);
        if (!base || !chall) return null;
        const baseIdx = experiments.indexOf(base);
        const challIdx = experiments.indexOf(chall);
        const basePM = base.practical_metrics;
        const challPM = chall.practical_metrics;
        const winner = (challPM?.best_f1 ?? 0) > (basePM?.best_f1 ?? 0) ? 'challenger' : 'baseline';

        return (
          <div key={pair.id} className="ab-pair">
            <div className="ab-pair-header">
              <h3>{pair.label}</h3>
              <span className="ab-pair-variable">测试变量: {pair.variable}</span>
            </div>
            {pair.method_note && (
              <div className="ab-method-note"><span className="ab-method-note-icon">&#9432;</span>{pair.method_note}</div>
            )}
            <div className="ab-pair-cards">
              <div className={`ab-pair-card ${winner === 'baseline' ? 'winner' : ''}`} style={{ borderTopColor: COLORS[baseIdx % COLORS.length] }}>
                <div className="ab-pair-role">BASELINE {winner === 'baseline' && <span className="ab-winner-tag">WIN</span>}</div>
                <div className="ab-model-name" style={{ color: COLORS[baseIdx % COLORS.length] }}>{base.name}</div>
                <div className="ab-metric-row"><span className="ab-metric-label">AUC</span><span className="ab-metric-value">{base.auc.toFixed(3)}</span></div>
                {basePM && <>
                  <div className="ab-metric-row"><span className="ab-metric-label">Best F1</span><span className="ab-metric-value">{basePM.best_f1.toFixed(3)} @{(basePM.best_f1_threshold * 100).toFixed(0)}%</span></div>
                  <div className="ab-metric-row"><span className="ab-metric-label">Brier</span><span className="ab-metric-value">{basePM.brier_score.toFixed(4)}</span></div>
                </>}
              </div>
              <div className="ab-pair-vs">VS</div>
              <div className={`ab-pair-card ${winner === 'challenger' ? 'winner' : ''}`} style={{ borderTopColor: COLORS[challIdx % COLORS.length] }}>
                <div className="ab-pair-role">CHALLENGER {winner === 'challenger' && <span className="ab-winner-tag">WIN</span>}</div>
                <div className="ab-model-name" style={{ color: COLORS[challIdx % COLORS.length] }}>{chall.name}</div>
                <div className="ab-metric-row"><span className="ab-metric-label">AUC</span><span className="ab-metric-value">{chall.auc.toFixed(3)}</span></div>
                {challPM && <>
                  <div className="ab-metric-row"><span className="ab-metric-label">Best F1</span><span className="ab-metric-value">{challPM.best_f1.toFixed(3)} @{(challPM.best_f1_threshold * 100).toFixed(0)}%</span></div>
                  <div className="ab-metric-row"><span className="ab-metric-label">Brier</span><span className="ab-metric-value">{challPM.brier_score.toFixed(4)}</span></div>
                </>}
              </div>
            </div>
            <h4 className="lab-subsection-title">概率时间线</h4>
            <ProbTimeline baseline={base} challenger={chall} baseColor={COLORS[baseIdx % COLORS.length]} challColor={COLORS[challIdx % COLORS.length]} />
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

  if (error) return <div className="lab-container"><div className="lab-card"><p>Phase 3 data not available: {error}</p></div></div>;
  if (!data) return <div className="loading">Loading Phase 3 data...</div>;

  const { experiments, pairwise, feature_importances, summary, practical_summary } = data;

  return (
    <div className="lab-container">
      <header className="lab-header">
        <div>
          <h1>Ch.2 Non-linear Models</h1>
          <p className="lab-subtitle">Phase 3 · GBDT / RandomForest vs Logistic Regression</p>
        </div>
        <div className="lab-model-badge">
          <span className="lab-badge-version">Phase {data.phase}</span>
          {practical_summary && <span className="lab-badge-auc">Best F1: {practical_summary.best_f1.toFixed(3)}</span>}
        </div>
      </header>

      {/* Summary card */}
      <section className="lab-card">
        <div className="best-config-card" style={{ marginTop: 0 }}>
          <div className="best-config-header">
            <span className="best-config-badge">CURRENT STATUS</span>
            <div className="best-config-title">
              <span className="best-config-name">{summary.best_model}</span>
              {practical_summary && <span className="best-config-auc">F1 {practical_summary.best_f1.toFixed(3)}</span>}
            </div>
          </div>
          <p className="best-config-desc">
            所有模型均采用 Unbalanced 训练（见 Ch.2.1 指标探索），确保概率输出可直接用于决策。
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
              <span className="roadmap-stat-value">{experiments.length}</span>
              <span className="roadmap-stat-label">模型数</span>
            </div>
            {practical_summary && (
              <div className="roadmap-stat">
                <span className="roadmap-stat-value" style={{ color: '#3a82d6' }}>{practical_summary.best_brier.toFixed(4)}</span>
                <span className="roadmap-stat-label">Best Brier</span>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Step-by-step summary */}
      <section className="lab-card">
        <div className="ab-header">
          <h2>实验摘要</h2>
          <span className="ab-badge" style={{ background: '#eff6ff', color: '#1e40af', border: '1px solid #bfdbfe' }}>4 STEPS</span>
        </div>
        <div className="lab-table-wrap">
          <table className="lab-table">
            <thead>
              <tr><th>Step</th><th>做法</th><th>代表模型</th><th>F1</th><th>结论</th></tr>
            </thead>
            <tbody>
              <tr style={{ background: 'rgba(34,197,94,0.06)' }}>
                <td style={{ fontWeight: 600 }}>1 · 模型类型</td>
                <td style={{ fontSize: 12 }}>LR / GBDT / RF × Slim(10) / Full(23) — 全部 Unbalanced</td>
                <td style={{ fontWeight: 600, color: '#16a34a' }}>LR Slim</td>
                <td className="lab-td-mono" style={{ fontWeight: 700, color: '#16a34a' }}>0.588</td>
                <td style={{ fontSize: 12 }}>LR 最佳 · GBDT/RF 小样本下过拟合</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>2 · Regime 特征</td>
                <td style={{ fontSize: 12 }}>Fed利率方向 + CPI趋势 + 曲线倒挂 · 分 9 特征全量 / 2 特征精简</td>
                <td>LR +2特征</td>
                <td className="lab-td-mono">0.529</td>
                <td style={{ fontSize: 12, color: '#dc2626' }}>9 特征严重有害(0.208) · 2 特征仍不如 baseline</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>3 · 事件日历</td>
                <td style={{ fontSize: 12 }}>FOMC / CPI / NFP 前后天数 + 窗口标记 · 含 KitchenSink 全组合</td>
                <td>LR KitchenSink</td>
                <td className="lab-td-mono">0.541</td>
                <td style={{ fontSize: 12, color: '#dc2626' }}>事件特征无增量 · Brier=0.099 校准最佳</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>4 · 长期重测</td>
                <td style={{ fontSize: 12 }}>2005+ 数据(5333样本) · LR / GBDT / RF 全部重跑</td>
                <td>RF Ext</td>
                <td className="lab-td-mono">0.322</td>
                <td style={{ fontSize: 12, color: '#dc2626' }}>AUC 从 0.89→0.57 · 跨周期非平稳性是根本瓶颈</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: 12, padding: '10px 12px', background: '#f0fdf4', borderRadius: 6, border: '1px solid #bbf7d0' }}>
          <p style={{ margin: 0, fontSize: 13, fontWeight: 600, color: '#166534' }}>
            总结：LR Slim (10特征, F1=0.588) 至今未被超越。加特征 = 加噪声，换模型 = 加过拟合，加数据 = 加非平稳性。
          </p>
        </div>
      </section>

      {/* Overview table */}
      <section className="lab-card ab-section">
        <div className="ab-header">
          <h2>全量模型对比</h2>
          <span className="ab-badge">{experiments.length} MODELS</span>
        </div>
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
            LR 用系数绝对值，树模型用信息增益。排序差异揭示非线性交互效应。
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
            <span className="roadmap-check">&#10003;</span>
            <div className="roadmap-step-body">
              <div className="roadmap-step-title">Step 1 · 非线性模型 Baseline</div>
              <div className="roadmap-step-desc">LR Slim F1=0.588 仍为最佳 · GBDT/RF 在小样本下不如 LR</div>
            </div>
          </div>
          <div className="roadmap-item done">
            <span className="roadmap-check">&#10003;</span>
            <div className="roadmap-step-body">
              <div className="roadmap-step-title">Step 2 · Regime 特征</div>
              <div className="roadmap-step-desc">9 特征全量反而有害 · 2 特征精简版 F1=0.529 · 结论：当前数据周期单一，regime 信号是噪声</div>
            </div>
          </div>
          <div className="roadmap-item done">
            <span className="roadmap-check">&#10003;</span>
            <div className="roadmap-step-body">
              <div className="roadmap-step-title">Step 3 · 事件日历</div>
              <div className="roadmap-step-desc">FOMC/CPI/NFP 窗口特征 · LR+Events F1=0.516 · KitchenSink F1=0.541 · 均不如 Slim baseline</div>
            </div>
          </div>
          <div className="roadmap-item done">
            <span className="roadmap-check">&#10003;</span>
            <div className="roadmap-step-body">
              <div className="roadmap-step-title">Step 4 · 长期重测 (2005+)</div>
              <div className="roadmap-step-desc">5333 样本 · LR/GBDT/RF 均 F1≈0.31 · 长期数据 AUC 显著退化（0.57~0.60）· 与 Ch.1 结论一致：线性模型 + 跨周期非平稳性</div>
            </div>
          </div>
        </div>
      </section>

      <footer className="footer">
        <div>Ch.2 Non-linear Models · Unbalanced training · F1/Brier evaluation</div>
      </footer>
    </div>
  );
}

export function Phase3Lab() {
  return <ErrorBoundary><Phase3LabInner /></ErrorBoundary>;
}
