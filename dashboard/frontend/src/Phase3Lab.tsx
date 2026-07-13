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
  brier_score: number;
  base_rate: number;
  best_f1: number;
  best_f1_threshold: number;
  p_at_80: number;
  r_at_80: number;
  lift_at_80: number;
  mean_prob: number;
  median_prob: number;
  prob_gt50_pct: number;
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
interface PracticalSummary {
  best_f1_model: string; best_f1: number;
  best_brier_model: string; best_brier: number;
  best_lift_model: string; best_lift: number;
}
interface Phase3Data {
  phase: number; title: string;
  experiments: ExperimentData[];
  pairwise: PairwiseConfig[];
  feature_importances: Record<string, FeatureImp[]>;
  practical_summary?: PracticalSummary;
  summary: {
    lr_slim_auc: number; gbdt_slim_auc: number; gbdt_full_auc: number;
    rf_slim_auc: number; lr_full_auc: number;
    best_model: string; data_range: string; total_samples: number;
  };
}

const COLORS = ['#d6457a', '#3a82d6', '#16a34a', '#ea580c', '#8b5cf6', '#0d9488', '#b45309', '#6366f1'];

function MetricCell({ value, best, fmt = 'pct', higher = true }: { value: number; best: number; fmt?: 'pct' | 'num' | 'x'; higher?: boolean }) {
  const isBest = higher ? value >= best - 0.001 : value <= best + 0.001;
  const display = fmt === 'pct' ? `${(value * 100).toFixed(1)}%`
    : fmt === 'x' ? `${value.toFixed(1)}x`
    : value.toFixed(4);
  return (
    <td className="lab-td-mono" style={isBest ? { fontWeight: 700, color: '#16a34a' } : undefined}>
      {display}
    </td>
  );
}

function OverviewTable({ experiments }: { experiments: ExperimentData[] }) {
  const withPM = experiments.filter(e => e.practical_metrics);
  if (withPM.length === 0) return null;

  const bestAuc = Math.max(...withPM.map(e => e.auc));
  const bestF1 = Math.max(...withPM.map(e => e.practical_metrics!.best_f1));
  const bestBrier = Math.min(...withPM.map(e => e.practical_metrics!.brier_score));
  const bestLift = Math.max(...withPM.map(e => e.practical_metrics!.lift_at_80));
  const baseRate = withPM[0]?.practical_metrics?.base_rate ?? 0;

  return (
    <div className="ab-overview-table">
      <div className="lab-table-wrap" style={{ overflowX: 'auto' }}>
        <table className="lab-table">
          <thead>
            <tr>
              <th>模型</th>
              <th title="排序能力">AUC</th>
              <th title="校准准确度（越低越好）">Brier</th>
              <th title="输出均值 vs base rate 差距">Mean P</th>
              <th title="80%阈值精确率">P@80%</th>
              <th title="相对 base rate 提升">Lift@80%</th>
              <th title="最优 F1 及对应阈值">Best F1</th>
              <th title="超过 50% 的预测占比">P{'>'}50%</th>
            </tr>
          </thead>
          <tbody>
            {withPM.map((exp, i) => {
              const pm = exp.practical_metrics!;
              return (
                <tr key={i}>
                  <td style={{ color: COLORS[i % COLORS.length], fontWeight: 600, whiteSpace: 'nowrap' }}>
                    {exp.name}
                  </td>
                  <MetricCell value={exp.auc} best={bestAuc} fmt="num" />
                  <MetricCell value={pm.brier_score} best={bestBrier} fmt="num" higher={false} />
                  <td className="lab-td-mono" style={{
                    color: Math.abs(pm.mean_prob - baseRate) < 0.03 ? '#16a34a' : '#dc2626',
                  }}>
                    {(pm.mean_prob * 100).toFixed(1)}%
                    <span style={{ fontSize: 10, opacity: 0.6, marginLeft: 4 }}>
                      (base: {(baseRate * 100).toFixed(1)}%)
                    </span>
                  </td>
                  <MetricCell value={pm.p_at_80} best={Math.max(...withPM.map(e => e.practical_metrics!.p_at_80))} fmt="pct" />
                  <MetricCell value={pm.lift_at_80} best={bestLift} fmt="x" />
                  <td className="lab-td-mono" style={pm.best_f1 >= bestF1 - 0.001 ? { fontWeight: 700, color: '#16a34a' } : undefined}>
                    {pm.best_f1.toFixed(3)}
                    <span style={{ fontSize: 10, opacity: 0.6, marginLeft: 2 }}>
                      @{(pm.best_f1_threshold * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="lab-td-mono">{pm.prob_gt50_pct.toFixed(1)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p style={{ fontSize: 11, color: '#8a7882', marginTop: 8 }}>
        Base Rate = {(baseRate * 100).toFixed(1)}% · Mean P 接近 Base Rate = 概率校准良好 · Brier 越低 = 概率越准 · Lift@80% = P@80% / Base Rate
      </p>
    </div>
  );
}

function CalibrationInsight({ experiments }: { experiments: ExperimentData[] }) {
  const withPM = experiments.filter(e => e.practical_metrics);
  if (withPM.length === 0) return null;
  const baseRate = withPM[0]?.practical_metrics?.base_rate ?? 0;

  const calibrated = withPM.filter(e => Math.abs(e.practical_metrics!.mean_prob - baseRate) < 0.05);
  const overconfident = withPM.filter(e => e.practical_metrics!.mean_prob > baseRate + 0.05);

  return (
    <div className="best-config-card" style={{ borderLeftColor: '#3a82d6' }}>
      <div className="best-config-header">
        <span className="best-config-badge" style={{ background: '#eff6ff', color: '#1e40af' }}>CALIBRATION ANALYSIS</span>
      </div>
      <div style={{ fontSize: 13, lineHeight: 1.8 }}>
        <p><strong>核心发现：</strong>AUC 高不等于模型可用。关键在于概率是否校准。</p>
        <ul style={{ margin: '8px 0', paddingLeft: 20 }}>
          <li>
            <span style={{ color: '#16a34a', fontWeight: 600 }}>校准良好</span>（Mean P ≈ Base Rate {(baseRate * 100).toFixed(1)}%）：
            {calibrated.length > 0 ? calibrated.map(e => e.name).join(', ') : '无'}
          </li>
          <li>
            <span style={{ color: '#dc2626', fontWeight: 600 }}>过度自信</span>（Mean P 远超 Base Rate）：
            {overconfident.length > 0 ? overconfident.map(e => `${e.name} (${(e.practical_metrics!.mean_prob * 100).toFixed(1)}%)`).join(', ') : '无'}
          </li>
        </ul>
        <p style={{ fontSize: 12, color: '#6b5f63' }}>
          class_weight=balanced 上调正样本权重 → 模型倾向输出高概率 → 50% 阈值下大量误报。
          移除 balanced / 加 isotonic calibration → 概率回归真实频率 → 相同阈值下精确率更高。
        </p>
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

  const baseRate = baseline.practical_metrics?.base_rate ?? 0.12;

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(241, 216, 226, 0.6)" />
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
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(241, 216, 226, 0.6)" horizontal={false} />
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
        逐一对比 baseline 与 challenger，核心关注概率校准（Brier）和实战指标（F1, Lift）。
      </p>

      {pairwise.map((pair) => {
        const base = find(pair.baseline);
        const chall = find(pair.challenger);
        if (!base || !chall) return null;
        const baseIdx = experiments.indexOf(base);
        const challIdx = experiments.indexOf(chall);
        const basePM = base.practical_metrics;
        const challPM = chall.practical_metrics;
        const winnerMetric = (challPM?.best_f1 ?? 0) > (basePM?.best_f1 ?? 0) ? 'challenger' : 'baseline';

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
              <div className={`ab-pair-card ${winnerMetric === 'baseline' ? 'winner' : ''}`} style={{ borderTopColor: COLORS[baseIdx % COLORS.length] }}>
                <div className="ab-pair-role">
                  BASELINE {winnerMetric === 'baseline' && <span className="ab-winner-tag">WIN</span>}
                </div>
                <div className="ab-model-name" style={{ color: COLORS[baseIdx % COLORS.length] }}>{base.name}</div>
                <div className="ab-metric-row"><span className="ab-metric-label">AUC</span><span className="ab-metric-value">{base.auc.toFixed(3)}</span></div>
                {basePM && <>
                  <div className="ab-metric-row"><span className="ab-metric-label">Best F1</span><span className="ab-metric-value">{basePM.best_f1.toFixed(3)} @{(basePM.best_f1_threshold * 100).toFixed(0)}%</span></div>
                  <div className="ab-metric-row"><span className="ab-metric-label">Brier</span><span className="ab-metric-value">{basePM.brier_score.toFixed(4)}</span></div>
                  <div className="ab-metric-row"><span className="ab-metric-label">Mean P</span><span className="ab-metric-value">{(basePM.mean_prob * 100).toFixed(1)}%</span></div>
                  <div className="ab-metric-row"><span className="ab-metric-label">Lift@80%</span><span className="ab-metric-value">{basePM.lift_at_80.toFixed(1)}x</span></div>
                </>}
              </div>
              <div className="ab-pair-vs">VS</div>
              <div className={`ab-pair-card ${winnerMetric === 'challenger' ? 'winner' : ''}`} style={{ borderTopColor: COLORS[challIdx % COLORS.length] }}>
                <div className="ab-pair-role">
                  CHALLENGER {winnerMetric === 'challenger' && <span className="ab-winner-tag">WIN</span>}
                </div>
                <div className="ab-model-name" style={{ color: COLORS[challIdx % COLORS.length] }}>{chall.name}</div>
                <div className="ab-metric-row"><span className="ab-metric-label">AUC</span><span className="ab-metric-value">{chall.auc.toFixed(3)}</span></div>
                {challPM && <>
                  <div className="ab-metric-row"><span className="ab-metric-label">Best F1</span><span className="ab-metric-value">{challPM.best_f1.toFixed(3)} @{(challPM.best_f1_threshold * 100).toFixed(0)}%</span></div>
                  <div className="ab-metric-row"><span className="ab-metric-label">Brier</span><span className="ab-metric-value">{challPM.brier_score.toFixed(4)}</span></div>
                  <div className="ab-metric-row"><span className="ab-metric-label">Mean P</span><span className="ab-metric-value">{(challPM.mean_prob * 100).toFixed(1)}%</span></div>
                  <div className="ab-metric-row"><span className="ab-metric-label">Lift@80%</span><span className="ab-metric-value">{challPM.lift_at_80.toFixed(1)}x</span></div>
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

  if (error) return <div className="lab-container"><div className="lab-card"><p>Phase 3 data not available yet: {error}</p></div></div>;
  if (!data) return <div className="loading">Loading Phase 3 data...</div>;

  const { experiments, pairwise, feature_importances, summary, practical_summary } = data;

  return (
    <div className="lab-container">
      <header className="lab-header">
        <div>
          <h1>Ch.2 Model Evolution</h1>
          <p className="lab-subtitle">Phase 3 · 优化目标重定义 — 从 AUC 到 F1 / 校准 / Lift</p>
        </div>
        <div className="lab-model-badge">
          <span className="lab-badge-version">Phase {data.phase}</span>
          {practical_summary && <span className="lab-badge-auc">Best F1: {practical_summary.best_f1.toFixed(3)}</span>}
        </div>
      </header>

      {/* Key insight */}
      <section className="lab-card">
        <div className="best-config-card" style={{ marginTop: 0, borderLeftColor: '#dc2626' }}>
          <div className="best-config-header">
            <span className="best-config-badge" style={{ background: '#fef2f2', color: '#991b1b' }}>PROBLEM</span>
            <div className="best-config-title">
              <span className="best-config-name">AUC 高 ≠ 模型可用</span>
            </div>
          </div>
          <div style={{ fontSize: 13, lineHeight: 1.8 }}>
            <p>上一轮 GBDT Full AUC=0.897，但 P@50%=10%，相当于每 10 次预警有 9 次误报。</p>
            <p>根因：<code>class_weight=balanced</code> 让模型输出概率严重偏高（mean_prob 远超 base rate）。</p>
            <p style={{ fontWeight: 600, marginTop: 8 }}>新评估体系：</p>
            <ul style={{ margin: '4px 0', paddingLeft: 20 }}>
              <li><strong>Brier Score</strong> — 概率校准度（越低越好，衡量「说 80% 时是否真有 80% 概率」）</li>
              <li><strong>Best F1 @ 最优阈值</strong> — 精确率/召回率平衡点（实战决策指标）</li>
              <li><strong>Lift@80%</strong> — 80% 阈值下精确率 / Base Rate（{'>'} 1 才有预测价值）</li>
              <li><strong>Mean Prob vs Base Rate</strong> — 差距越小 = 校准越好</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Practical summary */}
      {practical_summary && (
        <section className="lab-card">
          <div className="roadmap-status-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
            <div className="roadmap-stat">
              <span className="roadmap-stat-value" style={{ color: '#16a34a', fontSize: 18 }}>{practical_summary.best_f1.toFixed(3)}</span>
              <span className="roadmap-stat-label">Best F1</span>
              <span style={{ fontSize: 10, color: '#8a7882' }}>{practical_summary.best_f1_model}</span>
            </div>
            <div className="roadmap-stat">
              <span className="roadmap-stat-value" style={{ color: '#3a82d6', fontSize: 18 }}>{practical_summary.best_brier.toFixed(4)}</span>
              <span className="roadmap-stat-label">Best Brier</span>
              <span style={{ fontSize: 10, color: '#8a7882' }}>{practical_summary.best_brier_model}</span>
            </div>
            <div className="roadmap-stat">
              <span className="roadmap-stat-value" style={{ color: '#ea580c', fontSize: 18 }}>{practical_summary.best_lift.toFixed(1)}x</span>
              <span className="roadmap-stat-label">Best Lift@80%</span>
              <span style={{ fontSize: 10, color: '#8a7882' }}>{practical_summary.best_lift_model}</span>
            </div>
            <div className="roadmap-stat">
              <span className="roadmap-stat-value text-sm">{summary.data_range}</span>
              <span className="roadmap-stat-label">{summary.total_samples.toLocaleString()} 样本</span>
            </div>
          </div>
        </section>
      )}

      {/* Full comparison table */}
      <section className="lab-card ab-section">
        <div className="ab-header">
          <h2>全量模型对比</h2>
          <span className="ab-badge" style={{ background: '#f0fdf4', color: '#166534', border: '1px solid #bbf7d0' }}>PRACTICAL METRICS</span>
          <span className="ab-badge">{experiments.length} MODELS</span>
        </div>
        <p className="lab-card-desc">
          核心指标变更：AUC 仅作参考，以 F1 / Brier / Lift 作为优化目标。绿色 = 该列最佳。
        </p>
        <OverviewTable experiments={experiments} />
      </section>

      {/* Calibration insight */}
      <section className="lab-card">
        <CalibrationInsight experiments={experiments} />
      </section>

      <PairwiseSection pairwise={pairwise} experiments={experiments} />

      {/* Feature importance */}
      {Object.keys(feature_importances).length > 0 && (
        <section className="lab-card">
          <div className="ab-header">
            <h2>特征重要性对比</h2>
            <span className="ab-badge">FEATURE IMP</span>
          </div>
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
              <div className="roadmap-step-title">Step 1 · 非线性模型 + 优化目标重定义</div>
              <div className="roadmap-step-detail">
                <span className="roadmap-metric-chip">GBDT / RF / LR</span>
                <span className="roadmap-metric-chip">Balanced vs Unbalanced</span>
                <span className="roadmap-metric-chip">Isotonic Calibration</span>
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
        <div>Ch.2 Model Evolution · 优化目标: F1 / Brier / Lift (非 AUC)</div>
      </footer>
    </div>
  );
}

export function Phase3Lab() {
  return <ErrorBoundary><Phase3LabInner /></ErrorBoundary>;
}
