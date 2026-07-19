import { useState, useEffect, useMemo, Component, type ReactNode } from 'react';
import { StackedProbSPChart } from './components/StackedProbSPChart';
import { LazyMount } from './components/LazyMount';
import { fetchDataJson } from './api';
import { mergeExperimentTimeline, downsample } from './utils/chart';
import { ResearchTrackNotice } from './components/ResearchTrackNotice';

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
interface SP500Point { date: string; sp500: number; }
interface ExperimentData {
  name: string; auc: number;
  threshold_analysis: ThresholdRow[];
  events_backtest: EventBacktest[];
  probability_timeline: { date: string; probability: number }[];
  sp500_timeline?: SP500Point[];
  practical_metrics?: PracticalMetrics;
}
interface PairwiseConfig { id: string; label: string; variable: string; baseline: string; challenger: string; method_note: string; }
interface PracticalSummary { best_f1_model: string; best_f1: number; best_brier_model: string; best_brier: number; best_lift_model: string; best_lift: number; }
interface MetricData {
  title: string;
  experiments: ExperimentData[];
  pairwise: PairwiseConfig[];
  practical_summary: PracticalSummary;
  summary: { data_range: string; total_samples: number; base_rate: number };
}

const COLORS = ['#d6457a', '#3a82d6', '#16a34a', '#ea580c', '#8b5cf6', '#0d9488', '#b45309'];

function ProbTimeline({ baseline, challenger, baseColor, challColor }: {
  baseline: ExperimentData; challenger: ExperimentData; baseColor: string; challColor: string; baseRate?: number;
}) {
  const sp500Timeline = baseline.sp500_timeline ?? challenger.sp500_timeline ?? [];

  const chartData = useMemo(() => {
    const full = mergeExperimentTimeline([baseline, challenger], sp500Timeline);
    return downsample(full);
  }, [baseline, challenger, sp500Timeline]);

  const series = useMemo(() => [
    { dataKey: 'prob_0', name: baseline.name, color: baseColor },
    { dataKey: 'prob_1', name: challenger.name, color: challColor },
  ], [baseline.name, challenger.name, baseColor, challColor]);

  return (
    <LazyMount minHeight={340}>
      <StackedProbSPChart
        data={chartData}
        series={series}
        probHeight={220}
        spHeight={110}
        showLegend
        showThreshold
      />
    </LazyMount>
  );
}

function MetricLabInner() {
  const [data, setData] = useState<MetricData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDataJson<MetricData>('metric_exploration.json')
      .then(d => setData(d))
      .catch(e => setError(e.message));
  }, []);

  if (error) return <div className="lab-container"><div className="lab-card"><p>Metric exploration data not available: {error}</p></div></div>;
  if (!data) return <div className="loading">Loading...</div>;

  const { experiments, pairwise, practical_summary, summary } = data;
  const baseRate = summary.base_rate;
  const find = (name: string) => experiments.find(e => e.name.includes(name));

  const balancedModels = experiments.filter(e => e.practical_metrics && e.practical_metrics.mean_prob > baseRate + 0.15);
  const calibratedModels = experiments.filter(e => e.practical_metrics && Math.abs(e.practical_metrics.mean_prob - baseRate) < 0.15);

  return (
    <div className="lab-container">
      <header className="lab-header">
        <div>
          <h1>Ch.2.1 Metric Exploration</h1>
          <p className="lab-subtitle">评估指标优化 · Balanced vs Unbalanced · 概率校准实验</p>
        </div>
        <div className="lab-model-badge">
          <span className="lab-badge-version">L3</span>
          <span className="lab-badge-auc">Best F1: {practical_summary.best_f1.toFixed(3)}</span>
        </div>
      </header>

      <ResearchTrackNotice track="risk-model" />

      {/* Problem statement */}
      <section className="lab-card">
        <div className="best-config-card" style={{ marginTop: 0, borderLeftColor: '#dc2626' }}>
          <div className="best-config-header">
            <span className="best-config-badge" style={{ background: '#fef2f2', color: '#991b1b' }}>PROBLEM</span>
            <div className="best-config-title"><span className="best-config-name">AUC 高 ≠ 模型可用</span></div>
          </div>
          <div style={{ fontSize: 13, lineHeight: 1.8 }}>
            <p>GBDT Full 的 AUC=0.897，但 P@50%=10%（10 次预警 9 次假的）。根因：</p>
            <p><code>class_weight=balanced</code> 让模型认为大跌是常见事件（上调正样本权重至 50%），输出概率整体虚高（mean 66~77%），远超真实 base rate {(baseRate * 100).toFixed(0)}%。</p>
          </div>
        </div>
      </section>

      {/* Metric definitions */}
      <section className="lab-card">
        <div className="ab-header">
          <h2>新评估指标体系</h2>
          <span className="ab-badge" style={{ background: '#eff6ff', color: '#1e40af', border: '1px solid #bfdbfe' }}>DEFINITIONS</span>
        </div>
        <div className="lab-table-wrap">
          <table className="lab-table">
            <thead><tr><th>指标</th><th>含义</th><th>理想值</th></tr></thead>
            <tbody>
              <tr><td style={{ fontWeight: 600 }}>Base Rate</td><td>历史上大跌天数占比，是「瞎猜」的精确率上限</td><td className="lab-td-mono">{(baseRate * 100).toFixed(1)}%</td></tr>
              <tr><td style={{ fontWeight: 600 }}>Mean Prob</td><td>模型输出概率的均值，应接近 Base Rate</td><td className="lab-td-mono">≈ {(baseRate * 100).toFixed(0)}%</td></tr>
              <tr><td style={{ fontWeight: 600 }}>Brier Score</td><td>预测概率 vs 实际结果的均方误差</td><td className="lab-td-mono">越低越好（完美=0）</td></tr>
              <tr><td style={{ fontWeight: 600 }}>Best F1</td><td>精确率与召回率的最优平衡点</td><td className="lab-td-mono">越高越好</td></tr>
              <tr><td style={{ fontWeight: 600 }}>P@80%</td><td>当模型输出 {'>'}80% 时的精确率</td><td className="lab-td-mono">{'>'}50%</td></tr>
              <tr><td style={{ fontWeight: 600 }}>Lift@80%</td><td>P@80% / Base Rate — 比瞎猜好多少倍</td><td className="lab-td-mono">{'>'}3x</td></tr>
              <tr><td style={{ fontWeight: 600 }}>P{'>'}50%</td><td>输出概率超过 50% 的预测占比（过高=概率虚高）</td><td className="lab-td-mono">{'<'}20%</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Full comparison table */}
      <section className="lab-card ab-section">
        <div className="ab-header">
          <h2>全量模型对比</h2>
          <span className="ab-badge">{experiments.length} MODELS</span>
        </div>
        <div className="lab-table-wrap" style={{ overflowX: 'auto' }}>
          <table className="lab-table">
            <thead>
              <tr>
                <th>模型</th><th>AUC</th><th>Brier ↓</th><th>Mean P</th>
                <th>P{'>'}50%</th><th>Best F1</th><th>@阈值</th><th>P@80%</th><th>Lift</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp, i) => {
                const pm = exp.practical_metrics;
                if (!pm) return null;
                const probOk = Math.abs(pm.mean_prob - baseRate) < 0.15;
                return (
                  <tr key={i}>
                    <td style={{ color: COLORS[i % COLORS.length], fontWeight: 600, whiteSpace: 'nowrap' }}>{exp.name}</td>
                    <td className="lab-td-mono">{exp.auc.toFixed(3)}</td>
                    <td className="lab-td-mono">{pm.brier_score.toFixed(4)}</td>
                    <td className="lab-td-mono" style={{ color: probOk ? '#16a34a' : '#dc2626', fontWeight: 600 }}>
                      {(pm.mean_prob * 100).toFixed(1)}%
                    </td>
                    <td className="lab-td-mono" style={{ color: pm.prob_gt50_pct > 30 ? '#dc2626' : '#16a34a' }}>
                      {pm.prob_gt50_pct.toFixed(1)}%
                    </td>
                    <td className="lab-td-mono" style={{ fontWeight: 600 }}>{pm.best_f1.toFixed(3)}</td>
                    <td className="lab-td-mono">{(pm.best_f1_threshold * 100).toFixed(0)}%</td>
                    <td className="lab-td-mono">{pm.p_at_80 > 0 ? `${(pm.p_at_80 * 100).toFixed(1)}%` : '—'}</td>
                    <td className="lab-td-mono">{pm.lift_at_80 > 0 ? `${pm.lift_at_80.toFixed(1)}x` : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p style={{ fontSize: 11, color: '#8a7882', marginTop: 8 }}>
          Base Rate = {(baseRate * 100).toFixed(1)}% · Mean P 绿色 = 接近 base rate · P{'>'}50% 红色 = 概率虚高
        </p>
      </section>

      {/* Calibration insight */}
      <section className="lab-card">
        <div className="best-config-card" style={{ borderLeftColor: '#3a82d6' }}>
          <div className="best-config-header">
            <span className="best-config-badge" style={{ background: '#eff6ff', color: '#1e40af' }}>CALIBRATION ANALYSIS</span>
          </div>
          <div style={{ fontSize: 13, lineHeight: 1.8 }}>
            <p><strong>核心发现：</strong>AUC 高不等于模型可用。关键在于概率是否校准。</p>
            <ul style={{ margin: '8px 0', paddingLeft: 20 }}>
              <li>
                <span style={{ color: '#dc2626', fontWeight: 600 }}>概率虚高</span>（Mean P 远超 {(baseRate * 100).toFixed(0)}%）：
                {balancedModels.length > 0 ? balancedModels.map(e => `${e.name} (${(e.practical_metrics!.mean_prob * 100).toFixed(0)}%)`).join(', ') : '无'}
              </li>
              <li>
                <span style={{ color: '#16a34a', fontWeight: 600 }}>概率合理</span>（Mean P 接近 base rate）：
                {calibratedModels.length > 0 ? calibratedModels.map(e => `${e.name} (${(e.practical_metrics!.mean_prob * 100).toFixed(0)}%)`).join(', ') : '无'}
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Pairwise comparisons */}
      <section className="lab-card ab-section">
        <div className="ab-header">
          <h2>对比实验</h2>
          <span className="ab-badge">{pairwise.length} PAIRS</span>
        </div>

        {pairwise.map(pair => {
          const base = find(pair.baseline);
          const chall = find(pair.challenger);
          if (!base || !chall) return null;
          const baseIdx = experiments.indexOf(base);
          const challIdx = experiments.indexOf(chall);
          const bPM = base.practical_metrics;
          const cPM = chall.practical_metrics;
          const winner = (cPM?.best_f1 ?? 0) > (bPM?.best_f1 ?? 0) ? 'challenger' : 'baseline';

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
                {[
                  { role: 'BASELINE', exp: base, pm: bPM, isWin: winner === 'baseline', idx: baseIdx },
                  { role: 'CHALLENGER', exp: chall, pm: cPM, isWin: winner === 'challenger', idx: challIdx },
                ].map(({ role, exp, pm, isWin, idx }) => (
                  <div key={role} className={`ab-pair-card ${isWin ? 'winner' : ''}`} style={{ borderTopColor: COLORS[idx % COLORS.length] }}>
                    <div className="ab-pair-role">{role} {isWin && <span className="ab-winner-tag">WIN</span>}</div>
                    <div className="ab-model-name" style={{ color: COLORS[idx % COLORS.length] }}>{exp.name}</div>
                    <div className="ab-metric-row"><span className="ab-metric-label">AUC</span><span className="ab-metric-value">{exp.auc.toFixed(3)}</span></div>
                    {pm && <>
                      <div className="ab-metric-row"><span className="ab-metric-label">Best F1</span><span className="ab-metric-value">{pm.best_f1.toFixed(3)} @{(pm.best_f1_threshold * 100).toFixed(0)}%</span></div>
                      <div className="ab-metric-row"><span className="ab-metric-label">Brier</span><span className="ab-metric-value">{pm.brier_score.toFixed(4)}</span></div>
                      <div className="ab-metric-row">
                        <span className="ab-metric-label">Mean P</span>
                        <span className="ab-metric-value" style={{ color: Math.abs(pm.mean_prob - baseRate) < 0.15 ? '#16a34a' : '#dc2626' }}>
                          {(pm.mean_prob * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="ab-metric-row"><span className="ab-metric-label">P{'>'}50%</span><span className="ab-metric-value">{pm.prob_gt50_pct.toFixed(1)}%</span></div>
                    </>}
                  </div>
                ))}
                <div className="ab-pair-vs">VS</div>
              </div>

              <h4 className="lab-subsection-title">概率时间线（红色虚线=50%，灰色虚线=base rate）</h4>
              <ProbTimeline baseline={base} challenger={chall} baseColor={COLORS[baseIdx % COLORS.length]} challColor={COLORS[challIdx % COLORS.length]} />
            </div>
          );
        })}
      </section>

      {/* Conclusion: metric decision */}
      <section className="lab-card">
        <div className="best-config-card" style={{ borderLeftColor: '#16a34a' }}>
          <div className="best-config-header">
            <span className="best-config-badge" style={{ background: '#f0fdf4', color: '#166534' }}>DECISION</span>
            <div className="best-config-title"><span className="best-config-name">后续统一评估标准</span></div>
          </div>
          <div style={{ fontSize: 13, lineHeight: 1.8 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', margin: '8px 0 12px' }}>
              <tbody>
                <tr style={{ borderBottom: '1px solid #f1d8e2' }}>
                  <td style={{ padding: '10px 8px', fontWeight: 700, color: '#16a34a', fontSize: 15, width: 100 }}>主指标</td>
                  <td style={{ padding: '10px 8px' }}>
                    <strong style={{ fontSize: 15 }}>Best F1</strong>
                    <span style={{ marginLeft: 8, fontSize: 12, color: '#6b5f63' }}>精确率 × 召回率的最优平衡 — 直接衡量「能不能拿来做交易决策」</span>
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: '10px 8px', fontWeight: 700, color: '#3a82d6', fontSize: 15 }}>辅指标</td>
                  <td style={{ padding: '10px 8px' }}>
                    <strong style={{ fontSize: 15 }}>Brier Score</strong>
                    <span style={{ marginLeft: 8, fontSize: 12, color: '#6b5f63' }}>概率校准度 — 确保 F1 不是来自虚高概率的巧合</span>
                  </td>
                </tr>
              </tbody>
            </table>

            <p style={{ fontSize: 12, color: '#6b5f63', marginTop: 4 }}>
              <strong>AUC</strong> 降级为参考项（排序能力 ≠ 实战决策能力）。其余指标（Lift、P@80%、Mean P）仅用于诊断分析。
            </p>

            <div style={{ marginTop: 12, padding: '10px 12px', background: '#f8fafc', borderRadius: 6, border: '1px solid #e2e8f0' }}>
              <p style={{ margin: 0, fontSize: 12, fontWeight: 600, color: '#475569' }}>实验结论摘要</p>
              <ul style={{ margin: '6px 0 0', paddingLeft: 18, fontSize: 12, color: '#475569', lineHeight: 1.7 }}>
                <li>去掉 <code>class_weight=balanced</code> 是最大单次改进（F1: 0.467 → 0.588）</li>
                <li>训练端修正（去 balanced）比后处理校准（isotonic）更有效</li>
                <li>后续所有实验统一使用 Unbalanced 训练</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <footer className="footer">
        <div>Ch.2.1 Metric Exploration · 主指标 F1 · 辅指标 Brier</div>
      </footer>
    </div>
  );
}

export function MetricLab() {
  return <ErrorBoundary><MetricLabInner /></ErrorBoundary>;
}
