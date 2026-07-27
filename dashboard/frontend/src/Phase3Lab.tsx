import { useState, useEffect, useMemo, Component, type ReactNode } from 'react';
import {
  BarChart, Bar, Cell, CartesianGrid,
  LineChart, Line, Legend, ReferenceLine,
  XAxis, YAxis, Tooltip,
  ResponsiveContainer,
} from 'recharts';
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
interface FeatureImp { feature: string; importance: number; }
interface PracticalSummary { best_f1_model: string; best_f1: number; best_brier_model: string; best_brier: number; best_lift_model: string; best_lift: number; }
interface CorrPair { feat_a: string; feat_b: string; spearman: number; }
interface VifEntry { feature: string; vif: number; }
interface RedundancyGroup { group: string; features: string[]; representative: string; }
interface CorrelationAnalysis {
  high_corr_pairs: CorrPair[];
  vif: VifEntry[];
  redundancy_groups: RedundancyGroup[];
  total_features: number;
  slim_features: string[];
  full_features: string[];
}
interface WFFoldResult {
  model: string; fold: number;
  train_period: string; test_period: string;
  train_n: number; test_n: number; train_pos: number; test_pos: number;
  auc: number;
  practical_metrics: PracticalMetrics;
}
interface WFSummary {
  n_folds: number;
  f1_mean: number; f1_std: number; f1_min: number; f1_max: number;
  brier_mean: number; brier_std: number;
  auc_mean: number;
}
interface WalkForwardData {
  title: string;
  design: Record<string, unknown>;
  data_range: string;
  total_samples: number;
  folds: Array<{ fold: number; train_start: string; train_end: string; test_start: string; test_end: string; train_n: number; test_n: number }>;
  results: WFFoldResult[];
  summary_by_model: Record<string, WFSummary>;
  single_split_baseline: Record<string, { best_f1: number; brier_score: number; auc: number; train_end: string; test_start: string }>;
  verdict: string[];
  decay?: {
    half_life_days: number;
    results: WFFoldResult[];
    summary_by_model: Record<string, WFSummary>;
  };
}
interface TargetGridRow {
  id: string;
  horizon_days: number;
  drawdown_pct: number;
  positive_rate: number;
  wf_f1_mean: number | null;
  wf_f1_std: number | null;
  wf_brier_mean: number | null;
  is_default: boolean;
}
interface TargetSensitivityData {
  title: string;
  grid: TargetGridRow[];
  default_config: TargetGridRow | null;
  best_wf_config: TargetGridRow | null;
  verdict: string[];
}
interface EpisodeEvalSummary {
  hit_at_10: number | null;
  hit_at_15: number | null;
  hit_at_20: number | null;
  mean_lead_days: number | null;
  total_episodes: number;
  false_alarms_per_year_mean: number | null;
  n_folds: number;
}
interface EpisodeEvalData {
  title: string;
  role: string;
  primary_metrics_unchanged: string;
  design: Record<string, unknown>;
  summary_by_model: Record<string, EpisodeEvalSummary>;
  verdict: string[];
}
interface Phase3Data {
  phase: number; title: string;
  experiments: ExperimentData[];
  pairwise: PairwiseConfig[];
  feature_importances: Record<string, FeatureImp[]>;
  practical_summary?: PracticalSummary;
  correlation_analysis?: CorrelationAnalysis;
  walk_forward?: WalkForwardData;
  target_sensitivity?: TargetSensitivityData;
  episode_eval?: EpisodeEvalData;
  regime_models?: RegimeModelsData;
  summary: {
    lr_slim_auc: number; lr_full_auc: number;
    gbdt_slim_auc: number; gbdt_full_auc: number;
    rf_slim_auc: number; rf_full_auc?: number;
    best_model: string; data_range: string; total_samples: number;
  };
}

interface RegimeModelWFSummary {
  auc_mean: number; auc_std: number;
  f1_mean: number; f1_std: number;
  brier_mean: number;
}
interface RegimeModelsData {
  title: string;
  data_range: string;
  n_samples: number;
  positive_rate: number;
  tight_regime_pct: number;
  walk_forward_config: { min_train_years: number; step_years: number; embargo_days: number; n_folds: number };
  walk_forward_summary: Record<string, RegimeModelWFSummary>;
  verdict: string[];
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

const JOURNEY_STEPS = [
  { step: 0, label: 'Ch.1 Baseline', f1: 0.400, brier: 0.18, note: 'LR Balanced + 23 features' },
  { step: 1, label: 'Slim 10特征', f1: 0.480, brier: 0.16, note: '去冗余：23→10 特征' },
  { step: 2, label: 'Embargo 20d', f1: 0.520, brier: 0.15, note: '防 look-ahead bias' },
  { step: 3, label: 'Unbalanced', f1: 0.588, brier: 0.10, note: '去 class_weight=balanced' },
  { step: 4, label: 'Percentile Clip', f1: 0.647, brier: 0.099, note: '修复 clip(-10,10) → percentile clipping' },
  { step: 5, label: '+Events', f1: 0.690, brier: 0.098, note: 'LR Slim+Events: FOMC/CPI/NFP 日历特征' },
  { step: 6, label: 'Events+Interact', f1: 0.688, brier: 0.080, note: 'Events + regime 交互项合并 — F1 持平，Brier 最优' },
];

function JourneyTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: typeof JOURNEY_STEPS[0] }> }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, padding: '8px 12px', fontSize: 12, boxShadow: '0 2px 8px rgba(0,0,0,0.08)' }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{d.label}</div>
      <div>F1: {d.f1.toFixed(3)} · Brier: {d.brier.toFixed(2)}</div>
      <div style={{ color: '#8a7882', marginTop: 2 }}>{d.note}</div>
    </div>
  );
}

function WalkForwardSection({ wf }: { wf: WalkForwardData }) {
  const models = Object.keys(wf.summary_by_model);
  const chartData = wf.folds.map(f => {
    const row: Record<string, number | string> = { fold: `F${f.fold}`, test: f.test_start.slice(0, 7) };
    for (const m of models) {
      const r = wf.results.find(x => x.fold === f.fold && x.model === m);
      if (r) row[m] = r.practical_metrics.best_f1;
    }
    return row;
  });

  return (
    <section className="lab-card">
      <div className="ab-header">
        <h2>Walk-Forward 验证</h2>
        <span className="ab-badge" style={{ background: '#fef3c7', color: '#92400e', border: '1px solid #fcd34d' }}>EXPANDING</span>
      </div>
      <p className="lab-card-desc">
        扩展窗口：训练从 {wf.data_range.split(' ~ ')[0]} 起点累积 · 测试 6 个月 · 步长 6 个月 · Embargo 20d ·
        每折 clip 仅用训练集分位数。对比单次 70/30 split 是否过拟合特定时段。
      </p>

      <div className="lab-table-wrap" style={{ marginBottom: 16 }}>
        <table className="lab-table">
          <thead>
            <tr><th>模型</th><th>单次 Split F1</th><th>WF 均值±σ</th><th>WF F1 范围</th><th>单次 Brier</th><th>WF Brier 均值</th></tr>
          </thead>
          <tbody>
            {models.map(m => {
              const s = wf.summary_by_model[m];
              const b = wf.single_split_baseline[m];
              return (
                <tr key={m}>
                  <td style={{ fontWeight: 600 }}>{m}</td>
                  <td className="lab-td-mono">{b?.best_f1?.toFixed(3) ?? '—'}</td>
                  <td className="lab-td-mono">{s.f1_mean.toFixed(3)} ± {s.f1_std.toFixed(3)}</td>
                  <td className="lab-td-mono">{s.f1_min.toFixed(3)} ~ {s.f1_max.toFixed(3)}</td>
                  <td className="lab-td-mono">{b?.brier_score?.toFixed(3) ?? '—'}</td>
                  <td className="lab-td-mono">{s.brier_mean.toFixed(3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(241,216,226,0.4)" />
          <XAxis dataKey="fold" tick={{ fontSize: 11, fill: '#5c4f56' }} />
          <YAxis domain={[0, 0.8]} tick={{ fontSize: 10, fill: '#16a34a' }} />
          <Tooltip />
          <Legend />
          {models.map((m, i) => (
            <Line key={m} type="monotone" dataKey={m} stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={{ r: 5 }} />
          ))}
        </LineChart>
      </ResponsiveContainer>

      <div className="lab-table-wrap" style={{ marginTop: 16 }}>
        <table className="lab-table" style={{ fontSize: 12 }}>
          <thead>
            <tr><th>Fold</th><th>测试期</th><th>模型</th><th>F1</th><th>Brier</th><th>AUC</th><th>正样本</th></tr>
          </thead>
          <tbody>
            {wf.results.map(r => (
              <tr key={`${r.fold}-${r.model}`}>
                <td>F{r.fold}</td>
                <td>{r.test_period}</td>
                <td>{r.model.replace('LR ', '')}</td>
                <td className="lab-td-mono" style={r.practical_metrics.best_f1 >= 0.6 ? { color: '#16a34a', fontWeight: 700 } : undefined}>
                  {r.practical_metrics.best_f1.toFixed(3)}
                </td>
                <td className="lab-td-mono">{r.practical_metrics.brier_score.toFixed(3)}</td>
                <td className="lab-td-mono" style={{ color: '#8a7882' }}>{r.auc.toFixed(3)}</td>
                <td>{r.test_pos}/{r.test_n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: 12, padding: '10px 12px', background: '#fffbeb', borderRadius: 8, border: '1px solid #fcd34d', fontSize: 12, lineHeight: 1.7 }}>
        <strong style={{ color: '#92400e' }}>结论：</strong>
        单次 split 的高 F1（0.69）主要来自 Fold 4 测试期（2025-10 ~ 2026-04），与固定 70/30 切分高度重叠。
        前 3 折 OOS F1 接近 0，说明模型<strong>尚未证明跨时段稳定</strong>。
      </div>

      {wf.decay && (
        <div style={{ marginTop: 16 }}>
          <h4 className="lab-subsection-title">时间衰减训练（half-life {wf.decay.half_life_days}d）</h4>
          <div className="lab-table-wrap">
            <table className="lab-table" style={{ fontSize: 12 }}>
              <thead>
                <tr><th>模型</th><th>无衰减 WF F1</th><th>+Decay WF F1</th><th>Δ F1</th><th>Decay Brier</th></tr>
              </thead>
              <tbody>
                {Object.keys(wf.summary_by_model).map(baseName => {
                  const base = wf.summary_by_model[baseName];
                  const decayName = `${baseName} +Decay`;
                  const decay = wf.decay!.summary_by_model[decayName];
                  if (!decay) return null;
                  const delta = decay.f1_mean - base.f1_mean;
                  return (
                    <tr key={baseName}>
                      <td>{baseName.replace('LR ', '')}</td>
                      <td className="lab-td-mono">{base.f1_mean.toFixed(3)} ± {base.f1_std.toFixed(3)}</td>
                      <td className="lab-td-mono">{decay.f1_mean.toFixed(3)} ± {decay.f1_std.toFixed(3)}</td>
                      <td className="lab-td-mono" style={{ color: delta > 0.02 ? '#16a34a' : delta < -0.02 ? '#dc2626' : undefined }}>
                        {delta >= 0 ? '+' : ''}{delta.toFixed(3)}
                      </td>
                      <td className="lab-td-mono">{decay.brier_mean.toFixed(3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}

function TargetSensitivitySection({ data }: { data: TargetSensitivityData }) {
  return (
    <section className="lab-card">
      <div className="ab-header">
        <h2>Target 敏感性网格</h2>
        <span className="ab-badge" style={{ background: '#faf5ff', color: '#6d28d9', border: '1px solid #ddd6fe' }}>9 CONFIGS</span>
      </div>
      <p className="lab-card-desc">
        固定 LR Slim+Events，扫描 horizon（10/20/40 天）× 回撤阈值（3%/5%/7%），每格跑 expanding WF。
      </p>
      <div className="lab-table-wrap">
        <table className="lab-table" style={{ fontSize: 12 }}>
          <thead>
            <tr><th>配置</th><th>Horizon</th><th>阈值</th><th>正样本率</th><th>WF F1 均值±σ</th><th>WF Brier</th><th>备注</th></tr>
          </thead>
          <tbody>
            {data.grid.map(r => (
              <tr key={r.id}>
                <td className="lab-td-mono">{r.id}</td>
                <td>{r.horizon_days}d</td>
                <td>{r.drawdown_pct}%</td>
                <td>{(r.positive_rate * 100).toFixed(1)}%</td>
                <td className="lab-td-mono" style={r.wf_f1_mean != null && r.wf_f1_mean >= 0.3 ? { color: '#16a34a', fontWeight: 600 } : undefined}>
                  {r.wf_f1_mean != null ? `${r.wf_f1_mean.toFixed(3)} ± ${(r.wf_f1_std ?? 0).toFixed(3)}` : '—'}
                </td>
                <td className="lab-td-mono">{r.wf_brier_mean?.toFixed(3) ?? '—'}</td>
                <td>{r.is_default ? '默认' : r.id === data.best_wf_config?.id ? 'WF最佳' : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.verdict.length > 0 && (
        <div style={{ marginTop: 12, padding: '10px 12px', background: '#faf5ff', borderRadius: 8, border: '1px solid #ddd6fe', fontSize: 12, lineHeight: 1.7 }}>
          {data.verdict.map((v, i) => <div key={i}>{v}</div>)}
        </div>
      )}
    </section>
  );
}

function RegimeModelsSection({ data }: { data: RegimeModelsData }) {
  const sorted = Object.entries(data.walk_forward_summary)
    .sort(([, a], [, b]) => b.f1_mean - a.f1_mean);
  const bestF1 = sorted[0]?.[1]?.f1_mean ?? 0;

  return (
    <section className="lab-card">
      <div className="ab-header">
        <h2>Regime-Conditional & Non-Linear (1990+)</h2>
        <span className="ab-badge" style={{ background: '#fef3c7', color: '#92400e', border: '1px solid #fcd34d' }}>
          {data.walk_forward_config.n_folds} folds · {data.n_samples} samples
        </span>
      </div>
      <p className="lab-card-desc">
        延长历史至 1990+（{data.n_samples} 交易日，正样本率 {(data.positive_rate * 100).toFixed(1)}%，
        tight regime {data.tight_regime_pct}%）。比较 LR vs GBDT × Slim vs Regime+Interact。
      </p>
      <div className="lab-table-wrap">
        <table className="lab-table">
          <thead>
            <tr><th>Model</th><th>WF F1</th><th>WF AUC</th><th>Brier ↓</th></tr>
          </thead>
          <tbody>
            {sorted.map(([name, s]) => (
              <tr key={name}>
                <td style={{ fontWeight: 600 }}>
                  {Math.abs(s.f1_mean - bestF1) < 0.001 && <span className="ab-best-tag">BEST</span>}
                  {name}
                </td>
                <td className="lab-td-mono" style={Math.abs(s.f1_mean - bestF1) < 0.001 ? { color: '#16a34a', fontWeight: 700 } : undefined}>
                  {s.f1_mean.toFixed(4)} ± {s.f1_std.toFixed(4)}
                </td>
                <td className="lab-td-mono">{s.auc_mean.toFixed(4)} ± {s.auc_std.toFixed(4)}</td>
                <td className="lab-td-mono">{s.brier_mean.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.verdict.length > 0 && (
        <div style={{ marginTop: 12, padding: '10px 14px', background: '#f8fafc', borderRadius: 8, borderLeft: '3px solid #8b5cf6' }}>
          <div style={{ fontWeight: 700, fontSize: 12, color: '#8b5cf6', marginBottom: 4 }}>Verdict</div>
          {data.verdict.map((v, i) => (
            <div key={i} style={{ fontSize: 12, color: '#374151', lineHeight: 1.7 }}>• {v}</div>
          ))}
        </div>
      )}
    </section>
  );
}

function EpisodeEvalSection({ data }: { data: EpisodeEvalData }) {
  const models = Object.keys(data.summary_by_model);
  return (
    <section className="lab-card">
      <div className="ab-header">
        <h2>Episode 评估（辅指标）</h2>
        <span className="ab-badge" style={{ background: '#f0f9ff', color: '#0369a1', border: '1px solid #bae6fd' }}>SUPPLEMENTARY</span>
      </div>
      <p className="lab-card-desc">
        按 crash 事件统计 Hit/Miss（非逐日 F1）。阈值 = train 上 best_f1_threshold；默认解读 Hit@20。
        Primary 仍为 Best F1 + Brier。
      </p>
      <div className="lab-table-wrap">
        <table className="lab-table" style={{ fontSize: 12 }}>
          <thead>
            <tr>
              <th>模型</th>
              <th>Hit@10</th>
              <th>Hit@15</th>
              <th>Hit@20</th>
              <th>Mean Lead</th>
              <th>Episodes</th>
              <th>FA / yr</th>
            </tr>
          </thead>
          <tbody>
            {models.map(name => {
              const s = data.summary_by_model[name];
              const fmt = (v: number | null) => v != null ? v.toFixed(3) : '—';
              return (
                <tr key={name}>
                  <td>{name.replace('LR ', '')}</td>
                  <td className="lab-td-mono">{fmt(s.hit_at_10)}</td>
                  <td className="lab-td-mono">{fmt(s.hit_at_15)}</td>
                  <td className="lab-td-mono" style={s.hit_at_20 != null && s.hit_at_20 >= 0.5 ? { color: '#16a34a', fontWeight: 600 } : undefined}>
                    {fmt(s.hit_at_20)}
                  </td>
                  <td className="lab-td-mono">{s.mean_lead_days != null ? `${s.mean_lead_days}d` : '—'}</td>
                  <td>{s.total_episodes}</td>
                  <td className="lab-td-mono">{s.false_alarms_per_year_mean?.toFixed(1) ?? '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {data.verdict.length > 0 && (
        <div style={{ marginTop: 12, padding: '10px 12px', background: '#f0f9ff', borderRadius: 8, border: '1px solid #bae6fd', fontSize: 12, lineHeight: 1.7 }}>
          {data.verdict.map((v, i) => <div key={i}>{v}</div>)}
        </div>
      )}
    </section>
  );
}

function OptimizationJourney() {
  return (
    <section className="lab-card">
      <div className="ab-header">
        <h2>Optimization Journey</h2>
        <span className="ab-badge" style={{ background: '#f0fdf4', color: '#166534', border: '1px solid #bbf7d0' }}>PROGRESS</span>
      </div>
      <p className="lab-card-desc">从 Ch.1 到 Ch.2 各优化步骤的 F1 / Brier 变化。每个拐点标注了对应的优化方法。</p>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={JOURNEY_STEPS} margin={{ top: 20, right: 30, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(241,216,226,0.4)" />
          <XAxis dataKey="label" tick={{ fontSize: 10, fill: '#5c4f56' }} angle={-25} textAnchor="end" height={60} />
          <YAxis yAxisId="f1" domain={[0, 0.8]} tick={{ fontSize: 10, fill: '#16a34a' }} label={{ value: 'F1', angle: -90, position: 'insideLeft', style: { fill: '#16a34a', fontSize: 11 } }} />
          <YAxis yAxisId="brier" orientation="right" domain={[0, 0.25]} tick={{ fontSize: 10, fill: '#dc2626' }} label={{ value: 'Brier', angle: 90, position: 'insideRight', style: { fill: '#dc2626', fontSize: 11 } }} />
          <Tooltip content={<JourneyTooltip />} />
          <Legend />
          <ReferenceLine yAxisId="f1" y={0.625} stroke="#16a34a" strokeDasharray="3 3" strokeOpacity={0.5} />
          <Line yAxisId="f1" type="monotone" dataKey="f1" stroke="#16a34a" strokeWidth={2.5} dot={{ r: 5, fill: '#16a34a' }} name="Best F1 ↑" />
          <Line yAxisId="brier" type="monotone" dataKey="brier" stroke="#dc2626" strokeWidth={2} dot={{ r: 4, fill: '#dc2626' }} name="Brier ↓" />
        </LineChart>
      </ResponsiveContainer>
    </section>
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
    fetchDataJson<Phase3Data>('phase3_metrics.json')
      .then(d => setData(d))
      .catch(e => setError(e.message));
  }, []);

  if (error) return <div className="lab-container"><div className="lab-card"><p>Phase 3 data not available: {error}</p></div></div>;
  if (!data) return <div className="loading">Loading Phase 3 data...</div>;

  const { experiments, pairwise, feature_importances, practical_summary, walk_forward, target_sensitivity, episode_eval, regime_models } = data;

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

      <ResearchTrackNotice track="risk-model" />

      <section className="lab-card" style={{ borderLeft: '4px solid #8b5cf6', background: '#faf5ff' }}>
        <div className="ab-header">
          <h2 style={{ color: '#6b21a8' }}>研究结论：为什么换算法没用</h2>
          <span className="ab-badge" style={{ background: '#f3e8ff', color: '#7c3aed', border: '1px solid #c4b5fd' }}>INSIGHT</span>
        </div>
        <div style={{ fontSize: 13, lineHeight: 1.9, color: '#374151' }}>
          <p style={{ margin: '0 0 10px', fontWeight: 600 }}>
            核心矛盾：每次崩盘/回撤的触发源不同（关税、疫情、杠杆爆仓），且同一宏观事件在不同市场环境下的影响也不同。
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, margin: '12px 0' }}>
            <div style={{ padding: '10px 12px', background: '#fff', borderRadius: 8, border: '1px solid #e9d5ff' }}>
              <div style={{ fontWeight: 700, color: '#dc2626', fontSize: 12 }}>换算法 → 几乎无效</div>
              <div style={{ fontSize: 12, marginTop: 4, color: '#6b7280' }}>
                LR F1=0.306 vs GBDT F1=0.312<br/>
                模型复杂度不是瓶颈，因为映射 f(features)→crash 本身不稳定
              </div>
            </div>
            <div style={{ padding: '10px 12px', background: '#fff', borderRadius: 8, border: '1px solid #bbf7d0' }}>
              <div style={{ fontWeight: 700, color: '#16a34a', fontSize: 12 }}>加数据 → 有效但有天花板</div>
              <div style={{ fontSize: 12, marginTop: 4, color: '#6b7280' }}>
                1000天 → 9000天，WF F1: 0.19→0.31 (+63%)<br/>
                见过的 pattern 越多，匹配概率越高，但新型崩盘仍会失败
              </div>
            </div>
          </div>
          <p style={{ margin: '10px 0 6px', fontWeight: 600 }}>验证：Fold 4 为何独占 F1=0.71？</p>
          <p style={{ margin: 0, fontSize: 12, color: '#6b7280' }}>
            因为训练集包含了 2025.4 关税冲击（41天正样本），而测试期的 2026.2 回撤特征指纹几乎相同
            （VIX~22、跌破50MA、利差走阔）。Fold 1-3 的训练集没见过类似 pattern → F1≈0。
          </p>
          <div style={{ marginTop: 12, padding: '10px 14px', background: '#f0fdf4', borderRadius: 8, border: '1px solid #86efac' }}>
            <div style={{ fontWeight: 700, fontSize: 12, color: '#166534' }}>下一步方向</div>
            <div style={{ fontSize: 12, color: '#374151', marginTop: 4, lineHeight: 1.8 }}>
              ① 换目标：不预测「是否崩盘」，改为度量「市场脆弱性」— 绕开触发源不可预测的问题<br/>
              ② 异常检测：检测微观因子是否偏离正常状态，不依赖崩盘类型<br/>
              ③ 增量信号：期权 skew、资金流等结构性数据，可能在 price 之前反应
            </div>
          </div>
        </div>
      </section>

      <OptimizationJourney />

      {walk_forward && <WalkForwardSection wf={walk_forward} />}

      {target_sensitivity && <TargetSensitivitySection data={target_sensitivity} />}

      {episode_eval && <EpisodeEvalSection data={episode_eval} />}

      {regime_models && <RegimeModelsSection data={regime_models} />}

      {/* ============ SECTION 1: 核心问题 ============ */}
      <section className="lab-card">
        <div className="best-config-card" style={{ marginTop: 0, borderLeftColor: '#3a82d6' }}>
          <div className="best-config-header">
            <span className="best-config-badge" style={{ background: '#eff6ff', color: '#1e40af' }}>CORE QUESTION</span>
          </div>
          <div style={{ fontSize: 14, lineHeight: 1.8 }}>
            <p style={{ margin: 0 }}>Ch.1 的 LR Slim（10特征，线性模型）已经是最佳 baseline。本章尝试三个方向突破：</p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, marginTop: 12 }}>
              <div style={{ padding: '10px 12px', background: '#f8fafc', borderRadius: 8, borderLeft: '3px solid #d6457a' }}>
                <div style={{ fontWeight: 700, color: '#d6457a', fontSize: 13 }}>A. 换模型</div>
                <div style={{ fontSize: 12, color: '#6b5f63', marginTop: 4 }}>线性 → 非线性（GBDT/RF），能否学到「VIX高 且 利差走阔」这类交互？</div>
              </div>
              <div style={{ padding: '10px 12px', background: '#f8fafc', borderRadius: 8, borderLeft: '3px solid #3a82d6' }}>
                <div style={{ fontWeight: 700, color: '#3a82d6', fontSize: 13 }}>B. 加特征</div>
                <div style={{ fontSize: 12, color: '#6b5f63', marginTop: 4 }}>引入宏观 regime（加息/降息/曲线倒挂）和事件日历（FOMC/CPI），是否提供增量信息？</div>
              </div>
              <div style={{ padding: '10px 12px', background: '#f8fafc', borderRadius: 8, borderLeft: '3px solid #16a34a' }}>
                <div style={{ fontWeight: 700, color: '#16a34a', fontSize: 13 }}>C. 加数据</div>
                <div style={{ fontSize: 12, color: '#6b5f63', marginTop: 4 }}>从 ~1000 天扩展到 ~5300 天（2005+），覆盖多个经济周期，模型是否更稳健？</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============ SECTION 2: 模型工具箱 ============ */}
      <section className="lab-card">
        <div className="ab-header">
          <h2>模型工具箱</h2>
          <span className="ab-badge" style={{ background: '#f0fdf4', color: '#166534', border: '1px solid #bbf7d0' }}>3 TYPES</span>
        </div>
        <div style={{ fontSize: 13, lineHeight: 1.7 }}>
          <div className="lab-table-wrap">
            <table className="lab-table">
              <thead>
                <tr><th>模型</th><th>一句话</th><th>核心优势</th><th>核心风险</th></tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ fontWeight: 700, color: '#d6457a' }}>LR</td>
                  <td style={{ fontSize: 12 }}>Logistic Regression — 每个特征乘一个权重再加总</td>
                  <td style={{ fontSize: 12 }}>概率天然校准 · 小样本稳定 · 可解释</td>
                  <td style={{ fontSize: 12 }}>不能学交互（如「A高且B高=危险」）</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 700, color: '#3a82d6' }}>GBDT</td>
                  <td style={{ fontSize: 12 }}>梯度提升决策树 — 逐棵树纠正前一棵的错误</td>
                  <td style={{ fontSize: 12 }}>自动学交互 · 抗共线性 · 结构化数据通常最强</td>
                  <td style={{ fontSize: 12 }}>小样本易过拟合 · 概率需校准</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 700, color: '#16a34a' }}>RF</td>
                  <td style={{ fontSize: 12 }}>随机森林 — 200 棵树并行投票取平均</td>
                  <td style={{ fontSize: 12 }}>方差低 · 不太过拟合 · 天然做特征选择</td>
                  <td style={{ fontSize: 12 }}>偏差高 · 概率不太准 · 大数据上才稳定</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: 11, color: '#8a7882', marginTop: 8 }}>
            所有模型统一 Unbalanced 训练 + Embargo 20d 隔离（详见 Ch.2.1）。Slim = 10 个量价特征，Full = 23 个全量特征。
          </p>
        </div>
      </section>

      {/* ============ SECTION 3: 实验地图 ============ */}
      <section className="lab-card">
        <div className="ab-header">
          <h2>实验地图</h2>
          <span className="ab-badge" style={{ background: '#eff6ff', color: '#1e40af', border: '1px solid #bfdbfe' }}>5 STEPS</span>
        </div>

        {/* Step 1 */}
        <div style={{ marginBottom: 16, padding: 12, borderRadius: 8, background: 'rgba(34,197,94,0.04)', border: '1px solid rgba(34,197,94,0.2)' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
            <span style={{ fontWeight: 700, color: '#16a34a', fontSize: 14 }}>Step 1 · 换模型</span>
            <span style={{ fontSize: 11, color: '#16a34a', background: '#f0fdf4', padding: '2px 6px', borderRadius: 4 }}>BEST</span>
          </div>
          <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.7 }}>
            <strong>假设：</strong>非线性模型（GBDT/RF）能捕捉特征交互，超越线性 LR。<br/>
            <strong>控制变量：</strong>相同特征（Slim 10 或 Full 23），仅换模型类型。<br/>
            <strong>核心差异：</strong>LR 是「加权求和」，GBDT 是「逐步纠错的决策树」，RF 是「多棵树投票」。
          </div>
          <div style={{ marginTop: 8, padding: '6px 10px', background: '#f0fdf4', borderRadius: 6, fontSize: 12 }}>
            <strong style={{ color: '#166534' }}>结论：LR Slim 仍为最佳（F1=0.647）。</strong>
            {' '}GBDT Slim F1=0.450, RF Slim F1=0.268 — 小样本下过拟合，简单模型更稳健。
          </div>
        </div>

        {/* Step 2 — Events */}
        <div style={{ marginBottom: 16, padding: 12, borderRadius: 8, background: 'rgba(234,88,12,0.04)', border: '1px solid rgba(234,88,12,0.2)' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
            <span style={{ fontWeight: 700, color: '#ea580c', fontSize: 14 }}>Step 2 · 事件日历特征</span>
            <span style={{ fontSize: 11, color: '#166534', background: '#f0fdf4', padding: '2px 6px', borderRadius: 4 }}>NEW BEST</span>
          </div>
          <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.7 }}>
            <strong>假设：</strong>三大宏观事件发布前后市场波动加剧，接近事件 = 更高风险。<br/>
            <strong>控制变量：</strong>在 Slim 10 特征基础上追加 9 个事件日历特征（共 19 特征），模型不变（LR / GBDT）。
          </div>
          <div style={{ marginTop: 10, padding: '8px 10px', background: '#fff', borderRadius: 6, border: '1px solid #e5e7eb', fontSize: 12, lineHeight: 1.8 }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>三大事件</div>
            <div><span style={{ fontWeight: 600, color: '#ea580c' }}>FOMC</span>（Federal Open Market Committee） — 美联储公开市场委员会，每年 8 次会议决定利率。利率决议直接影响全市场资产定价，是美股最大的单日波动源。</div>
            <div><span style={{ fontWeight: 600, color: '#ea580c' }}>CPI</span>（Consumer Price Index） — 消费者物价指数，每月公布。通胀数据超/低预期直接改变市场对加息/降息的预期。</div>
            <div><span style={{ fontWeight: 600, color: '#ea580c' }}>NFP</span>（Non-Farm Payrolls） — 非农就业报告，每月第一个周五公布。就业市场强弱影响美联储决策路径。</div>
          </div>
          <div style={{ marginTop: 8, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={{ padding: '8px 10px', background: '#fff', borderRadius: 6, border: '1px solid #e5e7eb', fontSize: 12 }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>新增 9 个特征</div>
              <div style={{ color: '#6b5f63', lineHeight: 1.6 }}>
                fomc_days_to · fomc_days_since · fomc_within_3d · fomc_within_7d ·
                cpi_days_to · cpi_days_since · cpi_within_3d · cpi_within_7d · nfp_within_3d
              </div>
            </div>
            <div style={{ padding: '8px 10px', background: '#fff', borderRadius: 6, border: '1px solid #e5e7eb', fontSize: 12 }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>为什么 FOMC 最重要？</div>
              <div style={{ color: '#6b5f63', lineHeight: 1.6 }}>
                特征重要性排名中，fomc_days_since 和 fomc_days_to 排前 3。
                FOMC 会后市场需要消化利率路径变化，"距上次 FOMC 多久" 反映了这种消化周期。
              </div>
            </div>
          </div>
          <div style={{ marginTop: 8, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={{ padding: '8px 10px', background: '#fff', borderRadius: 6, border: '1px solid #e5e7eb', fontSize: 12 }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>优化尝试 A: 精简到 4 特征</div>
              <div style={{ color: '#6b5f63' }}>只保留 fomc_days_since, fomc_days_to, cpi_within_3d, nfp_within_3d</div>
              <div style={{ color: '#dc2626', fontWeight: 600, marginTop: 4 }}>F1=0.632 — 精简反而下降，"冗余"特征有用</div>
            </div>
            <div style={{ padding: '8px 10px', background: '#fff', borderRadius: 6, border: '1px solid #e5e7eb', fontSize: 12 }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>优化尝试 B: 衰减函数变换</div>
              <div style={{ color: '#6b5f63' }}>{'exp(-days/5)'} 替换线性天数，捕捉"越近越危险"</div>
              <div style={{ color: '#dc2626', fontWeight: 600, marginTop: 4 }}>F1=0.552 — 非线性变换无效，线性距离已够用</div>
            </div>
          </div>
          <div style={{ marginTop: 8, padding: '6px 10px', background: '#f0fdf4', borderRadius: 6, fontSize: 12 }}>
            <strong style={{ color: '#166534' }}>结论：全量 9 特征 LR Slim+Events F1=0.690 仍为最佳。</strong>
            {' '}FOMC 时间邻近性是核心增量来源。精简和非线性变换均无法超越，说明模型已充分利用了线性距离信息。
          </div>
        </div>

        {/* Step 3 — Regime as Context */}
        <div style={{ marginBottom: 16, padding: 12, borderRadius: 8, background: 'rgba(139,92,246,0.04)', border: '1px solid rgba(139,92,246,0.2)' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
            <span style={{ fontWeight: 700, color: '#8b5cf6', fontSize: 14 }}>Step 3 · Regime 作为上下文</span>
            <span style={{ fontSize: 11, color: '#8b5cf6', background: '#faf5ff', padding: '2px 6px', borderRadius: 4 }}>NEW</span>
          </div>
          <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.7 }}>
            <strong>背景：</strong>之前直接加 regime 特征（curve_inverted, fed_hiking 等）无增量，且相关性分析发现它们与现有特征高度冗余（VIF {'>'} 10）。<br/>
            <strong>新思路：</strong>不再作为独立特征，而是将 regime 作为「上下文/条件」来使用。
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginTop: 8 }}>
            <div style={{ padding: '8px 10px', background: '#fff', borderRadius: 6, border: '1px solid #e5e7eb', fontSize: 12 }}>
              <div style={{ fontWeight: 600, marginBottom: 4, color: '#8b5cf6' }}>Scheme A: 条件建模</div>
              分 tight/normal 两个 regime 训练独立 LR，预测时根据当前 regime 切换模型。
            </div>
            <div style={{ padding: '8px 10px', background: '#fff', borderRadius: 6, border: '1px solid #e5e7eb', fontSize: 12 }}>
              <div style={{ fontWeight: 600, marginBottom: 4, color: '#8b5cf6' }}>Scheme B: 交互项</div>
              加 tight×vix_level、tight×credit_spread 等交互项，让模型学到条件效应。
            </div>
            <div style={{ padding: '8px 10px', background: '#fff', borderRadius: 6, border: '1px solid #e5e7eb', fontSize: 12 }}>
              <div style={{ fontWeight: 600, marginBottom: 4, color: '#8b5cf6' }}>Scheme C: 后处理校准</div>
              训练全局模型，根据各 regime 的实际崩盘率用 OOF 方法调整输出概率。
            </div>
          </div>
          <div style={{ marginTop: 8, padding: '6px 10px', background: '#faf5ff', borderRadius: 6, fontSize: 12 }}>
            <strong style={{ color: '#6d28d9' }}>结论：</strong>
            条件建模 F1=0.667 · 交互项 LR F1=0.667 (Brier=0.081 最佳!) · 后处理校准 F1=0.667。
            三种方案 F1 一致但 Brier 差异大 — 交互项方案的概率校准最好。
          </div>
        </div>

        {/* Step 4 */}
        <div style={{ marginBottom: 4, padding: 12, borderRadius: 8, background: 'rgba(220,38,38,0.04)', border: '1px solid rgba(220,38,38,0.2)' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
            <span style={{ fontWeight: 700, color: '#dc2626', fontSize: 14 }}>Step 4 · 长期数据重测</span>
          </div>
          <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.7 }}>
            <strong>假设：</strong>更多训练数据（5000+ vs ~950 天）应让模型更稳健。<br/>
            <strong>控制变量：</strong>相同的 Slim 10 特征 + 相同模型，仅扩展数据到 2005+。<br/>
            <strong>核心差异 vs Step 1：</strong>训练集跨越次贷危机(2008)、欧债危机(2011)、COVID(2020)、通胀加息(2022)等完全不同的市场 regime。
          </div>
          <div style={{ marginTop: 8, padding: '6px 10px', background: '#fef2f2', borderRadius: 6, fontSize: 12 }}>
            <strong style={{ color: '#991b1b' }}>结论：</strong>性能退化，根因是「非平稳性」：跨周期的统计规律太弱。待进一步数据质量优化。
          </div>
        </div>

        {/* Step 5 — Combined */}
        <div style={{ marginBottom: 4, padding: 12, borderRadius: 8, background: 'rgba(13,148,136,0.04)', border: '1px solid rgba(13,148,136,0.2)' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
            <span style={{ fontWeight: 700, color: '#0d9488', fontSize: 14 }}>Step 5 · Events + Interact 合并</span>
            <span style={{ fontSize: 11, color: '#0d9488', background: '#f0fdfa', padding: '2px 6px', borderRadius: 4 }}>BEST BRIER</span>
          </div>
          <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.7 }}>
            <strong>假设：</strong>Step 2 的 Events（F1 最佳）和 Step 3 的 Interact（Brier 最佳）捕捉的是不同信号，合并应该同时提升两个指标。<br/>
            <strong>控制变量：</strong>LR Slim + 全量 Events 9 特征 + 3 个 regime 交互项（共 22 特征）。<br/>
            <strong>核心差异：</strong>Events 捕捉「什么时候危险」（时间维度），Interact 捕捉「什么条件下更危险」（状态维度）。
          </div>
          <div style={{ marginTop: 8, padding: '6px 10px', background: '#f0fdfa', borderRadius: 6, fontSize: 12 }}>
            <strong style={{ color: '#0d9488' }}>结论：LR Events+Interact F1=0.688, Brier=0.080（全局最优 Brier！）。</strong>
            {' '}F1 与 Events 持平（0.690 vs 0.688），但 Brier 从 0.098 降至 0.080 — 概率校准显著提升。
            这是目前最平衡的模型：预测能力和概率准确度兼顾。
          </div>
        </div>

        {/* Overall conclusion */}
        <div style={{ marginTop: 16, padding: '12px 14px', background: '#f0fdf4', borderRadius: 8, border: '1px solid #bbf7d0' }}>
          <div style={{ fontWeight: 700, fontSize: 14, color: '#166534', marginBottom: 6 }}>总结</div>
          <div style={{ fontSize: 13, lineHeight: 1.7, color: '#374151' }}>
            单次 split 最佳：<strong> LR Slim+Events</strong> (F1=0.690) · <strong>LR Events+Interact</strong> (Brier=0.080)。
            Walk-forward 均值 F1≈0.19，高 F1 集中在 2025 Q4~2026 Q1 — 跨时段稳定性<strong>未通过</strong>。
            下一步：时间衰减 / target 网格 / regime 细化（见 WF 验证后讨论）。
          </div>
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

      {/* Correlation Analysis */}
      {data.correlation_analysis && (
        <section className="lab-card">
          <div className="ab-header">
            <h2>特征相关性分析</h2>
            <span className="ab-badge" style={{ background: '#fef3c7', color: '#92400e', border: '1px solid #fcd34d' }}>
              {data.correlation_analysis.total_features} FEATURES
            </span>
          </div>
          <p className="lab-card-desc">
            Spearman 相关性 + VIF 分析。高相关特征组已在 Slim 10 特征中去冗余。新增特征前需通过相关性检查。
          </p>

          {/* Redundancy groups */}
          <div style={{ marginBottom: 16 }}>
            <h4 style={{ fontSize: 13, color: '#374151', marginBottom: 8 }}>已识别冗余组</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 8 }}>
              {data.correlation_analysis.redundancy_groups.map(g => (
                <div key={g.group} style={{ padding: '8px 10px', background: '#f8fafc', borderRadius: 6, border: '1px solid #e5e7eb', fontSize: 12 }}>
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>{g.group}</div>
                  <div style={{ color: '#6b5f63' }}>{g.features.join(', ')}</div>
                  <div style={{ color: '#16a34a', fontWeight: 600, marginTop: 4 }}>代表: {g.representative}</div>
                </div>
              ))}
            </div>
          </div>

          {/* High corr pairs */}
          <div style={{ marginBottom: 16 }}>
            <h4 style={{ fontSize: 13, color: '#374151', marginBottom: 8 }}>高相关特征对 (|r| {'>'} 0.5)</h4>
            <div className="lab-table-wrap">
              <table className="lab-table" style={{ fontSize: 12 }}>
                <thead><tr><th>特征 A</th><th>特征 B</th><th>Spearman r</th></tr></thead>
                <tbody>
                  {data.correlation_analysis.high_corr_pairs.slice(0, 15).map((p, i) => (
                    <tr key={i}>
                      <td>{p.feat_a}</td>
                      <td>{p.feat_b}</td>
                      <td style={{ fontWeight: 600, color: Math.abs(p.spearman) > 0.7 ? '#dc2626' : '#b45309' }}>{p.spearman.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* VIF */}
          {data.correlation_analysis.vif.length > 0 && (
            <div>
              <h4 style={{ fontSize: 13, color: '#374151', marginBottom: 8 }}>VIF（Slim 10 特征）</h4>
              <div className="lab-table-wrap">
                <table className="lab-table" style={{ fontSize: 12 }}>
                  <thead><tr><th>特征</th><th>VIF</th><th>状态</th></tr></thead>
                  <tbody>
                    {data.correlation_analysis.vif.map(v => (
                      <tr key={v.feature}>
                        <td>{v.feature}</td>
                        <td className="lab-td-mono" style={{ fontWeight: 600 }}>{v.vif.toFixed(1)}</td>
                        <td style={{ color: v.vif >= 10 ? '#dc2626' : v.vif >= 5 ? '#b45309' : '#16a34a', fontWeight: 600 }}>
                          {v.vif >= 10 ? 'REDUNDANT' : v.vif >= 5 ? 'CAUTION' : 'OK'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>
      )}

      {/* Audit log */}
      <section className="lab-card">
        <div className="ab-header">
          <h2>审计记录</h2>
          <span className="ab-badge" style={{ background: '#fef3c7', color: '#92400e', border: '1px solid #fcd34d' }}>AUDIT LOG</span>
        </div>
        <div style={{ fontSize: 12, lineHeight: 1.7, color: '#6b5f63' }}>
          <div style={{ display: 'grid', gap: 6 }}>
            <div style={{ display: 'flex', gap: 8 }}>
              <span style={{ color: '#16a34a', fontWeight: 600, whiteSpace: 'nowrap' }}>FIXED</span>
              <span>clip(-10,10) 导致 vix_level 成为常数 → 改为 percentile clipping (1st-99th)</span>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <span style={{ color: '#16a34a', fontWeight: 600, whiteSpace: 'nowrap' }}>FIXED</span>
              <span>Scheme B 交互项特征名称不匹配 (vix_z → vix_level) → 已修正</span>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <span style={{ color: '#16a34a', fontWeight: 600, whiteSpace: 'nowrap' }}>FIXED</span>
              <span>Scheme A 条件建模 fallback 路径 probs_cond_all 未填充 → 已修正</span>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <span style={{ color: '#16a34a', fontWeight: 600, whiteSpace: 'nowrap' }}>FIXED</span>
              <span>Scheme C 校准系数使用 in-sample → 改为 OOF hold-out 方法</span>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <span style={{ color: '#16a34a', fontWeight: 600, whiteSpace: 'nowrap' }}>FIXED</span>
              <span>Step 4 baseline 的 sp500_timeline 误用短期数据 → 已改为长期</span>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <span style={{ color: '#16a34a', fontWeight: 600, whiteSpace: 'nowrap' }}>REMOVED</span>
              <span>Regime 直接特征实验（Regime2/Regime9）已删除 — 相关性分析证明信息冗余</span>
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
