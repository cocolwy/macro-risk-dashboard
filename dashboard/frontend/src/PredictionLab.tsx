import { useState, useEffect, useMemo } from 'react';
import {
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, ReferenceLine,
  CartesianGrid,
} from 'recharts';
import { downsample, mergeExperimentTimeline, CHART_TOOLTIP_STYLE } from './utils/chart';
import { StackedProbSPChart } from './components/StackedProbSPChart';

interface ModelInfo {
  name: string;
  target: string;
  features_count: number;
  training_samples: number;
  test_samples: number;
  positive_rate: number;
  train_period: string;
  test_period: string;
  roc_auc: number;
  last_updated: string;
}

interface ThresholdRow {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  alert_days: number;
  total_days: number;
  alert_pct: number;
}

interface FeatureWeight {
  feature: string;
  weight: number;
}

interface EventBacktest {
  name: string;
  event_date: string;
  drop_pct: number;
  first_alert_date: string | null;
  lead_days: number | null;
  max_probability: number;
  max_prob_date: string;
}

interface ProbPoint {
  date: string;
  probability: number;
}

interface SP500Point {
  date: string;
  sp500: number;
}

interface ExperimentData {
  name: string;
  auc: number;
  current_probability: number;
  current_signal: string;
  threshold_analysis: ThresholdRow[];
  events_backtest: EventBacktest[];
  probability_timeline: ProbPoint[];
}

interface WeightComparison {
  feature: string;
  ml_weight: number;
  human_weight: number;
  agree: string;
}

interface ModelMetrics {
  model_info: ModelInfo;
  current_prediction: { date: string; probability: number; signal: string };
  pr_curve: { threshold: number; precision: number; recall: number }[];
  roc_curve: { fpr: number; tpr: number }[];
  feature_importance: FeatureWeight[];
  threshold_analysis: ThresholdRow[];
  events_backtest: EventBacktest[];
  probability_timeline: ProbPoint[];
  sp500_timeline: SP500Point[];
  actual_drops: { date: string; max_drawdown: number }[];
  experiments?: ExperimentData[];
  weight_comparison?: WeightComparison[];
}

const DATA_BASE_URL = import.meta.env.PROD ? './data' : '/data';

const FEATURE_LABELS: Record<string, string> = {
  'credit_spread_10d_chg': '信用利差 10日变化',
  'breadth_10d_chg': '市场宽度 10日变化',
  'absorption_ratio_5d_chg': '耦合度 5日变化',
  'turbulence_5d_chg': '湍流度 5日变化',
  'vix_20d_chg': 'VIX 20日变化',
  'credit_spread_5d_chg': '信用利差 5日变化',
  'vix_5d_chg': 'VIX 5日变化',
  'sp500_vs_50ma': 'S&P vs 50MA',
  'absorption_ratio_10d_chg': '耦合度 10日变化',
  'credit_spread_level': '信用利差水平',
  'sp500_20d_ret': 'S&P 20日回报',
  'sp500_50d_ret': 'S&P 50日回报',
  'vix_volatility': 'VIX波动率',
  'vix_level': 'VIX水平',
  'vix_vs_20d_avg': 'VIX vs 20日均值',
  'term_spread_level': '期限利差水平',
  'term_spread_5d_chg': '期限利差 5日变化',
  'term_spread_20d_chg': '期限利差 20日变化',
  'breadth_level': '市场宽度水平',
  'turbulence_10d_chg': '湍流度 10日变化',
  'turbulence_20d_chg': '湍流度 20日变化',
  'absorption_ratio_20d_chg': '耦合度 20日变化',
  'vix_10d_chg': 'VIX 10日变化',
};

const EXP_COLORS = ['#d6457a', '#3a82d6', '#16a34a', '#dc2626', '#8b5cf6', '#06b6d4', '#ea580c', '#0d9488', '#a855f7', '#f97316', '#64748b'];

function signalColor(signal: string) {
  return signal === 'elevated' ? '#dc2626' : signal === 'watch' ? '#b45309' : '#16a34a';
}

type DateRange = '3m' | '6m' | '1y' | '2y' | 'all';
const RANGE_OPTIONS: { key: DateRange; label: string }[] = [
  { key: '3m', label: '3M' },
  { key: '6m', label: '6M' },
  { key: '1y', label: '1Y' },
  { key: '2y', label: '2Y' },
  { key: 'all', label: 'ALL' },
];

function filterByRange<T extends { date: string }>(data: T[], range: DateRange): T[] {
  if (range === 'all' || data.length === 0) return data;
  const lastDate = new Date(data[data.length - 1].date);
  const months = range === '3m' ? 3 : range === '6m' ? 6 : range === '1y' ? 12 : 24;
  const cutoff = new Date(lastDate);
  cutoff.setMonth(cutoff.getMonth() - months);
  const cutoffStr = cutoff.toISOString().slice(0, 10);
  return data.filter(d => d.date >= cutoffStr);
}

function ExperimentGroup({ title, desc, experiments, sp500Timeline, colors }: {
  title: string; desc: string; experiments: ExperimentData[]; sp500Timeline: SP500Point[]; colors: string[];
}) {
  const [range, setRange] = useState<DateRange>('1y');

  const chartData = useMemo(() => {
    const full = mergeExperimentTimeline(experiments, sp500Timeline);
    const filtered = filterByRange(full, range);
    return downsample(filtered);
  }, [experiments, sp500Timeline, range]);

  const allEvents = useMemo(() => {
    const events = new Map<string, EventBacktest>();
    experiments.forEach(exp => exp.events_backtest.forEach(e => {
      if (!events.has(e.name)) events.set(e.name, e);
    }));
    return events;
  }, [experiments]);

  const dropMarkers = useMemo(() =>
    [...allEvents.values()].map(e => ({
      date: e.event_date,
      label: e.name,
      dropPct: e.drop_pct,
    })),
    [allEvents],
  );

  return (
    <div className="ab-group">
      <h3 className="ab-group-title">{title}</h3>
      <p className="lab-card-desc">{desc}</p>

      <div className="ab-summary-row">
        {experiments.map((exp, i) => (
          <div key={exp.name} className="ab-summary-card" style={{ borderTopColor: colors[i] }}>
            <div className="ab-model-name" style={{ color: colors[i] }}>{exp.name}</div>
            <div className="ab-metric-row">
              <span className="ab-metric-label">AUC</span>
              <span className="ab-metric-value">{exp.auc}</span>
            </div>
            <div className="ab-metric-row">
              <span className="ab-metric-label">当前概率</span>
              <span className="ab-metric-value" style={{ color: signalColor(exp.current_signal) }}>
                {(exp.current_probability * 100).toFixed(1)}%
              </span>
            </div>
            <div className="ab-metric-row">
              <span className="ab-metric-label">信号</span>
              <span className="ab-signal-badge" style={{ background: signalColor(exp.current_signal) + '22', color: signalColor(exp.current_signal) }}>
                {exp.current_signal === 'elevated' ? 'ELEVATED' : exp.current_signal === 'watch' ? 'WATCH' : 'LOW'}
              </span>
            </div>
          </div>
        ))}
      </div>

      <div className="ab-chart-section">
        <div className="range-picker">
          {RANGE_OPTIONS.map(opt => (
            <button key={opt.key}
              className={`range-btn ${range === opt.key ? 'active' : ''}`}
              onClick={() => setRange(opt.key)}>
              {opt.label}
            </button>
          ))}
        </div>
        <StackedProbSPChart
          data={chartData}
          series={experiments.map((exp, i) => ({
            dataKey: `prob_${i}`,
            name: exp.name,
            color: colors[i],
          }))}
          dropEvents={dropMarkers}
          showLegend
          probHeight={260}
          spHeight={130}
        />
      </div>

      <div className="ab-tables-row">
        <div className="ab-table-half">
          <h4>阈值对比</h4>
          <div className="lab-table-wrap">
            <table className="lab-table lab-table-compact">
              <thead>
                <tr>
                  <th>阈值</th>
                  {experiments.map((exp, i) => (
                    <th key={i} colSpan={2} style={{ color: colors[i] }}>
                      {exp.name.length > 16 ? exp.name.replace('Logistic Regression', 'LR').replace('Extended', 'Ext') : exp.name}
                    </th>
                  ))}
                </tr>
                <tr><th></th>{experiments.map((_, i) => (<><th key={`p${i}`}>P</th><th key={`r${i}`}>R</th></>))}</tr>
              </thead>
              <tbody>
                {[0.3, 0.5, 0.7].map(thresh => (
                  <tr key={thresh} className={thresh === 0.5 ? 'lab-row-highlight' : ''}>
                    <td className="lab-td-mono">{(thresh * 100).toFixed(0)}%</td>
                    {experiments.map((exp, i) => {
                      const row = exp.threshold_analysis.find(r => r.threshold === thresh);
                      if (!row) return <><td key={`p${i}`}>-</td><td key={`r${i}`}>-</td></>;
                      return (<>
                        <td key={`p${i}`} style={{ color: row.precision > 0.3 ? '#16a34a' : row.precision > 0.15 ? '#b45309' : '#dc2626' }}>
                          {(row.precision * 100).toFixed(0)}%
                        </td>
                        <td key={`r${i}`} style={{ color: row.recall > 0.7 ? '#16a34a' : row.recall > 0.4 ? '#b45309' : '#dc2626' }}>
                          {(row.recall * 100).toFixed(0)}%
                        </td>
                      </>);
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="ab-table-half">
          <h4>历史事件</h4>
          <div className="lab-table-wrap">
            <table className="lab-table lab-table-compact">
              <thead>
                <tr>
                  <th>事件</th>
                  {experiments.map((exp, i) => (
                    <th key={i} colSpan={2} style={{ color: colors[i] }}>
                      {exp.name.length > 16 ? exp.name.replace('Logistic Regression', 'LR').replace('Extended', 'Ext') : exp.name}
                    </th>
                  ))}
                </tr>
                <tr><th></th>{experiments.map((_, i) => (<><th key={`a${i}`}>提前</th><th key={`m${i}`}>峰值</th></>))}</tr>
              </thead>
              <tbody>
                {[...allEvents.values()].map(evt => (
                  <tr key={evt.name}>
                    <td style={{ fontSize: 11 }}>{evt.name}</td>
                    {experiments.map((exp, i) => {
                      const e = exp.events_backtest.find(b => b.name === evt.name);
                      if (!e) return <><td key={`a${i}`} style={{ color: '#aaa' }}>-</td><td key={`m${i}`} style={{ color: '#aaa' }}>-</td></>;
                      return (<>
                        <td key={`a${i}`} style={{ color: e.lead_days ? '#16a34a' : '#dc2626' }}>
                          {e.lead_days ? `${e.lead_days}d` : 'miss'}
                        </td>
                        <td key={`m${i}`}>{(e.max_probability * 100).toFixed(0)}%</td>
                      </>);
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

interface PairwiseTest {
  label: string;
  variable: string;
  baseline: ExperimentData;
  challenger: ExperimentData;
  baseColor: string;
  challColor: string;
  methodNote?: string;
}

function buildPairwiseTests(experiments: ExperimentData[]): PairwiseTest[] {
  const find = (substr: string) => experiments.find(e => e.name.includes(substr));
  const findExact = (substr: string) => experiments.find(e => e.name === substr || e.name.startsWith(substr));
  const mlBase = find('Logistic Regression') ?? find('ML (');
  const human = find('Human Logic');
  const slim = findExact('ML Slim');
  const mlExt = find('ML Extended');
  const humanExt = find('Human Extended');
  const d1Short = experiments.find(e => e.name.includes('D1') && !e.name.includes('Ext'));
  const minShort = experiments.find(e => e.name.includes('MIN') && !e.name.includes('Ext'));
  const andShort = experiments.find(e => e.name.includes('AND') && !e.name.includes('Ext'));
  const d1Ext = experiments.find(e => e.name.includes('D1') && e.name.includes('Ext'));
  const minExt = experiments.find(e => e.name.includes('MIN') && e.name.includes('Ext'));
  const andExt = experiments.find(e => e.name.includes('AND') && e.name.includes('Ext'));

  const pairs: PairwiseTest[] = [];
  if (mlBase && human) pairs.push({
    label: 'Exp 1: 权重来源', variable: '自动学习 vs 人工逻辑',
    baseline: mlBase, challenger: human,
    baseColor: EXP_COLORS[0], challColor: EXP_COLORS[1],
  });
  if (mlBase && slim) pairs.push({
    label: 'Exp 2: 特征去冗余', variable: '23特征 vs 10特征（去共线性）',
    baseline: mlBase, challenger: slim,
    baseColor: EXP_COLORS[0], challColor: EXP_COLORS[2],
  });
  if (slim && d1Short) pairs.push({
    label: 'Exp 3: Embargo隔离', variable: '无隔离 vs 20天Embargo（防时序泄露）',
    baseline: slim, challenger: d1Short,
    baseColor: EXP_COLORS[2], challColor: EXP_COLORS[5],
    methodNote: '从本实验起引入 Embargo：训练集和测试集之间设置 20 天隔离带，防止因 20 日前瞻目标导致的数据泄露。后续实验（Exp 4+）均沿用此方法。',
  });
  if (d1Short && minShort && andShort) pairs.push({
    label: 'Exp 4: 双模型集成', variable: 'MIN(连续概率取较小值) vs AND(logical AND，二元0/1)',
    baseline: minShort, challenger: andShort,
    baseColor: EXP_COLORS[6], challColor: EXP_COLORS[9],
    methodNote: 'MIN = min(D1概率, Human概率)，保留连续概率可做阈值分析。AND = 两边都超过 50% 才输出 1，否则 0（二元信号，无灰度区间）。两者都基于 Embargo 纠偏后的 D1 + Human Logic。',
  });
  if (mlBase && mlExt) pairs.push({
    label: 'Exp 5a: ML 长期 vs 短期', variable: '短期 (~4年) vs 长期 (2005+, 20年) — ML模型',
    baseline: mlBase, challenger: mlExt,
    baseColor: EXP_COLORS[0], challColor: EXP_COLORS[3],
    methodNote: '测试更多历史数据是否提升 ML 模型效果。长期版使用 Embargo 纠偏。',
  });
  if (human && humanExt) pairs.push({
    label: 'Exp 5b: Human 长期 vs 短期', variable: '短期 (~4年) vs 长期 (2005+, 20年) — Human模型',
    baseline: human, challenger: humanExt,
    baseColor: EXP_COLORS[1], challColor: EXP_COLORS[4],
    methodNote: '测试更多历史数据是否提升 Human Logic 模型效果。长期版使用 Embargo 纠偏。',
  });
  if (d1Ext && minExt && andExt) pairs.push({
    label: 'Exp 6: 长期双模型集成', variable: '长期 MIN vs 长期 AND',
    baseline: minExt, challenger: andExt,
    baseColor: EXP_COLORS[7], challColor: EXP_COLORS[8],
    methodNote: '长期数据版本，同样使用 Embargo 纠偏。对比 MIN（连续）与 AND（二元）两种集成方式。',
  });
  return pairs;
}

function DeltaCell({ base, value, pct }: { base: number; value: number; pct?: boolean }) {
  const delta = value - base;
  const isUp = delta > 0.001;
  const isDown = delta < -0.001;
  const color = isUp ? '#16a34a' : isDown ? '#dc2626' : '#8a7882';
  const arrow = isUp ? '\u2191' : isDown ? '\u2193' : '';
  const display = pct ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
  const deltaStr = pct ? `${delta > 0 ? '+' : ''}${(delta * 100).toFixed(1)}` : `${delta > 0 ? '+' : ''}${delta.toFixed(3)}`;
  return (
    <td>
      <span style={{ fontWeight: 600 }}>{display}</span>
      {arrow && <span style={{ color, fontSize: 10, marginLeft: 4 }}>{arrow}{deltaStr}</span>}
    </td>
  );
}

function ABComparisonSection({ experiments, sp500Timeline }: { experiments: ExperimentData[]; sp500Timeline: SP500Point[] }) {
  const pairs = buildPairwiseTests(experiments);

  return (
    <section className="lab-card ab-section">
      <div className="ab-header">
        <h2>模型对比实验</h2>
        <span className="ab-badge">EXPERIMENT</span>
        <span className="ab-badge" style={{ background: 'rgba(34, 197, 94, 0.2)', color: '#16a34a' }}>{experiments.length} MODELS</span>
      </div>

      {/* Overall comparison table */}
      <div className="ab-overview-table">
        <table className="lab-table">
          <thead>
            <tr>
              <th>模型</th>
              <th>AUC</th>
              <th>P@50%</th>
              <th>R@50%</th>
              <th>当前信号</th>
              <th>数据量</th>
            </tr>
          </thead>
          <tbody>
            {experiments.map((exp, i) => {
              const row50 = exp.threshold_analysis.find(r => r.threshold === 0.5);
              const isBest = exp.auc === Math.max(...experiments.map(e => e.auc));
              return (
                <tr key={exp.name} style={{ background: isBest ? 'rgba(34, 197, 94, 0.06)' : undefined }}>
                  <td style={{ color: EXP_COLORS[i], fontWeight: 600 }}>
                    {isBest && <span className="ab-best-tag">BEST</span>}
                    {exp.name}
                  </td>
                  <td className="lab-td-mono" style={{ fontWeight: 700 }}>{exp.auc}</td>
                  <td style={{ color: (row50?.precision ?? 0) > 0.2 ? '#16a34a' : '#b45309' }}>
                    {row50 ? `${(row50.precision * 100).toFixed(1)}%` : '-'}
                  </td>
                  <td style={{ color: (row50?.recall ?? 0) > 0.7 ? '#16a34a' : '#b45309' }}>
                    {row50 ? `${(row50.recall * 100).toFixed(1)}%` : '-'}
                  </td>
                  <td>
                    <span className="ab-signal-badge" style={{ background: signalColor(exp.current_signal) + '22', color: signalColor(exp.current_signal) }}>
                      {(exp.current_probability * 100).toFixed(0)}% {exp.current_signal === 'elevated' ? 'HIGH' : exp.current_signal === 'watch' ? 'WATCH' : 'LOW'}
                    </span>
                  </td>
                  <td style={{ fontSize: 11, color: '#8a7882' }}>{exp.probability_timeline.length}天</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pairwise comparisons */}
      {pairs.map((pair, pi) => {
        const b50 = pair.baseline.threshold_analysis.find(r => r.threshold === 0.5);
        const c50 = pair.challenger.threshold_analysis.find(r => r.threshold === 0.5);
        const winner = pair.challenger.auc > pair.baseline.auc ? 'challenger' : 'baseline';

        return (
          <div key={pi} className="ab-pair">
            <div className="ab-pair-header">
              <h3>{pair.label}</h3>
              <span className="ab-pair-variable">测试变量: {pair.variable}</span>
              {pair.methodNote && (
                <div className="ab-method-note">
                  <span className="ab-method-note-icon">&#9432;</span>
                  {pair.methodNote}
                </div>
              )}
            </div>

            <div className="ab-pair-cards">
              <div className={`ab-pair-card ${winner === 'baseline' ? 'winner' : ''}`} style={{ borderTopColor: pair.baseColor }}>
                <div className="ab-pair-role">BASELINE {winner === 'baseline' && <span className="ab-winner-tag">WIN</span>}</div>
                <div className="ab-model-name" style={{ color: pair.baseColor }}>{pair.baseline.name}</div>
                <div className="ab-metric-row"><span className="ab-metric-label">AUC</span><span className="ab-metric-value">{pair.baseline.auc}</span></div>
                <div className="ab-metric-row"><span className="ab-metric-label">P@50%</span><span className="ab-metric-value">{b50 ? `${(b50.precision*100).toFixed(1)}%` : '-'}</span></div>
                <div className="ab-metric-row"><span className="ab-metric-label">R@50%</span><span className="ab-metric-value">{b50 ? `${(b50.recall*100).toFixed(1)}%` : '-'}</span></div>
              </div>
              <div className="ab-pair-vs">VS</div>
              <div className={`ab-pair-card ${winner === 'challenger' ? 'winner' : ''}`} style={{ borderTopColor: pair.challColor }}>
                <div className="ab-pair-role">CHALLENGER {winner === 'challenger' && <span className="ab-winner-tag">WIN</span>}</div>
                <div className="ab-model-name" style={{ color: pair.challColor }}>{pair.challenger.name}</div>
                <div className="ab-metric-row"><span className="ab-metric-label">AUC</span>
                  <DeltaCell base={pair.baseline.auc} value={pair.challenger.auc} />
                </div>
                <div className="ab-metric-row"><span className="ab-metric-label">P@50%</span>
                  {c50 && b50 ? <DeltaCell base={b50.precision} value={c50.precision} pct /> : <td>-</td>}
                </div>
                <div className="ab-metric-row"><span className="ab-metric-label">R@50%</span>
                  {c50 && b50 ? <DeltaCell base={b50.recall} value={c50.recall} pct /> : <td>-</td>}
                </div>
              </div>
            </div>

            <ExperimentGroup
              title="" desc=""
              experiments={[pair.baseline, pair.challenger]} sp500Timeline={sp500Timeline}
              colors={[pair.baseColor, pair.challColor]}
            />
          </div>
        );
      })}
    </section>
  );
}

function WeightComparisonSection({ data }: { data: WeightComparison[] }) {
  const sorted = [...data].sort((a, b) => Math.abs(b.ml_weight) - Math.abs(a.ml_weight));
  const maxW = Math.max(...sorted.map(d => Math.max(Math.abs(d.ml_weight), Math.abs(d.human_weight))));

  return (
    <section className="lab-card">
      <h2>权重对比: ML学习 vs 人工逻辑</h2>
      <p className="lab-card-desc">对比ML自动学到的权重与人工基于经济学逻辑设定的权重，方向一致=绿色</p>
      <div className="weight-grid">
        {sorted.slice(0, 18).map(row => (
          <div key={row.feature} className={`weight-row ${row.agree}`}>
            <span className="weight-label">{FEATURE_LABELS[row.feature] || row.feature}</span>
            <div className="weight-bars">
              <div className="weight-bar-pair">
                <span className="weight-bar-tag" style={{ color: '#d6457a' }}>ML</span>
                <div className="weight-bar-track">
                  <div className="weight-bar-fill" style={{
                    width: `${Math.abs(row.ml_weight) / maxW * 100}%`,
                    background: row.ml_weight > 0 ? '#dc2626' : '#16a34a',
                    marginLeft: row.ml_weight < 0 ? 'auto' : undefined,
                  }} />
                </div>
                <span className="weight-val">{row.ml_weight > 0 ? '+' : ''}{row.ml_weight.toFixed(3)}</span>
              </div>
              <div className="weight-bar-pair">
                <span className="weight-bar-tag" style={{ color: '#3a82d6' }}>HM</span>
                <div className="weight-bar-track">
                  <div className="weight-bar-fill" style={{
                    width: `${Math.abs(row.human_weight) / maxW * 100}%`,
                    background: row.human_weight > 0 ? '#dc2626' : '#16a34a',
                    marginLeft: row.human_weight < 0 ? 'auto' : undefined,
                  }} />
                </div>
                <span className="weight-val">{row.human_weight > 0 ? '+' : ''}{row.human_weight.toFixed(2)}</span>
              </div>
            </div>
            <span className={`weight-agree-badge ${row.agree}`}>
              {row.agree === 'same' ? 'AGREE' : row.agree === 'zero' ? 'N/A' : 'DIFF'}

            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

export function PredictionLab() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mainRange, setMainRange] = useState<DateRange>('1y');

  useEffect(() => {
    fetch(`${DATA_BASE_URL}/model_metrics.json`)
      .then(r => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then(setMetrics)
      .catch(e => setError(e.message));
  }, []);

  const probWithSP = useMemo(() => {
    if (!metrics) return [];
    const spMap = new Map(metrics.sp500_timeline.map(s => [s.date, s.sp500]));
    return metrics.probability_timeline.map(p => ({
      ...p,
      sp500: spMap.get(p.date),
    }));
  }, [metrics]);

  const mainChartData = useMemo(
    () => downsample(filterByRange(probWithSP, mainRange)),
    [probWithSP, mainRange],
  );

  const mainDropMarkers = useMemo(() =>
    (metrics?.events_backtest ?? []).map(e => ({
      date: e.event_date,
      label: e.name,
      dropPct: e.drop_pct,
    })),
    [metrics],
  );

  if (error) return <div className="lab-error">Failed to load model data: {error}</div>;
  if (!metrics) return <div className="lab-loading">Loading Prediction Lab...</div>;

  const { model_info, current_prediction, feature_importance, events_backtest } = metrics;

  const sigColor = signalColor(current_prediction.signal);

  return (
    <div className="lab-container">
      <header className="lab-header">
        <div>
          <h1>Prediction Lab</h1>
          <p className="lab-subtitle">研发中 - 预测模型监督与优化</p>
        </div>
        <div className="lab-model-badge">
          <span className="lab-badge-version">{model_info.name}</span>
          <span className="lab-badge-auc">AUC: {model_info.roc_auc}</span>
        </div>
      </header>

      {/* ===== AB TEST COMPARISON ===== */}
      {metrics.experiments && metrics.experiments.length > 1 && (
        <ABComparisonSection experiments={metrics.experiments} sp500Timeline={metrics.sp500_timeline} />
      )}

      {/* ===== INSIGHT: WHY LONG-TERM DATA PERFORMS WORSE ===== */}
      {metrics.experiments && metrics.experiments.some(e => e.name.includes('Ext')) && (
        <section className="lab-card insight-card">
          <div className="ab-header">
            <h2>为什么更多数据反而更差？</h2>
            <span className="ab-badge" style={{ background: '#fef3c7', color: '#92400e', border: '1px solid #fcd34d' }}>INSIGHT</span>
          </div>
          <p className="lab-card-desc" style={{ marginBottom: 16 }}>
            D1 Slim+Embargo 在短期数据上 AUC=0.86，但在长期数据（2005+，20年）上仅 0.60。核心原因是金融时序的<strong>非平稳性</strong>。
          </p>
          <div className="insight-reasons">
            <div className="insight-reason">
              <div className="insight-reason-num">1</div>
              <div className="insight-reason-body">
                <h4>特征质量断层</h4>
                <p>
                  <code>turbulence</code> 和 <code>absorption_ratio</code> 依赖 sector ETF（XLC 2018年才上市），训练集中 <strong>91% 为零值</strong>，
                  模型无法学习这两个关键特征，但测试时它们全部活跃。
                </p>
              </div>
            </div>
            <div className="insight-reason">
              <div className="insight-reason-num">2</div>
              <div className="insight-reason-body">
                <h4>特征分布漂移</h4>
                <p>
                  <code>term_spread</code> 训练期均值偏移 <strong>1.14 个标准差</strong>（2005-2020 正常利差 vs 2020-2026 倒挂），
                  <code>turbulence</code> 偏移 <strong>0.96 个标准差</strong>。模型在训练中没见过的分布上做预测。
                </p>
              </div>
            </div>
            <div className="insight-reason">
              <div className="insight-reason-num">3</div>
              <div className="insight-reason-body">
                <h4>跨体制噪声</h4>
                <p>
                  训练集跨 4 个市场体制（信贷泡沫、金融危机、零利率QE、加息周期），指标与崩盘的关系在不同体制下截然不同。
                  模型学到的是"平均"规律，不适用于任何特定体制。短期模型仅学近 2 年的当前体制，模式更一致。
                </p>
              </div>
            </div>
          </div>
          <div className="insight-conclusion">
            <strong>结论：</strong>数据越多≠越好。金融时序的最佳策略是<strong>滑动训练窗口</strong>（用最近 5-7 年），而非堆砌全部历史。
          </div>
        </section>
      )}

      {metrics.weight_comparison && (
        <WeightComparisonSection data={metrics.weight_comparison} />
      )}

      {/* ===== MODEL ARCHITECTURE EXPLAINER ===== */}
      <section className="lab-card">
        <div className="ab-header">
          <h2>模型原理</h2>
          <span className="ab-badge" style={{ background: '#f0fdf4', color: '#166534', border: '1px solid #bbf7d0' }}>EXPLAINER</span>
        </div>

        <div className="pipeline-section">
          <h3 className="pipeline-title">ML 模型 (Logistic Regression)</h3>
          <p className="lab-card-desc">自动从数据中学习每个特征的权重</p>
          <div className="pipeline-flow">
            <div className="pipeline-step">
              <div className="pipeline-step-num">1</div>
              <div className="pipeline-step-label">原始指标</div>
              <div className="pipeline-step-detail">VIX, 利差, 宽度 等 10-23 个指标的当日数值</div>
            </div>
            <div className="pipeline-arrow">→</div>
            <div className="pipeline-step">
              <div className="pipeline-step-num">2</div>
              <div className="pipeline-step-label">StandardScaler</div>
              <div className="pipeline-step-detail">z = (x − μ) / σ<br/>将每个指标标准化为"偏离训练期均值几个标准差"</div>
            </div>
            <div className="pipeline-arrow">→</div>
            <div className="pipeline-step highlight-ml">
              <div className="pipeline-step-num">3</div>
              <div className="pipeline-step-label">自动学习权重</div>
              <div className="pipeline-step-detail">LR 从训练数据自动学出每个特征的系数 → 加权求和</div>
            </div>
            <div className="pipeline-arrow">→</div>
            <div className="pipeline-step">
              <div className="pipeline-step-num">4</div>
              <div className="pipeline-step-label">Sigmoid</div>
              <div className="pipeline-step-detail">P = 1/(1+e⁻ˢ)<br/>得分转为 0~100% 概率</div>
            </div>
          </div>

          <div className="pipeline-label-row">
            <span className="pipeline-label-tag train">监督信号</span>
            <span className="pipeline-label-text">
              每一天的标签 = "未来 20 个交易日内 S&P 500 是否跌超 5%"。用历史已知价格回头标注，模型对比预测与标签来调整权重。
            </span>
          </div>
        </div>

        <div className="pipeline-divider" />

        <div className="pipeline-section">
          <h3 className="pipeline-title">Human Logic 模型</h3>
          <p className="lab-card-desc">权重固定（人工基于经济学直觉设定），但标准化和校准依赖训练数据</p>
          <div className="pipeline-flow">
            <div className="pipeline-step">
              <div className="pipeline-step-num">1</div>
              <div className="pipeline-step-label">原始指标</div>
              <div className="pipeline-step-detail">与 ML 模型完全相同的 23 个特征</div>
            </div>
            <div className="pipeline-arrow">→</div>
            <div className="pipeline-step" style={{ borderColor: '#f59e0b' }}>
              <div className="pipeline-step-num" style={{ background: '#f59e0b' }}>2</div>
              <div className="pipeline-step-label">StandardScaler</div>
              <div className="pipeline-step-detail">
                <strong>借用 ML 的 scaler</strong><br/>
                均值和标准差取决于训练集 → 不同训练期产生不同的 z-score
              </div>
            </div>
            <div className="pipeline-arrow">→</div>
            <div className="pipeline-step highlight-human">
              <div className="pipeline-step-num">3</div>
              <div className="pipeline-step-label">固定权重</div>
              <div className="pipeline-step-detail">
                人工设定：VIX↑=危险(+), 均线下方=危险(-×-=+), 利差走扩=危险(+)
              </div>
            </div>
            <div className="pipeline-arrow">→</div>
            <div className="pipeline-step" style={{ borderColor: '#f59e0b' }}>
              <div className="pipeline-step-num" style={{ background: '#f59e0b' }}>4</div>
              <div className="pipeline-step-label">校准 Sigmoid</div>
              <div className="pipeline-step-detail">
                调整 scale 和 bias 使平均输出 ≈ 训练集的大跌频率（base rate）
              </div>
            </div>
          </div>

          <div className="pipeline-label-row">
            <span className="pipeline-label-tag data-dep">数据依赖</span>
            <span className="pipeline-label-text">
              <strong>橙框 = 受训练数据影响。</strong>
              Step 2: Scaler 的均值/标准差来自训练集（短期 vs 长期算出不同 z-score）。
              Step 4: 校准基于训练集的大跌频率（短期 12% vs 长期 16.5%）。
              此外，零值率{'>'}50% 的特征会被自动关闭（长期数据中 turbulence 等 6 个特征被关闭）。
            </span>
          </div>
        </div>
      </section>

      {/* Current Signal */}
      <section className="lab-signal-card" style={{ borderColor: sigColor }}>
        <div className="lab-signal-main">
          <div className="lab-signal-prob" style={{ color: sigColor }}>
            {(current_prediction.probability * 100).toFixed(1)}%
          </div>
          <div className="lab-signal-meta">
            <div className="lab-signal-label">
              未来20日出现{'>'}5%回撤概率
            </div>
            <div className="lab-signal-date">{current_prediction.date}</div>
            <div className="lab-signal-note">
              精确率约17%，此信号仅供研究参考
            </div>
          </div>
        </div>
      </section>

      {/* Probability Timeline + S&P500 overlay */}
      <section className="lab-card">
        <h2>预测概率 vs 实际走势</h2>
        <p className="lab-card-desc">上图：模型崩盘概率（0–100%）；下图：S&P 500 独立纵轴，横轴对齐。概率超过 50% 虚线即为预警</p>
        <div className="range-picker">
          {RANGE_OPTIONS.map(opt => (
            <button key={opt.key}
              className={`range-btn ${mainRange === opt.key ? 'active' : ''}`}
              onClick={() => setMainRange(opt.key)}>
              {opt.label}
            </button>
          ))}
        </div>
        <div className="lab-chart-tall">
          <StackedProbSPChart
            data={mainChartData}
            series={[{ dataKey: 'probability', name: '崩盘概率', color: '#d6457a', type: 'area' }]}
            dropEvents={mainDropMarkers}
            probHeight={270}
            spHeight={130}
          />
        </div>
      </section>

      {/* Feature Importance */}
      <section className="lab-card">
        <h2>特征重要性 (ML model)</h2>
        <p className="lab-card-desc">正值=预示下跌，负值=预示安全。变化速度类特征（momentum）普遍比绝对水平更重要</p>
        <div className="lab-chart-tall">
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={feature_importance.slice(0, 15)} layout="vertical" margin={{ top: 5, right: 30, left: 140, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1d8e2" />
              <XAxis type="number" tick={{ fill: '#8a7882', fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" tick={{ fill: '#5a4452', fontSize: 11 }}
                tickFormatter={(v: string) => FEATURE_LABELS[v] || v} width={130} />
              <Tooltip contentStyle={CHART_TOOLTIP_STYLE}
                formatter={(v: number) => [v.toFixed(4), '权重']} />
              <ReferenceLine x={0} stroke="#d6e6f7" />
              <Bar dataKey="weight" isAnimationActive={false}>
                {feature_importance.slice(0, 15).map((entry, i) => (
                  <Cell key={i} fill={entry.weight > 0 ? '#dc2626' : '#16a34a'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Backtest Events */}
      <section className="lab-card">
        <h2>历史事件回测</h2>
        <p className="lab-card-desc">模型在已知历史大跌中的预警表现（样本外测试）</p>
        {events_backtest.length === 0 ? (
          <p className="lab-no-data">暂无样本外事件数据</p>
        ) : (
          <div className="lab-events-grid">
            {events_backtest.map(evt => (
              <div key={evt.name} className="lab-event-card">
                <div className="lab-event-name">{evt.name}</div>
                <div className="lab-event-drop" style={{ color: '#dc2626' }}>
                  跌幅 {evt.drop_pct}%
                </div>
                <div className="lab-event-detail">
                  {evt.first_alert_date ? (
                    <>
                      <span className="lab-event-success">提前 {evt.lead_days} 天预警</span>
                      <span className="lab-event-date">首次报警: {evt.first_alert_date}</span>
                      <span>最高概率: {(evt.max_probability * 100).toFixed(0)}%</span>
                    </>
                  ) : (
                    <span className="lab-event-miss">未触发预警 (最高 {(evt.max_probability * 100).toFixed(0)}%)</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Model Info */}
      <section className="lab-card lab-info-card">
        <h2>模型配置</h2>
        <div className="lab-info-grid">
          <div><span className="lab-info-key">模型</span><span>{model_info.name}</span></div>
          <div><span className="lab-info-key">目标</span><span>{model_info.target}</span></div>
          <div><span className="lab-info-key">特征数量</span><span>{model_info.features_count}</span></div>
          <div><span className="lab-info-key">训练样本</span><span>{model_info.training_samples} days</span></div>
          <div><span className="lab-info-key">测试样本</span><span>{model_info.test_samples} days</span></div>
          <div><span className="lab-info-key">正样本比例</span><span>{(model_info.positive_rate * 100).toFixed(1)}%</span></div>
          <div><span className="lab-info-key">训练期</span><span>{model_info.train_period}</span></div>
          <div><span className="lab-info-key">测试期</span><span>{model_info.test_period}</span></div>
          <div><span className="lab-info-key">数据更新</span><span>{model_info.last_updated}</span></div>
        </div>
      </section>

      {/* === PROJECT ROADMAP === */}
      <section className="lab-card roadmap-section">
        <div className="ab-header">
          <h2>项目进度与路线图</h2>
          <span className="ab-badge" style={{ background: '#f0fdf4', color: '#166534', border: '1px solid #bbf7d0' }}>ROADMAP</span>
        </div>

        <div className="roadmap-phase">
          <div className="roadmap-phase-header done">
            <span className="roadmap-phase-tag">Phase 1</span>
            <h3>基线建立</h3>
            <span className="roadmap-status done">DONE</span>
          </div>
          <div className="roadmap-items">
            <div className="roadmap-item done"><span className="roadmap-check">✓</span>风控指标看板（VIX / 利差 / 宽度 / 耦合度 / 湍流度 / 综合评分）</div>
            <div className="roadmap-item done"><span className="roadmap-check">✓</span>ML 基线模型（Logistic Regression, 23 特征, AUC 0.85）</div>
            <div className="roadmap-item done"><span className="roadmap-check">✓</span>Human Logic 基线（人工权重, AUC 0.81）</div>
            <div className="roadmap-item done"><span className="roadmap-check">✓</span>Prediction Lab 看板 + GitHub Actions 每日更新</div>
          </div>
        </div>

        <div className="roadmap-phase">
          <div className="roadmap-phase-header done">
            <span className="roadmap-phase-tag">Phase 2</span>
            <h3>实验框架与优化</h3>
            <span className="roadmap-status done">DONE</span>
          </div>
          <div className="roadmap-items">
            <div className="roadmap-item done"><span className="roadmap-check">✓</span>Exp 1: ML vs Human 权重对比 → ML 胜出</div>
            <div className="roadmap-item done"><span className="roadmap-check">✓</span>Exp 2: 特征去冗余（23→10 特征）→ AUC 0.85→0.89</div>
            <div className="roadmap-item done"><span className="roadmap-check">✓</span>Exp 3: Embargo 隔离（防数据泄露）→ 评估更准确</div>
            <div className="roadmap-item done"><span className="roadmap-check">✓</span>Exp 4: MIN/AND 双模型集成 → MIN AUC 0.80</div>
            <div className="roadmap-item done"><span className="roadmap-check">✓</span>Exp 5: 长期数据实验 → 结论: 更多数据≠更好（非平稳性）</div>
            <div className="roadmap-item done"><span className="roadmap-check">✓</span>模型原理文档 + 方法论标注</div>
          </div>
        </div>

        <div className="roadmap-phase">
          <div className="roadmap-phase-header done">
            <span className="roadmap-phase-tag">Phase 2.5</span>
            <h3>数据增强 (D1)</h3>
            <span className="roadmap-status done">DONE</span>
          </div>
          <div className="roadmap-items">
            <div className="roadmap-item done"><span className="roadmap-check">✓</span>D1 滑动窗口重采样 + Embargo 隔离</div>
            <div className="roadmap-item done"><span className="roadmap-check">✓</span>数据来源审计 + 覆盖分析</div>
          </div>
        </div>

        <div className="roadmap-phase">
          <div className="roadmap-phase-header next">
            <span className="roadmap-phase-tag">Phase 3</span>
            <h3>模型进化</h3>
            <span className="roadmap-status next">NEXT</span>
          </div>
          <div className="roadmap-items">
            <div className="roadmap-item pending"><span className="roadmap-dot" />滑动训练窗口: 用最近 5-7 年训练而非全量，应对非平稳性</div>
            <div className="roadmap-item pending"><span className="roadmap-dot" />非线性模型: XGBoost / Random Forest 对比 Logistic Regression</div>
            <div className="roadmap-item pending"><span className="roadmap-dot" />D2-D5 数据增强备选（SMOTE / 噪声注入 / 多阈值 / Bootstrap）</div>
            <div className="roadmap-item pending"><span className="roadmap-dot" />Human 模型去数据依赖: 使 Scaler/校准不依赖训练集</div>
          </div>
        </div>

        <div className="roadmap-phase">
          <div className="roadmap-phase-header future">
            <span className="roadmap-phase-tag">Phase 4</span>
            <h3>特征扩展</h3>
            <span className="roadmap-status future">FUTURE</span>
          </div>
          <div className="roadmap-items">
            <div className="roadmap-item pending"><span className="roadmap-dot" />新数据源: 期权隐含波动率曲面、资金流向、情绪指标</div>
            <div className="roadmap-item pending"><span className="roadmap-dot" />跨市场信号: 非美市场（A股 / 欧洲 / 日本）的联动预警</div>
            <div className="roadmap-item pending"><span className="roadmap-dot" />宏观日历事件: FOMC / CPI / 非农等事件前后的模式识别</div>
          </div>
        </div>

        <div className="roadmap-current-status">
          <h3>当前状态</h3>
          <div className="roadmap-status-grid">
            <div className="roadmap-stat">
              <span className="roadmap-stat-value">{metrics.experiments?.length ?? 0}</span>
              <span className="roadmap-stat-label">实验总数</span>
            </div>
            <div className="roadmap-stat">
              <span className="roadmap-stat-value">{Math.max(...(metrics.experiments ?? []).filter(e => !e.name.includes('Ext')).map(e => e.auc)).toFixed(2)}</span>
              <span className="roadmap-stat-label">最佳 AUC（短期）</span>
            </div>
            <div className="roadmap-stat">
              <span className="roadmap-stat-value">{metrics.model_info.training_samples + metrics.model_info.test_samples}</span>
              <span className="roadmap-stat-label">短期样本量</span>
            </div>
            <div className="roadmap-stat">
              <span className="roadmap-stat-value">{Math.round((metrics.model_info.training_samples + metrics.model_info.test_samples) * metrics.model_info.positive_rate)}</span>
              <span className="roadmap-stat-label">正样本数</span>
            </div>
          </div>
        </div>
      </section>

      <footer className="lab-footer">
        <div>此页面用于模型研发监督，数据每日更新。预测结果不构成投资建议。</div>
        <div style={{ marginTop: '4px', fontSize: '10px', opacity: 0.6, letterSpacing: '0.3px' }}>
          Built with <strong>auto-dashboard</strong> · design © Coco
        </div>
      </footer>
    </div>
  );
}
