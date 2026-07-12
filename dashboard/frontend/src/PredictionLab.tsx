import { useState, useEffect, useMemo } from 'react';
import {
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, ReferenceLine, Area, AreaChart,
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

const EXP_COLORS = ['#d6457a', '#3a82d6', '#16a34a', '#dc2626', '#8b5cf6', '#06b6d4'];

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
}

function buildPairwiseTests(experiments: ExperimentData[]): PairwiseTest[] {
  const find = (substr: string) => experiments.find(e => e.name.includes(substr));
  const mlBase = find('Logistic Regression') ?? find('ML (');
  const human = find('Human Logic');
  const slim = find('Slim');
  const mlExt = find('ML Extended');
  const humanExt = find('Human Extended');

  const d1 = find('D1');

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
  if (slim && d1) pairs.push({
    label: 'Exp 3: Embargo隔离', variable: '无隔离 vs 20天Embargo（防时序泄露）',
    baseline: slim, challenger: d1,
    baseColor: EXP_COLORS[2], challColor: EXP_COLORS[5],
  });
  if (mlExt && humanExt) pairs.push({
    label: 'Exp 4: 长期数据', variable: '20年数据: ML vs Human',
    baseline: mlExt, challenger: humanExt,
    baseColor: EXP_COLORS[3], challColor: EXP_COLORS[4],
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

  const { model_info, current_prediction, feature_importance, threshold_analysis, events_backtest } = metrics;

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
      {metrics.weight_comparison && (
        <WeightComparisonSection data={metrics.weight_comparison} />
      )}

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

      {/* Threshold Analysis */}
      <section className="lab-card">
        <h2>阈值选择分析</h2>
        <p className="lab-card-desc">不同概率阈值下的精确率、召回率和报警频率。阈值越高误报越少但可能漏报</p>
        <div className="lab-table-wrap">
          <table className="lab-table">
            <thead>
              <tr>
                <th>阈值</th>
                <th>精确率</th>
                <th>召回率</th>
                <th>F1</th>
                <th>报警天数</th>
                <th>报警比例</th>
              </tr>
            </thead>
            <tbody>
              {threshold_analysis.map(row => (
                <tr key={row.threshold} className={row.threshold === 0.5 ? 'lab-row-highlight' : ''}>
                  <td className="lab-td-mono">{(row.threshold * 100).toFixed(0)}%</td>
                  <td style={{ color: row.precision > 0.3 ? '#16a34a' : row.precision > 0.15 ? '#b45309' : '#dc2626' }}>
                    {(row.precision * 100).toFixed(1)}%
                  </td>
                  <td style={{ color: row.recall > 0.7 ? '#16a34a' : row.recall > 0.4 ? '#b45309' : '#dc2626' }}>
                    {(row.recall * 100).toFixed(1)}%
                  </td>
                  <td>{row.f1.toFixed(3)}</td>
                  <td>{row.alert_days} / {row.total_days}</td>
                  <td>{row.alert_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
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

      {/* ROC Curve */}
      <section className="lab-card">
        <h2>ROC 曲线 (AUC = {model_info.roc_auc})</h2>
        <p className="lab-card-desc">越靠左上角越好。0.5为随机猜测基线</p>
        <div className="lab-chart-square">
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={metrics.roc_curve} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1d8e2" />
              <XAxis dataKey="fpr" label={{ value: 'False Positive Rate', fill: '#8a7882', position: 'bottom', offset: 0 }} tick={{ fill: '#8a7882', fontSize: 11 }} />
              <YAxis label={{ value: 'True Positive Rate', fill: '#8a7882', angle: -90, position: 'insideLeft' }} tick={{ fill: '#8a7882', fontSize: 11 }} />
              <Area dataKey="tpr" fill="#8cc3ff" fillOpacity={0.3} stroke="#3a82d6" strokeWidth={2} isAnimationActive={false} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#d6e6f7" strokeDasharray="5 5" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
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

      {/* === DATA AUGMENTATION PROPOSALS === */}
      <section className="lab-card augment-section">
        <div className="ab-header">
          <h2>数据增强方案 (Experiment D)</h2>
          <span className="ab-badge" style={{ background: '#dceeff', color: '#3a82d6', border: '1px solid #bcdcff' }}>PROPOSAL</span>
        </div>
        <p className="lab-card-desc">
          当前核心问题：可用训练样本仅 ~740天，正样本（大跌）仅 ~94个。以下是可行的数据增强方案。
        </p>

        <div className="augment-grid">
          <div className="augment-card">
            <div className="augment-card-header">
              <span className="augment-id">D1</span>
              <h3>滑动窗口重采样</h3>
              <span className="augment-difficulty easy">易实现</span>
            </div>
            <p className="augment-desc">
              将当前固定的 20日前瞻窗口改为滑动步长=1天，每天都产生一条样本。
              理论上可将正样本数量增加 3-5x。
            </p>
            <div className="augment-pros-cons">
              <div className="augment-pro">
                <span className="augment-tag pro">优势</span>
                不引入合成数据，样本是真实市场状态
              </div>
              <div className="augment-con">
                <span className="augment-tag con">风险</span>
                相邻天高度相关（自相关），模型可能过拟合于时间连续性
              </div>
              <div className="augment-mitigation">
                <span className="augment-tag mit">缓解</span>
                使用 Embargo（隔离期）+ 时间序列交叉验证
              </div>
            </div>
          </div>

          <div className="augment-card">
            <div className="augment-card-header">
              <span className="augment-id">D2</span>
              <h3>SMOTE 过采样</h3>
              <span className="augment-difficulty medium">中等</span>
            </div>
            <p className="augment-desc">
              对正样本（大跌前特征向量）做插值合成，在特征空间中生成与真实正样本相近的虚拟样本，
              平衡正负比例至约 1:3。
            </p>
            <div className="augment-pros-cons">
              <div className="augment-pro">
                <span className="augment-tag pro">优势</span>
                经典方法，scikit-learn 直接支持。解决类别不平衡
              </div>
              <div className="augment-con">
                <span className="augment-tag con">风险</span>
                金融时序中特征空间的线性插值不一定反映真实市场状态
              </div>
              <div className="augment-mitigation">
                <span className="augment-tag mit">缓解</span>
                用 Borderline-SMOTE（只在决策边界附近合成）减少噪声
              </div>
            </div>
          </div>

          <div className="augment-card">
            <div className="augment-card-header">
              <span className="augment-id">D3</span>
              <h3>特征噪声注入</h3>
              <span className="augment-difficulty medium">中等</span>
            </div>
            <p className="augment-desc">
              对正样本添加高斯噪声（比如 +/- 5% 扰动），生成变体。模拟市场微观结构的随机性。
            </p>
            <div className="augment-pros-cons">
              <div className="augment-pro">
                <span className="augment-tag pro">优势</span>
                简单直接，可控噪声幅度
              </div>
              <div className="augment-con">
                <span className="augment-tag con">风险</span>
                可能产生不符合物理约束的特征组合（如VIX负值）
              </div>
              <div className="augment-mitigation">
                <span className="augment-tag mit">缓解</span>
                对合成样本做范围裁剪 + 物理约束检查
              </div>
            </div>
          </div>

          <div className="augment-card">
            <div className="augment-card-header">
              <span className="augment-id">D4</span>
              <h3>多阈值目标融合</h3>
              <span className="augment-difficulty easy">易实现</span>
            </div>
            <p className="augment-desc">
              保持主目标（大于5%跌幅），但用3%和8%阈值的标签做辅助任务。模型先学习「任何显著下跌」的模式，
              再 fine-tune 到 5% 目标。
            </p>
            <div className="augment-pros-cons">
              <div className="augment-pro">
                <span className="augment-tag pro">优势</span>
                不产生合成数据，利用不同严格度的真实标签
              </div>
              <div className="augment-con">
                <span className="augment-tag con">风险</span>
                3% 跌幅的模式可能与 5% 跌幅不同
              </div>
              <div className="augment-mitigation">
                <span className="augment-tag mit">缓解</span>
                两阶段训练：先预训练后微调
              </div>
            </div>
          </div>

          <div className="augment-card">
            <div className="augment-card-header">
              <span className="augment-id">D5</span>
              <h3>Bootstrap 集成</h3>
              <span className="augment-difficulty medium">中等</span>
            </div>
            <p className="augment-desc">
              对训练集做有放回重采样，训练多个模型（类似 Bagging），最终概率取平均。
              等效于增加数据多样性。
            </p>
            <div className="augment-pros-cons">
              <div className="augment-pro">
                <span className="augment-tag pro">优势</span>
                减少单模型方差，概率估计更稳定
              </div>
              <div className="augment-con">
                <span className="augment-tag con">风险</span>
                不增加信息量，计算成本高
              </div>
              <div className="augment-mitigation">
                <span className="augment-tag mit">缓解</span>
                配合其他增强方法一起使用效果更好
              </div>
            </div>
          </div>
        </div>

        <div className="augment-recommendation">
          <h3>推荐实验顺序</h3>
          <div className="augment-rec-list">
            <div className="augment-rec-item">
              <span className="augment-rec-num">1</span>
              <div>
                <strong>D1 滑动窗口</strong> — 最低风险，不引入合成数据，预计正样本 x3-5
              </div>
            </div>
            <div className="augment-rec-item">
              <span className="augment-rec-num">2</span>
              <div>
                <strong>D4 多阈值融合</strong> — 利用3%/8%的辅助标签做迁移学习
              </div>
            </div>
            <div className="augment-rec-item">
              <span className="augment-rec-num">3</span>
              <div>
                <strong>D2 SMOTE</strong> — 如果前两步效果有限，再尝试合成过采样
              </div>
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
