import { useState, useEffect, useMemo, Component, type ReactNode } from 'react';
import {
  BarChart, Bar, Cell, CartesianGrid,
  XAxis, YAxis, Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { StackedProbSPChart } from './components/StackedProbSPChart';
import { mergeExperimentTimeline, downsample } from './utils/chart';

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
    <StackedProbSPChart
      data={chartData}
      series={series}
      probHeight={220}
      spHeight={110}
      showLegend
      showThreshold
    />
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

  const { experiments, pairwise, feature_importances, practical_summary } = data;

  const f1Of = (name: string) =>
    experiments.find(e => e.name === name)?.practical_metrics?.best_f1;

  const regimeExtNames = ['LR Ext+Regime9', 'LR Ext+Regime2', 'GBDT Ext+Regime2'] as const;
  const regimeExtExps = regimeExtNames
    .map(n => experiments.find(e => e.name === n))
    .filter((e): e is ExperimentData => !!e && !!e.practical_metrics);
  const lrExtF1 = f1Of('LR Ext');
  const bestRegimeExt = regimeExtExps.length
    ? regimeExtExps.reduce((a, b) =>
        (a.practical_metrics!.best_f1 >= b.practical_metrics!.best_f1) ? a : b)
    : null;
  let regimeExtVerdict = '实验未跑';
  if (bestRegimeExt && lrExtF1 != null) {
    const f1 = bestRegimeExt.practical_metrics!.best_f1;
    const r9 = f1Of('LR Ext+Regime9');
    const r2 = f1Of('LR Ext+Regime2');
    if (f1 > lrExtF1 + 0.01) {
      regimeExtVerdict = `相对 LR Ext(${lrExtF1.toFixed(3)}) 有增量 — 多周期下 regime 信号成立`;
    } else if (r9 != null && r9 < lrExtF1 - 0.05) {
      regimeExtVerdict = `Regime9=${r9.toFixed(3)} / Regime2=${r2?.toFixed(3) ?? '—'} vs Ext=${lrExtF1.toFixed(3)} — 长样本仍无增量（但受分布偏移影响，见审计注意）`;
    } else {
      regimeExtVerdict = `最佳 ${bestRegimeExt.name} F1=${f1.toFixed(3)} vs LR Ext ${lrExtF1.toFixed(3)} — 多周期仍无明显增量`;
    }
  }

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
          <span className="ab-badge" style={{ background: '#eff6ff', color: '#1e40af', border: '1px solid #bfdbfe' }}>
            {regimeExtExps.length > 0 ? '5 STEPS' : '4 STEPS'}
          </span>
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
            <strong style={{ color: '#166534' }}>结论：LR Slim 仍为最佳（F1=0.625）。</strong>
            {' '}GBDT/RF 在 ~950 样本下过拟合：GBDT Slim F1=0.524, RF Slim F1=0.230。小数据量下，简单模型反而更稳健。
          </div>
        </div>

        {/* Step 2 */}
        <div style={{ marginBottom: 16, padding: 12, borderRadius: 8, background: 'rgba(58,130,214,0.04)', border: '1px solid rgba(58,130,214,0.2)' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
            <span style={{ fontWeight: 700, color: '#3a82d6', fontSize: 14 }}>Step 2 · 加 Regime 特征</span>
          </div>
          <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.7 }}>
            <strong>假设：</strong>宏观经济周期状态（加息/降息/曲线倒挂/通胀高企）能提供风险的「背景信号」。<br/>
            <strong>控制变量：</strong>在 Slim 10 特征基础上追加 regime 特征，模型固定为 LR/GBDT。<br/>
            <strong>核心差异（两种版本）：</strong>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8 }}>
            <div style={{ padding: '8px 10px', background: '#fff', borderRadius: 6, border: '1px solid #e5e7eb', fontSize: 12 }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>Regime9（全量 9 特征）</div>
              curve_inverted, curve_flat, fed_rate_level, fed_rate_chg_63d, fed_hiking, fed_cutting, cpi_yoy, cpi_accelerating, cpi_above_3
              <div style={{ color: '#dc2626', fontWeight: 600, marginTop: 4 }}>F1=0.232 — 严重有害</div>
            </div>
            <div style={{ padding: '8px 10px', background: '#fff', borderRadius: 6, border: '1px solid #e5e7eb', fontSize: 12 }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>Regime2（精简 2 特征）</div>
              curve_inverted（收益率曲线是否倒挂）+ fed_hiking（63日内是否加息 {'>'} 25bp）
              <div style={{ color: '#b45309', fontWeight: 600, marginTop: 4 }}>F1=0.545 — 弱于 baseline 但不有害</div>
            </div>
          </div>
          <div style={{ marginTop: 8, padding: '6px 10px', background: '#eff6ff', borderRadius: 6, fontSize: 12 }}>
            <strong style={{ color: '#1e40af' }}>结论：</strong>
            慢变量（联邦利率水平、CPI）与日频 target（20 天大跌）频率不匹配，反而引入噪声。仅「是否倒挂」接近有用。
          </div>
        </div>

        {/* Step 2d */}
        <div style={{ marginBottom: 16, padding: 12, borderRadius: 8, background: 'rgba(139,92,246,0.04)', border: '1px solid rgba(139,92,246,0.2)' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
            <span style={{ fontWeight: 700, color: '#8b5cf6', fontSize: 14 }}>Step 2d · Regime × 长期数据</span>
            <span style={{ fontSize: 11, color: '#92400e', background: '#fef3c7', padding: '2px 6px', borderRadius: 4 }}>AUDIT</span>
          </div>
          <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.7 }}>
            <strong>假设：</strong>Regime 在短期数据（~3年，仅 1 个加息周期）像噪声，是因为样本只覆盖 1 个经济周期。若用 2005+ 数据（3 个完整加息/降息周期），regime 应该有用。<br/>
            <strong>控制变量：</strong>相同特征 + 模型，仅将数据从 ~1000→5333 天。<br/>
            <strong>核心差异 vs Step 2：</strong>数据覆盖多个经济周期（2005-07 加息→ 2008 降息→ 2015-18 加息→ 2020 降息→ 2022-23 加息）。
          </div>
          <div style={{ marginTop: 8, padding: '6px 10px', background: '#faf5ff', borderRadius: 6, fontSize: 12 }}>
            <strong style={{ color: '#6d28d9' }}>结论：{regimeExtExps.length > 0 ? regimeExtVerdict : '待运行'}</strong>
          </div>
          <div style={{ marginTop: 6, padding: '6px 10px', background: '#fffbeb', borderRadius: 6, fontSize: 11, color: '#92400e' }}>
            <strong>审计注意：</strong>
            Regime 特征存在严重 train/test 分布偏移（curve_inverted: 6.5%→34.4%）。结论应读作「当前二值特征 + 固定 split 下无增量」，不等于 regime 假说本身无效。
          </div>
        </div>

        {/* Step 3 */}
        <div style={{ marginBottom: 16, padding: 12, borderRadius: 8, background: 'rgba(234,88,12,0.04)', border: '1px solid rgba(234,88,12,0.2)' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
            <span style={{ fontWeight: 700, color: '#ea580c', fontSize: 14 }}>Step 3 · 事件日历特征</span>
          </div>
          <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.7 }}>
            <strong>假设：</strong>重大经济事件发布前后（FOMC 利率决议、CPI 数据公布、非农就业报告）市场波动加剧，接近事件 = 更高风险。<br/>
            <strong>控制变量：</strong>在 Slim 基础上追加「距离最近 FOMC/CPI/NFP 的天数」和「是否在事件前后 3/7 天窗口内」。<br/>
            <strong>核心差异 vs Step 2：</strong>Regime 是「经济状态」（慢变量），Events 是「时间邻近性」（快变量/周期性）。
          </div>
          <div style={{ marginTop: 8, padding: '6px 10px', background: '#fff7ed', borderRadius: 6, fontSize: 12 }}>
            <strong style={{ color: '#c2410c' }}>结论：事件邻近性无预测增量。</strong>
            {' '}Slim+Events F1=0.500, KitchenSink (Slim+Regime2+Events) F1=0.556 — 均不超越 Slim baseline。
            大跌的发生不取决于「是否在 FOMC 前后」，而是取决于「冲击的意外程度」，这无法用日历捕捉。
          </div>
        </div>

        {/* Step 4 */}
        <div style={{ marginBottom: 4, padding: 12, borderRadius: 8, background: 'rgba(220,38,38,0.04)', border: '1px solid rgba(220,38,38,0.2)' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
            <span style={{ fontWeight: 700, color: '#dc2626', fontSize: 14 }}>Step 4 · 长期数据重测</span>
          </div>
          <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.7 }}>
            <strong>假设：</strong>更多训练数据（5333 vs 952 天）应让模型更稳健。<br/>
            <strong>控制变量：</strong>相同的 Slim 10 特征 + 相同模型，仅扩展数据到 2005+。<br/>
            <strong>核心差异 vs Step 1：</strong>训练集跨越次贷危机(2008)、欧债危机(2011)、COVID(2020)、通胀加息(2022)等完全不同的市场 regime。
          </div>
          <div style={{ marginTop: 8, padding: '6px 10px', background: '#fef2f2', borderRadius: 6, fontSize: 12 }}>
            <strong style={{ color: '#991b1b' }}>结论：性能显著退化。</strong>
            {' '}LR/GBDT/RF 全部 F1≈0.31, AUC 从 ~0.90 降至 ~0.57。
            根因是「非平稳性」：2008 年学到的模式（如「VIX {'>'} 30 = 危险」）在 2024 年的市场结构下不适用，跨周期的统计规律太弱。
          </div>
        </div>

        {/* Overall conclusion */}
        <div style={{ marginTop: 16, padding: '12px 14px', background: '#f0fdf4', borderRadius: 8, border: '1px solid #bbf7d0' }}>
          <div style={{ fontWeight: 700, fontSize: 14, color: '#166534', marginBottom: 6 }}>总结</div>
          <div style={{ fontSize: 13, lineHeight: 1.7, color: '#374151' }}>
            <strong>LR Slim (F1=0.625)</strong> 至今未被超越。三条突破路径均失败：
            <span style={{ color: '#d6457a' }}> 换模型</span> = 小样本过拟合 ·
            <span style={{ color: '#3a82d6' }}> 加特征</span> = 频率不匹配的噪声 ·
            <span style={{ color: '#16a34a' }}> 加数据</span> = 跨周期非平稳性。
            下一步应探索 regime-switching 模型或 LLM 定性分析（Ch.3 Multi-Agent）。
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
              <span>Step 4 baseline 的 sp500_timeline 误用短期数据 → 已改为长期</span>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <span style={{ color: '#16a34a', fontWeight: 600, whiteSpace: 'nowrap' }}>FIXED</span>
              <span>cpi_accelerating 用 diff(1) 导致 98.5% 为零 → 改为 diff(21)</span>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <span style={{ color: '#b45309', fontWeight: 600, whiteSpace: 'nowrap' }}>CAVEAT</span>
              <span>Regime 特征 train/test 分布偏移严重（curve_inverted 6.5%→34.4%），结论适用范围有限</span>
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
