import { useState, useEffect, useMemo, Component, type ReactNode } from 'react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { fetchDataJson } from './api';
import { downsample, CHART_TOOLTIP_STYLE } from './utils/chart';
import { LazyMount } from './components/LazyMount';

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

interface ReturnSummary { mean: number | null; median: number | null; hit_rate: number | null; count?: number; }
interface WindowCandidate {
  window: string; event_hit_rate: number | null; baseline_hit_rate: number | null;
  excess_hit_rate: number | null; event_mean_pct: number | null;
  p_value: number | null; phi: number | null; significant: boolean; verdict: string;
}
interface WindowSweepEntry {
  event: string;
  pre_windows_tested: string[];
  post_windows_tested: string[];
  best_pre: WindowCandidate;
  best_post: WindowCandidate;
  all_pre: WindowCandidate[];
  all_post: WindowCandidate[];
}
interface HypothesisEvent {
  event: string;
  best_pre_window?: string;
  best_post_window?: string;
  h1_pre_rise: {
    confirmed: boolean; mean_pct: number | null; hit_rate_up: number | null;
    baseline_hit_rate_up?: number | null; excess_hit_rate?: number | null;
    p_value?: number | null; verdict?: string;
  };
  h2_post_fall: {
    confirmed: boolean; mean_pct: number | null; hit_rate_down: number | null;
    baseline_hit_rate_down?: number | null; excess_hit_rate?: number | null;
    p_value?: number | null; verdict?: string;
  };
}
interface CorrSide {
  event_hit_rate_up?: number; event_hit_rate_down?: number;
  baseline_hit_rate_up?: number; baseline_hit_rate_down?: number;
  excess_hit_rate: number | null; event_mean_pct: number | null; baseline_mean_pct: number | null;
  event_n: number; baseline_n: number; z_stat: number | null; p_value: number | null;
  phi: number | null; significant: boolean; verdict: string;
}
interface CorrelationAnalysis {
  method: string;
  baseline_reference: {
    pre_7d: { hit_rate_up: number; mean_pct: number; n: number; description: string };
    post_1d: { hit_rate_down: number; mean_pct: number; n: number; description: string };
  };
  by_event: { event: string; best_pre_window?: string; best_post_window?: string; h1_pre_rise: CorrSide; h2_post_fall: CorrSide }[];
}
interface EventRow {
  date: string; trading_day: string; vix_at_event: number;
  pre_return_pct: number | null; pre_window: string;
  post_1d_return_pct?: number | null; post_2d_return_pct?: number | null;
  post_3d_return_pct?: number | null; post_5d_return_pct?: number | null;
  best_post_window?: string;
}
interface EventStudy {
  event_type: string; best_pre_window: string; best_post_window: string;
  events: EventRow[];
  summary: { count: number; pre: ReturnSummary; post: ReturnSummary };
}
interface HistoricalPeriod {
  label: string; date_range: string; n_events: number; error?: string;
  return_based?: { event_hit_rate: number | null; baseline_hit_rate: number | null; excess_hit_rate: number | null; p_value: number | null; significant: boolean | null; verdict: string };
  level_based?: { event_mean_vix: number | null; baseline_mean_vix: number | null; excess_vix: number | null; p_value: number | null; significant: boolean | null; verdict: string };
}
interface RegimeResult {
  regime: string; n: number; error?: string;
  hit_rate_up: number | null; mean_pct: number | null; p_value: number | null; significant: boolean;
  vix_range?: [number, number];
}
interface EventVolData {
  title: string; subtitle: string;
  primary_analysis?: {
    method: string;
    windows: { h1_pre: string; h2_post: string };
    baseline_stride: number;
    bonferroni_tests: number;
    bonferroni_alpha: number;
    baseline_reference: {
      pre: { hit_rate_up: number; mean_pct: number; n: number; description: string };
      post: { hit_rate_down: number; mean_pct: number; n: number; description: string };
    };
    conclusion_summary: string;
  };
  conclusion_summary?: string;
  hypothesis: { h1: string; h2: string; by_event: HypothesisEvent[] };
  correlation_analysis: CorrelationAnalysis;
  window_sweep: WindowSweepEntry[];
  sensitivity?: {
    covid_fomc: {
      description: string; events_remaining: number; pre_window: string;
      hit_rate_up: number; baseline_hit_rate_up: number; excess_hit_rate: number;
      p_value: number; significant: boolean; verdict: string;
    };
  };
  level_analysis?: {
    description_a: string; description_b: string; full_sample_mean_vix: number;
    method_a: Array<{ event: string; event_mean: number; baseline_mean: number; excess: number; p_value: number; significant: boolean; verdict: string; event_n: number; baseline_n: number }>;
    method_b: Array<{ event: string; event_mean: number; baseline_mean: number; excess: number; p_value: number; significant: boolean; verdict: string; event_n: number; baseline_n: number }>;
    note: string;
  };
  subsample_analysis?: Record<string, {
    label: string; date_range: string; note: string;
    by_event: Array<{
      event: string; n_events: number;
      return_based: { event_hit_rate: number | null; baseline_hit_rate: number | null; excess_hit_rate: number | null; p_value: number | null; verdict: string };
      level_based: { event_mean_vix: number | null; baseline_mean_vix: number | null; excess_vix: number | null; p_value: number | null; verdict: string };
    }>;
  }>;
  summary: { data_range: string; analysis_years?: number; instrument: string; total_trading_days: number; method: string; event_counts?: Record<string, number> };
  event_studies: EventStudy[];
  upcoming_events: { event: string; date: string }[];
  vix_timeline: { date: string; vix: number; daily_return_pct: number | null; event_flags: string[] }[];
  methodology?: {
    exploratory_sweep?: { warning: string; bonferroni_alpha: number; tests: number };
  };
  historical_replication?: {
    description: string; reference: string; split_date: string; vix_data_range: string;
    total_fomc_events: number; window: string; interpretation: string;
    periods: HistoricalPeriod[];
    error?: string;
  };
  conditional_analysis?: {
    description: string; window: string; total_events: number;
    overall_hit_rate: number; thresholds: { p33: number; p67: number };
    by_regime: RegimeResult[]; interpretation: string; error?: string;
  };
  skew_analysis?: {
    description: string; instrument: string; window: string; data_range: string;
    n_fomc_events: number;
    skew_stats: {
      mean_full_period: number; event_mean_change: number; baseline_mean_change: number | null;
      excess_change: number | null; hit_rate_up: number; p_value: number | null; significant: boolean;
    };
    recent_events: Array<{ date: string; skew_at_t5: number; skew_at_t1: number; skew_change: number }>;
    interpretation: string; error?: string;
  };
}

const EVENT_COLORS: Record<string, string> = { FOMC: '#ea580c', CPI: '#3a82d6', NFP: '#16a34a' };
const EVENT_TYPE_LABEL: Record<string, string> = { fomc: 'FOMC', cpi: 'CPI', nfp: 'NFP' };

function fmtPct(v: number | null | undefined, digits = 2) {
  if (v == null) return '—';
  return `${v > 0 ? '+' : ''}${v.toFixed(digits)}%`;
}

function sigBadge(p: number | null, alpha = 0.05) {
  if (p == null) return null;
  if (p < alpha) return <span style={{ fontSize: 10, color: '#166534', background: '#dcfce7', padding: '1px 5px', borderRadius: 4 }}>p&lt;{alpha}</span>;
  if (p < alpha * 2) return <span style={{ fontSize: 10, color: '#92400e', background: '#fef3c7', padding: '1px 5px', borderRadius: 4 }}>边际</span>;
  return null;
}

function pctColor(v: number | null, invert = false) {
  if (v == null) return '#6b7280';
  const up = invert ? v < 0 : v > 0;
  const down = invert ? v > 0 : v < 0;
  return up ? '#dc2626' : down ? '#16a34a' : '#6b7280';
}

function ConfirmBadge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span style={{
      fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 4,
      background: ok ? '#dcfce7' : '#fef2f2',
      color: ok ? '#166534' : '#991b1b',
    }}>{label}</span>
  );
}

function EventVolLabInner() {
  const [data, setData] = useState<EventVolData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeSweep, setActiveSweep] = useState('FOMC');

  useEffect(() => {
    fetchDataJson<EventVolData>('event_vix_analysis.json')
      .then(d => setData(d))
      .catch(e => setError(e.message));
  }, []);

  const chartTimeline = useMemo(() => {
    if (!data) return [];
    return downsample(data.vix_timeline.map(p => ({ date: p.date, vix: p.vix })));
  }, [data]);

  const preChartData = useMemo(() => {
    if (!data) return [];
    const base = data.correlation_analysis.baseline_reference.pre_7d.hit_rate_up * 100;
    return data.correlation_analysis.by_event.map(e => ({
      name: e.event,
      event_rate: ((e.h1_pre_rise as CorrSide).event_hit_rate_up ?? 0) * 100,
      baseline_rate: ((e.h1_pre_rise as CorrSide).baseline_hit_rate_up ?? base / 100) * 100,
      window: e.best_pre_window ?? '',
    }));
  }, [data]);

  const postChartData = useMemo(() => {
    if (!data) return [];
    const base = data.correlation_analysis.baseline_reference.post_1d.hit_rate_down * 100;
    return data.correlation_analysis.by_event.map(e => ({
      name: e.event,
      event_rate: ((e.h2_post_fall as CorrSide).event_hit_rate_down ?? 0) * 100,
      baseline_rate: ((e.h2_post_fall as CorrSide).baseline_hit_rate_down ?? base / 100) * 100,
      window: e.best_post_window ?? '',
    }));
  }, [data]);

  if (error) return <div className="lab-container"><div className="lab-card"><p>Event VIX data not available: {error}</p></div></div>;
  if (!data) return <div className="loading">Loading...</div>;

  const { hypothesis, correlation_analysis, window_sweep, summary, event_studies, upcoming_events, primary_analysis, conclusion_summary, sensitivity, methodology, level_analysis, subsample_analysis } = data;
  const bonferroniAlpha = primary_analysis?.bonferroni_alpha ?? 0.05;
  const sweep = window_sweep.find(s => s.event === activeSweep);
  const study = event_studies.find(s => EVENT_TYPE_LABEL[s.event_type] === activeSweep);

  return (
    <div className="lab-container">
      <header className="lab-header">
        <div>
          <h1>Event × VIX</h1>
          <p className="lab-subtitle">{data.subtitle}</p>
        </div>
        <div className="lab-model-badge">
          <span className="lab-badge-version">L2</span>
          <span className="lab-badge-auc">{summary.data_range}</span>
          <span className="lab-badge-auc">{summary.instrument}</span>
        </div>
      </header>

      {/* Hypotheses */}
      <section className="lab-card">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div style={{ padding: 14, borderRadius: 8, border: '1px solid #fed7aa', background: '#fff7ed' }}>
            <div style={{ fontWeight: 700, color: '#c2410c', marginBottom: 6 }}>H1 · {hypothesis.h1}</div>
            <div style={{ fontSize: 13, color: '#374151' }}>
              固定窗口 {primary_analysis?.windows.h1_pre ?? 'T-5~T-1'} · Bonferroni α={bonferroniAlpha}
            </div>
          </div>
          <div style={{ padding: 14, borderRadius: 8, border: '1px solid #bfdbfe', background: '#eff6ff' }}>
            <div style={{ fontWeight: 700, color: '#1d4ed8', marginBottom: 6 }}>H2 · {hypothesis.h2}</div>
            <div style={{ fontSize: 13, color: '#374151' }}>
              固定窗口 {primary_analysis?.windows.h2_post ?? 'T+1'} · 稀疏基准 stride={primary_analysis?.baseline_stride ?? 5}
            </div>
          </div>
        </div>
      </section>

      {/* Baseline reference */}
      <section className="lab-card">
        <h2 style={{ fontSize: 16, marginBottom: 10 }}>非事件日基准（对照组）</h2>
        <p style={{ fontSize: 13, color: '#6b7280', marginBottom: 12 }}>
          {correlation_analysis.method}
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div style={{ padding: 12, borderRadius: 8, background: '#f9fafb', border: '1px solid #e5e7eb', fontSize: 13 }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>{correlation_analysis.baseline_reference.pre_7d.description}</div>
            <div>VIX 上涨率 <strong>{(correlation_analysis.baseline_reference.pre_7d.hit_rate_up * 100).toFixed(1)}%</strong>（n={correlation_analysis.baseline_reference.pre_7d.n}）</div>
            <div style={{ color: '#6b7280' }}>均值 {fmtPct(correlation_analysis.baseline_reference.pre_7d.mean_pct)}</div>
          </div>
          <div style={{ padding: 12, borderRadius: 8, background: '#f9fafb', border: '1px solid #e5e7eb', fontSize: 13 }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>{correlation_analysis.baseline_reference.post_1d.description}</div>
            <div>VIX 下跌率 <strong>{(correlation_analysis.baseline_reference.post_1d.hit_rate_down * 100).toFixed(1)}%</strong>（n={correlation_analysis.baseline_reference.post_1d.n}）</div>
            <div style={{ color: '#6b7280' }}>均值 {fmtPct(correlation_analysis.baseline_reference.post_1d.mean_pct)}</div>
          </div>
        </div>
      </section>

      {/* Verdict cards — primary analysis */}
      <section className="lab-card">
        <h2 style={{ fontSize: 16, marginBottom: 12 }}>主结论（预注册固定窗口）</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
          {hypothesis.by_event.map(e => (
            <div key={e.event} style={{ padding: 14, borderRadius: 8, border: '1px solid #e5e7eb' }}>
              <div style={{ fontWeight: 700, color: EVENT_COLORS[e.event], marginBottom: 10 }}>
                {e.event}
                <div style={{ fontSize: 11, fontWeight: 400, color: '#6b7280' }}>
                  固定窗口：{e.best_pre_window} / {e.best_post_window}
                </div>
              </div>
              <div style={{ fontSize: 13, lineHeight: 2 }}>
                <div>
                  H1 上涨率：{(e.h1_pre_rise.hit_rate_up! * 100).toFixed(0)}% vs 基准 {(e.h1_pre_rise.baseline_hit_rate_up! * 100).toFixed(0)}%
                  <span style={{ marginLeft: 6, fontSize: 11, color: pctColor(e.h1_pre_rise.excess_hit_rate ?? null) }}>
                    ({e.h1_pre_rise.excess_hit_rate! > 0 ? '+' : ''}{(e.h1_pre_rise.excess_hit_rate! * 100).toFixed(0)}pp)
                  </span>
                  <div><ConfirmBadge ok={e.h1_pre_rise.confirmed} label={e.h1_pre_rise.verdict ?? ''} /> {sigBadge(e.h1_pre_rise.p_value ?? null, bonferroniAlpha)}</div>
                </div>
                <div>
                  H2 下跌率：{(e.h2_post_fall.hit_rate_down! * 100).toFixed(0)}% vs 基准 {(e.h2_post_fall.baseline_hit_rate_down! * 100).toFixed(0)}%
                  <span style={{ marginLeft: 6, fontSize: 11, color: pctColor(e.h2_post_fall.excess_hit_rate ?? null) }}>
                    ({e.h2_post_fall.excess_hit_rate! > 0 ? '+' : ''}{(e.h2_post_fall.excess_hit_rate! * 100).toFixed(0)}pp)
                  </span>
                  <div><ConfirmBadge ok={e.h2_post_fall.confirmed} label={e.h2_post_fall.verdict ?? ''} /> {sigBadge(e.h2_post_fall.p_value ?? null, bonferroniAlpha)}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 12, padding: '10px 12px', background: '#f0fdf4', borderRadius: 8, fontSize: 13, lineHeight: 1.7, border: '1px solid #bbf7d0' }}>
          <strong style={{ color: '#166534' }}>结论：</strong>
          {conclusion_summary ?? primary_analysis?.conclusion_summary ?? '—'}
        </div>
        {sensitivity?.covid_fomc && (
          <div style={{ marginTop: 10, padding: '10px 12px', background: '#fffbeb', borderRadius: 8, fontSize: 12, lineHeight: 1.6, border: '1px solid #fde68a', color: '#78350f' }}>
            <strong>敏感性分析：</strong>{sensitivity.covid_fomc.description} — FOMC 前 {sensitivity.covid_fomc.pre_window}{' '}
            {(sensitivity.covid_fomc.hit_rate_up * 100).toFixed(0)}% vs {(sensitivity.covid_fomc.baseline_hit_rate_up * 100).toFixed(0)}%，
            p={sensitivity.covid_fomc.p_value.toFixed(3)}（{sensitivity.covid_fomc.verdict}）
          </div>
        )}
      </section>

      {/* Level Analysis */}
      {level_analysis && (
        <section className="lab-card">
          <h2 style={{ fontSize: 16, marginBottom: 4 }}>水平比较分析（Level-Based）</h2>
          <p style={{ fontSize: 12, color: '#6b7280', marginBottom: 14 }}>
            与涨跌方向不同，这里直接比较 VIX <strong>绝对水平</strong>。
            VIX 的 30 天前瞻性意味着：如果市场在事件前普遍焦虑，VIX 水平应高于无事件期。
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {/* Method A */}
            <div>
              <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 8 }}>
                Method A · 窗口均值
              </div>
              <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 10 }}>{level_analysis.description_a}</div>
              <table className="ab-table" style={{ fontSize: 12 }}>
                <thead>
                  <tr><th>事件</th><th>事件窗口均值</th><th>基准均值</th><th>超额</th><th>p</th><th>判定</th></tr>
                </thead>
                <tbody>
                  {level_analysis.method_a.map(r => (
                    <tr key={r.event}>
                      <td><span style={{ color: EVENT_COLORS[r.event], fontWeight: 600 }}>{r.event}</span></td>
                      <td>{r.event_mean?.toFixed(1) ?? '—'}</td>
                      <td>{r.baseline_mean?.toFixed(1) ?? '—'}</td>
                      <td style={{ color: pctColor(r.excess) }}>{r.excess != null ? `${r.excess > 0 ? '+' : ''}${r.excess.toFixed(1)}` : '—'}</td>
                      <td>{r.p_value?.toFixed(3) ?? '—'} {sigBadge(r.p_value, bonferroniAlpha)}</td>
                      <td>{r.verdict}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {/* Method B */}
            <div>
              <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 8 }}>
                Method B · T-1 绝对水平
              </div>
              <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 10 }}>{level_analysis.description_b}</div>
              <table className="ab-table" style={{ fontSize: 12 }}>
                <thead>
                  <tr><th>事件</th><th>T-1 均值</th><th>全样本均值</th><th>超额</th><th>p</th><th>判定</th></tr>
                </thead>
                <tbody>
                  {level_analysis.method_b.map(r => (
                    <tr key={r.event}>
                      <td><span style={{ color: EVENT_COLORS[r.event], fontWeight: 600 }}>{r.event}</span></td>
                      <td>{r.event_mean?.toFixed(1) ?? '—'}</td>
                      <td>{r.baseline_mean?.toFixed(1) ?? '—'}</td>
                      <td style={{ color: pctColor(r.excess) }}>{r.excess != null ? `${r.excess > 0 ? '+' : ''}${r.excess.toFixed(1)}` : '—'}</td>
                      <td>{r.p_value?.toFixed(3) ?? '—'} {sigBadge(r.p_value, bonferroniAlpha)}</td>
                      <td>{r.verdict}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div style={{ marginTop: 12, padding: '10px 12px', background: '#fefce8', borderRadius: 8, fontSize: 12, lineHeight: 1.7, border: '1px solid #fef08a', color: '#713f12' }}>
            <strong>解读：</strong>
            两种水平比较方法均显示，事件前 VIX 水平与非事件日<strong>几乎相同</strong>（超额仅 ±0.2）。
            这说明 VIX 的"事件焦虑"在更长的周期里已被定价——因为 FOMC/CPI/NFP 每月都有，
            VIX 的 30 天前瞻窗口几乎始终在定价某个即将到来的事件。真正的"清醒期"很短，
            导致事件期 vs 非事件期的<strong>水平差别趋近于零</strong>。
            这也解释了为何涨跌方向（return-based）的信号更有意义：它问的是事件是否让 VIX <em>额外加速</em>，而不是水平本身是否更高。
          </div>
        </section>
      )}

      {/* Sub-sample Analysis */}
      {subsample_analysis && (
        <section className="lab-card">
          <h2 style={{ fontSize: 16, marginBottom: 4 }}>近期子样本分析</h2>
          <p style={{ fontSize: 12, color: '#6b7280', marginBottom: 14 }}>
            近期样本量较小（1y: n≈8~12 / 2y: n≈16~24），p 值仅供方向参考，不作主要结论依据。
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {Object.entries(subsample_analysis).map(([key, sub]) => (
              <div key={key} style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 14 }}>
                <div style={{ fontWeight: 700, marginBottom: 2 }}>{sub.label}</div>
                <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 10 }}>{sub.date_range}</div>
                <table className="ab-table" style={{ fontSize: 11 }}>
                  <thead>
                    <tr><th>事件</th><th>n</th><th>涨跌方向 p</th><th>水平均值 p</th><th>判定</th></tr>
                  </thead>
                  <tbody>
                    {sub.by_event.map(row => (
                      <tr key={row.event}>
                        <td><span style={{ color: EVENT_COLORS[row.event], fontWeight: 600 }}>{row.event}</span></td>
                        <td>{row.n_events}</td>
                        <td style={{ color: (row.return_based.p_value ?? 1) < 0.05 ? '#166534' : '#374151' }}>
                          {row.return_based.p_value?.toFixed(3) ?? '—'}
                        </td>
                        <td style={{ color: (row.level_based.p_value ?? 1) < 0.05 ? '#166534' : '#374151' }}>
                          {row.level_based.p_value?.toFixed(3) ?? '—'}
                        </td>
                        <td style={{ fontSize: 10 }}>{row.return_based.verdict}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 12, padding: '10px 12px', background: '#f0f9ff', borderRadius: 8, fontSize: 12, lineHeight: 1.7, border: '1px solid #bae6fd', color: '#0c4a6e' }}>
            <strong>近两年要点：</strong>
            2024-2026 美联储进入降息周期，利率不确定性格局与 2022-2023 加息期不同。
            近一年 FOMC 前 VIX 甚至呈负向（p=1.0），表明市场对降息节奏已高度预期，预期本身差异在收窄。
            近两年 NFP 前 return p=0.015 有亮点，但水平比较不支持，需更多数据验证。
          </div>
        </section>
      )}

      {/* Exploratory window sweep */}
      <section className="lab-card ab-section">
        <div className="ab-header">
          <div>
            <h2>探索性窗口扫描</h2>
            {methodology?.exploratory_sweep && (
              <p style={{ fontSize: 12, color: '#b45309', margin: '4px 0 0' }}>
                ⚠ {methodology.exploratory_sweep.warning}（Bonferroni α≈{methodology.exploratory_sweep.bonferroni_alpha}，{methodology.exploratory_sweep.tests} tests）
              </p>
            )}
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            {window_sweep.map(s => (
              <button key={s.event} onClick={() => setActiveSweep(s.event)}
                style={{
                  padding: '4px 12px', borderRadius: 6, fontSize: 12, cursor: 'pointer', border: '1px solid',
                  borderColor: activeSweep === s.event ? EVENT_COLORS[s.event] : '#e5e7eb',
                  background: activeSweep === s.event ? `${EVENT_COLORS[s.event]}15` : '#fff',
                  color: activeSweep === s.event ? EVENT_COLORS[s.event] : '#6b7280',
                  fontWeight: activeSweep === s.event ? 600 : 400,
                }}>{s.event}</button>
            ))}
          </div>
        </div>
        {sweep && (
          <div style={{ overflowX: 'auto' }}>
            <table className="ab-table" style={{ fontSize: 12 }}>
              <thead>
                <tr>
                  <th>假设</th><th>窗口</th><th>事件率</th><th>基准率</th><th>超额</th><th>均值</th><th>p</th><th>判定</th>
                </tr>
              </thead>
              <tbody>
                {sweep.all_pre.map(c => (
                  <tr key={c.window} style={{ background: c.window === sweep.best_pre.window ? '#fff7ed' : undefined }}>
                    <td>H1 发布前上涨</td>
                    <td><strong>{c.window}</strong>{c.window === sweep.best_pre.window ? ' ★' : ''}</td>
                    <td>{((c.event_hit_rate ?? 0) * 100).toFixed(1)}%</td>
                    <td>{((c.baseline_hit_rate ?? 0) * 100).toFixed(1)}%</td>
                    <td style={{ color: pctColor(c.excess_hit_rate) }}>{c.excess_hit_rate != null ? `${c.excess_hit_rate > 0 ? '+' : ''}${(c.excess_hit_rate * 100).toFixed(1)}pp` : '—'}</td>
                    <td>{fmtPct(c.event_mean_pct)}</td>
                    <td>{c.p_value?.toFixed(3) ?? '—'} {sigBadge(c.p_value)}</td>
                    <td>{c.verdict}</td>
                  </tr>
                ))}
                {sweep.all_post.map(c => (
                  <tr key={c.window} style={{ background: c.window === sweep.best_post.window ? '#eff6ff' : undefined }}>
                    <td>H2 发布后下跌</td>
                    <td><strong>{c.window}</strong>{c.window === sweep.best_post.window ? ' ★' : ''}</td>
                    <td>{((c.event_hit_rate ?? 0) * 100).toFixed(1)}%</td>
                    <td>{((c.baseline_hit_rate ?? 0) * 100).toFixed(1)}%</td>
                    <td style={{ color: pctColor(c.excess_hit_rate) }}>{c.excess_hit_rate != null ? `${c.excess_hit_rate > 0 ? '+' : ''}${(c.excess_hit_rate * 100).toFixed(1)}pp` : '—'}</td>
                    <td>{fmtPct(c.event_mean_pct)}</td>
                    <td>{c.p_value?.toFixed(3) ?? '—'} {sigBadge(c.p_value)}</td>
                    <td>{c.verdict}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Hit rate comparison charts */}
      <section className="lab-card">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div>
            <h3 style={{ fontSize: 14, marginBottom: 8 }}>H1 · 发布前 VIX 上涨率（事件 vs 基准）</h3>
            <LazyMount minHeight={240}>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={preChartData} margin={{ top: 5, right: 10, bottom: 5, left: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1d8e2" />
                  <XAxis dataKey="name" tick={{ fill: '#8a7882', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#8a7882', fontSize: 11 }} unit="%" domain={[0, 80]} />
                  <Tooltip contentStyle={CHART_TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(1)}%`]} />
                  <Bar dataKey="event_rate" name="事件日" fill="#dc2626" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="baseline_rate" name="非事件基准" fill="#9ca3af" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </LazyMount>
          </div>
          <div>
            <h3 style={{ fontSize: 14, marginBottom: 8 }}>H2 · 发布后 VIX 下跌率（事件 vs 基准）</h3>
            <LazyMount minHeight={240}>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={postChartData} margin={{ top: 5, right: 10, bottom: 5, left: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1d8e2" />
                  <XAxis dataKey="name" tick={{ fill: '#8a7882', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#8a7882', fontSize: 11 }} unit="%" domain={[0, 80]} />
                  <Tooltip contentStyle={CHART_TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(1)}%`]} />
                  <Bar dataKey="event_rate" name="事件日" fill="#16a34a" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="baseline_rate" name="非事件基准" fill="#9ca3af" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </LazyMount>
          </div>
        </div>
      </section>

      {/* Correlation detail table */}
      <section className="lab-card ab-section">
        <div className="ab-header"><h2>相关性统计明细</h2></div>
        <div style={{ overflowX: 'auto' }}>
          <table className="ab-table" style={{ fontSize: 12 }}>
            <thead>
              <tr>
                <th>事件</th><th>假设</th>
                <th>事件命中率</th><th>基准命中率</th><th>超额</th>
                <th>phi</th><th>z</th><th>p</th><th>判定</th>
              </tr>
            </thead>
            <tbody>
              {correlation_analysis.by_event.flatMap(e => [
                <tr key={`${e.event}-h1`}>
                  <td><span style={{ color: EVENT_COLORS[e.event], fontWeight: 600 }}>{e.event}</span></td>
                  <td>H1 {e.best_pre_window}</td>
                  <td>{((e.h1_pre_rise.event_hit_rate_up ?? 0) * 100).toFixed(1)}%</td>
                  <td>{((e.h1_pre_rise.baseline_hit_rate_up ?? 0) * 100).toFixed(1)}%</td>
                  <td style={{ fontWeight: 600, color: pctColor(e.h1_pre_rise.excess_hit_rate) }}>
                    {e.h1_pre_rise.excess_hit_rate != null ? `${e.h1_pre_rise.excess_hit_rate > 0 ? '+' : ''}${(e.h1_pre_rise.excess_hit_rate * 100).toFixed(1)}pp` : '—'}
                  </td>
                  <td>{e.h1_pre_rise.phi?.toFixed(4) ?? '—'}</td>
                  <td>{e.h1_pre_rise.z_stat?.toFixed(2) ?? '—'}</td>
                  <td>{e.h1_pre_rise.p_value?.toFixed(3) ?? '—'} {sigBadge(e.h1_pre_rise.p_value)}</td>
                  <td>{e.h1_pre_rise.verdict}</td>
                </tr>,
                <tr key={`${e.event}-h2`}>
                  <td><span style={{ color: EVENT_COLORS[e.event], fontWeight: 600 }}>{e.event}</span></td>
                  <td>H2 {e.best_post_window}</td>
                  <td>{((e.h2_post_fall.event_hit_rate_down ?? 0) * 100).toFixed(1)}%</td>
                  <td>{((e.h2_post_fall.baseline_hit_rate_down ?? 0) * 100).toFixed(1)}%</td>
                  <td style={{ fontWeight: 600, color: pctColor(e.h2_post_fall.excess_hit_rate) }}>
                    {e.h2_post_fall.excess_hit_rate != null ? `${e.h2_post_fall.excess_hit_rate > 0 ? '+' : ''}${(e.h2_post_fall.excess_hit_rate * 100).toFixed(1)}pp` : '—'}
                  </td>
                  <td>{e.h2_post_fall.phi?.toFixed(4) ?? '—'}</td>
                  <td>{e.h2_post_fall.z_stat?.toFixed(2) ?? '—'}</td>
                  <td>{e.h2_post_fall.p_value?.toFixed(3) ?? '—'} {sigBadge(e.h2_post_fall.p_value)}</td>
                  <td>{e.h2_post_fall.verdict}</td>
                </tr>,
              ])}
            </tbody>
          </table>
        </div>
      </section>

      {/* VIX timeline */}
      <section className="lab-card">
        <h2 style={{ fontSize: 16, marginBottom: 8 }}>VIX 走势</h2>
        <LazyMount minHeight={220}>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartTimeline} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1d8e2" />
              <XAxis dataKey="date" tick={{ fill: '#8a7882', fontSize: 10 }} tickFormatter={(d: string) => d.slice(0, 7)} minTickGap={80} />
              <YAxis tick={{ fill: '#8a7882', fontSize: 11 }} width={40} />
              <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
              <Line type="monotone" dataKey="vix" stroke="#d6457a" strokeWidth={1.5} dot={false} isAnimationActive={false} name="VIX" />
            </LineChart>
          </ResponsiveContainer>
        </LazyMount>
      </section>

      {/* Per-event detail */}
      <section className="lab-card ab-section">
        <div className="ab-header">
          <h2>逐事件明细（固定窗口）</h2>
          <div style={{ display: 'flex', gap: 6 }}>
            {event_studies.map(s => (
              <button key={s.event_type} onClick={() => setActiveSweep(EVENT_TYPE_LABEL[s.event_type])}
                style={{
                  padding: '4px 12px', borderRadius: 6, fontSize: 12, cursor: 'pointer', border: '1px solid',
                  borderColor: activeSweep === EVENT_TYPE_LABEL[s.event_type] ? EVENT_COLORS[EVENT_TYPE_LABEL[s.event_type]] : '#e5e7eb',
                  background: activeSweep === EVENT_TYPE_LABEL[s.event_type] ? `${EVENT_COLORS[EVENT_TYPE_LABEL[s.event_type]]}15` : '#fff',
                  color: activeSweep === EVENT_TYPE_LABEL[s.event_type] ? EVENT_COLORS[EVENT_TYPE_LABEL[s.event_type]] : '#6b7280',
                  fontWeight: activeSweep === EVENT_TYPE_LABEL[s.event_type] ? 600 : 400,
                }}>
                {EVENT_TYPE_LABEL[s.event_type]} ({s.summary.count})
              </button>
            ))}
          </div>
        </div>
        {study && (
          <>
            <div style={{ marginBottom: 12, fontSize: 12, color: '#6b7280' }}>
              固定窗口：发布前 <strong>{study.best_pre_window}</strong> · 发布后 <strong>{study.best_post_window}</strong>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 14 }}>
              <div style={{ padding: 10, borderRadius: 6, background: '#fff7ed', border: '1px solid #fed7aa', fontSize: 12 }}>
                <div style={{ fontWeight: 600 }}>发布前 {study.best_pre_window}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: pctColor(study.summary.pre.mean) }}>{fmtPct(study.summary.pre.mean)}</div>
                <div style={{ color: '#6b7280' }}>上涨率 {(study.summary.pre.hit_rate! * 100).toFixed(0)}%</div>
              </div>
              <div style={{ padding: 10, borderRadius: 6, background: '#eff6ff', border: '1px solid #bfdbfe', fontSize: 12 }}>
                <div style={{ fontWeight: 600 }}>发布后 {study.best_post_window}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: pctColor(study.summary.post.mean, true) }}>{fmtPct(study.summary.post.mean)}</div>
                <div style={{ color: '#6b7280' }}>下跌率 {(study.summary.post.hit_rate! * 100).toFixed(0)}%</div>
              </div>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="ab-table" style={{ fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>发布日</th><th>交易日</th><th>VIX</th>
                    <th>前 {study.best_pre_window}</th>
                    <th>后 T+1</th><th>后 T+3</th><th>后 T+5</th>
                  </tr>
                </thead>
                <tbody>
                  {study.events.map(e => (
                    <tr key={e.date}>
                      <td>{e.date}</td>
                      <td>{e.trading_day}</td>
                      <td>{e.vix_at_event.toFixed(1)}</td>
                      <td style={{ color: pctColor(e.pre_return_pct) }}>{fmtPct(e.pre_return_pct)}</td>
                      <td style={{ color: pctColor(e.post_1d_return_pct ?? null, true) }}>{fmtPct(e.post_1d_return_pct ?? null)}</td>
                      <td style={{ color: pctColor(e.post_3d_return_pct ?? null, true) }}>{fmtPct(e.post_3d_return_pct ?? null)}</td>
                      <td style={{ color: pctColor(e.post_5d_return_pct ?? null, true) }}>{fmtPct(e.post_5d_return_pct ?? null)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>

      {/* Historical Replication */}
      {data.historical_replication && !data.historical_replication.error && (
        <section className="lab-card">
          <h2 style={{ fontSize: 16, marginBottom: 4 }}>历史复现 · Lucca & Moench (2015)</h2>
          <p style={{ fontSize: 12, color: '#6b7280', marginBottom: 12 }}>
            {data.historical_replication.description}
            <br /><span style={{ color: '#4b5563' }}>VIX 数据范围：{data.historical_replication.vix_data_range} · 总 FOMC 次数：{data.historical_replication.total_fomc_events}</span>
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 14 }}>
            {data.historical_replication.periods.map(p => (
              <div key={p.label} style={{
                background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: '16px 18px',
              }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0', marginBottom: 4 }}>{p.label}</div>
                <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 8 }}>{p.date_range} · n={p.n_events}</div>
                {p.error ? (
                  <div style={{ color: '#ef4444', fontSize: 12 }}>{p.error}</div>
                ) : (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    <div style={{ background: '#1e293b', borderRadius: 6, padding: '8px 10px' }}>
                      <div style={{ fontSize: 10, color: '#6b7280', marginBottom: 2 }}>涨跌方向命中率</div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: (p.return_based?.event_hit_rate ?? 0) > 0.5 ? '#f97316' : '#9ca3af' }}>
                        {p.return_based?.event_hit_rate != null ? `${(p.return_based.event_hit_rate * 100).toFixed(0)}%` : '—'}
                      </div>
                      <div style={{ fontSize: 11, color: '#4b5563' }}>基准 {p.return_based?.baseline_hit_rate != null ? `${(p.return_based.baseline_hit_rate * 100).toFixed(0)}%` : '—'}</div>
                      <div style={{ marginTop: 4 }}>{sigBadge(p.return_based?.p_value ?? null)}</div>
                      <div style={{ fontSize: 10, color: '#4b5563', marginTop: 2 }}>p={p.return_based?.p_value?.toFixed(3) ?? '—'}</div>
                    </div>
                    <div style={{ background: '#1e293b', borderRadius: 6, padding: '8px 10px' }}>
                      <div style={{ fontSize: 10, color: '#6b7280', marginBottom: 2 }}>VIX 水平 (Method A)</div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: (p.level_based?.excess_vix ?? 0) > 0 ? '#f97316' : '#9ca3af' }}>
                        {p.level_based?.excess_vix != null ? `${p.level_based.excess_vix > 0 ? '+' : ''}${p.level_based.excess_vix.toFixed(1)}` : '—'}
                      </div>
                      <div style={{ fontSize: 11, color: '#4b5563' }}>事件均值 vs 基准差异</div>
                      <div style={{ marginTop: 4 }}>{sigBadge(p.level_based?.p_value ?? null)}</div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
          <div style={{
            background: '#0c1a2e', border: '1px solid #1e3a5f', borderRadius: 8, padding: '10px 14px',
            fontSize: 12, color: '#93c5fd',
          }}>
            💡 {data.historical_replication.interpretation}
          </div>
        </section>
      )}

      {/* Conditional Analysis */}
      {data.conditional_analysis && !data.conditional_analysis.error && (
        <section className="lab-card">
          <h2 style={{ fontSize: 16, marginBottom: 4 }}>条件因子分析 · VIX 三档分组</h2>
          <p style={{ fontSize: 12, color: '#6b7280', marginBottom: 12 }}>
            {data.conditional_analysis.description}
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 14 }}>
            {data.conditional_analysis.by_regime.map((r, i) => {
              const colors = ['#22c55e', '#f59e0b', '#ef4444'];
              const color = colors[i] ?? '#6b7280';
              return (
                <div key={r.regime} style={{
                  background: '#0f172a', border: `1px solid ${color}30`,
                  borderRadius: 10, padding: '16px 18px',
                }}>
                  <div style={{ fontSize: 11, color, fontWeight: 600, marginBottom: 6 }}>
                    {['低波动', '中波动', '高波动'][i]}
                  </div>
                  <div style={{ fontSize: 10, color: '#4b5563', marginBottom: 8 }}>
                    {r.vix_range ? `VIX ${r.vix_range[0]}~${r.vix_range[1]}` : ''} · n={r.n}
                  </div>
                  {r.error ? (
                    <div style={{ color: '#6b7280', fontSize: 12 }}>{r.error}</div>
                  ) : (
                    <>
                      <div style={{ fontSize: 28, fontWeight: 800, color }}>
                        {r.hit_rate_up != null ? `${(r.hit_rate_up * 100).toFixed(0)}%` : '—'}
                      </div>
                      <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 4 }}>T-5~T-1 上涨率</div>
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                        {sigBadge(r.p_value)}
                        <span style={{ fontSize: 10, color: '#4b5563' }}>p={r.p_value?.toFixed(3) ?? '—'}</span>
                      </div>
                    </>
                  )}
                </div>
              );
            })}
          </div>
          <div style={{
            background: '#0c1a2e', border: '1px solid #1e3a5f', borderRadius: 8, padding: '10px 14px',
            fontSize: 12, color: '#93c5fd',
          }}>
            💡 {data.conditional_analysis.interpretation}
          </div>
        </section>
      )}

      {/* SKEW Analysis */}
      {data.skew_analysis && !data.skew_analysis.error && (
        <section className="lab-card">
          <h2 style={{ fontSize: 16, marginBottom: 4 }}>隐含波动率偏斜 · CBOE SKEW 指数</h2>
          <p style={{ fontSize: 12, color: '#6b7280', marginBottom: 12 }}>
            {data.skew_analysis.description}
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 14 }}>
            {[
              { label: 'FOMC 前 SKEW 变化', value: data.skew_analysis.skew_stats.event_mean_change, suffix: 'pts', color: (data.skew_analysis.skew_stats.event_mean_change ?? 0) > 0 ? '#f97316' : '#9ca3af' },
              { label: '非事件日基准变化', value: data.skew_analysis.skew_stats.baseline_mean_change, suffix: 'pts', color: '#9ca3af' },
              { label: '超额偏斜变化', value: data.skew_analysis.skew_stats.excess_change, suffix: 'pts', color: (data.skew_analysis.skew_stats.excess_change ?? 0) > 0 ? '#f97316' : '#6b7280' },
              { label: 'SKEW 上升命中率', value: data.skew_analysis.skew_stats.hit_rate_up != null ? data.skew_analysis.skew_stats.hit_rate_up * 100 : null, suffix: '%', color: '#6b7280' },
            ].map(m => (
              <div key={m.label} style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 8, padding: '12px 14px' }}>
                <div style={{ fontSize: 10, color: '#6b7280', marginBottom: 4 }}>{m.label}</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: m.color }}>
                  {m.value != null ? `${m.value > 0 ? '+' : ''}${m.value.toFixed(2)}${m.suffix}` : '—'}
                </div>
              </div>
            ))}
          </div>
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 6 }}>
              p值：{data.skew_analysis.skew_stats.p_value?.toFixed(3) ?? '—'} ·
              {data.skew_analysis.skew_stats.significant
                ? <span style={{ color: '#f97316' }}> SKEW 显著上升</span>
                : <span style={{ color: '#6b7280' }}> SKEW 无显著差异</span>}
              {' '}· n={data.skew_analysis.n_fomc_events} · {data.skew_analysis.data_range}
            </div>
          </div>
          <div style={{
            background: '#0c1a2e', border: '1px solid #1e3a5f', borderRadius: 8, padding: '10px 14px',
            fontSize: 12, color: '#93c5fd',
          }}>
            💡 {data.skew_analysis.interpretation}
          </div>
        </section>
      )}

      {/* Upcoming */}
      <section className="lab-card">
        <h2 style={{ fontSize: 16, marginBottom: 10 }}>即将到来的事件</h2>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {upcoming_events.map(e => (
            <div key={`${e.event}-${e.date}`} style={{
              padding: '8px 14px', borderRadius: 8, fontSize: 13,
              border: `1px solid ${EVENT_COLORS[e.event]}40`, background: `${EVENT_COLORS[e.event]}08`,
            }}>
              <span style={{ fontWeight: 600, color: EVENT_COLORS[e.event] }}>{e.event}</span>
              <span style={{ marginLeft: 8 }}>{e.date}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export function EventVolLab() {
  return <ErrorBoundary><EventVolLabInner /></ErrorBoundary>;
}
