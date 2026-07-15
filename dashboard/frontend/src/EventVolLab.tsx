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
interface EventVolData {
  title: string; subtitle: string;
  hypothesis: { h1: string; h2: string; by_event: HypothesisEvent[] };
  correlation_analysis: CorrelationAnalysis;
  window_sweep: WindowSweepEntry[];
  summary: { data_range: string; instrument: string; total_trading_days: number; method: string };
  event_studies: EventStudy[];
  upcoming_events: { event: string; date: string }[];
  vix_timeline: { date: string; vix: number; daily_return_pct: number | null; event_flags: string[] }[];
}

const EVENT_COLORS: Record<string, string> = { FOMC: '#ea580c', CPI: '#3a82d6', NFP: '#16a34a' };
const EVENT_TYPE_LABEL: Record<string, string> = { fomc: 'FOMC', cpi: 'CPI', nfp: 'NFP' };

function fmtPct(v: number | null | undefined, digits = 2) {
  if (v == null) return '—';
  return `${v > 0 ? '+' : ''}${v.toFixed(digits)}%`;
}

function sigBadge(p: number | null) {
  if (p == null) return null;
  if (p < 0.05) return <span style={{ fontSize: 10, color: '#166534', background: '#dcfce7', padding: '1px 5px', borderRadius: 4 }}>p&lt;0.05</span>;
  if (p < 0.1) return <span style={{ fontSize: 10, color: '#92400e', background: '#fef3c7', padding: '1px 5px', borderRadius: 4 }}>p&lt;0.1</span>;
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

  const { hypothesis, correlation_analysis, window_sweep, summary, event_studies, upcoming_events } = data;
  const sweep = window_sweep.find(s => s.event === activeSweep);
  const study = event_studies.find(s => EVENT_TYPE_LABEL[s.event_type] === activeSweep);

  return (
    <div className="lab-container">
      <header className="lab-header">
        <div>
          <h1>Ch.2.2 Event × VIX</h1>
          <p className="lab-subtitle">{data.subtitle}</p>
        </div>
        <div className="lab-model-badge">
          <span className="lab-badge-version">L3</span>
          <span className="lab-badge-auc">{summary.instrument}</span>
        </div>
      </header>

      {/* Hypotheses */}
      <section className="lab-card">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div style={{ padding: 14, borderRadius: 8, border: '1px solid #fed7aa', background: '#fff7ed' }}>
            <div style={{ fontWeight: 700, color: '#c2410c', marginBottom: 6 }}>H1 · {hypothesis.h1}</div>
            <div style={{ fontSize: 13, color: '#374151' }}>各事件独立扫描多个窗口，选最优</div>
          </div>
          <div style={{ padding: 14, borderRadius: 8, border: '1px solid #bfdbfe', background: '#eff6ff' }}>
            <div style={{ fontWeight: 700, color: '#1d4ed8', marginBottom: 6 }}>H2 · {hypothesis.h2}</div>
            <div style={{ fontSize: 13, color: '#374151' }}>各事件独立扫描多个窗口，选最优</div>
          </div>
        </div>
      </section>

      {/* Baseline reference */}
      <section className="lab-card">
        <h2 style={{ fontSize: 16, marginBottom: 10 }}>非事件日基准（对照组）</h2>
        <p style={{ fontSize: 13, color: '#6b7280', marginBottom: 12 }}>
          {correlation_analysis.method} · 只有和基准对比才能判断事件是否真正相关
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

      {/* Verdict cards */}
      <section className="lab-card">
        <h2 style={{ fontSize: 16, marginBottom: 12 }}>相关性验证（事件 vs 基准）</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
          {hypothesis.by_event.map(e => (
            <div key={e.event} style={{ padding: 14, borderRadius: 8, border: '1px solid #e5e7eb' }}>
              <div style={{ fontWeight: 700, color: EVENT_COLORS[e.event], marginBottom: 10 }}>
                {e.event}
                <div style={{ fontSize: 11, fontWeight: 400, color: '#6b7280' }}>
                  最优窗口：{e.best_pre_window} / {e.best_post_window}
                </div>
              </div>
              <div style={{ fontSize: 13, lineHeight: 2 }}>
                <div>
                  H1 上涨率：{(e.h1_pre_rise.hit_rate_up! * 100).toFixed(0)}% vs 基准 {(e.h1_pre_rise.baseline_hit_rate_up! * 100).toFixed(0)}%
                  <span style={{ marginLeft: 6, fontSize: 11, color: pctColor(e.h1_pre_rise.excess_hit_rate ?? null) }}>
                    ({e.h1_pre_rise.excess_hit_rate! > 0 ? '+' : ''}{(e.h1_pre_rise.excess_hit_rate! * 100).toFixed(0)}pp)
                  </span>
                  <div><ConfirmBadge ok={e.h1_pre_rise.confirmed} label={e.h1_pre_rise.verdict ?? ''} /> {sigBadge(e.h1_pre_rise.p_value ?? null)}</div>
                </div>
                <div>
                  H2 下跌率：{(e.h2_post_fall.hit_rate_down! * 100).toFixed(0)}% vs 基准 {(e.h2_post_fall.baseline_hit_rate_down! * 100).toFixed(0)}%
                  <span style={{ marginLeft: 6, fontSize: 11, color: pctColor(e.h2_post_fall.excess_hit_rate ?? null) }}>
                    ({e.h2_post_fall.excess_hit_rate! > 0 ? '+' : ''}{(e.h2_post_fall.excess_hit_rate! * 100).toFixed(0)}pp)
                  </span>
                  <div><ConfirmBadge ok={e.h2_post_fall.confirmed} label={e.h2_post_fall.verdict ?? ''} /> {sigBadge(e.h2_post_fall.p_value ?? null)}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 12, padding: '10px 12px', background: '#f0fdf4', borderRadius: 8, fontSize: 13, lineHeight: 1.7, border: '1px solid #bbf7d0' }}>
          <strong style={{ color: '#166534' }}>结论（各事件最优窗口）：</strong>
          <strong>FOMC 前 T-3~T-1</strong>显著正相关（上涨率 69% vs 基准 45%，p=0.003）；
          <strong>CPI 前 T-7~T-1</strong>显著正相关（62% vs 46%，p=0.026）；
          NFP 前 T-5~T-1 有差异但不显著（56% vs 46%，p=0.12）。
          三类事件发布后 VIX 下跌率均未显著高于同窗口基准。
        </div>
      </section>

      {/* Window sweep table */}
      <section className="lab-card ab-section">
        <div className="ab-header">
          <h2>时间窗口扫描</h2>
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
          <h2>逐事件明细（最优窗口）</h2>
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
              最优窗口：发布前 <strong>{study.best_pre_window}</strong> · 发布后 <strong>{study.best_post_window}</strong>
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
