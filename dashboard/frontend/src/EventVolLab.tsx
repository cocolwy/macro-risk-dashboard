import { useState, useEffect, useMemo, Component, type ReactNode } from 'react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell,
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

interface CorrResult { rho: number | null; p_value: number | null; n: number; }
interface FeatureCorr {
  feature: string; event: string; label: string;
  daily_return: CorrResult; return_5d: CorrResult;
}
interface WindowRow {
  feature: string; label: string; days_count: number;
  mean_daily_return_pct: number; mean_5d_return_pct: number;
  baseline_daily_return_pct: number; baseline_5d_return_pct: number;
  excess_daily_return_pct: number; excess_5d_return_pct: number;
  hit_rate_up: number;
}
interface EventRow {
  date: string; uvix_at_event: number;
  pre_5d_return_pct: number | null; pre_3d_return_pct: number | null;
  pre_1d_return_pct: number | null; post_1d_return_pct: number | null;
}
interface EventStudy {
  event_type: string; events: EventRow[];
  summary: {
    count: number;
    pre_5d: { mean: number | null; median: number | null; hit_rate: number | null };
    pre_3d: { mean: number | null; median: number | null; hit_rate: number | null };
    pre_1d: { mean: number | null; median: number | null; hit_rate: number | null };
  };
}
interface TimelinePoint {
  date: string; uvix: number; daily_return_pct: number | null; event_flags: string[];
}
interface EventVolData {
  title: string; subtitle: string;
  summary: {
    data_range: string; uvix_inception: string; total_trading_days: number;
    event_features: string[];
    strongest_correlations: { feature: string; label: string; rho: number; p_value: number }[];
  };
  feature_correlations: FeatureCorr[];
  window_comparison: WindowRow[];
  event_studies: EventStudy[];
  upcoming_events: { event: string; date: string }[];
  uvix_timeline: TimelinePoint[];
  methodology: Record<string, string>;
}

const EVENT_COLORS: Record<string, string> = {
  FOMC: '#ea580c', CPI: '#3a82d6', NFP: '#16a34a',
};
const EVENT_TYPE_LABEL: Record<string, string> = {
  fomc: 'FOMC', cpi: 'CPI', nfp: 'NFP',
};

function fmtPct(v: number | null | undefined, digits = 2) {
  if (v == null) return '—';
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(digits)}%`;
}

function fmtRho(v: number | null) {
  if (v == null) return '—';
  return v.toFixed(4);
}

function sigBadge(p: number | null) {
  if (p == null) return null;
  if (p < 0.05) return <span style={{ fontSize: 10, color: '#166534', background: '#dcfce7', padding: '1px 5px', borderRadius: 4 }}>p&lt;0.05</span>;
  if (p < 0.1) return <span style={{ fontSize: 10, color: '#92400e', background: '#fef3c7', padding: '1px 5px', borderRadius: 4 }}>p&lt;0.1</span>;
  return null;
}

function pctColor(v: number | null) {
  if (v == null) return '#6b7280';
  return v > 0 ? '#dc2626' : v < 0 ? '#16a34a' : '#6b7280';
}

function EventVolLabInner() {
  const [data, setData] = useState<EventVolData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeStudy, setActiveStudy] = useState('fomc');

  useEffect(() => {
    fetchDataJson<EventVolData>('event_uvix_analysis.json')
      .then(d => setData(d))
      .catch(e => setError(e.message));
  }, []);

  const chartTimeline = useMemo(() => {
    if (!data) return [];
    return downsample(data.uvix_timeline.map(p => ({
      date: p.date,
      uvix: p.uvix,
      hasEvent: p.event_flags.length > 0 ? 1 : 0,
      eventLabel: p.event_flags.join('+') || '',
    })));
  }, [data]);

  const windowChartData = useMemo(() => {
    if (!data) return [];
    return data.window_comparison.map(w => ({
      name: w.label,
      excess_5d: w.excess_5d_return_pct,
      hit_rate: w.hit_rate_up * 100,
    }));
  }, [data]);

  if (error) return <div className="lab-container"><div className="lab-card"><p>Event UVIX data not available: {error}</p></div></div>;
  if (!data) return <div className="loading">Loading...</div>;

  const { summary, feature_correlations, window_comparison, event_studies, upcoming_events } = data;
  const study = event_studies.find(s => s.event_type === activeStudy);

  return (
    <div className="lab-container">
      <header className="lab-header">
        <div>
          <h1>Ch.2.2 Event × UVIX</h1>
          <p className="lab-subtitle">{data.subtitle}</p>
        </div>
        <div className="lab-model-badge">
          <span className="lab-badge-version">L3</span>
          <span className="lab-badge-auc">{summary.total_trading_days} 交易日</span>
        </div>
      </header>

      {/* Problem / hypothesis */}
      <section className="lab-card">
        <div className="best-config-card" style={{ marginTop: 0, borderLeftColor: '#ea580c' }}>
          <div className="best-config-header">
            <span className="best-config-badge" style={{ background: '#fff7ed', color: '#c2410c' }}>HYPOTHESIS</span>
            <div className="best-config-title"><span className="best-config-name">事件发布前 UVIX 是否系统性上涨？</span></div>
          </div>
          <div style={{ fontSize: 13, lineHeight: 1.8 }}>
            <p>复用 Ch.2 <strong>+Events</strong> 的 9 个事件日历特征（与 <code>predict_model.build_event_features()</code> 完全一致）：</p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginTop: 8 }}>
              {[
                { ev: 'FOMC', feats: ['days_to', 'days_since', 'within_3d', 'within_7d'] },
                { ev: 'CPI', feats: ['days_to', 'days_since', 'within_3d', 'within_7d'] },
                { ev: 'NFP', feats: ['within_3d'] },
              ].map(g => (
                <div key={g.ev} style={{ padding: '8px 10px', background: '#fff', borderRadius: 6, border: '1px solid #e5e7eb', fontSize: 12 }}>
                  <div style={{ fontWeight: 600, color: EVENT_COLORS[g.ev], marginBottom: 4 }}>{g.ev}</div>
                  {g.feats.map(f => <div key={f} style={{ color: '#6b7280' }}>{g.ev.toLowerCase()}_{f}</div>)}
                </div>
              ))}
            </div>
            <p style={{ marginTop: 10 }}>
              数据范围 {summary.data_range}（UVIX 上市 {summary.uvix_inception}）·
              分析方法：Spearman 相关 + 事件窗口超额收益 + 逐事件 pre-release 回报
            </p>
          </div>
        </div>
      </section>

      {/* Key findings */}
      <section className="lab-card">
        <h2 style={{ fontSize: 16, marginBottom: 12 }}>核心发现</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
          {event_studies.map(s => {
            const label = EVENT_TYPE_LABEL[s.event_type];
            const pre5 = s.summary.pre_5d;
            return (
              <div key={s.event_type} style={{ padding: 14, borderRadius: 8, border: '1px solid #e5e7eb', background: '#fafafa' }}>
                <div style={{ fontWeight: 700, color: EVENT_COLORS[label], marginBottom: 6 }}>{label} ({s.summary.count} 次)</div>
                <div style={{ fontSize: 13, lineHeight: 1.7 }}>
                  <div>发布前 5 日 UVIX 均值：<span style={{ color: pctColor(pre5.mean), fontWeight: 600 }}>{fmtPct(pre5.mean)}</span></div>
                  <div>上涨命中率：<span style={{ fontWeight: 600 }}>{pre5.hit_rate != null ? `${(pre5.hit_rate * 100).toFixed(0)}%` : '—'}</span></div>
                </div>
              </div>
            );
          })}
        </div>
        <div style={{ marginTop: 12, padding: '10px 12px', background: '#fef2f2', borderRadius: 8, fontSize: 13, lineHeight: 1.7, border: '1px solid #fecaca' }}>
          <strong style={{ color: '#991b1b' }}>结论：</strong>
          UVIX 在 FOMC/CPI 发布前<strong>并未</strong>系统性上涨 — 5 日 pre-release 均值均为负，FOMC 命中率仅 21%。
          NFP 前 1 日有微弱正向（均值 +2.9%），但相关性均不显著（|ρ| &lt; 0.06）。
          与 LR Slim+Events 模型增量（F1=0.690）不同，UVIX 价格本身对事件窗口的响应较弱。
        </div>
      </section>

      {/* 9-feature correlation table */}
      <section className="lab-card ab-section">
        <div className="ab-header"><h2>9 特征 × UVIX 相关性</h2></div>
        <div style={{ overflowX: 'auto' }}>
          <table className="ab-table" style={{ fontSize: 13 }}>
            <thead>
              <tr>
                <th>事件</th><th>特征</th><th>标签</th>
                <th>日收益 ρ</th><th>p</th>
                <th>5日收益 ρ</th><th>p</th>
              </tr>
            </thead>
            <tbody>
              {feature_correlations.map(row => (
                <tr key={row.feature}>
                  <td><span style={{ color: EVENT_COLORS[row.event], fontWeight: 600 }}>{row.event}</span></td>
                  <td><code style={{ fontSize: 11 }}>{row.feature}</code></td>
                  <td>{row.label}</td>
                  <td style={{ fontWeight: 600, color: pctColor(row.daily_return.rho) }}>{fmtRho(row.daily_return.rho)}</td>
                  <td>{row.daily_return.p_value?.toFixed(3) ?? '—'} {sigBadge(row.daily_return.p_value)}</td>
                  <td style={{ fontWeight: 600, color: pctColor(row.return_5d.rho) }}>{fmtRho(row.return_5d.rho)}</td>
                  <td>{row.return_5d.p_value?.toFixed(3) ?? '—'} {sigBadge(row.return_5d.p_value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Window comparison chart */}
      <section className="lab-card">
        <h2 style={{ fontSize: 16, marginBottom: 4 }}>事件窗口超额收益（vs 全样本基准）</h2>
        <p style={{ fontSize: 12, color: '#6b7280', marginBottom: 12 }}>within_3d / within_7d 窗口内 UVIX 5 日收益 − 全样本 5 日均值</p>
        <LazyMount minHeight={260}>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={windowChartData} margin={{ top: 5, right: 20, bottom: 5, left: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1d8e2" />
              <XAxis dataKey="name" tick={{ fill: '#8a7882', fontSize: 11 }} />
              <YAxis tick={{ fill: '#8a7882', fontSize: 11 }} unit="%" />
              <Tooltip contentStyle={CHART_TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(2)}%`, '超额5日收益']} />
              <ReferenceLine y={0} stroke="#9ca3af" strokeDasharray="4 4" />
              <Bar dataKey="excess_5d" name="超额5日收益" radius={[4, 4, 0, 0]}>
                {windowChartData.map((entry, i) => (
                  <Cell key={i} fill={entry.excess_5d >= 0 ? '#dc2626' : '#16a34a'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </LazyMount>
        <div style={{ overflowX: 'auto', marginTop: 12 }}>
          <table className="ab-table" style={{ fontSize: 12 }}>
            <thead>
              <tr>
                <th>窗口</th><th>天数</th><th>日均收益</th><th>5日收益</th>
                <th>超额5日</th><th>上涨命中率</th>
              </tr>
            </thead>
            <tbody>
              {window_comparison.map(w => (
                <tr key={w.feature}>
                  <td>{w.label}</td>
                  <td>{w.days_count}</td>
                  <td style={{ color: pctColor(w.mean_daily_return_pct) }}>{fmtPct(w.mean_daily_return_pct, 3)}</td>
                  <td style={{ color: pctColor(w.mean_5d_return_pct) }}>{fmtPct(w.mean_5d_return_pct, 3)}</td>
                  <td style={{ fontWeight: 600, color: pctColor(w.excess_5d_return_pct) }}>{fmtPct(w.excess_5d_return_pct, 3)}</td>
                  <td>{(w.hit_rate_up * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* UVIX timeline */}
      <section className="lab-card">
        <h2 style={{ fontSize: 16, marginBottom: 4 }}>UVIX 走势 + 事件窗口标记</h2>
        <p style={{ fontSize: 12, color: '#6b7280', marginBottom: 8 }}>within_3d 窗口内的交易日高亮</p>
        <LazyMount minHeight={240}>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={chartTimeline} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1d8e2" />
              <XAxis dataKey="date" tick={{ fill: '#8a7882', fontSize: 10 }} tickFormatter={(d: string) => d.slice(0, 7)} minTickGap={80} />
              <YAxis tick={{ fill: '#8a7882', fontSize: 11 }} width={55} />
              <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
              <Line type="monotone" dataKey="uvix" stroke="#d6457a" strokeWidth={1.5} dot={false} isAnimationActive={false} name="UVIX" />
            </LineChart>
          </ResponsiveContainer>
        </LazyMount>
      </section>

      {/* Per-event study */}
      <section className="lab-card ab-section">
        <div className="ab-header">
          <h2>逐事件 Pre-Release 回报</h2>
          <div style={{ display: 'flex', gap: 6 }}>
            {event_studies.map(s => (
              <button key={s.event_type}
                onClick={() => setActiveStudy(s.event_type)}
                style={{
                  padding: '4px 12px', borderRadius: 6, fontSize: 12, cursor: 'pointer', border: '1px solid',
                  borderColor: activeStudy === s.event_type ? EVENT_COLORS[EVENT_TYPE_LABEL[s.event_type]] : '#e5e7eb',
                  background: activeStudy === s.event_type ? `${EVENT_COLORS[EVENT_TYPE_LABEL[s.event_type]]}15` : '#fff',
                  color: activeStudy === s.event_type ? EVENT_COLORS[EVENT_TYPE_LABEL[s.event_type]] : '#6b7280',
                  fontWeight: activeStudy === s.event_type ? 600 : 400,
                }}>
                {EVENT_TYPE_LABEL[s.event_type]}
              </button>
            ))}
          </div>
        </div>
        {study && (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 14 }}>
              {(['pre_5d', 'pre_3d', 'pre_1d'] as const).map(key => {
                const s = study.summary[key];
                const labels = { pre_5d: 'T-5~T-1', pre_3d: 'T-3~T-1', pre_1d: 'T-1' };
                return (
                  <div key={key} style={{ padding: 10, borderRadius: 6, background: '#f9fafb', border: '1px solid #e5e7eb', fontSize: 12 }}>
                    <div style={{ fontWeight: 600, marginBottom: 4 }}>{labels[key]} 均值</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: pctColor(s.mean) }}>{fmtPct(s.mean)}</div>
                    <div style={{ color: '#6b7280' }}>中位数 {fmtPct(s.median)} · 命中率 {s.hit_rate != null ? `${(s.hit_rate * 100).toFixed(0)}%` : '—'}</div>
                  </div>
                );
              })}
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="ab-table" style={{ fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>发布日</th><th>UVIX</th>
                    <th>前5日</th><th>前3日</th><th>前1日</th><th>后1日</th>
                  </tr>
                </thead>
                <tbody>
                  {study.events.map(e => (
                    <tr key={e.date}>
                      <td>{e.date}</td>
                      <td>{e.uvix_at_event.toFixed(1)}</td>
                      <td style={{ color: pctColor(e.pre_5d_return_pct) }}>{fmtPct(e.pre_5d_return_pct)}</td>
                      <td style={{ color: pctColor(e.pre_3d_return_pct) }}>{fmtPct(e.pre_3d_return_pct)}</td>
                      <td style={{ color: pctColor(e.pre_1d_return_pct) }}>{fmtPct(e.pre_1d_return_pct)}</td>
                      <td style={{ color: pctColor(e.post_1d_return_pct) }}>{fmtPct(e.post_1d_return_pct)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>

      {/* Upcoming events */}
      <section className="lab-card">
        <h2 style={{ fontSize: 16, marginBottom: 10 }}>即将到来的事件</h2>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {upcoming_events.map(e => (
            <div key={`${e.event}-${e.date}`} style={{
              padding: '8px 14px', borderRadius: 8, fontSize: 13,
              border: `1px solid ${EVENT_COLORS[e.event]}40`,
              background: `${EVENT_COLORS[e.event]}08`,
            }}>
              <span style={{ fontWeight: 600, color: EVENT_COLORS[e.event] }}>{e.event}</span>
              <span style={{ marginLeft: 8, color: '#374151' }}>{e.date}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export function EventVolLab() {
  return (
    <ErrorBoundary>
      <EventVolLabInner />
    </ErrorBoundary>
  );
}
