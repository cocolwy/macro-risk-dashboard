import { useState, useEffect, Component, type ReactNode } from 'react';
import { fetchDataJson } from './api';

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

interface Headline {
  title: string;
  source: string;
  published: string;
  url: string;
}

interface NewsData {
  fetched_at: string;
  headlines: Headline[];
}

interface ThemeEntry {
  summary: string;
  signal: 'risk-off' | 'neutral' | 'risk-on' | string;
}

interface AgentReport {
  timestamp: string;
  risk_score: number;
  risk_level: 'low' | 'moderate' | 'elevated' | 'high' | 'critical' | string;
  ml_probability: number;
  signal_basis?: string;
  themes?: {
    rates_fed?: ThemeEntry;
    geopolitics_energy?: ThemeEntry;
    equities_earnings?: ThemeEntry;
    credit_liquidity?: ThemeEntry;
  };
  key_risks: string[];
  news_analysis?: string;
  recommendation: string;
  reasoning: string;
  provider?: string;
}

const LEVEL_META: Record<string, { label: string; color: string; bg: string; border: string }> = {
  low: { label: 'LOW', color: '#16a34a', bg: '#f0fdf4', border: '#86efac' },
  moderate: { label: 'MODERATE', color: '#2563eb', bg: '#eff6ff', border: '#93c5fd' },
  elevated: { label: 'ELEVATED', color: '#d97706', bg: '#fffbeb', border: '#fcd34d' },
  high: { label: 'HIGH', color: '#ea580c', bg: '#fff7ed', border: '#fdba74' },
  critical: { label: 'CRITICAL', color: '#dc2626', bg: '#fef2f2', border: '#fca5a5' },
};

const SOURCE_COLORS: Record<string, string> = {
  Reuters: '#b45309',
  CNBC: '#1d4ed8',
  MarketWatch: '#0f766e',
};

const SIGNAL_META: Record<string, { label: string; color: string; bg: string; border: string }> = {
  'risk-off': { label: '风险规避', color: '#dc2626', bg: '#fef2f2', border: '#fca5a5' },
  'neutral':  { label: '中性',     color: '#6b7280', bg: '#f9fafb', border: '#d1d5db' },
  'risk-on':  { label: '风险偏好', color: '#16a34a', bg: '#f0fdf4', border: '#86efac' },
};

const THEME_CONFIG = [
  { key: 'rates_fed',           icon: '🏦', title: '央行 / 利率',   en: 'Rates & Fed' },
  { key: 'geopolitics_energy',  icon: '🌍', title: '地缘 / 能源',   en: 'Geopolitics & Energy' },
  { key: 'equities_earnings',   icon: '📈', title: '股市 / 财报',   en: 'Equities & Earnings' },
  { key: 'credit_liquidity',    icon: '💧', title: '信用 / 流动性', en: 'Credit & Liquidity' },
] as const;

function SignalPill({ signal }: { signal: string }) {
  const meta = SIGNAL_META[signal] ?? SIGNAL_META.neutral;
  return (
    <span style={{
      fontSize: 11, fontWeight: 600, letterSpacing: '0.04em', padding: '2px 8px',
      borderRadius: 10, color: meta.color, background: meta.bg,
      border: `1px solid ${meta.border}`, whiteSpace: 'nowrap',
    }}>
      {meta.label}
    </span>
  );
}

function formatTime(iso: string | undefined): string {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString('zh-CN', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', timeZoneName: 'short',
    });
  } catch {
    return iso;
  }
}

function AgentLabInner() {
  const [report, setReport] = useState<AgentReport | null>(null);
  const [news, setNews] = useState<NewsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetchDataJson<AgentReport>('agent_report.json'),
      fetchDataJson<NewsData>('news_headlines.json'),
    ])
      .then(([rep, newsData]) => {
        setReport(rep);
        setNews(newsData);
      })
      .catch(e => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="lab-container">
        <div className="lab-card">
          <p>Agent Lab data not available: {error}</p>
          <p className="lab-card-desc">
            需先运行 <code>news_fetcher.py</code> 与 <code>agent_pipeline.py</code> 生成数据。
          </p>
        </div>
      </div>
    );
  }
  if (!report || !news) return <div className="loading">Loading...</div>;

  const level = LEVEL_META[report.risk_level] ?? LEVEL_META.elevated;
  const mlPct = report.ml_probability * 100;
  const agentAsPct = report.risk_score;

  return (
    <div className="lab-container">
      <header className="lab-header">
        <div>
          <h1>News Agent</h1>
          <p className="lab-subtitle">RSS 新闻 + ML 概率 → Claude / OpenAI / 本地规则引擎研报</p>
        </div>
        <div className="lab-model-badge">
          <span className="lab-badge-version">
            L1 · {(() => {
              const p = report.provider || 'local';
              if (p.startsWith('anthropic') || p.includes('claude')) return 'Claude';
              if (p.startsWith('openai')) return 'OpenAI';
              return p;
            })()}
          </span>
          <span className="lab-badge-auc" style={{ color: level.color }}>
            Score {report.risk_score} · {level.label}
          </span>
        </div>
      </header>

      {report.provider && !report.provider.startsWith('anthropic') && !report.provider.startsWith('openai') && (
        <div className="lab-signal-note" style={{ marginBottom: 16, color: '#b45309' }}>
          当前研报来源：{report.provider}（不是 Claude/ChatGPT）。若已配置 API Key，请查看 Actions 日志中的 key length / 报错信息。
        </div>
      )}

      {/* Risk score gauge */}
      <section className="lab-signal-card" style={{ borderColor: level.color }}>
        <div className="lab-signal-main">
          <div className="lab-signal-prob" style={{ color: level.color }}>
            {report.risk_score}
          </div>
          <div className="lab-signal-meta">
            <div className="lab-signal-label">
              Agent 综合风险评分
              <span
                className="agent-level-pill"
                style={{ color: level.color, background: level.bg, borderColor: level.border }}
              >
                {level.label}
              </span>
            </div>
            <div className="lab-signal-date">更新于 {formatTime(report.timestamp)}</div>
            <div className="lab-signal-note">
              0–100 量表 · 融合 ML 大跌概率与当日财经新闻语义
            </div>
          </div>
        </div>
      </section>

      {/* ML vs Agent comparison */}
      <section className="lab-card">
        <div className="ab-header">
          <h2>ML 概率 vs Agent 评分</h2>
          <span className="ab-badge" style={{ background: '#eff6ff', color: '#1e40af', border: '1px solid #bfdbfe' }}>
            COMPARISON
          </span>
        </div>
        <p className="lab-card-desc">
          ML 输出为未来 20 日出现 &gt;5% 回撤的概率；Agent 评分是在新闻语境下的 0–100 综合风险分。
        </p>
        <div className="agent-compare-grid">
          <div className="agent-compare-card">
            <div className="agent-compare-label">ML Crash Probability</div>
            <div className="agent-compare-value" style={{ color: '#d6457a' }}>{mlPct.toFixed(1)}%</div>
            <div className="agent-compare-bar">
              <div className="agent-compare-fill" style={{ width: `${Math.min(100, mlPct)}%`, background: '#d6457a' }} />
            </div>
            <div className="agent-compare-hint">model_metrics.json · current_prediction</div>
          </div>
          <div className="agent-compare-card">
            <div className="agent-compare-label">Agent Risk Score</div>
            <div className="agent-compare-value" style={{ color: level.color }}>{agentAsPct}</div>
            <div className="agent-compare-bar">
              <div className="agent-compare-fill" style={{ width: `${agentAsPct}%`, background: level.color }} />
            </div>
            <div className="agent-compare-hint">Claude / LLM · news + ML synthesis</div>
          </div>
        </div>
      </section>

      {/* Key risks */}
      {report.key_risks?.length > 0 && (
        <section className="lab-card">
          <div className="ab-header">
            <h2>关键风险</h2>
            <span className="ab-badge">{report.key_risks.length} items</span>
          </div>
          <ul className="agent-risk-list">
            {report.key_risks.map((risk, i) => (
              <li key={i}>{risk}</li>
            ))}
          </ul>
        </section>
      )}

      {/* Themes (structured analysis) */}
      {report.themes ? (
        <section className="lab-card">
          <div className="ab-header">
            <h2>分主题解读</h2>
            {report.signal_basis && (
              <span className="ab-badge" style={{ background: '#f5f3ff', color: '#6d28d9', border: '1px solid #c4b5fd' }}>
                {report.signal_basis}
              </span>
            )}
          </div>
          <p className="lab-card-desc">
            四维度独立分析 · 每维度附信号方向（风险偏好 / 中性 / 风险规避）
          </p>
          <div className="agent-themes-grid">
            {THEME_CONFIG.map(({ key, icon, title }) => {
              const theme = report.themes?.[key as keyof typeof report.themes];
              if (!theme) return null;
              return (
                <div key={key} className="agent-theme-card">
                  <div className="agent-theme-header">
                    <span className="agent-theme-icon">{icon}</span>
                    <span className="agent-theme-title">{title}</span>
                    <SignalPill signal={theme.signal} />
                  </div>
                  <p className="agent-prose" style={{ marginTop: 8, marginBottom: 0 }}>
                    {theme.summary || '今日无明显相关信号。'}
                  </p>
                </div>
              );
            })}
          </div>
        </section>
      ) : report.news_analysis ? (
        /* Backward compat: old single-block format */
        <section className="lab-card">
          <div className="ab-header"><h2>新闻解读</h2></div>
          <p className="agent-prose">{report.news_analysis}</p>
        </section>
      ) : null}

      {/* Recommendation + reasoning */}
      <section className="lab-card">
        <div className="ab-header">
          <h2>建议 &amp; 推理</h2>
        </div>
        <div className="agent-text-block">
          <h3 className="lab-subsection-title" style={{ marginTop: 0 }}>操作建议</h3>
          <p className="agent-prose">{report.recommendation || '—'}</p>
          <h3 className="lab-subsection-title">评分推理</h3>
          <p className="agent-prose agent-reasoning">{report.reasoning || '—'}</p>
        </div>
      </section>

      {/* Headlines */}
      <section className="lab-card">
        <div className="ab-header">
          <h2>今日财经新闻</h2>
          <span className="ab-badge">{news.headlines.length} headlines</span>
        </div>
        <p className="lab-card-desc">
          抓取时间 {formatTime(news.fetched_at)} · Reuters / CNBC / MarketWatch RSS
        </p>
        {news.headlines.length === 0 ? (
          <p style={{ color: '#8a7882', fontSize: 13 }}>暂无标题</p>
        ) : (
          <ul className="agent-headline-list">
            {news.headlines.map((h, i) => (
              <li key={i} className="agent-headline-item">
                <span
                  className="agent-source-tag"
                  style={{ color: SOURCE_COLORS[h.source] || '#5c4f56' }}
                >
                  {h.source}
                </span>
                {h.url ? (
                  <a href={h.url} target="_blank" rel="noopener noreferrer" className="agent-headline-title">
                    {h.title}
                  </a>
                ) : (
                  <span className="agent-headline-title">{h.title}</span>
                )}
                <span className="agent-headline-time">{formatTime(h.published)}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

export function AgentLab() {
  return (
    <ErrorBoundary>
      <AgentLabInner />
    </ErrorBoundary>
  );
}
