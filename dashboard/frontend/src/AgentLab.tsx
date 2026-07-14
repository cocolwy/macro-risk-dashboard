import { useState, useEffect, Component, type ReactNode } from 'react';

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

interface AgentReport {
  timestamp: string;
  risk_score: number;
  risk_level: 'low' | 'moderate' | 'elevated' | 'high' | 'critical' | string;
  ml_probability: number;
  key_risks: string[];
  news_analysis: string;
  recommendation: string;
  reasoning: string;
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
    const base = import.meta.env.BASE_URL || '/';
    const bust = `t=${Date.now()}`;
    Promise.all([
      fetch(`${base}data/agent_report.json?${bust}`).then(r => {
        if (!r.ok) throw new Error(`agent_report.json HTTP ${r.status}`);
        return r.json();
      }),
      fetch(`${base}data/news_headlines.json?${bust}`).then(r => {
        if (!r.ok) throw new Error(`news_headlines.json HTTP ${r.status}`);
        return r.json();
      }),
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
          <h1>Ch.3 News Agent</h1>
          <p className="lab-subtitle">RSS 新闻 + ML 概率 → Claude 宏观风险研报</p>
        </div>
        <div className="lab-model-badge">
          <span className="lab-badge-version">L3</span>
          <span className="lab-badge-auc" style={{ color: level.color }}>
            Score {report.risk_score} · {level.label}
          </span>
        </div>
      </header>

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
            <div className="agent-compare-hint">Claude · news + ML synthesis</div>
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

      {/* Analysis + recommendation + reasoning */}
      <section className="lab-card">
        <div className="ab-header">
          <h2>Agent 分析</h2>
        </div>
        <div className="agent-text-block">
          <h3 className="lab-subsection-title" style={{ marginTop: 0 }}>新闻解读</h3>
          <p className="agent-prose">{report.news_analysis || '—'}</p>
          <h3 className="lab-subsection-title">建议</h3>
          <p className="agent-prose">{report.recommendation || '—'}</p>
          <h3 className="lab-subsection-title">推理过程</h3>
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
