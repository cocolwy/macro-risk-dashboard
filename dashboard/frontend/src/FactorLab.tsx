import { useState, useEffect } from 'react';
import { ResearchTrackNotice } from './components/ResearchTrackNotice';

interface FactorMethod {
  name: string;
  description: string;
  result: string;
}

interface FactorReference {
  title: string;
  authors: string;
  journal: string;
  year: number;
  note: string;
}

interface FactorData {
  instrument: string;
  source: string;
  primary_period: string;
  n_trading_days: number;
  events?: Record<string, number>;
}

interface PipelineStage {
  id: string;
  name: string;
  status: 'pending' | 'in_progress' | 'done' | 'blocked' | 'skipped';
  summary?: string;
}

interface FactorGoal {
  trading_objective: string;
  research_instrument: string;
  tradable_instrument: string;
  mapping_gap: string;
  horizon: string;
}

interface Factor {
  id: string;
  name: string;
  name_zh: string;
  hypothesis: string;
  motivation: string;
  status: 'confirmed' | 'conditional' | 'weak' | 'dead' | 'pending';
  status_note: string;
  current_stage?: string;
  case_file?: string;
  experiment_hash?: string;
  tags: string[];
  goal?: FactorGoal;
  pipeline_stages?: PipelineStage[];
  data: FactorData;
  methods: FactorMethod[];
  key_findings: string[];
  verdict: string;
  next_steps: string[];
  references: FactorReference[];
  created: string;
  last_updated: string;
}

interface FactorLog {
  version: string;
  description: string;
  updated: string;
  pipeline?: {
    doc: string;
    readme: string;
    stages: string[];
    stage_names: Record<string, string>;
  };
  factors: Factor[];
}

const STATUS_CONFIG: Record<string, { label: string; accent: string; className: string; icon: string }> = {
  confirmed:   { label: 'Confirmed',   accent: '#16a34a', className: 'deck-status-confirmed',   icon: '✓' },
  conditional: { label: 'Conditional', accent: '#b45309', className: 'deck-status-conditional', icon: '≈' },
  weak:        { label: 'Weak',        accent: '#ea580c', className: 'deck-status-weak',        icon: '~' },
  dead:        { label: 'Dead',        accent: '#dc2626', className: 'deck-status-dead',        icon: '✗' },
  pending:     { label: 'Pending',     accent: '#8a7882', className: 'deck-status-pending',     icon: '?' },
};

const STAGE_STYLE: Record<string, { color: string; bg: string; border: string }> = {
  done:        { color: '#16a34a', bg: '#f0fdf4', border: '#bbf7d0' },
  in_progress: { color: '#b45309', bg: '#fffbeb', border: '#fde68a' },
  pending:     { color: '#8a7882', bg: '#fdfbfc', border: '#f1d8e2' },
  blocked:     { color: '#dc2626', bg: '#fef2f2', border: '#fecaca' },
  skipped:     { color: '#8a7882', bg: '#f9fafb', border: '#e5e7eb' },
};

function StatusBadge({ status }: { status: string }) {
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.pending;
  return (
    <span className={`deck-status-badge ${cfg.className}`}>
      {cfg.icon} {cfg.label}
    </span>
  );
}

function TagPill({ tag }: { tag: string }) {
  return <span className="deck-tag">{tag}</span>;
}

function PipelineStrip({ stages, compact = false }: { stages: PipelineStage[]; compact?: boolean }) {
  return (
    <div className={`deck-pipeline-strip${compact ? ' compact' : ''}`}>
      {stages.map(s => {
        const st = STAGE_STYLE[s.status] ?? STAGE_STYLE.pending;
        return (
          <div
            key={s.id}
            className="deck-pipeline-step"
            title={`${s.id} ${s.name}: ${s.status}${s.summary ? ` — ${s.summary}` : ''}`}
            style={{ background: st.bg, borderColor: st.border }}
          >
            <div className="deck-pipeline-id" style={{ color: st.color }}>{s.id}</div>
            {!compact && (
              <>
                <div className="deck-pipeline-name">{s.name}</div>
                <div className="deck-pipeline-status">{s.status}</div>
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}

function MethodRow({ m, idx }: { m: FactorMethod; idx: number }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="deck-method-row">
      <button type="button" className="deck-method-btn" onClick={() => setOpen(o => !o)}>
        <span style={{ color: 'var(--text-3)', fontSize: 11, minWidth: 24 }}>{String(idx + 1).padStart(2, '0')}</span>
        <span className="deck-method-name">{m.name}</span>
        <span style={{ marginLeft: 'auto', color: 'var(--text-3)', fontSize: 14 }}>{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <>
          <p className="lab-card-desc" style={{ marginLeft: 32, marginBottom: 6 }}>{m.description}</p>
          <div className="deck-method-result">
            <span style={{ color: 'var(--text-3)', marginRight: 6 }}>→</span>
            {m.result}
          </div>
        </>
      )}
    </div>
  );
}

function FactorCard({ factor, onSelect }: { factor: Factor; onSelect: () => void }) {
  const cfg = STATUS_CONFIG[factor.status] ?? STATUS_CONFIG.pending;
  return (
    <div
      className="deck-factor-card"
      style={{ ['--status-accent' as string]: cfg.accent }}
      onClick={onSelect}
    >
      <div className="deck-factor-head">
        <span className="deck-factor-id">{factor.id}</span>
        <div style={{ flex: 1 }}>
          <div className="deck-factor-title">{factor.name}</div>
          <div className="deck-factor-subtitle">{factor.name_zh}</div>
        </div>
        <StatusBadge status={factor.status} />
      </div>
      {factor.goal && (
        <div className="deck-goal-snippet">
          Goal: {factor.goal.tradable_instrument} · {factor.goal.trading_objective}
        </div>
      )}
      <p className="deck-hypothesis">{factor.hypothesis}</p>
      {factor.pipeline_stages && <PipelineStrip stages={factor.pipeline_stages} compact />}
      <div className="deck-tags" style={{ marginTop: 12 }}>
        {factor.tags.map(t => <TagPill key={t} tag={t} />)}
      </div>
      <div className="deck-meta">
        {factor.current_stage ? `Stage ${factor.current_stage} · ` : ''}
        {factor.data.primary_period} · {factor.data.n_trading_days.toLocaleString()} trading days
      </div>
    </div>
  );
}

function FactorDetail({ factor, onBack }: { factor: Factor; onBack: () => void }) {
  return (
    <div>
      <button type="button" className="deck-back-btn" onClick={onBack}>← Back</button>

      <section className="lab-card">
        <div className="deck-factor-head">
          <span className="deck-factor-id">{factor.id}</span>
          <div style={{ flex: 1 }}>
            <h2 style={{ margin: 0, fontSize: 22 }}>{factor.name}</h2>
            <p className="lab-card-desc" style={{ marginBottom: 0 }}>{factor.name_zh}</p>
          </div>
          <StatusBadge status={factor.status} />
        </div>
        <p style={{ fontSize: 13, color: 'var(--text-2)', lineHeight: 1.7, marginTop: 14 }}>
          <strong style={{ color: 'var(--text-3)' }}>Hypothesis: </strong>{factor.hypothesis}
        </p>
        <p style={{ fontSize: 13, color: 'var(--text-2)', lineHeight: 1.7 }}>
          <strong style={{ color: 'var(--text-3)' }}>Motivation: </strong>{factor.motivation}
        </p>
        <p className="deck-meta" style={{ marginTop: 12 }}>
          {factor.data.instrument} · {factor.data.source} · {factor.data.primary_period} · {factor.data.n_trading_days.toLocaleString()} days
          {factor.data.events && (
            <span> · FOMC×{factor.data.events.fomc} / CPI×{factor.data.events.cpi} / NFP×{factor.data.events.nfp}</span>
          )}
        </p>
        <div className="deck-tags" style={{ marginTop: 10 }}>
          {factor.tags.map(t => <TagPill key={t} tag={t} />)}
        </div>
        {(factor.case_file || factor.experiment_hash) && (
          <p className="deck-meta" style={{ marginTop: 10 }}>
            {factor.case_file && <>Case: <code>{factor.case_file}</code></>}
            {factor.experiment_hash && (
              <a href={factor.experiment_hash} style={{ marginLeft: 12, color: 'var(--pink-500)' }}>
                实验页 {factor.experiment_hash}
              </a>
            )}
          </p>
        )}
      </section>

      {factor.goal && (
        <section className="lab-card deck-goal-card">
          <h2>S0 · Trading Goal</h2>
          <p style={{ fontSize: 13, color: 'var(--text-1)', lineHeight: 1.7, marginBottom: 8 }}>
            {factor.goal.trading_objective}
          </p>
          <div className="lab-info-grid">
            <div><span className="lab-info-key">Research</span><span>{factor.goal.research_instrument}</span></div>
            <div><span className="lab-info-key">Tradable</span><span>{factor.goal.tradable_instrument}</span></div>
          </div>
          <p style={{ fontSize: 12, color: 'var(--accent-yellow)', marginTop: 10, lineHeight: 1.6 }}>
            Mapping gap: {factor.goal.mapping_gap}
          </p>
          <p className="deck-meta">Horizon: {factor.goal.horizon}</p>
        </section>
      )}

      {factor.pipeline_stages && (
        <section className="lab-card">
          <h2>
            Pipeline S0–S7
            {factor.current_stage && (
              <span style={{ color: 'var(--accent-yellow)', fontWeight: 400, fontSize: 13, marginLeft: 8 }}>
                current: {factor.current_stage}
              </span>
            )}
          </h2>
          <p className="lab-card-desc">协议见仓库 factor_pipeline/PIPELINE.md</p>
          <PipelineStrip stages={factor.pipeline_stages} />
          <div style={{ marginTop: 16 }}>
            {factor.pipeline_stages.map(s => (
              <div key={s.id} className="deck-pipeline-row">
                <span className="deck-pipeline-row-id" style={{ color: STAGE_STYLE[s.status]?.color }}>
                  {s.id}
                </span>
                <span className="deck-pipeline-row-name">{s.name}</span>
                <span className="deck-pipeline-row-summary">{s.summary ?? '—'}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      <div className="deck-two-col">
        <section className="lab-card" style={{ marginBottom: 0 }}>
          <h2>Key Findings</h2>
          <ul className="deck-findings" style={{ margin: 0, paddingLeft: 16 }}>
            {factor.key_findings.map((f, i) => <li key={i}>{f}</li>)}
          </ul>
        </section>
        <section className="lab-card" style={{ marginBottom: 0 }}>
          <h2>Verdict</h2>
          <div className="deck-verdict-box">{factor.verdict}</div>
          <h2 style={{ fontSize: 15, marginBottom: 8 }}>Next Steps</h2>
          <ul className="deck-next-steps" style={{ margin: 0, paddingLeft: 16 }}>
            {factor.next_steps.map((s, i) => <li key={i}>{s}</li>)}
          </ul>
        </section>
      </div>

      <section className="lab-card">
        <h2>Methods Tried <span style={{ color: 'var(--text-3)', fontWeight: 400, fontSize: 13 }}>({factor.methods.length})</span></h2>
        {factor.methods.map((m, i) => <MethodRow key={m.name} m={m} idx={i} />)}
      </section>

      <div className="deck-note-box">
        <strong>Status Note: </strong>{factor.status_note}
      </div>

      {factor.references.length > 0 && (
        <section className="lab-card">
          <h2>References</h2>
          {factor.references.map((r, i) => (
            <div key={i} className="deck-ref-item">
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)' }}>{r.title} ({r.year})</div>
              <div style={{ fontSize: 12, color: 'var(--text-3)', marginBottom: 4 }}>{r.authors} · {r.journal}</div>
              <div style={{ fontSize: 12, color: 'var(--text-2)', fontStyle: 'italic' }}>{r.note}</div>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}

export function FactorLab() {
  const [log, setLog] = useState<FactorLog | null>(null);
  const [selected, setSelected] = useState<Factor | null>(null);
  const [filter, setFilter] = useState<string>('all');

  useEffect(() => {
    const base = import.meta.env.BASE_URL ?? '/';
    fetch(`${base}data/factor_research_log.json`)
      .then(r => r.json())
      .then(setLog)
      .catch(() => {});
  }, []);

  if (!log) {
    return <div className="lab-loading">Loading factor log…</div>;
  }

  if (selected) {
    return (
      <div className="lab-container">
        <ResearchTrackNotice track="alpha" />
        <FactorDetail factor={selected} onBack={() => setSelected(null)} />
      </div>
    );
  }

  const statuses = ['all', ...Array.from(new Set(log.factors.map(f => f.status)))];
  const visible = filter === 'all' ? log.factors : log.factors.filter(f => f.status === filter);
  const counts = Object.fromEntries(
    Object.keys(STATUS_CONFIG).map(s => [s, log.factors.filter(f => f.status === s).length])
  );

  return (
    <div className="lab-container">
      <header className="deck-header">
        <div className="deck-kicker">ALPHA DECK</div>
        <div className="lab-header" style={{ marginBottom: 0 }}>
          <div>
            <h1>Factor Research Log</h1>
            <p className="lab-subtitle">S0–S7 流水线记录 · 假设 · 检验 · 可交易性 · 结论</p>
          </div>
        </div>
        {log.pipeline && (
          <div className="deck-protocol">
            流水线协议：仓库内 <code>{log.pipeline.doc}</code>
            {' · '}用法 <code>{log.pipeline.readme}</code>
            {' · '}阶段 {log.pipeline.stages.join(' → ')}
          </div>
        )}
      </header>

      <ResearchTrackNotice track="alpha" />

      <div className="deck-stats">
        {Object.entries(STATUS_CONFIG).map(([status, cfg]) => (
          <div key={status} className="deck-stat">
            <div className="deck-stat-num" style={{ color: cfg.accent }}>{counts[status] ?? 0}</div>
            <div className="deck-stat-label">{cfg.label}</div>
          </div>
        ))}
      </div>

      <div className="deck-filters">
        {statuses.map(s => (
          <button
            key={s}
            type="button"
            className={`deck-filter-btn${filter === s ? ' active' : ''}`}
            onClick={() => setFilter(s)}
          >
            {s === 'all' ? `All (${log.factors.length})` : `${STATUS_CONFIG[s]?.label ?? s} (${counts[s] ?? 0})`}
          </button>
        ))}
      </div>

      <div className="deck-factor-list">
        {visible.map(f => (
          <FactorCard key={f.id} factor={f} onSelect={() => setSelected(f)} />
        ))}
        {visible.length === 0 && (
          <div className="lab-no-data">No factors with this status yet.</div>
        )}
      </div>

      <div className="deck-footer">
        Alpha Deck v{log.version} · Last updated {log.updated} · {log.factors.length} factors
      </div>
    </div>
  );
}
