import { useState, useEffect } from 'react';

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

const STATUS_CONFIG: Record<string, { label: string; color: string; bg: string; icon: string }> = {
  confirmed: { label: 'Confirmed', color: '#10b981', bg: '#064e3b', icon: '✓' },
  conditional: { label: 'Conditional', color: '#f59e0b', bg: '#451a03', icon: '≈' },
  weak:        { label: 'Weak',        color: '#f97316', bg: '#431407', icon: '~' },
  dead:        { label: 'Dead',        color: '#ef4444', bg: '#450a0a', icon: '✗' },
  pending:     { label: 'Pending',     color: '#6b7280', bg: '#111827', icon: '?' },
};

const STAGE_COLOR: Record<string, string> = {
  done: '#10b981',
  in_progress: '#f59e0b',
  pending: '#4b5563',
  blocked: '#ef4444',
  skipped: '#374151',
};

function StatusBadge({ status }: { status: string }) {
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.pending;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '2px 10px', borderRadius: 12,
      fontSize: 12, fontWeight: 600,
      color: cfg.color, background: cfg.bg,
      border: `1px solid ${cfg.color}40`,
    }}>
      {cfg.icon} {cfg.label}
    </span>
  );
}

function TagPill({ tag }: { tag: string }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 8,
      fontSize: 11, background: '#1e3a5f', color: '#93c5fd', border: '1px solid #1d4ed8',
    }}>
      {tag}
    </span>
  );
}

function PipelineStrip({ stages, compact = false }: { stages: PipelineStage[]; compact?: boolean }) {
  return (
    <div style={{ display: 'flex', gap: 4, flexWrap: compact ? 'nowrap' : 'wrap', overflowX: compact ? 'auto' : undefined }}>
      {stages.map((s, i) => {
        const color = STAGE_COLOR[s.status] ?? STAGE_COLOR.pending;
        return (
          <div
            key={s.id}
            title={`${s.id} ${s.name}: ${s.status}${s.summary ? ` — ${s.summary}` : ''}`}
            style={{
              flex: compact ? '0 0 auto' : 1,
              minWidth: compact ? 44 : 0,
              background: `${color}18`,
              border: `1px solid ${color}55`,
              borderRadius: 6,
              padding: compact ? '4px 6px' : '8px 6px',
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: 10, fontFamily: 'monospace', color, fontWeight: 700 }}>{s.id}</div>
            {!compact && (
              <>
                <div style={{ fontSize: 10, color: '#9ca3af', marginTop: 2 }}>{s.name}</div>
                <div style={{ fontSize: 9, color: '#6b7280', marginTop: 2 }}>{s.status}</div>
              </>
            )}
            {compact && i < stages.length - 1 && <span style={{ display: 'none' }} />}
          </div>
        );
      })}
    </div>
  );
}

function MethodRow({ m, idx }: { m: FactorMethod; idx: number }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ borderBottom: '1px solid #1f2937', marginBottom: 2 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          width: '100%', background: 'none', border: 'none',
          padding: '8px 0', cursor: 'pointer', textAlign: 'left', color: '#d1d5db',
        }}
      >
        <span style={{ color: '#4b5563', fontSize: 11, minWidth: 24 }}>{String(idx + 1).padStart(2, '0')}</span>
        <span style={{ fontSize: 12, fontFamily: 'monospace', color: '#60a5fa' }}>{m.name}</span>
        <span style={{ marginLeft: 'auto', color: '#4b5563', fontSize: 14 }}>{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div style={{ paddingLeft: 32, paddingBottom: 10 }}>
          <p style={{ fontSize: 12, color: '#9ca3af', margin: '0 0 6px' }}>{m.description}</p>
          <div style={{
            fontSize: 12, background: '#0f172a', border: '1px solid #1e293b',
            borderRadius: 6, padding: '6px 10px', color: '#e2e8f0',
          }}>
            <span style={{ color: '#64748b', marginRight: 6 }}>→</span>
            {m.result}
          </div>
        </div>
      )}
    </div>
  );
}

function FactorCard({ factor, onSelect }: { factor: Factor; onSelect: () => void }) {
  const cfg = STATUS_CONFIG[factor.status] ?? STATUS_CONFIG.pending;
  return (
    <div
      onClick={onSelect}
      style={{
        background: '#111827', border: `1px solid ${cfg.color}40`,
        borderRadius: 12, padding: '20px 24px', cursor: 'pointer',
        transition: 'border-color 0.2s',
      }}
      onMouseEnter={e => (e.currentTarget.style.borderColor = cfg.color)}
      onMouseLeave={e => (e.currentTarget.style.borderColor = `${cfg.color}40`)}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 10 }}>
        <span style={{ fontFamily: 'monospace', fontSize: 12, color: '#4b5563', paddingTop: 2 }}>
          {factor.id}
        </span>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 16, fontWeight: 700, color: '#f9fafb', marginBottom: 2 }}>{factor.name}</div>
          <div style={{ fontSize: 12, color: '#6b7280' }}>{factor.name_zh}</div>
        </div>
        <StatusBadge status={factor.status} />
      </div>
      {factor.goal && (
        <div style={{
          fontSize: 12, color: '#fbbf24', background: '#451a0311',
          border: '1px solid #92400e55', borderRadius: 8, padding: '8px 10px', marginBottom: 10,
        }}>
          Goal: {factor.goal.tradable_instrument} · {factor.goal.trading_objective}
        </div>
      )}
      <p style={{ fontSize: 13, color: '#9ca3af', margin: '0 0 12px', lineHeight: 1.6 }}>
        {factor.hypothesis}
      </p>
      {factor.pipeline_stages && (
        <div style={{ marginBottom: 12 }}>
          <PipelineStrip stages={factor.pipeline_stages} compact />
        </div>
      )}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
        {factor.tags.map(t => <TagPill key={t} tag={t} />)}
      </div>
      <div style={{ fontSize: 11, color: '#4b5563' }}>
        {factor.current_stage ? `Stage ${factor.current_stage} · ` : ''}
        {factor.data.primary_period} · {factor.data.n_trading_days.toLocaleString()} trading days
      </div>
    </div>
  );
}

function FactorDetail({ factor, onBack }: { factor: Factor; onBack: () => void }) {
  return (
    <div>
      <button
        onClick={onBack}
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          background: '#1f2937', border: '1px solid #374151', borderRadius: 8,
          padding: '6px 14px', color: '#9ca3af', cursor: 'pointer', fontSize: 13,
          marginBottom: 24,
        }}
      >
        ← Back
      </button>

      <div style={{
        background: '#111827', border: '1px solid #1f2937',
        borderRadius: 12, padding: '24px 28px', marginBottom: 20,
      }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 12 }}>
          <span style={{ fontFamily: 'monospace', fontSize: 13, color: '#4b5563', paddingTop: 4 }}>{factor.id}</span>
          <div>
            <h2 style={{ margin: 0, color: '#f9fafb', fontSize: 22 }}>{factor.name}</h2>
            <div style={{ fontSize: 13, color: '#6b7280', marginTop: 2 }}>{factor.name_zh}</div>
          </div>
          <div style={{ marginLeft: 'auto' }}><StatusBadge status={factor.status} /></div>
        </div>
        <div style={{ fontSize: 13, color: '#9ca3af', lineHeight: 1.7, marginBottom: 10 }}>
          <strong style={{ color: '#6b7280' }}>Hypothesis: </strong>{factor.hypothesis}
        </div>
        <div style={{ fontSize: 13, color: '#9ca3af', lineHeight: 1.7, marginBottom: 14 }}>
          <strong style={{ color: '#6b7280' }}>Motivation: </strong>{factor.motivation}
        </div>
        <div style={{ fontSize: 12, color: '#4b5563', marginBottom: 12 }}>
          {factor.data.instrument} · {factor.data.source} · {factor.data.primary_period} · {factor.data.n_trading_days.toLocaleString()} days
          {factor.data.events && (
            <span style={{ marginLeft: 8 }}>
              · FOMC×{factor.data.events.fomc} / CPI×{factor.data.events.cpi} / NFP×{factor.data.events.nfp}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
          {factor.tags.map(t => <TagPill key={t} tag={t} />)}
        </div>
        {(factor.case_file || factor.experiment_hash) && (
          <div style={{ fontSize: 11, color: '#6b7280' }}>
            {factor.case_file && <span>Case: <code style={{ color: '#93c5fd' }}>{factor.case_file}</code></span>}
            {factor.experiment_hash && (
              <a href={factor.experiment_hash} style={{ marginLeft: 12, color: '#60a5fa' }}>
                实验页 {factor.experiment_hash}
              </a>
            )}
          </div>
        )}
      </div>

      {/* Goal */}
      {factor.goal && (
        <div style={{
          background: '#1c1408', border: '1px solid #92400e',
          borderRadius: 12, padding: '18px 22px', marginBottom: 20,
        }}>
          <h3 style={{ color: '#fbbf24', margin: '0 0 10px', fontSize: 15 }}>S0 · Trading Goal</h3>
          <div style={{ fontSize: 13, color: '#fde68a', lineHeight: 1.7, marginBottom: 8 }}>
            {factor.goal.trading_objective}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, fontSize: 12 }}>
            <div style={{ color: '#9ca3af' }}>
              Research: <span style={{ color: '#e5e7eb' }}>{factor.goal.research_instrument}</span>
            </div>
            <div style={{ color: '#9ca3af' }}>
              Tradable: <span style={{ color: '#e5e7eb' }}>{factor.goal.tradable_instrument}</span>
            </div>
          </div>
          <div style={{ fontSize: 12, color: '#f97316', marginTop: 10, lineHeight: 1.6 }}>
            Mapping gap: {factor.goal.mapping_gap}
          </div>
          <div style={{ fontSize: 11, color: '#78716c', marginTop: 6 }}>Horizon: {factor.goal.horizon}</div>
        </div>
      )}

      {/* Pipeline */}
      {factor.pipeline_stages && (
        <div style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 12, padding: '20px 24px', marginBottom: 20 }}>
          <h3 style={{ color: '#e5e7eb', margin: '0 0 6px', fontSize: 15 }}>
            Pipeline S0–S7
            {factor.current_stage && (
              <span style={{ color: '#f59e0b', fontWeight: 400, fontSize: 13, marginLeft: 8 }}>
                current: {factor.current_stage}
              </span>
            )}
          </h3>
          <p style={{ fontSize: 11, color: '#4b5563', margin: '0 0 14px' }}>
            协议见仓库 factor_pipeline/PIPELINE.md
          </p>
          <PipelineStrip stages={factor.pipeline_stages} />
          <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
            {factor.pipeline_stages.map(s => (
              <div key={s.id} style={{ display: 'flex', gap: 10, fontSize: 12, lineHeight: 1.5 }}>
                <span style={{
                  fontFamily: 'monospace', fontWeight: 700, minWidth: 28,
                  color: STAGE_COLOR[s.status] ?? '#6b7280',
                }}>{s.id}</span>
                <span style={{ color: '#9ca3af', minWidth: 72 }}>{s.name}</span>
                <span style={{ color: '#6b7280', flex: 1 }}>{s.summary ?? '—'}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <div style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 12, padding: '20px 24px' }}>
          <h3 style={{ color: '#e5e7eb', margin: '0 0 14px', fontSize: 15 }}>Key Findings</h3>
          <ul style={{ margin: 0, padding: '0 0 0 16px' }}>
            {factor.key_findings.map((f, i) => (
              <li key={i} style={{ fontSize: 13, color: '#9ca3af', lineHeight: 1.7, marginBottom: 4 }}>{f}</li>
            ))}
          </ul>
        </div>

        <div style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 12, padding: '20px 24px' }}>
          <h3 style={{ color: '#e5e7eb', margin: '0 0 10px', fontSize: 15 }}>Verdict</h3>
          <div style={{
            fontSize: 13, color: '#fbbf24', background: '#451a03',
            border: '1px solid #92400e', borderRadius: 8, padding: '10px 14px', marginBottom: 14,
            lineHeight: 1.7,
          }}>
            {factor.verdict}
          </div>
          <h3 style={{ color: '#e5e7eb', margin: '0 0 10px', fontSize: 15 }}>Next Steps</h3>
          <ul style={{ margin: 0, padding: '0 0 0 16px' }}>
            {factor.next_steps.map((s, i) => (
              <li key={i} style={{ fontSize: 13, color: '#60a5fa', lineHeight: 1.7, marginBottom: 4 }}>{s}</li>
            ))}
          </ul>
        </div>
      </div>

      <div style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 12, padding: '20px 24px', marginBottom: 20 }}>
        <h3 style={{ color: '#e5e7eb', margin: '0 0 14px', fontSize: 15 }}>
          Methods Tried <span style={{ color: '#4b5563', fontWeight: 400, fontSize: 13 }}>({factor.methods.length})</span>
        </h3>
        {factor.methods.map((m, i) => <MethodRow key={m.name} m={m} idx={i} />)}
      </div>

      <div style={{
        background: '#0f172a', border: '1px solid #1e3a5f', borderRadius: 10,
        padding: '14px 18px', marginBottom: 20,
        fontSize: 13, color: '#93c5fd', lineHeight: 1.7,
      }}>
        <strong style={{ color: '#3b82f6' }}>Status Note: </strong>{factor.status_note}
      </div>

      {factor.references.length > 0 && (
        <div style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 12, padding: '20px 24px' }}>
          <h3 style={{ color: '#e5e7eb', margin: '0 0 14px', fontSize: 15 }}>References</h3>
          {factor.references.map((r, i) => (
            <div key={i} style={{ marginBottom: 12, paddingLeft: 12, borderLeft: '2px solid #1f2937' }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#d1d5db' }}>
                {r.title} ({r.year})
              </div>
              <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>
                {r.authors} · {r.journal}
              </div>
              <div style={{ fontSize: 12, color: '#9ca3af', fontStyle: 'italic' }}>{r.note}</div>
            </div>
          ))}
        </div>
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
    return <div style={{ textAlign: 'center', padding: 60, color: '#6b7280' }}>Loading factor log…</div>;
  }

  if (selected) {
    return (
      <div className="container" style={{ maxWidth: 960, margin: '0 auto', padding: '32px 24px' }}>
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
    <div className="container" style={{ maxWidth: 960, margin: '0 auto', padding: '32px 24px' }}>
      <div style={{ marginBottom: 28 }}>
        <div style={{ fontSize: 12, color: '#4b5563', marginBottom: 6, letterSpacing: 1 }}>ALPHA DECK</div>
        <h1 style={{ margin: 0, fontSize: 28, fontWeight: 800, color: '#f9fafb' }}>Factor Research Log</h1>
        <p style={{ margin: '8px 0 0', color: '#6b7280', fontSize: 14 }}>
          S0–S7 流水线记录 · 假设 · 检验 · 可交易性 · 结论
        </p>
        {log.pipeline && (
          <div style={{
            marginTop: 14, fontSize: 12, color: '#93c5fd',
            background: '#0f172a', border: '1px solid #1e3a5f', borderRadius: 8, padding: '10px 14px',
          }}>
            流水线协议：仓库内 <code style={{ color: '#e2e8f0' }}>{log.pipeline.doc}</code>
            {' · '}用法 <code style={{ color: '#e2e8f0' }}>{log.pipeline.readme}</code>
            {' · '}阶段 {log.pipeline.stages.join(' → ')}
          </div>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 10, marginBottom: 28 }}>
        {Object.entries(STATUS_CONFIG).map(([status, cfg]) => (
          <div key={status} style={{
            background: '#111827', border: `1px solid ${cfg.color}30`,
            borderRadius: 10, padding: '12px 14px', textAlign: 'center',
          }}>
            <div style={{ fontSize: 22, fontWeight: 800, color: cfg.color }}>{counts[status] ?? 0}</div>
            <div style={{ fontSize: 11, color: '#6b7280', marginTop: 2 }}>{cfg.label}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {statuses.map(s => (
          <button
            key={s}
            onClick={() => setFilter(s)}
            style={{
              padding: '5px 14px', borderRadius: 8, cursor: 'pointer', fontSize: 12, fontWeight: 600,
              background: filter === s ? '#1d4ed8' : '#1f2937',
              color: filter === s ? '#fff' : '#9ca3af',
              border: filter === s ? '1px solid #2563eb' : '1px solid #374151',
            }}
          >
            {s === 'all' ? `All (${log.factors.length})` : `${STATUS_CONFIG[s]?.label ?? s} (${counts[s] ?? 0})`}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {visible.map(f => (
          <FactorCard key={f.id} factor={f} onSelect={() => setSelected(f)} />
        ))}
        {visible.length === 0 && (
          <div style={{ textAlign: 'center', padding: 40, color: '#4b5563' }}>No factors with this status yet.</div>
        )}
      </div>

      <div style={{ marginTop: 40, textAlign: 'right', fontSize: 11, color: '#374151' }}>
        Alpha Deck v{log.version} · Last updated {log.updated} · {log.factors.length} factors
      </div>
    </div>
  );
}
