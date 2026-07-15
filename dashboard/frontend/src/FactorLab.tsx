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

interface Factor {
  id: string;
  name: string;
  name_zh: string;
  hypothesis: string;
  motivation: string;
  status: 'confirmed' | 'conditional' | 'weak' | 'dead' | 'pending';
  status_note: string;
  tags: string[];
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
  factors: Factor[];
}

const STATUS_CONFIG: Record<string, { label: string; color: string; bg: string; icon: string }> = {
  confirmed: { label: 'Confirmed', color: '#10b981', bg: '#064e3b', icon: '✓' },
  conditional: { label: 'Conditional', color: '#f59e0b', bg: '#451a03', icon: '≈' },
  weak:        { label: 'Weak',        color: '#f97316', bg: '#431407', icon: '~' },
  dead:        { label: 'Dead',        color: '#ef4444', bg: '#450a0a', icon: '✗' },
  pending:     { label: 'Pending',     color: '#6b7280', bg: '#111827', icon: '?' },
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
      <p style={{ fontSize: 13, color: '#9ca3af', margin: '0 0 12px', lineHeight: 1.6 }}>
        {factor.hypothesis}
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
        {factor.tags.map(t => <TagPill key={t} tag={t} />)}
      </div>
      <div style={{ fontSize: 11, color: '#4b5563' }}>
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

      {/* Header */}
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
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {factor.tags.map(t => <TagPill key={t} tag={t} />)}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        {/* Key Findings */}
        <div style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 12, padding: '20px 24px' }}>
          <h3 style={{ color: '#e5e7eb', margin: '0 0 14px', fontSize: 15 }}>Key Findings</h3>
          <ul style={{ margin: 0, padding: '0 0 0 16px' }}>
            {factor.key_findings.map((f, i) => (
              <li key={i} style={{ fontSize: 13, color: '#9ca3af', lineHeight: 1.7, marginBottom: 4 }}>{f}</li>
            ))}
          </ul>
        </div>

        {/* Verdict + Next Steps */}
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

      {/* Methods */}
      <div style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 12, padding: '20px 24px', marginBottom: 20 }}>
        <h3 style={{ color: '#e5e7eb', margin: '0 0 14px', fontSize: 15 }}>
          Methods Tried <span style={{ color: '#4b5563', fontWeight: 400, fontSize: 13 }}>({factor.methods.length})</span>
        </h3>
        {factor.methods.map((m, i) => <MethodRow key={m.name} m={m} idx={i} />)}
      </div>

      {/* Status Note */}
      <div style={{
        background: '#0f172a', border: '1px solid #1e3a5f', borderRadius: 10,
        padding: '14px 18px', marginBottom: 20,
        fontSize: 13, color: '#93c5fd', lineHeight: 1.7,
      }}>
        <strong style={{ color: '#3b82f6' }}>Status Note: </strong>{factor.status_note}
      </div>

      {/* References */}
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
      <div className="container" style={{ maxWidth: 900, margin: '0 auto', padding: '32px 24px' }}>
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
    <div className="container" style={{ maxWidth: 900, margin: '0 auto', padding: '32px 24px' }}>
      {/* Page Header */}
      <div style={{ marginBottom: 32 }}>
        <div style={{ fontSize: 12, color: '#4b5563', marginBottom: 6, letterSpacing: 1 }}>ALPHA DECK</div>
        <h1 style={{ margin: 0, fontSize: 28, fontWeight: 800, color: '#f9fafb' }}>Factor Research Log</h1>
        <p style={{ margin: '8px 0 0', color: '#6b7280', fontSize: 14 }}>
          系统化记录每个量化因子的假设、检验方法、发现与结论
        </p>
      </div>

      {/* Summary Cards */}
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

      {/* Filter Tabs */}
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

      {/* Factor List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {visible.map(f => (
          <FactorCard key={f.id} factor={f} onSelect={() => setSelected(f)} />
        ))}
        {visible.length === 0 && (
          <div style={{ textAlign: 'center', padding: 40, color: '#4b5563' }}>No factors with this status yet.</div>
        )}
      </div>

      {/* Footer */}
      <div style={{ marginTop: 40, textAlign: 'right', fontSize: 11, color: '#374151' }}>
        Alpha Deck v{log.version} · Last updated {log.updated} · {log.factors.length} factors
      </div>
    </div>
  );
}
