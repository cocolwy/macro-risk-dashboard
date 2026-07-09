import { AlertInfo } from '../api';

interface CausalFlowProps {
  alerts: Record<string, AlertInfo>;
}

interface FlowNode {
  id: string;
  label: string;
  sublabel: string;
  tier: string;
  leadTime: string;
}

const NODES: FlowNode[] = [
  { id: 'term_spread', label: 'Term Spread', sublabel: 'Yield Curve Inversion', tier: 'macro', leadTime: '12-18mo' },
  { id: 'credit_spread', label: 'Credit Spread', sublabel: 'HY Bond Stress', tier: 'credit', leadTime: '3-6mo' },
  { id: 'absorption_ratio', label: 'Absorption Ratio', sublabel: 'Market Coupling', tier: 'structure', leadTime: '1-3mo' },
  { id: 'breadth', label: 'Market Breadth', sublabel: 'Internal Weakness', tier: 'structure', leadTime: '1-3mo' },
  { id: 'turbulence', label: 'Turbulence', sublabel: 'Regime Breakdown', tier: 'signal', leadTime: 'Days' },
  { id: 'vix', label: 'VIX', sublabel: 'Fear Pricing', tier: 'signal', leadTime: 'Days' },
];

const TIER_CONFIG: Record<string, { label: string; color: string }> = {
  macro: { label: 'Macro Layer (12-18mo lead)', color: '#4a9eff' },
  credit: { label: 'Credit Layer (3-6mo lead)', color: '#a78bfa' },
  structure: { label: 'Structure Layer (1-3mo lead)', color: '#fbbf24' },
  signal: { label: 'Signal Layer (concurrent)', color: '#f87171' },
};

function getLevelColor(level?: string): string {
  if (level === 'danger') return '#f87171';
  if (level === 'warning') return '#fbbf24';
  return '#34d399';
}

function getLevelGlow(level?: string): string {
  if (level === 'danger') return '0 0 12px rgba(248, 113, 113, 0.6)';
  if (level === 'warning') return '0 0 8px rgba(251, 191, 36, 0.4)';
  return 'none';
}

export function CausalFlow({ alerts }: CausalFlowProps) {
  const tiers = ['macro', 'credit', 'structure', 'signal'];

  return (
    <div className="chart-card full-width" style={{ padding: '24px' }}>
      <h3>Indicator Causal Chain</h3>
      <div className="subtitle">
        Macro signals propagate downward — earlier layers give more lead time
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '20px' }}>
        {tiers.map((tier, tierIdx) => {
          const tierNodes = NODES.filter((n) => n.tier === tier);
          const config = TIER_CONFIG[tier];

          return (
            <div key={tier}>
              <div style={{
                fontSize: '11px',
                color: config.color,
                fontWeight: 600,
                marginBottom: '8px',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
              }}>
                {config.label}
              </div>

              <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                {tierNodes.map((node) => {
                  const alert = alerts[node.id];
                  const level = alert?.level;
                  const borderColor = getLevelColor(level);
                  const glow = getLevelGlow(level);

                  return (
                    <div
                      key={node.id}
                      style={{
                        flex: '1 1 200px',
                        background: level === 'danger'
                          ? 'rgba(248, 113, 113, 0.08)'
                          : level === 'warning'
                            ? 'rgba(251, 191, 36, 0.05)'
                            : 'var(--bg-secondary)',
                        border: `1.5px solid ${borderColor}`,
                        borderRadius: '10px',
                        padding: '14px 16px',
                        boxShadow: glow,
                        transition: 'all 0.3s ease',
                        animation: level === 'danger' ? 'pulse-danger 2s infinite' : undefined,
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <div style={{ fontWeight: 600, fontSize: '14px' }}>{node.label}</div>
                          <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                            {node.sublabel}
                          </div>
                        </div>
                        <div style={{
                          width: '10px',
                          height: '10px',
                          borderRadius: '50%',
                          background: borderColor,
                          boxShadow: glow,
                        }} />
                      </div>
                      {alert && (
                        <div style={{
                          marginTop: '8px',
                          fontSize: '11px',
                          color: borderColor,
                          fontWeight: 500,
                        }}>
                          {alert.message}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {tierIdx < tiers.length - 1 && (
                <div style={{
                  textAlign: 'center',
                  color: 'var(--text-secondary)',
                  fontSize: '18px',
                  marginTop: '12px',
                  opacity: 0.5,
                }}>
                  ▼
                </div>
              )}
            </div>
          );
        })}

        <div style={{
          textAlign: 'center',
          marginTop: '8px',
          padding: '12px',
          background: 'rgba(248, 113, 113, 0.05)',
          border: '1px dashed rgba(248, 113, 113, 0.3)',
          borderRadius: '8px',
          fontSize: '13px',
          color: 'var(--text-secondary)',
        }}>
          ▼ Market Crash / Major Drawdown
        </div>
      </div>
    </div>
  );
}
