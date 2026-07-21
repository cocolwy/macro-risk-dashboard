import { useEffect, useState } from 'react';
import { useSpring } from 'framer-motion';

function AnimatedNumber({ value, color }: { value: number; color: string }) {
  const spring = useSpring(0, { damping: 30, stiffness: 100 });
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    spring.set(value);
    const unsub = spring.on('change', (v: number) => setDisplay(Math.round(v)));
    return unsub;
  }, [value, spring]);

  return (
    <div style={{
      fontSize: '72px',
      fontWeight: 800,
      color,
      marginTop: '16px',
      lineHeight: 1,
      letterSpacing: '-0.04em',
      textShadow: `0 0 40px ${color}30`,
    }}>
      {display}
    </div>
  );
}

interface ScoreGaugeProps {
  score: number;
  label: string;
  level: string;
  action: string;
  components: Record<string, number>;
  momentum?: Record<string, { current: number; week_change: number; month_change: number }>;
}

const COMPONENT_LABELS: Record<string, string> = {
  term_spread: 'Term Spread',
  credit_spread: 'Credit Spread',
  vix: 'VIX',
  absorption_ratio: 'Absorption Ratio',
  turbulence: 'Turbulence',
  breadth: 'Market Breadth',
};

function getScoreColor(score: number): string {
  if (score <= 20) return '#16a34a';
  if (score <= 40) return '#22c55e';
  if (score <= 60) return '#f59e0b';
  if (score <= 80) return '#ea580c';
  return '#dc2626';
}

function getMomentumArrow(change: number): string {
  if (change > 0.5) return '↑';
  if (change < -0.5) return '↓';
  return '→';
}

function getMomentumColor(change: number, invertGood: boolean = false): string {
  const isUp = change > 0.5;
  const isDown = change < -0.5;
  if (invertGood) {
    if (isUp) return '#dc2626';
    if (isDown) return '#16a34a';
  } else {
    if (isUp) return '#16a34a';
    if (isDown) return '#dc2626';
  }
  return 'var(--text-3)';
}

export function ScoreGauge({ score, label, action, components, momentum }: ScoreGaugeProps) {
  const color = getScoreColor(score);

  return (
    <div className="chart-card full-width" style={{ textAlign: 'center', padding: '28px' }}>
      <h3>Composite Risk Score</h3>
      <div className="subtitle">Weighted aggregate of all indicators with acceleration & persistence filters</div>

      <AnimatedNumber value={Math.round(score)} color={color} />
      <div style={{
        fontSize: '16px',
        fontWeight: 600,
        color,
        marginTop: '8px',
        letterSpacing: '-0.01em',
      }}>
        {label}
      </div>
      <div style={{
        fontSize: '13px',
        color: 'var(--text-2)',
        marginTop: '6px',
      }}>
        {action}
      </div>

      <div style={{
        margin: '20px auto',
        maxWidth: '400px',
        height: '6px',
        background: 'rgba(241, 216, 226, 0.3)',
        borderRadius: '3px',
        position: 'relative',
        overflow: 'hidden',
      }}>
        <div style={{
          width: `${score}%`,
          height: '100%',
          background: `linear-gradient(90deg, #16a34a, #f59e0b, #dc2626)`,
          borderRadius: '3px',
          transition: 'width 0.6s cubic-bezier(0.2, 0, 0, 1)',
        }} />
        {[20, 40, 60, 80].map((t) => (
          <div key={t} style={{
            position: 'absolute',
            left: `${t}%`,
            top: 0,
            bottom: 0,
            width: '1px',
            background: 'rgba(255,255,255,0.25)',
          }} />
        ))}
      </div>

      {/* Scale labels */}
      <div style={{ display: 'flex', justifyContent: 'space-between', maxWidth: '400px', margin: '0 auto 20px', fontSize: '10px', color: 'var(--text-3)' }}>
        <span>Low</span>
        <span>Moderate</span>
        <span>Elevated</span>
        <span>High</span>
        <span>Extreme</span>
      </div>

      {/* Component breakdown */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
        gap: '8px',
        marginTop: '16px',
        textAlign: 'left',
      }}>
        {Object.entries(components).map(([key, value]) => {
          const compColor = getScoreColor(value);
          const mom = momentum?.[key];
          const isInvertedMomentum = ['term_spread', 'breadth'].indexOf(key) === -1;
          return (
            <div key={key} style={{
              padding: '8px 12px',
              background: 'var(--bg-2)',
              borderRadius: '6px',
              borderLeft: `3px solid ${compColor}`,
              border: '1px solid var(--border)',
              borderLeftWidth: '3px',
              borderLeftColor: compColor,
            }}>
              <div style={{ fontSize: '11px', color: 'var(--text-2)' }}>
                {COMPONENT_LABELS[key] || key}
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px', marginTop: '2px' }}>
                <span style={{ fontSize: '16px', fontWeight: 700, color: compColor }}>
                  {Math.round(value)}
                </span>
                {mom && (
                  <span style={{ fontSize: '11px', color: getMomentumColor(mom.week_change, isInvertedMomentum) }}>
                    1W: {getMomentumArrow(mom.week_change)}{mom.week_change > 0 ? '+' : ''}{mom.week_change.toFixed(2)}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
