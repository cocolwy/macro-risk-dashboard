import { AlertInfo } from '../api';

interface AlertsPanelProps {
  alerts: Record<string, AlertInfo>;
}

const INDICATOR_LABELS: Record<string, string> = {
  term_spread: 'Term Spread',
  credit_spread: 'Credit Spread',
  vix: 'VIX',
  absorption_ratio: 'Absorption Ratio',
  turbulence: 'Turbulence',
  breadth: 'Market Breadth',
};

export function AlertsPanel({ alerts }: AlertsPanelProps) {
  const entries = Object.entries(alerts);
  if (entries.length === 0) return null;

  return (
    <div className="alerts-bar">
      {entries.map(([key, alert]) => (
        <div key={key} className={`alert-badge ${alert.level}`}>
          <span className={`dot ${alert.level}`} />
          <span>{INDICATOR_LABELS[key] || key}: {alert.message}</span>
        </div>
      ))}
    </div>
  );
}
