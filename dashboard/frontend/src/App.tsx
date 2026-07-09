import { useState, useEffect } from 'react';
import { fetchAllData, DataPoint, Summary } from './api';
import { ChartCard } from './components/ChartCard';
import { MultiLineChart } from './components/MultiLineChart';
import { AlertsPanel } from './components/AlertsPanel';
import { CausalFlow } from './components/CausalFlow';

interface DashboardData {
  summary: Summary;
  termSpread: DataPoint[];
  creditSpread: DataPoint[];
  vix: DataPoint[];
  sp500: DataPoint[];
  breadth: DataPoint[];
  absorptionRatio: DataPoint[];
  turbulence: DataPoint[];
}

function getLatestValue(data: DataPoint[], key: string): string {
  if (data.length === 0) return '--';
  const latest = data[data.length - 1];
  const val = latest[key];
  if (val === undefined) return '--';
  return typeof val === 'number' ? val.toFixed(2) : String(val);
}

export default function App() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAllData()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="loading">Loading macro risk data...</div>;
  }

  if (error || !data) {
    return (
      <div className="app">
        <div className="error-banner">
          Failed to load data: {error || 'Unknown error'}. 
          Make sure to run the data fetcher first: <code>python dashboard/fetch_macro_data.py</code>
        </div>
      </div>
    );
  }

  const { summary } = data;

  const hasDanger = Object.values(summary.alerts).some((a) => a.level === 'danger');
  const hasWarning = Object.values(summary.alerts).some((a) => a.level === 'warning');

  return (
    <div className="app">
      {hasDanger && (
        <div className="global-danger-banner">
          RISK ALERT — One or more indicators have entered danger zone
        </div>
      )}
      {!hasDanger && hasWarning && (
        <div className="global-warning-banner">
          CAUTION — Elevated readings detected
        </div>
      )}

      <header className="header">
        <h1>Macro Risk Dashboard</h1>
        <span className="updated">
          Last updated: {summary.last_updated ? new Date(summary.last_updated).toLocaleString() : 'N/A'}
        </span>
      </header>

      <AlertsPanel alerts={summary.alerts} />

      <CausalFlow alerts={summary.alerts} />

      <div className="charts-grid">
        <ChartCard
          title="US Treasury Term Spread (10Y - 2Y)"
          subtitle="Negative = yield curve inverted, recession warning"
          data={data.termSpread}
          dataKey="term_spread_10y2y"
          color="#4a9eff"
          currentValue={`${getLatestValue(data.termSpread, 'term_spread_10y2y')}%`}
          referenceLine={{ y: 0, label: 'Inversion', color: '#f87171' }}
          type="area"
          gradientId="termSpreadGrad"
          alertLevel={summary.alerts.term_spread?.level}
        />

        <ChartCard
          title="VIX (Fear Index)"
          subtitle="CBOE Volatility Index — >20 elevated, >30 extreme"
          data={data.vix}
          dataKey="vix"
          color="#fbbf24"
          currentValue={getLatestValue(data.vix, 'vix')}
          referenceLine={{ y: 20, label: 'Elevated', color: '#fbbf24' }}
          type="area"
          gradientId="vixGrad"
          alertLevel={summary.alerts.vix?.level}
        />

        <MultiLineChart
          title="Credit Spreads"
          subtitle="ICE BofA High Yield & Investment Grade OAS (percentage points)"
          data={data.creditSpread}
          lines={[
            { key: 'high_yield_spread', color: '#f87171', name: 'High Yield' },
            { key: 'investment_grade_spread', color: '#4a9eff', name: 'Investment Grade' },
          ]}
        />

        <ChartCard
          title="S&P 500"
          subtitle="US large cap benchmark"
          data={data.sp500}
          dataKey="sp500"
          color="#34d399"
          currentValue={getLatestValue(data.sp500, 'sp500')}
          type="area"
          gradientId="sp500Grad"
        />

        <ChartCard
          title="Absorption Ratio"
          subtitle="Market coupling (PCA on sector ETFs) — high = fragile, systemic risk"
          data={data.absorptionRatio}
          dataKey="absorption_ratio"
          color="#a78bfa"
          currentValue={getLatestValue(data.absorptionRatio, 'absorption_ratio')}
          alertLevel={summary.alerts.absorption_ratio?.level}
        />

        <ChartCard
          title="Turbulence Index"
          subtitle="Mahalanobis distance — spikes = market regime breakdown"
          data={data.turbulence}
          dataKey="turbulence"
          color="#fb923c"
          currentValue={getLatestValue(data.turbulence, 'turbulence')}
          type="area"
          gradientId="turbGrad"
          alertLevel={summary.alerts.turbulence?.level}
        />

        <ChartCard
          title="Market Breadth"
          subtitle="S&P 500: % days above 200-day MA (rolling 20d)"
          data={data.breadth}
          dataKey="pct_above_200ma"
          color="#34d399"
          currentValue={`${getLatestValue(data.breadth, 'pct_above_200ma')}%`}
          className="full-width"
        />
      </div>

      <footer className="footer">
        Data sources: FRED (US Treasury, Credit Spreads), Yahoo Finance (VIX, S&P 500, Sector ETFs).
        Updated daily via GitHub Actions.
      </footer>
    </div>
  );
}
