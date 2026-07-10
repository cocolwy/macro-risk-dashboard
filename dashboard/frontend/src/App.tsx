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
          explanation={"期限利差 = 10年期国债收益率 - 2年期国债收益率。正常情况下为正值（长期利率高于短期）。当它变为负值（倒挂），意味着债券市场在说「未来经济会很差」，历史上每次美国衰退前12-18个月都出现过倒挂。红线以下 = 危险区域。"}
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
          explanation={"VIX是通过标普500期权价格计算出的「恐慌指数」，反映市场对未来30天波动率的预期。低于20=平静，20-30=紧张，高于30=恐慌（2020年3月曾飙到82）。它是同步指标——不能提前预警，但能告诉你「市场现在有多害怕」。"}
        />

        <MultiLineChart
          title="Credit Spreads"
          subtitle="ICE BofA High Yield & Investment Grade OAS (percentage points)"
          data={data.creditSpread}
          lines={[
            { key: 'high_yield_spread', color: '#f87171', name: 'High Yield' },
            { key: 'investment_grade_spread', color: '#4a9eff', name: 'Investment Grade' },
          ]}
          explanation={"信用利差 = 企业债收益率 - 同期限国债收益率。它衡量「市场觉得企业多大概率还不起钱」。High Yield（垃圾债）利差超过6% = 极度恐慌，3-4% = 正常。利差急剧走阔说明资金在逃离风险资产，往往比股市下跌更早反应。"}
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
          explanation={"标普500指数，美国大盘股基准。放在这里作为参照——当上面的风险指标恶化时，观察S&P 500是否开始下跌。如果风险指标已经亮黄/红灯但S&P还在涨，说明市场还在「最后的疯狂」，反而更危险。"}
        />

        <ChartCard
          title="Absorption Ratio"
          subtitle="Market coupling (PCA on sector ETFs) — high = fragile, systemic risk"
          data={data.absorptionRatio}
          dataKey="absorption_ratio"
          color="#a78bfa"
          currentValue={getLatestValue(data.absorptionRatio, 'absorption_ratio')}
          alertLevel={summary.alerts.absorption_ratio?.level}
          explanation={"吸收比率：用PCA分析11个行业ETF，看前几个主成分能解释多少总波动。值越高说明所有行业被同一个力量驱动（高度耦合），分散化失效。类比：多米诺骨牌排得越紧，倒一个全部倒。当AR突然升高1个标准差以上时 = 系统脆弱。"}
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
          explanation={"湍流指数：用马氏距离衡量「今天多资产的表现有多异常」。它不仅看涨跌幅度，更看资产间相关性是否崩塌（比如股债同跌，而历史上它们负相关）。飙升 = 市场在「失序」，正常的避险逻辑不再有效，通常发生在大跌的开始阶段。"}
        />

        <ChartCard
          title="Market Breadth"
          subtitle="% of 11 sector ETFs above their 200-day moving average"
          data={data.breadth}
          dataKey="pct_above_200ma"
          color="#34d399"
          currentValue={`${getLatestValue(data.breadth, 'pct_above_200ma')}%`}
          className="full-width"
          explanation={"市场宽度：11个标普行业ETF中，有多少比例当前价格在各自的200日均线之上。100%=所有行业都在上升趋势中，0%=全部跌破趋势线。关键信号：如果S&P 500还在涨但这个比例在下降 =「顶部背离」，说明上涨只靠少数行业撑着，多数行业已经走弱。"}
        />
      </div>

      <footer className="footer">
        Data sources: FRED (US Treasury, Credit Spreads), Yahoo Finance (VIX, S&P 500, Sector ETFs).
        Updated daily via GitHub Actions.
      </footer>
    </div>
  );
}
