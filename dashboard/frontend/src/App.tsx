import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { fetchAllData, fetchDataJson, resolveCompositeScore, DataPoint, Summary, MomentumData } from './api';
import { ChartCard } from './components/ChartCard';
import { MultiLineChart } from './components/MultiLineChart';
import { AlertsPanel } from './components/AlertsPanel';
import { CausalFlow } from './components/CausalFlow';
import { SectorTable } from './components/SectorTable';
import { ScoreGauge } from './components/ScoreGauge';

const staggerContainer = {
  animate: { transition: { staggerChildren: 0.04 } },
};
const staggerItem = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.3, ease: [0.25, 0.1, 0.25, 1] } },
};

interface DashboardData {
  summary: Summary;
  termSpread: DataPoint[];
  creditSpread: DataPoint[];
  vix: DataPoint[];
  sp500: DataPoint[];
  breadth: DataPoint[];
  sectors: DataPoint[];
  absorptionRatio: DataPoint[];
  turbulence: DataPoint[];
  compositeScore: DataPoint[];
  momentum: MomentumData;
}

interface ProductionModel {
  id: string;
  name: string;
  role: string;
  current_probability: number;
  current_signal: string;
  auc: number;
  n_features: number;
  practical_metrics: {
    best_f1: number;
    brier_score: number;
  };
}
interface ProductionModelsBlock {
  models: ProductionModel[];
  walk_forward?: {
    status: string;
    message: string;
    summary_by_model: Record<string, { f1_mean: number; f1_std: number }>;
  };
}
interface ModelMetricsData {
  production_models?: ProductionModelsBlock;
}

function signalColor(signal: string) {
  return signal === 'elevated' ? '#dc2626' : signal === 'watch' ? '#b45309' : '#16a34a';
}

function CrashModelSummary({ block }: { block: ProductionModelsBlock }) {
  const wf = block.walk_forward;
  return (
    <div className="crash-model-summary" style={{
      background: 'rgba(255, 255, 255, 0.72)', backdropFilter: 'blur(16px) saturate(180%)',
      WebkitBackdropFilter: 'blur(16px) saturate(180%)',
      borderRadius: 16, padding: '16px 18px',
      border: '1px solid rgba(226, 232, 240, 0.65)', marginBottom: 20,
      boxShadow: '0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 10 }}>
        <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700, letterSpacing: '-0.01em' }}>ML Crash Risk（Ch.2）</h3>
        <span style={{
          fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 4,
          background: '#fef3c7', color: '#92400e', border: '1px solid #fcd34d',
        }}>RESEARCH</span>
      </div>
      {wf && (
        <div style={{
          fontSize: 11, padding: '6px 10px', marginBottom: 12, borderRadius: 6,
          background: '#fffbeb', border: '1px solid #fcd34d', color: '#92400e', lineHeight: 1.6,
        }}>
          {wf.message}
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        {block.models.map((m, i) => {
          const probPct = (m.current_probability * 100).toFixed(1);
          const wfRow = wf?.summary_by_model?.[m.name];
          return (
            <div key={m.id} style={{
              padding: '10px 12px', borderRadius: 8,
              border: `2px solid ${i === 0 ? '#d6457a' : '#0d9488'}`,
              background: i === 0 ? 'rgba(214,69,122,0.03)' : 'rgba(13,148,136,0.03)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
                <span style={{ fontSize: 12, fontWeight: 700 }}>{m.name.replace('LR ', '')}</span>
                <span style={{ fontSize: 10, fontWeight: 600, color: i === 0 ? '#d6457a' : '#0d9488' }}>{m.role}</span>
              </div>
              <div style={{ fontSize: 24, fontWeight: 700, letterSpacing: '-0.02em', color: signalColor(m.current_signal), marginBottom: 2 }}>
                {probPct}%
              </div>
              <div style={{ fontSize: 11, color: '#6b7280' }}>
                F1 {m.practical_metrics.best_f1.toFixed(3)} · Brier {m.practical_metrics.brier_score.toFixed(3)}
              </div>
              {wfRow && (
                <div style={{ fontSize: 10, color: '#92400e', marginTop: 2 }}>
                  WF F1: {wfRow.f1_mean.toFixed(3)} ± {wfRow.f1_std.toFixed(3)}
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div style={{ fontSize: 11, color: '#6b7280', marginTop: 10 }}>
        Target: 20d &gt;5% 回撤 · 详见{' '}
        <a href="#ch1" style={{ color: '#3a82d6' }}>Ch.1 双轨模型</a> ·{' '}
        <a href="#ch2" style={{ color: '#3a82d6' }}>Ch.2 Walk-forward</a>
      </div>
    </div>
  );
}

function computeFragilityPosition(vixData: DataPoint[], arData: DataPoint[]) {
  if (vixData.length < 60 || arData.length < 60) return null;

  const window = Math.min(240, vixData.length);
  const smoothing = 5;

  const vixVals = vixData.slice(-window).map(d => d.vix as number).filter(v => v != null);
  const arVals = arData.slice(-window).map(d => d.absorption_ratio as number).filter(v => v != null);

  if (vixVals.length < 60 || arVals.length < 60) return null;

  const latestVix = vixVals[vixVals.length - 1];
  const latestAr = arVals[arVals.length - 1];
  const vixPctile = vixVals.filter(v => v <= latestVix).length / vixVals.length;
  const arPctile = arVals.filter(v => v <= latestAr).length / arVals.length;
  const fragility = vixPctile * arPctile;

  const recentFragilities: number[] = [];
  for (let i = Math.max(0, vixVals.length - smoothing); i < vixVals.length; i++) {
    const v = vixVals[i];
    const a = arVals[Math.min(i, arVals.length - 1)];
    const vp = vixVals.filter(x => x <= v).length / vixVals.length;
    const ap = arVals.filter(x => x <= a).length / arVals.length;
    recentFragilities.push(vp * ap);
  }
  const smoothedFragility = recentFragilities.reduce((s, x) => s + x, 0) / recentFragilities.length;
  const position = Math.max(0.2, Math.min(1.0, 1 - smoothedFragility));

  return { fragility: smoothedFragility, position, vixPctile, arPctile, rawFragility: fragility };
}

function PositionCard({ vixData, arData }: { vixData: DataPoint[]; arData: DataPoint[] }) {
  const result = computeFragilityPosition(vixData, arData);
  if (!result) return null;

  const posPct = (result.position * 100).toFixed(0);
  const posColor = result.position >= 0.7 ? '#16a34a' : result.position >= 0.5 ? '#b45309' : '#dc2626';

  return (
    <div style={{
      background: 'rgba(255, 255, 255, 0.72)', backdropFilter: 'blur(16px) saturate(180%)',
      WebkitBackdropFilter: 'blur(16px) saturate(180%)',
      borderRadius: 16, padding: '16px 18px',
      border: '1px solid rgba(226, 232, 240, 0.65)', marginBottom: 20,
      boxShadow: '0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700, letterSpacing: '-0.01em' }}>Position Sizing（脆弱性仓位）</h3>
        <span style={{ fontSize: 10, padding: '2px 8px', borderRadius: 4, background: '#fef3c7', color: '#92400e', border: '1px solid #fcd34d', fontWeight: 600 }}>
          EXPERIMENTAL
        </span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 10 }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 28, fontWeight: 700, letterSpacing: '-0.02em', color: posColor }}>{posPct}%</div>
          <div style={{ fontSize: 11, color: '#6b7280' }}>建议仓位</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 20, fontWeight: 600, color: '#6b7280' }}>{(result.fragility * 100).toFixed(0)}%</div>
          <div style={{ fontSize: 11, color: '#6b7280' }}>脆弱度</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#6b7280', lineHeight: 1.6 }}>
            VIX pctile: {(result.vixPctile * 100).toFixed(0)}%<br/>
            AR pctile: {(result.arPctile * 100).toFixed(0)}%
          </div>
        </div>
      </div>
      <div style={{ fontSize: 11, color: '#9ca3af', borderTop: '1px solid #f1f5f9', paddingTop: 8 }}>
        公式: position = 1 - smooth(VIX_pctile × AR_pctile, 5d) · 下限 20%<br/>
        ⚠️ 无提前效应（VIX 是同步指标），仅参考 · 详见{' '}
        <a href="#ch3-risk" style={{ color: '#7c3aed' }}>Ch.3</a>
      </div>
    </div>
  );
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
  const [modelBlock, setModelBlock] = useState<ProductionModelsBlock | null>(null);

  useEffect(() => {
    fetchAllData()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
    fetchDataJson<ModelMetricsData>('model_metrics.json')
      .then(d => setModelBlock(d.production_models ?? null))
      .catch(() => {});
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
  const compositeScoreInfo = resolveCompositeScore(summary, data.compositeScore);

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

      {compositeScoreInfo && (
        <ScoreGauge
          score={compositeScoreInfo.score}
          label={compositeScoreInfo.label}
          level={compositeScoreInfo.level}
          action={compositeScoreInfo.action}
          components={compositeScoreInfo.components}
          momentum={data.momentum}
        />
      )}

      {modelBlock && <CrashModelSummary block={modelBlock} />}

      {data.vix.length > 0 && data.absorptionRatio.length > 0 && (
        <PositionCard vixData={data.vix} arData={data.absorptionRatio} />
      )}

      {data.compositeScore.length > 0 && (
        <div className="charts-grid" style={{ marginBottom: '20px' }}>
          <ChartCard
            title="Risk Score History"
            subtitle="Composite score over time — higher = more dangerous"
            data={data.compositeScore}
            dataKey="composite_score"
            color="#d6457a"
            type="area"
            gradientId="scoreGrad"
            referenceLine={{ y: 60, label: 'High Risk', color: '#dc2626' }}
            className="full-width"
            explanation={"综合风险评分的历史走势。高于60分=高风险区间（橙红色），需要认真对待。注意观察评分上升的速度——缓慢上升可能只是波动，突然跳升更值得警惕。"}
          />
        </div>
      )}

      <CausalFlow alerts={summary.alerts} />

      <motion.div className="charts-grid" variants={staggerContainer} initial="initial" animate="animate">
        <motion.div variants={staggerItem}><ChartCard
          title="US Treasury Term Spread (10Y - 2Y)"
          subtitle="Negative = yield curve inverted, recession warning"
          data={data.termSpread}
          dataKey="term_spread_10y2y"
          color="#3a82d6"
          currentValue={`${getLatestValue(data.termSpread, 'term_spread_10y2y')}%`}
          referenceLine={{ y: 0, label: 'Inversion', color: '#dc2626' }}
          type="area"
          gradientId="termSpreadGrad"
          alertLevel={summary.alerts.term_spread?.level}
          explanation={"期限利差 = 10年期国债收益率 - 2年期国债收益率。正常情况下为正值（长期利率高于短期）。当它变为负值（倒挂），意味着债券市场在说「未来经济会很差」，历史上每次美国衰退前12-18个月都出现过倒挂。红线以下 = 危险区域。"}
        /></motion.div>

        <motion.div variants={staggerItem}><ChartCard
          title="VIX (Fear Index)"
          subtitle="CBOE Volatility Index — >20 elevated, >30 extreme"
          data={data.vix}
          dataKey="vix"
          color="#b45309"
          currentValue={getLatestValue(data.vix, 'vix')}
          referenceLine={{ y: 20, label: 'Elevated', color: '#b45309' }}
          type="area"
          gradientId="vixGrad"
          alertLevel={summary.alerts.vix?.level}
          explanation={"VIX是通过标普500期权价格计算出的「恐慌指数」，反映市场对未来30天波动率的预期。低于20=平静，20-30=紧张，高于30=恐慌（2020年3月曾飙到82）。它是同步指标——不能提前预警，但能告诉你「市场现在有多害怕」。"}
        /></motion.div>

        <motion.div variants={staggerItem}><MultiLineChart
          title="Credit Spreads"
          subtitle="ICE BofA High Yield & Investment Grade OAS (percentage points)"
          data={data.creditSpread}
          lines={[
            { key: 'high_yield_spread', color: '#d6457a', name: 'High Yield' },
            { key: 'investment_grade_spread', color: '#3a82d6', name: 'Investment Grade' },
          ]}
          explanation={"信用利差 = 企业债收益率 - 同期限国债收益率。它衡量「市场觉得企业多大概率还不起钱」。High Yield（垃圾债）利差超过6% = 极度恐慌，3-4% = 正常。利差急剧走阔说明资金在逃离风险资产，往往比股市下跌更早反应。"}
        /></motion.div>

        <motion.div variants={staggerItem}><ChartCard
          title="S&P 500"
          subtitle="US large cap benchmark"
          data={data.sp500}
          dataKey="sp500"
          color="#16a34a"
          currentValue={getLatestValue(data.sp500, 'sp500')}
          type="area"
          gradientId="sp500Grad"
          explanation={"标普500指数，美国大盘股基准。放在这里作为参照——当上面的风险指标恶化时，观察S&P 500是否开始下跌。如果风险指标已经亮黄/红灯但S&P还在涨，说明市场还在「最后的疯狂」，反而更危险。"}
        /></motion.div>

        <motion.div variants={staggerItem}><ChartCard
          title="Absorption Ratio"
          subtitle="Market coupling (PCA on sector ETFs) — high = fragile, systemic risk"
          data={data.absorptionRatio}
          dataKey="absorption_ratio"
          color="#8b5cf6"
          currentValue={getLatestValue(data.absorptionRatio, 'absorption_ratio')}
          alertLevel={summary.alerts.absorption_ratio?.level}
          explanation={"吸收比率：用PCA分析11个行业ETF，看前几个主成分能解释多少总波动。值越高说明所有行业被同一个力量驱动（高度耦合），分散化失效。类比：多米诺骨牌排得越紧，倒一个全部倒。当AR突然升高1个标准差以上时 = 系统脆弱。"}
        /></motion.div>

        <motion.div variants={staggerItem}><ChartCard
          title="Turbulence Index"
          subtitle="Mahalanobis distance — spikes = market regime breakdown"
          data={data.turbulence}
          dataKey="turbulence"
          color="#ea580c"
          currentValue={getLatestValue(data.turbulence, 'turbulence')}
          type="area"
          gradientId="turbGrad"
          alertLevel={summary.alerts.turbulence?.level}
          explanation={"湍流指数：用马氏距离衡量「今天多资产的表现有多异常」。它不仅看涨跌幅度，更看资产间相关性是否崩塌（比如股债同跌，而历史上它们负相关）。飙升 = 市场在「失序」，正常的避险逻辑不再有效，通常发生在大跌的开始阶段。"}
        /></motion.div>

        <motion.div variants={staggerItem} className="full-width"><ChartCard
          title="Market Breadth"
          subtitle="% of 11 sector ETFs above their 200-day moving average"
          data={data.breadth}
          dataKey="pct_above_200ma"
          color="#16a34a"
          currentValue={`${getLatestValue(data.breadth, 'pct_above_200ma')}%`}
          className="full-width"
          explanation={"市场宽度：11个标普行业ETF中，有多少比例当前价格在各自的200日均线之上。100%=所有行业都在上升趋势中，0%=全部跌破趋势线。关键信号：如果S&P 500还在涨但这个比例在下降 =「顶部背离」，说明上涨只靠少数行业撑着，多数行业已经走弱。"}
        /></motion.div>

        <motion.div variants={staggerItem} className="full-width"><SectorTable data={data.sectors} /></motion.div>
      </motion.div>

      <footer className="footer">
        <div>Data sources: FRED (US Treasury, Credit Spreads), Yahoo Finance (VIX, S&P 500, Sector ETFs). Updated daily via GitHub Actions.</div>
        <div style={{ marginTop: '4px', fontSize: '10px', opacity: 0.6, letterSpacing: '0.3px' }}>
          Built with <strong>auto-dashboard</strong> · design © Coco
        </div>
      </footer>
    </div>
  );
}
