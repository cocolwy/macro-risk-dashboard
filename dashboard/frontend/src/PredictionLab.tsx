import { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, ReferenceLine, Area, AreaChart,
  ComposedChart, CartesianGrid, Legend
} from 'recharts';

interface ModelInfo {
  name: string;
  target: string;
  features_count: number;
  training_samples: number;
  test_samples: number;
  positive_rate: number;
  train_period: string;
  test_period: string;
  roc_auc: number;
  last_updated: string;
}

interface ThresholdRow {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  alert_days: number;
  total_days: number;
  alert_pct: number;
}

interface FeatureWeight {
  feature: string;
  weight: number;
}

interface EventBacktest {
  name: string;
  event_date: string;
  drop_pct: number;
  first_alert_date: string | null;
  lead_days: number | null;
  max_probability: number;
  max_prob_date: string;
}

interface ProbPoint {
  date: string;
  probability: number;
}

interface SP500Point {
  date: string;
  sp500: number;
}

interface ModelMetrics {
  model_info: ModelInfo;
  current_prediction: { date: string; probability: number; signal: string };
  pr_curve: { threshold: number; precision: number; recall: number }[];
  roc_curve: { fpr: number; tpr: number }[];
  feature_importance: FeatureWeight[];
  threshold_analysis: ThresholdRow[];
  events_backtest: EventBacktest[];
  probability_timeline: ProbPoint[];
  sp500_timeline: SP500Point[];
  actual_drops: { date: string; max_drawdown: number }[];
}

const DATA_BASE_URL = import.meta.env.PROD ? './data' : '/data';

const FEATURE_LABELS: Record<string, string> = {
  'credit_spread_10d_chg': '信用利差 10日变化',
  'breadth_10d_chg': '市场宽度 10日变化',
  'absorption_ratio_5d_chg': '耦合度 5日变化',
  'turbulence_5d_chg': '湍流度 5日变化',
  'vix_20d_chg': 'VIX 20日变化',
  'credit_spread_5d_chg': '信用利差 5日变化',
  'vix_5d_chg': 'VIX 5日变化',
  'sp500_vs_50ma': 'S&P vs 50MA',
  'absorption_ratio_10d_chg': '耦合度 10日变化',
  'credit_spread_level': '信用利差水平',
  'sp500_20d_ret': 'S&P 20日回报',
  'sp500_50d_ret': 'S&P 50日回报',
  'vix_volatility': 'VIX波动率',
  'vix_level': 'VIX水平',
  'vix_vs_20d_avg': 'VIX vs 20日均值',
  'term_spread_level': '期限利差水平',
  'term_spread_5d_chg': '期限利差 5日变化',
  'term_spread_20d_chg': '期限利差 20日变化',
  'breadth_level': '市场宽度水平',
  'turbulence_10d_chg': '湍流度 10日变化',
  'turbulence_20d_chg': '湍流度 20日变化',
  'absorption_ratio_20d_chg': '耦合度 20日变化',
  'vix_10d_chg': 'VIX 10日变化',
};

export function PredictionLab() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${DATA_BASE_URL}/model_metrics.json`)
      .then(r => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then(setMetrics)
      .catch(e => setError(e.message));
  }, []);

  if (error) return <div className="lab-error">Failed to load model data: {error}</div>;
  if (!metrics) return <div className="lab-loading">Loading Prediction Lab...</div>;

  const { model_info, current_prediction, feature_importance, threshold_analysis, events_backtest } = metrics;

  const probWithSP = metrics.probability_timeline.map(p => {
    const sp = metrics.sp500_timeline.find(s => s.date === p.date);
    return { ...p, sp500: sp?.sp500 };
  });

  const signalColor = current_prediction.signal === 'elevated' ? '#ef4444'
    : current_prediction.signal === 'watch' ? '#f59e0b' : '#22c55e';

  return (
    <div className="lab-container">
      <header className="lab-header">
        <div>
          <h1>Prediction Lab</h1>
          <p className="lab-subtitle">研发中 - 预测模型监督与优化</p>
        </div>
        <div className="lab-model-badge">
          <span className="lab-badge-version">{model_info.name}</span>
          <span className="lab-badge-auc">AUC: {model_info.roc_auc}</span>
        </div>
      </header>

      {/* Current Signal */}
      <section className="lab-signal-card" style={{ borderColor: signalColor }}>
        <div className="lab-signal-main">
          <div className="lab-signal-prob" style={{ color: signalColor }}>
            {(current_prediction.probability * 100).toFixed(1)}%
          </div>
          <div className="lab-signal-meta">
            <div className="lab-signal-label">
              未来20日出现{'>'}5%回撤概率
            </div>
            <div className="lab-signal-date">{current_prediction.date}</div>
            <div className="lab-signal-note">
              精确率约17%，此信号仅供研究参考
            </div>
          </div>
        </div>
      </section>

      {/* Probability Timeline + S&P500 overlay */}
      <section className="lab-card">
        <h2>预测概率 vs 实际走势</h2>
        <p className="lab-card-desc">橙色区域为模型输出的崩盘概率，蓝线为S&P 500走势。概率超过50%虚线即为预警</p>
        <div className="lab-chart-tall">
          <ResponsiveContainer width="100%" height={320}>
            <ComposedChart data={probWithSP} margin={{ top: 10, right: 40, left: 10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="date" tick={{ fill: '#999', fontSize: 11 }} interval={Math.floor(probWithSP.length / 8)} />
              <YAxis yAxisId="prob" domain={[0, 1]} tick={{ fill: '#f59e0b', fontSize: 11 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
              <YAxis yAxisId="sp" orientation="right" tick={{ fill: '#60a5fa', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#1a1a2e', border: '1px solid #333' }}
                formatter={(value: number, name: string) => {
                  if (name === 'probability') return [`${(value * 100).toFixed(1)}%`, '崩盘概率'];
                  return [value?.toFixed(0), 'S&P 500'];
                }}
              />
              <ReferenceLine yAxisId="prob" y={0.5} stroke="#ef4444" strokeDasharray="5 5" label={{ value: "50% 阈值", fill: '#ef4444', fontSize: 11 }} />
              <Area yAxisId="prob" dataKey="probability" fill="#f59e0b" fillOpacity={0.3} stroke="#f59e0b" strokeWidth={1.5} />
              <Line yAxisId="sp" dataKey="sp500" stroke="#60a5fa" strokeWidth={1.5} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Threshold Analysis */}
      <section className="lab-card">
        <h2>阈值选择分析</h2>
        <p className="lab-card-desc">不同概率阈值下的精确率、召回率和报警频率。阈值越高误报越少但可能漏报</p>
        <div className="lab-table-wrap">
          <table className="lab-table">
            <thead>
              <tr>
                <th>阈值</th>
                <th>精确率</th>
                <th>召回率</th>
                <th>F1</th>
                <th>报警天数</th>
                <th>报警比例</th>
              </tr>
            </thead>
            <tbody>
              {threshold_analysis.map(row => (
                <tr key={row.threshold} className={row.threshold === 0.5 ? 'lab-row-highlight' : ''}>
                  <td className="lab-td-mono">{(row.threshold * 100).toFixed(0)}%</td>
                  <td style={{ color: row.precision > 0.3 ? '#22c55e' : row.precision > 0.15 ? '#f59e0b' : '#ef4444' }}>
                    {(row.precision * 100).toFixed(1)}%
                  </td>
                  <td style={{ color: row.recall > 0.7 ? '#22c55e' : row.recall > 0.4 ? '#f59e0b' : '#ef4444' }}>
                    {(row.recall * 100).toFixed(1)}%
                  </td>
                  <td>{row.f1.toFixed(3)}</td>
                  <td>{row.alert_days} / {row.total_days}</td>
                  <td>{row.alert_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Feature Importance */}
      <section className="lab-card">
        <h2>特征重要性</h2>
        <p className="lab-card-desc">正值=预示下跌，负值=预示安全。变化速度类特征（momentum）普遍比绝对水平更重要</p>
        <div className="lab-chart-tall">
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={feature_importance.slice(0, 15)} layout="vertical" margin={{ top: 5, right: 30, left: 140, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis type="number" tick={{ fill: '#999', fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" tick={{ fill: '#ccc', fontSize: 11 }}
                tickFormatter={(v: string) => FEATURE_LABELS[v] || v} width={130} />
              <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #333' }}
                formatter={(v: number) => [v.toFixed(4), '权重']} />
              <ReferenceLine x={0} stroke="#666" />
              <Bar dataKey="weight">
                {feature_importance.slice(0, 15).map((entry, i) => (
                  <Cell key={i} fill={entry.weight > 0 ? '#ef4444' : '#22c55e'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Backtest Events */}
      <section className="lab-card">
        <h2>历史事件回测</h2>
        <p className="lab-card-desc">模型在已知历史大跌中的预警表现（样本外测试）</p>
        {events_backtest.length === 0 ? (
          <p className="lab-no-data">暂无样本外事件数据</p>
        ) : (
          <div className="lab-events-grid">
            {events_backtest.map(evt => (
              <div key={evt.name} className="lab-event-card">
                <div className="lab-event-name">{evt.name}</div>
                <div className="lab-event-drop" style={{ color: '#ef4444' }}>
                  跌幅 {evt.drop_pct}%
                </div>
                <div className="lab-event-detail">
                  {evt.first_alert_date ? (
                    <>
                      <span className="lab-event-success">提前 {evt.lead_days} 天预警</span>
                      <span className="lab-event-date">首次报警: {evt.first_alert_date}</span>
                      <span>最高概率: {(evt.max_probability * 100).toFixed(0)}%</span>
                    </>
                  ) : (
                    <span className="lab-event-miss">未触发预警 (最高 {(evt.max_probability * 100).toFixed(0)}%)</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* ROC Curve */}
      <section className="lab-card">
        <h2>ROC 曲线 (AUC = {model_info.roc_auc})</h2>
        <p className="lab-card-desc">越靠左上角越好。0.5为随机猜测基线</p>
        <div className="lab-chart-square">
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={metrics.roc_curve} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="fpr" label={{ value: 'False Positive Rate', fill: '#999', position: 'bottom', offset: 0 }} tick={{ fill: '#999', fontSize: 11 }} />
              <YAxis label={{ value: 'True Positive Rate', fill: '#999', angle: -90, position: 'insideLeft' }} tick={{ fill: '#999', fontSize: 11 }} />
              <Area dataKey="tpr" fill="#8b5cf6" fillOpacity={0.3} stroke="#8b5cf6" strokeWidth={2} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#666" strokeDasharray="5 5" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Model Info */}
      <section className="lab-card lab-info-card">
        <h2>模型配置</h2>
        <div className="lab-info-grid">
          <div><span className="lab-info-key">模型</span><span>{model_info.name}</span></div>
          <div><span className="lab-info-key">目标</span><span>{model_info.target}</span></div>
          <div><span className="lab-info-key">特征数量</span><span>{model_info.features_count}</span></div>
          <div><span className="lab-info-key">训练样本</span><span>{model_info.training_samples} days</span></div>
          <div><span className="lab-info-key">测试样本</span><span>{model_info.test_samples} days</span></div>
          <div><span className="lab-info-key">正样本比例</span><span>{(model_info.positive_rate * 100).toFixed(1)}%</span></div>
          <div><span className="lab-info-key">训练期</span><span>{model_info.train_period}</span></div>
          <div><span className="lab-info-key">测试期</span><span>{model_info.test_period}</span></div>
          <div><span className="lab-info-key">数据更新</span><span>{model_info.last_updated}</span></div>
        </div>
      </section>

      <footer className="lab-footer">
        <p>此页面用于模型研发监督，数据每日更新。预测结果不构成投资建议。</p>
      </footer>
    </div>
  );
}
