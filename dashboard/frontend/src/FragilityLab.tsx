import { useState, useEffect } from 'react';
import { fetchDataJson } from './api';
import { ResearchTrackNotice } from './components/ResearchTrackNotice';

interface FragilityModelResult {
  model_name: string;
  target_name: string;
  n_folds: number;
  r2_mean: number;
  r2_std: number;
  rank_corr_mean: number;
  rank_corr_std: number;
  mse_mean: number;
}

interface FragilityData {
  title: string;
  hypothesis: string;
  data_range: string;
  n_samples: number;
  targets: {
    name: string;
    description: string;
    stats: { mean: number; std: number; min: number; max: number };
  }[];
  walk_forward_results: FragilityModelResult[];
  verdict: string[];
}

export function FragilityLab() {
  const [data, setData] = useState<FragilityData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDataJson<FragilityData>('fragility_metrics.json')
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="lab-container">
      <header className="lab-header">
        <div className="lab-title-row">
          <h1>Ch.3 Fragility & Anomaly Detection</h1>
          <span className="lab-badge-version">Research</span>
        </div>
      </header>

      <ResearchTrackNotice track="risk-model" />

      <section className="lab-card" style={{ borderLeft: '4px solid #8b5cf6', background: '#faf5ff' }}>
        <h2 style={{ color: '#6b21a8', margin: '0 0 10px' }}>研究方向</h2>
        <div style={{ fontSize: 13, lineHeight: 1.9, color: '#374151' }}>
          <p style={{ margin: '0 0 8px' }}>
            <strong>核心转变：</strong>不再预测「是否崩盘」（binary），改为度量「市场脆弱性」（continuous）。
          </p>
          <p style={{ margin: '0 0 12px', color: '#6b7280' }}>
            动机：崩盘触发源不可预测（关税、疫情、杠杆爆仓），但脆弱性是可观测的市场结构属性。
            脆弱的市场遇到任何冲击都会放大；健康的市场能吸收冲击。
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
            <div style={{ padding: '10px 12px', background: '#fff', borderRadius: 8, border: '1px solid #e9d5ff' }}>
              <div style={{ fontWeight: 700, color: '#7c3aed', fontSize: 12 }}>Target A: Vol Surprise</div>
              <div style={{ fontSize: 11, color: '#6b7280', marginTop: 4 }}>
                realized_vol_20d / current_VIX<br/>高值 = 实际波动超过预期 = 市场失控
              </div>
            </div>
            <div style={{ padding: '10px 12px', background: '#fff', borderRadius: 8, border: '1px solid #e9d5ff' }}>
              <div style={{ fontWeight: 700, color: '#7c3aed', fontSize: 12 }}>Target B: Continuous Max DD</div>
              <div style={{ fontSize: 11, color: '#6b7280', marginTop: 4 }}>
                max_drawdown_20d（连续值，不二值化）<br/>直接度量未来损失幅度
              </div>
            </div>
            <div style={{ padding: '10px 12px', background: '#fff', borderRadius: 8, border: '1px solid #e9d5ff' }}>
              <div style={{ fontWeight: 700, color: '#7c3aed', fontSize: 12 }}>Target C: Structural Fragility</div>
              <div style={{ fontSize: 11, color: '#6b7280', marginTop: 4 }}>
                absorption_ratio × VIX_percentile<br/>耦合度高 × 恐慌水平 = 结构性脆弱
              </div>
            </div>
          </div>
          <p style={{ margin: '12px 0 0', fontSize: 12, color: '#6b7280' }}>
            仓位公式：<code>position_size = base × (1 - fragility_score)</code> — 无需选阈值，天然连续。
          </p>
        </div>
      </section>

      {loading && <div className="loading">Loading fragility experiment data...</div>}

      {data && (
        <>
          <section className="lab-card">
            <h2>Walk-Forward Results ({data.data_range})</h2>
            <p className="lab-card-desc">{data.n_samples} samples · {data.walk_forward_results[0]?.n_folds ?? '?'} folds</p>
            <div className="lab-table-wrap">
              <table className="lab-table">
                <thead>
                  <tr>
                    <th>Model × Target</th>
                    <th>Rank Corr ↑</th>
                    <th>R² ↑</th>
                    <th>MSE ↓</th>
                  </tr>
                </thead>
                <tbody>
                  {data.walk_forward_results
                    .sort((a, b) => b.rank_corr_mean - a.rank_corr_mean)
                    .map((r, i) => (
                      <tr key={i}>
                        <td style={{ fontWeight: 600 }}>
                          {i === 0 && <span className="ab-best-tag">BEST</span>}
                          {r.model_name} · {r.target_name}
                        </td>
                        <td className="lab-td-mono" style={i === 0 ? { color: '#16a34a', fontWeight: 700 } : undefined}>
                          {r.rank_corr_mean.toFixed(4)} ± {r.rank_corr_std.toFixed(4)}
                        </td>
                        <td className="lab-td-mono">{r.r2_mean.toFixed(4)} ± {r.r2_std.toFixed(4)}</td>
                        <td className="lab-td-mono">{r.mse_mean.toFixed(4)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </section>

          {data.verdict.length > 0 && (
            <section className="lab-card">
              <h2>Verdict</h2>
              <div style={{ padding: '10px 14px', background: '#f0fdf4', borderRadius: 8, border: '1px solid #86efac' }}>
                {data.verdict.map((v, i) => (
                  <div key={i} style={{ fontSize: 13, color: '#374151', lineHeight: 1.8 }}>• {v}</div>
                ))}
              </div>
            </section>
          )}
        </>
      )}

      {!loading && !data && (
        <section className="lab-card">
          <p style={{ color: '#6b7280', fontSize: 13 }}>
            实验尚未运行。数据将在 <code>experiment_fragility.py</code> 完成后出现。
          </p>
        </section>
      )}
    </div>
  );
}
