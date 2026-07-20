import { useEffect, useState } from 'react';
import { fetchDataJson } from './api';

interface StatementTable {
  periods: string[];
  rows: { label: string; values: (number | null)[] }[];
}

interface FundamentalsData {
  title: string;
  subtitle: string;
  ticker: string;
  name: string;
  as_of: string;
  disclaimer: string;
  snapshot: {
    price: number;
    market_cap: number;
    ltm_revenue: number;
    ltm_revenue_growth?: number | null;
    ltm_ebit?: number;
    ltm_ebitda?: number;
    gross_margin?: number | null;
    operating_margin?: number | null;
    forward_eps?: number | null;
    trailing_eps?: number | null;
    cash?: number;
    total_debt?: number;
  };
  annual: {
    income: StatementTable;
    cashflow: StatementTable;
    balance: StatementTable;
  };
  quarterly: {
    income: StatementTable;
    cashflow: StatementTable;
  };
  estimates?: {
    earnings?: Record<string, string>[];
    revenue?: Record<string, string>[];
  };
}

function fmtUsd(n?: number | null, digits = 2) {
  if (n == null || Number.isNaN(n)) return '—';
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: digits })}`;
}

function fmtCell(v: number | null, label: string) {
  if (v == null || Number.isNaN(v)) return '—';
  if (label.includes('EPS') || Math.abs(v) < 1000) {
    return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  const abs = Math.abs(v);
  if (abs >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function fmtPct(n?: number | null) {
  if (n == null || Number.isNaN(n)) return '—';
  return `${(n * 100).toFixed(1)}%`;
}

function StatementBlock({ title, table }: { title: string; table: StatementTable }) {
  if (!table?.periods?.length) {
    return (
      <section className="lab-card">
        <h2 className="lab-subsection-title">{title}</h2>
        <p className="val-muted">暂无数据</p>
      </section>
    );
  }
  return (
    <section className="lab-card">
      <h2 className="lab-subsection-title">{title}</h2>
      <div className="val-table-scroll">
        <table className="val-table">
          <thead>
            <tr>
              <th>科目</th>
              {table.periods.map(p => <th key={p}>{p}</th>)}
            </tr>
          </thead>
          <tbody>
            {table.rows.map(r => (
              <tr key={r.label}>
                <td><strong>{r.label}</strong></td>
                {r.values.map((v, i) => (
                  <td key={i}>{fmtCell(v, r.label)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function FundamentalsLab() {
  const [data, setData] = useState<FundamentalsData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [mode, setMode] = useState<'annual' | 'quarterly'>('annual');

  useEffect(() => {
    fetchDataJson<FundamentalsData>('nvda_fundamentals.json')
      .then(setData)
      .catch(e => setErr(String(e)));
  }, []);

  if (err) {
    return (
      <div className="lab-container">
        <div className="lab-card"><h2>加载失败</h2><pre>{err}</pre></div>
      </div>
    );
  }
  if (!data) {
    return <div className="lab-container"><div className="loading">Loading fundamentals…</div></div>;
  }

  const s = data.snapshot;

  return (
    <div className="lab-container val-lab">
      <header className="lab-header">
        <div>
          <h1>{data.title}</h1>
          <p className="lab-subtitle">{data.subtitle} · 更新 {data.as_of}</p>
        </div>
        <div className="lab-model-badge">
          <span className="lab-badge-version">{data.ticker}</span>
          <a className="lab-track-link" href="#valuation">估值 Valuation →</a>
        </div>
      </header>

      <div className="lab-track-notice">
        <span className="lab-track-label">财报页</span>
        <span className="lab-track-desc">{data.name} · 利润表 / 现金流 / 资产负债表</span>
        <span className="lab-track-sep">·</span>
        <span className="lab-track-hint">{data.disclaimer}</span>
      </div>

      <section className="lab-card val-verdict">
        <div className="val-kpi-row">
          <div className="val-kpi"><span>现价</span><strong>{fmtUsd(s.price)}</strong></div>
          <div className="val-kpi"><span>市值</span><strong>{fmtCell(s.market_cap, '')}</strong></div>
          <div className="val-kpi"><span>LTM 收入</span><strong>{fmtCell(s.ltm_revenue, '')}</strong></div>
          <div className="val-kpi"><span>LTM 增速</span><strong>{fmtPct(s.ltm_revenue_growth)}</strong></div>
          <div className="val-kpi"><span>毛利率</span><strong>{fmtPct(s.gross_margin)}</strong></div>
          <div className="val-kpi"><span>经营利润率</span><strong>{fmtPct(s.operating_margin)}</strong></div>
          <div className="val-kpi"><span>TTM EPS</span><strong>{fmtUsd(s.trailing_eps)}</strong></div>
          <div className="val-kpi"><span>Fwd EPS</span><strong>{fmtUsd(s.forward_eps)}</strong></div>
        </div>
      </section>

      <div className="val-mode-toggle">
        <button
          type="button"
          className={mode === 'annual' ? 'nav-btn nav-active' : 'nav-btn'}
          onClick={() => setMode('annual')}
        >
          年度 Annual
        </button>
        <button
          type="button"
          className={mode === 'quarterly' ? 'nav-btn nav-active' : 'nav-btn'}
          onClick={() => setMode('quarterly')}
        >
          季度 Quarterly
        </button>
      </div>

      {mode === 'annual' ? (
        <>
          <StatementBlock title="利润表 · Income Statement（年度）" table={data.annual.income} />
          <StatementBlock title="现金流量表 · Cash Flow（年度）" table={data.annual.cashflow} />
          <StatementBlock title="资产负债表 · Balance Sheet（年度）" table={data.annual.balance} />
        </>
      ) : (
        <>
          <StatementBlock title="利润表 · Income Statement（季度）" table={data.quarterly.income} />
          <StatementBlock title="现金流量表 · Cash Flow（季度）" table={data.quarterly.cashflow} />
        </>
      )}

      {(data.estimates?.earnings?.length || data.estimates?.revenue?.length) ? (
        <section className="lab-card">
          <h2 className="lab-subsection-title">分析师预期（原始表）</h2>
          {data.estimates.earnings?.length ? (
            <>
              <h3 className="val-mini-title">Earnings Estimate</h3>
              <div className="val-table-scroll">
                <table className="val-table">
                  <thead>
                    <tr>
                      {Object.keys(data.estimates.earnings[0]).map(k => <th key={k}>{k}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {data.estimates.earnings.map((row, i) => (
                      <tr key={i}>
                        {Object.values(row).map((v, j) => <td key={j}>{v}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : null}
          {data.estimates.revenue?.length ? (
            <>
              <h3 className="val-mini-title">Revenue Estimate</h3>
              <div className="val-table-scroll">
                <table className="val-table">
                  <thead>
                    <tr>
                      {Object.keys(data.estimates.revenue[0]).map(k => <th key={k}>{k}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {data.estimates.revenue.map((row, i) => (
                      <tr key={i}>
                        {Object.values(row).map((v, j) => <td key={j}>{v}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}

export default FundamentalsLab;
