import { useState } from 'react';
import { motion } from 'framer-motion';
import { staggerItem } from './motionPresets';

interface StatementTable {
  periods: string[];
  rows: { label: string; values: (number | null)[] }[];
}

export interface FundamentalsData {
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

function StatementBlock({ title, table }: { title: string; table: StatementTable }) {
  if (!table?.periods?.length) {
    return (
      <motion.section className="lab-card" variants={staggerItem}>
        <h2 className="lab-subsection-title">{title}</h2>
        <p className="val-muted">暂无数据</p>
      </motion.section>
    );
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

  return (
    <motion.section className="lab-card" variants={staggerItem}>
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
    </motion.section>
  );
}

export function FundamentalsPanel({ data }: { data: FundamentalsData }) {
  const [mode, setMode] = useState<'annual' | 'quarterly'>('annual');

  return (
    <>
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
        <motion.section className="lab-card" variants={staggerItem}>
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
        </motion.section>
      ) : null}
    </>
  );
}
