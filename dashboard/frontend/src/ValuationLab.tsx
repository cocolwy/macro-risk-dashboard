import { useEffect, useState } from 'react';
import { fetchDataJson } from './api';

interface MethodRow {
  name: string;
  price: number | null;
  weight: number;
  note: string;
}

interface ProjectionRow {
  year: number;
  growth: number;
  revenue: number;
  ebit: number;
  nopat: number;
  da: number;
  capex: number;
  nwc: number;
  fcff: number;
}

interface GlossaryItem {
  term: string;
  zh: string;
  def: string;
}

interface ValuationData {
  title: string;
  subtitle: string;
  ticker: string;
  name: string;
  as_of: string;
  disclaimer: string;
  snapshot: {
    price: number;
    market_cap: number;
    beta: number;
    sector?: string;
    industry?: string;
    chg_3m?: number | null;
    chg_12m?: number | null;
    ltm_revenue: number;
    ltm_revenue_growth?: number | null;
    gross_margin?: number | null;
    operating_margin?: number | null;
    forward_eps?: number | null;
    forward_pe?: number | null;
    ev_ebitda?: number | null;
  };
  verdict: {
    blended: number;
    dcf: number;
    relative: number;
    current: number;
    upside: number;
    headline: string;
  };
  methods: MethodRow[];
  dcf: {
    assumptions: Record<string, number | string | number[]>;
    projection: ProjectionRow[];
    bridge: Record<string, number>;
    sensitivity: {
      wacc_grid: number[];
      g_grid: number[];
      rows: { wacc: number; prices: number[] }[];
      base_wacc: number;
      base_g: number;
    };
    scenarios: { name: string; price: number; levers: string }[];
  };
  relative: {
    peers: {
      ticker: string;
      pe_fwd?: number | null;
      ev_rev?: number | null;
      ev_ebitda?: number | null;
      gross_margin?: number | null;
      revenue_growth?: number | null;
    }[];
    medians: { pe_fwd?: number | null; ev_rev?: number | null; ev_ebitda?: number | null };
    premium: number;
    self_multiples: Record<string, number | null | undefined>;
  };
  risks: string[];
  glossary: GlossaryItem[];
  dcf_explain: { what: string; steps: string[]; intuition: string };
}

function fmtUsd(n?: number | null, digits = 0) {
  if (n == null || Number.isNaN(n)) return '—';
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits })}`;
}

function fmtB(n?: number | null) {
  if (n == null || Number.isNaN(n)) return '—';
  return `$${(n / 1e9).toFixed(1)}B`;
}

function fmtPct(n?: number | null, digits = 1) {
  if (n == null || Number.isNaN(n)) return '—';
  return `${(n * 100).toFixed(digits)}%`;
}

function fmtMult(n?: number | null) {
  if (n == null || Number.isNaN(n)) return '—';
  return `${n.toFixed(1)}×`;
}

const ASSUMPTION_ROWS: { key: string; label: string; hint: string; format: (a: Record<string, unknown>) => string }[] = [
  { key: 'y1_growth', label: 'Y1 收入增速', hint: '显式预测第 1 年收入增长率（共识/历史起算）', format: a => fmtPct(a.y1_growth as number) },
  { key: 'growth_path', label: 'Y1→Y5 增速路径', hint: '从高速增长淡出到接近永久增长', format: a => ((a.growth_path as number[]) || []).map(g => fmtPct(g, 0)).join(' → ') },
  { key: 'ebit_margin', label: 'EBIT margin', hint: '经营利润率（近 3 年中位）', format: a => fmtPct(a.ebit_margin as number) },
  { key: 'da_pct_rev', label: 'D&A / 收入', hint: '折旧摊销占收入比', format: a => fmtPct(a.da_pct_rev as number) },
  { key: 'capex_pct_rev', label: 'CapEx / 收入', hint: '资本开支占收入比', format: a => fmtPct(a.capex_pct_rev as number) },
  { key: 'nwc_pct_rev', label: 'ΔNWC / 收入', hint: '营运资金占用占收入比', format: a => fmtPct(a.nwc_pct_rev as number) },
  { key: 'tax_rate', label: '税率', hint: '有效税率（15–30% 区间约束）', format: a => fmtPct(a.tax_rate as number, 0) },
  { key: 'rf', label: 'rf（无风险利率）', hint: '10Y 美债收益率', format: a => fmtPct(a.rf as number, 2) },
  { key: 'beta', label: 'Beta (β)', hint: '相对市场波动敏感度', format: a => (a.beta as number).toFixed(2) },
  { key: 'erp', label: 'ERP', hint: '股权风险溢价', format: a => fmtPct(a.erp as number, 1) },
  { key: 'ke', label: 'ke（股权成本）', hint: 'rf + β × ERP', format: a => fmtPct(a.ke as number, 1) },
  { key: 'kd', label: 'kd（债务成本）', hint: '有效借款利率', format: a => fmtPct(a.kd as number, 1) },
  { key: 'wacc', label: 'WACC', hint: '加权平均资本成本 = 折现率', format: a => fmtPct(a.wacc as number, 1) },
  { key: 'terminal_g', label: '永久增长 g', hint: '终值期永续增长率', format: a => fmtPct(a.terminal_g as number, 1) },
  { key: 'exit_ebitda_multiple', label: '退出倍数', hint: 'Y5 EBITDA × EV/EBITDA', format: a => fmtMult(a.exit_ebitda_multiple as number) },
];

export function ValuationLab() {
  const [data, setData] = useState<ValuationData | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetchDataJson<ValuationData>('nvda_valuation.json')
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
    return <div className="lab-container"><div className="loading">Loading valuation…</div></div>;
  }

  const s = data.snapshot;
  const a = data.dcf.assumptions as Record<string, unknown>;
  const upsideCls = data.verdict.upside >= 0 ? 'val-up' : 'val-down';

  return (
    <div className="lab-container val-lab">
      <header className="lab-header">
        <div>
          <h1>{data.title}</h1>
          <p className="lab-subtitle">{data.subtitle} · 更新 {data.as_of}</p>
        </div>
        <div className="lab-model-badge">
          <span className="lab-badge-version">{data.ticker}</span>
          <a className="lab-track-link" href="#fundamentals">财报 Fundamentals →</a>
        </div>
      </header>

      <div className="lab-track-notice">
        <span className="lab-track-label">研究线</span>
        <span className="lab-track-desc">基本面估值（DCF / 相对估值），独立于 Alpha Deck 因子线与 Risk 崩盘模型</span>
        <span className="lab-track-sep">·</span>
        <span className="lab-track-hint">{data.disclaimer}</span>
      </div>

      <section className="lab-card val-verdict">
        <div className="val-verdict-main">
          <div className="val-verdict-label">混合公允价值</div>
          <div className="val-verdict-price">{fmtUsd(data.verdict.blended, 0)}</div>
          <div className={`val-verdict-upside ${upsideCls}`}>
            vs 现价 {fmtUsd(data.verdict.current, 2)} · {fmtPct(data.verdict.upside, 0)}
          </div>
        </div>
        <p className="val-verdict-headline">{data.verdict.headline}</p>
        <div className="val-kpi-row">
          <div className="val-kpi"><span>市值</span><strong>{fmtB(s.market_cap)}</strong></div>
          <div className="val-kpi"><span>LTM 收入</span><strong>{fmtB(s.ltm_revenue)}</strong></div>
          <div className="val-kpi"><span>LTM 增速</span><strong>{fmtPct(s.ltm_revenue_growth)}</strong></div>
          <div className="val-kpi"><span>毛利率</span><strong>{fmtPct(s.gross_margin)}</strong></div>
          <div className="val-kpi"><span>Fwd P/E</span><strong>{fmtMult(s.forward_pe)}</strong></div>
          <div className="val-kpi"><span>Beta</span><strong>{s.beta.toFixed(2)}</strong></div>
        </div>
      </section>

      <section className="lab-card">
        <h2 className="lab-subsection-title">三方法汇总</h2>
        <table className="val-table">
          <thead>
            <tr><th>方法</th><th>隐含价</th><th>权重</th><th>说明</th></tr>
          </thead>
          <tbody>
            {data.methods.map(m => (
              <tr key={m.name}>
                <td><strong>{m.name}</strong></td>
                <td>{m.price == null ? '—' : fmtUsd(m.price, 0)}</td>
                <td>{fmtPct(m.weight, 0)}</td>
                <td className="val-muted">{m.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="lab-card val-explain">
        <h2 className="lab-subsection-title">DCF 是什么？</h2>
        <p className="val-explain-what">{data.dcf_explain.what}</p>
        <p className="val-explain-intuition"><em>{data.dcf_explain.intuition}</em></p>
        <ol className="val-explain-steps">
          {data.dcf_explain.steps.map((step, i) => (
            <li key={i}>{step}</li>
          ))}
        </ol>
      </section>

      <section className="lab-card">
        <h2 className="lab-subsection-title">DCF 假设（Base）</h2>
        <p className="lab-section-hint">点开术语可对照下方完整名词表。</p>
        <table className="val-table">
          <thead>
            <tr><th>参数</th><th>取值</th><th>含义</th></tr>
          </thead>
          <tbody>
            {ASSUMPTION_ROWS.map(r => (
              <tr key={r.key}>
                <td><strong>{r.label}</strong></td>
                <td>{r.format(a)}</td>
                <td className="val-muted">{r.hint}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="lab-card">
        <h2 className="lab-subsection-title">5 年 FCFF 投影</h2>
        <table className="val-table">
          <thead>
            <tr>
              <th>年</th><th>增速</th><th>收入</th><th>EBIT</th><th>NOPAT</th><th>FCFF</th>
            </tr>
          </thead>
          <tbody>
            {data.dcf.projection.map(p => (
              <tr key={p.year}>
                <td>Y{p.year}</td>
                <td>{fmtPct(p.growth, 0)}</td>
                <td>{fmtB(p.revenue)}</td>
                <td>{fmtB(p.ebit)}</td>
                <td>{fmtB(p.nopat)}</td>
                <td><strong>{fmtB(p.fcff)}</strong></td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="val-bridge">
          <div>PV(FCFF) <strong>{fmtB(data.dcf.bridge.pv_fcff)}</strong></div>
          <div>PV(终值) <strong>{fmtB(data.dcf.bridge.pv_tv)}</strong></div>
          <div>企业价值 EV <strong>{fmtB(data.dcf.bridge.enterprise_value)}</strong></div>
          <div>股权价值 <strong>{fmtB(data.dcf.bridge.equity_value)}</strong></div>
          <div>隐含股价 <strong>{fmtUsd(data.dcf.bridge.implied_price, 0)}</strong></div>
          <div>终值占 EV <strong>{fmtPct(data.dcf.bridge.tv_share_of_ev)}</strong></div>
        </div>
      </section>

      <section className="lab-card">
        <h2 className="lab-subsection-title">敏感性 · WACC × g（$/股）</h2>
        <table className="val-table val-sens">
          <thead>
            <tr>
              <th>WACC \ g</th>
              {data.dcf.sensitivity.g_grid.map(g => (
                <th key={g}>{fmtPct(g, 1)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.dcf.sensitivity.rows.map(row => (
              <tr key={row.wacc}>
                <td>{fmtPct(row.wacc, 1)}</td>
                {row.prices.map((p, i) => {
                  const isBase =
                    Math.abs(row.wacc - data.dcf.sensitivity.base_wacc) < 1e-6 &&
                    Math.abs(data.dcf.sensitivity.g_grid[i] - data.dcf.sensitivity.base_g) < 1e-9;
                  return <td key={i} className={isBase ? 'val-base-cell' : ''}>{p.toFixed(0)}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="lab-card">
        <h2 className="lab-subsection-title">情景</h2>
        <table className="val-table">
          <thead>
            <tr><th>情景</th><th>隐含价</th><th>杠杆</th></tr>
          </thead>
          <tbody>
            {data.dcf.scenarios.map(sc => (
              <tr key={sc.name}>
                <td><strong>{sc.name}</strong></td>
                <td>{fmtUsd(sc.price, 0)}</td>
                <td className="val-muted">{sc.levers}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="lab-card">
        <h2 className="lab-subsection-title">同业相对估值</h2>
        <table className="val-table">
          <thead>
            <tr>
              <th>Ticker</th><th>P/E fwd</th><th>EV/Sales</th><th>EV/EBITDA</th><th>毛利率</th><th>收入增速</th>
            </tr>
          </thead>
          <tbody>
            <tr className="val-self-row">
              <td><strong>{data.ticker}</strong></td>
              <td>{fmtMult(data.relative.self_multiples.pe_fwd as number | undefined)}</td>
              <td>{fmtMult(data.relative.self_multiples.ev_rev as number | undefined)}</td>
              <td>{fmtMult(data.relative.self_multiples.ev_ebitda as number | undefined)}</td>
              <td>{fmtPct(s.gross_margin)}</td>
              <td>{fmtPct(s.ltm_revenue_growth)}</td>
            </tr>
            {data.relative.peers.map(p => (
              <tr key={p.ticker}>
                <td>{p.ticker}</td>
                <td>{fmtMult(p.pe_fwd)}</td>
                <td>{fmtMult(p.ev_rev)}</td>
                <td>{fmtMult(p.ev_ebitda)}</td>
                <td>{fmtPct(p.gross_margin)}</td>
                <td>{fmtPct(p.revenue_growth)}</td>
              </tr>
            ))}
            <tr>
              <td><strong>同业中位</strong></td>
              <td>{fmtMult(data.relative.medians.pe_fwd)}</td>
              <td>{fmtMult(data.relative.medians.ev_rev)}</td>
              <td>{fmtMult(data.relative.medians.ev_ebitda)}</td>
              <td colSpan={2} className="val-muted">相对估值溢价 ×{data.relative.premium}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section className="lab-card">
        <h2 className="lab-subsection-title">关键风险</h2>
        <ul className="val-risks">
          {data.risks.map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      </section>

      <section className="lab-card" id="glossary">
        <h2 className="lab-subsection-title">名词表 · Base 假设术语</h2>
        <div className="val-glossary">
          {data.glossary.map(g => (
            <div key={g.term} className="val-glossary-item">
              <div className="val-glossary-term">{g.term} <span>{g.zh}</span></div>
              <p>{g.def}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export default ValuationLab;
