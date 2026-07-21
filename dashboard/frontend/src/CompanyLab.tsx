import { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { fetchDataJson } from './api';
import { ValuationPanel, type ValuationData } from './ValuationLab';
import { FundamentalsPanel, type FundamentalsData } from './FundamentalsLab';
import { companyTabFromHash, hashForCompanyTab, type CompanyTab } from './siteNav';
import { pageFade, staggerContainer, staggerItem, tabFade } from './motionPresets';

function fmtUsd(n?: number | null, digits = 2) {
  if (n == null || Number.isNaN(n)) return '—';
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: digits })}`;
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

export function CompanyLab() {
  const [tab, setTab] = useState<CompanyTab>(() => companyTabFromHash(window.location.hash));
  const [valuation, setValuation] = useState<ValuationData | null>(null);
  const [fundamentals, setFundamentals] = useState<FundamentalsData | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const onHashChange = () => setTab(companyTabFromHash(window.location.hash));
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    Promise.all([
      fetchDataJson<ValuationData>('nvda_valuation.json'),
      fetchDataJson<FundamentalsData>('nvda_fundamentals.json'),
    ])
      .then(([v, f]) => {
        setValuation(v);
        setFundamentals(f);
      })
      .catch(e => setErr(String(e)));
  }, []);

  const switchTab = (next: CompanyTab) => {
    const hash = hashForCompanyTab(next);
    if (window.location.hash !== hash) {
      window.location.hash = hash.slice(1);
    }
    setTab(next);
  };

  if (err) {
    return (
      <div className="lab-container">
        <div className="lab-card"><h2>加载失败</h2><pre>{err}</pre></div>
      </div>
    );
  }

  if (!valuation || !fundamentals) {
    return <div className="lab-container"><div className="loading">Loading company research…</div></div>;
  }

  const vs = valuation.snapshot;
  const fs = fundamentals.snapshot;
  const asOf = valuation.as_of || fundamentals.as_of;

  return (
    <div className="lab-container val-lab">
      <motion.header className="lab-header" {...pageFade}>
        <div>
          <h1>Company Research · {valuation.ticker}</h1>
          <p className="lab-subtitle">
            {valuation.name} · DCF / 相对估值 / 三表财报 · 更新 {asOf}
          </p>
        </div>
        <div className="lab-model-badge">
          <span className="lab-badge-version">{valuation.ticker}</span>
        </div>
      </motion.header>

      <motion.div className="lab-track-notice" {...pageFade} transition={{ ...pageFade.animate.transition, delay: 0.04 }}>
        <span className="lab-track-label">研究线</span>
        <span className="lab-track-desc">基本面研究（估值 + 财报），独立于 Alpha Deck 因子线与 Risk 崩盘模型</span>
        <span className="lab-track-sep">·</span>
        <span className="lab-track-hint">{valuation.disclaimer}</span>
      </motion.div>

      <motion.section className="lab-card val-verdict" variants={staggerItem} initial="initial" animate="animate">
        <div className="val-kpi-row">
          <div className="val-kpi"><span>现价</span><strong>{fmtUsd(vs.price ?? fs.price)}</strong></div>
          <div className="val-kpi"><span>市值</span><strong>{fmtB(vs.market_cap ?? fs.market_cap)}</strong></div>
          <div className="val-kpi"><span>LTM 收入</span><strong>{fmtB(vs.ltm_revenue ?? fs.ltm_revenue)}</strong></div>
          <div className="val-kpi"><span>LTM 增速</span><strong>{fmtPct(vs.ltm_revenue_growth ?? fs.ltm_revenue_growth)}</strong></div>
          <div className="val-kpi"><span>毛利率</span><strong>{fmtPct(vs.gross_margin ?? fs.gross_margin)}</strong></div>
          <div className="val-kpi"><span>经营利润率</span><strong>{fmtPct(vs.operating_margin ?? fs.operating_margin)}</strong></div>
          <div className="val-kpi"><span>Fwd P/E</span><strong>{fmtMult(vs.forward_pe)}</strong></div>
          <div className="val-kpi"><span>Beta</span><strong>{vs.beta.toFixed(2)}</strong></div>
          <div className="val-kpi"><span>TTM EPS</span><strong>{fmtUsd(fs.trailing_eps)}</strong></div>
          <div className="val-kpi"><span>Fwd EPS</span><strong>{fmtUsd(fs.forward_eps ?? vs.forward_eps)}</strong></div>
        </div>
      </motion.section>

      <div className="val-mode-toggle company-tab-bar">
        <button
          type="button"
          className={tab === 'valuation' ? 'nav-btn nav-active' : 'nav-btn'}
          onClick={() => switchTab('valuation')}
        >
          估值 Valuation
        </button>
        <button
          type="button"
          className={tab === 'fundamentals' ? 'nav-btn nav-active' : 'nav-btn'}
          onClick={() => switchTab('fundamentals')}
        >
          财报 Fundamentals
        </button>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={tab}
          className="company-tab-panel"
          variants={tabFade}
          initial="initial"
          animate="animate"
          exit="exit"
        >
          <motion.div variants={staggerContainer} initial="initial" animate="animate">
            {tab === 'valuation'
              ? <ValuationPanel data={valuation} />
              : <FundamentalsPanel data={fundamentals} />}
          </motion.div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

export default CompanyLab;
