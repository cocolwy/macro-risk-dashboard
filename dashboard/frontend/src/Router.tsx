import { useState, useEffect, lazy, Suspense } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Home } from './Home';
import { Breadcrumb } from './components/Breadcrumb';
import { pageFromHash, hashForPage, SITE_NAV, PageId } from './siteNav';

const pageTransition = {
  initial: { opacity: 0, y: 6 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.25, ease: [0.25, 0.1, 0.25, 1] } },
  exit: { opacity: 0, transition: { duration: 0.12 } },
};

const App = lazy(() => import('./App'));
const PredictionLab = lazy(() => import('./PredictionLab').then(m => ({ default: m.PredictionLab })));
const Phase3Lab = lazy(() => import('./Phase3Lab').then(m => ({ default: m.Phase3Lab })));
const FragilityLab = lazy(() => import('./FragilityLab').then(m => ({ default: m.FragilityLab })));
const MetricLab = lazy(() => import('./MetricLab').then(m => ({ default: m.MetricLab })));
const EventVolLab = lazy(() => import('./EventVolLab').then(m => ({ default: m.EventVolLab })));
const AgentLab = lazy(() => import('./AgentLab').then(m => ({ default: m.AgentLab })));
const ProjectBoard = lazy(() => import('./ProjectBoard').then(m => ({ default: m.ProjectBoard })));
const FactorLab = lazy(() => import('./FactorLab').then(m => ({ default: m.FactorLab })));
const CompanyLab = lazy(() => import('./CompanyLab').then(m => ({ default: m.CompanyLab })));

const RISK_CHILDREN: PageId[] = ['ch1', 'ch2', 'ch2_1', 'ch3_risk'];

function getInitialPage(): PageId {
  return pageFromHash(window.location.hash);
}

export function Router() {
  const [page, setPage] = useState<PageId>(getInitialPage);

  useEffect(() => {
    const onHashChange = () => setPage(getInitialPage());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const navigate = (p: PageId) => {
    window.location.hash = hashForPage(p);
    setPage(p);
  };

  const isRiskChild = RISK_CHILDREN.includes(page);

  return (
    <>
      <nav className="top-nav">
        <button
          className={`nav-btn ${page === 'home' ? 'nav-active' : ''}`}
          onClick={() => navigate('home')}
        >
          Home
        </button>

        <button
          className={`nav-btn ${page === 'pipeline' ? 'nav-active' : ''}`}
          onClick={() => navigate('pipeline')}
        >
          <span className="nav-level">L1</span>
          Pipeline
        </button>

        <button
          className={`nav-btn ${page === 'ch3' ? 'nav-active' : ''}`}
          onClick={() => navigate('ch3')}
        >
          <span className="nav-level">L1</span>
          News Agent
          {SITE_NAV.ch3.badge && <span className="nav-badge-dev">{SITE_NAV.ch3.badge}</span>}
        </button>

        <button
          className={`nav-btn ${page === 'factorlab' ? 'nav-active' : ''}`}
          onClick={() => navigate('factorlab')}
        >
          <span className="nav-level">L1</span>
          Alpha Deck
          {SITE_NAV.factorlab.badge && <span className="nav-badge-dev">{SITE_NAV.factorlab.badge}</span>}
        </button>

        <div className="nav-group">
          <button
            className={`nav-btn ${page === 'risk' ? 'nav-active' : ''}`}
            onClick={() => navigate('risk')}
          >
            <span className="nav-level">L2</span>
            Risk Dashboard
          </button>
          <div className="nav-sub">
            {RISK_CHILDREN.map(id => (
              <button
                key={id}
                className={`nav-btn nav-btn-sub ${page === id ? 'nav-active' : ''}`}
                onClick={() => navigate(id)}
              >
                <span className="nav-level">L3</span>
                {SITE_NAV[id].title.replace('Ch.', 'Ch')}
                {SITE_NAV[id].badge && <span className="nav-badge-dev">{SITE_NAV[id].badge}</span>}
              </button>
            ))}
          </div>
        </div>

        <button
          className={`nav-btn ${page === 'ch2_2' ? 'nav-active' : ''}`}
          onClick={() => navigate('ch2_2')}
        >
          <span className="nav-level">L2</span>
          Event × VIX
          {SITE_NAV.ch2_2.badge && <span className="nav-badge-dev">{SITE_NAV.ch2_2.badge}</span>}
        </button>

        <button
          className={`nav-btn ${page === 'company' ? 'nav-active' : ''}`}
          onClick={() => navigate('company')}
        >
          <span className="nav-level">L2</span>
          Company
          {SITE_NAV.company.badge && <span className="nav-badge-dev">{SITE_NAV.company.badge}</span>}
        </button>
      </nav>

      {page !== 'home' && (
        <div className="page-context">
          <Breadcrumb page={page} onNavigate={navigate} />
          {(isRiskChild || SITE_NAV[page].level === 2) && (
            <div className="page-context-meta">
              {SITE_NAV[page].subtitle}
            </div>
          )}
        </div>
      )}

      <AnimatePresence mode="wait">
        <motion.div key={page} {...pageTransition}>
          <Suspense fallback={<div className="loading">Loading...</div>}>
            {page === 'home' && <Home onNavigate={navigate} />}
            {page === 'pipeline' && <ProjectBoard />}
            {page === 'risk' && <App />}
            {page === 'ch1' && <PredictionLab />}
            {page === 'ch2' && <Phase3Lab />}
            {page === 'ch3_risk' && <FragilityLab />}
            {page === 'ch2_1' && <MetricLab />}
            {page === 'ch2_2' && <EventVolLab />}
            {page === 'ch3' && <AgentLab />}
            {page === 'factorlab' && <FactorLab />}
            {page === 'company' && <CompanyLab />}
          </Suspense>
        </motion.div>
      </AnimatePresence>
    </>
  );
}
