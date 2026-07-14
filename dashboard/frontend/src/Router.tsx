import { useState, useEffect, lazy, Suspense } from 'react';
import { Home } from './Home';
import { Breadcrumb } from './components/Breadcrumb';
import { pageFromHash, hashForPage, SITE_NAV, PageId } from './siteNav';

const App = lazy(() => import('./App'));
const PredictionLab = lazy(() => import('./PredictionLab').then(m => ({ default: m.PredictionLab })));
const Phase3Lab = lazy(() => import('./Phase3Lab').then(m => ({ default: m.Phase3Lab })));
const MetricLab = lazy(() => import('./MetricLab').then(m => ({ default: m.MetricLab })));
const AgentLab = lazy(() => import('./AgentLab').then(m => ({ default: m.AgentLab })));
const ProjectBoard = lazy(() => import('./ProjectBoard').then(m => ({ default: m.ProjectBoard })));

const RISK_CHILDREN: PageId[] = ['ch1', 'ch2', 'ch2_1'];

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
      </nav>

      {page !== 'home' && (
        <div className="page-context">
          <Breadcrumb page={page} onNavigate={navigate} />
          {isRiskChild && (
            <div className="page-context-meta">
              {SITE_NAV[page].subtitle}
            </div>
          )}
        </div>
      )}

      <Suspense fallback={<div className="loading">Loading...</div>}>
        {page === 'home' && <Home onNavigate={navigate} />}
        {page === 'pipeline' && <ProjectBoard />}
        {page === 'risk' && <App />}
        {page === 'ch1' && <PredictionLab />}
        {page === 'ch2' && <Phase3Lab />}
        {page === 'ch2_1' && <MetricLab />}
        {page === 'ch3' && <AgentLab />}
      </Suspense>
    </>
  );
}
