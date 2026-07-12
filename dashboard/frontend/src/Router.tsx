import { useState, useEffect, lazy, Suspense } from 'react';

const App = lazy(() => import('./App'));
const PredictionLab = lazy(() => import('./PredictionLab').then(m => ({ default: m.PredictionLab })));

type Page = 'dashboard' | 'lab';

function getInitialPage(): Page {
  const hash = window.location.hash.replace('#', '');
  return hash === 'lab' ? 'lab' : 'dashboard';
}

export function Router() {
  const [page, setPage] = useState<Page>(getInitialPage);

  useEffect(() => {
    const onHashChange = () => setPage(getInitialPage());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const navigate = (p: Page) => {
    window.location.hash = p === 'dashboard' ? '' : p;
    setPage(p);
  };

  return (
    <>
      <nav className="top-nav">
        <button
          className={`nav-btn ${page === 'dashboard' ? 'nav-active' : ''}`}
          onClick={() => navigate('dashboard')}
        >
          Macro Risk Monitor
        </button>
        <button
          className={`nav-btn ${page === 'lab' ? 'nav-active' : ''}`}
          onClick={() => navigate('lab')}
        >
          Prediction Lab
          <span className="nav-badge-dev">DEV</span>
        </button>
      </nav>
      <Suspense fallback={<div className="loading">Loading...</div>}>
        {page === 'dashboard' ? <App /> : <PredictionLab />}
      </Suspense>
    </>
  );
}
