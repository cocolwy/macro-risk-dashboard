import { useState, useEffect, lazy, Suspense } from 'react';

const App = lazy(() => import('./App'));
const PredictionLab = lazy(() => import('./PredictionLab').then(m => ({ default: m.PredictionLab })));
const Phase3Lab = lazy(() => import('./Phase3Lab').then(m => ({ default: m.Phase3Lab })));
const ProjectBoard = lazy(() => import('./ProjectBoard').then(m => ({ default: m.ProjectBoard })));

type Page = 'dashboard' | 'lab' | 'phase3' | 'board';

function getInitialPage(): Page {
  const hash = window.location.hash.replace('#', '');
  if (hash === 'lab') return 'lab';
  if (hash === 'phase3') return 'phase3';
  if (hash === 'board') return 'board';
  return 'dashboard';
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
          Ch.1 Linear Models
        </button>
        <button
          className={`nav-btn ${page === 'phase3' ? 'nav-active' : ''}`}
          onClick={() => navigate('phase3')}
        >
          Ch.2 Model Evolution
          <span className="nav-badge-dev">NEW</span>
        </button>
        <button
          className={`nav-btn ${page === 'board' ? 'nav-active' : ''}`}
          onClick={() => navigate('board')}
        >
          Quant Project Board
        </button>
      </nav>
      <Suspense fallback={<div className="loading">Loading...</div>}>
        {page === 'dashboard' ? <App /> : page === 'lab' ? <PredictionLab /> : page === 'phase3' ? <Phase3Lab /> : <ProjectBoard />}
      </Suspense>
    </>
  );
}
