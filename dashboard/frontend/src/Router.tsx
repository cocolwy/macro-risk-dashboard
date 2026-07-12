import { useState, useEffect } from 'react';
import App from './App';
import { PredictionLab } from './PredictionLab';

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
      {page === 'dashboard' ? <App /> : <PredictionLab />}
    </>
  );
}
