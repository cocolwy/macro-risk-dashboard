import { useState, useEffect, lazy, Suspense } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Home } from './Home';
import { pageFromHash, hashForPage, SITE_NAV, SIDEBAR_TREE, PageId } from './siteNav';

const pageTransition = {
  initial: { opacity: 0, y: 6 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.25, ease: [0.25, 0.1, 0.25, 1] as [number, number, number, number] } },
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
const ResearchSurvey = lazy(() => import('./ResearchSurvey').then(m => ({ default: m.ResearchSurvey })));

function getInitialPage(): PageId {
  return pageFromHash(window.location.hash);
}

function Sidebar({ page, onNavigate }: { page: PageId; onNavigate: (p: PageId) => void }) {
  const isChildOf = (parentId: PageId) => {
    const item = SITE_NAV[page];
    return item.parent === parentId;
  };

  return (
    <aside className="sidebar">
      <button
        className={`sidebar-home ${page === 'home' ? 'sidebar-active' : ''}`}
        onClick={() => onNavigate('home')}
      >
        <span className="sidebar-home-icon">◆</span>
        <span className="sidebar-home-text">Quant Hub</span>
      </button>

      {SIDEBAR_TREE.map(group => (
        <div key={group.label} className="sidebar-group">
          <div className="sidebar-group-label">{group.label}</div>
          {group.items.map(({ id, children }) => {
            const item = SITE_NAV[id];
            const isActive = page === id;
            const hasActiveChild = children?.includes(page) ?? false;
            const showChildren = children && (isActive || hasActiveChild || isChildOf(id));

            return (
              <div key={id} className="sidebar-tree-node">
                <button
                  className={`sidebar-item ${isActive ? 'sidebar-active' : ''} ${hasActiveChild ? 'sidebar-parent-active' : ''}`}
                  onClick={() => onNavigate(id)}
                >
                  <span className="sidebar-item-title">{item.title}</span>
                  {item.badge && <span className="sidebar-badge">{item.badge}</span>}
                  {children && (
                    <span className={`sidebar-chevron ${showChildren ? 'sidebar-chevron-open' : ''}`}>›</span>
                  )}
                </button>

                {children && (
                  <div className={`sidebar-children ${showChildren ? 'sidebar-children-open' : ''}`}>
                    {children.map(childId => {
                      const child = SITE_NAV[childId];
                      return (
                        <button
                          key={childId}
                          className={`sidebar-item sidebar-item-child ${page === childId ? 'sidebar-active' : ''}`}
                          onClick={() => onNavigate(childId)}
                        >
                          <span className="sidebar-item-title">{child.title}</span>
                          {child.badge && <span className="sidebar-badge">{child.badge}</span>}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </aside>
  );
}

function usePersistedState(key: string, initial: boolean): [boolean, (v: boolean | ((prev: boolean) => boolean)) => void] {
  const [value, setValue] = useState(() => {
    try { const s = localStorage.getItem(key); return s !== null ? s === 'true' : initial; }
    catch { return initial; }
  });
  const set = (v: boolean | ((prev: boolean) => boolean)) => {
    setValue(prev => {
      const next = typeof v === 'function' ? v(prev) : v;
      try { localStorage.setItem(key, String(next)); } catch { /* noop */ }
      return next;
    });
  };
  return [value, set];
}

export function Router() {
  const [page, setPage] = useState<PageId>(getInitialPage);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [collapsed, setCollapsed] = usePersistedState('sidebar-collapsed', false);

  useEffect(() => {
    const onHashChange = () => setPage(getInitialPage());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const navigate = (p: PageId) => {
    window.location.hash = hashForPage(p);
    setPage(p);
    setSidebarOpen(false);
  };

  return (
    <div className={`layout ${collapsed ? 'layout-collapsed' : ''}`}>
      {/* Mobile hamburger */}
      <button
        className="sidebar-toggle"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        aria-label="Toggle navigation"
      >
        <span className={`sidebar-toggle-icon ${sidebarOpen ? 'sidebar-toggle-open' : ''}`} />
      </button>

      {sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      <div className={`sidebar-wrapper ${sidebarOpen ? 'sidebar-wrapper-open' : ''} ${collapsed ? 'sidebar-wrapper-collapsed' : ''}`}>
        <Sidebar page={page} onNavigate={navigate} />
        <button
          className="sidebar-collapse-btn"
          onClick={() => setCollapsed(c => !c)}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <span className={`sidebar-collapse-icon ${collapsed ? 'sidebar-collapse-icon-flip' : ''}`}>‹</span>
        </button>
      </div>

      <main className="layout-main">
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
              {page === 'survey' && <ResearchSurvey />}
            </Suspense>
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
