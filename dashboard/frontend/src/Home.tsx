import { motion } from 'framer-motion';
import { SITE_NAV, HOME_SECTIONS, HOME_TODOS, PageId } from './siteNav';

const staggerGrid = {
  animate: { transition: { staggerChildren: 0.04 } },
};
const cardPop = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.28, ease: [0.25, 0.1, 0.25, 1] } },
};

interface HomeProps {
  onNavigate: (page: PageId) => void;
}

export function Home({ onNavigate }: HomeProps) {
  return (
    <div className="home-container">
      <header className="home-header">
        <h1>{SITE_NAV.home.title}</h1>
      </header>

      <section className="home-section home-todos-section">
        <div className="home-section-title">{HOME_TODOS.title}</div>
        <p className="home-section-hint">{HOME_TODOS.hint}</p>
        <ul className="home-todo-list">
          {HOME_TODOS.items.map(item => (
            <li key={item.id} className="home-todo-item">
              <span className="home-todo-id">{item.factorId}</span>
              <div className="home-todo-body">
                <div className="home-todo-title">{item.title}</div>
                <div className="home-todo-summary">{item.summary}</div>
                <div className="home-todo-meta">
                  <span className="home-todo-stage">{item.stage}</span>
                  <code className="home-todo-case">{item.caseFile}</code>
                </div>
              </div>
              <button type="button" className="home-todo-link" onClick={() => onNavigate('factorlab')}>
                Alpha Deck →
              </button>
            </li>
          ))}
        </ul>
      </section>

      {HOME_SECTIONS.map(section => (
        <section key={section.level} className="home-section">
          <div className="home-section-title">{section.label}</div>

          <motion.div
            className={`home-grid ${section.level === 3 ? 'home-grid-nested' : ''}`}
            variants={staggerGrid}
            initial="initial"
            animate="animate"
          >
            {section.items.map(id => {
              const item = SITE_NAV[id];
              return (
                <motion.button key={id} className="home-card" onClick={() => onNavigate(id)} variants={cardPop}>
                  <span className="home-card-accent" />
                  {item.badge && <span className="home-card-badge">{item.badge}</span>}
                  <h3>{item.title}</h3>
                  <p className="home-card-desc">{item.subtitle}</p>
                  {item.metrics && <div className="home-card-meta">{item.metrics}</div>}
                </motion.button>
              );
            })}
          </motion.div>
        </section>
      ))}

      <footer className="home-footer">
        <div>Quant Research Hub · Macro Risk Dashboard</div>
        <div className="home-footer-credit">Built with auto-dashboard · design © Coco</div>
      </footer>
    </div>
  );
}
