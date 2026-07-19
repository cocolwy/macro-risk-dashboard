import { SITE_NAV, HOME_SECTIONS, PageId } from './siteNav';

interface HomeProps {
  onNavigate: (page: PageId) => void;
}

export function Home({ onNavigate }: HomeProps) {
  return (
    <div className="home-container">
      <header className="home-header">
        <h1>{SITE_NAV.home.title}</h1>
        <p className="home-subtitle">
          两条独立研究线：<strong>Alpha Deck</strong>（单因子 S0–S7）与 <strong>风控模型 Ch.1→Ch.2</strong>（LR/GBDT 崩盘预测）
        </p>
      </header>

      {HOME_SECTIONS.map(section => (
        <section key={section.level} className="home-section">
          <div className="home-section-title">{section.label}</div>
          {'hint' in section && section.hint && (
            <p className="home-section-hint">{section.hint}</p>
          )}

          {section.level === 3 && section.parent && (
            <div className="home-parent-hint">
              隶属 <button className="home-parent-link" onClick={() => onNavigate(section.parent!)}>
                {SITE_NAV[section.parent].title}
              </button>
            </div>
          )}

          <div className={`home-grid ${section.level === 3 ? 'home-grid-nested' : ''}`}>
            {section.items.map(id => {
              const item = SITE_NAV[id];
              return (
                <button key={id} className="home-card" onClick={() => onNavigate(id)}>
                  <span className="home-card-accent" />
                  <span className={`home-card-level l${item.level}`}>L{item.level}</span>
                  {item.badge && <span className="home-card-badge">{item.badge}</span>}
                  <h3>{item.title}</h3>
                  <p className="home-card-desc">{item.subtitle}</p>
                  {item.metrics && <div className="home-card-meta">{item.metrics}</div>}
                </button>
              );
            })}
          </div>
        </section>
      ))}

      <footer className="home-footer">
        <div>Quant Research Hub · Macro Risk Dashboard</div>
        <div className="home-footer-credit">Built with auto-dashboard · design © Coco</div>
      </footer>
    </div>
  );
}
