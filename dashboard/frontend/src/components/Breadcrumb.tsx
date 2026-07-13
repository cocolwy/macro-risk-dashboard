import { breadcrumbTrail, PageId } from '../siteNav';

interface BreadcrumbProps {
  page: PageId;
  onNavigate: (page: PageId) => void;
}

export function Breadcrumb({ page, onNavigate }: BreadcrumbProps) {
  if (page === 'home') return null;

  const trail = breadcrumbTrail(page);

  return (
    <nav className="breadcrumb" aria-label="Breadcrumb">
      {trail.map((item, i) => (
        <span key={item.id} className="breadcrumb-segment">
          {i > 0 && <span className="breadcrumb-sep">/</span>}
          {i < trail.length - 1 ? (
            <button className="breadcrumb-link" onClick={() => onNavigate(item.id)}>
              {item.title}
            </button>
          ) : (
            <span className="breadcrumb-current">{item.title}</span>
          )}
        </span>
      ))}
    </nav>
  );
}
