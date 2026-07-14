import { useState, useEffect, useRef, type ReactNode } from 'react';

interface LazyMountProps {
  children: ReactNode;
  fallback?: ReactNode;
  minHeight?: number;
  rootMargin?: string;
}

/** Defer heavy chart rendering until the section scrolls near the viewport. */
export function LazyMount({
  children,
  fallback,
  minHeight = 320,
  rootMargin = '300px 0px',
}: LazyMountProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [rootMargin]);

  return (
    <div ref={ref} style={{ minHeight: visible ? undefined : minHeight }}>
      {visible
        ? children
        : (fallback ?? <div className="chart-placeholder" style={{ minHeight }}>加载图表…</div>)}
    </div>
  );
}
