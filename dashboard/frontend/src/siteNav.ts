export type PageId = 'home' | 'pipeline' | 'risk' | 'ch1' | 'ch2' | 'ch2_1';

export interface NavItem {
  id: PageId;
  hash: string;
  title: string;
  subtitle: string;
  level: 0 | 1 | 2 | 3;
  parent?: PageId;
  badge?: string;
  metrics?: string;
}

/** Canonical site hierarchy — keep in sync with .cursor/rules/dashboard-structure.mdc */
export const SITE_NAV: Record<PageId, NavItem> = {
  home: {
    id: 'home',
    hash: '',
    title: 'Quant Research Hub',
    subtitle: '量化研究看板索引',
    level: 0,
  },
  pipeline: {
    id: 'pipeline',
    hash: 'pipeline',
    title: 'Multi-Agent Pipeline',
    subtitle: '6-Agent 因子研究 · 组合构建 · Qlib 回测',
    level: 1,
    metrics: '107→459 stocks · 5 phases',
  },
  risk: {
    id: 'risk',
    hash: 'risk',
    title: 'Macro Risk Dashboard',
    subtitle: '宏观风控指标监控 · 综合风险评分 · 每日更新',
    level: 2,
    metrics: '7 indicators · composite score',
  },
  ch1: {
    id: 'ch1',
    hash: 'ch1',
    title: 'Ch.1 Linear Models',
    subtitle: 'Logistic Regression 实验台 · AB 对比 · 权重分析',
    level: 3,
    parent: 'risk',
    metrics: 'D1 Slim+Embargo AUC 0.861',
  },
  ch2: {
    id: 'ch2',
    hash: 'ch2',
    title: 'Ch.2 Non-linear Models',
    subtitle: 'GBDT / RandomForest vs LR · Unbalanced 训练 · Phase 3',
    level: 3,
    parent: 'risk',
    metrics: 'GBDT vs LR walk-forward',
  },
  ch2_1: {
    id: 'ch2_1',
    hash: 'ch2-1',
    title: 'Ch.2.1 Metric Exploration',
    subtitle: '评估指标优化 · Balanced vs Unbalanced · 概率校准实验',
    level: 3,
    parent: 'risk',
    badge: 'NEW',
    metrics: 'F1 / Brier / Lift analysis',
  },
};

const HASH_ALIASES: Record<string, PageId> = {
  '': 'home',
  home: 'home',
  pipeline: 'pipeline',
  board: 'pipeline',
  risk: 'risk',
  dashboard: 'risk',
  ch1: 'ch1',
  lab: 'ch1',
  ch2: 'ch2',
  phase3: 'ch2',
  'ch2-1': 'ch2_1',
  metrics: 'ch2_1',
};

export function pageFromHash(hash: string): PageId {
  const key = hash.replace('#', '').trim();
  return HASH_ALIASES[key] ?? 'home';
}

export function hashForPage(page: PageId): string {
  const item = SITE_NAV[page];
  return item.hash ? `#${item.hash}` : '';
}

export function breadcrumbTrail(page: PageId): NavItem[] {
  if (page === 'home') return [SITE_NAV.home];
  const trail: NavItem[] = [SITE_NAV.home];
  const current = SITE_NAV[page];
  if (current.parent) trail.push(SITE_NAV[current.parent]);
  trail.push(current);
  return trail;
}

export const HOME_SECTIONS = [
  {
    level: 1,
    label: '一级 · 量化整体架构',
    items: ['pipeline'] as PageId[],
  },
  {
    level: 2,
    label: '二级 · 项目',
    items: ['risk'] as PageId[],
  },
  {
    level: 3,
    label: '三级 · 风控因子探索',
    parent: 'risk' as PageId,
    items: ['ch1', 'ch2', 'ch2_1'] as PageId[],
  },
];
