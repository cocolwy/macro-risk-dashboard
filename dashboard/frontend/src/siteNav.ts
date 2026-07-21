export type PageId =
  | 'home'
  | 'pipeline'
  | 'risk'
  | 'ch1'
  | 'ch2'
  | 'ch2_1'
  | 'ch2_2'
  | 'ch3'
  | 'ch3_risk'
  | 'factorlab'
  | 'valuation'
  | 'fundamentals';

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
  ch3: {
    id: 'ch3',
    hash: 'ch3',
    title: 'News Agent',
    subtitle: 'RSS 新闻 + ML 概率 · Claude 宏观风险研报',
    level: 1,
    badge: 'NEW',
    metrics: 'risk score · news synthesis',
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
    subtitle: '风控模型线 · Logistic Regression · AB 对比 · 非 Alpha Deck 因子流水线',
    level: 3,
    parent: 'risk',
    metrics: 'D1 Slim+Embargo AUC 0.861',
  },
  ch2: {
    id: 'ch2',
    hash: 'ch2',
    title: 'Ch.2 Non-linear Models',
    subtitle: '风控模型线 · GBDT / RF vs LR · Phase 3 · 非 Alpha Deck',
    level: 3,
    parent: 'risk',
    metrics: 'GBDT vs LR walk-forward',
  },
  ch3_risk: {
    id: 'ch3_risk',
    hash: 'ch3-risk',
    title: 'Ch.3 Fragility & Anomaly',
    subtitle: '风控模型线 · 脆弱性目标 · 异常检测 · 非 Alpha Deck',
    level: 3,
    parent: 'risk',
    badge: 'NEW',
    metrics: 'fragility target · anomaly detection',
  },
  ch2_1: {
    id: 'ch2_1',
    hash: 'ch2-1',
    title: 'Ch.2.1 Metric Exploration',
    subtitle: '风控模型线 · F1 / Brier / 校准 · 非 Alpha Deck',
    level: 3,
    parent: 'risk',
    metrics: 'F1 / Brier / Lift analysis',
  },
  ch2_2: {
    id: 'ch2_2',
    hash: 'ch2-2',
    title: 'Event × VIX',
    subtitle: 'FOMC / CPI / NFP 宏观数据发布前后 · VIX 事件研究',
    level: 2,
    badge: 'NEW',
    metrics: 'VIX event study · H1/H2',
  },
  factorlab: {
    id: 'factorlab',
    hash: 'factorlab',
    title: 'Alpha Deck',
    subtitle: '因子线 · S0–S7 单因子挖掘 · 与 Risk Ch.1/Ch.2 模型实验独立',
    level: 1,
    badge: 'NEW',
    metrics: 'F001/F002/F003 dead',
  },
  valuation: {
    id: 'valuation',
    hash: 'valuation',
    title: 'Company Valuation',
    subtitle: 'DCF + 相对估值三角 · NVDA 案例 · 名词表',
    level: 2,
    badge: 'NEW',
    metrics: 'DCF · peers · sensitivity',
  },
  fundamentals: {
    id: 'fundamentals',
    hash: 'fundamentals',
    title: 'Company Fundamentals',
    subtitle: '利润表 · 现金流 · 资产负债表 · 分析师预期',
    level: 2,
    badge: 'NEW',
    metrics: 'annual · quarterly · estimates',
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
  'ch2-2': 'ch2_2',
  eventvol: 'ch2_2',
  ch3: 'ch3',
  agent: 'ch3',
  'ch3-risk': 'ch3_risk',
  fragility: 'ch3_risk',
  factorlab: 'factorlab',
  alphadeck: 'factorlab',
  valuation: 'valuation',
  dcf: 'valuation',
  fundamentals: 'fundamentals',
  earnings: 'fundamentals',
  financials: 'fundamentals',
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
    hint: 'Alpha Deck = 单因子线；Pipeline / News Agent = 其他架构项目',
    items: ['pipeline', 'ch3', 'factorlab'] as PageId[],
  },
  {
    level: 2,
    label: '二级 · 项目',
    items: ['risk', 'ch2_2', 'valuation', 'fundamentals'] as PageId[],
  },
  {
    level: 3,
    label: '三级 · 风控模型实验（Ch.1 → Ch.2，独立于 Alpha Deck）',
    hint: 'LR / GBDT 模型演进，服务 #risk 看板；不是因子流水线',
    parent: 'risk' as PageId,
    items: ['ch1', 'ch2', 'ch2_1', 'ch3_risk'] as PageId[],
  },
];

/** Home 页待办 — 预注册完成、尚未开工的因子实验 */
export interface HomeTodo {
  id: string;
  factorId: string;
  title: string;
  stage: string;
  status: 'registered' | 'in_progress' | 'done';
  summary: string;
  caseFile: string;
}

export const HOME_TODOS = {
  updated: '2026-07-19',
  title: 'Alpha Deck · 已预注册 · 待开工',
  hint: '当前无待开工因子；F001–F003 均已 dead。',
  items: [] as HomeTodo[],
};
