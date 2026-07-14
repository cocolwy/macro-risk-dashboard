export interface DataPoint {
  date: string;
  [key: string]: string | number;
}

export interface AlertInfo {
  level: 'ok' | 'warning' | 'danger';
  message: string;
}

export interface Summary {
  alerts: Record<string, AlertInfo>;
  last_updated: string;
  indicators_available: string[];
  composite_score?: {
    score: number;
    label: string;
    level: string;
    color: string;
    action: string;
    components: Record<string, number>;
    date: string;
  };
}

export interface MomentumData {
  [key: string]: { current: number; week_change: number; month_change: number };
}

const DATA_BASE_URL = import.meta.env.PROD
  ? './data'
  : '/data';

function getScoreLabel(score: number) {
  if (score <= 20) return { level: 'low', label: 'Low Risk', color: '#34d399', action: '正常持仓' };
  if (score <= 40) return { level: 'moderate', label: 'Moderate', color: '#86efac', action: '保持关注' };
  if (score <= 60) return { level: 'elevated', label: 'Elevated', color: '#fbbf24', action: '审视仓位，准备防御' };
  if (score <= 80) return { level: 'high', label: 'High Risk', color: '#fb923c', action: '减仓/增加对冲' };
  return { level: 'extreme', label: 'Extreme', color: '#f87171', action: '大幅减仓/现金为王' };
}

export function resolveCompositeScore(
  summary: Summary,
  series: DataPoint[],
): NonNullable<Summary['composite_score']> | undefined {
  if (summary.composite_score) return summary.composite_score;
  if (series.length === 0) return undefined;

  const latest = series[series.length - 1];
  const score = Number(latest.composite_score);
  if (!Number.isFinite(score)) return undefined;

  const label = getScoreLabel(score);
  return {
    score,
    label: label.label,
    level: label.level,
    color: label.color,
    action: label.action,
    components: (latest.components as unknown as Record<string, number> | undefined) ?? {},
    date: String(latest.date),
  };
}

export async function fetchDataJson<T>(filename: string): Promise<T> {
  return fetchJson<T>(filename);
}

async function fetchJson<T>(filename: string): Promise<T> {
  // Always bust cache — GitHub Pages / browsers otherwise keep stale dashboard JSON.
  const cacheBust = `?_=${Date.now()}`;
  const resp = await fetch(`${DATA_BASE_URL}/${filename}${cacheBust}`);
  if (!resp.ok) throw new Error(`Failed to fetch ${filename}: ${resp.status}`);
  return resp.json();
}

export async function fetchSummary(): Promise<Summary> {
  return fetchJson<Summary>('summary.json');
}

export async function fetchIndicator(name: string): Promise<DataPoint[]> {
  return fetchJson<DataPoint[]>(`${name}.json`);
}

export async function fetchAllData(): Promise<{
  summary: Summary;
  termSpread: DataPoint[];
  creditSpread: DataPoint[];
  vix: DataPoint[];
  sp500: DataPoint[];
  breadth: DataPoint[];
  sectors: DataPoint[];
  absorptionRatio: DataPoint[];
  turbulence: DataPoint[];
  compositeScore: DataPoint[];
  momentum: MomentumData;
}> {
  const [summary, termSpread, creditSpread, vix, sp500, breadth, sectors, absorptionRatio, turbulence, compositeScore, momentum] =
    await Promise.all([
      fetchSummary(),
      fetchIndicator('term_spread').catch(() => []),
      fetchIndicator('credit_spread').catch(() => []),
      fetchIndicator('vix').catch(() => []),
      fetchIndicator('sp500').catch(() => []),
      fetchIndicator('breadth').catch(() => []),
      fetchIndicator('sectors').catch(() => []),
      fetchIndicator('absorption_ratio').catch(() => []),
      fetchIndicator('turbulence').catch(() => []),
      fetchIndicator('composite_score').catch(() => []),
      fetchJson<MomentumData>('momentum.json').catch(() => ({})),
    ]);

  return { summary, termSpread, creditSpread, vix, sp500, breadth, sectors, absorptionRatio, turbulence, compositeScore, momentum };
}
