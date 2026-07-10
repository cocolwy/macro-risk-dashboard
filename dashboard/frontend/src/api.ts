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

async function fetchJson<T>(filename: string): Promise<T> {
  const resp = await fetch(`${DATA_BASE_URL}/${filename}`);
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
