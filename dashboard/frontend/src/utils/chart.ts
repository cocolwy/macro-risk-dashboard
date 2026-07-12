export function downsample<T>(data: T[], maxPoints = 500): T[] {
  if (data.length <= maxPoints) return data;
  const step = Math.ceil(data.length / maxPoints);
  return data.filter((_, i) => i % step === 0 || i === data.length - 1);
}

export interface TimelinePoint {
  date: string;
  sp500?: number;
  [key: string]: string | number | undefined;
}

export function mergeExperimentTimeline(
  experiments: { probability_timeline: { date: string; probability: number }[] }[],
  sp500Timeline: { date: string; sp500: number }[],
): TimelinePoint[] {
  const sp500Map = new Map(sp500Timeline.map(s => [s.date, s.sp500]));
  const probMaps = experiments.map(exp =>
    new Map(exp.probability_timeline.map(p => [p.date, p.probability])),
  );

  const baseTimeline = experiments.reduce((longest, e) =>
    e.probability_timeline.length > longest.length ? e.probability_timeline : longest,
    [] as { date: string; probability: number }[],
  );

  return baseTimeline.map(p => {
    const row: TimelinePoint = { date: p.date, sp500: sp500Map.get(p.date) };
    probMaps.forEach((map, i) => {
      row[`prob_${i}`] = map.get(p.date);
    });
    return row;
  });
}

export const CHART_TOOLTIP_STYLE = {
  background: '#ffffff',
  border: '1px solid #f1d8e2',
  borderRadius: 8,
  boxShadow: '0 4px 12px rgba(255,168,196,0.12)',
} as const;

export function getSP500Domain(data: { sp500?: number }[]): [number, number] {
  const values = data.map(d => d.sp500).filter((v): v is number => v != null && Number.isFinite(v));
  if (values.length === 0) return [3000, 6000];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = Math.max((max - min) * 0.08, max * 0.01);
  return [Math.floor(min - pad), Math.ceil(max + pad)];
}

export const STACKED_CHART_MARGIN = { left: 10, right: 20 } as const;
export const STACKED_YAXIS_WIDTH = 48;

