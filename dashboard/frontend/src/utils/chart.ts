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
