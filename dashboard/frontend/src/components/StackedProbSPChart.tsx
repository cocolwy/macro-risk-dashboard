import { useId, useMemo } from 'react';
import {
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, Area, AreaChart,
  CartesianGrid, Line, LineChart, Legend,
} from 'recharts';
import {
  CHART_TOOLTIP_STYLE, getSP500Domain, STACKED_CHART_MARGIN, STACKED_YAXIS_WIDTH,
  TimelinePoint,
} from '../utils/chart';

export interface ProbSeries {
  dataKey: string;
  name: string;
  color: string;
  type?: 'line' | 'area';
  strokeDasharray?: string;
}

export interface DropMarker {
  date: string;
  label: string;
  dropPct: number;
}

interface StackedProbSPChartProps {
  data: TimelinePoint[];
  series: ProbSeries[];
  probHeight?: number;
  spHeight?: number;
  showLegend?: boolean;
  showThreshold?: boolean;
  dropEvents?: DropMarker[];
}

const dateTick = (d: string) => d.slice(5, 10).replace('-', '/');

function filterVisibleDrops(data: TimelinePoint[], drops?: DropMarker[]): DropMarker[] {
  if (!drops?.length || !data.length) return [];
  const minDate = data[0].date;
  const maxDate = data[data.length - 1].date;
  return drops.filter(e => e.date >= minDate && e.date <= maxDate);
}

function DropReferenceLines({ events, opacity = 1 }: { events: DropMarker[]; opacity?: number }) {
  return (
    <>
      {events.map(evt => (
        <ReferenceLine
          key={evt.date}
          x={evt.date}
          stroke={`rgba(220, 38, 38, ${opacity})`}
          strokeDasharray="4 3"
          strokeWidth={1.5}
          ifOverflow="hidden"
        />
      ))}
    </>
  );
}

export function StackedProbSPChart({
  data,
  series,
  probHeight = 260,
  spHeight = 130,
  showLegend = false,
  showThreshold = true,
  dropEvents,
}: StackedProbSPChartProps) {
  const syncId = useId();
  const spDomain = useMemo(() => getSP500Domain(data), [data]);
  const visibleDrops = useMemo(() => filterVisibleDrops(data, dropEvents), [data, dropEvents]);
  const dropDateSet = useMemo(() => new Set(visibleDrops.map(d => d.date)), [visibleDrops]);

  const probMargin = { top: showLegend ? 40 : 16, right: STACKED_CHART_MARGIN.right, left: STACKED_CHART_MARGIN.left, bottom: 8 };
  const spMargin = { top: 8, right: STACKED_CHART_MARGIN.right, left: STACKED_CHART_MARGIN.left, bottom: 8 };

  const hasArea = series.some(s => s.type === 'area');

  const spTooltipFormatter = (value: number, _name: string, item: { payload?: TimelinePoint }) => {
    const date = item.payload?.date;
    const drop = visibleDrops.find(d => d.date === date);
    if (drop) return [`${value?.toFixed(0)} (${drop.dropPct}%)`, drop.label];
    return [value?.toFixed(0), 'S&P 500'];
  };

  const probChart = hasArea ? (
    <AreaChart data={data} margin={probMargin} syncId={syncId}>
      <CartesianGrid strokeDasharray="3 3" stroke="#f1d8e2" vertical={false} />
      <XAxis dataKey="date" hide />
      <YAxis
        domain={[0, 1]}
        width={STACKED_YAXIS_WIDTH}
        tick={{ fill: '#8a7882', fontSize: 11 }}
        tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
      />
      <Tooltip
        contentStyle={CHART_TOOLTIP_STYLE}
        formatter={(value: number, name: string) => {
          const s = series.find(x => x.name === name || x.dataKey === name);
          return [`${(value * 100).toFixed(1)}%`, s?.name ?? name];
        }}
        labelFormatter={(label) => String(label)}
      />
      {showLegend && <Legend verticalAlign="top" height={32} iconType="plainline" />}
      <DropReferenceLines events={visibleDrops} opacity={0.25} />
      {showThreshold && (
        <ReferenceLine y={0.5} stroke="#dc2626" strokeDasharray="5 5"
          label={{ value: '50%', fill: '#dc2626', fontSize: 10, position: 'right' }} />
      )}
      {series.map(s => (
        <Area
          key={s.dataKey}
          type="monotone"
          dataKey={s.dataKey}
          name={s.name}
          stroke={s.color}
          fill={s.color}
          fillOpacity={0.2}
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
      ))}
    </AreaChart>
  ) : (
    <LineChart data={data} margin={probMargin} syncId={syncId}>
      <CartesianGrid strokeDasharray="3 3" stroke="#f1d8e2" vertical={false} />
      <XAxis dataKey="date" hide />
      <YAxis
        domain={[0, 1]}
        width={STACKED_YAXIS_WIDTH}
        tick={{ fill: '#8a7882', fontSize: 11 }}
        tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
      />
      <Tooltip
        contentStyle={CHART_TOOLTIP_STYLE}
        formatter={(value: number, name: string) => {
          const s = series.find(x => x.name === name || x.dataKey === name);
          return [`${(value * 100).toFixed(1)}%`, s?.name ?? name];
        }}
        labelFormatter={(label) => String(label)}
      />
      {showLegend && <Legend verticalAlign="top" height={32} iconType="plainline" />}
      <DropReferenceLines events={visibleDrops} opacity={0.25} />
      {showThreshold && (
        <ReferenceLine y={0.5} stroke="#dc2626" strokeDasharray="5 5"
          label={{ value: '50%', fill: '#dc2626', fontSize: 10, position: 'right' }} />
      )}
      {series.map(s => (
        <Line
          key={s.dataKey}
          type="monotone"
          dataKey={s.dataKey}
          name={s.name}
          stroke={s.color}
          strokeWidth={1.5}
          strokeDasharray={s.strokeDasharray}
          dot={false}
          isAnimationActive={false}
        />
      ))}
    </LineChart>
  );

  return (
    <div className="stacked-charts">
      <div className="stacked-charts-prob">
        <ResponsiveContainer width="100%" height={probHeight}>
          {probChart}
        </ResponsiveContainer>
      </div>

      <div className="stacked-charts-divider" />

      <div className="stacked-charts-sp">
        <div className="stacked-sp-header">
          <span className="stacked-sp-label">S&P 500</span>
          {visibleDrops.length > 0 && (
            <span className="stacked-sp-hint">
              <span className="drop-marker-swatch" /> 红色虚线 = 历史大跌
            </span>
          )}
        </div>
        <ResponsiveContainer width="100%" height={spHeight}>
          <LineChart data={data} margin={spMargin} syncId={syncId}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1d8e2" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fill: '#8a7882', fontSize: 10 }}
              tickFormatter={dateTick}
              minTickGap={50}
              axisLine={{ stroke: '#f1d8e2' }}
              tickLine={false}
            />
            <YAxis
              domain={spDomain}
              width={STACKED_YAXIS_WIDTH}
              tick={{ fill: '#3a82d6', fontSize: 10 }}
              tickFormatter={(v: number) => v.toFixed(0)}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={CHART_TOOLTIP_STYLE}
              formatter={spTooltipFormatter as (value: number, name: string) => [string, string]}
              labelFormatter={(label) => String(label)}
            />
            <DropReferenceLines events={visibleDrops} />
            {visibleDrops.map(evt => (
              <ReferenceLine
                key={`lbl-${evt.date}`}
                x={evt.date}
                stroke="transparent"
                label={{
                  value: `${evt.dropPct}%`,
                  position: 'insideTopLeft',
                  fill: '#dc2626',
                  fontSize: 9,
                  fontWeight: 600,
                  offset: 4,
                }}
              />
            ))}
            <Line
              type="monotone"
              dataKey="sp500"
              stroke="#3a82d6"
              strokeWidth={2}
              dot={(props: { cx?: number; cy?: number; payload?: TimelinePoint }) => {
                const { cx, cy, payload } = props;
                if (cx == null || cy == null || !payload?.date || !dropDateSet.has(payload.date)) return <></>;
                return <circle cx={cx} cy={cy} r={5} fill="#dc2626" stroke="#fff" strokeWidth={2} />;
              }}
              activeDot={{ r: 4, fill: '#3a82d6', stroke: '#fff', strokeWidth: 2 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
        {visibleDrops.length > 0 && (
          <div className="stacked-drop-tags">
            {visibleDrops.map(evt => (
              <span key={evt.date} className="stacked-drop-tag" title={evt.label}>
                {evt.label} <strong>{evt.dropPct}%</strong>
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
