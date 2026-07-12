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

interface StackedProbSPChartProps {
  data: TimelinePoint[];
  series: ProbSeries[];
  probHeight?: number;
  spHeight?: number;
  showLegend?: boolean;
  showThreshold?: boolean;
}

const dateTick = (d: string) => d.slice(5, 10).replace('-', '/');

export function StackedProbSPChart({
  data,
  series,
  probHeight = 220,
  spHeight = 110,
  showLegend = false,
  showThreshold = true,
}: StackedProbSPChartProps) {
  const syncId = useId();
  const spDomain = useMemo(() => getSP500Domain(data), [data]);

  const probMargin = { top: showLegend ? 36 : 10, right: STACKED_CHART_MARGIN.right, left: STACKED_CHART_MARGIN.left, bottom: 0 };
  const spMargin = { top: 4, right: STACKED_CHART_MARGIN.right, left: STACKED_CHART_MARGIN.left, bottom: 5 };

  const hasArea = series.some(s => s.type === 'area');

  return (
    <div className="stacked-charts">
      <ResponsiveContainer width="100%" height={probHeight}>
        {hasArea ? (
          <AreaChart data={data} margin={probMargin} syncId={syncId}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1d8e2" />
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
                const s = series.find(x => x.dataKey === name);
                return [`${(value * 100).toFixed(1)}%`, s?.name ?? name];
              }}
              labelFormatter={(label) => String(label)}
            />
            {showLegend && (
              <Legend verticalAlign="top" height={28}
                formatter={(value: string) => series.find(s => s.dataKey === value)?.name ?? value}
              />
            )}
            {showThreshold && (
              <ReferenceLine y={0.5} stroke="#dc2626" strokeDasharray="5 5"
                label={{ value: '50%', fill: '#dc2626', fontSize: 10, position: 'right' }} />
            )}
            {series.map(s => (
              <Area
                key={s.dataKey}
                type="monotone"
                dataKey={s.dataKey}
                stroke={s.color}
                fill={s.color}
                fillOpacity={0.25}
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
                name={s.dataKey}
              />
            ))}
          </AreaChart>
        ) : (
          <LineChart data={data} margin={probMargin} syncId={syncId}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1d8e2" />
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
                const s = series.find(x => x.dataKey === name);
                return [`${(value * 100).toFixed(1)}%`, s?.name ?? name];
              }}
              labelFormatter={(label) => String(label)}
            />
            {showLegend && (
              <Legend verticalAlign="top" height={28}
                formatter={(value: string) => series.find(s => s.dataKey === value)?.name ?? value}
              />
            )}
            {showThreshold && (
              <ReferenceLine y={0.5} stroke="#dc2626" strokeDasharray="5 5"
                label={{ value: '50%', fill: '#dc2626', fontSize: 10, position: 'right' }} />
            )}
            {series.map(s => (
              <Line
                key={s.dataKey}
                type="monotone"
                dataKey={s.dataKey}
                stroke={s.color}
                strokeWidth={1.5}
                strokeDasharray={s.strokeDasharray}
                dot={false}
                isAnimationActive={false}
                name={s.dataKey}
              />
            ))}
          </LineChart>
        )}
      </ResponsiveContainer>

      <div className="stacked-sp-label">S&P 500</div>

      <ResponsiveContainer width="100%" height={spHeight}>
        <LineChart data={data} margin={spMargin} syncId={syncId}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1d8e2" />
          <XAxis
            dataKey="date"
            tick={{ fill: '#8a7882', fontSize: 10 }}
            tickFormatter={dateTick}
            minTickGap={50}
          />
          <YAxis
            domain={spDomain}
            width={STACKED_YAXIS_WIDTH}
            tick={{ fill: '#3a82d6', fontSize: 10 }}
            tickFormatter={(v: number) => v.toFixed(0)}
          />
          <Tooltip
            contentStyle={CHART_TOOLTIP_STYLE}
            formatter={(value: number) => [value?.toFixed(0), 'S&P 500']}
            labelFormatter={(label) => String(label)}
          />
          <Line
            type="monotone"
            dataKey="sp500"
            stroke="#3a82d6"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
