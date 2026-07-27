import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  AreaChart,
} from 'recharts';
import { DataPoint } from '../api';
import { downsample, CHART_TOOLTIP_STYLE } from '../utils/chart';

interface ChartCardProps {
  title: string;
  subtitle?: string;
  data: DataPoint[];
  dataKey: string;
  color?: string;
  referenceLine?: { y: number; label: string; color: string };
  className?: string;
  currentValue?: string;
  type?: 'line' | 'area';
  gradientId?: string;
  alertLevel?: 'ok' | 'warning' | 'danger';
  explanation?: string;
}

export function ChartCard({
  title,
  subtitle,
  data,
  dataKey,
  color = '#4a9eff',
  referenceLine,
  className = '',
  currentValue,
  type = 'line',
  gradientId,
  alertLevel,
  explanation,
}: ChartCardProps) {
  const displayData = downsample(data);

  const alertStyle = alertLevel === 'danger'
    ? { borderColor: '#fca5a5', boxShadow: '0 4px 16px rgba(220, 38, 38, 0.12)', animation: 'pulse-danger 2s infinite' }
    : alertLevel === 'warning'
      ? { borderColor: '#fcd34d', boxShadow: '0 4px 12px rgba(245, 158, 11, 0.1)' }
      : {};

  return (
    <div className={`chart-card ${className}`} style={alertStyle}>
      <h3>{title}</h3>
      {subtitle && <div className="subtitle">{subtitle}</div>}
      {currentValue && <div className="current-value" style={{ color }}>{currentValue}</div>}
      <ResponsiveContainer width="100%" height={200}>
        {type === 'area' ? (
          <AreaChart data={displayData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
            <defs>
              <linearGradient id={gradientId || 'gradient'} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                <stop offset="95%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1d8e2" />
            <XAxis
              dataKey="date"
              tick={{ fill: '#8a7882', fontSize: 11 }}
              tickFormatter={(d: string) => d.slice(0, 7)}
              minTickGap={60}
            />
            <YAxis tick={{ fill: '#8a7882', fontSize: 11 }} width={50} />
            <Tooltip contentStyle={CHART_TOOLTIP_STYLE} labelStyle={{ color: '#5a4452' }} />
            {referenceLine && (
              <ReferenceLine y={referenceLine.y} stroke={referenceLine.color} strokeDasharray="5 5" label={{ value: referenceLine.label, fill: referenceLine.color, fontSize: 11 }} />
            )}
            <Area type="monotone" dataKey={dataKey} stroke={color} fill={`url(#${gradientId || 'gradient'})`} strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </AreaChart>
        ) : (
          <LineChart data={displayData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1d8e2" />
            <XAxis
              dataKey="date"
              tick={{ fill: '#8a7882', fontSize: 11 }}
              tickFormatter={(d: string) => d.slice(0, 7)}
              minTickGap={60}
            />
            <YAxis tick={{ fill: '#8a7882', fontSize: 11 }} width={50} />
            <Tooltip contentStyle={CHART_TOOLTIP_STYLE} labelStyle={{ color: '#5a4452' }} />
            {referenceLine && (
              <ReferenceLine y={referenceLine.y} stroke={referenceLine.color} strokeDasharray="5 5" label={{ value: referenceLine.label, fill: referenceLine.color, fontSize: 11 }} />
            )}
            <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </LineChart>
        )}
      </ResponsiveContainer>
      {explanation && (
        <div style={{
          marginTop: '12px',
          padding: '10px 12px',
          background: 'var(--bg-3)',
          borderLeft: '3px solid var(--blue-300)',
          borderRadius: '0 8px 8px 0',
          fontSize: '12px',
          lineHeight: '1.7',
          color: 'var(--blue-500)',
          letterSpacing: '0.005em',
        }}>
          {explanation}
        </div>
      )}
    </div>
  );
}
