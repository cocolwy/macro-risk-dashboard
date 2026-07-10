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
  const displayData = data.length > 500
    ? data.filter((_, i) => i % Math.ceil(data.length / 500) === 0 || i === data.length - 1)
    : data;

  const alertStyle = alertLevel === 'danger'
    ? { borderColor: 'rgba(248, 113, 113, 0.7)', boxShadow: '0 0 15px rgba(248, 113, 113, 0.2)', animation: 'pulse-danger 2s infinite' }
    : alertLevel === 'warning'
      ? { borderColor: 'rgba(251, 191, 36, 0.5)', boxShadow: '0 0 10px rgba(251, 191, 36, 0.15)' }
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
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2f45" />
            <XAxis
              dataKey="date"
              tick={{ fill: '#9aa0b4', fontSize: 11 }}
              tickFormatter={(d: string) => d.slice(0, 7)}
              minTickGap={60}
            />
            <YAxis tick={{ fill: '#9aa0b4', fontSize: 11 }} width={50} />
            <Tooltip
              contentStyle={{ background: '#1e2235', border: '1px solid #2a2f45', borderRadius: 8 }}
              labelStyle={{ color: '#9aa0b4' }}
            />
            {referenceLine && (
              <ReferenceLine y={referenceLine.y} stroke={referenceLine.color} strokeDasharray="5 5" label={{ value: referenceLine.label, fill: referenceLine.color, fontSize: 11 }} />
            )}
            <Area type="monotone" dataKey={dataKey} stroke={color} fill={`url(#${gradientId || 'gradient'})`} strokeWidth={1.5} dot={false} />
          </AreaChart>
        ) : (
          <LineChart data={displayData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2f45" />
            <XAxis
              dataKey="date"
              tick={{ fill: '#9aa0b4', fontSize: 11 }}
              tickFormatter={(d: string) => d.slice(0, 7)}
              minTickGap={60}
            />
            <YAxis tick={{ fill: '#9aa0b4', fontSize: 11 }} width={50} />
            <Tooltip
              contentStyle={{ background: '#1e2235', border: '1px solid #2a2f45', borderRadius: 8 }}
              labelStyle={{ color: '#9aa0b4' }}
            />
            {referenceLine && (
              <ReferenceLine y={referenceLine.y} stroke={referenceLine.color} strokeDasharray="5 5" label={{ value: referenceLine.label, fill: referenceLine.color, fontSize: 11 }} />
            )}
            <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={1.5} dot={false} />
          </LineChart>
        )}
      </ResponsiveContainer>
      {explanation && (
        <div style={{
          marginTop: '12px',
          padding: '10px 12px',
          background: 'rgba(74, 158, 255, 0.04)',
          border: '1px solid rgba(74, 158, 255, 0.12)',
          borderRadius: '8px',
          fontSize: '12px',
          lineHeight: '1.6',
          color: 'var(--text-secondary)',
        }}>
          {explanation}
        </div>
      )}
    </div>
  );
}
