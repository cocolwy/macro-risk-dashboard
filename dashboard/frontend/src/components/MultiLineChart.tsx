import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { DataPoint } from '../api';
import { downsample, CHART_TOOLTIP_STYLE } from '../utils/chart';

interface MultiLineChartProps {
  title: string;
  subtitle?: string;
  data: DataPoint[];
  lines: { key: string; color: string; name: string }[];
  className?: string;
  explanation?: string;
}

export function MultiLineChart({ title, subtitle, data, lines, className = '', explanation }: MultiLineChartProps) {
  const displayData = downsample(data);

  return (
    <div className={`chart-card ${className}`}>
      <h3>{title}</h3>
      {subtitle && <div className="subtitle">{subtitle}</div>}
      <ResponsiveContainer width="100%" height={220}>
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
          <Legend wrapperStyle={{ fontSize: 12, color: '#5a4452' }} />
          {lines.map((l) => (
            <Line key={l.key} type="monotone" dataKey={l.key} stroke={l.color} name={l.name} strokeWidth={1.5} dot={false} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
      {explanation && (
        <div style={{
          marginTop: '12px',
          padding: '10px 12px',
          background: '#f3f8ff',
          borderLeft: '3px solid #8cc3ff',
          borderRadius: '0 8px 8px 0',
          fontSize: '12px',
          lineHeight: '1.6',
          color: '#3a82d6',
        }}>
          {explanation}
        </div>
      )}
    </div>
  );
}
