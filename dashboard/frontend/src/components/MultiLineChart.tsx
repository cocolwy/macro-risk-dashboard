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

interface MultiLineChartProps {
  title: string;
  subtitle?: string;
  data: DataPoint[];
  lines: { key: string; color: string; name: string }[];
  className?: string;
}

export function MultiLineChart({ title, subtitle, data, lines, className = '' }: MultiLineChartProps) {
  const displayData = data.length > 500
    ? data.filter((_, i) => i % Math.ceil(data.length / 500) === 0 || i === data.length - 1)
    : data;

  return (
    <div className={`chart-card ${className}`}>
      <h3>{title}</h3>
      {subtitle && <div className="subtitle">{subtitle}</div>}
      <ResponsiveContainer width="100%" height={220}>
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
          <Legend wrapperStyle={{ fontSize: 12, color: '#9aa0b4' }} />
          {lines.map((l) => (
            <Line key={l.key} type="monotone" dataKey={l.key} stroke={l.color} name={l.name} strokeWidth={1.5} dot={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
