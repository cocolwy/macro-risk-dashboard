import { DataPoint } from '../api';

interface SectorTableProps {
  data: DataPoint[];
}

export function SectorTable({ data }: SectorTableProps) {
  if (data.length === 0) return null;

  const latest = data[data.length - 1];
  const sectors = Object.keys(latest)
    .filter((k) => k !== 'date')
    .sort((a, b) => (latest[a] as number) - (latest[b] as number));

  return (
    <div className="chart-card full-width">
      <h3>Sector Health — Distance from 200-Day MA</h3>
      <div className="subtitle">Each sector ETF's current price vs its 200-day moving average</div>

      <div style={{ marginTop: '16px' }}>
        {sectors.map((sector) => {
          const value = latest[sector] as number;
          const isBelow = value < 0;
          const barColor = isBelow
            ? '#f87171'
            : value < 3
              ? '#fbbf24'
              : '#34d399';
          const barWidth = Math.min(Math.abs(value) * 3, 100);

          return (
            <div key={sector} style={{ display: 'flex', alignItems: 'center', marginBottom: '8px', gap: '12px' }}>
              <div style={{ width: '140px', fontSize: '12px', fontWeight: 500, color: 'var(--text-primary)' }}>
                {SECTOR_LABELS[sector] || sector}
              </div>
              <div style={{ flex: 1, position: 'relative', height: '20px', background: 'var(--bg-secondary)', borderRadius: '4px', overflow: 'hidden' }}>
                {isBelow ? (
                  <div style={{
                    position: 'absolute',
                    right: '50%',
                    width: `${barWidth / 2}%`,
                    height: '100%',
                    background: barColor,
                    borderRadius: '4px 0 0 4px',
                    opacity: 0.8,
                  }} />
                ) : (
                  <div style={{
                    position: 'absolute',
                    left: '50%',
                    width: `${barWidth / 2}%`,
                    height: '100%',
                    background: barColor,
                    borderRadius: '0 4px 4px 0',
                    opacity: 0.8,
                  }} />
                )}
                <div style={{
                  position: 'absolute',
                  left: '50%',
                  top: 0,
                  bottom: 0,
                  width: '1px',
                  background: 'rgba(255,255,255,0.2)',
                }} />
              </div>
              <div style={{
                width: '60px',
                textAlign: 'right',
                fontSize: '12px',
                fontWeight: 600,
                color: barColor,
              }}>
                {value > 0 ? '+' : ''}{value.toFixed(1)}%
              </div>
            </div>
          );
        })}
      </div>

      <div style={{
        marginTop: '14px',
        padding: '10px 12px',
        background: 'rgba(74, 158, 255, 0.04)',
        border: '1px solid rgba(74, 158, 255, 0.12)',
        borderRadius: '8px',
        fontSize: '12px',
        lineHeight: '1.6',
        color: 'var(--text-secondary)',
      }}>
        各行业ETF当前价格偏离200日均线的百分比。红色=跌破200MA（趋势走弱），绿色=在200MA上方（趋势健康）。当多数行业转红时，说明市场内部在恶化，即使指数还在高位也要警惕。
      </div>
    </div>
  );
}

const SECTOR_LABELS: Record<string, string> = {
  XLK: '🖥️ Tech 科技',
  XLI: '🏭 Industrial 工业',
  XLV: '🏥 Health 医疗',
  XLE: '⛽ Energy 能源',
  XLF: '🏦 Financial 金融',
  XLRE: '🏠 Real Estate 地产',
  XLB: '⚒️ Materials 原材料',
  XLP: '🛒 Staples 必选消费',
  XLU: '💡 Utilities 公用事业',
  XLY: '🛍️ Discretionary 可选消费',
  XLC: '📡 Communication 通信',
};
