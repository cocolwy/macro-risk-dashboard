/** Clarifies Alpha Deck vs Risk Model lab are separate research tracks. */
export function ResearchTrackNotice({ track }: { track: 'alpha' | 'risk-model' }) {
  if (track === 'alpha') {
    return (
      <div className="lab-track-notice">
        <span className="lab-track-label">研究线 A · Alpha Deck</span>
        <span className="lab-track-desc">单因子挖掘（S0–S7 · 假设 → 检验 → 可交易性）</span>
        <span className="lab-track-sep">|</span>
        <span className="lab-track-other">另一条独立线：</span>
        <a className="lab-track-link" href="#ch1">风控模型实验 Ch.1→Ch.2</a>
        <span className="lab-track-hint">（综合崩盘概率，不走因子流水线）</span>
      </div>
    );
  }

  return (
    <div className="lab-track-notice lab-track-notice-risk">
      <span className="lab-track-label">研究线 B · 风控模型实验</span>
      <span className="lab-track-desc">LR / GBDT 等模型演进 · 服务 Macro Risk Dashboard</span>
      <span className="lab-track-sep">|</span>
      <span className="lab-track-other">另一条独立线：</span>
      <a className="lab-track-link" href="#factorlab">Alpha Deck</a>
      <span className="lab-track-hint">（单因子 S0–S7，如 F001 UVIX）</span>
    </div>
  );
}
