import { useMemo } from 'react';

const SEVERITY_COLORS = {
  critical: '#FF4444',
  high: '#FF8C00',
  medium: '#FFD700',
  low: '#00FF88',
};

const SEVERITY_ORDER = ['critical', 'high', 'medium', 'low'];

export default function SeverityBreakdown({ incidents = [] }) {
  const counts = useMemo(() => {
    const map = { critical: 0, high: 0, medium: 0, low: 0 };
    incidents.forEach((inc) => {
      const sev = (inc.severity || '').toLowerCase();
      if (sev in map) map[sev]++;
    });
    return map;
  }, [incidents]);

  const total = useMemo(
    () => Object.values(counts).reduce((a, b) => a + b, 0),
    [counts]
  );

  return (
    <div className="panel">
      <div className="panel-header">SEVERITY DISTRIBUTION</div>

      {/* Stacked bar */}
      <div className="w-full h-2 rounded-full overflow-hidden flex bg-[#30363D]">
        {SEVERITY_ORDER.map((sev) => {
          const pct = total > 0 ? (counts[sev] / total) * 100 : 0;
          return (
            <div
              key={sev}
              className="h-full transition-all duration-700 ease-in-out"
              style={{
                width: `${pct}%`,
                backgroundColor: SEVERITY_COLORS[sev],
                minWidth: pct > 0 ? '2px' : '0px',
              }}
            />
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex items-center justify-between mt-3 gap-2 flex-wrap">
        {SEVERITY_ORDER.map((sev) => (
          <div key={sev} className="flex items-center gap-1.5">
            <span
              className="inline-block w-2 h-2 rounded-full"
              style={{ backgroundColor: SEVERITY_COLORS[sev] }}
            />
            <span className="text-[#8B949E] text-[0.65rem] uppercase tracking-wide">
              {sev}
            </span>
            <span
              className="text-[0.7rem] font-[family-name:var(--font-mono)] font-semibold"
              style={{ color: SEVERITY_COLORS[sev] }}
            >
              {counts[sev]}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
