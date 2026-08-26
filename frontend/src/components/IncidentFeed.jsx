import { useRef, useEffect, useMemo } from 'react';
import { formatDistanceToNow, format } from 'date-fns';

const SEVERITY_COLORS = {
  critical: '#FF4444',
  high: '#FF8C00',
  medium: '#FFD700',
  low: '#00FF88',
};

export default function IncidentFeed({ incidents = [], selectedIncidentId, onSelectIncident }) {
  const prevLengthRef = useRef(incidents.length);

  const sortedIncidents = useMemo(
    () => [...incidents].sort((a, b) => new Date(b.detected_at) - new Date(a.detected_at)),
    [incidents]
  );

  const newCount = incidents.length - prevLengthRef.current;

  useEffect(() => {
    prevLengthRef.current = incidents.length;
  }, [incidents.length]);

  return (
    <div className="panel flex flex-col h-full bg-[var(--color-bg-card)]">
      {/* Header */}
      <div className="p-3.5 border-b border-[var(--color-border)] flex items-center justify-between bg-[rgba(255,255,255,0.02)]">
        <div className="flex items-center gap-2">
          <span className="flex h-2 w-2 relative">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--color-critical)] opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-[var(--color-critical)]"></span>
          </span>
          <span className="text-[10px] font-bold uppercase tracking-widest text-[#F0F6FC]">
            Incident Feed
          </span>
        </div>
        <span className="bg-[rgba(255,255,255,0.05)] border border-[var(--color-border)] text-[#8B949E] text-[9px] font-mono px-2 py-0.5 rounded">
          {incidents.length} Active
        </span>
      </div>

      {/* Incident List */}
      <div className="overflow-y-auto flex-1 divide-y divide-[rgba(255,255,255,0.05)]">
        {sortedIncidents.length === 0 && (
          <div className="px-4 py-8 text-center text-[#8B949E] text-xs italic">
            Awaiting telemetry data...
          </div>
        )}

        {sortedIncidents.map((incident, index) => {
          const severity = incident.severity || 'low';
          const color = SEVERITY_COLORS[severity] || SEVERITY_COLORS.low;
          const isSelected = incident.incident_id === selectedIncidentId;
          const isNew = newCount > 0 && index < newCount;

          const detectedDate = new Date(incident.detected_at);
          const relativeTime = formatDistanceToNow(detectedDate, { addSuffix: true });
          const absoluteTime = format(detectedDate, "yyyy-MM-dd HH:mm:ss") + " UTC";

          const firstAsset =
            incident.affected_assets && incident.affected_assets.length > 0
              ? incident.affected_assets[0]
              : 'Unknown Asset';

          return (
            <div
              key={incident.incident_id}
              onClick={() => onSelectIncident(incident.incident_id)}
              title={absoluteTime}
              className={`
                p-3.5 cursor-pointer transition-all border-l-4 group relative select-none
                ${isSelected
                  ? 'bg-[rgba(0,212,255,0.08)] border-l-[var(--color-accent)]'
                  : 'border-l-transparent hover:bg-[rgba(255,255,255,0.02)]'
                }
                ${isNew ? 'bg-[rgba(0,212,255,0.15)] animate-pulse' : ''}
              `}
              style={{
                borderLeftColor: isSelected ? undefined : color,
              }}
            >
              <div className="flex items-start justify-between mb-1.5">
                <div className="flex flex-col gap-0.5">
                  <span className="font-mono text-[11px] font-bold text-[#F0F6FC] tracking-wider">
                    {incident.incident_id}
                  </span>
                  <span className="text-[9px] text-[#8B949E]">
                    {relativeTime}
                  </span>
                </div>
                <div className="flex flex-col items-end gap-1">
                  <span
                    className="text-base font-bold font-mono leading-none"
                    style={{ color, textShadow: `0 0 10px ${color}33` }}
                  >
                    {incident.risk_score}
                  </span>
                  <span
                    className="text-[8px] font-extrabold uppercase tracking-widest px-1 py-0.25 rounded"
                    style={{
                      color,
                      backgroundColor: `${color}15`,
                      border: `1px solid ${color}33`,
                    }}
                  >
                    {severity}
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between text-[10px] text-[#8B949E] font-mono mt-2">
                <span className="bg-[#161B22] border border-[var(--color-border)] rounded px-1.5 py-0.5 truncate max-w-[150px]">
                  {firstAsset}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
