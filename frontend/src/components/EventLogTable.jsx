import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { format } from 'date-fns';

const GRID_TEMPLATE = '120px 100px 140px 1fr 100px 120px';
const ROW_HEIGHT = 32;

const STATUS_COLORS = {
  success: '#00FF88',
  allowed: '#00FF88',
  passed: '#00FF88',
  ok: '#00FF88',
  failure: '#FF4444',
  failed: '#FF4444',
  blocked: '#FF4444',
  denied: '#FF4444',
  rejected: '#FF4444',
  error: '#FF4444',
};

function getStatusColor(status) {
  if (!status) return '#8B949E';
  const key = status.toLowerCase();
  return STATUS_COLORS[key] || '#8B949E';
}

function formatTimestamp(ts) {
  try {
    const date = new Date(ts);
    if (isNaN(date.getTime())) return String(ts);
    return format(date, 'HH:mm:ss.SSS');
  } catch {
    return String(ts);
  }
}

const TableHeader = () => (
  <div
    className="sticky top-0 z-20 bg-[var(--color-bg-secondary)] border-b border-[var(--color-border)] text-[#8B949E] text-[9px] uppercase font-bold tracking-wider"
    style={{
      display: 'grid',
      gridTemplateColumns: GRID_TEMPLATE,
      height: ROW_HEIGHT,
      alignItems: 'center',
      paddingLeft: 12,
      paddingRight: 12,
    }}
  >
    <span>Timestamp</span>
    <span>Source Type</span>
    <span>Source IP</span>
    <span>Action</span>
    <span>Status</span>
    <span>MITRE Tag</span>
  </div>
);

function EventRow({ event, index }) {
  const isEven = index % 2 === 0;
  const hasMitre = Boolean(event.mitre_tag);
  const statusColor = getStatusColor(event.status);

  return (
    <div
      className={`
        transition-colors duration-150 select-none
        ${isEven ? 'bg-[#0D1117]' : 'bg-[#161B22]/40'}
        ${hasMitre 
          ? 'border-l-[3px] border-[var(--color-critical)] text-[#F0F6FC] bg-[rgba(255,68,68,0.02)]' 
          : 'border-l-[3px] border-transparent text-[#8B949E]'
        }
        hover:bg-[rgba(0,212,255,0.03)]
      `}
      style={{
        display: 'grid',
        gridTemplateColumns: GRID_TEMPLATE,
        height: ROW_HEIGHT,
        alignItems: 'center',
        paddingLeft: 12,
        paddingRight: 12,
      }}
    >
      {/* Timestamp */}
      <span className="font-mono text-[10px] opacity-90 truncate">
        {formatTimestamp(event.timestamp)}
      </span>

      {/* Source Type */}
      <span className="flex items-center">
        <span className="bg-[rgba(255,255,255,0.03)] border border-[var(--color-border)] text-[var(--color-accent)] rounded px-1.5 py-0.25 text-[9px] font-mono truncate max-w-[85px]">
          {event.source_type || '—'}
        </span>
      </span>

      {/* Source IP */}
      <span className="font-mono text-[10px] truncate">
        {event.src_ip || '—'}
      </span>

      {/* Action */}
      <span className="text-[10px] truncate pr-2">
        {event.action || '—'}
      </span>

      {/* Status */}
      <span
        className="text-[10px] font-bold truncate uppercase"
        style={{ color: statusColor }}
      >
        {event.status || '—'}
      </span>

      {/* MITRE Tag */}
      <span className="flex items-center">
        {hasMitre ? (
          <span className="bg-[rgba(255,68,68,0.1)] border border-[rgba(255,68,68,0.2)] text-[var(--color-critical)] rounded px-1.5 py-0.25 text-[9px] font-mono font-bold truncate max-w-[100px]">
            {event.mitre_tag}
          </span>
        ) : (
          <span className="text-[10px] text-gray-600 font-mono">—</span>
        )}
      </span>
    </div>
  );
}

export default function EventLogTable({ events = [] }) {
  const containerRef = useRef(null);
  const [isHovered, setIsHovered] = useState(false);

  // Limit to last 100 events
  const displayEvents = useMemo(
    () => (events.length > 100 ? events.slice(-100) : events),
    [events]
  );

  // Auto-scroll to bottom when new events arrive (unless hovered)
  useEffect(() => {
    if (!isHovered && containerRef.current && displayEvents.length > 0) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [displayEvents, isHovered]);

  const handleMouseEnter = useCallback(() => setIsHovered(true), []);
  const handleMouseLeave = useCallback(() => setIsHovered(false), []);

  return (
    <div className="panel flex flex-col h-full bg-[var(--color-bg-card)]">
      {/* Panel Header */}
      <div className="p-3 border-b border-[var(--color-border)] flex items-center justify-between bg-[rgba(0,212,255,0.03)] flex-none">
        <div className="flex items-center gap-2">
          <svg
            className="w-3.5 h-3.5 text-[var(--color-accent)]"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
          <span className="text-[10px] font-bold uppercase tracking-widest text-white">Live Event Log</span>
          <span className="bg-[rgba(255,255,255,0.05)] text-[#8B949E] text-[9px] font-mono rounded-full px-2 py-0.25">
            {displayEvents.length}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="relative flex h-1.5 w-1.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#00FF88] opacity-75" />
            <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[#00FF88]" />
          </span>
          <span className="text-[9px] font-bold text-[#00FF88] tracking-widest uppercase animate-pulse">
            STREAMING
          </span>
        </div>
      </div>

      {/* Table Body Container */}
      <div
        className="flex-1 flex flex-col min-h-0"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <TableHeader />

        {displayEvents.length > 0 ? (
          <div
            ref={containerRef}
            className="overflow-y-auto flex-1 custom-scrollbar min-h-0 bg-[#0D1117]/80"
          >
            {displayEvents.map((event, index) => (
              <EventRow
                key={`${event.timestamp}-${index}`}
                event={event}
                index={index}
              />
            ))}
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-[#8B949E] text-xs italic py-8">
            No events streaming. Check backend connection.
          </div>
        )}
      </div>
    </div>
  );
}
