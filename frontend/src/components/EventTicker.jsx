import { useMemo } from 'react';
import { format } from 'date-fns';

export default function EventTicker({ events = [] }) {
  const recentEvents = useMemo(
    () => events.slice(-20),
    [events]
  );

  if (recentEvents.length === 0) {
    return (
      <div className="bg-[var(--color-bg-card)] border border-[var(--color-border)] rounded-lg px-4 py-2 h-9 flex items-center">
        <span className="text-[#8B949E] text-xs font-mono">
          Awaiting threat telemetry…
        </span>
      </div>
    );
  }

  const renderItem = (event, index, keyPrefix) => {
    let ts = '00:00:00';
    try {
      ts = format(new Date(event.timestamp), 'HH:mm:ss');
    } catch {}

    return (
      <span
        key={`${keyPrefix}-${index}`}
        className="text-[var(--color-accent)] font-mono text-xs whitespace-nowrap mr-8 inline-flex items-center"
      >
        <span className="text-gray-500 mr-1">[</span>
        {ts}
        <span className="text-gray-500 mx-1">]</span>
        <span className="font-bold">{event.technique_id || 'UNKNOWN'}</span>
        <span className="text-[#8B949E] mx-1">—</span>
        <span className="text-[#E6EDF3]">{event.technique_name || 'Unknown'}</span>
      </span>
    );
  };

  return (
    <div className="bg-[var(--color-bg-card)] border border-[var(--color-border)] rounded-lg px-4 py-2 h-9 overflow-hidden relative">
      <div className="animate-marquee whitespace-nowrap">
        {recentEvents.map((e, i) => renderItem(e, i, 'a'))}
        {recentEvents.map((e, i) => renderItem(e, i, 'b'))}
      </div>
    </div>
  );
}
