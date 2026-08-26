import { useMemo } from 'react';

function formatNumber(n) {
  if (n == null) return '0';
  return Number(n).toLocaleString();
}

export default function SummaryStats({ incidents = [], metrics = {} }) {
  const counts = useMemo(() => {
    const total = incidents.length;
    const critical = incidents.filter((i) => i.severity === 'critical').length;
    const high = incidents.filter((i) => i.severity === 'high').length;
    const activeThreats = critical + high;
    const eventsProcessed = metrics.total_events_processed ?? 0;
    return { total, critical, activeThreats, eventsProcessed };
  }, [incidents, metrics]);

  const cards = [
    {
      label: 'INCIDENTS TODAY',
      value: counts.total,
      color: '#00D4FF',
    },
    {
      label: 'CRITICAL THREATS',
      value: counts.critical,
      color: '#FF4444',
    },
    {
      label: 'ACTIVE THREATS',
      value: counts.activeThreats,
      color: '#FF8C00',
    },
    {
      label: 'EVENTS PROCESSED',
      value: counts.eventsProcessed,
      color: '#F0F6FC',
      formatted: true,
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-2 w-full">
      {cards.map((card) => (
        <div
          key={card.label}
          className="bg-[var(--color-bg-card)] border border-[var(--color-border)] rounded-md p-2.5 flex flex-col justify-between hover:border-[var(--color-accent)] transition-all"
        >
          <span className="text-[7.5px] font-bold uppercase tracking-wider text-[#8B949E]">
            {card.label}
          </span>
          <span
            className="text-lg font-black font-mono mt-1"
            style={{ color: card.color, textShadow: `0 0 10px ${card.color}22` }}
          >
            {card.formatted ? formatNumber(card.value) : card.value}
          </span>
        </div>
      ))}
    </div>
  );
}
