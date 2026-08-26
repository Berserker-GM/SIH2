import { useMemo } from 'react';

export default function AttackerLeaderboard({ incidents = [] }) {
  const topAttackers = useMemo(() => {
    const freq = {};
    incidents.forEach((inc) => {
      const assets = inc.affected_assets ?? [];
      assets.forEach((ip) => {
        freq[ip] = (freq[ip] || 0) + 1;
      });
    });
    return Object.entries(freq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
  }, [incidents]);

  return (
    <div className="panel flex flex-col h-full bg-[var(--color-bg-card)]">
      <div className="p-3 border-b border-[var(--color-border)] flex items-center gap-2 bg-[rgba(0,212,255,0.03)]">
        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="var(--color-accent)" strokeWidth={2.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
        </svg>
        <span className="text-[10px] font-bold uppercase tracking-widest text-white">Attacker IP Leaderboard</span>
      </div>

      <div className="p-3 flex flex-col justify-center">
        {topAttackers.length === 0 ? (
          <p className="text-[#8B949E] text-xs italic text-center py-4">Awaiting data...</p>
        ) : (
          <div className="flex flex-col gap-2">
            {topAttackers.map(([ip, count], idx) => (
              <div
                key={ip}
                className="flex items-center justify-between p-2 rounded bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.03)] hover:border-[var(--color-accent)] transition-all"
              >
                <div className="flex items-center gap-2">
                  <span className="text-[#8B949E] text-[10px] font-mono font-bold w-4">{idx + 1}.</span>
                  <span className="text-[#F0F6FC] text-xs font-mono truncate max-w-[120px]">
                    {ip}
                  </span>
                </div>
                <span className="bg-[rgba(0,212,255,0.1)] border border-[rgba(0,212,255,0.2)] rounded px-1.5 py-0.5 text-[var(--color-accent)] text-[10px] font-mono font-bold">
                  {count} Hits
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
