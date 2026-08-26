import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

export default function TechniqueChart({ incidents = [] }) {
  // Count technique IDs across all incidents
  const techniqueCounts = {};
  incidents.forEach(inc => {
    if (inc.attack_chain) {
      inc.attack_chain.forEach(step => {
        const tid = step.technique_id || 'UNKNOWN';
        techniqueCounts[tid] = (techniqueCounts[tid] || 0) + 1;
      });
    }
  });

  const data = Object.keys(techniqueCounts)
    .map(tid => ({ techniqueId: tid, count: techniqueCounts[tid] }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8); // Top 8 techniques

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-[#161B22] border border-[var(--color-border)] rounded p-2 text-[#F0F6FC] font-mono text-[10px] shadow-lg">
          <p className="font-bold text-[var(--color-accent)]">{payload[0].payload.techniqueId}</p>
          <p className="mt-1 text-gray-400">Hits: <span className="text-white font-bold">{payload[0].value}</span></p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="panel flex flex-col h-full bg-[var(--color-bg-card)]">
      <div className="p-3 border-b border-[var(--color-border)] flex items-center gap-2 bg-[rgba(0,212,255,0.03)]">
        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="var(--color-accent)" strokeWidth={2.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        <span className="text-[10px] font-bold uppercase tracking-widest text-white">Attack Pattern Frequency</span>
      </div>
      <div className="flex-1 w-full p-3 pl-0 min-h-[140px]">
        {data.length === 0 ? (
          <div className="h-full flex items-center justify-center text-[#8B949E] text-xs italic">
            No patterns recorded
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 5, right: 5, left: -25, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
              <XAxis dataKey="techniqueId" stroke="var(--color-text-secondary)" tick={{ fill: 'var(--color-text-secondary)', fontSize: 8, fontFamily: 'var(--font-mono)' }} />
              <YAxis stroke="var(--color-text-secondary)" tick={{ fill: 'var(--color-text-secondary)', fontSize: 8 }} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0,212,255,0.05)' }} />
              <Bar dataKey="count" fill="var(--color-accent)" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
