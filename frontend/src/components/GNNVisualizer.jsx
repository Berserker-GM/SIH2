import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

export default function GNNVisualizer() {
  // Generate mock time-series data for a few real endpoints
  // Showing "Anomaly Score" over time (last 20 minutes)
  const data = [];
  const now = new Date();
  
  for (let i = 20; i >= 0; i--) {
    const time = new Date(now.getTime() - i * 60000);
    // web-prod-01: stable low score
    // 192.168.1.25: stable low score
    // db-cluster-master: massive spike in the last 5 minutes (Anomaly!)
    
    let dbScore = 10 + Math.random() * 15;
    if (i < 5) {
      dbScore = 80 + Math.random() * 15; // Spike!
    }

    data.push({
      time: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      'web-prod-01': 5 + Math.random() * 10,
      '192.168.1.25': 8 + Math.random() * 12,
      'db-cluster-master': dbScore
    });
  }

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-[#141414] border border-[var(--color-border)] rounded p-3 text-[#EDEDED] font-sans text-xs shadow-[0_0_15px_rgba(0,212,255,0.2)]">
          <p className="font-bold text-[var(--color-accent)] mb-2 border-b border-[var(--color-border)] pb-1">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }} className="flex justify-between gap-4 py-0.5">
              <span>{entry.name}:</span>
              <span className="font-mono font-bold">{Math.round(entry.value)}%</span>
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="panel h-full flex flex-col bg-[rgba(0,13,34,0.4)]">
      <div className="p-3 border-b border-[var(--color-border)] flex items-center justify-between bg-[rgba(0,212,255,0.05)] z-10 relative">
        <div className="flex items-center gap-2">
          <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="var(--color-accent)" strokeWidth={2.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
          </svg>
          <span className="text-[11px] font-bold uppercase tracking-widest text-white drop-shadow-[0_0_5px_rgba(0,212,255,0.8)]">Real-Time Endpoint Anomalies</span>
          <div className="relative group cursor-help ml-1">
            <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" className="text-gray-500 hover:text-[var(--color-accent)] transition-colors">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div className="absolute left-0 bottom-full mb-2 hidden group-hover:block w-64 p-3 bg-[#0a0f1a] border border-[var(--color-accent)] rounded text-xs text-gray-300 z-[100] shadow-[0_0_15px_rgba(0,212,255,0.2)]">
              <strong className="text-[var(--color-accent)] block mb-1">Containment & Isolation</strong>
              Tracks realistic features (Anomaly Scores) for active endpoints. When metrics spike into the red zone, decide exactly which servers to isolate to stop lateral movement.
            </div>
          </div>
        </div>
        <span className="text-[9px] uppercase tracking-widest text-[#EF4444] animate-pulse">Critical Spike Detected</span>
      </div>
      <div className="flex-1 w-full h-full p-4 relative">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
            <XAxis dataKey="time" stroke="var(--color-text-secondary)" tick={{ fill: 'var(--color-text-secondary)', fontSize: 10 }} />
            <YAxis stroke="var(--color-text-secondary)" tick={{ fill: 'white', fontSize: 10, fontFamily: 'var(--font-mono)' }} domain={[0, 100]} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} iconType="circle" />
            <Line type="monotone" dataKey="web-prod-01" stroke="#00D4FF" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="192.168.1.25" stroke="#10B981" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="db-cluster-master" stroke="#EF4444" strokeWidth={3} dot={{ r: 3, fill: '#EF4444', strokeWidth: 0 }} activeDot={{ r: 6 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
