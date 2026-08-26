import { useState, useEffect } from 'react';
import { format } from 'date-fns';

export default function Header({ connected }) {
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  const formattedTime = format(now, 'yyyy-MM-dd HH:mm:ss');

  return (
    <header className="flex items-center justify-between px-6 py-4 bg-[var(--color-bg-secondary)] border-b border-[var(--color-border)] shrink-0 z-50">
      {/* Title / Logo */}
      <div className="flex items-center gap-3">
        <span className="font-mono text-xl font-black tracking-[0.25em] text-[var(--color-accent)] drop-shadow-[0_0_8px_rgba(0,212,255,0.4)]">
          NORMATIVE
        </span>
        <span className="bg-[rgba(0,212,255,0.1)] text-[var(--color-accent)] border border-[rgba(0,212,255,0.2)] text-[9px] font-mono font-bold px-1.5 py-0.5 rounded tracking-widest uppercase">
          SOC OPERATOR
        </span>
      </div>

      {/* Clock & Status */}
      <div className="flex items-center gap-6">
        <span className="text-[#8B949E] text-xs font-mono tracking-wider">
          {formattedTime} UTC
        </span>
        
        <div className="flex items-center gap-2">
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              connected
                ? 'bg-[#00FF88] shadow-[0_0_8px_2px_rgba(0,255,136,0.4)]'
                : 'bg-[#FF4444] shadow-[0_0_8px_2px_rgba(255,68,68,0.4)] animate-pulse'
            }`}
          />
          <span
            className={`text-[10px] font-mono font-bold tracking-wider ${
              connected ? 'text-[#00FF88]' : 'text-[#FF4444]'
            }`}
          >
            {connected ? 'CONNECTED' : 'DISCONNECTED'}
          </span>
        </div>
      </div>
    </header>
  );
}
