import { useMemo } from 'react';

function fprColor(fpr) {
  if (fpr < 5) return '#00FF88'; // Green
  if (fpr <= 15) return '#FFD700'; // Yellow
  return '#FF4444'; // Red
}

function FprGauge({ fpr }) {
  const radius = 28;
  const strokeWidth = 4;
  const normalizedRadius = radius - strokeWidth / 2;
  const circumference = 2 * Math.PI * normalizedRadius;
  const clampedFpr = Math.min(Math.max(fpr, 0), 100);
  const offset = circumference - (clampedFpr / 100) * circumference;
  const color = fprColor(fpr);

  return (
    <div className="flex items-center gap-3 bg-[rgba(22,27,34,0.6)] border border-[var(--color-border)] p-2.5 rounded-lg">
      <div className="relative flex items-center justify-center" style={{ width: radius * 2, height: radius * 2 }}>
        <svg width={radius * 2} height={radius * 2} className="transform -rotate-90">
          <circle
            cx={radius}
            cy={radius}
            r={normalizedRadius}
            fill="transparent"
            stroke="rgba(255,255,255,0.05)"
            strokeWidth={strokeWidth}
          />
          <circle
            cx={radius}
            cy={radius}
            r={normalizedRadius}
            fill="transparent"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="transition-all duration-700 ease-in-out"
          />
        </svg>
        <span
          className="absolute text-[10px] font-bold font-mono"
          style={{ color }}
        >
          {fpr.toFixed(1)}%
        </span>
      </div>
      <div className="flex flex-col">
        <span className="text-[9px] uppercase tracking-wider text-[var(--color-text-secondary)] font-bold">FPR STATUS</span>
        <span className="text-[8px] text-[#8B949E] uppercase">
          {fpr < 5 ? 'EXCELLENT' : fpr <= 15 ? 'WARNING' : 'CRITICAL'}
        </span>
      </div>
    </div>
  );
}

function normPct(val) {
  if (val == null || isNaN(val)) return 0;
  const num = Number(val);
  if (num <= 0) return 0;
  if (num > 100) return num / 100;
  if (num > 1) return num;
  return num * 100;
}

export default function ModelMetrics({ metrics = {} }) {
  const vals = useMemo(() => {
    const fpr = normPct(metrics.fpr ?? 0.03);
    const precision = normPct(metrics.precision ?? 0.94);
    const recall = normPct(metrics.recall ?? 0.91);
    const f1 = normPct(metrics.f1 ?? 0.925);
    const accuracy = normPct(metrics.accuracy ?? 0.96);
    const totalAnomalies = metrics.total_anomalies_detected ?? 0;
    const totalEvents = metrics.total_events_processed ?? 0;
    const anomalyRate = totalEvents > 0 ? (totalAnomalies / totalEvents) * 100 : 0.2;
    return { fpr, precision, recall, f1, accuracy, anomalyRate };
  }, [metrics]);

  const tiles = [
    { label: 'FPR', value: `${vals.fpr.toFixed(1)}%` },
    { label: 'Precision', value: `${vals.precision.toFixed(1)}%` },
    { label: 'Recall', value: `${vals.recall.toFixed(1)}%` },
    { label: 'F1 Score', value: `${vals.f1.toFixed(1)}%` },
    { label: 'Accuracy', value: `${vals.accuracy.toFixed(1)}%` },
    { label: 'Anomaly Rate', value: `${vals.anomalyRate.toFixed(1)}%` },
  ];

  return (
    <div className="panel flex flex-col h-full">
      <div className="p-3 border-b border-[var(--color-border)] flex items-center justify-between bg-[rgba(0,212,255,0.03)]">
        <div className="flex items-center gap-2">
          <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="var(--color-accent)" strokeWidth={2.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          <span className="text-[10px] font-bold uppercase tracking-widest text-white">Model Health</span>
        </div>
        <div className="relative group cursor-help">
          <svg width="12" height="12" fill="none" viewBox="0 0 24 24" stroke="currentColor" className="text-gray-500 hover:text-[var(--color-accent)] transition-colors">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div className="absolute right-0 top-full mt-2 hidden group-hover:block w-56 p-2.5 bg-[#0a0f1a] border border-[var(--color-accent)] rounded text-[10px] text-gray-300 z-[100] shadow-lg leading-relaxed">
            <strong className="text-[var(--color-accent)] block mb-1">Model Telemetry</strong>
            Real-time pipeline metrics showing overall FPR, accuracy, and rate of anomaly detections.
          </div>
        </div>
      </div>

      <div className="p-3 flex flex-col gap-3 h-full justify-between">
        {/* FPR Gauge */}
        <FprGauge fpr={vals.fpr} />

        {/* 6 Metric Tiles */}
        <div className="grid grid-cols-3 gap-2">
          {tiles.map((t) => (
            <div
              key={t.label}
              className="bg-[rgba(22,27,34,0.5)] border border-[var(--color-border)] rounded-md p-2 text-center flex flex-col items-center justify-center transition-all hover:border-[var(--color-accent)] hover:bg-[rgba(0,212,255,0.05)]"
            >
              <span className="text-sm font-bold font-mono text-white leading-tight">
                {t.value}
              </span>
              <span className="text-[7px] text-[var(--color-text-secondary)] font-bold uppercase tracking-wider mt-0.5 whitespace-nowrap">
                {t.label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
