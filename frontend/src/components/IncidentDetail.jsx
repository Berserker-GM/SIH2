// import { useMemo } from 'react';
import { useMemo } from 'react';
import CounterfactualPanel from './CounterfactualPanel';
import { format } from 'date-fns';

const SEVERITY_COLORS = {
  critical: '#FF4444',
  high: '#FF8C00',
  medium: '#FFD700',
  low: '#00FF88',
};

function RobotIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0 drop-shadow-[0_0_4px_var(--color-accent)]">
      <rect x="3" y="11" width="18" height="10" rx="2" />
      <circle cx="12" cy="5" r="2" />
      <path d="M12 7v4" />
      <line x1="8" y1="16" x2="8" y2="16.01" />
      <line x1="16" y1="16" x2="16" y2="16.01" />
    </svg>
  );
}

function RiskGauge({ score, color }) {
  // Semicircle arc logic: radius=40, arc length = pi * 40 ≈ 125.66
  const arcLength = 125.66;
  const offset = arcLength - (score / 100) * arcLength;

  return (
    <div className="flex flex-col items-center justify-center p-3 bg-[rgba(22,27,34,0.4)] border border-[var(--color-border)] rounded-lg">
      <div className="relative w-44 h-24 flex items-end justify-center overflow-hidden">
        <svg viewBox="0 0 100 55" className="w-full h-full">
          {/* Background track */}
          <path
            d="M 10 50 A 40 40 0 0 1 90 50"
            fill="none"
            stroke="rgba(255, 255, 255, 0.05)"
            strokeWidth="8"
            strokeLinecap="round"
          />
          {/* Active gauge segment */}
          <path
            d="M 10 50 A 40 40 0 0 1 90 50"
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={arcLength}
            strokeDashoffset={offset}
            className="transition-all duration-1000 ease-out"
            style={{ filter: `drop-shadow(0 0 3px ${color})` }}
          />
        </svg>
        <div className="absolute bottom-1 flex flex-col items-center">
          <span className="text-3xl font-black font-mono tracking-tight text-white leading-none">
            {score}
          </span>
          <span className="text-[7px] font-bold uppercase tracking-wider text-[#8B949E] mt-1">
            RISK INDEX
          </span>
        </div>
      </div>
    </div>
  );
}

function AnomalyBar({ label, score }) {
  // Convert 0-1 range to percentage
  const pct = Math.min(Math.max(score * 100, 0), 100);
  return (
    <div className="flex items-center gap-3">
      <span className="text-[9px] text-[#8B949E] font-bold uppercase w-12 text-right tracking-wider">
        {label}
      </span>
      <div className="flex-1 bg-[rgba(255,255,255,0.05)] h-2 rounded-full overflow-hidden border border-[rgba(255,255,255,0.02)]">
        <div
          className="h-full bg-[var(--color-accent)] rounded-full transition-all duration-1000 ease-out"
          style={{ width: `${pct}%`, filter: 'drop-shadow(0 0 4px var(--color-accent))' }}
        />
      </div>
      <span className="text-[10px] text-[#F0F6FC] font-mono w-8 text-right font-bold">
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}

export default function IncidentDetail({ incident }) {
  if (!incident) {
    return (
      <div className="panel flex items-center justify-center text-[#8B949E] text-xs italic h-full bg-[var(--color-bg-card)]">
        Select an incident to view deep analysis.
      </div>
    );
  }

  const severity = incident.severity || 'low';
  const color = SEVERITY_COLORS[severity] || SEVERITY_COLORS.low;
  const anomalyScores = incident.anomaly_scores || {};
  const llmContext = incident.llm_context || {};
  const mitreTags = llmContext.mitre_tags || [];
  const summary = llmContext.summary || '';
  const predictedAttack = incident.predicted_next || null;
  const affectedAssets = incident.affected_assets || [];

  return (
    <div className="panel animate-fade-in flex flex-col bg-[var(--color-bg-card)]">
    
      {/* Header */}
      <div className="p-4 border-b border-[var(--color-border)] bg-[rgba(255,255,255,0.02)] flex items-center justify-between">
        <div className="flex flex-col gap-0.5">
          <span className="font-mono text-[11px] font-bold text-[#F0F6FC] tracking-wider">
            {incident.incident_id}
          </span>
          <span className="text-[8px] text-[#8B949E]">
            DETECTED AT: {format(new Date(incident.detected_at), 'yyyy-MM-dd HH:mm:ss')} UTC
          </span>
        </div>
        <span
          className="text-[8px] font-extrabold uppercase tracking-widest px-2 py-0.5 rounded"
          style={{ color, backgroundColor: `${color}15`, border: `1px solid ${color}33` }}
        >
          {severity}
        </span>
      </div>

      <div className="p-4 flex flex-col gap-4.5">
        {/* Risk Gauge */}
        <RiskGauge score={incident.risk_score} color={color} />

        {/* LLM summary blockquote */}
        {summary && (
          <div className="flex flex-col gap-1.5">
            <span className="text-[8px] font-bold uppercase tracking-widest text-[#8B949E]">
              Analyst Insight
            </span>
            <blockquote className="flex gap-2.5 bg-[rgba(0,212,255,0.03)] border-l-2 border-[var(--color-accent)] p-3 rounded-r-md text-xs text-[#E6EDF3] leading-relaxed">
              <RobotIcon />
              <div>{summary}</div>
            </blockquote>
          </div>
        )}

        {/* Predicted Next Stage */}
        {predictedAttack && (
          <div className="flex flex-col gap-1.5">
            <span className="text-[8px] font-bold uppercase tracking-widest text-[#8B949E]">
              Next Phase Prediction
            </span>
            <div className="bg-[rgba(255,140,0,0.05)] border border-[rgba(255,140,0,0.2)] rounded-lg p-3 flex items-center justify-between animate-pulse">
              <div className="flex flex-col gap-0.5">
                <span className="text-xs font-bold text-[#FF8C00]">
                  {predictedAttack.technique_name || 'Unknown Technique'}
                </span>
                {predictedAttack.technique_id && (
                  <span className="text-[9px] text-[#8B949E] font-mono">
                    {predictedAttack.technique_id}
                  </span>
                )}
              </div>
              <span className="text-xl font-black text-[#FF8C00] font-mono leading-none">
                {Math.round((predictedAttack.probability || 0) * 100)}%
              </span>
            </div>
          </div>
        )}

        {/* MITRE Tags */}
        {mitreTags.length > 0 && (
          <div className="flex flex-col gap-1.5">
            <span className="text-[8px] font-bold uppercase tracking-widest text-[#8B949E]">
              MITRE ATT&CK Indicators
            </span>
            <div className="flex flex-wrap gap-1.5">
              {mitreTags.map((tag) => (
                <a
                  key={tag}
                  href={`https://attack.mitre.org/techniques/${tag}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="bg-[rgba(255,255,255,0.03)] border border-[var(--color-border)] hover:border-[var(--color-accent)] hover:text-white rounded px-2 py-0.5 text-[9px] font-mono text-[#8B949E] transition-colors"
                >
                  {tag}
                </a>
              ))}
            </div>
          </div>
        )}

        {/* Compromised Assets */}
        {affectedAssets.length > 0 && (
          <div className="flex flex-col gap-1.5">
            <span className="text-[8px] font-bold uppercase tracking-widest text-[#8B949E]">
              Affected Scope
            </span>
            <div className="flex flex-wrap gap-1.5">
              {affectedAssets.map((asset, i) => (
                <span
                  key={`${asset}-${i}`}
                  className="bg-[#161B22] border border-[var(--color-border)] text-[var(--color-accent)] rounded px-2 py-0.5 text-[9px] font-mono"
                >
                  {asset}
                </span>
              ))}
            </div>
          </div>
        )}
        {/* Counterfactual — minimum intervention */}
        <div className="flex flex-col gap-1.5">
          <span className="text-[8px] font-bold uppercase tracking-widest text-[#8B949E]">
            Minimum Intervention to Stop Attack
          </span>
          <CounterfactualPanel incidentId={incident.incident_id} incident={incident} />
        </div>

        {/* Engine Anomaly Scores */}
        <div className="flex flex-col gap-2 bg-[rgba(22,27,34,0.4)] border border-[var(--color-border)] p-3 rounded-lg">
          <span className="text-[8px] font-bold uppercase tracking-widest text-[#8B949E] mb-1">
            Engine Anomaly Scores
          </span>
          <div className="flex flex-col gap-2">
            <AnomalyBar label="LSTM" score={anomalyScores.lstm ?? 0} />
            <AnomalyBar label="IForest" score={anomalyScores.iforest ?? 0} />
            <AnomalyBar label="GNN" score={anomalyScores.gnn ?? 0} />
          </div>
        </div>

        {/* One-click isolate button */}
        {affectedAssets.length > 0 && (
          <button
            onClick={() => alert(`ISOLATION COMMAND SENT\n\nHost: ${affectedAssets[0]}\nStatus: Isolated from network\nTime: ${new Date().toUTCString()}\n\nResponse team has been notified.`)}
            className="w-full py-2.5 px-3 bg-[rgba(255,68,68,0.08)] border border-[#FF4444] text-[#FF4444] text-[10px] font-black uppercase tracking-widest rounded-lg hover:bg-[rgba(255,68,68,0.18)] transition-all active:scale-95 cursor-pointer"
          >
            ⚡ Isolate Host — {affectedAssets[0]}
          </button>
        )}
      </div>
    </div>
  );
}

