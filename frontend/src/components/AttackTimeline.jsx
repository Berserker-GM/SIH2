export default function AttackTimeline({ incident }) {
  if (!incident) {
    return (
      <div className="panel">
        <div className="panel-header">ATTACK CHAIN</div>
        <div className="flex items-center justify-center h-32">
          <span className="text-[#6E7681] text-sm italic">
            Select an incident to view attack chain
          </span>
        </div>
      </div>
    );
  }

  const chain = incident.attack_chain || [];
  const predicted = incident.predicted_next;

  return (
    <div className="panel">
      <div className="panel-header">
        ATTACK CHAIN
        <span className="text-[#00D4FF] text-xs ml-2 font-[family-name:var(--font-mono)]">
          {incident.incident_id}
        </span>
      </div>

      <div className="overflow-x-auto pb-2">
        <div className="flex flex-row items-stretch gap-0 min-w-max px-1 py-2">
          {chain.map((step, idx) => (
            <div key={idx} className="flex items-center animate-fade-in">
              {/* Technique node */}
              <div className="bg-[#21262D] border border-[#30363D] rounded-lg p-3 min-w-[160px] flex flex-col gap-1.5">
                {/* MITRE badge */}
                <span className="inline-flex self-start bg-[#00D4FF]/10 text-[#00D4FF] rounded px-2 py-0.5 text-xs font-[family-name:var(--font-mono)]">
                  {step.technique_id}
                </span>

                {/* Technique name */}
                <span className="text-sm font-medium text-[#E6EDF3] leading-snug">
                  {step.technique_name}
                </span>

                {/* Tactic */}
                <span className="text-xs text-[#8B949E] italic">
                  {step.tactic}
                </span>

                {/* Confidence bar */}
                <div className="mt-1">
                  <div className="flex items-center justify-between mb-0.5">
                    <span className="text-[10px] text-[#6E7681]">
                      confidence
                    </span>
                    <span className="text-[10px] text-[#00D4FF] font-[family-name:var(--font-mono)]">
                      {step.confidence}%
                    </span>
                  </div>
                  <div className="w-full h-1 bg-[#30363D] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[#00D4FF] rounded-full transition-all duration-500"
                      style={{ width: `${step.confidence}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Arrow connector */}
              {(idx < chain.length - 1 || predicted) && (
                <div className="flex items-center px-1.5">
                  <svg
                    width="28"
                    height="16"
                    viewBox="0 0 28 16"
                    fill="none"
                    className="flex-shrink-0"
                  >
                    <line
                      x1="0"
                      y1="8"
                      x2="22"
                      y2="8"
                      stroke="#484F58"
                      strokeWidth="1.5"
                    />
                    <polyline
                      points="18,3 24,8 18,13"
                      stroke="#484F58"
                      strokeWidth="1.5"
                      fill="none"
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                  </svg>
                </div>
              )}
            </div>
          ))}

          {/* Predicted next node */}
          {predicted && (
            <div className="flex items-center animate-fade-in">
              <div className="border-dashed border-2 border-[#FF8C00] bg-[#FF8C00]/5 rounded-lg p-3 min-w-[160px] flex flex-col gap-1.5">
                <span className="inline-flex self-start bg-[#FF8C00]/15 text-[#FF8C00] rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider font-[family-name:var(--font-mono)]">
                  Predicted Next
                </span>

                <span className="text-sm font-medium text-[#E6EDF3] leading-snug">
                  {predicted.technique_name}
                </span>

                {predicted.technique_id && (
                  <span className="text-xs text-[#8B949E] font-[family-name:var(--font-mono)]">
                    {predicted.technique_id}
                  </span>
                )}

                <div className="mt-1 flex items-baseline gap-1">
                  <span className="text-2xl font-bold text-[#FF8C00] font-[family-name:var(--font-mono)]">
                    {Math.round((predicted.probability || 0) * 100)}%
                  </span>
                  <span className="text-[10px] text-[#8B949E]">
                    probability
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
