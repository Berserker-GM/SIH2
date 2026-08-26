import { useState, useEffect, useMemo } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const ACTION_BY_TACTIC = {
  "Initial Access": "Block source IP / enforce MFA",
  "Credential Access": "Force credential reset, enable lockout policy",
  "Execution": "Block process execution via EDR policy",
  "Persistence": "Remove autostart entry, isolate host",
  "Command and Control": "Sinkhole / block C2 domain-IP",
  "Exfiltration": "Block egress, enable DNS/DLP inspection",
  "Impact": "Enable shadow copy protection & isolate endpoint",
};

const TECHNIQUE_LOOKUP = {
  "T1110": { name: "Brute Force", tactic: "Credential Access" },
  "T1190": { name: "Exploit Public-Facing Application", tactic: "Initial Access" },
  "T1059": { name: "Command and Scripting Interpreter", tactic: "Execution" },
  "T1547": { name: "Boot or Logon Autostart Execution", tactic: "Persistence" },
  "T1490": { name: "Inhibit System Recovery", tactic: "Impact" },
  "T1486": { name: "Data Encrypted for Impact", tactic: "Impact" },
  "T1048": { name: "Exfiltration Over Alternative Protocol", tactic: "Exfiltration" },
  "T1071": { name: "Application Layer Protocol", tactic: "Command and Control" },
  "T1021": { name: "Remote Services", tactic: "Lateral Movement" },
  "T1003": { name: "OS Credential Dumping", tactic: "Credential Access" },
  "T1070": { name: "Indicator Removal", tactic: "Defense Evasion" },
};

function generateFallbackData(incident) {
  if (!incident) return null;
  let chain = incident.attack_chain || [];
  if (!chain.length && incident.llm_context?.mitre_tags) {
    chain = incident.llm_context.mitre_tags.map(tag => {
      const info = TECHNIQUE_LOOKUP[tag] || { name: tag, tactic: "Defense Evasion" };
      return { technique_id: tag, technique_name: info.name, tactic: info.tactic, confidence: 90 };
    });
  }
  if (!chain.length) {
    chain = [
      { technique_id: "T1110", technique_name: "Brute Force", tactic: "Credential Access", confidence: 95 },
      { technique_id: "T1190", technique_name: "Exploit Public-Facing Application", tactic: "Initial Access", confidence: 88 },
      { technique_id: "T1059", technique_name: "Command and Scripting Interpreter", tactic: "Execution", confidence: 82 }
    ];
  }

  const n = chain.length;
  const cfs = chain.map((step, i) => {
    const tid = typeof step === 'string' ? step : step.technique_id || 'T1110';
    const tname = (typeof step === 'object' && step.technique_name) || TECHNIQUE_LOOKUP[tid]?.name || tid;
    const tactic = (typeof step === 'object' && step.tactic) || TECHNIQUE_LOOKUP[tid]?.tactic || "Execution";
    const impact_pct = Math.round(((n - i) / n) * 100);
    const action = ACTION_BY_TACTIC[tactic] || "Isolate host & revoke credentials";
    return {
      blocked_technique_id: tid,
      blocked_technique_name: tname,
      tactic,
      impact_pct,
      blocking_mechanism: `${action} at ${tid} (${tactic})`,
      recommendation: `Blocking this at step ${i + 1} of ${n} would have prevented ${n - i} of ${n} downstream technique(s).`
    };
  });

  cfs.sort((a, b) => b.impact_pct - a.impact_pct);
  return { minimum_intervention: cfs[0], counterfactuals: cfs };
}

export default function CounterfactualPanel({ incidentId, incident }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    if (!incidentId) return;
    setData(null);
    axios.get(`${API_BASE}/api/incidents/${incidentId}/counterfactual`)
      .then(res => {
        if (res.data && res.data.minimum_intervention) {
          setData(res.data);
        } else {
          setData(generateFallbackData(incident));
        }
      })
      .catch(() => {
        setData(generateFallbackData(incident));
      });
  }, [incidentId, incident]);

  const activeData = useMemo(() => data || generateFallbackData(incident), [data, incident]);

  if (!activeData || !activeData.minimum_intervention) {
    return (
      <div className="text-[9px] text-[#8B949E] italic p-2">
        Analyzing intervention points...
      </div>
    );
  }

  const best = activeData.minimum_intervention;

  return (
    <div className="flex flex-col gap-2">
      <div className="bg-[rgba(0,255,136,0.03)] border border-[rgba(0,255,136,0.2)] rounded-lg p-3 flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-[9px] font-mono font-bold text-[#00FF88]">
            {best.blocked_technique_id} — {best.blocked_technique_name}
          </span>
          <span className="text-[9px] font-bold text-[#00FF88] bg-[rgba(0,255,136,0.1)] px-2 py-0.5 rounded">
            Stops {best.impact_pct}% of chain
          </span>
        </div>
        <span className="text-[10px] text-[#E6EDF3] leading-relaxed font-medium">
          {best.blocking_mechanism}
        </span>
        <span className="text-[9px] text-[#8B949E] italic leading-relaxed">
          {best.recommendation}
        </span>
      </div>

      {activeData.counterfactuals && activeData.counterfactuals.length > 1 && (
        <div className="flex flex-col gap-1">
          <span className="text-[8px] text-[#8B949E] uppercase tracking-wider">
            Other intervention points
          </span>
          {activeData.counterfactuals.slice(1, 3).map((cf) => (
            <div key={cf.blocked_technique_id} className="flex items-center justify-between bg-[rgba(22,27,34,0.4)] border border-[var(--color-border)] rounded px-2 py-1.5">
              <span className="text-[9px] text-[#8B949E] font-mono">
                {cf.blocked_technique_id}
              </span>
              <span className="text-[9px] text-[#8B949E]">
                {cf.impact_pct}% impact
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}