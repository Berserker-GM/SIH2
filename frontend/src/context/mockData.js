export function generateMockData() {
  const TECHNIQUES = [
    { id: 'T1110', name: 'Brute Force', tactic: 'Credential Access' },
    { id: 'T1059', name: 'Command and Scripting Interpreter', tactic: 'Execution' },
    { id: 'T1053', name: 'Scheduled Task/Job', tactic: 'Persistence' },
    { id: 'T1021', name: 'Remote Services', tactic: 'Lateral Movement' },
    { id: 'T1486', name: 'Data Encrypted for Impact', tactic: 'Impact' },
    { id: 'T1071', name: 'Application Layer Protocol', tactic: 'Command and Control' },
    { id: 'T1048', name: 'Exfiltration Over Alternative Protocol', tactic: 'Exfiltration' },
    { id: 'T1070', name: 'Indicator Removal', tactic: 'Defense Evasion' },
    { id: 'T1547', name: 'Boot or Logon Autostart Execution', tactic: 'Persistence' },
    { id: 'T1078', name: 'Valid Accounts', tactic: 'Defense Evasion' },
    { id: 'T1003', name: 'OS Credential Dumping', tactic: 'Credential Access' },
    { id: 'T1569', name: 'System Services', tactic: 'Execution' },
  ];

  const SEVERITIES = ['critical', 'high', 'medium', 'low'];
  const EXTERNAL_IPS = ['203.0.113.99', '45.33.32.156', '198.51.100.23', '185.220.101.1', '91.219.237.22'];
  const INTERNAL_IPS = ['192.168.1.10', '192.168.1.25', '192.168.2.100', '10.0.0.50', '10.0.1.15'];
  const HOSTNAMES = ['dc01.corp.local', 'web-prod-01', 'db-master', 'fileserver', 'mail-gw'];
  const SUMMARIES = [
    'Telemetry indicates a sustained brute-force campaign targeting Edge Gateway (T1110), immediately followed by lateral movement to internal subnets. Execution of unverified scripts detected. Recommend immediate isolation of affected nodes and credential rotation.',
    'Anomalous credential access patterns observed. A compromised external actor is leveraging stolen session tokens to traverse the network perimeter, with subsequent queries directed at critical database instances. High confidence of privilege escalation attempts.',
    'Pre-ransomware indicators detected. Multiple endpoints show simultaneous abnormal file I/O operations and mass encryption behaviors following the execution of a malicious payload via PowerShell. Critical severity: initiate automated containment protocol.',
    'Behavioral models flag an Advanced Persistent Threat (APT) signature. Prolonged, low-volume network reconnaissance correlated with unauthorized scheduled task creation (T1053) for persistence. The threat actor is likely establishing long-term backdoor access.',
    'High-volume data exfiltration attempt blocked. Analysis reveals an attempt to tunnel encrypted payloads over DNS to a known malicious C2 infrastructure. Network segmentation and DNS sinkholing strongly advised.',
    'Active defense evasion tactics identified. Security logs and event indicators are being systematically cleared across multiple compromised hosts. The adversary is actively attempting to obscure their attack path. Forensic imaging required.',
  ];

  const now = Date.now();
  const incidents = [];

  for (let i = 0; i < 24; i++) {
    const severity = SEVERITIES[Math.floor(Math.random() * SEVERITIES.length)];
    const riskScore = severity === 'critical' ? 75 + Math.floor(Math.random() * 25) :
                      severity === 'high' ? 55 + Math.floor(Math.random() * 20) :
                      severity === 'medium' ? 30 + Math.floor(Math.random() * 25) :
                      5 + Math.floor(Math.random() * 25);

    const chainLength = 2 + Math.floor(Math.random() * 4);
    const startIdx = Math.floor(Math.random() * (TECHNIQUES.length - chainLength));
    const attack_chain = [];
    for (let j = 0; j < chainLength; j++) {
      const tech = TECHNIQUES[(startIdx + j) % TECHNIQUES.length];
      attack_chain.push({
        technique_id: tech.id,
        technique_name: tech.name,
        tactic: tech.tactic,
        confidence: 55 + Math.floor(Math.random() * 45),
        timestamp: new Date(now - (24 - i) * 3600000 + j * 900000).toISOString(),
      });
    }

    const nextTech = TECHNIQUES[(startIdx + chainLength) % TECHNIQUES.length];
    const affectedAssets = [
      EXTERNAL_IPS[Math.floor(Math.random() * EXTERNAL_IPS.length)],
      ...(Math.random() > 0.3 ? [INTERNAL_IPS[Math.floor(Math.random() * INTERNAL_IPS.length)]] : []),
      ...(Math.random() > 0.5 ? [HOSTNAMES[Math.floor(Math.random() * HOSTNAMES.length)]] : []),
    ];

    incidents.push({
      incident_id: `INC-${String(1000 + i).padStart(4, '0')}`,
      severity,
      risk_score: riskScore,
      detected_at: new Date(now - (24 - i) * 3600000 + Math.floor(Math.random() * 1800000)).toISOString(),
      affected_assets: affectedAssets,
      attack_chain,
      predicted_next: {
        technique_name: nextTech.name,
        technique_id: nextTech.id,
        probability: 0.55 + Math.random() * 0.4,
      },
      anomaly_scores: {
        lstm: 40 + Math.floor(Math.random() * 55),
        iforest: 30 + Math.floor(Math.random() * 65),
        gnn: 45 + Math.floor(Math.random() * 50),
      },
      model_metrics: {
        fpr: 1.5 + Math.random() * 8,
        precision: 82 + Math.random() * 15,
        recall: 78 + Math.random() * 18,
        f1: 80 + Math.random() * 16,
        accuracy: 90 + Math.random() * 8,
      },
      llm_context: {
        summary: SUMMARIES[Math.floor(Math.random() * SUMMARIES.length)],
        mitre_tags: attack_chain.map(s => s.technique_id),
        severity_reason: `Risk score ${riskScore} exceeds threshold. Multiple high-confidence attack stages detected.`,
      },
    });
  }

  const metrics = {
    fpr: 3.2 + Math.random() * 5,
    precision: 89 + Math.random() * 8,
    recall: 84 + Math.random() * 12,
    f1: 86 + Math.random() * 10,
    accuracy: 93 + Math.random() * 5,
    total_events_processed: 148000 + Math.floor(Math.random() * 50000),
    total_anomalies_detected: 340 + Math.floor(Math.random() * 200),
  };

  const SOURCE_TYPES = ['syslog', 'netflow', 'sysmon', 'auth'];
  const ACTIONS = [
    'SSH login attempt', 'HTTP GET /admin', 'Process spawned: cmd.exe',
    'File created: payload.exe', 'Registry modified: Run key', 'DNS query: c2.evil.com',
    'Outbound connection: 443', 'Failed authentication', 'Privilege escalation',
    'File integrity change', 'Cron job modified', 'Shadow copy deleted',
    'Lateral movement via SMB', 'Data upload 2.3GB', 'Port scan detected',
  ];
  const STATUSES = ['success', 'blocked', 'failed', 'allowed'];

  const events = [];
  for (let i = 0; i < 100; i++) {
    const hasMitre = Math.random() > 0.6;
    events.push({
      timestamp: new Date(now - (100 - i) * 30000).toISOString(),
      source_type: SOURCE_TYPES[Math.floor(Math.random() * SOURCE_TYPES.length)],
      src_ip: Math.random() > 0.4
        ? EXTERNAL_IPS[Math.floor(Math.random() * EXTERNAL_IPS.length)]
        : INTERNAL_IPS[Math.floor(Math.random() * INTERNAL_IPS.length)],
      action: ACTIONS[Math.floor(Math.random() * ACTIONS.length)],
      status: STATUSES[Math.floor(Math.random() * STATUSES.length)],
      mitre_tag: hasMitre ? TECHNIQUES[Math.floor(Math.random() * TECHNIQUES.length)].id : null,
    });
  }

  const tickerEvents = [];
  incidents.forEach(inc => {
    inc.attack_chain.forEach(step => {
      tickerEvents.push({
        timestamp: step.timestamp,
        technique_id: step.technique_id,
        technique_name: step.technique_name,
      });
    });
  });
  tickerEvents.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

  return { incidents, metrics, events, tickerEvents: tickerEvents.slice(0, 20) };
}

export function fetchMockIncidentDetail(id, incidentsList) {
  return new Promise((resolve) => {
    setTimeout(() => {
      const inc = incidentsList.find((i) => i.incident_id === id);
      if (inc) {
        // Enriched detail view
        resolve({
          ...inc,
          enrichment: {
            process_tree: ["cmd.exe", "powershell.exe", "svchost.exe"],
            related_ips: ["192.168.1.100", "45.33.32.156"],
            extracted_credentials: ["admin", "root"],
          }
        });
      } else {
        resolve(null);
      }
    }, 400); // Simulate network latency
  });
}

export function fetchMockIncidentGraph(id) {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Generate a realistic attack graph for the given incident
      const nodes = [
        { id: '1', type: 'input', data: { label: 'Attacker IP' }, position: { x: 0, y: 150 } },
        { id: '2', data: { label: 'Firewall (External)' }, position: { x: 250, y: 150 } },
        { id: '3', data: { label: 'Web Server (DMZ)' }, position: { x: 500, y: 150 } },
        { id: '4', data: { label: 'Internal Switch' }, position: { x: 750, y: 150 } },
        { id: '5', type: 'output', data: { label: 'Database (PII)' }, position: { x: 1000, y: 50 } },
        { id: '6', type: 'output', data: { label: 'Domain Controller' }, position: { x: 1000, y: 250 } },
      ];

      const edges = [
        { id: 'e1-2', source: '1', target: '2', animated: true, style: { stroke: '#EF4444' } },
        { id: 'e2-3', source: '2', target: '3', animated: true, style: { stroke: '#EF4444' } },
        { id: 'e3-4', source: '3', target: '4', animated: true, style: { stroke: '#F97316' } },
        { id: 'e4-5', source: '4', target: '5', animated: false, style: { stroke: '#FACC15' } },
        { id: 'e4-6', source: '4', target: '6', animated: true, style: { stroke: '#EF4444' } },
      ];

      resolve({ nodes, edges });
    }, 600);
  });
}
