import redis
import json
import uuid
from datetime import datetime, timezone, timedelta
import random

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

TECHNIQUES = [
    {"technique_id": "T1110", "technique_name": "Brute Force", "tactic": "Credential Access"},
    {"technique_id": "T1190", "technique_name": "Exploit Public-Facing Application", "tactic": "Initial Access"},
    {"technique_id": "T1059", "technique_name": "Command and Scripting Interpreter", "tactic": "Execution"},
    {"technique_id": "T1547", "technique_name": "Boot or Logon Autostart Execution", "tactic": "Persistence"},
    {"technique_id": "T1490", "technique_name": "Inhibit System Recovery", "tactic": "Impact"},
    {"technique_id": "T1486", "technique_name": "Data Encrypted for Impact", "tactic": "Impact"},
    {"technique_id": "T1048", "technique_name": "Exfiltration Over Alternative Protocol", "tactic": "Exfiltration"},
    {"technique_id": "T1071", "technique_name": "Application Layer Protocol", "tactic": "Command and Control"},
    {"technique_id": "T1021", "technique_name": "Remote Services", "tactic": "Lateral Movement"},
    {"technique_id": "T1003", "technique_name": "OS Credential Dumping", "tactic": "Credential Access"},
]

SEVERITIES = ["critical", "high", "medium", "low"]
EXTERNAL_IPS = ["203.0.113.99", "45.33.32.156", "198.51.100.23", "185.220.101.1", "91.219.237.22"]
INTERNAL_IPS = ["192.168.1.10", "192.168.1.25", "192.168.2.100", "10.0.0.50", "10.0.1.15"]

SUMMARIES = [
    "Telemetry indicates a sustained brute-force campaign targeting the edge gateway, immediately followed by lateral movement to internal subnets. Execution of unverified scripts detected. Recommend immediate isolation of affected nodes and credential rotation.",
    "Anomalous credential access patterns observed. A compromised external actor is leveraging stolen session tokens to traverse the network perimeter with subsequent queries directed at critical database instances.",
    "Pre-ransomware indicators detected. Multiple endpoints show simultaneous abnormal file I/O operations and mass encryption behaviors following execution of a malicious payload via PowerShell. Initiate automated containment protocol.",
    "Behavioral models flag an Advanced Persistent Threat signature. Prolonged low-volume network reconnaissance correlated with unauthorized scheduled task creation for persistence. Threat actor establishing long-term backdoor access.",
    "High-volume data exfiltration attempt blocked. Analysis reveals an attempt to tunnel encrypted payloads over DNS to known malicious C2 infrastructure. Network segmentation and DNS sinkholing strongly advised.",
]

now = datetime.now(timezone.utc)
created = 0

for i in range(10):
    severity = random.choice(SEVERITIES)
    risk_score = {
        "critical": random.randint(80, 99),
        "high": random.randint(60, 79),
        "medium": random.randint(35, 59),
        "low": random.randint(10, 34),
    }[severity]

    chain_length = random.randint(2, 5)
    start_idx = random.randint(0, len(TECHNIQUES) - chain_length)
    attack_chain = []
    for j in range(chain_length):
        tech = TECHNIQUES[(start_idx + j) % len(TECHNIQUES)]
        attack_chain.append({
            **tech,
            "confidence": random.randint(65, 98),
            "timestamp": (now - timedelta(hours=random.randint(1, 24))).isoformat(),
        })

    next_tech = TECHNIQUES[(start_idx + chain_length) % len(TECHNIQUES)]
    src_ip = random.choice(EXTERNAL_IPS)

    incident = {
        "incident_id": f"INC-{str(uuid.uuid4())[:8].upper()}",
        "severity": severity,
        "risk_score": risk_score,
        "detected_at": (now - timedelta(hours=random.randint(1, 24))).isoformat(),
        "timestamp": (now - timedelta(hours=random.randint(1, 24))).isoformat(),
        "affected_assets": [src_ip, random.choice(INTERNAL_IPS)],
        "attack_chain": attack_chain,
        "predicted_next": {
            "technique_id": next_tech["technique_id"],
            "technique_name": next_tech["technique_name"],
            "tactic": next_tech["tactic"],
            "probability": round(random.uniform(0.55, 0.95), 2),
        },
        "anomaly_scores": {
            "lstm": round(random.uniform(0.4, 0.95), 2),
            "iforest": round(random.uniform(0.35, 0.90), 2),
            "gnn": round(random.uniform(0.45, 0.92), 2),
        },
        "llm_context": {
            "summary": random.choice(SUMMARIES),
            "mitre_tags": [s["technique_id"] for s in attack_chain],
            "severity_reason": f"Risk score {risk_score} with {chain_length} confirmed attack stages detected.",
        },
        "src_ip": src_ip,
    }

    r.lpush("incidents:new", json.dumps(incident))
    r.publish("incidents:new", json.dumps(incident))
    created += 1
    print(f"Created {incident['incident_id']} | {severity.upper()} | Risk: {risk_score}/100")

print(f"\nTotal incidents created: {created}")