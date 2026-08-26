#!/usr/bin/env python3
# =============================================================================
# normative/tests/test_dev1.py
#
# Run:
#   python -m pytest tests/test_dev1.py -v
#   python tests/test_dev1.py                (standalone)
# =============================================================================

import json
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Ensure the project root (Normnative/) is in sys.path so absolute imports work.
# This file is at tests/test_dev1.py → project root = parent directory (..)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now absolute imports from the project root
from src.ingestor.parsers.auth_parser import parse as parse_auth
from src.ingestor.parsers.network_parser import parse as parse_network, detect_payload_flags
from src.ingestor.parsers.endpoint_parser import parse as parse_endpoint
from src.ingestor.parsers.syslog_parser import parse as parse_syslog
from src.ingestor.feature_extractor import FeatureExtractor, FEATURE_ORDER, INPUT_DIM
from src.ingestor.mitre_lookup import match_rule, MITRE_RULES, MITRE_BY_TECHNIQUE
# ─────────────────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name: str, condition: bool, detail: str = ""):
    icon = PASS if condition else FAIL
    results.append((name, condition))
    print(f"  {icon}  {name}" + (f"  ({detail})" if detail else ""))
    return condition


# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Auth Parser ──────────────────────────────────────────────────────────")

SSH_FAIL_LINE  = "Jan 15 14:22:01 web01 sshd[1234]: Failed password for root from 203.0.113.99 port 44210 ssh2"
SSH_SUCCESS_LINE = "Jan 15 14:23:11 web01 sshd[1234]: Accepted password for root from 203.0.113.99 port 44210 ssh2"

evt = parse_auth(SSH_FAIL_LINE, "192.168.1.20")
check("SSH fail parses",           evt is not None)
check("source_type = auth",        evt and evt["source_type"] == "auth")
check("status = FAILURE",          evt and evt["status"] == "FAILURE")
check("src_ip captured",           evt and evt["src_ip"] == "203.0.113.99")
check("action = LOGIN_ATTEMPT",    evt and evt["action"] == "LOGIN_ATTEMPT")
check("payload_flags is list",     evt and isinstance(evt["payload_flags"], list))
check("all required keys present", evt and all(k in evt for k in [
    "event_id","timestamp","source_type","src_ip","dest_ip",
    "src_port","dest_port","action","payload_flags","raw"
]))

evt2 = parse_auth(SSH_SUCCESS_LINE, "192.168.1.20")
check("SSH success parses",        evt2 is not None)
check("status = SUCCESS",          evt2 and evt2["status"] == "SUCCESS")
check("user captured",             evt2 and evt2["user"] == "root")

# Windows JSON event
win_evt = json.dumps({"EventID": 4625, "SubjectUserName": "admin", "IpAddress": "10.0.0.99", "IpPort": "3389"})
evt3 = parse_auth(win_evt, "192.168.1.5")
check("Windows 4625 (fail) parses",  evt3 is not None)
check("Windows FAILURE status",       evt3 and evt3["status"] == "FAILURE")

win_ok = json.dumps({"EventID": 4624, "SubjectUserName": "jsmith", "IpAddress": "192.168.1.50", "IpPort": "0"})
evt4 = parse_auth(win_ok, "192.168.1.5")
check("Windows 4624 (success) parses", evt4 is not None)
check("Windows SUCCESS status",         evt4 and evt4["status"] == "SUCCESS")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Network Parser ───────────────────────────────────────────────────────")

APACHE_BENIGN  = '198.51.100.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 2326'
APACHE_LFI     = '203.0.113.99 - - [10/Oct/2000:13:56:01 -0700] "GET /page?file=../../../../etc/passwd HTTP/1.1" 200 512'
APACHE_404     = '198.51.100.5 - - [10/Oct/2000:13:57:00 -0700] "GET /missing HTTP/1.1" 404 123'

e = parse_network(APACHE_BENIGN, "192.168.1.10")
check("Apache benign parses",         e is not None)
check("action = HTTP_REQUEST",        e and e["action"] == "HTTP_REQUEST")
check("http_status = 200",            e and e["http_status"] == 200)
check("bytes_sent captured",          e and e["bytes_sent"] == 2326)
check("no payload flags on benign",   e and e["payload_flags"] == [])

e2 = parse_network(APACHE_LFI, "192.168.1.10")
check("Apache LFI parses",            e2 is not None)
check("LFI flag detected",            e2 and "LFI" in e2["payload_flags"])
check("TRAVERSAL flag detected",      e2 and "TRAVERSAL" in e2["payload_flags"])
check("status SUCCESS on 200",        e2 and e2["status"] == "SUCCESS")

# Payload detection
check("detect_payload_flags LFI",     "LFI" in detect_payload_flags("../etc/passwd"))
check("detect_payload_flags RCE",     "RCE" in detect_payload_flags("whoami"))
check("detect_payload_flags SQLi",    "SQLi" in detect_payload_flags("union select"))
check("detect_payload_flags SSRF",    "SSRF" in detect_payload_flags("169.254.169.254"))
check("detect_payload_flags empty",   detect_payload_flags("") == [])

# Netflow JSON
netflow = json.dumps({"src_ip": "192.168.1.30", "dest_ip": "45.33.32.156",
                       "bytes_sent": 75000000, "src_port": 43000, "dest_port": 443})
e3 = parse_network(netflow, "192.168.1.30")
check("Netflow JSON parses",          e3 is not None)
check("bytes_sent 75MB",              e3 and e3["bytes_sent"] == 75_000_000)
check("dest_ip captured",             e3 and e3["dest_ip"] == "45.33.32.156")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Endpoint Parser ──────────────────────────────────────────────────────")

SYSMON_PROC = json.dumps({
    "EventID": 1, "Image": "/bin/bash", "ParentImage": "/usr/sbin/nginx",
    "CommandLine": "/bin/bash -i", "User": "www-data"
})
SYSMON_FILE = json.dumps({
    "EventID": 11, "TargetFilename": "/etc/cron.d/backdoor",
    "Image": "/bin/bash", "User": "www-data"
})
SYSMON_REG = json.dumps({
    "EventID": 13,
    "TargetObject": r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run\malware",
    "Image": "notepad.exe", "User": "jsmith"
})

e = parse_endpoint(SYSMON_PROC, "192.168.1.10")
check("Sysmon proc create parses",    e is not None)
check("action = PROCESS_SPAWN",       e and e["action"] == "PROCESS_SPAWN")
check("parent_proc = nginx",          e and e["parent_proc"] == "nginx")
check("child_proc = /bin/bash",       e and e["child_proc"] == "/bin/bash")
check("non_admin_user = True",        e and e["non_admin_user"] == True)

e2 = parse_endpoint(SYSMON_FILE, "192.168.1.10")
check("Sysmon file create parses",    e2 is not None)
check("action = FILE_WRITE",          e2 and e2["action"] == "FILE_WRITE")
check("file_path captured",           e2 and e2["file_path"] == "/etc/cron.d/backdoor")

e3 = parse_endpoint(SYSMON_REG, "192.168.1.10")
check("Sysmon registry write parses", e3 is not None)
check("action = REGISTRY_WRITE",      e3 and e3["action"] == "REGISTRY_WRITE")
check("registry path captured",       e3 and "CurrentVersion" in (e3["file_path"] or ""))

# Generic process JSON
generic = json.dumps({
    "source_type": "endpoint", "parent_proc": "apache2", "child_proc": "/bin/sh",
    "user": "daemon", "src_ip": "10.0.0.5", "cpu_pct": 5.0
})
e4 = parse_endpoint(generic, "10.0.0.5")
check("Generic process JSON parses",  e4 is not None)
check("parent_proc = apache2",        e4 and "apache" in (e4["parent_proc"] or ""))


# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Syslog Parser ────────────────────────────────────────────────────────")

CRON_EDIT  = "Jun 15 14:23:11 web01 cron[999]: (www-data) REPLACE (www-data)"
BULK_FILES = "Jun 15 14:30:00 web01 cryptd[1234]: 312 files encrypted cpu:87.3"
FILE_INTEG = "Jun 15 14:31:00 web01 aide[555]: changed: /etc/passwd"

e = parse_syslog(CRON_EDIT, "192.168.1.10")
check("Cron edit parses",             e is not None)
check("action = CRON_MODIFY",         e and e["action"] == "CRON_MODIFY")
check("non_admin_user = True",        e and e["non_admin_user"] == True)
check("file_path contains cron.d",    e and e["file_path"] and "cron" in e["file_path"])

e2 = parse_syslog(BULK_FILES, "192.168.1.10")
check("Bulk file op parses",          e2 is not None)
check("action = BULK_FILE_OPERATION", e2 and e2["action"] == "BULK_FILE_OPERATION")
check("file_ops = 312",               e2 and e2["file_ops"] == 312)
check("cpu_pct = 87.3",               e2 and abs((e2["cpu_pct"] or 0) - 87.3) < 0.01)

e3 = parse_syslog(FILE_INTEG, "192.168.1.10")
check("File integrity change parses", e3 is not None)
check("action = FILE_INTEGRITY_CHANGE", e3 and e3["action"] == "FILE_INTEGRITY_CHANGE")
check("file_path = /etc/passwd",      e3 and e3["file_path"] == "/etc/passwd")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Feature Extractor ────────────────────────────────────────────────────")

check(f"INPUT_DIM = {INPUT_DIM}",     INPUT_DIM == 11)
check(f"FEATURE_ORDER has 11 items",  len(FEATURE_ORDER) == 11)
check("feature_vector[0] = login_fail_count",    FEATURE_ORDER[0] == "login_fail_count")
check("feature_vector[3] = bytes_sent_total",    FEATURE_ORDER[3] == "bytes_sent_total")
check("feature_vector[8] = entropy_dest_ports",  FEATURE_ORDER[8] == "entropy_dest_ports")

extractor = FeatureExtractor(window_size=5)   # small window for test

# Build 5 events to fill window
base_event = {
    "event_id": "test-id", "timestamp": "2024-01-15T14:00:00Z",
    "source_type": "network", "src_ip": "10.0.0.1", "dest_ip": "8.8.8.8",
    "src_port": 54321, "dest_port": 443, "user": None, "action": "HTTP_REQUEST",
    "status": "SUCCESS", "process_name": None, "parent_proc": None, "child_proc": None,
    "bytes_sent": 1000, "uri_query": "/test", "payload_flags": [],
    "file_path": None, "file_ops": None, "cpu_pct": None, "non_admin_user": None,
    "http_status": 200, "raw": {},
}
import uuid
ml_vec = None
for i in range(5):
    import copy
    evt = copy.deepcopy(base_event)
    evt["event_id"] = str(uuid.uuid4())
    evt["dest_port"] = 443 + i  # vary port for entropy
    ml_vec, graph_edge = extractor.process(evt)

check("ml_vector emitted after window full",     ml_vec is not None)
check("ml_vector has feature_vector key",         ml_vec and "feature_vector" in ml_vec)
check("feature_vector length = 11",              ml_vec and len(ml_vec["feature_vector"]) == 11)
check("ml_vector has window_events",              ml_vec and len(ml_vec["window_events"]) == 5)
check("ml_vector has host_key",                   ml_vec and ml_vec["host_key"] == "10.0.0.1")
check("graph_edge has required keys",             all(k in graph_edge for k in
    ["edge_id", "src_node", "dst_node", "edge_type", "weight", "timestamp", "event_id"]))
check("graph_edge src_node = src_ip",             graph_edge["src_node"] == "10.0.0.1")
check("graph_edge weight 0-1",                    0.0 <= graph_edge["weight"] <= 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
print("\n── MITRE Lookup ─────────────────────────────────────────────────────────")

check("7 rules defined",              len(MITRE_RULES) == 7)
check("All rule_ids present",         all(f"RULE_00{i}" in {r["rule_id"] for r in MITRE_RULES} for i in range(1,8)))
check("All technique_ids present",    all(t in MITRE_BY_TECHNIQUE for t in
    ["T1110","T1190","T1059","T1490","T1486","T1048","T1547"]))
check("All rules have llm_description", all(r.get("llm_description") for r in MITRE_RULES))
check("All rules have severity_weight", all(isinstance(r.get("severity_weight"), float) for r in MITRE_RULES))

# T1110: Brute Force
brute_force_window = []
for _ in range(6):
    brute_force_window.append({
        "source_type": "auth", "src_ip": "203.0.113.99",
        "action": "LOGIN_ATTEMPT", "status": "FAILURE",
        "payload_flags": [], "http_status": None, "bytes_sent": None,
        "parent_proc": None, "child_proc": None, "process_name": None,
        "file_path": None, "file_ops": None, "cpu_pct": None, "non_admin_user": None,
    })
success_evt = {**brute_force_window[0], "status": "SUCCESS"}
matched = match_rule(success_evt, recent_events=brute_force_window)
check("T1110 Brute Force fires after 6 fails",   any(r["technique_id"] == "T1110" for r in matched))

# T1190: Web Exploit
exploit_evt = {
    "source_type": "network", "src_ip": "203.0.113.99",
    "action": "HTTP_REQUEST", "status": "SUCCESS", "http_status": 200,
    "payload_flags": ["LFI"], "bytes_sent": 512, "uri_query": "/page?file=../etc/passwd",
    "parent_proc": None, "child_proc": None, "process_name": None,
    "file_path": None, "file_ops": None, "cpu_pct": None, "non_admin_user": None,
}
matched2 = match_rule(exploit_evt)
check("T1190 Web Exploit fires on LFI+200",      any(r["technique_id"] == "T1190" for r in matched2))

# T1059: Shell Spawn
shell_evt = {
    "source_type": "endpoint", "src_ip": "192.168.1.10",
    "action": "PROCESS_SPAWN", "status": None, "http_status": None,
    "payload_flags": [], "bytes_sent": None, "uri_query": None,
    "parent_proc": "nginx", "child_proc": "/bin/bash",
    "process_name": "/bin/bash", "file_path": None, "file_ops": None,
    "cpu_pct": None, "non_admin_user": True,
}
matched3 = match_rule(shell_evt)
check("T1059 Shell Spawn fires on nginx→/bin/bash", any(r["technique_id"] == "T1059" for r in matched3))

# T1490: Shadow Copy
shadow_evt = {
    "source_type": "endpoint", "src_ip": "192.168.1.10",
    "action": "delete shadows", "process_name": "vssadmin.exe",
    "status": None, "http_status": None, "payload_flags": [], "bytes_sent": None,
    "uri_query": None, "parent_proc": None, "child_proc": None,
    "file_path": None, "file_ops": None, "cpu_pct": None, "non_admin_user": False,
}
matched4 = match_rule(shadow_evt)
check("T1490 Shadow Copy fires on vssadmin",       any(r["technique_id"] == "T1490" for r in matched4))

# T1486: Ransomware
ransom_evt = {
    "source_type": "syslog", "src_ip": "192.168.1.10",
    "action": "BULK_FILE_OPERATION", "status": None, "http_status": None,
    "payload_flags": [], "bytes_sent": None, "uri_query": None,
    "parent_proc": None, "child_proc": None, "process_name": "cryptd",
    "file_path": None, "file_ops": 312, "cpu_pct": 87.3, "non_admin_user": None,
}
matched5 = match_rule(ransom_evt)
check("T1486 Ransomware fires on 312 ops + 87% CPU", any(r["technique_id"] == "T1486" for r in matched5))

# T1048: Exfiltration
exfil_evt = {
    "source_type": "network", "src_ip": "192.168.1.30", "dest_ip": "45.33.32.156",
    "action": "NETWORK_FLOW", "status": None, "http_status": None,
    "payload_flags": [], "bytes_sent": 75_000_000, "uri_query": None,
    "parent_proc": None, "child_proc": None, "process_name": None,
    "file_path": None, "file_ops": None, "cpu_pct": None, "non_admin_user": None,
}
matched6 = match_rule(exfil_evt)
check("T1048 Exfil fires on 75MB to external IP",  any(r["technique_id"] == "T1048" for r in matched6))

# T1547: Persistence
persist_evt = {
    "source_type": "syslog", "src_ip": "192.168.1.10",
    "action": "CRON_MODIFY", "status": None, "http_status": None,
    "payload_flags": [], "bytes_sent": None, "uri_query": None,
    "parent_proc": None, "child_proc": None, "process_name": "cron",
    "file_path": "/etc/cron.d/backdoor", "file_ops": 1, "cpu_pct": None,
    "non_admin_user": True,
}
matched7 = match_rule(persist_evt)
check("T1547 Persistence fires on /etc/cron.d non-admin", any(r["technique_id"] == "T1547" for r in matched7))

# No false positive on benign
benign = {
    "source_type": "network", "src_ip": "198.51.100.1", "dest_ip": "192.168.1.10",
    "action": "HTTP_REQUEST", "status": "SUCCESS", "http_status": 200,
    "payload_flags": [], "bytes_sent": 4200, "uri_query": "/index.html",
    "parent_proc": None, "child_proc": None, "process_name": None,
    "file_path": None, "file_ops": None, "cpu_pct": None, "non_admin_user": None,
}
matched_benign = match_rule(benign)
check("No MITRE hit on benign traffic",            len(matched_benign) == 0)


# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Integration: Normalized Event Schema ─────────────────────────────────")

REQUIRED_KEYS = [
    "event_id", "timestamp", "source_type", "src_ip", "dest_ip",
    "src_port", "dest_port", "user", "action", "status", "process_name",
    "parent_proc", "child_proc", "bytes_sent", "uri_query", "payload_flags",
    "file_path", "file_ops", "cpu_pct", "non_admin_user", "http_status", "raw",
]
all_parsers_output = [
    parse_auth(SSH_FAIL_LINE, "192.168.1.20"),
    parse_network(APACHE_LFI, "192.168.1.10"),
    parse_endpoint(SYSMON_PROC, "192.168.1.10"),
    parse_syslog(CRON_EDIT, "192.168.1.10"),
]
for evt in all_parsers_output:
    if evt:
        missing = [k for k in REQUIRED_KEYS if k not in evt]
        check(f"All 22 keys in {evt['source_type']} event", len(missing) == 0,
              detail=f"missing: {missing}" if missing else "")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Summary ──────────────────────────────────────────────────────────────")
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
total  = len(results)
print(f"\n  Total: {total}   ✅ Passed: {passed}   ❌ Failed: {failed}\n")

if failed > 0:
    print("  Failed tests:")
    for name, ok in results:
        if not ok:
            print(f"    ✗  {name}")
    sys.exit(1)
else:
    print("  All tests passed. Dev 1 pipeline is integration-ready. 🚀\n")