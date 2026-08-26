# =============================================================================
# normative/ingestor/mitre_lookup.py
# MITRE ATT&CK Lookup Table — 7 rules covering the full attack chain
# Imported by: incident_assembler.py (Dev 3), feature_extractor.py (Dev 1)
# DO NOT rename keys — Dev 3 depends on exact field names
# =============================================================================

MITRE_RULES = [
    {
        "rule_id":          "RULE_001",
        "technique_id":     "T1110",
        "technique_name":   "Brute Force",
        "tactic":           "Credential Access",
        "source_types":     ["auth"],
        "match_logic":      "brute_force_then_success",
        "threshold_fails":  5,
        "window_seconds":   60,
        "llm_description":  (
            "Multiple failed login attempts followed by a successful authentication, "
            "indicating a password guessing or brute-force attack."
        ),
        "severity_weight":  0.50, # Medium
    },
    {
        "rule_id":          "RULE_002",
        "technique_id":     "T1190",
        "technique_name":   "Exploit Public-Facing Application",
        "tactic":           "Initial Access",
        "source_types":     ["network"],
        "match_logic":      "http_exploit_payload",
        "payload_flags":    ["LFI", "RCE", "SQLi", "SSRF", "XSS", "TRAVERSAL"],
        "required_status":  200,
        "llm_description":  (
            "Adversary exploited a web application vulnerability (e.g., LFI, RCE) "
            "to gain unauthorized access to the server."
        ),
        "severity_weight":  0.65, # High
    },
    {
        "rule_id":          "RULE_003",
        "technique_id":     "T1059",
        "technique_name":   "Command and Scripting Interpreter",
        "tactic":           "Execution",
        "source_types":     ["endpoint"],
        "match_logic":      "webserver_shell_spawn",
        "parent_procs":     ["nginx", "apache", "apache2", "httpd", "php-fpm", "tomcat"],
        "child_procs":      ["/bin/sh", "/bin/bash", "/bin/dash", "cmd.exe", "powershell.exe"],
        "llm_description":  (
            "Web server process spawned an interactive shell, indicating a successful "
            "remote code execution (RCE) or webshell deployment."
        ),
        "severity_weight":  0.80, # Critical
    },
    {
        "rule_id":          "RULE_004",
        "technique_id":     "T1490",
        "technique_name":   "Inhibit System Recovery",
        "tactic":           "Impact",
        "source_types":     ["endpoint"],
        "match_logic":      "shadow_copy_deletion",
        "process_names":    ["vssadmin.exe", "wbadmin.exe"],
        "action_patterns":  ["delete shadows", "delete catalog", "resize shadowstorage"],
        "llm_description":  (
            "Adversary attempted to delete volume shadow copies or system backups "
            "to prevent data recovery prior to ransomware execution."
        ),
        "severity_weight":  0.75, # High
    },
    {
        "rule_id":              "RULE_005",
        "technique_id":         "T1486",
        "technique_name":       "Data Encrypted for Impact",
        "tactic":               "Impact",
        "source_types":         ["syslog", "endpoint"],
        "match_logic":          "ransomware_file_pattern",
        "file_ops_threshold":   200,
        "cpu_pct_threshold":    70.0,
        "window_seconds":       30,
        "llm_description":      (
            "Automated, high-velocity file modification and encryption patterns "
            "consistent with an active ransomware deployment."
        ),
        "severity_weight":      1.00, # Critical
    },
    {
        "rule_id":                  "RULE_006",
        "technique_id":             "T1048",
        "technique_name":           "Exfiltration Over Alternative Protocol",
        "tactic":                   "Exfiltration",
        "source_types":             ["network"],
        "match_logic":              "large_outbound_transfer",
        "bytes_sent_threshold":     50_000_000,   # 50 MB
        "require_unknown_dest":     True,
        "llm_description":          (
            "Unusually large outbound data transfer to an unverified external "
            "destination, indicating a data breach or exfiltration event."
        ),
        "severity_weight":          0.85, # Critical
    },
    {
        "rule_id":              "RULE_007",
        "technique_id":         "T1547",
        "technique_name":       "Boot or Logon Autostart Execution",
        "tactic":               "Persistence",
        "source_types":         ["endpoint", "syslog"],
        "match_logic":          "persistence_mechanism",
        "watched_paths":        [
            "/etc/cron.d", "/etc/cron.daily", "/etc/cron.hourly",
            "/etc/cron.weekly", "/etc/rc.local", "/etc/init.d",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
            r"HKLM\Software\Microsoft\Windows\CurrentVersion\Run",
        ],
        "require_non_admin":    True,
        "llm_description":      (
            "Creation of persistent system tasks or registry modifications designed "
            "to maintain access across server reboots."
        ),
        "severity_weight":      0.45, # Medium
    },
]

# Fast lookup by technique_id
MITRE_BY_TECHNIQUE = {r["technique_id"]: r for r in MITRE_RULES}

# Fast lookup by rule_id
MITRE_BY_RULE = {r["rule_id"]: r for r in MITRE_RULES}

# Known internal IP ranges (used by RULE_006 unknown dest check)
KNOWN_INTERNAL_RANGES = [
    "10.", "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
    "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
    "172.30.", "172.31.", "192.168.", "127.",
]


def is_unknown_external(ip: str) -> bool:
    """Returns True if ip is NOT in any known internal range."""
    if not ip:
        return False
    return not any(ip.startswith(prefix) for prefix in KNOWN_INTERNAL_RANGES)


def match_rule(event: dict, recent_events: list = None) -> list:
    """
    Evaluate all MITRE rules against a normalized event (and optional
    recent event window for stateful rules).

    Returns: list of matched MITRE_RULES entries
    """
    recent_events = recent_events or []
    matched = []

    for rule in MITRE_RULES:
        # Source type filter
        if event.get("source_type") not in rule["source_types"]:
            continue

        logic = rule["match_logic"]

        # ── RULE_001: Brute Force ─────────────────────────────────────────
        if logic == "brute_force_then_success":
            if event.get("status") == "SUCCESS" and event.get("action") == "LOGIN_ATTEMPT":
                src = event.get("src_ip")
                fail_count = sum(
                    1 for e in recent_events
                    if e.get("src_ip") == src
                    and e.get("status") == "FAILURE"
                    and e.get("action") == "LOGIN_ATTEMPT"
                )
                if fail_count >= rule["threshold_fails"]:
                    matched.append(rule)

        # ── RULE_002: Web Exploit ─────────────────────────────────────────
        elif logic == "http_exploit_payload":
            flags = event.get("payload_flags", [])
            http_ok = event.get("http_status") == rule["required_status"]
            has_payload = any(f in rule["payload_flags"] for f in flags)
            if http_ok and has_payload:
                matched.append(rule)

        # ── RULE_003: Shell Spawn ─────────────────────────────────────────
        elif logic == "webserver_shell_spawn":
            parent = (event.get("parent_proc") or "").lower()
            child = (event.get("child_proc") or "").lower()
            parent_match = any(p in parent for p in rule["parent_procs"])
            child_match = any(c in child for c in rule["child_procs"])
            if parent_match and child_match:
                matched.append(rule)

        # ── RULE_004: Shadow Copy Deletion ────────────────────────────────
        elif logic == "shadow_copy_deletion":
            proc = (event.get("process_name") or "").lower()
            action = (event.get("action") or "").lower()
            proc_match = any(p.lower() in proc for p in rule["process_names"])
            action_match = any(pat in action for pat in rule["action_patterns"])
            if proc_match and action_match:
                matched.append(rule)

        # ── RULE_005: Ransomware File Pattern ─────────────────────────────
        elif logic == "ransomware_file_pattern":
            file_ops = event.get("file_ops") or 0
            cpu = event.get("cpu_pct") or 0.0
            if (file_ops >= rule["file_ops_threshold"]
                    and cpu >= rule["cpu_pct_threshold"]):
                matched.append(rule)

        # ── RULE_006: Large Outbound Transfer ─────────────────────────────
        elif logic == "large_outbound_transfer":
            sent = event.get("bytes_sent") or 0
            dest = event.get("dest_ip") or ""
            unknown = is_unknown_external(dest) if rule["require_unknown_dest"] else True
            if sent >= rule["bytes_sent_threshold"] and unknown:
                matched.append(rule)

        # ── RULE_007: Persistence Mechanism ───────────────────────────────
        elif logic == "persistence_mechanism":
            path = event.get("file_path") or ""
            non_admin = event.get("non_admin_user") or False
            path_match = any(p in path for p in rule["watched_paths"])
            if path_match and (not rule["require_non_admin"] or non_admin):
                matched.append(rule)

    return matched
