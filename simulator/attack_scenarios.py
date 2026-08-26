# =============================================================================
# normative/ingestor/simulator/attack_scenarios.py
# Scripted attack event sequences for the demo simulator.
# Each event is a partial Normalized Event Object — the simulator fills
# event_id and timestamp at publish time.
# =============================================================================

# ── IP constants (also hardcoded in Dev 4's ThreatMap) ───────────────────────
ATTACKER_IP   = "203.0.113.99"    # Beijing, China       (39.9042, 116.4074)
EXFIL_IP      = "45.33.32.156"    # Fremont, CA, USA     (37.5485, -121.9886)
INTERNAL_WEB  = "192.168.1.10"    # internal web server
INTERNAL_AUTH = "192.168.1.20"    # internal auth server
INTERNAL_DB   = "192.168.1.30"    # internal DB server


# ── Benign baseline events (played first for ~90 seconds) ────────────────────
# These train the IForest baseline and give the LSTM its "normal" window.
# All benign events now use ATTACKER_IP as source to fill the ML window quickly.

BENIGN_EVENTS = [
    # Normal web traffic from attacker IP to internal web
    {
        "source_type": "network", "src_ip": ATTACKER_IP, "dest_ip": INTERNAL_WEB,
        "src_port": 54321, "dest_port": 443, "action": "HTTP_REQUEST",
        "status": "SUCCESS", "http_status": 200, "bytes_sent": 4200,
        "uri_query": "/index.html", "payload_flags": [],
        "user": None, "process_name": None, "parent_proc": None,
        "child_proc": None, "file_path": None, "file_ops": None,
        "cpu_pct": None, "non_admin_user": None,
    },
    {
        "source_type": "network", "src_ip": ATTACKER_IP, "dest_ip": INTERNAL_WEB,
        "src_port": 55000, "dest_port": 443, "action": "HTTP_REQUEST",
        "status": "SUCCESS", "http_status": 200, "bytes_sent": 8100,
        "uri_query": "/api/products", "payload_flags": [],
        "user": None, "process_name": None, "parent_proc": None,
        "child_proc": None, "file_path": None, "file_ops": None,
        "cpu_pct": None, "non_admin_user": None,
    },
    # Normal login from attacker IP to auth server
    {
        "source_type": "auth", "src_ip": ATTACKER_IP, "dest_ip": INTERNAL_AUTH,
        "src_port": 0, "dest_port": 22, "action": "LOGIN_ATTEMPT",
        "status": "SUCCESS", "user": "devops", "process_name": None,
        "parent_proc": None, "child_proc": None, "bytes_sent": None,
        "uri_query": None, "payload_flags": [], "file_path": None,
        "file_ops": None, "cpu_pct": None, "non_admin_user": True,
        "http_status": None,
    },
    # Normal file write on internal web – keep src_ip = ATTACKER_IP for consistency
    {
        "source_type": "endpoint", "src_ip": ATTACKER_IP, "dest_ip": INTERNAL_WEB,
        "src_port": 0, "dest_port": 0, "action": "FILE_WRITE",
        "status": None, "user": "www-data", "process_name": "nginx",
        "parent_proc": "systemd", "child_proc": None, "bytes_sent": None,
        "uri_query": None, "payload_flags": [], "file_path": "/var/log/nginx/access.log",
        "file_ops": 3, "cpu_pct": 12.5, "non_admin_user": True, "http_status": None,
    },
]


# ── APT Attack Chain — 8 steps, ordered by delay_after_prev_sec ──────────────
# All attack events now use ATTACKER_IP as src_ip to fill the ML window on the
# same host that generated the benign baseline.

APT_SCENARIO = [

    # ── Step 1: T1110 Brute Force (12 failures then 1 success) ───────────────
    # Repeat this block 12x via simulator before sending the SUCCESS
    {
        "step": 1, "technique_id": "T1110", "delay_after_prev_sec": 0,
        "repeat": 12,
        "event": {
            "source_type": "auth", "src_ip": ATTACKER_IP, "dest_ip": INTERNAL_AUTH,
            "src_port": 44210, "dest_port": 22, "action": "LOGIN_ATTEMPT",
            "status": "FAILURE", "user": "root", "process_name": None,
            "parent_proc": None, "child_proc": None, "bytes_sent": None,
            "uri_query": None, "payload_flags": [], "file_path": None,
            "file_ops": None, "cpu_pct": None, "non_admin_user": None,
            "http_status": None,
        },
    },
    {
        "step": 1, "technique_id": "T1110", "delay_after_prev_sec": 2,
        "repeat": 1,
        "event": {
            "source_type": "auth", "src_ip": ATTACKER_IP, "dest_ip": INTERNAL_AUTH,
            "src_port": 44210, "dest_port": 22, "action": "LOGIN_ATTEMPT",
            "status": "SUCCESS", "user": "root", "process_name": None,
            "parent_proc": None, "child_proc": None, "bytes_sent": None,
            "uri_query": None, "payload_flags": [], "file_path": None,
            "file_ops": None, "cpu_pct": None, "non_admin_user": None,
            "http_status": None,
        },
    },

    # ── Step 2: T1190 Web Exploit — LFI via uri_query ────────────────────────
    {
        "step": 2, "technique_id": "T1190", "delay_after_prev_sec": 15,
        "repeat": 1,
        "event": {
            "source_type": "network", "src_ip": ATTACKER_IP, "dest_ip": INTERNAL_WEB,
            "src_port": 51234, "dest_port": 80, "action": "HTTP_REQUEST",
            "status": "SUCCESS", "user": None, "process_name": None,
            "parent_proc": None, "child_proc": None, "bytes_sent": 512,
            "uri_query": "/page?file=../../../../etc/passwd",
            "payload_flags": ["LFI", "TRAVERSAL"],
            "file_path": None, "file_ops": None, "cpu_pct": None,
            "non_admin_user": None, "http_status": 200,
        },
    },

    # ── Step 3: T1059 Shell Spawn — nginx forks /bin/bash ────────────────────
    {
        "step": 3, "technique_id": "T1059", "delay_after_prev_sec": 20,
        "repeat": 1,
        "event": {
            "source_type": "endpoint", "src_ip": ATTACKER_IP, "dest_ip": INTERNAL_WEB,
            "src_port": 0, "dest_port": 0, "action": "PROCESS_SPAWN",
            "status": None, "user": "www-data", "process_name": "/bin/bash",
            "parent_proc": "nginx", "child_proc": "/bin/bash",
            "bytes_sent": None, "uri_query": None, "payload_flags": [],
            "file_path": None, "file_ops": None, "cpu_pct": 8.3,
            "non_admin_user": True, "http_status": None,
        },
    },

    # ── Step 4: T1547 Persistence — cron modification by www-data ────────────
    {
        "step": 4, "technique_id": "T1547", "delay_after_prev_sec": 30,
        "repeat": 1,
        "event": {
            "source_type": "syslog", "src_ip": ATTACKER_IP, "dest_ip": INTERNAL_WEB,
            "src_port": 0, "dest_port": 0, "action": "CRON_MODIFY",
            "status": None, "user": "www-data", "process_name": "cron",
            "parent_proc": None, "child_proc": None, "bytes_sent": None,
            "uri_query": None, "payload_flags": [],
            "file_path": "/etc/cron.d/www-data",
            "file_ops": 1, "cpu_pct": None, "non_admin_user": True,
            "http_status": None,
        },
    },

    # ── Step 5: T1490 Shadow Copy Deletion ───────────────────────────────────
    {
        "step": 5, "technique_id": "T1490", "delay_after_prev_sec": 40,
        "repeat": 1,
        "event": {
            "source_type": "endpoint", "src_ip": ATTACKER_IP, "dest_ip": INTERNAL_WEB,
            "src_port": 0, "dest_port": 0, "action": "delete shadows",
            "status": None, "user": "root", "process_name": "vssadmin.exe",
            "parent_proc": "/bin/bash", "child_proc": None,
            "bytes_sent": None, "uri_query": None, "payload_flags": [],
            "file_path": None, "file_ops": None, "cpu_pct": 5.1,
            "non_admin_user": False, "http_status": None,
        },
    },

    # ── Step 6: T1486 Ransomware File Encryption ─────────────────────────────
    {
        "step": 6, "technique_id": "T1486", "delay_after_prev_sec": 45,
        "repeat": 1,
        "event": {
            "source_type": "syslog", "src_ip": ATTACKER_IP, "dest_ip": INTERNAL_WEB,
            "src_port": 0, "dest_port": 0, "action": "BULK_FILE_OPERATION",
            "status": None, "user": None, "process_name": "cryptd",
            "parent_proc": None, "child_proc": None,
            "bytes_sent": None, "uri_query": None, "payload_flags": [],
            "file_path": None, "file_ops": 312, "cpu_pct": 87.3,
            "non_admin_user": None, "http_status": None,
        },
    },

    # ── Step 7: T1048 Data Exfiltration — 75MB outbound ──────────────────────
    {
        "step": 7, "technique_id": "T1048", "delay_after_prev_sec": 60,
        "repeat": 1,
        "event": {
            "source_type": "network", "src_ip": ATTACKER_IP, "dest_ip": EXFIL_IP,
            "src_port": 43000, "dest_port": 443, "action": "NETWORK_FLOW",
            "status": None, "user": None, "process_name": None,
            "parent_proc": None, "child_proc": None,
            "bytes_sent": 75_000_000, "uri_query": None, "payload_flags": [],
            "file_path": None, "file_ops": None, "cpu_pct": None,
            "non_admin_user": None, "http_status": None,
        },
    },
]


# ── Rapid attack replay (speed 4x for tight demos < 3 min) ───────────────────
APT_SCENARIO_FAST = [
    {**step, "delay_after_prev_sec": max(1, step["delay_after_prev_sec"] // 4)}
    for step in APT_SCENARIO
]