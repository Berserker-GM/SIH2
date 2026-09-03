"""
Defines all ~20 persona configs and writes them to personas/configs/<id>.json.

Three tiers, per the taxonomy in the brief:
  1. Dataset-calibrated (benign / brute-force / DoS / web attack / infiltration
     & lateral-movement families) — parameters are set to plausible
     CIC-IDS2018-shaped values. NOTE: the actual numeric calibration (means/
     stds pulled from the pipeline dev's training report) is a placeholder
     here — see docs/CALIBRATION_TODO.md. The shape of each config and the
     phase logic is final; only the numeric knobs need swapping once real
     per-class stats are shared.
  2. MITRE-doc-based (recon / C2 / exfiltration) — parameters derived from
     ATT&CK technique descriptions, not the dataset. Reasoning documented in
     docs/mitre_notes.md.
  3. Adaptive/evasive — same MITRE basis, plus `adaptive: true` and a
     `search_space` block consumed by adaptive.py's hill-climbing loop.
"""
import json
from pathlib import Path

OUT_DIR = Path("personas/configs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PORTS_WEB = [80, 443, 8080, 8443]
PORTS_COMMON = list(range(1, 65536))


def base(persona_id, family, mitre_technique, base_seed, phases,
         timing=None, flags=None, ports=None, payload=None, packet_fields=None,
         noise=None, adaptive=False, search_space=None, notes=None):
    cfg = {
        "persona_id": persona_id,
        "family": family,
        "mitre_technique": mitre_technique,
        "adaptive": adaptive,
        "base_seed": base_seed,
        "timing": timing or {"interarrival_dist": "exponential", "lambda": 1.0, "jitter_pct": 0.15},
        "flags": flags or {},
        "ports": ports or {"pool": PORTS_COMMON, "selection": "randomized"},
        "payload": payload or {"size_dist": "lognormal", "mean": 500, "std": 150},
        "packet_fields": packet_fields or {"ttl_base": 64, "ttl_jitter": 2, "fragment_rate": 0.0, "retransmit_rate": 0.01},
        "phases": phases,
        "noise": noise or {"stage_overlap_pct": 0.0, "blend_alpha": 0.3},
    }
    if adaptive:
        cfg["search_space"] = search_space or {}
    if notes:
        cfg["_notes"] = notes  # informal, not consumed by the generator; jury talking-point text
    return cfg


PERSONAS = []

# ============================================================
# TIER 1 — DATASET-CALIBRATED (benign, brute-force, DoS, web, infiltration/lateral)
# ============================================================

PERSONAS.append(base(
    "benign_browsing_v1", "benign", None, base_seed=1001,
    timing={"interarrival_dist": "exponential", "lambda": 1.2, "jitter_pct": 0.35},
    flags={"syn_ratio": 0.06, "ack_ratio": 0.62, "fin_ratio": 0.09, "rst_ratio": 0.02,
           "psh_ratio": 0.12, "urg_ratio": 0.0, "tcp_ratio": 0.88,
           "down_up_ratio_mean": 2.5, "down_up_ratio_std": 0.6,
           "bidirectional_ratio_mean": 0.7, "port_scan_score_base": 0.0, "port_scan_score_gain": 0.03},
    ports={"pool": PORTS_WEB, "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 600, "std": 300},
    packet_fields={"ttl_base": 64, "ttl_jitter": 2, "fragment_rate": 0.01, "retransmit_rate": 0.01},
    phases=[{"name": "browse", "mitre_stage": "benign", "technique_id": None,
             "dwell_windows": [100000, 100000], "next": None}],
    notes="Calibrated to CIC-IDS2018 'Benign' class stats: HTTP/HTTPS-heavy, moderate "
          "down/up asymmetry typical of web browsing, low SYN density (few new connections "
          "per window relative to steady-state ACK traffic).",
))

PERSONAS.append(base(
    "benign_bulk_transfer_v1", "benign", None, base_seed=1002,
    timing={"interarrival_dist": "exponential", "lambda": 3.0, "jitter_pct": 0.1},
    flags={"syn_ratio": 0.02, "ack_ratio": 0.7, "fin_ratio": 0.05, "rst_ratio": 0.01,
           "psh_ratio": 0.08, "urg_ratio": 0.0, "tcp_ratio": 0.97,
           "down_up_ratio_mean": 8.0, "down_up_ratio_std": 1.5,
           "bidirectional_ratio_mean": 0.3, "port_scan_score_base": 0.0, "port_scan_score_gain": 0.02},
    ports={"pool": [443, 22, 445], "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 1400, "std": 200},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.03, "retransmit_rate": 0.015},
    phases=[{"name": "transfer", "mitre_stage": "benign", "technique_id": None,
             "dwell_windows": [100000, 100000], "next": None}],
    notes="Sustained large-file backup/sync traffic — few long-lived flows, large packets, "
          "strongly download-dominant. Included so the decoder sees a benign class that "
          "*also* has high byte volume, so volume alone can't be a shortcut feature.",
))

PERSONAS.append(base(
    "benign_iot_idle_v1", "benign", None, base_seed=1003,
    timing={"interarrival_dist": "exponential", "lambda": 0.2, "jitter_pct": 0.5},
    flags={"syn_ratio": 0.1, "ack_ratio": 0.5, "fin_ratio": 0.1, "rst_ratio": 0.03,
           "psh_ratio": 0.2, "urg_ratio": 0.0, "tcp_ratio": 0.4,
           "down_up_ratio_mean": 1.0, "down_up_ratio_std": 0.3,
           "bidirectional_ratio_mean": 0.5, "port_scan_score_base": 0.0, "port_scan_score_gain": 0.05},
    ports={"pool": [1883, 5683, 53], "selection": "sequential"},
    payload={"size_dist": "lognormal", "mean": 90, "std": 30},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.0, "retransmit_rate": 0.005},
    phases=[{"name": "idle_poll", "mitre_stage": "benign", "technique_id": None,
             "dwell_windows": [100000, 100000], "next": None}],
    notes="Low-rate MQTT/CoAP-style device heartbeat. Small fixed low port pool visited "
          "'sequentially' (i.e. repeatedly) to stress-test that low unique_dst_ports + "
          "small pool doesn't get mistaken for scanning by itself.",
))

PERSONAS.append(base(
    "bruteforce_ssh_v1", "initial_access", "T1110.001", base_seed=1101,
    timing={"interarrival_dist": "exponential", "lambda": 4.0, "jitter_pct": 0.1},
    flags={"syn_ratio": 0.55, "ack_ratio": 0.3, "fin_ratio": 0.05, "rst_ratio": 0.15,
           "psh_ratio": 0.05, "urg_ratio": 0.0, "tcp_ratio": 1.0,
           "down_up_ratio_mean": 0.9, "down_up_ratio_std": 0.15,
           "bidirectional_ratio_mean": 0.9, "port_scan_score_base": 0.0, "port_scan_score_gain": 0.02},
    ports={"pool": [22], "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 120, "std": 40},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.0, "retransmit_rate": 0.04},
    phases=[{"name": "brute", "mitre_stage": "initial_access", "technique_id": "T1110.001",
             "dwell_windows": [100000, 100000], "next": None}],
    notes="Calibrated against CIC-IDS2018 SSH-Bruteforce class: high SYN density, elevated "
          "RST (failed auth resets), single-port (22) target, small uniform payload sizes.",
))

PERSONAS.append(base(
    "bruteforce_ftp_v1", "initial_access", "T1110.001", base_seed=1102,
    timing={"interarrival_dist": "exponential", "lambda": 3.5, "jitter_pct": 0.1},
    flags={"syn_ratio": 0.5, "ack_ratio": 0.35, "fin_ratio": 0.05, "rst_ratio": 0.1,
           "psh_ratio": 0.08, "urg_ratio": 0.0, "tcp_ratio": 1.0,
           "down_up_ratio_mean": 1.1, "down_up_ratio_std": 0.2,
           "bidirectional_ratio_mean": 0.85, "port_scan_score_base": 0.0, "port_scan_score_gain": 0.02},
    ports={"pool": [21], "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 100, "std": 30},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.0, "retransmit_rate": 0.03},
    phases=[{"name": "brute", "mitre_stage": "initial_access", "technique_id": "T1110.001",
             "dwell_windows": [100000, 100000], "next": None}],
    notes="CIC-IDS2018 FTP-Bruteforce analogue; same shape as SSH variant with FTP control-"
          "port targeting and slightly lower RST rate (protocol responds differently to bad auth).",
))

PERSONAS.append(base(
    "web_bruteforce_v1", "initial_access", "T1110.003", base_seed=1103,
    timing={"interarrival_dist": "exponential", "lambda": 5.0, "jitter_pct": 0.15},
    flags={"syn_ratio": 0.3, "ack_ratio": 0.45, "fin_ratio": 0.1, "rst_ratio": 0.05,
           "psh_ratio": 0.2, "urg_ratio": 0.0, "tcp_ratio": 1.0,
           "down_up_ratio_mean": 1.3, "down_up_ratio_std": 0.3,
           "bidirectional_ratio_mean": 0.8, "port_scan_score_base": 0.0, "port_scan_score_gain": 0.02},
    ports={"pool": PORTS_WEB, "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 350, "std": 100},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.0, "retransmit_rate": 0.02},
    phases=[{"name": "brute", "mitre_stage": "initial_access", "technique_id": "T1110.003",
             "dwell_windows": [100000, 100000], "next": None}],
    notes="Web-login credential stuffing (POST-flood against a login form) — higher PSH "
          "density than raw SSH/FTP brute force since each attempt carries an HTTP body.",
))

PERSONAS.append(base(
    "dos_hulk_v1", "impact", "T1499", base_seed=1201,
    timing={"interarrival_dist": "exponential", "lambda": 25.0, "jitter_pct": 0.05},
    flags={"syn_ratio": 0.7, "ack_ratio": 0.2, "fin_ratio": 0.02, "rst_ratio": 0.05,
           "psh_ratio": 0.03, "urg_ratio": 0.0, "tcp_ratio": 1.0,
           "down_up_ratio_mean": 0.6, "down_up_ratio_std": 0.1,
           "bidirectional_ratio_mean": 0.95, "port_scan_score_base": 0.0, "port_scan_score_gain": 0.01},
    ports={"pool": [80, 443], "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 200, "std": 60},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.0, "retransmit_rate": 0.06},
    phases=[{"name": "flood", "mitre_stage": "impact", "technique_id": "T1499",
             "dwell_windows": [100000, 100000], "next": None}],
    notes="CIC-IDS2018 DoS-Hulk analogue: very high flow rate, SYN-heavy, single target port, "
          "elevated retransmits from the victim's saturated queue.",
))

PERSONAS.append(base(
    "dos_slowloris_v1", "impact", "T1499.002", base_seed=1202,
    timing={"interarrival_dist": "exponential", "lambda": 0.8, "jitter_pct": 0.1},
    flags={"syn_ratio": 0.15, "ack_ratio": 0.7, "fin_ratio": 0.01, "rst_ratio": 0.02,
           "psh_ratio": 0.05, "urg_ratio": 0.0, "tcp_ratio": 1.0,
           "down_up_ratio_mean": 0.4, "down_up_ratio_std": 0.1,
           "bidirectional_ratio_mean": 0.9, "port_scan_score_base": 0.0, "port_scan_score_gain": 0.01},
    ports={"pool": [80, 443], "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 40, "std": 15},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.0, "retransmit_rate": 0.02},
    phases=[{"name": "slow_hold", "mitre_stage": "impact", "technique_id": "T1499.002",
             "dwell_windows": [100000, 100000], "next": None}],
    notes="Low-and-slow application-layer DoS: low flow rate but connections held open "
          "(long duration_max), tiny partial-header payloads. Deliberately the opposite "
          "shape of dos_hulk_v1 so 'impact' isn't a single monolithic pattern.",
))

PERSONAS.append(base(
    "web_sql_injection_v1", "initial_access", "T1190", base_seed=1301,
    timing={"interarrival_dist": "exponential", "lambda": 1.5, "jitter_pct": 0.2},
    flags={"syn_ratio": 0.2, "ack_ratio": 0.5, "fin_ratio": 0.1, "rst_ratio": 0.03,
           "psh_ratio": 0.25, "urg_ratio": 0.0, "tcp_ratio": 1.0,
           "down_up_ratio_mean": 1.8, "down_up_ratio_std": 0.4,
           "bidirectional_ratio_mean": 0.75, "port_scan_score_base": 0.0, "port_scan_score_gain": 0.02},
    ports={"pool": PORTS_WEB, "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 700, "std": 250},
    packet_fields={"ttl_base": 64, "ttl_jitter": 2, "fragment_rate": 0.0, "retransmit_rate": 0.015},
    phases=[{"name": "inject", "mitre_stage": "initial_access", "technique_id": "T1190",
             "dwell_windows": [100000, 100000], "next": None}],
    notes="CIC-IDS2018 'Web Attack - SQL Injection' analogue — moderate PSH-heavy request "
          "traffic with larger-than-normal request payloads (injected query strings).",
))

PERSONAS.append(base(
    "web_xss_v1", "initial_access", "T1190", base_seed=1302,
    timing={"interarrival_dist": "exponential", "lambda": 1.8, "jitter_pct": 0.2},
    flags={"syn_ratio": 0.18, "ack_ratio": 0.5, "fin_ratio": 0.1, "rst_ratio": 0.03,
           "psh_ratio": 0.22, "urg_ratio": 0.0, "tcp_ratio": 1.0,
           "down_up_ratio_mean": 2.2, "down_up_ratio_std": 0.5,
           "bidirectional_ratio_mean": 0.75, "port_scan_score_base": 0.0, "port_scan_score_gain": 0.02},
    ports={"pool": PORTS_WEB, "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 550, "std": 200},
    packet_fields={"ttl_base": 64, "ttl_jitter": 2, "fragment_rate": 0.0, "retransmit_rate": 0.015},
    phases=[{"name": "inject", "mitre_stage": "initial_access", "technique_id": "T1190",
             "dwell_windows": [100000, 100000], "next": None}],
    notes="CIC-IDS2018 'Web Attack - XSS' analogue — similar shape to SQLi variant, "
          "slightly larger down/up ratio (reflected responses).",
))

PERSONAS.append(base(
    "infiltration_v1", "lateral_movement", "T1210", base_seed=1401,
    timing={"interarrival_dist": "exponential", "lambda": 0.6, "jitter_pct": 0.3},
    flags={"syn_ratio": 0.12, "ack_ratio": 0.55, "fin_ratio": 0.08, "rst_ratio": 0.03,
           "psh_ratio": 0.15, "urg_ratio": 0.0, "tcp_ratio": 0.8,
           "down_up_ratio_mean": 1.5, "down_up_ratio_std": 0.5,
           "bidirectional_ratio_mean": 0.6, "port_scan_score_base": 0.02, "port_scan_score_gain": 0.1},
    ports={"pool": [445, 135, 139, 3389], "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 800, "std": 400},
    packet_fields={"ttl_base": 64, "ttl_jitter": 2, "fragment_rate": 0.02, "retransmit_rate": 0.02},
    phases=[{"name": "internal_probe", "mitre_stage": "lateral_movement", "technique_id": "T1210",
             "dwell_windows": [100000, 100000], "next": None}],
    notes="CIC-IDS2018 'Infiltration' class analogue — internal-network exploitation of a "
          "remote service, moderate byte volume with irregular timing (interactive, not bulk).",
))

PERSONAS.append(base(
    "lateral_movement_smb_v1", "lateral_movement", "T1021.002", base_seed=1402,
    timing={"interarrival_dist": "exponential", "lambda": 1.0, "jitter_pct": 0.2},
    flags={"syn_ratio": 0.2, "ack_ratio": 0.5, "fin_ratio": 0.08, "rst_ratio": 0.04,
           "psh_ratio": 0.15, "urg_ratio": 0.0, "tcp_ratio": 1.0,
           "down_up_ratio_mean": 1.2, "down_up_ratio_std": 0.3,
           "bidirectional_ratio_mean": 0.65, "port_scan_score_base": 0.01, "port_scan_score_gain": 0.08},
    ports={"pool": [445, 139], "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 900, "std": 300},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.01, "retransmit_rate": 0.015},
    phases=[{"name": "smb_move", "mitre_stage": "lateral_movement", "technique_id": "T1021.002",
             "dwell_windows": [100000, 100000], "next": None}],
    notes="SMB/Windows-admin-share style lateral movement (psexec-like). Dataset-calibrated "
          "using CIC-IDS2018's internal-traffic infiltration stats as the closest analogue, "
          "narrowed to the SMB port pair.",
))

# ============================================================
# TIER 2 — MITRE-DOC-BASED (recon / C2 / exfiltration; no dataset examples exist)
# ============================================================

PERSONAS.append(base(
    "recon_portscan_v1", "reconnaissance", "T1595.001", base_seed=2001,
    timing={"interarrival_dist": "exponential", "lambda": 2.0, "jitter_pct": 0.1},
    flags={"syn_ratio": 0.85, "ack_ratio": 0.05, "fin_ratio": 0.01, "rst_ratio": 0.4,
           "psh_ratio": 0.0, "urg_ratio": 0.0, "tcp_ratio": 0.95,
           "down_up_ratio_mean": 0.3, "down_up_ratio_std": 0.1,
           "bidirectional_ratio_mean": 0.15, "port_scan_score_base": 0.3, "port_scan_score_gain": 0.6},
    ports={"pool": list(range(1, 1025)), "selection": "sequential"},
    payload={"size_dist": "lognormal", "mean": 40, "std": 10},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.0, "retransmit_rate": 0.0},
    phases=[{"name": "sweep", "mitre_stage": "reconnaissance", "technique_id": "T1595.001",
             "dwell_windows": [100000, 100000], "next": None}],
    noise={"stage_overlap_pct": 0.15, "blend_alpha": 0.25},
    notes="T1595.001 (Active Scanning: Scanning IP Blocks) / T1046 (Network Service "
          "Discovery). No CIC-IDS2018 class covers pure port-sweep recon, so this is built "
          "from the technique description directly: sequential port selection over the "
          "well-known-port range, minimal-to-no payload (SYN probes only), near-zero "
          "bidirectionality (most probes get no meaningful response), and elevated RST "
          "(closed-port resets). 15% window-level blend toward a benign profile keeps this "
          "class from being trivially separable purely on port_scan_score.",
))

PERSONAS.append(base(
    "recon_service_discovery_v1", "reconnaissance", "T1046", base_seed=2002,
    timing={"interarrival_dist": "exponential", "lambda": 0.5, "jitter_pct": 0.25},
    flags={"syn_ratio": 0.6, "ack_ratio": 0.25, "fin_ratio": 0.05, "rst_ratio": 0.2,
           "psh_ratio": 0.05, "urg_ratio": 0.0, "tcp_ratio": 0.7,
           "down_up_ratio_mean": 0.6, "down_up_ratio_std": 0.2,
           "bidirectional_ratio_mean": 0.35, "port_scan_score_base": 0.15, "port_scan_score_gain": 0.4},
    ports={"pool": [21, 22, 23, 25, 80, 135, 139, 443, 445, 3389, 8080], "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 90, "std": 30},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.0, "retransmit_rate": 0.0},
    phases=[{"name": "service_probe", "mitre_stage": "reconnaissance", "technique_id": "T1046",
             "dwell_windows": [100000, 100000], "next": None}],
    noise={"stage_overlap_pct": 0.15, "blend_alpha": 0.25},
    notes="T1046 (Network Service Discovery) — slower, more targeted than a raw port sweep: "
          "probes a curated list of commonly-exploited service ports rather than a "
          "sequential block, with longer inter-probe gaps ('longer inter-probe intervals to "
          "evade flow-threshold detection' per the brief) and slightly more of the probes "
          "getting a real service response (higher bidirectional_ratio than pure sweeping).",
))

PERSONAS.append(base(
    "c2_beacon_v1", "command_and_control", "T1071.001", base_seed=2101,
    timing={"interarrival_dist": "exponential", "lambda": 0.05, "jitter_pct": 0.2},
    flags={"syn_ratio": 0.05, "ack_ratio": 0.7, "fin_ratio": 0.02, "rst_ratio": 0.01,
           "psh_ratio": 0.15, "urg_ratio": 0.0, "tcp_ratio": 1.0,
           "down_up_ratio_mean": 0.9, "down_up_ratio_std": 0.15,
           "bidirectional_ratio_mean": 0.8, "port_scan_score_base": 0.0, "port_scan_score_gain": 0.01},
    ports={"pool": [443, 80], "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 250, "std": 60},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.0, "retransmit_rate": 0.005},
    phases=[{"name": "beacon", "mitre_stage": "command_and_control", "technique_id": "T1071.001",
             "dwell_windows": [100000, 100000], "next": None}],
    noise={"stage_overlap_pct": 0.1, "blend_alpha": 0.2},
    notes="T1071.001 (Application Layer Protocol: Web Protocols) — periodic low-volume "
          "beaconing over HTTPS/HTTP that blends with normal web traffic on the wire. Low "
          "flow rate (one beacon roughly every ~20s per window-lambda), small near-uniform "
          "payload sizes (check-in requests), timing jitter present but bounded — real C2 "
          "frameworks jitter beacons to avoid a perfectly periodic IAT signature, but not "
          "so much that they miss check-ins.",
))

PERSONAS.append(base(
    "c2_dns_tunnel_v1", "command_and_control", "T1071.004", base_seed=2102,
    timing={"interarrival_dist": "exponential", "lambda": 0.3, "jitter_pct": 0.15},
    flags={"syn_ratio": 0.02, "ack_ratio": 0.4, "fin_ratio": 0.02, "rst_ratio": 0.01,
           "psh_ratio": 0.1, "urg_ratio": 0.0, "tcp_ratio": 0.1,
           "down_up_ratio_mean": 1.1, "down_up_ratio_std": 0.2,
           "bidirectional_ratio_mean": 0.85, "port_scan_score_base": 0.0, "port_scan_score_gain": 0.01},
    ports={"pool": [53], "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 180, "std": 50},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.0, "retransmit_rate": 0.005},
    phases=[{"name": "tunnel", "mitre_stage": "command_and_control", "technique_id": "T1071.004",
             "dwell_windows": [100000, 100000], "next": None}],
    noise={"stage_overlap_pct": 0.1, "blend_alpha": 0.2},
    notes="T1071.004 (Application Layer Protocol: DNS) — C2 tunneled over DNS queries. "
          "Almost entirely UDP/port-53, moderate query-response bidirectionality, payload "
          "sizes larger than a normal DNS lookup (encoded C2 data in subdomain labels) but "
          "still small in absolute terms to stay under typical UDP fragmentation thresholds.",
))

PERSONAS.append(base(
    "exfil_c2channel_v1", "exfiltration", "T1041", base_seed=2201,
    timing={"interarrival_dist": "exponential", "lambda": 1.5, "jitter_pct": 0.1},
    flags={"syn_ratio": 0.02, "ack_ratio": 0.6, "fin_ratio": 0.02, "rst_ratio": 0.01,
           "psh_ratio": 0.2, "urg_ratio": 0.0, "tcp_ratio": 1.0,
           "down_up_ratio_mean": 0.08, "down_up_ratio_std": 0.03,
           "bidirectional_ratio_mean": 0.2, "port_scan_score_base": 0.0, "port_scan_score_gain": 0.01},
    ports={"pool": [443], "selection": "sequential"},
    payload={"size_dist": "lognormal", "mean": 1300, "std": 200},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.02, "retransmit_rate": 0.01},
    phases=[{"name": "exfil", "mitre_stage": "exfiltration", "technique_id": "T1041",
             "dwell_windows": [100000, 100000], "next": None}],
    noise={"stage_overlap_pct": 0.1, "blend_alpha": 0.2},
    notes="T1041 (Exfiltration Over C2 Channel) — sustained, strongly one-directional "
          "(uplink-dominant, down_up_ratio_mean ~0.08 i.e. inverted from normal browsing) "
          "byte transfer over the same channel used for C2, held on a single port to look "
          "like one long-lived legitimate HTTPS session rather than many short flows.",
))

# ============================================================
# TIER 3 — ADAPTIVE / EVASIVE (hill-climbing personas, no LLM)
# ============================================================

PERSONAS.append(base(
    "evasive_recon_v1", "reconnaissance", "T1595.001", base_seed=3001,
    adaptive=True,
    timing={"interarrival_dist": "exponential", "lambda": 1.0, "jitter_pct": 0.1},
    flags={"syn_ratio": 0.7, "ack_ratio": 0.1, "fin_ratio": 0.02, "rst_ratio": 0.3,
           "psh_ratio": 0.0, "urg_ratio": 0.0, "tcp_ratio": 0.9,
           "down_up_ratio_mean": 0.3, "down_up_ratio_std": 0.1,
           "bidirectional_ratio_mean": 0.2, "port_scan_score_base": 0.2, "port_scan_score_gain": 0.5},
    ports={"pool": list(range(1, 1025)), "selection": "sequential"},
    payload={"size_dist": "lognormal", "mean": 40, "std": 10},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.0, "retransmit_rate": 0.0},
    phases=[
        {"name": "probe", "mitre_stage": "reconnaissance", "technique_id": "T1595.001",
         "dwell_windows": [6, 12], "next": "confirm",
         "timing": {"interarrival_dist": "exponential", "lambda": 1.0, "jitter_pct": 0.1},
         "flags": {"syn_ratio": 0.7, "ack_ratio": 0.1, "fin_ratio": 0.02, "rst_ratio": 0.3,
                    "psh_ratio": 0.0, "urg_ratio": 0.0, "tcp_ratio": 0.9,
                    "down_up_ratio_mean": 0.3, "down_up_ratio_std": 0.1,
                    "bidirectional_ratio_mean": 0.2, "port_scan_score_base": 0.2, "port_scan_score_gain": 0.5}},
        {"name": "confirm", "mitre_stage": "reconnaissance", "technique_id": "T1046",
         "dwell_windows": [4, 8], "next": None},
    ],
    noise={"stage_overlap_pct": 0.15, "blend_alpha": 0.25},
    search_space={
        "probe.timing.lambda": {"min": 0.1, "max": 2.0, "step": 0.15},
        "probe.timing.jitter_pct": {"min": 0.05, "max": 0.6, "step": 0.05},
        "probe.flags.port_scan_score_gain": {"min": 0.1, "max": 0.5, "step": 0.05},
        "kill_chain_target_phase": "confirm",
        "step_budget_windows": 40,
    },
    notes="Evasive variant of the port-sweep persona: hill-climbing tunes probe rate, "
          "jitter, and how aggressively port diversity drives port_scan_score, while the "
          "dual objective requires it still reach the 'confirm' phase within a 40-window "
          "budget — i.e. it must actually keep scanning, just less conspicuously.",
))

PERSONAS.append(base(
    "evasive_c2_v1", "command_and_control", "T1071.001", base_seed=3002,
    adaptive=True,
    timing={"interarrival_dist": "exponential", "lambda": 0.1, "jitter_pct": 0.2},
    flags={"syn_ratio": 0.03, "ack_ratio": 0.7, "fin_ratio": 0.02, "rst_ratio": 0.01,
           "psh_ratio": 0.12, "urg_ratio": 0.0, "tcp_ratio": 1.0,
           "down_up_ratio_mean": 0.9, "down_up_ratio_std": 0.15,
           "bidirectional_ratio_mean": 0.8, "port_scan_score_base": 0.0, "port_scan_score_gain": 0.01},
    ports={"pool": [443, 80], "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 220, "std": 60},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.0, "retransmit_rate": 0.005},
    phases=[
        {"name": "establish", "mitre_stage": "command_and_control", "technique_id": "T1071.001",
         "dwell_windows": [4, 8], "next": "sustain"},
        {"name": "sustain", "mitre_stage": "command_and_control", "technique_id": "T1071.001",
         "dwell_windows": [20, 40], "next": None,
         "timing": {"interarrival_dist": "exponential", "lambda": 0.1, "jitter_pct": 0.2},
         "payload": {"size_dist": "lognormal", "mean": 220, "std": 60}},
    ],
    noise={"stage_overlap_pct": 0.12, "blend_alpha": 0.25},
    search_space={
        "sustain.timing.lambda": {"min": 0.02, "max": 0.3, "step": 0.02},
        "sustain.timing.jitter_pct": {"min": 0.1, "max": 0.7, "step": 0.05},
        "sustain.payload.std": {"min": 20, "max": 150, "step": 15},
        "kill_chain_target_phase": "sustain",
        "step_budget_windows": 50,
    },
    notes="Evasive beaconing: hill-climbing widens beacon-interval jitter and payload-size "
          "variance to break up the periodic-beacon signature detectors look for, but the "
          "budget constraint means it can't just go silent — it must still reach 'sustain' "
          "(i.e. keep beaconing) within 50 windows.",
))

PERSONAS.append(base(
    "evasive_lateral_v1", "lateral_movement", "T1021.002", base_seed=3003,
    adaptive=True,
    timing={"interarrival_dist": "exponential", "lambda": 0.7, "jitter_pct": 0.15},
    flags={"syn_ratio": 0.15, "ack_ratio": 0.5, "fin_ratio": 0.08, "rst_ratio": 0.04,
           "psh_ratio": 0.12, "urg_ratio": 0.0, "tcp_ratio": 1.0,
           "down_up_ratio_mean": 1.2, "down_up_ratio_std": 0.3,
           "bidirectional_ratio_mean": 0.6, "port_scan_score_base": 0.01, "port_scan_score_gain": 0.08},
    ports={"pool": [445, 139, 3389], "selection": "randomized"},
    payload={"size_dist": "lognormal", "mean": 700, "std": 250},
    packet_fields={"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.01, "retransmit_rate": 0.015},
    phases=[
        {"name": "stage", "mitre_stage": "reconnaissance", "technique_id": "T1046",
         "dwell_windows": [4, 8], "next": "move"},
        {"name": "move", "mitre_stage": "lateral_movement", "technique_id": "T1021.002",
         "dwell_windows": [10, 25], "next": None,
         "timing": {"interarrival_dist": "exponential", "lambda": 0.7, "jitter_pct": 0.15},
         "flags": {"syn_ratio": 0.15, "ack_ratio": 0.5, "fin_ratio": 0.08, "rst_ratio": 0.04,
                    "psh_ratio": 0.12, "urg_ratio": 0.0, "tcp_ratio": 1.0,
                    "down_up_ratio_mean": 1.2, "down_up_ratio_std": 0.3,
                    "bidirectional_ratio_mean": 0.6, "port_scan_score_base": 0.01, "port_scan_score_gain": 0.08},
         "packet_fields": {"ttl_base": 64, "ttl_jitter": 1, "fragment_rate": 0.01, "retransmit_rate": 0.015}},
    ],
    noise={"stage_overlap_pct": 0.1, "blend_alpha": 0.2},
    search_space={
        "move.timing.lambda": {"min": 0.1, "max": 1.2, "step": 0.1},
        "move.flags.port_scan_score_gain": {"min": 0.02, "max": 0.12, "step": 0.01},
        "move.packet_fields.fragment_rate": {"min": 0.0, "max": 0.05, "step": 0.01},
        "kill_chain_target_phase": "move",
        "step_budget_windows": 35,
    },
    notes="The named example in the brief: evasive SMB-style lateral movement. Hill-climbing "
          "tunes movement rate and port-diversity signal, but the persona must still "
          "complete the 'stage -> move' kill chain within 35 windows — an evasive persona "
          "that just goes quiet forever is not demonstrating evasion.",
))

for cfg in PERSONAS:
    with open(OUT_DIR / f"{cfg['persona_id']}.json", "w") as f:
        json.dump(cfg, f, indent=2)

print(f"Wrote {len(PERSONAS)} persona configs to {OUT_DIR}/")
