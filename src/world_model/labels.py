"""Map CIC-IDS attack labels → MITRE ATT&CK stages used by the SIH problem."""

from __future__ import annotations

# SIH requested stages, plus Impact (DoS/DDoS are first-class in CIC-IDS).
STAGE_NAMES = [
    "benign",
    "reconnaissance",
    "initial_access",
    "lateral_movement",
    "command_and_control",
    "exfiltration",
    "impact",
]

STAGE_ID = {name: i for i, name in enumerate(STAGE_NAMES)}

# Representative MITRE technique for each SIH stage (CIC has no exfil label).
STAGE_TECHNIQUE: dict[str, tuple[str, str]] = {
    "benign": ("", ""),
    "reconnaissance": ("T1046", "Network Service Discovery"),
    "initial_access": ("T1110", "Brute Force"),
    "lateral_movement": ("T1021", "Remote Services"),
    "command_and_control": ("T1071", "Application Layer Protocol"),
    "exfiltration": ("T1048", "Exfiltration Over Alternative Protocol"),
    "impact": ("T1498", "Network Denial of Service"),
}

# (stage, mitre_technique_id, mitre_technique_name)
_LABEL_TABLE: dict[str, tuple[str, str, str]] = {
    "benign": ("benign", "", ""),
    "ftp-bruteforce": ("initial_access", "T1110", "Brute Force"),
    "ssh-bruteforce": ("initial_access", "T1110", "Brute Force"),
    "brute force -web": ("initial_access", "T1110", "Brute Force"),
    "brute force -xss": ("initial_access", "T1190", "Exploit Public-Facing Application"),
    "sql injection": ("initial_access", "T1190", "Exploit Public-Facing Application"),
    "web attack - brute force": ("initial_access", "T1110", "Brute Force"),
    "web attack - xss": ("initial_access", "T1190", "Exploit Public-Facing Application"),
    "web attack - sql injection": ("initial_access", "T1190", "Exploit Public-Facing Application"),
    "infilteration": ("lateral_movement", "T1021", "Remote Services"),
    "infiltration": ("lateral_movement", "T1021", "Remote Services"),
    "bot": ("command_and_control", "T1071", "Application Layer Protocol"),
    "botnet": ("command_and_control", "T1071", "Application Layer Protocol"),
    "portscan": ("reconnaissance", "T1046", "Network Service Discovery"),
    "port scan": ("reconnaissance", "T1046", "Network Service Discovery"),
    "heartbleed": ("initial_access", "T1190", "Exploit Public-Facing Application"),
    "dos attacks-hulk": ("impact", "T1498", "Network Denial of Service"),
    "dos attacks-slowhttptest": ("impact", "T1498", "Network Denial of Service"),
    "dos attacks-goldeneye": ("impact", "T1498", "Network Denial of Service"),
    "dos attacks-slowloris": ("impact", "T1498", "Network Denial of Service"),
    "ddos attacks-loic-http": ("impact", "T1498", "Network Denial of Service"),
    "ddos attack-loic-udp": ("impact", "T1498", "Network Denial of Service"),
    "ddos attack-hoic": ("impact", "T1498", "Network Denial of Service"),
    "ddos": ("impact", "T1498", "Network Denial of Service"),
    "dos": ("impact", "T1498", "Network Denial of Service"),
}

# Severity used when a window contains mixed attack labels (higher wins).
_STAGE_SEVERITY = {
    "benign": 0,
    "reconnaissance": 1,
    "initial_access": 2,
    "command_and_control": 3,
    "lateral_movement": 4,
    "exfiltration": 5,
    "impact": 4,
}


def technique_for_stage(stage: str | int) -> tuple[str, str]:
    """Map a SIH stage name or stage_id to (technique_id, technique_name)."""
    if not isinstance(stage, str):
        idx = int(stage)
        if idx < 0 or idx >= len(STAGE_NAMES):
            raise KeyError(f"unknown stage_id {stage}")
        stage = STAGE_NAMES[idx]
    if stage not in STAGE_TECHNIQUE:
        raise KeyError(f"unknown stage {stage!r}")
    return STAGE_TECHNIQUE[stage]


def _norm_label(raw: str) -> str:
    text = str(raw or "").strip().casefold()
    text = text.replace("–", "-").replace("—", "-")
    return " ".join(text.split())


def map_label(raw: str) -> dict:
    key = _norm_label(raw)
    if key in _LABEL_TABLE:
        stage, tid, tname = _LABEL_TABLE[key]
    elif "infiltrat" in key:
        stage, tid, tname = _LABEL_TABLE["infiltration"]
    elif "portscan" in key.replace(" ", "") or "port scan" in key:
        stage, tid, tname = _LABEL_TABLE["portscan"]
    elif key in {"", "label", "nan", "none"}:
        stage, tid, tname = _LABEL_TABLE["benign"]
    elif "benign" in key:
        stage, tid, tname = _LABEL_TABLE["benign"]
    else:
        # Unknown attack string: treat as malicious initial access, do not drop.
        stage, tid, tname = ("initial_access", "T1190", "Exploit Public-Facing Application")

    return {
        "raw_label": str(raw),
        "is_attack": stage != "benign",
        "is_infiltration": stage == "lateral_movement",
        "stage": stage,
        "stage_id": STAGE_ID[stage],
        "technique_id": tid,
        "technique_name": tname,
        "severity": _STAGE_SEVERITY[stage],
    }


def pick_window_stage(mapped_rows: list[dict]) -> dict:
    """Worst-case / highest-severity label inside a time window."""
    if not mapped_rows:
        return map_label("Benign")
    return max(mapped_rows, key=lambda m: m["severity"])
