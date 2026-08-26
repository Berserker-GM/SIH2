# =============================================================================
# normative/ingestor/parsers/auth_parser.py
# Parses SSH / RDP / PAM authentication log lines into Normalized Event Objects
# Handles: /var/log/auth.log, /var/log/secure, Windows Security Event logs (JSON)
# =============================================================================

import re
import uuid
from datetime import datetime, timezone
from typing import Optional


# ── Regex patterns for common auth log formats ───────────────────────────────

# sshd: Failed password for root from 203.0.113.99 port 22 ssh2
_SSH_FAIL = re.compile(
    r"sshd.*?(?P<status>Failed|Invalid user|error: maximum authentication)"
    r".*?(?:for(?: invalid user)? (?P<user>\S+))?"
    r".*?from (?P<src_ip>[\d\.]+) port (?P<src_port>\d+)",
    re.IGNORECASE,
)

# sshd: Accepted password for ubuntu from 203.0.113.99 port 22 ssh2
_SSH_SUCCESS = re.compile(
    r"sshd.*?Accepted (?:password|publickey|keyboard-interactive)"
    r" for (?P<user>\S+) from (?P<src_ip>[\d\.]+) port (?P<src_port>\d+)",
    re.IGNORECASE,
)

# su: pam_unix(su:auth): authentication failure; ... rhost=10.0.0.1 user=root
_PAM_FAIL = re.compile(
    r"pam_unix.*?authentication failure"
    r"(?:.*?rhost=(?P<src_ip>[\d\.]+))?"
    r"(?:.*?user=(?P<user>\S+))?",
    re.IGNORECASE,
)

# Windows JSON event (preprocessed): {"EventID": 4625, "SubjectUserName": "admin", ...}
_WIN_LOGON_FAIL = 4625
_WIN_LOGON_SUCCESS = 4624
_WIN_RDP_SUCCESS = 4648


def parse(raw_line: str, source_host: str = "unknown") -> Optional[dict]:
    """
    Parse a raw auth log line into a Normalized Event Object.

    Args:
        raw_line:    Raw log string (syslog format or JSON string)
        source_host: Hostname/IP of the machine that generated the log

    Returns:
        Normalized Event Object dict, or None if line is not parseable
    """
    raw_line = raw_line.strip()
    if not raw_line:
        return None

    # Try JSON first (Windows Security Event format)
    if raw_line.startswith("{"):
        return _parse_windows_json(raw_line, source_host)

    # Try sshd success
    m = _SSH_SUCCESS.search(raw_line)
    if m:
        return _build_event(
            source_type="auth",
            src_ip=m.group("src_ip"),
            dest_ip=source_host,
            src_port=int(m.group("src_port")),
            dest_port=22,
            user=m.group("user"),
            action="LOGIN_ATTEMPT",
            status="SUCCESS",
            raw={"line": raw_line},
        )

    # Try sshd failure
    m = _SSH_FAIL.search(raw_line)
    if m:
        return _build_event(
            source_type="auth",
            src_ip=m.group("src_ip") or "0.0.0.0",
            dest_ip=source_host,
            src_port=int(m.group("src_port") or 0),
            dest_port=22,
            user=m.group("user"),
            action="LOGIN_ATTEMPT",
            status="FAILURE",
            raw={"line": raw_line},
        )

    # Try PAM failure
    m = _PAM_FAIL.search(raw_line)
    if m:
        return _build_event(
            source_type="auth",
            src_ip=m.group("src_ip") or source_host,
            dest_ip=source_host,
            src_port=0,
            dest_port=0,
            user=m.group("user"),
            action="LOGIN_ATTEMPT",
            status="FAILURE",
            raw={"line": raw_line},
        )

    return None


def _parse_windows_json(raw_line: str, source_host: str) -> Optional[dict]:
    import json
    try:
        evt = json.loads(raw_line)
    except Exception:
        return None

    event_id = evt.get("EventID")
    if event_id not in (_WIN_LOGON_FAIL, _WIN_LOGON_SUCCESS, _WIN_RDP_SUCCESS):
        return None

    status = "SUCCESS" if event_id in (_WIN_LOGON_SUCCESS, _WIN_RDP_SUCCESS) else "FAILURE"
    port = 3389 if event_id == _WIN_RDP_SUCCESS else 445

    return _build_event(
        source_type="auth",
        src_ip=evt.get("IpAddress", "0.0.0.0"),
        dest_ip=source_host,
        src_port=int(evt.get("IpPort", 0) or 0),
        dest_port=port,
        user=evt.get("SubjectUserName") or evt.get("TargetUserName"),
        action="LOGIN_ATTEMPT",
        status=status,
        raw=evt,
    )


def _build_event(
    source_type: str,
    src_ip: str,
    dest_ip: str,
    src_port: int,
    dest_port: int,
    user,
    action: str,
    status: str,
    raw: dict,
) -> dict:
    return {
        "event_id":       str(uuid.uuid4()),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "source_type":    source_type,
        "src_ip":         src_ip or "0.0.0.0",
        "dest_ip":        dest_ip or "0.0.0.0",
        "src_port":       int(src_port or 0),
        "dest_port":      int(dest_port or 0),
        "user":           user,
        "action":         action,
        "status":         status,
        "process_name":   None,
        "parent_proc":    None,
        "child_proc":     None,
        "bytes_sent":     None,
        "uri_query":      None,
        "payload_flags":  [],
        "file_path":      None,
        "file_ops":       None,
        "cpu_pct":        None,
        "non_admin_user": None,
        "http_status":    None,
        "raw":            raw,
    }
