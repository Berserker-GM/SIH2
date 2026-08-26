# =============================================================================
# normative/ingestor/parsers/syslog_parser.py
# Parses system events: kernel messages, cron, file integrity, resource spikes
# Handles: /var/log/syslog, /var/log/messages, journald JSON output
# =============================================================================

import re
import uuid
from datetime import datetime, timezone
from typing import Optional

# Syslog line pattern: Jun 15 14:23:11 hostname process[pid]: message
_SYSLOG_HEADER = re.compile(
    r"(?P<month>\w+)\s+(?P<day>\d+)\s+(?P<time>[\d:]+)\s+"
    r"(?P<host>\S+)\s+(?P<process>[^\[:]+?)(?:\[(?P<pid>\d+)\])?:\s+"
    r"(?P<message>.+)"
)

# Cron job patterns
_CRON_CMD = re.compile(r"CMD\s+\((?P<cmd>.+)\)", re.IGNORECASE)
_CRON_EDIT = re.compile(r"(?:REPLACE|BEGIN EDIT|LIST)\s+(?P<user>\S+)", re.IGNORECASE)

# OOM / resource spike
_OOM = re.compile(r"Out of memory|oom_kill_process|Killed process", re.IGNORECASE)
_HIGH_CPU = re.compile(r"(?:cpu|load average)[^\d]+([\d\.]+)", re.IGNORECASE)

# File integrity (AIDE, Tripwire, auditd file watch)
_FILE_INTEGRITY = re.compile(
    r"(?:changed|modified|added|removed|deleted)[:\s]+(?P<path>/[\S]+)",
    re.IGNORECASE,
)

# Ransomware-like bulk file activity (custom detector messages)
_BULK_FILE = re.compile(
    r"(?P<ops>\d+)\s+file(?:s)?\s+(?:modified|encrypted|renamed|deleted)"
    r"(?:\s+cpu[:\s]+(?P<cpu>[\d\.]+))?",
    re.IGNORECASE,
)


def parse(raw_line: str, source_host: str = "unknown") -> Optional[dict]:
    """
    Parse a raw syslog line into a Normalized Event Object.

    Args:
        raw_line:    Raw log string (syslog format or journald JSON)
        source_host: Hostname/IP of the system

    Returns:
        Normalized Event Object dict, or None if not parseable
    """
    raw_line = raw_line.strip()
    if not raw_line:
        return None

    if raw_line.startswith("{"):
        return _parse_journald_json(raw_line, source_host)

    return _parse_syslog_line(raw_line, source_host)


def _parse_syslog_line(raw_line: str, source_host: str) -> Optional[dict]:
    m = _SYSLOG_HEADER.match(raw_line)
    if not m:
        return None

    process = m.group("process").strip().lower()
    message = m.group("message").strip()
    host = m.group("host") or source_host

    # ── Cron modifications ───────────────────────────────────────────────
    if "cron" in process:
        return _handle_cron(message, host, raw_line)

    # ── OOM / resource events ────────────────────────────────────────────
    if _OOM.search(message):
        return _build_event(
            source_type="syslog",
            src_ip=host,
            dest_ip=host,
            user=None,
            action="SYSTEM_OOM",
            status=None,
            process_name=process,
            file_path=None,
            file_ops=None,
            cpu_pct=None,
            non_admin_user=None,
            raw={"line": raw_line},
        )

    # ── Bulk file operations (ransomware detector output) ─────────────────
    bm = _BULK_FILE.search(message)
    if bm:
        ops = int(bm.group("ops") or 0)
        cpu = float(bm.group("cpu") or 0.0)
        return _build_event(
            source_type="syslog",
            src_ip=host,
            dest_ip=host,
            user=None,
            action="BULK_FILE_OPERATION",
            status=None,
            process_name=process,
            file_path=None,
            file_ops=ops,
            cpu_pct=cpu,
            non_admin_user=None,
            raw={"line": raw_line},
        )

    # ── File integrity change ─────────────────────────────────────────────
    fm = _FILE_INTEGRITY.search(message)
    if fm:
        return _build_event(
            source_type="syslog",
            src_ip=host,
            dest_ip=host,
            user=None,
            action="FILE_INTEGRITY_CHANGE",
            status=None,
            process_name=process,
            file_path=fm.group("path"),
            file_ops=1,
            cpu_pct=None,
            non_admin_user=None,
            raw={"line": raw_line},
        )

    return None


def _handle_cron(message: str, host: str, raw_line: str) -> Optional[dict]:
    # Cron edit = persistence mechanism (T1547)
    em = _CRON_EDIT.search(message)
    if em:
        user = em.group("user")
        # Any cron edit by a non-standard user is suspicious
        non_admin = user.lower() not in {"root", "admin", "cron"}
        return _build_event(
            source_type="syslog",
            src_ip=host,
            dest_ip=host,
            user=user,
            action="CRON_MODIFY",
            status=None,
            process_name="cron",
            file_path=f"/etc/cron.d/{user}",
            file_ops=1,
            cpu_pct=None,
            non_admin_user=non_admin,
            raw={"line": raw_line},
        )

    # Cron execution (less interesting, skip unless needed)
    return None


def _parse_journald_json(raw_line: str, source_host: str) -> Optional[dict]:
    import json
    try:
        evt = json.loads(raw_line)
    except Exception:
        return None

    message = evt.get("MESSAGE", "")
    unit = evt.get("_SYSTEMD_UNIT", "")
    host = evt.get("_HOSTNAME") or source_host
    user = evt.get("_UID") or evt.get("SYSLOG_IDENTIFIER")

    # Re-use line parser on the message field
    synthetic = f"Jan  1 00:00:00 {host} {unit}: {message}"
    result = _parse_syslog_line(synthetic, host)
    if result:
        result["raw"] = evt
        if user:
            result["user"] = str(user)
    return result


def _build_event(
    source_type, src_ip, dest_ip, user, action, status,
    process_name, file_path, file_ops, cpu_pct, non_admin_user, raw
) -> dict:
    return {
        "event_id":       str(uuid.uuid4()),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "source_type":    source_type,
        "src_ip":         src_ip or "0.0.0.0",
        "dest_ip":        dest_ip or "0.0.0.0",
        "src_port":       0,
        "dest_port":      0,
        "user":           user,
        "action":         action,
        "status":         status,
        "process_name":   process_name,
        "parent_proc":    None,
        "child_proc":     None,
        "bytes_sent":     None,
        "uri_query":      None,
        "payload_flags":  [],
        "file_path":      file_path,
        "file_ops":       file_ops,
        "cpu_pct":        cpu_pct,
        "non_admin_user": non_admin_user,
        "http_status":    None,
        "raw":            raw,
    }
