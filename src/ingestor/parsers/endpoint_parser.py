# =============================================================================
# normative/ingestor/parsers/endpoint_parser.py
# Parses Sysmon (Windows) and Linux Auditd endpoint events
# Handles: Sysmon EventID 1 (process create), Auditd execve, file events
# =============================================================================

import re
import uuid
from datetime import datetime, timezone
from typing import Optional

# Admin users/processes — used to set non_admin_user flag
_ADMIN_USERS = {"root", "SYSTEM", "Administrator", "admin", "wheel"}
_ADMIN_PROCESSES = {"svchost.exe", "services.exe", "lsass.exe", "wininit.exe"}

# Sysmon EventIDs we care about
_SYSMON_PROCESS_CREATE = 1
_SYSMON_FILE_CREATE    = 11
_SYSMON_REGISTRY_SET   = 13
_SYSMON_NETWORK_CONN   = 3

# Auditd syscall names for file operations
_FILE_WRITE_SYSCALLS = {"write", "pwrite64", "writev", "rename", "renameat", "unlink", "unlinkat"}


def parse(raw_line: str, source_host: str = "unknown") -> Optional[dict]:
    """
    Parse a raw endpoint log line into a Normalized Event Object.

    Args:
        raw_line:    Raw log string (JSON for Sysmon/Auditd)
        source_host: Hostname/IP of the endpoint

    Returns:
        Normalized Event Object dict, or None if not parseable
    """
    raw_line = raw_line.strip()
    if not raw_line:
        return None

    if raw_line.startswith("{"):
        return _parse_json(raw_line, source_host)

    # Plain auditd key=value format
    if "type=SYSCALL" in raw_line or "type=EXECVE" in raw_line:
        return _parse_auditd_kv(raw_line, source_host)

    return None


def _parse_json(raw_line: str, source_host: str) -> Optional[dict]:
    import json
    try:
        evt = json.loads(raw_line)
    except Exception:
        return None

    # Sysmon JSON (e.g. from Winlogbeat)
    event_id = int(evt.get("EventID") or evt.get("event_id") or 0)

    if event_id == _SYSMON_PROCESS_CREATE:
        return _parse_sysmon_process(evt, source_host)
    if event_id == _SYSMON_FILE_CREATE:
        return _parse_sysmon_file(evt, source_host)
    if event_id == _SYSMON_REGISTRY_SET:
        return _parse_sysmon_registry(evt, source_host)

    # Generic endpoint JSON (e.g. from osquery, EDR)
    if "parent_proc" in evt or "parent_process" in evt or "ParentImage" in evt:
        return _parse_generic_process(evt, source_host)
    if "file_path" in evt or "TargetFilename" in evt:
        return _parse_generic_file(evt, source_host)

    return None


def _parse_sysmon_process(evt: dict, source_host: str) -> dict:
    image = evt.get("Image") or evt.get("process_name", "")
    parent_image = evt.get("ParentImage") or evt.get("parent_proc", "")
    cmd = evt.get("CommandLine") or ""
    user = evt.get("User") or evt.get("user")

    return _build_event(
        source_type="endpoint",
        src_ip=source_host,
        dest_ip=source_host,
        user=user,
        action="PROCESS_SPAWN",
        status=None,
        process_name=_basename(image),
        parent_proc=_basename(parent_image),
        child_proc=image,   # full path preserved — MITRE RULE_003 matches on /bin/bash etc.
        file_path=None,
        file_ops=None,
        cpu_pct=None,
        non_admin_user=_is_non_admin(user, image),
        raw=evt,
    )


def _parse_sysmon_file(evt: dict, source_host: str) -> dict:
    path = evt.get("TargetFilename") or evt.get("file_path", "")
    user = evt.get("User") or evt.get("user")
    return _build_event(
        source_type="endpoint",
        src_ip=source_host,
        dest_ip=source_host,
        user=user,
        action="FILE_WRITE",
        status=None,
        process_name=_basename(evt.get("Image", "")),
        parent_proc=None,
        child_proc=None,
        file_path=path,
        file_ops=1,
        cpu_pct=None,
        non_admin_user=_is_non_admin(user, ""),
        raw=evt,
    )


def _parse_sysmon_registry(evt: dict, source_host: str) -> dict:
    path = evt.get("TargetObject") or evt.get("file_path", "")
    user = evt.get("User") or evt.get("user")
    return _build_event(
        source_type="endpoint",
        src_ip=source_host,
        dest_ip=source_host,
        user=user,
        action="REGISTRY_WRITE",
        status=None,
        process_name=_basename(evt.get("Image", "")),
        parent_proc=None,
        child_proc=None,
        file_path=path,
        file_ops=1,
        cpu_pct=None,
        non_admin_user=_is_non_admin(user, ""),
        raw=evt,
    )


def _parse_generic_process(evt: dict, source_host: str) -> dict:
    parent = evt.get("parent_proc") or evt.get("parent_process") or evt.get("ParentImage", "")
    child = evt.get("child_proc") or evt.get("process_name") or evt.get("Image", "")
    user = evt.get("user") or evt.get("User")
    return _build_event(
        source_type="endpoint",
        src_ip=evt.get("src_ip", source_host),
        dest_ip=source_host,
        user=user,
        action="PROCESS_SPAWN",
        status=None,
        process_name=_basename(child),
        parent_proc=_basename(parent),
        child_proc=_basename(child),
        file_path=evt.get("file_path"),
        file_ops=evt.get("file_ops"),
        cpu_pct=evt.get("cpu_pct"),
        non_admin_user=_is_non_admin(user, child),
        raw=evt,
    )


def _parse_generic_file(evt: dict, source_host: str) -> dict:
    path = evt.get("file_path") or evt.get("TargetFilename", "")
    user = evt.get("user") or evt.get("User")
    return _build_event(
        source_type="endpoint",
        src_ip=source_host,
        dest_ip=source_host,
        user=user,
        action="FILE_WRITE",
        status=None,
        process_name=evt.get("process_name"),
        parent_proc=None,
        child_proc=None,
        file_path=path,
        file_ops=int(evt.get("file_ops") or 1),
        cpu_pct=evt.get("cpu_pct"),
        non_admin_user=_is_non_admin(user, ""),
        raw=evt,
    )


def _parse_auditd_kv(raw_line: str, source_host: str) -> Optional[dict]:
    """Parse auditd key=value format log lines."""
    kv = dict(re.findall(r'(\w+)=(?:"([^"]*)"|([\S]*))', raw_line))
    cleaned = {k: (v1 or v2) for k, (v1, v2) in
               [(k, (m[1], m[2])) for k, m in
                [(k, re.search(r'=(?:"([^"]*)"|([\S]*))', f'{k}={v}'))
                 for k, v in kv.items()] if m]}

    syscall = cleaned.get("syscall", "")
    exe = cleaned.get("exe", "")
    user = cleaned.get("auid") or cleaned.get("uid")

    action = "FILE_WRITE" if syscall in _FILE_WRITE_SYSCALLS else "PROCESS_SPAWN"

    return _build_event(
        source_type="endpoint",
        src_ip=source_host,
        dest_ip=source_host,
        user=user,
        action=action,
        status=None,
        process_name=_basename(exe),
        parent_proc=_basename(cleaned.get("ppid_exe", "")),
        child_proc=None,
        file_path=cleaned.get("name"),
        file_ops=1 if action == "FILE_WRITE" else None,
        cpu_pct=None,
        non_admin_user=_is_non_admin(user, exe),
        raw={"line": raw_line},
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _basename(path: str) -> Optional[str]:
    if not path:
        return None
    # Handle both Unix and Windows paths
    return path.replace("\\", "/").rstrip("/").split("/")[-1].lower() or None


def _is_non_admin(user: Optional[str], process: str) -> bool:
    if not user:
        return False
    user_lower = user.lower()
    if user_lower in {u.lower() for u in _ADMIN_USERS}:
        return False
    if any(p.lower() in process.lower() for p in _ADMIN_PROCESSES):
        return False
    return True


def _build_event(
    source_type, src_ip, dest_ip, user, action, status,
    process_name, parent_proc, child_proc, file_path,
    file_ops, cpu_pct, non_admin_user, raw
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
        "parent_proc":    parent_proc,
        "child_proc":     child_proc,
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
