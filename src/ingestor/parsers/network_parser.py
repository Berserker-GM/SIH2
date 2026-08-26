# =============================================================================
# normative/ingestor/parsers/network_parser.py
# Parses HTTP access logs and Netflow records into Normalized Event Objects
# Handles: Apache/Nginx combined log format, JSON access logs, Netflow JSON
# =============================================================================

import re
import uuid
from datetime import datetime, timezone
from typing import Optional

# ── Exploit payload detection patterns ───────────────────────────────────────

_PAYLOAD_PATTERNS = {
    "LFI":       [r"\.\./", r"etc/passwd", r"etc/shadow", r"proc/self", r"win\.ini"],
    "RCE":       [r"whoami", r"cat\s+/", r"ls\s+-", r"id;", r"cmd\.exe", r";bash", r"\|sh"],
    "SQLi":      [r"union\s+select", r"or\s+1=1", r"drop\s+table", r";\s*select", r"'--"],
    "SSRF":      [r"169\.254\.169\.254", r"localhost", r"127\.0\.0\.1", r"metadata\."],
    "XSS":       [r"<script", r"javascript:", r"onerror=", r"onload="],
    "TRAVERSAL": [r"%2e%2e", r"%252e", r"\.\.\\", r"\.\.%2f", r"\.\./"],
}

_COMPILED_PATTERNS = {
    flag: [re.compile(p, re.IGNORECASE) for p in patterns]
    for flag, patterns in _PAYLOAD_PATTERNS.items()
}

# Apache/Nginx combined log:
# 203.0.113.99 - frank [10/Oct/2000:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 2326
_APACHE_LOG = re.compile(
    r'(?P<src_ip>[\d\.]+)\s+-\s+(?P<user>\S+)\s+\[.*?\]\s+'
    r'"(?P<method>\S+)\s+(?P<uri>\S+)\s+HTTP/[\d\.]+"\s+'
    r'(?P<status>\d{3})\s+(?P<bytes>\d+|-)',
    re.IGNORECASE,
)


def parse(raw_line: str, source_host: str = "unknown") -> Optional[dict]:
    """
    Parse a raw network log line into a Normalized Event Object.

    Args:
        raw_line:    Raw log string (Apache/Nginx combined or JSON)
        source_host: IP/hostname of the web server or netflow exporter

    Returns:
        Normalized Event Object dict, or None if not parseable
    """
    raw_line = raw_line.strip()
    if not raw_line:
        return None

    if raw_line.startswith("{"):
        return _parse_json(raw_line, source_host)

    m = _APACHE_LOG.match(raw_line)
    if m:
        return _parse_apache(m, source_host, raw_line)

    return None


def _parse_apache(m: re.Match, source_host: str, raw_line: str) -> dict:
    uri = m.group("uri")
    http_status = int(m.group("status"))
    bytes_sent_raw = m.group("bytes")
    bytes_sent = int(bytes_sent_raw) if bytes_sent_raw != "-" else 0
    user = m.group("user") if m.group("user") != "-" else None

    payload_flags = detect_payload_flags(uri)

    return {
        "event_id":       str(uuid.uuid4()),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "source_type":    "network",
        "src_ip":         m.group("src_ip"),
        "dest_ip":        source_host,
        "src_port":       0,
        "dest_port":      443 if http_status else 80,
        "user":           user,
        "action":         "HTTP_REQUEST",
        "status":         "SUCCESS" if http_status < 400 else "FAILURE",
        "process_name":   None,
        "parent_proc":    None,
        "child_proc":     None,
        "bytes_sent":     bytes_sent,
        "uri_query":      uri,
        "payload_flags":  payload_flags,
        "file_path":      None,
        "file_ops":       None,
        "cpu_pct":        None,
        "non_admin_user": None,
        "http_status":    http_status,
        "raw":            {"line": raw_line},
    }


def _parse_json(raw_line: str, source_host: str) -> Optional[dict]:
    import json
    try:
        evt = json.loads(raw_line)
    except Exception:
        return None

    # Support two formats: HTTP access log JSON and Netflow JSON
    if "bytes_sent" in evt or "bytes" in evt:
        return _parse_netflow_json(evt, source_host)
    if "uri" in evt or "request" in evt:
        return _parse_http_json(evt, source_host)
    return None


def _parse_http_json(evt: dict, source_host: str) -> dict:
    uri = evt.get("uri") or evt.get("request", "")
    http_status = int(evt.get("status", 0))
    payload_flags = detect_payload_flags(uri)
    payload_flags += detect_payload_flags(evt.get("body", ""))

    return {
        "event_id":       str(uuid.uuid4()),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "source_type":    "network",
        "src_ip":         evt.get("src_ip") or evt.get("remote_addr", "0.0.0.0"),
        "dest_ip":        source_host,
        "src_port":       int(evt.get("src_port", 0) or 0),
        "dest_port":      int(evt.get("dest_port", 80) or 80),
        "user":           evt.get("user") or evt.get("username"),
        "action":         "HTTP_REQUEST",
        "status":         "SUCCESS" if http_status < 400 else "FAILURE",
        "process_name":   None,
        "parent_proc":    None,
        "child_proc":     None,
        "bytes_sent":     int(evt.get("bytes_sent", 0) or 0),
        "uri_query":      uri,
        "payload_flags":  list(set(payload_flags)),
        "file_path":      None,
        "file_ops":       None,
        "cpu_pct":        None,
        "non_admin_user": None,
        "http_status":    http_status,
        "raw":            evt,
    }


def _parse_netflow_json(evt: dict, source_host: str) -> dict:
    bytes_sent = int(evt.get("bytes_sent") or evt.get("bytes") or 0)
    dest_ip = evt.get("dest_ip") or evt.get("dst_ip") or source_host

    return {
        "event_id":       str(uuid.uuid4()),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "source_type":    "network",
        "src_ip":         evt.get("src_ip", "0.0.0.0"),
        "dest_ip":        dest_ip,
        "src_port":       int(evt.get("src_port", 0) or 0),
        "dest_port":      int(evt.get("dest_port", 0) or 0),
        "user":           None,
        "action":         "NETWORK_FLOW",
        "status":         None,
        "process_name":   None,
        "parent_proc":    None,
        "child_proc":     None,
        "bytes_sent":     bytes_sent,
        "uri_query":      None,
        "payload_flags":  [],
        "file_path":      None,
        "file_ops":       None,
        "cpu_pct":        None,
        "non_admin_user": None,
        "http_status":    None,
        "raw":            evt,
    }


def detect_payload_flags(text: str) -> list:
    """Scan text for known exploit payload patterns. Returns list of flag strings."""
    if not text:
        return []
    flags = []
    for flag, patterns in _COMPILED_PATTERNS.items():
        if any(p.search(text) for p in patterns):
            flags.append(flag)
    return flags
