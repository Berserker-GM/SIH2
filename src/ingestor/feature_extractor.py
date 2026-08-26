# =============================================================================
# normative/ingestor/feature_extractor.py
# Produces two output types from normalized events:
#   1. ML_FEATURE_VECTOR  → Redis channel: features:ml   (consumed by Dev 2)
#   2. GRAPH_EDGE         → Redis channel: features:graph (consumed by Dev 3)
#
# FEATURE_ORDER (index matters — Dev 2 LSTM must match this exactly):
#   0  login_fail_count
#   1  login_success_count
#   2  unique_dest_ips
#   3  bytes_sent_total
#   4  file_ops_count
#   5  cpu_pct_avg
#   6  payload_flag_count
#   7  inter_arrival_ms_avg
#   8  entropy_dest_ports
#   9  src_port_norm        (src_port / 65535)
#   10 dest_port_norm       (dest_port / 65535)
# =============================================================================

import uuid
import math
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

# FEATURE_ORDER — share this with Dev 2, must be identical
FEATURE_ORDER = [
    "login_fail_count",
    "login_success_count",
    "unique_dest_ips",
    "bytes_sent_total",
    "file_ops_count",
    "cpu_pct_avg",
    "payload_flag_count",
    "inter_arrival_ms_avg",
    "entropy_dest_ports",
    "src_port_norm",
    "dest_port_norm",
]

WINDOW_SIZE = 5         # events per sliding window per host
INPUT_DIM   = len(FEATURE_ORDER)   # = 11, passed to Dev 2 as input_dim


class FeatureExtractor:
    """
    Stateful feature extractor. Maintains per-host sliding windows
    and emits ML vectors + graph edges for every incoming event.
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        # host_key → deque of normalized events
        self._windows: dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

    def process(self, event: dict) -> tuple[Optional[dict], dict]:
        """
        Process one normalized event.

        Returns:
            (ml_vector, graph_edge)
            ml_vector is None until the window is full for this host.
            graph_edge is always returned.
        """
        host_key = event.get("src_ip", "unknown")
        self._windows[host_key].append(event)

        graph_edge = self._make_graph_edge(event)

        window = list(self._windows[host_key])
        if len(window) < self.window_size:
            return None, graph_edge

        ml_vector = self._make_ml_vector(host_key, window)
        return ml_vector, graph_edge

    # ── ML Feature Vector ─────────────────────────────────────────────────

    def _make_ml_vector(self, host_key: str, window: list) -> dict:
        login_fails   = sum(1 for e in window if e.get("action") == "LOGIN_ATTEMPT" and e.get("status") == "FAILURE")
        login_success = sum(1 for e in window if e.get("action") == "LOGIN_ATTEMPT" and e.get("status") == "SUCCESS")
        unique_dests  = len({e.get("dest_ip") for e in window if e.get("dest_ip")})
        bytes_total   = sum(e.get("bytes_sent") or 0 for e in window)
        file_ops      = sum(e.get("file_ops") or 0 for e in window)
        cpu_values    = [e["cpu_pct"] for e in window if e.get("cpu_pct") is not None]
        cpu_avg       = sum(cpu_values) / len(cpu_values) if cpu_values else 0.0
        payload_count = sum(len(e.get("payload_flags") or []) for e in window)

        # Inter-arrival time in ms
        timestamps = []
        for e in window:
            try:
                ts = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
                timestamps.append(ts.timestamp() * 1000)
            except Exception:
                pass
        if len(timestamps) > 1:
            diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            inter_arrival_avg = sum(diffs) / len(diffs)
        else:
            inter_arrival_avg = 0.0

        # Shannon entropy of destination ports
        dest_ports = [e.get("dest_port") for e in window if e.get("dest_port")]
        entropy = _shannon_entropy(dest_ports)

        latest = window[-1]
        src_port_norm  = (latest.get("src_port") or 0) / 65535
        dest_port_norm = (latest.get("dest_port") or 0) / 65535

        return {
            "event_id":            latest["event_id"],
            "host_key":            host_key,
            "window_events":       [e["event_id"] for e in window],
            # Named features (for readability)
            "login_fail_count":    login_fails,
            "login_success_count": login_success,
            "unique_dest_ips":     unique_dests,
            "bytes_sent_total":    bytes_total,
            "file_ops_count":      file_ops,
            "cpu_pct_avg":         round(cpu_avg, 4),
            "payload_flag_count":  payload_count,
            "inter_arrival_ms_avg":round(inter_arrival_avg, 4),
            "entropy_dest_ports":  round(entropy, 4),
            "src_port_norm":       round(src_port_norm, 6),
            "dest_port_norm":      round(dest_port_norm, 6),
            # Flat vector in FEATURE_ORDER (what the LSTM actually consumes)
            "feature_vector":      [
                login_fails,
                login_success,
                unique_dests,
                bytes_total,
                file_ops,
                round(cpu_avg, 4),
                payload_count,
                round(inter_arrival_avg, 4),
                round(entropy, 4),
                round(src_port_norm, 6),
                round(dest_port_norm, 6),
            ],
        }

    # ── Graph Edge ────────────────────────────────────────────────────────

    def _make_graph_edge(self, event: dict) -> dict:
        src_node = event.get("src_ip") or "unknown"
        action   = event.get("action", "")

        # Determine edge type and destination node
        if action in ("LOGIN_ATTEMPT", "PROCESS_SPAWN") and event.get("user"):
            dst_node  = event["user"]
            edge_type = "auth"
        elif action == "PROCESS_SPAWN" and event.get("child_proc"):
            dst_node  = event["child_proc"]
            edge_type = "process_spawn"
        elif action in ("FILE_WRITE", "FILE_INTEGRITY_CHANGE", "REGISTRY_WRITE", "CRON_MODIFY") and event.get("file_path"):
            dst_node  = event["file_path"]
            edge_type = "file_access"
        elif event.get("dest_ip"):
            dst_node  = event["dest_ip"]
            edge_type = "network"
        else:
            dst_node  = event.get("dest_ip") or "unknown"
            edge_type = "network"

        # Weight: normalize bytes_sent (cap at 100MB), else 1.0
        bytes_sent = event.get("bytes_sent") or 0
        weight = min(bytes_sent / 100_000_000, 1.0) if bytes_sent else 1.0

        return {
            "edge_id":   str(uuid.uuid4()),
            "src_node":  src_node,
            "dst_node":  dst_node,
            "edge_type": edge_type,
            "weight":    round(weight, 6),
            "timestamp": event.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "event_id":  event["event_id"],
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _shannon_entropy(values: list) -> float:
    """Shannon entropy of a list of values (e.g. dest ports)."""
    if not values:
        return 0.0
    from collections import Counter
    counts = Counter(values)
    total  = len(values)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())
