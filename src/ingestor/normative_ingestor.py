#!/usr/bin/env python3
# =============================================================================
# normative/ingestor/normative_ingestor.py
# Main ingestor service loop.
#
# Flow:
#   1. Subscribes to Redis list events:raw (populated by simulator or real agents)
#   2. Auto-detects source_type and routes to correct parser
#   3. Validates and publishes Normalized Event Object to events:normalized
#   4. Passes event to FeatureExtractor → publishes ML vector + graph edge
#   5. Runs MITRE rule matching on every event (stateful window per host)
#   6. Prints structured log to stdout
#
# Usage:
#   python normative_ingestor.py
#   python normative_ingestor.py --redis-host localhost --redis-port 6379
# =============================================================================

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Add project root (Normnative/) to Python path so absolute imports work.
# This file is at src/ingestor/normative_ingestor.py → project root = parent.parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

# Now absolute imports from project root
from src.ingestor.parsers import PARSER_MAP
from src.ingestor.feature_extractor import FeatureExtractor
from src.ingestor.mitre_lookup import match_rule, MITRE_BY_TECHNIQUE

# ── Required keys every Normalized Event Object must have ────────────────────
REQUIRED_KEYS = {
    "event_id", "timestamp", "source_type",
    "src_ip", "dest_ip", "src_port", "dest_port",
    "action", "payload_flags", "raw",
}


class NormativeIngestor:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        import redis
        self.r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.extractor = FeatureExtractor()
        # Per-host recent event windows for stateful MITRE rules (brute force etc.)
        self._host_windows: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._stats = {"processed": 0, "anomalies": 0, "errors": 0, "mitre_hits": 0}

    def run(self):
        print("[INGESTOR] Normative ingestor started. Listening on events:raw ...")
        while True:
            try:
                # Blocking right-pop with 2s timeout
                result = self.r.brpop("events:raw", timeout=2)
                if result is None:
                    continue
                _, raw_payload = result
                self._handle_raw(raw_payload)
            except KeyboardInterrupt:
                print("\n[INGESTOR] Shutting down.")
                self._print_stats()
                break
            except Exception as e:
                self._stats["errors"] += 1
                print(f"[INGESTOR] Error: {e}")

    def _handle_raw(self, raw_payload: str):
        # ── 1. Parse raw payload ─────────────────────────────────────────
        try:
            raw = json.loads(raw_payload)
        except json.JSONDecodeError:
            # Treat as plain log line — try all parsers
            raw = {"_line": raw_payload}

        # ── 2. If already a partial normalized event (from simulator) ────
        if "source_type" in raw and "action" in raw:
            event = self._fill_defaults(raw)
        else:
            # ── 3. Route to correct parser ───────────────────────────────
            event = self._route_to_parser(raw)
            if event is None:
                return

        # ── 4. Validate ──────────────────────────────────────────────────
        if not self._validate(event):
            self._stats["errors"] += 1
            return

        # ── 5. Publish normalized event ──────────────────────────────────
        payload = json.dumps(event)
        self.r.lpush("events:normalized", payload)
        self.r.publish("events:normalized", payload)

        # ── 6. Update host window ─────────────────────────────────────────
        host_key = event.get("src_ip", "unknown")
        self._host_windows[host_key].append(event)

        # ── 7. MITRE rule matching ────────────────────────────────────────
        recent = list(self._host_windows[host_key])
        matched_rules = match_rule(event, recent_events=recent)
        if matched_rules:
            self._stats["mitre_hits"] += len(matched_rules)
            for rule in matched_rules:
                self._log_mitre_hit(event, rule)

        # ── 8. Feature extraction ────────────────────────────────────────
        ml_vector, graph_edge = self.extractor.process(event)

        self.r.lpush("features:graph", json.dumps(graph_edge))
        self.r.publish("features:graph", json.dumps(graph_edge))

        if ml_vector is not None:
            ml_vector["matched_rules"] = matched_rules
            self.r.lpush("features:ml", json.dumps(ml_vector))
            self.r.publish("features:ml", json.dumps(ml_vector))

        # ── 9. Stats + log ───────────────────────────────────────────────
        self._stats["processed"] += 1
        if self._stats["processed"] % 50 == 0:
            self._print_stats()

        self._log_event(event, matched_rules)

    def _route_to_parser(self, raw: dict) -> dict | None:
        """Try parsers in priority order. Return first successful parse."""
        line = raw.get("_line", json.dumps(raw))
        source_host = raw.get("source_host", "unknown")

        # Auth: SSH/RDP keywords
        if any(k in line for k in ("sshd", "pam_unix", "EventID", "LOGIN")):
            return PARSER_MAP["auth"](line, source_host)
        # Endpoint: process/file keywords
        if any(k in line for k in ("PROCESS_SPAWN", "FILE_WRITE", "parent_proc",
                                    "ParentImage", "SYSCALL", "EXECVE")):
            return PARSER_MAP["endpoint"](line, source_host)
        # Syslog: cron/bulk/kernel
        if any(k in line for k in ("cron", "CRON", "kernel", "ransomware", "file_ops")):
            return PARSER_MAP["syslog"](line, source_host)
        # Network: HTTP/netflow
        if any(k in line for k in ("HTTP", "GET ", "POST ", "bytes_sent", "uri")):
            return PARSER_MAP["network"](line, source_host)

        # Last resort: try each parser
        for name, parser_fn in PARSER_MAP.items():
            result = parser_fn(line, source_host)
            if result is not None:
                return result
        return None

    def _fill_defaults(self, raw: dict) -> dict:
        """For events already structured (e.g. from simulator), fill missing keys."""
        import uuid
        defaults = {
            "event_id":       str(uuid.uuid4()),
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "src_ip":         "0.0.0.0",
            "dest_ip":        "0.0.0.0",
            "src_port":       0,
            "dest_port":      0,
            "user":           None,
            "status":         None,
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
            "raw":            {},
        }
        for k, v in defaults.items():
            raw.setdefault(k, v)
        return raw

    def _validate(self, event: dict) -> bool:
        missing = REQUIRED_KEYS - set(event.keys())
        if missing:
            print(f"[INGESTOR] Invalid event — missing keys: {missing}")
            return False
        return True

    def _log_event(self, event: dict, matched_rules: list):
        ts   = event.get("timestamp", "")[:19]
        src  = event.get("src_ip", "?")
        act  = event.get("action", "?")
        stype = event.get("source_type", "?")
        rule_str = ""
        if matched_rules:
            ids = ", ".join(r["technique_id"] for r in matched_rules)
            rule_str = f"  ⚠  MITRE: {ids}"
        print(f"[INGESTOR] {ts}  {stype:<10}  {src:<18}  {act:<25}{rule_str}")

    def _log_mitre_hit(self, event: dict, rule: dict):
        print(f"[INGESTOR] 🔴 MITRE HIT  {rule['rule_id']}  {rule['technique_id']}  "
              f"{rule['technique_name']}  [{rule['tactic']}]  "
              f"src={event.get('src_ip')}")

    def _print_stats(self):
        print(f"\n[INGESTOR] ── Stats ──  processed={self._stats['processed']}  "
              f"mitre_hits={self._stats['mitre_hits']}  errors={self._stats['errors']}\n")


def main():
    import os
    parser = argparse.ArgumentParser(description="Normative Ingestor Service")
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", 6379)))
    args = parser.parse_args()

    try:
        import redis
        r = redis.Redis(host=args.redis_host, port=args.redis_port)
        r.ping()
        print(f"[INGESTOR] Redis connected at {args.redis_host}:{args.redis_port}")
    except Exception as e:
        print(f"[INGESTOR] Cannot connect to Redis: {e}")
        print("[INGESTOR] Start Redis with: redis-server")
        sys.exit(1)

    ingestor = NormativeIngestor(args.redis_host, args.redis_port)
    ingestor.run()


if __name__ == "__main__":
    main()