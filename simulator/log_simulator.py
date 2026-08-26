#!/usr/bin/env python3
# =============================================================================
# normative/simulator/log_simulator.py
#
# Usage:
#   python log_simulator.py                    # full APT, real-time speed
#   python log_simulator.py --speed 4          # 4× faster
#   python log_simulator.py --scenario fast    # compressed demo scenario
#   python log_simulator.py --dry-run          # print events, no Redis
#
# Publishes to:
#   Redis list    events:raw      (ingestor reads from here)
#   Redis pubsub  events:stream   (optional live consumers)
# =============================================================================

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Add project root (Normnative/) to Python path so absolute imports work.
# This file is at simulator/log_simulator.py → project root = parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Now we can import using absolute package paths
from simulator.attack_scenarios import APT_SCENARIO, APT_SCENARIO_FAST, BENIGN_EVENTS
# ─────────────────────────────────────────────────────────────────────────────


def make_event(partial: dict) -> dict:
    """Stamp a partial event dict with event_id and timestamp."""
    event = dict(partial)
    event["event_id"] = str(uuid.uuid4())
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    # Ensure all required keys exist (fill missing with None / [])
    defaults = {
        "source_type": "syslog", "src_ip": "0.0.0.0", "dest_ip": "0.0.0.0",
        "src_port": 0, "dest_port": 0, "user": None, "action": "UNKNOWN",
        "status": None, "process_name": None, "parent_proc": None,
        "child_proc": None, "bytes_sent": None, "uri_query": None,
        "payload_flags": [], "file_path": None, "file_ops": None,
        "cpu_pct": None, "non_admin_user": None, "http_status": None,
        "raw": {},
    }
    for k, v in defaults.items():
        event.setdefault(k, v)
    event["raw"] = {"simulated": True, "technique_id": event.pop("technique_id", None)}
    event.pop("step", None)
    event.pop("delay_after_prev_sec", None)
    event.pop("repeat", None)
    return event


def publish(r, event: dict, dry_run: bool):
    payload = json.dumps(event)
    if dry_run:
        src = event.get("src_ip", "?")
        action = event.get("action", "?")
        status = event.get("status") or ""
        tech = event.get("raw", {}).get("technique_id") or "benign"
        print(f"  [{tech:>8}]  {src:<18}  {action:<25}  {status}")
        return
    r.lpush("events:raw", payload)
    r.publish("events:stream", payload)


def run_benign_phase(r, duration_sec: int, speed: float, dry_run: bool):
    """Publish benign baseline events for `duration_sec` seconds."""
    print(f"\n[SIMULATOR] Benign baseline phase ({duration_sec}s at {speed}× speed)...")
    end = time.time() + (duration_sec / speed)
    idx = 0
    while time.time() < end:
        event = make_event(dict(BENIGN_EVENTS[idx % len(BENIGN_EVENTS)]))
        publish(r, event, dry_run)
        idx += 1
        time.sleep(1.5 / speed)
    print(f"[SIMULATOR] Benign phase complete. {idx} events published.")


def run_attack_phase(r, scenario: list, speed: float, dry_run: bool):
    """Replay the APT scenario step by step."""
    print(f"\n[SIMULATOR] *** ATTACK PHASE STARTING *** ({speed}× speed)\n")
    import random
    scenario_copy = list(scenario)
    random.shuffle(scenario_copy)
    total = 0
    for step_def in scenario_copy:
        step_num   = step_def["step"]
        tech_id    = step_def["technique_id"]
        delay      = step_def["delay_after_prev_sec"]
        repeat     = step_def.get("repeat", 1)
        event_tmpl = dict(step_def["event"])
        event_tmpl["technique_id"] = tech_id
        event_tmpl["step"] = step_num

        if delay > 0:
            actual_delay = delay / speed
            print(f"[SIMULATOR] Waiting {actual_delay:.1f}s before step {step_num} ({tech_id})...")
            time.sleep(actual_delay)

        print(f"[SIMULATOR] Step {step_num} | {tech_id} | ×{repeat}")
        for i in range(repeat):
            event = make_event(dict(event_tmpl))
            publish(r, event, dry_run)
            total += 1
            if repeat > 1:
                time.sleep(0.8 / speed)   # slight spacing between repeated events

    print(f"\n[SIMULATOR] Attack phase complete. {total} attack events published.")


def main():
    parser = argparse.ArgumentParser(description="Normative Log Simulator")
    parser.add_argument("--speed",    type=float, default=1.0,
                        help="Playback speed multiplier (default 1.0, demo use 4.0)")
    parser.add_argument("--scenario", choices=["full", "fast"], default="full",
                        help="full = real timing, fast = compressed 4× scenario")
    parser.add_argument("--benign-duration", type=int, default=90,
                        help="Seconds of benign baseline before attack (default 90)")
    parser.add_argument("--no-benign", action="store_true",
                        help="Skip benign phase, jump straight to attack")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print events to stdout, do not publish to Redis")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    scenario = APT_SCENARIO_FAST if args.scenario == "fast" else APT_SCENARIO
    speed    = args.speed

    r = None
    if not args.dry_run:
        try:
            import redis 
            r = redis.Redis(host=args.redis_host, port=args.redis_port,
                            decode_responses=True)
            r.ping()
            print(f"[SIMULATOR] Connected to Redis at {args.redis_host}:{args.redis_port}")
        except Exception as e:
            print(f"[SIMULATOR] Redis connection failed: {e}")
            print("[SIMULATOR] Falling back to dry-run mode.")
            args.dry_run = True

    print(f"[SIMULATOR] Starting Normative demo simulation")
    print(f"[SIMULATOR] Scenario: {args.scenario} | Speed: {speed}× | Dry-run: {args.dry_run}")

    if not args.no_benign:
        run_benign_phase(r, args.benign_duration, speed, args.dry_run)

    run_attack_phase(r, scenario, speed, args.dry_run)

    print("\n[SIMULATOR] Full simulation complete.")


if __name__ == "__main__":
    main()