"""
validate.py — Dev 2 (Omkar)
Quick smoke test: generate 100 synthetic events and confirm all scores in [0.0, 1.0].

Run before integrating with the rest of the team:
    cd normative/ml_models
    python validate.py
"""

import json
import random
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
from anomaly_scorer import score_window
from lstm_autoencoder import WINDOW_SIZE, FEATURE_ORDER as LSTM_FEATURES
from isolation_forest import FEATURE_ORDER as IFOREST_FEATURES


def make_fake_fv(event_id: str, is_attack: bool = False) -> dict:
    """Synthetic ML_FEATURE_VECTOR dict."""
    base = {
        "event_id":             event_id,
        "host_key":             "192.168.1.100",
        "window_events":        [],
        "login_fail_count":     random.randint(0, 5),
        "login_success_count":  random.randint(0, 3),
        "unique_dest_ips":      random.randint(1, 10),
        "bytes_sent_total":     random.uniform(100, 10_000),
        "file_ops_count":       random.randint(0, 20),
        "cpu_pct_avg":          random.uniform(5, 60),
        "payload_flag_count":   random.randint(0, 3),
        "inter_arrival_ms_avg": random.uniform(10, 500),
        "entropy_dest_ports":   random.uniform(0, 4),
        "src_port":             random.uniform(0, 1),   # pre-normalised
        "dest_port":            random.uniform(0, 1),   # pre-normalised
    }
    if is_attack:
        # Exaggerate features to simulate attack
        base["login_fail_count"]    = random.randint(50, 200)
        base["bytes_sent_total"]    = random.uniform(1_000_000, 5_000_000)
        base["payload_flag_count"]  = random.randint(10, 50)
    return base


def run_validation(n_events: int = 100) -> None:
    print(f"[VALIDATE] Generating {n_events} synthetic events …\n")

    host_key  = "192.168.1.100"
    window    = []
    results   = []
    errors    = []

    for i in range(n_events):
        is_attack = i >= 70    # last 30 events are "attack"
        fv = make_fake_fv(event_id=f"evt_{i:04d}", is_attack=is_attack)
        window.append(fv)

        if len(window) < WINDOW_SIZE:
            continue

        score_obj = score_window(host_key, window[-WINDOW_SIZE:])
        results.append(score_obj)

        # Validate ranges
        for field in ("lstm_score", "iforest_score", "composite_ml_score"):
            v = score_obj[field]
            if not (0.0 <= v <= 1.0):
                errors.append(f"Event {score_obj['event_id']}: {field}={v} OUT OF RANGE")

        if i % 10 == 0 or is_attack:
            tag = "ATTACK" if is_attack else "benign"
            print(
                f"  [{tag:6s}] evt_{i:04d} "
                f"lstm={score_obj['lstm_score']:.3f}  "
                f"iforest={score_obj['iforest_score']:.3f}  "
                f"composite={score_obj['composite_ml_score']:.3f}  "
                f"anomaly={score_obj['is_anomaly']}"
            )

    print(f"\n[VALIDATE] Scored {len(results)} windows.")

    if errors:
        print(f"\n❌  {len(errors)} range violations found:")
        for e in errors:
            print(f"    {e}")
        sys.exit(1)
    else:
        print("✅  All scores in [0.0, 1.0] — validation passed.")

    # Dump last result as schema reference
    if results:
        print("\n[SCHEMA] Last ANOMALY_SCORE_OBJECT:")
        print(json.dumps(results[-1], indent=2))


if __name__ == "__main__":
    run_validation(n_events=100)
