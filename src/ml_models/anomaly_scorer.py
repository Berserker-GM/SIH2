"""
anomaly_scorer.py — Dev 2 (Omkar)
Combines LSTM Autoencoder + Isolation Forest scores, publishes ANOMALY_SCORE_OBJECT
to Redis channel scores:anomaly.

Redis channels:
  SUBSCRIBE : features:ml          — ML_FEATURE_VECTOR per event (from Dev 1)
  LPUSH+PUBLISH: scores:anomaly    — ANOMALY_SCORE_OBJECT (consumed by Dev 3 GNN)
"""

import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Add project root (Normnative/) to Python path so absolute imports work.
# This file is at src/ml_models/anomaly_scorer.py → project root = parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import collections
from datetime import datetime, timezone

import redis

from src.ml_models.lstm_autoencoder import score_window_lstm, WINDOW_SIZE, ANOMALY_THRESHOLD
from src.ml_models.isolation_forest import get_scorer
from src.ml_models.schema_validator import validate_ml_feature_vector, validate_anomaly_score

import os
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REDIS_HOST   = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT   = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB     = int(os.getenv("REDIS_DB", 0))

COMPOSITE_LSTM_WEIGHT   = 0.6
COMPOSITE_IFOREST_WEIGHT = 0.4
COMPOSITE_THRESHOLD      = 0.6   # is_anomaly = True if composite > this

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SCORER] %(message)s")
log = logging.getLogger("anomaly_scorer")


# ---------------------------------------------------------------------------
# Sliding-window buffer per host_key
# ---------------------------------------------------------------------------
# host_key → deque of ML_FEATURE_VECTOR dicts (max len = WINDOW_SIZE)
_windows: dict[str, collections.deque] = collections.defaultdict(
    lambda: collections.deque(maxlen=WINDOW_SIZE)
)


# ---------------------------------------------------------------------------
# Core scoring function — exact output contract per spec
# ---------------------------------------------------------------------------
def score_window(host_key: str, feature_vectors: list[dict]) -> dict:
    """
    Input:  list of ML_FEATURE_VECTOR dicts (len == WINDOW_SIZE)
    Output: ANOMALY_SCORE_OBJECT — published to Redis channel scores:anomaly
    """
    lstm_score    = score_window_lstm(feature_vectors)
    iforest_score = get_scorer().score(feature_vectors[-1])   # point score on latest event

    composite = round(
        COMPOSITE_LSTM_WEIGHT * lstm_score + COMPOSITE_IFOREST_WEIGHT * iforest_score, 6
    )
    
    # Aggregate all matched rules across the entire sliding window to preserve the attack chain
    all_matched_rules = []
    seen_rule_ids = set()
    for fv in feature_vectors:
        for rule in fv.get("matched_rules", []):
            if rule["rule_id"] not in seen_rule_ids:
                seen_rule_ids.add(rule["rule_id"])
                all_matched_rules.append(rule)

    max_severity = max([r.get("severity_weight", 0.0) for r in all_matched_rules], default=0.0)
    is_anomaly = (composite >= COMPOSITE_THRESHOLD) or (max_severity >= 0.8)

    return {
        "event_id":           feature_vectors[-1]["event_id"],
        "src_ip":             host_key,
        "matched_rules":      all_matched_rules,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "lstm_score":         round(lstm_score, 6),
        "iforest_score":      round(iforest_score, 6),
        "composite_ml_score": composite,
        "is_anomaly":         is_anomaly,
        "window_events":      [fv["event_id"] for fv in feature_vectors],
    }


# ---------------------------------------------------------------------------
# Redis publisher helper
# ---------------------------------------------------------------------------
def publish_score(r: redis.Redis, score_obj: dict) -> None:
    # Validate outbound object against the shared integration contract
    ok, err = validate_anomaly_score(score_obj)
    if not ok:
        log.error("AnomalyScoreObject contract violation — NOT publishing: %s", err)
        return
    payload = json.dumps(score_obj)
    r.lpush("scores:anomaly", payload)
    r.publish("scores:anomaly", payload)
    if score_obj["is_anomaly"]:
        log.warning(
            "ANOMALY detected | host=%s composite=%.3f",
            score_obj["event_id"],
            score_obj["composite_ml_score"],
        )


# ---------------------------------------------------------------------------
# Main subscriber loop
# ---------------------------------------------------------------------------
def run() -> None:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    pubsub = r.pubsub()
    pubsub.subscribe("features:ml")

    iforest = get_scorer()
    log.info("Subscribed to features:ml — waiting for events …")

    benign_phase = True   # first 500 events used for iForest bootstrap

    for message in pubsub.listen():
        if message["type"] != "message":
            continue

        try:
            fv: dict = json.loads(message["data"])
        except json.JSONDecodeError as exc:
            log.error("Bad JSON on features:ml: %s", exc)
            continue

        # --- Validate inbound ML_FEATURE_VECTOR against contract ---
        ok, err = validate_ml_feature_vector(fv)
        if not ok:
            log.error("Contract violation on features:ml — dropping event: %s", err)
            continue

        host_key: str = fv.get("host_key", "unknown")

        # --- iForest bootstrap during benign phase ---
        if benign_phase:
            fitted = iforest.add_benign_event(fv)
            if fitted:
                benign_phase = False
                iforest.save()

        # --- Accumulate sliding window ---
        _windows[host_key].append(fv)

        if len(_windows[host_key]) < WINDOW_SIZE:
            # Not enough history yet — skip scoring
            continue

        window = list(_windows[host_key])   # snapshot

        try:
            score_obj = score_window(host_key, window)
        except Exception as exc:
            log.error("Scoring error for host %s: %s", host_key, exc, exc_info=True)
            continue

        publish_score(r, score_obj)

        if score_obj.get("is_anomaly"):
            _windows[host_key].clear()


if __name__ == "__main__":
    run()