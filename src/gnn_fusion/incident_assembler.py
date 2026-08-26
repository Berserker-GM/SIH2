"""
incident_assembler.py — Dev 3 (Ronit)  [MAIN SERVICE ENTRY POINT]

Subscribes to Redis channel scores:anomaly (ANOMALY_SCORE_OBJECTs from Dev 2).
For each anomaly flagged (is_anomaly=True), runs GNN inference, assembles
the complete Incident JSON, and:
  1. Publishes to Redis channel incidents:new  (LPUSH + PUBLISH)
  2. Writes to /app/output/incidents/{incident_id}.json

Incident JSON schema is the final contract consumed by Dev 4 (dashboard + LLM report).
"""

import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Add project root (Normnative/) to Python path so absolute imports work.
# This file is at src/gnn_fusion/incident_assembler.py → project root = parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import redis

from src.gnn_fusion.graph_builder import AttackGraph, GraphEdgeSubscriber
from src.gnn_fusion.gnn_model import GNNInference
from src.gnn_fusion.mitre_transitions import TECHNIQUE_NAMES, TECHNIQUE_TACTICS, get_top_next_technique
from src.gnn_fusion.risk_scorer import aggregate_mitre_severity, compute_risk_score, compute_severity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB   = int(os.getenv("REDIS_DB", 0))

ANOMALY_CHANNEL  = "scores:anomaly"
INCIDENT_CHANNEL = "incidents:new"
OUTPUT_DIR       = Path(os.getenv("INCIDENT_OUTPUT_DIR", "/app/output/incidents"))

GNN_WEIGHTS_PATH = os.getenv("GNN_WEIGHTS_PATH", "/app/gnn/models/gnn_weights.pt")

# Composite ML score threshold — only process events Dev 2 already flagged
ANOMALY_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Node feature accumulator
# Builds per-node feature dicts from incoming ANOMALY_SCORE_OBJECTs and
# the AttackGraph for graph topology features.
# ---------------------------------------------------------------------------

class NodeFeatureAccumulator:
    """
    Maintains a rolling aggregate of ML feature observations per node (IP).
    Provides the node_feature_map required by GNNInference.infer().
    """

    def __init__(self, attack_graph: AttackGraph):
        self.attack_graph = attack_graph
        self._store: dict[str, dict] = {}   # node_id → aggregated features
        self._lock = threading.Lock()

    def update_from_anomaly_score(self, score_obj: dict, src_ip: str) -> None:
        """
        Incorporate a new ANOMALY_SCORE_OBJECT into the per-node feature store.
        Graph topology features (in_degree, out_degree, avg_edge_weight) are
        read live from AttackGraph at inference time via get_features().
        """
        with self._lock:
            node = self._store.setdefault(src_ip, {
                "login_fail_count":    0.0,
                "login_success_count": 0.0,
                "unique_dest_ips":     0.0,
                "bytes_sent_total":    0.0,
                "file_ops_count":      0.0,
                "cpu_pct_avg":         0.0,
                "payload_flag_count":  0.0,
                "lstm_score_avg":      0.0,
                "iforest_score_avg":   0.0,
                "_lstm_count":         0,
                "_iforest_count":      0,
            })

            # Running average for ML scores
            n_lstm = node["_lstm_count"] + 1
            node["lstm_score_avg"] = (
                node["lstm_score_avg"] * node["_lstm_count"] + score_obj.get("lstm_score", 0.0)
            ) / n_lstm
            node["_lstm_count"] = n_lstm

            n_if = node["_iforest_count"] + 1
            node["iforest_score_avg"] = (
                node["iforest_score_avg"] * node["_iforest_count"] + score_obj.get("iforest_score", 0.0)
            ) / n_if
            node["_iforest_count"] = n_if

    def get_features(self, node_id: str) -> dict:
        """
        Return the full 12-feature dict for a node, injecting live graph topology.
        """
        with self._lock:
            base = dict(self._store.get(node_id, {}))

        # Inject live topology from AttackGraph
        base["in_degree"]       = float(self.attack_graph.in_degree(node_id))
        base["out_degree"]      = float(self.attack_graph.out_degree(node_id))
        base["avg_edge_weight"] = self.attack_graph.avg_edge_weight(node_id)

        return base

    def get_feature_map(self, nodes: list[str]) -> dict:
        """Return {node_id: feature_dict} for a list of nodes."""
        return {n: self.get_features(n) for n in nodes}


# ---------------------------------------------------------------------------
# Incident assembler
# ---------------------------------------------------------------------------

def assemble_incident(
    anomaly_scores: list[dict],   # ANOMALY_SCORE_OBJECTs contributing to this incident
    gnn_output: dict,             # GNN_OUTPUT from NormativeGNN
    matched_rules: list[dict],    # MITRE_RULES entries matched by Dev 1
) -> dict:
    """
    Assemble the complete Incident JSON (final contract for Dev 4).

    Publishes to:  Redis channel incidents:new  (LPUSH + PUBLISH)
    Also writes to: /app/output/incidents/{incident_id}.json

    Args:
        anomaly_scores : list of ANOMALY_SCORE_OBJECTs (at least one required)
        gnn_output     : GNN_OUTPUT dict from gnn_model.py
        matched_rules  : list of MITRE rule dicts from Dev 1

    Returns:
        dict — the complete Incident JSON
    """
    if not anomaly_scores:
        raise ValueError("assemble_incident requires at least one anomaly_score object")

    now_utc = datetime.now(timezone.utc)
    incident_id = f"INC-{now_utc.strftime('%Y%m%d-%H%M%S')}"

    # Risk scoring
    latest_score = anomaly_scores[-1]
    composite_ml = latest_score.get("composite_ml_score", 0.0)
    gnn_conf     = gnn_output.get("gnn_confidence", 0.0)
    mitre_weight = aggregate_mitre_severity(matched_rules)

    risk_score = compute_risk_score(composite_ml, gnn_conf, mitre_weight)
    severity   = compute_severity(risk_score)

    # Attack chain — one entry per matched rule, ordered by event timestamp
    sorted_rules = sorted(
        matched_rules,
        key=lambda r: r.get("timestamp", ""),
    )
    attack_chain = []
    for step_num, rule in enumerate(sorted_rules, start=1):
        attack_chain.append({
            "step":           step_num,
            "technique_id":   rule.get("technique_id", "UNKNOWN"),
            "technique_name": rule.get("technique_name", TECHNIQUE_NAMES.get(rule.get("technique_id", ""), "Unknown")),
            "tactic":         rule.get("tactic", TECHNIQUE_TACTICS.get(rule.get("technique_id", ""), "Unknown")),
            "confidence":     gnn_conf,
            "timestamp":      rule.get("timestamp", now_utc.isoformat()),
            "event_ids":      rule.get("event_ids", []),
        })

    # Predicted next technique
    next_technique_probs: dict = gnn_output.get("next_technique_probs", {})
    predicted_next = _resolve_predicted_next(next_technique_probs, matched_rules)

    # All contributing event_ids
    raw_event_ids: list[str] = []
    for score in anomaly_scores:
        raw_event_ids.extend(score.get("window_events", []))
    # Also pull from matched rules
    for rule in matched_rules:
        raw_event_ids.extend(rule.get("event_ids", []))
    raw_event_ids = list(dict.fromkeys(raw_event_ids))  # deduplicate, preserve order

    # LLM context block
    technique_names_in_chain = [r.get("technique_name", "") for r in sorted_rules]
    last_rule = sorted_rules[-1] if sorted_rules else {}
    llm_context = {
        "summary":         " → ".join(technique_names_in_chain) if technique_names_in_chain else "No MITRE techniques matched.",
        "mitre_tags":      [r.get("technique_id", "") for r in sorted_rules],
        "severity_reason": last_rule.get("llm_description", f"Risk score {risk_score} driven by ML anomaly and GNN classification."),
    }

    # Anomaly scores block
    anomaly_scores_block = {
        "lstm":    latest_score.get("lstm_score", 0.0),
        "iforest": latest_score.get("iforest_score", 0.0),
        "gnn":     gnn_conf,
    }

    # model_metrics is populated by Dev 4 at render time from /api/metrics
    model_metrics: dict = {}

    incident = {
        "incident_id":     incident_id,
        "detected_at":     now_utc.isoformat(),
        "severity":        severity,
        "risk_score":      risk_score,
        "affected_assets": [gnn_output.get("node_id", "unknown")],
        "attack_chain":    attack_chain,
        "predicted_next":  predicted_next,
        "anomaly_scores":  anomaly_scores_block,
        "model_metrics":   model_metrics,
        "llm_context":     llm_context,
        "raw_event_ids":   raw_event_ids,
    }

    return incident


def _resolve_predicted_next(
    next_technique_probs: dict,
    matched_rules: list[dict],
) -> dict:
    """
    Select the highest-probability next technique from the GNN output.
    Falls back to TRANSITION_MATRIX directly if GNN probs are empty.
    """
    if next_technique_probs:
        best_tid = max(next_technique_probs, key=next_technique_probs.get)
        best_prob = next_technique_probs[best_tid]
    else:
        # Fallback: use the last matched rule's technique to seed transitions
        last_tid = matched_rules[-1].get("technique_id") if matched_rules else None
        result = get_top_next_technique(last_tid) if last_tid else None
        if result:
            best_tid, best_prob = result
        else:
            return {
                "technique_id":   "UNKNOWN",
                "technique_name": "Unknown",
                "tactic":         "Unknown",
                "probability":    0.0,
            }

    return {
        "technique_id":   best_tid,
        "technique_name": TECHNIQUE_NAMES.get(best_tid, "Unknown"),
        "tactic":         TECHNIQUE_TACTICS.get(best_tid, "Unknown"),
        "probability":    round(best_prob, 4),
    }


# ---------------------------------------------------------------------------
# Redis publish + disk write
# ---------------------------------------------------------------------------

def publish_incident(incident: dict, redis_client: redis.Redis) -> None:
    """
    LPUSH and PUBLISH to Redis incidents:new channel.
    Also writes the incident to /app/output/incidents/{incident_id}.json.
    """
    payload = json.dumps(incident, default=str)

    # Redis: LPUSH keeps a list of recent incidents; PUBLISH notifies consumers
    redis_client.lpush(INCIDENT_CHANNEL, payload)
    redis_client.publish(INCIDENT_CHANNEL, payload)
    logger.info(
        "Published incident %s (severity=%s, risk=%d) to Redis",
        incident["incident_id"], incident["severity"], incident["risk_score"],
    )

    # Disk write
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{incident['incident_id']}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(incident, fh, indent=2, default=str)
    logger.info("Incident written to disk: %s", out_path)


# ---------------------------------------------------------------------------
# Main service loop
# ---------------------------------------------------------------------------

class IncidentAssemblerService:
    """
    Main service that ties together:
      - AttackGraph (fed by GraphEdgeSubscriber on features:graph)
      - GNNInference
      - NodeFeatureAccumulator
      - Incident assembly and publishing
    Subscribes to scores:anomaly and processes flagged anomalies.
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.attack_graph = AttackGraph()
        self.feature_accumulator = NodeFeatureAccumulator(self.attack_graph)
        self.gnn = GNNInference(weights_path=GNN_WEIGHTS_PATH)

        # Start graph edge subscriber in background
        self.edge_subscriber = GraphEdgeSubscriber(self.attack_graph, redis_client)
        self.edge_subscriber.start()

        self._stop = threading.Event()

    def run(self) -> None:
        """Block and process anomaly score messages from Dev 2."""
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(ANOMALY_CHANNEL)
        logger.info("IncidentAssemblerService listening on %s", ANOMALY_CHANNEL)

        for message in pubsub.listen():
            if self._stop.is_set():
                break
            if message["type"] != "message":
                continue
            try:
                self._handle_anomaly_score(message["data"])
            except Exception as exc:
                logger.exception("Unhandled error processing anomaly score: %s", exc)

    def stop(self) -> None:
        self._stop.set()
        self.edge_subscriber.stop()

    # ------------------------------------------------------------------

    def _handle_anomaly_score(self, raw: bytes | str) -> None:
        """Process one ANOMALY_SCORE_OBJECT from Dev 2."""
        try:
            score_obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("Failed to parse ANOMALY_SCORE_OBJECT: %s", exc)
            return

        # Only process anomalies above threshold
        if not score_obj.get("is_anomaly", False):
            return

        event_id = score_obj.get("event_id", "unknown")
        logger.info("Processing anomaly: event_id=%s score=%.3f", event_id, score_obj.get("composite_ml_score"))

        # Identify the source node — in production this comes from the paired
        # Normalized Event Object. For now, derive from event_id or use a sentinel.
        # Dev 1 should include src_ip in the ANOMALY_SCORE_OBJECT or a sidecar lookup.
        src_ip = score_obj.get("src_ip", "unknown_node")

        # Update feature accumulator
        self.feature_accumulator.update_from_anomaly_score(score_obj, src_ip)

        # Build node feature map for the subgraph around src_ip
        subgraph = self.attack_graph.get_subgraph(src_ip, hops=2)
        subgraph_nodes = list(subgraph.nodes()) or [src_ip]
        node_feature_map = self.feature_accumulator.get_feature_map(subgraph_nodes)

        # Determine current technique from matched rules (passed via score_obj if available)
        # In production, Dev 1 publishes matched_rules alongside or as a separate object.
        # We accept them embedded in the score_obj for flexibility.
        matched_rules: list[dict] = score_obj.get("matched_rules", [])
        current_technique: Optional[str] = (
            matched_rules[-1]["technique_id"] if matched_rules else None
        )

        # GNN inference
        gnn_output = self.gnn.infer(
            attack_graph=self.attack_graph,
            target_node=src_ip,
            node_feature_map=node_feature_map,
            current_technique=current_technique,
        )

        # Assemble incident
        incident = assemble_incident(
            anomaly_scores=[score_obj],
            gnn_output=gnn_output,
            matched_rules=matched_rules,
        )

        # Publish and persist
        publish_incident(incident, self.redis_client)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Starting NORMATIVE Incident Assembler Service (Dev 3)")

    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
    )

    # Basic connectivity check
    try:
        redis_client.ping()
        logger.info("Redis connection OK — %s:%d db=%d", REDIS_HOST, REDIS_PORT, REDIS_DB)
    except redis.ConnectionError as exc:
        logger.critical("Cannot connect to Redis: %s", exc)
        raise SystemExit(1) from exc

    service = IncidentAssemblerService(redis_client)
    try:
        service.run()
    except KeyboardInterrupt:
        logger.info("Shutting down IncidentAssemblerService")
        service.stop()


if __name__ == "__main__":
    main()