"""
graph_builder.py — Dev 3 (Ronit)

Maintains an in-memory directed attack graph as a NetworkX DiGraph.
Subscribes to Redis channel features:graph and processes GRAPH_EDGE objects
published by Dev 1.

Each node represents an entity (IP, user, process).
Each directed edge represents an interaction (network conn, auth attempt, spawn).
"""

import json
import logging
import threading

import networkx as nx
import redis

logger = logging.getLogger(__name__)


class AttackGraph:
    """
    Thread-safe directed attack graph.

    Nodes: entity identifiers — IP addresses, usernames, process names.
    Edges: directed interactions with metadata from Dev 1's GRAPH_EDGE objects.
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self._lock = threading.Lock()

    def add_edge(self, edge: dict) -> None:
        """
        Consume a GRAPH_EDGE object from Dev 1 and insert it into the graph.

        Expected edge keys:
            edge_id   : str   — unique identifier for this edge event
            src_node  : str   — source entity (IP / user / process)
            dst_node  : str   — destination entity
            edge_type : str   — e.g. "network_conn" | "auth_attempt" | "proc_spawn"
            weight    : float — edge weight (e.g. anomaly contribution)
            timestamp : str   — ISO 8601
            event_id  : str   — originating Normalized Event Object event_id
        """
        required = {"src_node", "dst_node", "edge_type", "weight", "timestamp", "event_id"}
        missing = required - edge.keys()
        if missing:
            logger.warning("GRAPH_EDGE missing fields %s — skipping: %s", missing, edge)
            return

        with self._lock:
            src = edge["src_node"]
            dst = edge["dst_node"]

            # If a parallel edge already exists, keep the one with higher weight
            # (NetworkX DiGraph only stores one edge per src→dst pair)
            if self.G.has_edge(src, dst):
                existing_weight = self.G[src][dst].get("weight", 0)
                if edge["weight"] <= existing_weight:
                    # Still append the event_id so we don't lose attribution
                    existing_ids = self.G[src][dst].get("event_ids", [])
                    existing_ids.append(edge["event_id"])
                    self.G[src][dst]["event_ids"] = existing_ids
                    return

            self.G.add_edge(
                src,
                dst,
                edge_type=edge["edge_type"],
                weight=edge["weight"],
                timestamp=edge["timestamp"],
                event_id=edge["event_id"],
                event_ids=[edge["event_id"]],  # list for multi-event aggregation
            )

            logger.debug("Edge added: %s → %s [%s]", src, dst, edge["edge_type"])

    def get_subgraph(self, node: str, hops: int = 2) -> nx.DiGraph:
        """
        Return the ego graph around a suspicious node (flagged by Dev 2).

        Args:
            node : the entity ID to centre the subgraph on
            hops : radius in hops (default 2)

        Returns:
            nx.DiGraph — subgraph; empty DiGraph if node not present
        """
        with self._lock:
            if node not in self.G:
                logger.warning("Node %s not in AttackGraph — returning empty subgraph", node)
                return nx.DiGraph()
            return nx.ego_graph(self.G, node, radius=hops, undirected=False)

    # ------------------------------------------------------------------
    # Node feature helpers — used by gnn_model.py to build feature vectors
    # ------------------------------------------------------------------

    def in_degree(self, node: str) -> int:
        with self._lock:
            return self.G.in_degree(node) if node in self.G else 0

    def out_degree(self, node: str) -> int:
        with self._lock:
            return self.G.out_degree(node) if node in self.G else 0

    def avg_edge_weight(self, node: str) -> float:
        """Average weight of all edges incident on node (in + out)."""
        with self._lock:
            if node not in self.G:
                return 0.0
            edges = list(self.G.in_edges(node, data=True)) + list(self.G.out_edges(node, data=True))
            if not edges:
                return 0.0
            return sum(d.get("weight", 0.0) for _, _, d in edges) / len(edges)

    def node_count(self) -> int:
        with self._lock:
            return self.G.number_of_nodes()

    def edge_count(self) -> int:
        with self._lock:
            return self.G.number_of_edges()


# ---------------------------------------------------------------------------
# Redis subscriber — runs as a background thread
# ---------------------------------------------------------------------------

class GraphEdgeSubscriber:
    """
    Subscribes to Redis channel features:graph and feeds edges into AttackGraph.

    Usage:
        attack_graph = AttackGraph()
        subscriber = GraphEdgeSubscriber(attack_graph, redis_client)
        subscriber.start()
    """

    CHANNEL = "features:graph"

    def __init__(self, attack_graph: AttackGraph, redis_client: redis.Redis):
        self.attack_graph = attack_graph
        self.redis_client = redis_client
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._listen, daemon=True, name="GraphEdgeSubscriber")
        self._thread.start()
        logger.info("GraphEdgeSubscriber started — listening on %s", self.CHANNEL)

    def stop(self) -> None:
        self._stop_event.set()

    def _listen(self) -> None:
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(self.CHANNEL)

        for message in pubsub.listen():
            if self._stop_event.is_set():
                break
            if message["type"] != "message":
                continue
            try:
                edge = json.loads(message["data"])
                self.attack_graph.add_edge(edge)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.error("Failed to parse GRAPH_EDGE: %s — raw: %s", exc, message["data"])
