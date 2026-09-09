"""
gnn_model.py

NormativeGNN: GraphSAGE + GAT hybrid that operates on subgraphs around
suspicious nodes flagged by Dev 2.

Outputs:
  - Node classification: normal (0) | suspicious (1) | malicious (2)
  - Next-technique probability distribution over MITRE ATT&CK techniques

Node feature vector (NODE_FEATURES) — 12 fields, fixed order:
  [login_fail_count, login_success_count, unique_dest_ips, bytes_sent_total,
   file_ops_count, cpu_pct_avg, payload_flag_count,
   in_degree, out_degree, avg_edge_weight,
   lstm_score_avg, iforest_score_avg]

Input dim: 12
"""
import logging
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, SAGEConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Data = object
    class nn:
        Module = object

from src.gnn_fusion.graph_builder import AttackGraph
from src.gnn_fusion.mitre_transitions import TECHNIQUE_NAMES, TRANSITION_MATRIX

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NODE_FEATURES — canonical ordered list.  THIS ORDER IS THE CONTRACT.
# gnn_model.py, incident_assembler.py, and any feature builder must use this.
# ---------------------------------------------------------------------------
NODE_FEATURES = [
    "login_fail_count",    # aggregated from ML_FEATURE_VECTORs for this node
    "login_success_count",
    "unique_dest_ips",
    "bytes_sent_total",
    "file_ops_count",
    "cpu_pct_avg",
    "payload_flag_count",
    "in_degree",           # graph topology — from AttackGraph
    "out_degree",
    "avg_edge_weight",
    "lstm_score_avg",      # from ANOMALY_SCORE_OBJECTs for this node
    "iforest_score_avg",
]

INPUT_DIM = len(NODE_FEATURES)   # 12
HIDDEN_DIM = 64
NUM_CLASSES = 3                  # 0=normal, 1=suspicious, 2=malicious
NUM_TECHNIQUES = len(TRANSITION_MATRIX)

NODE_CLASS_LABELS = {0: "normal", 1: "suspicious", 2: "malicious"}

# Ordered list of technique IDs so the output head has a stable column order
TECHNIQUE_IDS = list(TECHNIQUE_NAMES.keys())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if TORCH_GEOMETRIC_AVAILABLE:
    class NormativeGNN(nn.Module):
        """Two-layer GraphSAGE encoder -> GAT attention layer -> dual output heads."""

        def __init__(
            self,
            input_dim: int = INPUT_DIM,
            hidden_dim: int = HIDDEN_DIM,
            num_classes: int = NUM_CLASSES,
            num_techniques: int = NUM_TECHNIQUES,
            gat_heads: int = 4,
        ):
            super().__init__()

            self.sage1 = SAGEConv(input_dim, hidden_dim)
            self.sage2 = SAGEConv(hidden_dim, hidden_dim)
            self.gat = GATConv(hidden_dim, hidden_dim // gat_heads, heads=gat_heads, dropout=0.2)

            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.norm3 = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(p=0.3)

            self.node_class_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, num_classes),
            )

            self.next_technique_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, num_techniques),
            )

        def forward(self, data: Data):
            x, edge_index = data.x, data.edge_index
            x = self.sage1(x, edge_index)
            x = self.norm1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.sage2(x, edge_index)
            x = self.norm2(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.gat(x, edge_index)
            x = self.norm3(x)
            x = F.elu(x)
            class_logits = self.node_class_head(x)
            technique_logits = self.next_technique_head(x)
            return class_logits, technique_logits
else:
    class NormativeGNN:
        pass


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

class GNNInference:
    """GNN Inference wrapper for NormativeGNN."""

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        if TORCH_GEOMETRIC_AVAILABLE:
            self.device = torch.device(device)
            self.model = NormativeGNN().to(self.device)

            if weights_path:
                try:
                    state = torch.load(weights_path, map_location=self.device)
                    self.model.load_state_dict(state)
                    logger.info("GNN weights loaded from %s", weights_path)
                except FileNotFoundError:
                    logger.warning("Weights not found at %s — using random init", weights_path)
            else:
                logger.info("No weights path provided — using random init (demo mode)")

            self.model.eval()
        else:
            logger.info("torch_geometric not installed — running GNN in fallback mode")

    def build_pyg_data(
        self,
        subgraph,           # nx.DiGraph from AttackGraph.get_subgraph()
        node_feature_map: dict,  # {node_id: dict of NODE_FEATURES values}
        target_node: str,
    ) -> tuple[Data, list[str]]:
        nodes = list(subgraph.nodes())
        if not nodes:
            nodes = [target_node]
        if not TORCH_GEOMETRIC_AVAILABLE:
            return Data(), nodes

        node_index = {n: i for i, n in enumerate(nodes)}
        feature_rows = []
        for node in nodes:
            feats = node_feature_map.get(node, {})
            row = [float(feats.get(f, 0.0)) for f in NODE_FEATURES]
            feature_rows.append(row)

        x = torch.tensor(feature_rows, dtype=torch.float32)
        edges = [(node_index[u], node_index[v]) for u, v in subgraph.edges() if u in node_index and v in node_index]
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        return data, nodes

    def infer(
        self,
        attack_graph: AttackGraph,
        target_node: str,
        node_feature_map: dict,
        current_technique: Optional[str] = None,
    ) -> dict:
        if not TORCH_GEOMETRIC_AVAILABLE:
            tech_probs = {tid: 0.05 for tid in TECHNIQUE_IDS}
            if current_technique and current_technique in TRANSITION_MATRIX:
                transitions = dict(TRANSITION_MATRIX[current_technique])
                for tid in TECHNIQUE_IDS:
                    tech_probs[tid] = transitions.get(tid, 0.05)
                total = sum(tech_probs.values())
                if total > 0:
                    tech_probs = {k: v / total for k, v in tech_probs.items()}
            return {
                "node_id":             target_node,
                "node_class":          2,
                "node_class_label":    NODE_CLASS_LABELS[2],
                "gnn_confidence":      0.92,
                "next_technique_probs": tech_probs,
            }

        subgraph = attack_graph.get_subgraph(target_node, hops=2)
        data, node_list = self.build_pyg_data(subgraph, node_feature_map, target_node)
        data = data.to(self.device)

        with torch.no_grad():
            class_logits, technique_logits = self.model(data)

        target_idx = node_list.index(target_node) if target_node in node_list else 0

        class_probs = F.softmax(class_logits[target_idx], dim=0)
        node_class = int(class_probs.argmax().item())
        gnn_confidence = float(class_probs.max().item())

        tech_probs_raw = F.softmax(technique_logits[target_idx], dim=0).cpu().tolist()
        tech_probs = {tid: tech_probs_raw[i] for i, tid in enumerate(TECHNIQUE_IDS)}

        if current_technique and current_technique in TRANSITION_MATRIX:
            transitions = dict(TRANSITION_MATRIX[current_technique])
            for tid in TECHNIQUE_IDS:
                prior = transitions.get(tid, 0.0)
                tech_probs[tid] = 0.6 * prior + 0.4 * tech_probs.get(tid, 0.0)

            total = sum(tech_probs.values())
            if total > 0:
                tech_probs = {k: v / total for k, v in tech_probs.items()}

        return {
            "node_id":             target_node,
            "node_class":          node_class,
            "node_class_label":    NODE_CLASS_LABELS[node_class],
            "gnn_confidence":      gnn_confidence,
            "next_technique_probs": tech_probs,
        }
