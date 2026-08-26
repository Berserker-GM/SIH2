"""Per-window communication graph G_t alongside the 32-d feature vector S_t.

SIH asks to represent network state using feature vectors AND graphs.
S_t stays the frozen 32-d LSTM input (cic_schema.STATE_FEATURE_ORDER).
G_t is a directed flow graph for the same 5-second bin:

  host mode        — Src IP / Dst IP present (GeneratedLabelledFlows / PCAP)
  service mode     — CIC ML CSV default: dest ports only, sources unobserved
  host_service     — source hosts → dest ports when only Src IP is present

This is NOT a GNN world model and does not change input_dim=32.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.world_model.cic_schema import bind_columns

MAX_UI_NODES = 40
MAX_UI_EDGES = 64

_PROTO = {1: "icmp", 6: "tcp", 17: "udp"}


def _col(df: pd.DataFrame, bound: dict[str, str], field: str) -> pd.Series:
    if field not in bound:
        return pd.Series(0, index=df.index, dtype="float64")
    return pd.to_numeric(df[bound[field]], errors="coerce")


def _proto_name(value: float | int) -> str:
    try:
        return _PROTO.get(int(value), str(int(value)))
    except (TypeError, ValueError):
        return "?"


def _clean_ip(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.casefold() in {"nan", "none", "nat", "0", "0.0.0.0"}:
        return ""
    return text


def _series_ips(frame: pd.DataFrame, bound: dict[str, str], field: str) -> list[str] | None:
    if field not in bound:
        return None
    values = [_clean_ip(v) for v in frame[bound[field]].tolist()]
    if not any(values):
        return None
    return values


def _empty_graph(note: str) -> dict[str, Any]:
    return {
        "mode": "empty",
        "has_src_ip": False,
        "has_dst_ip": False,
        "note": note,
        "n_flows": 0,
        "n_nodes": 0,
        "n_edges": 0,
        "max_in_degree": 0,
        "max_out_degree": 0,
        "density": 0.0,
        "hub": None,
        "nodes": [],
        "edges": [],
    }


def build_window_graph(
    frame: pd.DataFrame,
    bound: dict[str, str] | None = None,
    *,
    max_nodes: int = MAX_UI_NODES,
    max_edges: int = MAX_UI_EDGES,
) -> dict[str, Any]:
    """Build G_t from the flows that landed in one 5-second window."""
    if frame is None or frame.empty:
        return _empty_graph("empty window")

    bound = bound or bind_columns(list(frame.columns))
    src_ips = _series_ips(frame, bound, "src_ip")
    dst_ips = _series_ips(frame, bound, "dst_ip")
    ports = _col(frame, bound, "dst_port").fillna(0).to_numpy()
    protos = _col(frame, bound, "protocol").fillna(0).to_numpy()
    bytes_fwd = _col(frame, bound, "totlen_fwd").fillna(0).to_numpy()
    bytes_bwd = _col(frame, bound, "totlen_bwd").fillna(0).to_numpy()
    pkts_fwd = _col(frame, bound, "tot_fwd_pkts").fillna(0).to_numpy()
    pkts_bwd = _col(frame, bound, "tot_bwd_pkts").fillna(0).to_numpy()

    n_flows = int(len(frame))
    has_src = src_ips is not None
    has_dst = dst_ips is not None

    if has_src and has_dst:
        mode = "host"
        note = "Host graph: Src IP → Dst IP (same 5s bin as S_t)."
    elif has_src:
        mode = "host_service"
        note = "Bipartite graph: Src IP → dest port (Dst IP absent)."
    else:
        mode = "service"
        note = (
            "Service graph: CIC ML CSVs omit Src/Dst IP, so sources are one "
            "unobserved-client node and destinations are dest ports. Host graphs "
            "need GeneratedLabelledFlows or PCAP."
        )

    edge_acc: dict[tuple[str, str], dict[str, float]] = {}
    node_meta: dict[str, dict[str, Any]] = {}

    def touch(node_id: str, kind: str, label: str) -> None:
        slot = node_meta.get(node_id)
        if slot is None:
            node_meta[node_id] = {
                "id": node_id,
                "kind": kind,
                "label": label,
                "in_degree": 0,
                "out_degree": 0,
                "bytes": 0.0,
                "flows": 0.0,
            }
            return
        if kind != "source" and slot["kind"] == "source":
            slot["kind"] = kind
            slot["label"] = label

    if mode == "service":
        touch("src:unobserved", "source", "unobserved sources")

    for i in range(n_flows):
        port = int(ports[i]) if i < len(ports) else 0
        proto = _proto_name(protos[i] if i < len(protos) else 0)
        nbytes = float(bytes_fwd[i] + bytes_bwd[i]) if i < len(bytes_fwd) else 0.0
        npkts = float(pkts_fwd[i] + pkts_bwd[i]) if i < len(pkts_fwd) else 0.0

        if mode == "host":
            src_id = f"h:{src_ips[i] or 'unknown'}"
            dst_id = f"h:{dst_ips[i] or 'unknown'}"
            touch(src_id, "host", src_ips[i] or "unknown")
            touch(dst_id, "host", dst_ips[i] or "unknown")
        elif mode == "host_service":
            src_id = f"h:{src_ips[i] or 'unknown'}"
            dst_id = f"p:{port}"
            touch(src_id, "host", src_ips[i] or "unknown")
            touch(dst_id, "service", f"{port}/{proto}")
        else:
            src_id = "src:unobserved"
            dst_id = f"p:{port}"
            touch(dst_id, "service", f"{port}/{proto}")

        key = (src_id, dst_id)
        slot = edge_acc.get(key)
        if slot is None:
            slot = {
                "flows": 0.0,
                "bytes": 0.0,
                "pkts": 0.0,
                "protocol": proto,
                "dst_port": port,
            }
            edge_acc[key] = slot
        slot["flows"] += 1.0
        slot["bytes"] += nbytes
        slot["pkts"] += npkts

        node_meta[src_id]["bytes"] += nbytes
        node_meta[src_id]["flows"] += 1.0
        node_meta[dst_id]["bytes"] += nbytes
        node_meta[dst_id]["flows"] += 1.0

    for src_id, dst_id in edge_acc:
        node_meta[src_id]["out_degree"] += 1
        node_meta[dst_id]["in_degree"] += 1

    n_nodes = len(node_meta)
    n_edges = len(edge_acc)
    possible = n_nodes * (n_nodes - 1)
    density = float(n_edges / possible) if possible > 0 else 0.0
    max_in = max((n["in_degree"] for n in node_meta.values()), default=0)
    max_out = max((n["out_degree"] for n in node_meta.values()), default=0)

    hub = None
    dests = [n for n in node_meta.values() if n["kind"] != "source"]
    if dests:
        hub_node = max(dests, key=lambda n: (n["in_degree"], n["bytes"]))
        hub = {
            "id": hub_node["id"],
            "label": hub_node["label"],
            "kind": hub_node["kind"],
            "in_degree": int(hub_node["in_degree"]),
            "bytes": float(hub_node["bytes"]),
        }

    ranked_nodes = sorted(
        node_meta.values(),
        key=lambda n: (n["in_degree"] + n["out_degree"], n["bytes"]),
        reverse=True,
    )
    keep_ids = {n["id"] for n in ranked_nodes[: max(1, max_nodes)]}
    ui_nodes = [
        {
            "id": n["id"],
            "kind": n["kind"],
            "label": n["label"],
            "in_degree": int(n["in_degree"]),
            "out_degree": int(n["out_degree"]),
            "bytes": float(n["bytes"]),
            "flows": float(n["flows"]),
        }
        for n in ranked_nodes
        if n["id"] in keep_ids
    ]

    ranked_edges = sorted(
        (
            {
                "src": src,
                "dst": dst,
                "flows": float(st["flows"]),
                "bytes": float(st["bytes"]),
                "pkts": float(st["pkts"]),
                "protocol": st["protocol"],
                "dst_port": int(st["dst_port"]),
            }
            for (src, dst), st in edge_acc.items()
            if src in keep_ids and dst in keep_ids
        ),
        key=lambda e: (e["flows"], e["bytes"]),
        reverse=True,
    )[:max_edges]

    return {
        "mode": mode,
        "has_src_ip": bool(has_src),
        "has_dst_ip": bool(has_dst),
        "note": note,
        "n_flows": n_flows,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "max_in_degree": int(max_in),
        "max_out_degree": int(max_out),
        "density": density,
        "hub": hub,
        "nodes": ui_nodes,
        "edges": ranked_edges,
    }
