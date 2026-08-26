#!/usr/bin/env python3
"""Network state as feature vector S_t AND communication graph G_t."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.world_model.cic_schema import INPUT_DIM, bind_columns
from src.world_model.labels import STAGE_ID
from src.world_model.prepare_dataset import generate_demo_flows
from src.world_model.state_graph import build_window_graph
from src.world_model.windows import WindowConfig, build_state_windows

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition))
    icon = "PASS" if condition else "FAIL"
    extra = f"  ({detail})" if detail else ""
    print(f"  {icon}  {name}{extra}")


print("\n-- Service graph from CIC ML schema (no IPs) --------------------------")
df = generate_demo_flows(n_windows=24, flows_per_window=6)
benign = df[df["Label"] == "Benign"].head(6)
scan = df[df["Label"] == "Infilteration"].head(12)

g_benign = build_window_graph(benign)
g_scan = build_window_graph(scan)
check("benign is service mode", g_benign["mode"] == "service")
check("scan is service mode", g_scan["mode"] == "service")
check("no IPs claimed", g_benign["has_src_ip"] is False and g_benign["has_dst_ip"] is False)
check("unobserved source node present", any(n["id"] == "src:unobserved" for n in g_benign["nodes"]))
check("benign has dest-port nodes", g_benign["n_nodes"] >= 2)
check("scan fan-out > benign", g_scan["max_out_degree"] > g_benign["max_out_degree"])
check("scan has more dest nodes", g_scan["n_nodes"] > g_benign["n_nodes"])
check("edges have flows/bytes", all("flows" in e and "bytes" in e for e in g_scan["edges"]))
check("hub is a dest port", (g_scan["hub"] or {}).get("kind") in {"service", "host"})


print("\n-- Host graph when Src/Dst IP columns exist ---------------------------")
host_df = df.copy()
host_df["Src IP"] = ["10.0.0." + str(1 + (i % 3)) for i in range(len(host_df))]
host_df["Dst IP"] = ["192.168.1." + str(10 + (i % 5)) for i in range(len(host_df))]
bound = bind_columns(list(host_df.columns))
check("binds Src IP", bound.get("src_ip") == "Src IP")
check("binds Dst IP", bound.get("dst_ip") == "Dst IP")

g_host = build_window_graph(host_df.head(12), bound)
check("host mode when both IPs present", g_host["mode"] == "host")
check("has_src_ip true", g_host["has_src_ip"] is True)
check("has_dst_ip true", g_host["has_dst_ip"] is True)
check("host nodes are IPs", all(n["kind"] in {"host"} for n in g_host["nodes"]))
check("host edges src->dst", all(e["src"].startswith("h:") and e["dst"].startswith("h:") for e in g_host["edges"]))

only_src = df.copy()
only_src["Src IP"] = ["203.0.113.99"] * len(only_src)
g_hs = build_window_graph(only_src.head(8))
check("host_service when only Src IP", g_hs["mode"] == "host_service")


print("\n-- Windows keep 32-d S_t; graphs are optional sidecar -----------------")
plain = build_state_windows(df, WindowConfig(window_seconds=5, horizon_k=6))
with_g = build_state_windows(
    df, WindowConfig(window_seconds=5, horizon_k=6), include_graphs=True,
)
check("default NPZ path has no graphs", "graphs" not in plain)
check("S_t still 32-d", plain["states"].shape[1] == INPUT_DIM)
check("graphs do not change S_t", np.allclose(plain["states"], with_g["states"]))
check("graphs aligned with states", len(with_g["graphs"]) == len(with_g["states"]))
check("last graph is service (demo CSV)", with_g["graphs"][-1]["mode"] == "service")
infil_ids = np.where(with_g["stage_id"] == STAGE_ID["lateral_movement"])[0]
check("demo still contains infiltration windows", infil_ids.size >= 1)
infil_graph = with_g["graphs"][int(infil_ids[-1])] if infil_ids.size else {"max_out_degree": 0}
check("infiltration window has star fan-out", int(infil_graph.get("max_out_degree") or 0) >= 6)


print("\n-- Summary ------------------------------------------------------------")
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"\n  Total: {len(results)}   Passed: {passed}   Failed: {failed}\n")
if failed:
    sys.exit(1)
print("  Dual state representation (vector + graph) is ready.\n")
